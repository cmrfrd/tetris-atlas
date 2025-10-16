use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Device, Shape, Tensor};
use candle_nn::loss::cross_entropy;
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
// use rand::Rng; // no longer needed
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::{debug, info};

use crate::checkpointer::Checkpointer;
use crate::data::TetrisDatasetGenerator;
use crate::grad_accum::{GradientAccumulator, get_l2_norm};
use crate::modules::{
    Conv2dConfig, ConvBlockSpec, ConvEncoder, ConvEncoderConfig, FiLM, FiLMConfig, Mlp, MlpConfig,
};
use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOrientationDistTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceOrientationTensor, TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement};
use crate::wrapped_tensor::WrappedTensor;

/// Simple goal-conditioned policy over placements using policy gradient (REINFORCE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGoalPolicyConfig {
    pub piece_embedding_dim: usize,

    pub board_encoder_config: ConvEncoderConfig,

    pub goal_film_config: FiLMConfig,
    pub piece_film_config: FiLMConfig,

    pub head_mlp_config: MlpConfig, // outputs logits over orientations
    pub value_mlp_config: MlpConfig, // outputs scalar value
}

#[derive(Debug, Clone)]
pub struct TetrisGoalPolicy {
    piece_embedding: Embedding,
    board_encoder: ConvEncoder,
    goal_film: FiLM,
    piece_film: FiLM,
    head_mlp: Mlp,
    value_mlp: Mlp,
}

impl TetrisGoalPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisGoalPolicyConfig) -> Result<Self> {
        let piece_embedding = embedding(
            TetrisPiece::NUM_PIECES,
            cfg.piece_embedding_dim,
            vb.pp("piece_embedding"),
        )?;
        let board_encoder = ConvEncoder::init(&vb.pp("board_encoder"), &cfg.board_encoder_config)?;
        let goal_film = FiLM::init(&vb.pp("goal_film"), &cfg.goal_film_config)?;
        let piece_film = FiLM::init(&vb.pp("piece_film"), &cfg.piece_film_config)?;
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;
        let value_mlp = Mlp::init(&vb.pp("value_mlp"), &cfg.value_mlp_config)?;
        Ok(Self {
            piece_embedding,
            board_encoder,
            goal_film,
            piece_film,
            head_mlp,
            value_mlp,
        })
    }

    /// Forward producing unmasked orientation logits [B, NUM_ORIENTATIONS]
    pub fn forward_logits(
        &self,
        goal_board: &TetrisBoardsTensor,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let (b, _t) = current_board.shape_tuple();

        // Encode boards as images [B,1,H,W] -> [B,D]
        let goal_img = goal_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;
        let cur_img = current_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;

        let goal_embed = self.board_encoder.forward(&goal_img)?; // [B, D]
        let cur_embed = self.board_encoder.forward(&cur_img)?; // [B, D]

        // Condition current on goal and piece
        let x = self.goal_film.forward(&cur_embed, &goal_embed)?; // [B, D]
        let piece_embed = self.piece_embedding.forward(current_piece)?.squeeze(1)?; // [B, D]
        let x = self.piece_film.forward(&x, &piece_embed)?; // [B, D]

        // Head -> orientation logits [B, NUM_ORIENTATIONS]
        let logits = self.head_mlp.forward(&x)?; // [B, O]
        TetrisPieceOrientationLogitsTensor::try_from(logits)
    }

    /// Compute masked action distribution over orientations, masking to those valid for current piece.
    pub fn forward_masked(
        &self,
        goal_board: &TetrisBoardsTensor,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(goal_board, current_board, current_piece)?; // [B,O]
        let mask = create_orientation_mask(current_piece)?; // [B,O] (0/1)
        // keep valid, set invalid to -inf
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        TetrisPieceOrientationLogitsTensor::try_from(masked)
    }

    /// Forward producing masked orientation logits and state value [B,O], [B]
    pub fn forward_masked_with_value(
        &self,
        goal_board: &TetrisBoardsTensor,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<(TetrisPieceOrientationLogitsTensor, Tensor)> {
        let (b, _t) = current_board.shape_tuple();

        // Encode boards as images [B,1,H,W] -> [B,D]
        let goal_img = goal_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;
        let cur_img = current_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;

        let goal_embed = self.board_encoder.forward(&goal_img)?; // [B, D]
        let cur_embed = self.board_encoder.forward(&cur_img)?; // [B, D]

        // Condition current on goal and piece
        let x = self.goal_film.forward(&cur_embed, &goal_embed)?; // [B, D]
        let piece_embed = self.piece_embedding.forward(current_piece)?.squeeze(1)?; // [B, D]
        let x = self.piece_film.forward(&x, &piece_embed)?; // [B, D]

        // Heads
        let logits = self.head_mlp.forward(&x)?; // [B, O]
        let value = self.value_mlp.forward(&x)?.squeeze(1)?; // [B]

        // Mask invalid orientations
        let mask = create_orientation_mask(current_piece)?; // [B,O] (0/1)
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(&logits, &neg_inf)?;
        Ok((TetrisPieceOrientationLogitsTensor::try_from(masked)?, value))
    }
}

/// Train a simple goal-conditioned placement policy with REINFORCE.
pub fn train_goal_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = Device::new_cuda(0).unwrap();

    // Hyperparameters
    const NUM_ITERATIONS: usize = 1_000_000;
    const BATCH_SIZE: usize = 64;
    const ACCUM_STEPS: usize = 4;
    const ROLLOUT_STEPS: usize = 64; // number of steps per trajectory
    const GAMMA: f32 = 0.98; // discount factor
    const CHECKPOINT_INTERVAL: usize = 10_000;
    let model_dim = 32;

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);

    let policy_cfg = TetrisGoalPolicyConfig {
        piece_embedding_dim: model_dim,
        board_encoder_config: ConvEncoderConfig {
            blocks: vec![
                ConvBlockSpec {
                    in_channels: 1,
                    out_channels: 32,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 32,
                },
                ConvBlockSpec {
                    in_channels: 32,
                    out_channels: 32,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 32,
                },
                ConvBlockSpec {
                    in_channels: 32,
                    out_channels: 16,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 16,
                },
                ConvBlockSpec {
                    in_channels: 16,
                    out_channels: 8,
                    kernel_size: 3,
                    conv_cfg: Conv2dConfig {
                        padding: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                    },
                    gn_groups: 8,
                },
            ],
            input_hw: (TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
            mlp: MlpConfig {
                hidden_size: 8 * TetrisBoard::HEIGHT * TetrisBoard::WIDTH,
                intermediate_size: 3 * model_dim,
                output_size: model_dim,
            },
        },
        goal_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: 2 * model_dim,
            output_dim: model_dim,
        },
        piece_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: 2 * model_dim,
            output_dim: model_dim,
        },
        head_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPieceOrientation::NUM_ORIENTATIONS,
        },
        value_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: 1,
        },
    };

    let policy = TetrisGoalPolicy::init(&vb, &policy_cfg)?;
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(model_params.clone(), ParamsAdamW::default())?;
    let mut grad_accum = GradientAccumulator::new(ACCUM_STEPS);

    // Summary + checkpointing (match transition model style)
    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let config_path = dir.join("policy_config.json");
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&policy_cfg).unwrap(),
        );
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    let generator = TetrisDatasetGenerator::new();
    let mut rng = rand::rng();

    {
        let seq_len = 12;
        for i in 0..10_000 {
            let datum_sequence =
                generator.gen_sequence((0..8).into(), BATCH_SIZE, seq_len, &device, &mut rng)?;

            let goal_board = datum_sequence.result_boards.last().unwrap().clone();

            // Precompute discounted returns along the supervised trajectory
            let mut pt_rewards: Vec<Tensor> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let current_board = &datum_sequence.current_boards[t];
                let next_board = &datum_sequence.result_boards[t];

                let eq_cur = current_board
                    .inner()
                    .eq(goal_board.inner())?
                    .to_dtype(DType::F32)?; // [B,T]
                let eq_next = next_board
                    .inner()
                    .eq(goal_board.inner())?
                    .to_dtype(DType::F32)?; // [B,T]
                let sim_cur = eq_cur.mean(D::Minus1)?; // [B]
                let sim_next = eq_next.mean(D::Minus1)?; // [B]
                let delta = (sim_next.clone() - &sim_cur)?; // [B]
                let sim_shape = sim_next.dims();
                let board_size_b =
                    Tensor::new(TetrisBoard::SIZE as f32, &device)?.broadcast_as(sim_shape)?; // [B]
                let exact = (eq_next.sum(D::Minus1)?)
                    .eq(&board_size_b)?
                    .to_dtype(DType::F32)?; // [B]
                let reward = ((&delta * 0.5)? + (&exact * 1.0)?)?; // [B]
                pt_rewards.push(reward);
            }
            let gamma_t = Tensor::new(GAMMA, &device)?.broadcast_as(pt_rewards[0].dims())?;
            let mut pt_returns: Vec<Tensor> = Vec::with_capacity(seq_len);
            let mut running = pt_rewards.last().unwrap().zeros_like()?; // [B]
            for t in (0..seq_len).rev() {
                running = (&pt_rewards[t] + &(running.clone() * &gamma_t)?)?; // [B]
                pt_returns.push(running.clone());
            }
            pt_returns.reverse();

            for t in 0..seq_len {
                let current_board = &datum_sequence.current_boards[t];
                let current_piece =
                    TetrisPieceTensor::from_pieces(&datum_sequence.pieces[t], &device)?;

                let (masked_logits, value_pred) = policy.forward_masked_with_value(
                    &goal_board,
                    &current_board,
                    &current_piece,
                )?;

                let mean_entropy_loss = masked_logits.into_dist()?.entropy()?.mean_all()?;
                let mean_entropy_loss_value = mean_entropy_loss.to_scalar::<f32>()?;

                let output_orientation = masked_logits;
                let target_orientation = &datum_sequence.orientations[t];

                let output_orientation_loss = cross_entropy(
                    &output_orientation
                        .inner()
                        .reshape(&[BATCH_SIZE, TetrisPieceOrientation::NUM_ORIENTATIONS])?,
                    &target_orientation.inner().reshape(&[BATCH_SIZE])?,
                )?;
                let output_orientation_loss_value = output_orientation_loss.to_scalar::<f32>()?;

                // Value pretrain loss to discounted return target
                let value_target = &pt_returns[t]; // [B]
                let v_err = (&value_pred - value_target)?; // [B]
                let value_loss = v_err.sqr()?.mean_all()?; // scalar
                let value_loss_value = value_loss.to_scalar::<f32>()?;

                // Joint loss
                let total_pretrain_loss =
                    (&output_orientation_loss + &value_loss.affine(0.5f64, 0.0)?)?;

                let grads = total_pretrain_loss.backward()?;
                let grad_norm = 0.0; // get_l2_norm(&grads, &model_params)?;

                info!(
                    "Iteration {} | OrLoss: {:.4} | VLoss: {:.4} | Entropy: {:.4} | Grad: {:.4}",
                    i,
                    output_orientation_loss_value,
                    value_loss_value,
                    mean_entropy_loss_value,
                    grad_norm
                );

                grad_accum.accumulate(grads, &model_params)?;

                summary_writer.as_mut().map(|s| {
                    s.add_scalar(
                        "pretrain/orientation_loss",
                        output_orientation_loss_value,
                        i,
                    );
                    s.add_scalar("pretrain/value_loss", value_loss_value, i);
                    s.add_scalar("pretrain/mean_entropy", mean_entropy_loss_value, i);
                    s.add_scalar("pretrain/grad_norm", grad_norm, i);
                });

                let should_step =
                    grad_accum.apply_and_reset(&mut optimizer, &model_params, Some(0.5_f64))?;

                if should_step {
                    debug!("Stepping model");
                }
            }
        }
    }

    // PPO hyperparameters
    const PPO_EPOCHS: usize = 4;
    const VF_COEF: f32 = 0.5;
    const ENTROPY_COEF: f32 = 0.01;
    const SURVIVE_REWARD: f32 = 0.01; // small reward for surviving a step
    const SHAPING_EXTRA_WEIGHT: f32 = 0.25; // reward for reducing extra blocks
    const SHAPING_MISSING_WEIGHT: f32 = 0.25; // reward for reducing missing blocks

    for i in 0..NUM_ITERATIONS {
        // Sample random goal board and random current board
        let goal_games =
            generator.gen_uniform_sampled_gameset((2..5).into(), BATCH_SIZE, &mut rng)?;
        let goal_board = TetrisBoardsTensor::from_gameset(goal_games, &device)?; // [B,T]

        let mut current_games =
            generator.gen_uniform_sampled_gameset((5..5).into(), BATCH_SIZE, &mut rng)?; // start random state

        // Rollout buffers
        let mut states_goal: Vec<TetrisBoardsTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut states_board: Vec<TetrisBoardsTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut states_piece: Vec<TetrisPieceTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut actions: Vec<TetrisPieceOrientationTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut logp_old: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS); // [B]
        let mut values_old: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS); // [B]
        let mut rewards: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS); // [B]

        // Rollout loop
        for _t in 0..ROLLOUT_STEPS {
            // State tensors
            let current_board = TetrisBoardsTensor::from_gameset(current_games, &device)?; // [B,T]
            let current_pieces_vec = current_games.current_pieces().to_vec();
            let current_piece =
                TetrisPieceTensor::from_pieces(&current_pieces_vec.as_slice(), &device)?; // [B,1]

            // Policy forward
            let (masked_logits, value) =
                policy.forward_masked_with_value(&goal_board, &current_board, &current_piece)?; // [B,O], [B]

            // Temperature 1.0 for PPO behavior and updates (detach to avoid graph retention)
            let tempered = masked_logits.inner().detach(); // [B,O]
            let logp_all = candle_nn::ops::log_softmax(&tempered, D::Minus1)?; // [B,O]

            // Sample actions
            let sampled_orientations = masked_logits.sample(1.0)?; // [B,1]
            let chosen_one_hot = TetrisPieceOrientationDistTensor::from_orientations_tensor(
                sampled_orientations.clone(),
            )?; // [B,O]

            // log π_old(a|s)
            let selected_log_prob = (logp_all.clone() * chosen_one_hot.inner())?.sum(D::Minus1)?; // [B]

            // Save rollout
            states_goal.push(goal_board.clone());
            states_board.push(current_board.clone());
            states_piece.push(current_piece.clone());
            actions.push(sampled_orientations.clone());
            logp_old.push(selected_log_prob.detach());
            values_old.push(value.detach());

            // Env step
            let placements: Vec<TetrisPiecePlacement> = current_pieces_vec
                .iter()
                .zip(sampled_orientations.into_orientations()?)
                .map(|(&piece, orientation)| TetrisPiecePlacement { piece, orientation })
                .collect();
            let lost_flags = current_games.apply_placement(&placements);
            let next_board = TetrisBoardsTensor::from_gameset(current_games, &device)?; // [B,T]

            // Dense reward: delta similarity + exact-match bonus
            let eq_cur = states_board
                .last()
                .unwrap()
                .inner()
                .eq(goal_board.inner())?
                .to_dtype(DType::F32)?; // [B,T]
            let eq_next = next_board
                .inner()
                .eq(goal_board.inner())?
                .to_dtype(DType::F32)?; // [B,T]
            let sim_cur = eq_cur.mean(D::Minus1)?; // [B]
            let sim_next = eq_next.mean(D::Minus1)?; // [B]
            let delta = (sim_next.clone() - &sim_cur)?; // [B]
            let sim_shape = sim_next.dims();
            let board_size_b =
                Tensor::new(TetrisBoard::SIZE as f32, &device)?.broadcast_as(sim_shape)?; // [B]
            let exact = (eq_next.sum(D::Minus1)?)
                .eq(&board_size_b)?
                .to_dtype(DType::F32)?; // [B]
            let reward_base = ((&delta * 0.5)? + (&exact * 1.0)?)?; // [B]

            // Line-clear–friendly shaping: reward reductions in "extra" and "missing" counts
            // extra = cells present in current but not in goal; missing = in goal but not in current
            let cur_f = current_board.inner().to_dtype(DType::F32)?; // [B,T]
            let goal_f = goal_board.inner().to_dtype(DType::F32)?; // [B,T]
            let next_f = next_board.inner().to_dtype(DType::F32)?; // [B,T]
            let one_cur = Tensor::new(1.0f32, &device)?.broadcast_as(cur_f.dims())?; // [B,T]

            let extra_cur = (&cur_f * &((&one_cur - &goal_f)?))?; // [B,T]
            let extra_next = (&next_f * &((&one_cur - &goal_f)?))?; // [B,T]
            let missing_cur = (&goal_f * &((&one_cur - &cur_f)?))?; // [B,T]
            let missing_next = (&goal_f * &((&one_cur - &next_f)?))?; // [B,T]

            let extra_cur_sum = extra_cur.sum(D::Minus1)?; // [B]
            let extra_next_sum = extra_next.sum(D::Minus1)?; // [B]
            let missing_cur_sum = missing_cur.sum(D::Minus1)?; // [B]
            let missing_next_sum = missing_next.sum(D::Minus1)?; // [B]

            let extra_delta = (&extra_cur_sum - &extra_next_sum)?; // [B] positive when removing extra
            let missing_delta = (&missing_cur_sum - &missing_next_sum)?; // [B] positive when filling missing

            let extra_term =
                (extra_delta / &board_size_b)?.affine(SHAPING_EXTRA_WEIGHT as f64, 0.0)?; // [B]
            let missing_term =
                (missing_delta / &board_size_b)?.affine(SHAPING_MISSING_WEIGHT as f64, 0.0)?; // [B]
            let shaping = (&extra_term + &missing_term)?; // [B]

            let reward_base = (&reward_base + &shaping)?; // [B]
            // Zero reward for lost games, add small survive bonus otherwise
            let not_lost_vec: Vec<f32> = lost_flags
                .into_iter()
                .map(|flag| {
                    let is_lost: bool = flag.into();
                    if is_lost { 0.0 } else { 1.0 }
                })
                .collect();
            let mask = Tensor::from_vec(
                not_lost_vec,
                Shape::from_dims(&[current_pieces_vec.len()]),
                &device,
            )?; // [B]
            let bonus = Tensor::new(SURVIVE_REWARD, &device)?.broadcast_as(mask.dims())?; // [B]
            let reward = ((&reward_base * &mask)? + (&mask * &bonus)?)?; // [B]
            rewards.push(reward);
        }

        // Compute discounted returns per step: R_t = r_t + gamma * R_{t+1}
        // Detach rewards so returns do not hold the forward graph
        let gamma_t = Tensor::new(GAMMA, &device)?.broadcast_as(rewards[0].dims())?;
        let mut returns: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut running = rewards.last().unwrap().detach().zeros_like()?; // [B]
        for t in (0..ROLLOUT_STEPS).rev() {
            running = (&rewards[t].detach() + &(running * &gamma_t)?)?; // [B]
            returns.push(running.clone());
        }
        returns.reverse();

        // Log average cumulative reward (undiscounted) across the rollout
        let mut avg_cumulative_reward: f32 = 0.0;
        for t in 0..ROLLOUT_STEPS {
            avg_cumulative_reward += rewards[t].mean_all()?.to_scalar::<f32>()?;
        }
        summary_writer
            .as_mut()
            .map(|s| s.add_scalar("rewards/avg_cumulative", avg_cumulative_reward, i));

        // Advantages: A_t = R_t - V_old(s_t)
        let mut advantages: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS);
        for t in 0..ROLLOUT_STEPS {
            let adv = (&returns[t].detach() - &values_old[t].detach())?; // [B]
            advantages.push(adv);
        }

        // PPO update over multiple epochs (full batch)
        let mut last_policy_loss_scalar: f32 = 0.0;
        let mut last_value_loss_scalar: f32 = 0.0;
        let mut last_entropy_scalar: f32 = 0.0;
        let mut last_kl_scalar: f32 = 0.0;

        for _epoch in 0..PPO_EPOCHS {
            let mut policy_loss_sum = Tensor::zeros((), DType::F32, &device)?;
            let mut value_loss_sum = Tensor::zeros((), DType::F32, &device)?;
            let mut entropy_sum = Tensor::zeros((), DType::F32, &device)?;
            let mut kl_sum = Tensor::zeros((), DType::F32, &device)?;

            for t in 0..ROLLOUT_STEPS {
                // Recompute logits/value under current policy
                let (masked_logits_new, value_new) = policy.forward_masked_with_value(
                    &states_goal[t],
                    &states_board[t],
                    &states_piece[t],
                )?; // [B,O], [B]
                // Use non-detached logits to allow gradients w.r.t. current policy
                let logp_all_new =
                    candle_nn::ops::log_softmax(masked_logits_new.inner(), D::Minus1)?; // [B,O]
                let probs_new = candle_nn::ops::softmax(masked_logits_new.inner(), D::Minus1)?; // [B,O]

                // Select log prob of taken action
                let chosen_one_hot =
                    TetrisPieceOrientationDistTensor::from_orientations_tensor(actions[t].clone())?; // [B,O]
                let logp_new = (logp_all_new.clone() * chosen_one_hot.inner())?.sum(D::Minus1)?; // [B]
                let logp_old_detached = logp_old[t].detach(); // [B]

                // Ratio
                let ratio = (logp_new.clone() - &logp_old_detached)?.exp()?; // [B]

                // Unclipped surrogate
                let surr1 = (&ratio * &advantages[t])?; // [B]
                let step_policy_loss = surr1.mean_all()?.neg()?; // scalar
                policy_loss_sum = (policy_loss_sum + step_policy_loss)?;

                // Value loss
                let v_err = (&value_new - &returns[t].detach())?; // [B]
                let v_mse = v_err.sqr()?.mean_all()?; // scalar
                value_loss_sum = (value_loss_sum + v_mse)?;

                // Entropy bonus from new policy
                let plogp = (&probs_new * &logp_all_new)?; // [B,O]
                let h = plogp.sum(D::Minus1)?.neg()?.mean_all()?; // scalar
                entropy_sum = (entropy_sum + h)?;

                // KL approx
                let kl_step = (&logp_old_detached - &logp_new)?.mean_all()?; // scalar
                kl_sum = (kl_sum + kl_step)?;
            }

            // Average across rollout steps
            let denom = Tensor::new(ROLLOUT_STEPS as f32, &device)?;
            let policy_loss = (policy_loss_sum / &denom)?;
            let value_loss = (value_loss_sum / &denom)?;
            let entropy = (entropy_sum / &denom)?;
            let kl = (kl_sum / &denom)?;

            // Total loss
            let vf_term = value_loss.affine(VF_COEF as f64, 0.0)?; // VF_COEF * value_loss
            let ent_term = entropy.affine(ENTROPY_COEF as f64, 0.0)?; // ENTROPY_COEF * entropy
            let total_loss = ((&policy_loss + &vf_term)? - &ent_term)?;

            let loss_value = total_loss.to_scalar::<f32>()?;
            last_policy_loss_scalar = policy_loss.to_scalar::<f32>()?;
            last_value_loss_scalar = value_loss.to_scalar::<f32>()?;
            last_entropy_scalar = entropy.to_scalar::<f32>()?;
            last_kl_scalar = kl.to_scalar::<f32>()?;

            // Backward + optimize (with grad accumulation)
            let grads = total_loss.backward()?;
            let grad_norm = get_l2_norm(&grads)?;
            grad_accum.accumulate(grads, &model_params)?;
            let stepped = grad_accum.apply_and_reset(&mut optimizer, &model_params, Some(1.0))?;

            // Logging
            summary_writer.as_mut().map(|s| {
                s.add_scalar("ppo/total_loss", loss_value, i);
                s.add_scalar("ppo/policy_loss", last_policy_loss_scalar, i);
                s.add_scalar("ppo/value_loss", last_value_loss_scalar, i);
                s.add_scalar("ppo/entropy", last_entropy_scalar, i);
                s.add_scalar("ppo/kl", last_kl_scalar, i);
                s.add_scalar("grad_norm", grad_norm, i);
            });

            // checkpoint each epoch end
            if let Some(ref checkpointer) = checkpointer {
                let _ = checkpointer.checkpoint_item(i, &model_varmap, None, None);
            }

            info!(
                "[{}] ppo total {:>8.4} | pol {:>8.4} | val {:>8.4} | ent {:>7.4} | kl {:>7.4}{}",
                i,
                loss_value,
                last_policy_loss_scalar,
                last_value_loss_scalar,
                last_entropy_scalar,
                last_kl_scalar,
                if stepped { " | step" } else { "" }
            );
        }
    }

    // final checkpoint
    if let Some(ref checkpointer) = checkpointer {
        let _ = checkpointer.force_checkpoint_item(NUM_ITERATIONS, &model_varmap, None, None);
    }

    Ok(())
}
