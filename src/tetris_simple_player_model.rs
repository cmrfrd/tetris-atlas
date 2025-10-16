use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Device, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
// use rand::Rng; // no longer needed
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::info;

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

/// Simple goal-conditioned policy over placements using REINFORCE (vanilla policy gradient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGoalPolicyConfig {
    pub piece_embedding_dim: usize,

    pub board_encoder_config: ConvEncoderConfig,

    pub piece_film_config: FiLMConfig,

    pub head_mlp_config: MlpConfig, // outputs logits over orientations
    pub value_mlp_config: MlpConfig, // outputs scalar value
}

#[derive(Debug, Clone)]
pub struct TetrisGoalPolicy {
    piece_embedding: Embedding,
    board_encoder: ConvEncoder,
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
        let piece_film = FiLM::init(&vb.pp("piece_film"), &cfg.piece_film_config)?;
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;
        let value_mlp = Mlp::init(&vb.pp("value_mlp"), &cfg.value_mlp_config)?;
        Ok(Self {
            piece_embedding,
            board_encoder,
            piece_film,
            head_mlp,
            value_mlp,
        })
    }

    /// Forward producing unmasked orientation logits [B, NUM_ORIENTATIONS]
    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let (b, _t) = current_board.shape_tuple();

        // Encode boards as images [B,1,H,W] -> [B,D]
        let cur_img = current_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;
        let cur_embed = self.board_encoder.forward(&cur_img)?; // [B, D]

        // Condition current on goal and piece
        let piece_embed = self.piece_embedding.forward(current_piece)?.squeeze(1)?; // [B, D]
        let x = self.piece_film.forward(&cur_embed, &piece_embed)?; // [B, D]

        // Head -> orientation logits [B, NUM_ORIENTATIONS]
        let logits = self.head_mlp.forward(&x)?; // [B, O]
        TetrisPieceOrientationLogitsTensor::try_from(logits)
    }

    /// Compute masked action distribution over orientations, masking to those valid for current piece.
    pub fn forward_masked(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(current_board, current_piece)?; // [B,O]
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
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<(TetrisPieceOrientationLogitsTensor, Tensor)> {
        let (b, _t) = current_board.shape_tuple();

        // Encode boards as images [B,1,H,W] -> [B,D]
        let cur_img = current_board
            .reshape(&[b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH])?
            .to_dtype(DType::F32)?;
        let cur_embed = self.board_encoder.forward(&cur_img)?; // [B, D]

        // Condition current on goal and piece
        let piece_embed = self.piece_embedding.forward(current_piece)?.squeeze(1)?; // [B, D]
        let x = self.piece_film.forward(&cur_embed, &piece_embed)?; // [B, D]

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

/// Train a simple goal-conditioned placement policy with REINFORCE (vanilla policy gradient).
pub fn train_goal_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = Device::new_cuda(0).unwrap();

    // Hyperparameters - optimized for STABLE training (prioritizing consistency over efficiency)
    const NUM_ITERATIONS: usize = 200_000;
    const BATCH_SIZE: usize = 32; // Large batch = more stable gradient estimates
    const ACCUM_STEPS: usize = 8; // No accumulation = simpler, more stable
    const ROLLOUT_STEPS: usize = 32; // Longer rollouts for better credit assignment
    const CHECKPOINT_INTERVAL: usize = 10_000;
    const CLIP_GRAD_MAX_NORM: f64 = 1.0; // Gradient clipping for stability
    const SAMPLE_TEMPERATURE: f32 = 0.1; // Higher temperature = more exploration = more stable
    let model_dim = 64; // Increased capacity for complex patterns

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
    let datum_rx =
        generator.spawn_transition_channel((3..7).into(), BATCH_SIZE, device.clone(), 64);

    // Hyperparameters - STABILITY-FOCUSED SETUP
    const GAMMA: f32 = 0.99; // Discount factor - helps with credit assignment
    const VF_COEF: f32 = 0.8;
    const ENTROPY_BONUS_COEF: f32 = 0.0; // BONUS for exploration (not penalty!)
    const ADVANTAGE_CLIP_RANGE: f32 = 10.0; // Larger clip range = allow bigger updates
    const VALUE_CLIP_RANGE: f32 = 100.0; // Clip value predictions to prevent explosion

    for i in 0..NUM_ITERATIONS {
        // Sample random goal board and random current board
        let span = tracing::info_span!("datum_recv", iteration = i, batch = BATCH_SIZE);
        let _enter = span.enter();
        let datum = datum_rx.recv().expect("prefetch thread stopped");
        let mut current_games = datum.current_gameset;

        // Pre-allocate rollout buffers for better performance
        // Store inputs (boards, pieces) for recomputation rather than outputs (logits, values with gradients)
        // This enables arbitrarily long rollouts without GPU memory issues
        let mut stored_boards: Vec<TetrisBoardsTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut stored_pieces: Vec<TetrisPieceTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut actions: Vec<TetrisPieceOrientationTensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut values: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS); // Value predictions V(s) - detached
        let mut rewards: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS);
        let mut dones: Vec<Tensor> = Vec::with_capacity(ROLLOUT_STEPS); // Terminal flags [B] per timestep
        let mut dead_games: Vec<bool> = vec![false; BATCH_SIZE]; // Track which games are dead
        let mut total_lines_cleared: Vec<usize> = vec![0; BATCH_SIZE]; // Track total lines cleared

        // Store previous board heights to calculate lines cleared
        let mut prev_heights: Vec<usize> = current_games
            .boards()
            .map(|board| board.height() as usize)
            .to_vec();

        // Rollout loop
        for _t in 0..ROLLOUT_STEPS {
            // State tensors
            let current_board = TetrisBoardsTensor::from_gameset(current_games, &device)?; // [B,T]
            let current_pieces_vec = current_games.current_pieces().to_vec();
            let current_piece =
                TetrisPieceTensor::from_pieces(&current_pieces_vec.as_slice(), &device)?; // [B,1]

            // Store inputs for later recomputation during loss calculation
            stored_boards.push(current_board.clone());
            stored_pieces.push(current_piece.clone());

            // Policy forward with value prediction
            let (masked_logits, value) =
                policy.forward_masked_with_value(&current_board, &current_piece)?; // [B,O], [B]

            // Detach outputs immediately to free computational graph (memory efficiency for long rollouts)
            let masked_logits_detached =
                TetrisPieceOrientationLogitsTensor::try_from(masked_logits.inner().detach())?;
            let value_detached = value.detach();

            // Sample actions with temperature (using detached logits)
            let sampled_orientations = masked_logits_detached.sample(SAMPLE_TEMPERATURE)?; // [B,1]

            // Save rollout data - only detached tensors (no computational graph stored)
            actions.push(sampled_orientations.clone());
            values.push(value_detached); // Store detached value for advantage calculation

            // Env step
            let placements: Vec<TetrisPiecePlacement> = current_pieces_vec
                .iter()
                .zip(sampled_orientations.into_orientations()?)
                .map(|(&piece, orientation)| TetrisPiecePlacement { piece, orientation })
                .collect();
            let lost_flags = current_games.apply_placement(&placements);

            // Get new heights to calculate lines cleared
            let new_heights: Vec<usize> = current_games
                .boards()
                .map(|board| board.height() as usize)
                .to_vec();

            // Calculate rewards - only reward when lines are cleared
            let step_rewards: Vec<f32> = (0..BATCH_SIZE)
                .map(|i| {
                    if dead_games[i] {
                        // Already dead from previous step - no reward
                        0.0
                    } else {
                        // Calculate how many lines were cleared this step
                        let lines_cleared = prev_heights[i].saturating_sub(new_heights[i]) as f32;

                        // Reward for clearing lines + small bonus for staying alive
                        // 1 line = 1.0 reward
                        // 2 lines = 2.0 reward
                        // 3 lines = 3.0 reward
                        // 4 lines (Tetris) = 4.0 reward
                        // Staying alive = 0.1 reward (encourages exploration)
                        lines_cleared * 2.0 + 0.1
                    }
                })
                .collect();

            let reward = Tensor::from_vec(step_rewards, Shape::from_dims(&[BATCH_SIZE]), &device)?;
            rewards.push(reward);

            // NOW update piece counts and dead games tracker AFTER reward calculation
            for (i, flag) in lost_flags.iter().enumerate() {
                if !dead_games[i] {
                    // Calculate lines cleared
                    let lines_cleared = prev_heights[i].saturating_sub(new_heights[i]);
                    total_lines_cleared[i] += lines_cleared;

                    if bool::from(*flag) {
                        // Mark as dead for future steps
                        dead_games[i] = true;
                    }
                }
            }

            // Update previous heights for next iteration
            prev_heights = new_heights;

            // Store terminal flags (1.0 = done, 0.0 = not done)
            let done_flags: Vec<f32> = dead_games
                .iter()
                .map(|&is_dead| if is_dead { 1.0 } else { 0.0 })
                .collect();
            let done_tensor =
                Tensor::from_vec(done_flags, Shape::from_dims(&[BATCH_SIZE]), &device)?;
            dones.push(done_tensor);

            // Early termination: if all games are dead, no point continuing rollout
            if dead_games.iter().all(|&d| d) {
                break;
            }
        }

        // Track actual rollout length (may be less than ROLLOUT_STEPS if early termination)
        let actual_rollout_steps = actions.len();

        // Compute discounted returns for credit assignment
        let mut returns: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);

        // Compute returns backwards with discounting
        let batch_zeros = Tensor::zeros((BATCH_SIZE,), DType::F32, &device)?;
        let mut running_return = batch_zeros;

        for t in (0..actual_rollout_steps).rev() {
            // done[t] represents if state s_{t+1} is terminal (after taking action at t)
            let done_t = &dones[t]; // [B] (1.0 = done, 0.0 = not done)
            let not_done_t = (Tensor::new(1.0f32, &device)?.broadcast_as(done_t.dims())? - done_t)?; // [B] (0.0 = done, 1.0 = not done)

            // R_t = r_t + γ * (1 - done_t) * R_{t+1}
            // Discounting helps with credit assignment - more recent actions matter more
            let future_return = (&running_return * &not_done_t)?.affine(GAMMA as f64, 0.0)?;
            running_return = (&rewards[t] + future_return)?;

            // Detach returns to prevent gradients flowing back through reward computation
            returns.push(running_return.detach());
        }

        returns.reverse();

        // Compute advantages = Returns - Value predictions (baseline)
        // This is key for credit assignment: tells us which actions were better than expected
        let mut advantages: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);
        for t in 0..actual_rollout_steps {
            let advantage = (&returns[t] - &values[t].detach())?; // Detach value to not affect its gradient here
            advantages.push(advantage);
        }

        // Clip advantages to prevent extreme policy updates (STABILITY IMPROVEMENT)
        // This prevents catastrophic forgetting when rare large advantages occur
        let mut advantages_clipped: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);
        for t in 0..actual_rollout_steps {
            let adv_clipped = advantages[t].clamp(-ADVANTAGE_CLIP_RANGE, ADVANTAGE_CLIP_RANGE)?;
            advantages_clipped.push(adv_clipped);
        }
        let advantages = advantages_clipped; // Use clipped advantages directly (no standardization for max stability)

        // Extract raw data for histogram logging
        let returns_stacked = Tensor::stack(&returns, 0)?; // [T, B]
        let returns_data: Vec<f64> = returns_stacked
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        let values_stacked = Tensor::stack(&values, 0)?; // [T, B]
        let values_data: Vec<f64> = values_stacked
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        let advantages_stacked = Tensor::stack(&advantages, 0)?; // [T, B]
        let advantages_data: Vec<f64> = advantages_stacked
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        // Track advantage clipping statistics
        let num_advantages_total = advantages_data.len() as f32;
        let num_advantages_clipped = advantages_data
            .iter()
            .filter(|&&a| a.abs() >= ADVANTAGE_CLIP_RANGE as f64 - 0.01) // Small epsilon for float comparison
            .count() as f32;
        let advantage_clip_fraction = num_advantages_clipped / num_advantages_total;

        // Track advantage magnitude (detect value function overfitting)
        let advantage_mean_abs: f64 =
            advantages_data.iter().map(|&a| a.abs()).sum::<f64>() / num_advantages_total as f64;
        let advantage_std: f64 = {
            let mean: f64 = advantages_data.iter().sum::<f64>() / num_advantages_total as f64;
            let variance: f64 = advantages_data
                .iter()
                .map(|&a| (a - mean).powi(2))
                .sum::<f64>()
                / num_advantages_total as f64;
            variance.sqrt()
        };

        // Calculate value function explained variance: 1 - Var(advantages) / Var(returns)
        // High explained variance (>0.9) with low advantage magnitude indicates overfitting
        let returns_std: f64 = {
            let mean: f64 = returns_data.iter().sum::<f64>() / returns_data.len() as f64;
            let variance: f64 = returns_data
                .iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>()
                / returns_data.len() as f64;
            variance.sqrt()
        };
        let explained_variance = if returns_std > 1e-8 {
            1.0 - (advantage_std / returns_std)
        } else {
            0.0
        };

        // Extract reward data for histogram logging
        let rewards_stacked = Tensor::stack(&rewards, 0)?; // [T, B]
        let rewards_data: Vec<f64> = rewards_stacked
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        let total_batch_reward = rewards_data.iter().sum::<f64>() as f32;
        let avg_reward_per_timestep =
            total_batch_reward / (actual_rollout_steps * BATCH_SIZE) as f32;
        let total_death_penalties = rewards_data.iter().filter(|&&r| r < 0.0).count();
        let death_rate = total_death_penalties as f32 / (actual_rollout_steps * BATCH_SIZE) as f32;

        summary_writer.as_mut().map(|s| {
            // Scalar summaries for quick reference
            s.add_scalar("rewards/total_batch", total_batch_reward, i);
            s.add_scalar("rewards/avg_per_timestep", avg_reward_per_timestep, i);
            s.add_scalar("rewards/death_rate", death_rate, i);

            // Histogram distributions for detailed analysis
            s.add_histogram("distributions/rewards", &rewards_data, None, i);
            s.add_histogram("distributions/returns", &returns_data, None, i);
            s.add_histogram("distributions/values", &values_data, None, i);
            s.add_histogram("distributions/advantages", &advantages_data, None, i);

            // Advantage clipping statistics (for stability monitoring)
            s.add_scalar(
                "stability/advantage_clip_fraction",
                advantage_clip_fraction,
                i,
            );

            // Value function overfitting diagnostics
            s.add_scalar("stability/advantage_mean_abs", advantage_mean_abs as f32, i);
            s.add_scalar("stability/advantage_std", advantage_std as f32, i);
            s.add_scalar("stability/explained_variance", explained_variance as f32, i);

            // Rollout statistics (early termination tracking)
            s.add_scalar("rollout/actual_steps", actual_rollout_steps as f32, i);
            s.add_scalar(
                "rollout/utilization",
                actual_rollout_steps as f32 / ROLLOUT_STEPS as f32,
                i,
            );
        });

        // Piece count logging - get directly from gameset
        let piece_counts_from_games = current_games.piece_counts().to_vec();
        let mean_pieces = piece_counts_from_games.iter().sum::<u32>() as f32 / BATCH_SIZE as f32;

        // Convert piece counts to histogram data
        let piece_counts_data: Vec<f64> = piece_counts_from_games
            .iter()
            .map(|&count| count as f64)
            .collect::<Vec<_>>();

        // Count how many games survived the full rollout
        let games_alive = dead_games.iter().filter(|&&d| !d).count() as f32;
        let survival_rate = games_alive / BATCH_SIZE as f32;

        summary_writer.as_mut().map(|s| {
            s.add_scalar("pieces/mean", mean_pieces, i);
            s.add_scalar("pieces/survival_rate", survival_rate, i);
            s.add_histogram("distributions/pieces", &piece_counts_data, Some(8), i);
        });

        // Compute loss from stored rollout data (recompute forward pass with gradients)
        let mut policy_loss_tensors: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);
        let mut value_loss_tensors: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);
        let mut entropy_tensors: Vec<Tensor> = Vec::with_capacity(actual_rollout_steps);

        for t in 0..actual_rollout_steps {
            // Recompute forward pass with gradients enabled for backprop
            let (stored_logits, stored_value) =
                policy.forward_masked_with_value(&stored_boards[t], &stored_pieces[t])?;

            // Compute log probabilities for gradient flow
            let logp_all = candle_nn::ops::log_softmax(stored_logits.inner(), D::Minus1)?;

            // Get log prob of the action that was taken
            let chosen_one_hot =
                TetrisPieceOrientationDistTensor::from_orientations_tensor(actions[t].clone())?;
            let action_log_prob = (logp_all.clone() * chosen_one_hot.inner())?.sum(D::Minus1)?;

            // Policy objective: log π(a|s) * advantage
            // Advantages are already detached (computed from detached returns and values)
            let adv_t = &advantages[t]; // [B]
            let policy_obj = (&action_log_prob * adv_t)?.mean_all()?; // scalar
            policy_loss_tensors.push(policy_obj);

            // Value loss: MSE between predicted value and actual return
            let return_t = &returns[t]; // [B]
            let value_error = (&stored_value - return_t)?;
            let value_error_clipped = value_error.clamp(-VALUE_CLIP_RANGE, VALUE_CLIP_RANGE)?;
            let value_loss = (&value_error_clipped * &value_error_clipped)?.mean_all()?;
            value_loss_tensors.push(value_loss);

            // Entropy
            let probs = candle_nn::ops::softmax(stored_logits.inner(), D::Minus1)?;
            let entropy_t = (&probs * &logp_all)?.sum(D::Minus1)?.neg()?.mean_all()?;
            entropy_tensors.push(entropy_t);
        }

        // Stack and mean the objectives (keep as tensors for autograd)
        let policy_objective = Tensor::stack(&policy_loss_tensors, 0)?.mean_all()?;
        let value_loss = Tensor::stack(&value_loss_tensors, 0)?.mean_all()?;
        let entropy = Tensor::stack(&entropy_tensors, 0)?.mean_all()?;

        // Total loss: -policy_objective + value_loss - entropy_bonus
        let total_loss = (&policy_objective.neg()? + &value_loss.affine(VF_COEF as f64, 0.0)?
            - &entropy.affine(ENTROPY_BONUS_COEF as f64, 0.0)?)?;

        // Backward + optimize
        let grads = total_loss.backward()?;
        let last_grad_norm = get_l2_norm(&grads)?;
        grad_accum.accumulate(grads, &model_params)?;
        let _stepped =
            grad_accum.apply_and_reset(&mut optimizer, &model_params, Some(CLIP_GRAD_MAX_NORM))?;

        // Extract scalars for logging
        let policy_loss = policy_objective.neg()?.to_scalar::<f32>()?;
        let value_loss_scalar = value_loss.to_scalar::<f32>()?;
        let entropy_scalar = entropy.to_scalar::<f32>()?;
        let total_loss_scalar =
            policy_loss + VF_COEF * value_loss_scalar - ENTROPY_BONUS_COEF * entropy_scalar;

        // Logging
        summary_writer.as_mut().map(|s| {
            s.add_scalar("reinforce/total_loss", total_loss_scalar, i);
            s.add_scalar("reinforce/policy_loss", policy_loss, i);
            s.add_scalar("reinforce/value_loss", value_loss_scalar, i);
            s.add_scalar("reinforce/entropy", entropy_scalar, i);
            s.add_scalar(
                "reinforce/learning_rate",
                optimizer.learning_rate() as f32,
                i,
            );
            s.add_scalar("grad_norm", last_grad_norm, i);
        });

        // Checkpoint
        if let Some(ref checkpointer) = checkpointer {
            let _ = checkpointer.checkpoint_item(i, &model_varmap, None, None);
        }

        info!(
            "[{}] reinforce total {:>8.4} | pol {:>8.4} | val {:>8.4} | lr {:.1e} | pieces: μ={:>5.1} | grad_norm={:.4}",
            i,
            total_loss_scalar,
            policy_loss,
            value_loss_scalar,
            optimizer.learning_rate(),
            mean_pieces,
            last_grad_norm
        );
    }

    // final checkpoint
    if let Some(ref checkpointer) = checkpointer {
        let _ = checkpointer.force_checkpoint_item(NUM_ITERATIONS, &model_varmap, None, None);
    }

    Ok(())
}
