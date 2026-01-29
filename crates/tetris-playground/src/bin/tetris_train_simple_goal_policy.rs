use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Device, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use tensorboard::summary_writer::SummaryWriter;
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement};
use tetris_ml::{
    checkpointer::Checkpointer,
    device,
    grad_accum::GradientAccumulator,
    modules::{
        AttnMlp, AttnMlpConfig, Conv2dConfig, ConvBlockSpec, ConvEncoder, ConvEncoderConfig, FiLM,
        FiLMConfig,
    },
    set_global_threadpool,
};
use tetris_ml::ops::{create_orientation_mask, get_l2_norm};
use tetris_ml::tensors::{
    TetrisBoardsTensor, TetrisPieceOrientationDistTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceTensor,
};
use tetris_ml::wrapped_tensor::WrappedTensor;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

/// Simple goal-conditioned policy over placements using REINFORCE (vanilla policy gradient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGoalPolicyConfig {
    pub piece_embedding_dim: usize,

    pub board_encoder_config: ConvEncoderConfig,

    pub piece_film_config: FiLMConfig,

    pub head_mlp_config: AttnMlpConfig, // outputs logits over orientations
    pub value_mlp_config: AttnMlpConfig, // outputs scalar value
}

#[derive(Debug, Clone)]
pub struct TetrisGoalPolicy {
    piece_embedding: Embedding,
    board_encoder: ConvEncoder,
    piece_film: FiLM,
    head_mlp: AttnMlp,
    value_mlp: AttnMlp,
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
        let head_mlp = AttnMlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;
        let value_mlp = AttnMlp::init(&vb.pp("value_mlp"), &cfg.value_mlp_config)?;
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

/// Compute curriculum-based rewards proportional to distance from mean
///
/// Reward structure:
/// - Rewards are proportional to distance from curriculum mean_game_length
/// - Games above mean get positive rewards: (count - mean) / mean
/// - Games below mean get negative rewards: (count - mean) / mean
/// - Clipped to [-1.0, 2.0] (asymmetric to encourage exceeding the target)
/// - Still alive: 0.0 (reward only at termination)
///
/// This provides continuous gradient signal that adapts to curriculum difficulty
fn compute_reward(
    lost_flags: &[tetris_game::IsLost],
    piece_counts: &[u32],
    mean_game_length: f32,
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let rewards: Vec<f32> = lost_flags
        .iter()
        .zip(piece_counts.iter())
        .map(|(&flag, &count)| {
            if bool::from(flag) {
                // Continuous reward scaled by relative distance from curriculum mean
                let distance = count as f32 - mean_game_length;
                let normalized = distance / mean_game_length.max(1.0);
                normalized.clamp(0.0, 1.0) // Asymmetric: bigger rewards for exceeding
            } else {
                0.0
            }
        })
        .collect();

    Ok(Tensor::from_vec(
        rewards,
        Shape::from_dims(&[batch_size]),
        device,
    )?)
}

/// Create a tensor indicating which games are in terminal states
///
/// Returns [B] tensor where 1.0 = done (game lost), 0.0 = not done
fn create_done_tensor(
    lost_flags: &[tetris_game::IsLost],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let done_values: Vec<f32> = lost_flags
        .iter()
        .map(|&flag| if bool::from(flag) { 1.0 } else { 0.0 })
        .collect();

    Ok(Tensor::from_vec(
        done_values,
        Shape::from_dims(&[batch_size]),
        device,
    )?)
}

/// Compute advantages using Generalized Advantage Estimation (GAE)
///
/// GAE formula:
///   δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
///   A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
///
/// # Arguments
/// * `rewards` - Vector of reward tensors [B] for each timestep
/// * `values` - Vector of value predictions [B] for each timestep
/// * `dones` - Vector of terminal flags [B] for each timestep
/// * `bootstrap_value` - Value prediction [B] for final state (s_{T+1})
/// * `gamma` - Discount factor (typically 0.99)
/// * `gae_lambda` - GAE lambda parameter (typically 0.95)
fn compute_gae_advantages(
    rewards: &[Tensor],
    values: &[Tensor],
    dones: &[Tensor],
    bootstrap_value: Tensor,
    gamma: f32,
    gae_lambda: f32,
) -> Result<Vec<Tensor>> {
    let num_steps = rewards.len();

    let mut advantages = Vec::with_capacity(num_steps);
    let mut gae = Tensor::zeros_like(&bootstrap_value)?;

    // Compute GAE backwards from T to 0
    for t in (0..num_steps).rev() {
        // Get next value: V(s_{t+1})
        let next_value = if t == num_steps - 1 {
            &bootstrap_value // Bootstrap from final state
        } else {
            &values[t + 1]
        };

        // Compute mask: (1 - done_t)
        // If done_t=1, then next_value should be 0 (no future returns)
        let not_done = (Tensor::ones_like(&dones[t])? - &dones[t])?;

        // Compute TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        let next_value_masked = (next_value * &not_done)?;
        let delta = (
            &rewards[t] +                                    // r_t
            next_value_masked.affine(gamma as f64, 0.0)? -   // γ * V(s_{t+1}) * (1 - done)
            &values[t]
            // -V(s_t)
        )?;

        // Compute GAE: A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
        let gae_masked = (&gae * &not_done)?;
        gae = (&delta + gae_masked.affine((gamma * gae_lambda) as f64, 0.0)?)?;

        advantages.push(gae.clone());
    }

    // Reverse to get chronological order [0..T]
    advantages.reverse();

    Ok(advantages)
}

/// Normalize advantages across batch and time for stable gradients
fn normalize_advantages(advantages: &[Tensor]) -> Result<Vec<Tensor>> {
    // Stack to [T, B]
    let stacked = Tensor::stack(advantages, 0)?;
    let shape = stacked.shape().clone();

    // Compute mean and std across all (timestep, game) pairs
    let mean = stacked.mean_all()?;

    // Compute variance: E[(x - mean)^2]
    let centered = (stacked - mean.broadcast_as(&shape)?)?;
    let variance = (&centered * &centered)?.mean_all()?;
    let std = (variance + 1e-8)?.sqrt()?; // Add epsilon for numerical stability

    // Normalize: (A - mean) / std
    let normalized_stacked = (centered / std.broadcast_as(&shape)?)?;

    // Unstack back to Vec<Tensor>
    let mut result = Vec::with_capacity(advantages.len());
    for t in 0..advantages.len() {
        result.push(normalized_stacked.get(t)?);
    }

    Ok(result)
}

/// Train a simple goal-conditioned placement policy with streaming rollouts
pub fn train_goal_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = device();

    // Hyperparameters
    const NUM_TOTAL_GAMES: usize = 10_000_000;
    const MIN_UPDATE_INTERVAL: usize = 64;
    const MAX_UPDATE_INTERVAL: usize = 64;
    const UPDATE_INTERVAL_SCALE: f32 = 1.0;
    const BATCH_SIZE: usize = 64;
    const ACCUM_STEPS: usize = 1;
    const CHECKPOINT_INTERVAL_GAMES: usize = 5_000;
    const CLIP_GRAD_MAX_VALUE: f64 = 5.0;
    const CLIP_GRAD_MAX_NORM: f64 = 1.0;
    const INITIAL_TEMPERATURE: f32 = 1.0;
    const FINAL_TEMPERATURE: f32 = 0.9;
    const TEMPERATURE_HALFLIFE_GAMES: f32 = 1_000_000.0;
    const GAMMA: f32 = 0.99;
    const GAE_LAMBDA: f32 = 0.90;
    const VF_COEF: f32 = 1.0;
    const POLICY_CLIP: f32 = 0.5;
    const INITIAL_ENTROPY_COEF: f32 = 0.1;
    const FINAL_ENTROPY_COEF: f32 = 0.01;
    const ENTROPY_HALFLIFE_GAMES: f32 = 100_000.0;
    const MIN_ENTROPY: f32 = 0.1;
    let model_dim = 64;

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, tetris_ml::fdtype(), &device);

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
            mlp: AttnMlpConfig {
                hidden_size: 8 * TetrisBoard::HEIGHT * TetrisBoard::WIDTH,
                intermediate_size: 3 * model_dim,
                output_size: model_dim,
                dropout: None,
            },
        },
        piece_film_config: FiLMConfig {
            cond_dim: model_dim,
            feat_dim: model_dim,
            hidden: 3 * model_dim,
            output_dim: model_dim,
        },
        head_mlp_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 3 * model_dim,
            output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            dropout: None,
        },
        value_mlp_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 3 * model_dim,
            output_size: 1,
            dropout: None,
        },
    };

    let policy = TetrisGoalPolicy::init(&vb, &policy_cfg)?;
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(model_params.clone(), ParamsAdamW::default())?;
    let mut grad_accum = GradientAccumulator::new(ACCUM_STEPS);

    // Summary + checkpointing
    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let config_path = dir.join("policy_config.json");
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&policy_cfg).unwrap(),
        );
        Checkpointer::new(CHECKPOINT_INTERVAL_GAMES, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    // Initialize rolling batch
    let mut current_games = tetris_game::TetrisGameSet::new(BATCH_SIZE);
    let mut total_games_completed = 0;
    let mut update_count = 0;

    // Curriculum learning: track rolling mean game length
    let mut mean_game_length: f32 = 8.0; // Start at minimum threshold
    const MIN_THRESHOLD: f32 = 8.0;
    const SMOOTHING: f32 = 0.9995; // Exponential moving average weight (higher = slower adaptation)

    println!(
        "Starting training: target {} games, batch size {}",
        NUM_TOTAL_GAMES, BATCH_SIZE
    );
    println!(
        "Dynamic UPDATE_INTERVAL: min={}, scale={:.1}× mean_game_length | {} gradient accumulation steps",
        MIN_UPDATE_INTERVAL, UPDATE_INTERVAL_SCALE, ACCUM_STEPS
    );
    println!(
        "Temperature decay: {:.2} → {:.2} (exponential, halflife={:.0} games)",
        INITIAL_TEMPERATURE, FINAL_TEMPERATURE, TEMPERATURE_HALFLIFE_GAMES
    );
    println!(
        "Curriculum rewards: continuous (count-mean)/mean, clipped to [-1.0, 2.0] | Starting mean: {:.1} pieces",
        mean_game_length
    );
    println!(
        "Entropy bonus: {:.3} → {:.3} (exponential, halflife={:.0} games) | Smoothing: {:.3}",
        INITIAL_ENTROPY_COEF, FINAL_ENTROPY_COEF, ENTROPY_HALFLIFE_GAMES, SMOOTHING
    );

    // Main training loop
    while total_games_completed < NUM_TOTAL_GAMES {
        // === Calculate exponentially decaying temperature ===
        // Temperature decays exponentially: T(t) = T_final + (T_initial - T_final) * exp(-ln(2) * t / halflife)
        let decay_factor =
            (-0.693147 * total_games_completed as f32 / TEMPERATURE_HALFLIFE_GAMES).exp();
        let temperature = (FINAL_TEMPERATURE
            + (INITIAL_TEMPERATURE - FINAL_TEMPERATURE) * decay_factor)
            .max(FINAL_TEMPERATURE);

        // === Calculate exponentially decaying entropy coefficient ===
        let entropy_decay_factor =
            (-0.693147 * total_games_completed as f32 / ENTROPY_HALFLIFE_GAMES).exp();
        let entropy_coef = (FINAL_ENTROPY_COEF
            + (INITIAL_ENTROPY_COEF - FINAL_ENTROPY_COEF) * entropy_decay_factor)
            .max(FINAL_ENTROPY_COEF);

        // === Calculate dynamic UPDATE_INTERVAL based on mean game length ===
        let update_interval = ((UPDATE_INTERVAL_SCALE * mean_game_length).ceil() as usize)
            .max(MIN_UPDATE_INTERVAL)
            .min(MAX_UPDATE_INTERVAL);

        // === Collect mini-rollout (update_interval steps) ===
        let mut stored_logits = Vec::with_capacity(update_interval);
        let mut stored_values = Vec::with_capacity(update_interval);
        let mut actions = Vec::with_capacity(update_interval);
        let mut values_detached = Vec::with_capacity(update_interval);
        let mut rewards = Vec::with_capacity(update_interval);
        let mut dones = Vec::with_capacity(update_interval);

        for _t in 0..update_interval {
            // Forward pass (all games regardless of their age)
            let current_board = TetrisBoardsTensor::from_gameset(&current_games, &device)?;
            let current_pieces_vec = current_games.current_pieces().to_vec();
            let current_piece = TetrisPieceTensor::from_pieces(&current_pieces_vec, &device)?;

            // Forward pass WITH gradients - only do this once!
            let (masked_logits, value) =
                policy.forward_masked_with_value(&current_board, &current_piece)?;

            // Store tensors WITH gradients for loss computation later
            stored_logits.push(masked_logits.clone());
            stored_values.push(value.clone());

            // Store detached copy for GAE computation (no gradients needed there)
            values_detached.push(value.detach());

            // Sample actions - sampling doesn't need gradients, it's stochastic
            let sampled_orientations = masked_logits.sample(temperature, &current_piece)?;
            actions.push(sampled_orientations.clone());

            // Execute actions
            let placements: Vec<TetrisPiecePlacement> = current_pieces_vec
                .iter()
                .zip(sampled_orientations.into_orientations()?)
                .map(|(&piece, orientation)| TetrisPiecePlacement { piece, orientation })
                .collect();
            let lost_flags = current_games.apply_placement(&placements);

            // Get final piece counts BEFORE reset (complete game length for dying games)
            let final_piece_counts = current_games.piece_counts();
            let final_piece_counts_vec: Vec<u32> = final_piece_counts.iter().copied().collect();

            // Update curriculum mean for games that just completed
            for (i, &is_lost) in lost_flags.iter().enumerate() {
                if bool::from(is_lost) {
                    let final_length = final_piece_counts_vec[i] as f32;
                    // Exponential moving average
                    mean_game_length =
                        SMOOTHING * mean_game_length + (1.0 - SMOOTHING) * final_length;
                    mean_game_length = mean_game_length.max(MIN_THRESHOLD);
                }
            }

            // Compute curriculum-based rewards
            let reward = compute_reward(
                &lost_flags,
                &final_piece_counts_vec,
                mean_game_length,
                BATCH_SIZE,
                &device,
            )?;
            rewards.push(reward);

            // Create done tensor
            let done = create_done_tensor(&lost_flags, BATCH_SIZE, &device)?;
            dones.push(done);

            // Reset lost games and track completions
            let num_completed = current_games.reset_lost_games();
            total_games_completed += num_completed;

            // Check if we've hit our target
            if total_games_completed >= NUM_TOTAL_GAMES {
                break;
            }
        }

        let actual_rollout_steps = actions.len();

        // === Bootstrap from current state ===
        let final_board = TetrisBoardsTensor::from_gameset(&current_games, &device)?;
        let final_pieces_vec = current_games.current_pieces().to_vec();
        let final_piece = TetrisPieceTensor::from_pieces(&final_pieces_vec, &device)?;
        let (_, bootstrap_value) = policy.forward_masked_with_value(&final_board, &final_piece)?;

        // === Compute advantages using GAE ===
        let advantages = compute_gae_advantages(
            &rewards,
            &values_detached,
            &dones,
            bootstrap_value.detach(),
            GAMMA,
            GAE_LAMBDA,
        )?;

        // Compute advantage statistics before normalization
        let advantages_stacked = Tensor::stack(&advantages, 0)?;
        let adv_mean_unnorm = advantages_stacked.mean_all()?.to_scalar::<f32>()?;

        // Normalize advantages for stable gradients
        let advantages = normalize_advantages(&advantages)?;

        // === Compute loss (use stored tensors - no forward pass needed!) ===
        let mut policy_losses = Vec::new();
        let mut value_losses = Vec::new();
        let mut entropy_tensors = Vec::new();

        for t in 0..actual_rollout_steps {
            // Use stored tensors with gradients - no recomputation!
            let logits = &stored_logits[t];
            let value = &stored_values[t];

            // Policy loss with advantage clipping (PPO-style stability)
            let log_probs = candle_nn::ops::log_softmax(logits.inner(), D::Minus1)?;
            let action_one_hot =
                TetrisPieceOrientationDistTensor::from_orientations_tensor(actions[t].clone())?;
            let action_log_prob = (log_probs.clone() * action_one_hot.inner())?.sum(D::Minus1)?;

            // Clip advantages to prevent large policy updates
            let advantages_clipped =
                advantages[t].clamp(-POLICY_CLIP as f64, POLICY_CLIP as f64)?;
            let policy_loss = (&action_log_prob * &advantages_clipped)?.mean_all()?;
            policy_losses.push(policy_loss);

            // Value loss - use CLIPPED advantages for stable target
            let value_target = (&advantages_clipped + &values_detached[t])?;
            let value_error = (value - &value_target)?;
            let value_loss = (&value_error * &value_error)?.mean_all()?;
            value_losses.push(value_loss);

            // Entropy bonus
            let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;
            let entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?.mean_all()?;
            entropy_tensors.push(entropy);
        }

        let policy_obj = Tensor::stack(&policy_losses, 0)?.mean_all()?;
        let value_loss = Tensor::stack(&value_losses, 0)?.mean_all()?;
        let entropy = Tensor::stack(&entropy_tensors, 0)?.mean_all()?;

        // Entropy floor: prevent collapse by penalizing low entropy
        let entropy_scalar = entropy.to_scalar::<f32>()?;
        let entropy_penalty = if entropy_scalar < MIN_ENTROPY {
            let deficit = MIN_ENTROPY - entropy_scalar;
            Tensor::new((deficit * deficit * 10.0) as f64, &device)?
        } else {
            Tensor::new(0.0f64, &device)?
        };

        // Total loss: -policy_objective + value_loss - entropy_bonus + entropy_penalty
        let total_loss = (&policy_obj.neg()? + &value_loss.affine(VF_COEF as f64, 0.0)?
            - &entropy.affine(entropy_coef as f64, 0.0)?
            + &entropy_penalty)?;

        // === Update ===
        let grads = total_loss.backward()?;
        let last_grad_norm = get_l2_norm(&grads)?;
        grad_accum.accumulate(grads, &model_params)?;
        let _stepped = grad_accum.apply_and_reset(
            &mut optimizer,
            &model_params,
            Some(CLIP_GRAD_MAX_NORM),
            Some(CLIP_GRAD_MAX_VALUE),
        )?;

        update_count += 1;

        // === Compute reward distribution for histogram ===
        let rewards_stacked = Tensor::stack(&rewards, 0)?; // [T, B]
        let rewards_flat = rewards_stacked.flatten_all()?;
        let rewards_histogram: Vec<f64> = rewards_flat
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        // === Compute normalized advantages distribution for histogram ===
        let advantages_norm_stacked = Tensor::stack(&advantages, 0)?; // [T, B]
        let advantages_norm_flat = advantages_norm_stacked.flatten_all()?;
        let advantages_norm_histogram: Vec<f64> = advantages_norm_flat
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        // Extract scalars for logging
        let policy_loss_scalar = policy_obj.neg()?.to_scalar::<f32>()?;
        let value_loss_scalar = value_loss.to_scalar::<f32>()?;
        let entropy_scalar = entropy.to_scalar::<f32>()?;
        let total_loss_scalar = total_loss.to_scalar::<f32>()?;

        // Print update progress every iteration
        if update_count % 10 == 0 || update_count == 1 {
            println!(
                "[Update {}] Games completed: {}/{} ({:.1}%) | Loss: {:.4}",
                update_count,
                total_games_completed,
                NUM_TOTAL_GAMES,
                (total_games_completed as f32 / NUM_TOTAL_GAMES as f32) * 100.0,
                total_loss_scalar
            );
        }

        // === Logging ===
        summary_writer.as_mut().map(|s| {
            s.add_scalar(
                "progress/total_games_completed",
                total_games_completed as f32,
                update_count,
            );
            s.add_scalar("loss/total", total_loss_scalar, update_count);
            s.add_scalar("loss/policy", policy_loss_scalar, update_count);
            s.add_scalar("loss/value", value_loss_scalar, update_count);
            s.add_scalar("loss/entropy", entropy_scalar, update_count);
            s.add_scalar("training/grad_norm", last_grad_norm, update_count);
            s.add_scalar("training/temperature", temperature, update_count);
            s.add_scalar("training/entropy_coef", entropy_coef, update_count);
            s.add_scalar(
                "curriculum/update_interval",
                update_interval as f32,
                update_count,
            );
            s.add_scalar("training/advantage_mean", adv_mean_unnorm, update_count);

            // Curriculum learning
            s.add_scalar(
                "curriculum/mean_game_length",
                mean_game_length,
                update_count,
            );

            // Reward distribution
            s.add_histogram(
                "distribution/rewards",
                &rewards_histogram,
                None,
                update_count,
            );

            // Normalized advantages distribution (should be ~N(0,1))
            s.add_histogram(
                "distribution/advantages_normalized",
                &advantages_norm_histogram,
                None,
                update_count,
            );

            // Log current piece counts distribution (shows age diversity)
            let piece_counts: Vec<f64> = current_games
                .piece_counts()
                .iter()
                .map(|&c| c as f64)
                .collect();
            s.add_histogram("distribution/game_ages", &piece_counts, None, update_count);
        });

        // === Checkpointing ===
        if let Some(ref checkpointer) = checkpointer {
            if total_games_completed % CHECKPOINT_INTERVAL_GAMES < BATCH_SIZE {
                let _ =
                    checkpointer.checkpoint_item(total_games_completed, &model_varmap, None, None);
            }
        }

        // === Progress logging ===
        if update_count % 100 == 0 {
            let piece_counts = current_games.piece_counts();
            let mean_age = piece_counts.iter().sum::<u32>() as f32 / BATCH_SIZE as f32;
            let min_age = piece_counts.iter().min().unwrap_or(&0);
            let max_age = piece_counts.iter().max().unwrap_or(&0);

            println!(
                "\n[Update {}] ========================================",
                update_count
            );
            println!(
                "  Progress: {}/{} games ({:.1}%)",
                total_games_completed,
                NUM_TOTAL_GAMES,
                (total_games_completed as f32 / NUM_TOTAL_GAMES as f32) * 100.0
            );
            println!(
                "  Loss: total={:.4} | policy={:.4} | value={:.4} | entropy={:.4}",
                total_loss_scalar, policy_loss_scalar, value_loss_scalar, entropy_scalar
            );
            println!(
                "  Game ages: mean={:.1} | min={} | max={}",
                mean_age, min_age, max_age
            );
            println!(
                "  Curriculum: target_length={:.1} pieces | update_interval={}",
                mean_game_length, update_interval
            );
            println!(
                "  Training: lr={:.1e} | temp={:.3} | entropy_coef={:.4} | grad_norm={:.4}",
                optimizer.learning_rate(),
                temperature,
                entropy_coef,
                last_grad_norm
            );
            println!("========================================\n");

            info!(
                "[{}] Games: {}/{} | Loss: {:.4} (pol={:.4}, val={:.4}, ent={:.4}) | Ages: μ={:.1} [{}-{}] | lr={:.1e} | temp={:.3} | ent_coef={:.4} | grad={:.4}",
                update_count,
                total_games_completed,
                NUM_TOTAL_GAMES,
                total_loss_scalar,
                policy_loss_scalar,
                value_loss_scalar,
                entropy_scalar,
                mean_age,
                min_age,
                max_age,
                optimizer.learning_rate(),
                temperature,
                entropy_coef,
                last_grad_norm,
            );
        }
    }

    // Final checkpoint
    if let Some(ref checkpointer) = checkpointer {
        let _ =
            checkpointer.force_checkpoint_item(total_games_completed, &model_varmap, None, None);
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║           TRAINING COMPLETE!                           ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("  Total games completed: {}", total_games_completed);
    println!("  Total updates: {}", update_count);
    println!("  Final learning rate: {:.1e}", optimizer.learning_rate());

    // Final game age statistics
    let final_piece_counts = current_games.piece_counts();
    let final_mean_age = final_piece_counts.iter().sum::<u32>() as f32 / BATCH_SIZE as f32;
    let final_min_age = final_piece_counts.iter().min().unwrap_or(&0);
    let final_max_age = final_piece_counts.iter().max().unwrap_or(&0);
    println!(
        "  Final game ages: mean={:.1} | min={} | max={}",
        final_mean_age, final_min_age, final_max_age
    );
    println!("════════════════════════════════════════════════════════\n");

    info!("Training complete! Total games: {}", total_games_completed);

    Ok(())
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[arg(long, help = "Path to save the tensorboard logs")]
    logdir: Option<String>,

    #[arg(long, help = "Path to load/save model checkpoints")]
    checkpoint_dir: Option<String>,

    #[arg(long, help = "Name of the training run")]
    run_name: String,
}

fn main() {
    info!("Starting tetris simple goal policy training");
    set_global_threadpool();

    let cli = Cli::parse();

    let verbosity = cli.verbose.saturating_add(2).clamp(0, 5);
    let level = Level::from_str(verbosity.to_string().as_str()).unwrap();

    let registry = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::filter::LevelFilter::from_level(level));

    registry.init();
    info!("Logging initialized at level: {}", level);

    let ulid = ulid::Ulid::new().to_string();
    let run_name = format!("{}_{}", cli.run_name, ulid);
    let logdir = cli.logdir.as_ref().map(|s| {
        let path = std::path::Path::new(s).join(&run_name);
        std::fs::create_dir_all(&path).expect("Failed to create log directory");
        path
    });
    let checkpoint_dir = cli.checkpoint_dir.as_ref().map(|s| {
        let path = std::path::Path::new(s).join(&run_name);
        std::fs::create_dir_all(&path).expect("Failed to create checkpoint directory");
        path
    });

    train_goal_policy(run_name, logdir, checkpoint_dir).unwrap();
}
