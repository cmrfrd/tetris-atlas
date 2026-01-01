use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
use itertools::MultiUnzip;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::info;

use crate::checkpointer::Checkpointer;
use crate::grad_accum::{GradientAccumulator, get_l2_norm};
use crate::modules::{
    CausalSelfAttentionConfig, DynamicTanhConfig, Mlp, MlpConfig, TransformerBlockConfig,
    TransformerBody, TransformerBodyConfig,
};
use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOrientationLogitsTensor, TetrisPieceOrientationTensor,
    TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement};
use crate::wrapped_tensor::WrappedTensor;
use crate::{device, fdtype};

/// Simple policy over placements using supervised learning on high-performing trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGoalPolicyConfig {
    /// Model dimension used throughout the transformer.
    pub d_model: usize,
    /// Number of transformer blocks.
    pub num_blocks: usize,
    /// Self-attention configuration.
    pub attn_config: CausalSelfAttentionConfig,
    /// MLP used inside each transformer block.
    pub block_mlp_config: MlpConfig,
    /// Head MLP (pooled embedding -> logits over orientations).
    pub head_mlp_config: MlpConfig,
    /// Value head MLP (pooled embedding -> scalar value).
    pub value_mlp_config: MlpConfig,
    /// Whether to apply a causal mask in attention (generally false for board evaluation).
    #[serde(default)]
    pub with_causal_mask: bool,
}

#[derive(Debug, Clone)]
pub struct TetrisGoalPolicy {
    // Embeddings
    token_embed: Embedding, // [0/1] -> D
    pos_embed: Embedding,   // [0..199] -> D
    piece_embed: Embedding, // piece id -> D (broadcast-added as conditioning)

    // Transformer body over the 200 board tokens.
    body: TransformerBody,

    // Head: pooled embedding -> logits over orientations.
    head_mlp: Mlp,
    // Value head: pooled embedding -> scalar value estimate.
    value_mlp: Mlp,

    // Config
    #[allow(dead_code)]
    d_model: usize,
    with_causal_mask: bool,
}

impl TetrisGoalPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisGoalPolicyConfig) -> Result<Self> {
        // Token embedding: 2 states (empty/filled) -> d_model
        let token_embed = embedding(
            TetrisBoard::NUM_TETRIS_CELL_STATES,
            cfg.d_model,
            vb.pp("token_embed"),
        )?;

        // Positional embedding: 200 positions -> d_model
        let pos_embed = embedding(TetrisBoard::SIZE, cfg.d_model, vb.pp("pos_embed"))?;

        // Piece embedding: piece id -> d_model (conditioning)
        let piece_embed = embedding(TetrisPiece::NUM_PIECES, cfg.d_model, vb.pp("piece_embed"))?;

        // Transformer body
        let body_cfg = TransformerBodyConfig {
            blocks_config: TransformerBlockConfig {
                dyn_tanh_config: DynamicTanhConfig {
                    alpha_init_value: 1.0,
                    normalized_shape: cfg.d_model,
                },
                attn_config: cfg.attn_config.clone(),
                mlp_config: cfg.block_mlp_config.clone(),
            },
            num_blocks: cfg.num_blocks,
            dyn_tanh_config: DynamicTanhConfig {
                alpha_init_value: 1.0,
                normalized_shape: cfg.d_model,
            },
        };
        let body = TransformerBody::init(&vb.pp("transformer_body"), &body_cfg)?;

        // Head MLP
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;

        // Value head MLP
        let value_mlp = Mlp::init(&vb.pp("value_mlp"), &cfg.value_mlp_config)?;

        Ok(Self {
            token_embed,
            pos_embed,
            piece_embed,
            body,
            head_mlp,
            value_mlp,
            d_model: cfg.d_model,
            with_causal_mask: cfg.with_causal_mask,
        })
    }

    fn forward_pooled(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let (b, _t) = current_board.shape_tuple(); // [B, 200]

        // Board tokens: [B, 200] (u8) -> [B, 200] (u32 indices)
        let tokens = current_board.inner().to_dtype(DType::U32)?;

        // Token + position embeddings: [B, 200, D]
        let x_tokens = self.token_embed.forward(&tokens)?;

        let device = tokens.device();
        let pos_ids = Tensor::arange(0, TetrisBoard::SIZE as u32, device)?
            .to_dtype(DType::U32)?
            .reshape(&[1, TetrisBoard::SIZE])?
            .repeat(&[b, 1])?; // [B, 200]
        let x_pos = self.pos_embed.forward(&pos_ids)?;

        // Piece conditioning: [B, D] -> [B, 200, D] (broadcast)
        let piece_embed = self.piece_embed.forward(current_piece)?.squeeze(1)?; // [B, D]
        let piece_b = piece_embed.unsqueeze(1)?.broadcast_as(x_tokens.dims())?;

        // Combine embeddings
        let x = ((&x_tokens + &x_pos)? + &piece_b)?; // [B, 200, D]

        // Transformer body
        let x = self.body.forward(x, self.with_causal_mask)?; // [B, 200, D]

        // Mean pool over tokens -> [B, D]
        Ok(x.mean(1)?)
    }

    pub fn forward_logits_and_value(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<(TetrisPieceOrientationLogitsTensor, Tensor)> {
        let pooled = self.forward_pooled(current_board, current_piece)?; // [B, D]

        // Policy head -> orientation logits [B, O]
        let logits = self.head_mlp.forward(&pooled)?; // [B, O]
        let logits = TetrisPieceOrientationLogitsTensor::try_from(logits)?;

        // Value head -> scalar [B]
        let value = self.value_mlp.forward(&pooled)?; // [B, 1] (expected)
        let value = value.squeeze(1)?; // [B]

        Ok((logits, value))
    }

    /// Forward producing unmasked orientation logits [B, NUM_ORIENTATIONS]
    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let (logits, _value) = self.forward_logits_and_value(current_board, current_piece)?;
        Ok(logits)
    }

    /// Forward producing a scalar value estimate [B] for (board, piece).
    pub fn forward_value(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let (_logits, value) = self.forward_logits_and_value(current_board, current_piece)?;
        Ok(value)
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

    pub fn forward_masked_with_value(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<(TetrisPieceOrientationLogitsTensor, Tensor)> {
        let (logits, value) = self.forward_logits_and_value(current_board, current_piece)?; // [B,O], [B]
        let mask = create_orientation_mask(current_piece)?; // [B,O] (0/1)
        // keep valid, set invalid to -inf
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        Ok((TetrisPieceOrientationLogitsTensor::try_from(masked)?, value))
    }
}

/// A single trajectory recording all states and actions for one game
#[derive(Debug, Clone)]
struct Trajectory {
    boards: Vec<TetrisBoard>,
    pieces: Vec<TetrisPiece>,
    orientations: Vec<TetrisPieceOrientation>,
    game_length: usize,
    lines_cleared: u32,
}

/// Train a policy-gradient style policy (historically "exceed the mean" trajectories).
pub fn train_tetris_policy_gradients(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    // Hyperparameters - SIMPLIFIED FOR STABILITY
    const NUM_ITERATIONS: usize = 100_000;
    const NUM_TRAJECTORIES_PER_BATCH: usize = 512;
    const PARALLEL_GAMES: usize = 512;
    const KEEP_TOP_PERCENT: f32 = 0.25; // Keep best 25% of trajectories by length
    const MINI_BATCH_SIZE: usize = 512;
    const CHECKPOINT_INTERVAL: usize = 100;
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0);
    const CLIP_GRAD_MAX_VALUE: Option<f64> = None;
    const LEARNING_RATE: f64 = 0.0001; // Conservative fixed rate
    const TEMPERATURE: f32 = 1.0; // Fixed temperature - no decay
    const ENTROPY_WEIGHT: f32 = 0.01; // Entropy bonus for exploration

    // Reward computation parameters
    const LENGTH_EXPONENT: f32 = 1.5; // Exponential scaling for trajectory length
    const LINES_CLEARED_WEIGHT: f32 = 1.0; // Weight for lines cleared score
    const REWARD_TIME_DECAY: f32 = 0.999; // Exponential timestep decay: r_t *= decay^t
    const REWARD_PER_PIECE: f32 = 0.5; // Constant reward per piece played (per timestep)

    // Training loop parameters
    const NUM_EPOCHS_PER_BATCH: usize = 1; // Number of epochs per batch
    const SAMPLE_PERCENT_TRANSITIONS: f32 = 0.8; // Percent of transitions to sample per epoch
    const MAX_NUM_MINI_BATCHES: usize = 100; // Cap on number of mini-batches
    const MAX_WEIGHTED_LOSS: f32 = 10.0; // Clamp weighted loss to prevent extreme values
    const VALUE_LOSS_WEIGHT: f64 = 0.5;

    let model_dim = 64;
    let num_blocks = 4;

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    let policy_cfg = TetrisGoalPolicyConfig {
        d_model: model_dim,
        num_blocks,
        attn_config: CausalSelfAttentionConfig {
            d_model: model_dim,
            n_attention_heads: 8,
            n_kv_heads: 8,
            rope_theta: 10_000.0,
            max_position_embeddings: TetrisBoard::SIZE,
        },
        block_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: model_dim,
            dropout: None,
        },
        head_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            dropout: None,
        },
        value_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: 1,
            dropout: None,
        },
        with_causal_mask: false,
    };

    let policy = TetrisGoalPolicy::init(&vb, &policy_cfg)?;
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params.clone(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..ParamsAdamW::default()
        },
    )?;
    let mut grad_accum = GradientAccumulator::new(1);

    // Summary + checkpointing
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

    let mut total_games_completed = 0;
    let mut mean_game_length: f32 = 10.0; // Initial estimate

    println!("\n=== SIMPLIFIED IMITATION LEARNING ===");
    println!(
        "Starting training: {} iterations, batch size {}",
        NUM_ITERATIONS, NUM_TRAJECTORIES_PER_BATCH
    );
    println!(
        "Strategy: Keep top {:.0}% longest games, train on their actions",
        KEEP_TOP_PERCENT * 100.0
    );
    println!("Temperature: {:.2} (fixed, no decay)", TEMPERATURE);
    println!("Learning rate: {:.4} (fixed)", LEARNING_RATE);
    println!("Entropy bonus: {:.3}", ENTROPY_WEIGHT);
    println!("Model capacity: {} dims", model_dim);
    println!("Mini-batch size: {}", MINI_BATCH_SIZE);

    // Main training loop - iteration-based
    for iteration in 0..NUM_ITERATIONS {
        // === Phase 1: Collect trajectories (curriculum-adaptive batch size) ===
        println!(
            "\n[Iteration {}] Collecting {} game trajectories ({} parallel games)...",
            iteration, NUM_TRAJECTORIES_PER_BATCH, PARALLEL_GAMES
        );

        // Initialize parallel games
        let mut active_games = crate::tetris::TetrisGameSet::new(PARALLEL_GAMES);

        // Track trajectory for each of the PARALLEL_GAMES game slots
        // Preallocate with expected average game length to reduce reallocations
        const EXPECTED_AVG_GAME_LENGTH: usize = 100;
        let mut trajectories_in_progress: Vec<(
            Vec<TetrisBoard>,
            Vec<TetrisPiece>,
            Vec<TetrisPieceOrientation>,
        )> = (0..PARALLEL_GAMES)
            .map(|_| {
                (
                    Vec::with_capacity(EXPECTED_AVG_GAME_LENGTH),
                    Vec::with_capacity(EXPECTED_AVG_GAME_LENGTH),
                    Vec::with_capacity(EXPECTED_AVG_GAME_LENGTH),
                )
            })
            .collect();
        let mut trajectories: Vec<Trajectory> = Vec::with_capacity(NUM_TRAJECTORIES_PER_BATCH);

        while trajectories.len() < NUM_TRAJECTORIES_PER_BATCH {
            // Batched forward pass for all PARALLEL_GAMES at once
            let game_boards: Vec<TetrisBoard> = active_games.boards().to_vec();
            let current_board = TetrisBoardsTensor::from_boards(&game_boards, &device)?;
            let current_pieces_vec: Vec<TetrisPiece> = active_games.current_pieces().to_vec();
            let current_piece = TetrisPieceTensor::from_pieces(&current_pieces_vec, &device)?;

            // Sample actions for all parallel games with mask
            let logits = policy.forward_logits(&current_board, &current_piece)?;
            let sampled_orientations = logits.sample(TEMPERATURE, &current_piece)?;
            let orientations = sampled_orientations.into_orientations()?;

            // Add to all trajectories in parallel
            trajectories_in_progress
                .par_iter_mut()
                .zip(game_boards.par_iter())
                .zip(current_pieces_vec.par_iter())
                .zip(orientations.par_iter())
                .for_each(|(((trajectory, game_board), piece), orientation)| {
                    trajectory.0.push(game_board.clone());
                    trajectory.1.push(piece.clone());
                    trajectory.2.push(orientation.clone());
                });

            #[derive(Copy, Clone)]
            struct SendPtr<T>(*mut T);
            unsafe impl<T> Send for SendPtr<T> {}
            unsafe impl<T> Sync for SendPtr<T> {}
            impl<T> SendPtr<T> {
                fn as_ptr(self) -> *mut T {
                    self.0
                }
            }

            let games_ptr = SendPtr(&mut active_games as *mut crate::tetris::TetrisGameSet);

            let lost: Vec<bool> = (0..PARALLEL_GAMES)
                .into_par_iter()
                .zip(&orientations)
                .zip(&current_pieces_vec)
                .map(move |((i, &orientation), &piece)| {
                    let result = unsafe {
                        let games = &mut *games_ptr.as_ptr();
                        games[i].apply_placement(TetrisPiecePlacement { piece, orientation })
                    };
                    bool::from(result.is_lost)
                })
                .collect();

            let trajectory_results = lost
                .iter()
                .enumerate()
                .filter_map(|(i, &is_lost)| {
                    let result = bool::from(is_lost).then(|| {
                        // Take ownership without cloning - much faster!
                        Trajectory {
                            boards: std::mem::take(&mut trajectories_in_progress[i].0),
                            pieces: std::mem::take(&mut trajectories_in_progress[i].1),
                            orientations: std::mem::take(&mut trajectories_in_progress[i].2),
                            game_length: active_games[i].piece_count as usize,
                            lines_cleared: active_games[i].lines_cleared,
                        }
                    });

                    if result.is_some() {
                        // Vectors are already empty from mem::take, just reset the game
                        active_games[i].reset(None);
                    }

                    result
                })
                .collect::<Vec<_>>();

            trajectories.extend(trajectory_results);
        }

        total_games_completed += NUM_TRAJECTORIES_PER_BATCH;
        println!("  ✓ Collection complete!");

        // === Phase 2: Compute per-timestep rewards with exponential length scaling and linear decay ===
        println!("  Computing rewards with exponential trajectory-length weighting...");

        // Find max trajectory length for monitoring
        let max_trajectory_length = trajectories
            .iter()
            .map(|t| t.game_length)
            .max()
            .unwrap_or(1) as f32;

        // Step 1: Compute raw rewards with exponential length scaling and linear decay
        let mut all_raw_rewards: Vec<f32> = Vec::new();
        let trajectories_with_raw_rewards: Vec<(Trajectory, Vec<f32>)> = trajectories
            .into_iter()
            .map(|traj| {
                let traj_length = traj.game_length as f32;
                let lines_cleared_score = traj.lines_cleared as f32;

                let length_score = traj_length.powf(LENGTH_EXPONENT);
                let weighted_lines = lines_cleared_score * LINES_CLEARED_WEIGHT;

                let rewards: Vec<f32> = (0..traj.game_length)
                    .map(|t| {
                        // Exponential decay: earlier timesteps matter more
                        // At t=0: decay=1.0, then decay^t.
                        let decay_factor = REWARD_TIME_DECAY.powi(t as i32);

                        // Combined reward: balanced contribution from both signals
                        let reward =
                            (REWARD_PER_PIECE + weighted_lines + length_score) * decay_factor;
                        reward
                    })
                    .collect();

                // Collect for global standardization
                all_raw_rewards.extend(&rewards);

                (traj, rewards)
            })
            .collect();

        // Track statistics for monitoring (raw reward scale)
        let mean_reward = all_raw_rewards.iter().sum::<f32>() / all_raw_rewards.len() as f32;
        let min_raw_reward = all_raw_rewards
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let max_raw_reward = all_raw_rewards
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let raw_range = max_raw_reward - min_raw_reward;

        // Advantage stats (for monitoring only). Training uses a learned baseline V(s).
        let all_advantages: Vec<f32> = all_raw_rewards.iter().map(|r| r - mean_reward).collect();
        let advantage_variance =
            all_advantages.iter().map(|a| a.powi(2)).sum::<f32>() / all_advantages.len() as f32;
        let advantage_std = advantage_variance.sqrt().max(1e-8);
        let mean_advantage = all_advantages.iter().sum::<f32>() / all_advantages.len() as f32;
        let min_advantage = all_advantages
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let max_advantage = all_advantages
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);

        // Track game length and lines cleared for monitoring
        mean_game_length = trajectories_with_raw_rewards
            .iter()
            .map(|(t, _)| t.game_length as f32)
            .sum::<f32>()
            / trajectories_with_raw_rewards.len() as f32;

        let mean_lines_cleared = trajectories_with_raw_rewards
            .iter()
            .map(|(t, _)| t.lines_cleared as f32)
            .sum::<f32>()
            / trajectories_with_raw_rewards.len() as f32;

        let total_lines_cleared: u32 = trajectories_with_raw_rewards
            .iter()
            .map(|(t, _)| t.lines_cleared)
            .sum();

        println!(
            "  Game length: mean={:.1}, max={:.0} pieces",
            mean_game_length, max_trajectory_length
        );
        println!(
            "  Lines cleared: mean={:.1}, total={} (efficiency: {:.2} lines/piece)",
            mean_lines_cleared,
            total_lines_cleared,
            mean_lines_cleared / mean_game_length
        );
        println!(
            "  Raw rewards: min={:.1}, max={:.1}, range={:.1}",
            min_raw_reward, max_raw_reward, raw_range
        );
        println!(
            "  Advantages (baseline={:.3}, std={:.3}): mean={:.4}, range=[{:.3}, {:.3}]",
            mean_reward, advantage_std, mean_advantage, min_advantage, max_advantage
        );

        // === Phase 3: Supervised learning with prioritized trajectory sampling ===
        // Step 1: Build per-trajectory transition pools with prioritization weights
        // Store raw rewards as training targets for the value head.
        let trajectory_pools: Vec<Vec<(TetrisBoard, TetrisPiece, TetrisPieceOrientation, f32)>> =
            trajectories_with_raw_rewards
                .iter()
                .map(|(traj, rewards)| {
                    (0..traj.game_length)
                        .map(|i| {
                            (
                                traj.boards[i].clone(),
                                traj.pieces[i],
                                traj.orientations[i],
                                rewards[i],
                            )
                        })
                        .collect()
                })
                .collect();

        // Prioritization weights: longer trajectories sampled more often
        // Using sqrt for moderate prioritization (linear would be too aggressive)
        let trajectory_weights: Vec<f64> = trajectories_with_raw_rewards
            .iter()
            .map(|(traj, _)| (traj.game_length as f64).sqrt())
            .collect();

        let total_transitions: usize = trajectory_pools.iter().map(|p| p.len()).sum();
        println!(
            "  Available transitions: {} (prioritized by trajectory length)",
            total_transitions
        );

        if total_transitions == 0 {
            println!("  WARNING: No transitions available, skipping update");
            continue;
        }

        // Step 2: Multi-epoch training with fresh prioritized sampling each epoch
        let mut rng = rand::rng();
        let mut epoch_avg_loss = 0.0;
        let mut epoch_avg_policy_loss = 0.0;
        let mut epoch_avg_value_loss = 0.0;
        let mut epoch_avg_accuracy = 0.0;
        let mut epoch_avg_entropy = 0.0;
        let mut epoch_avg_grad_norm = 0.0;

        // Create weighted distribution for trajectory sampling
        use rand::distr::{Distribution, weighted::WeightedIndex};
        let weight_dist = WeightedIndex::new(&trajectory_weights).unwrap();

        for epoch in 0..NUM_EPOCHS_PER_BATCH {
            // Sample a fresh subset of transitions using prioritized sampling
            let num_to_sample =
                (total_transitions as f32 * SAMPLE_PERCENT_TRANSITIONS).ceil() as usize;
            let num_to_sample = num_to_sample.min(total_transitions);

            let mut sampled_transitions: Vec<(
                TetrisBoard,
                TetrisPiece,
                TetrisPieceOrientation,
                f32,
            )> = (0..num_to_sample)
                .map(|_| {
                    // Sample a trajectory according to weights (longer = higher probability)
                    let traj_idx = weight_dist.sample(&mut rng);
                    let traj_pool = &trajectory_pools[traj_idx];
                    // Uniformly sample a transition from that trajectory
                    let time_idx = rng.random_range(0..traj_pool.len());
                    traj_pool[time_idx].clone()
                })
                .collect();
            sampled_transitions.shuffle(&mut rng);

            let calculated_batches =
                (num_to_sample as f32 / MINI_BATCH_SIZE as f32).ceil() as usize;
            let num_mini_batches = calculated_batches.min(MAX_NUM_MINI_BATCHES);

            if epoch == 0 {
                println!(
                    "  Training {} epochs × {} mini-batches (size {}, capped at {})",
                    NUM_EPOCHS_PER_BATCH, num_mini_batches, MINI_BATCH_SIZE, MAX_NUM_MINI_BATCHES
                );
                println!(
                    "  Sampling {} transitions per epoch ({:.1}%)",
                    num_to_sample,
                    SAMPLE_PERCENT_TRANSITIONS * 100.0
                );
            }

            let mut total_loss_accum = 0.0;
            let mut total_policy_loss_accum = 0.0;
            let mut total_value_loss_accum = 0.0;
            let mut total_accuracy_accum = 0.0;
            let mut total_entropy_accum = 0.0;
            let mut total_grad_norm_accum = 0.0;
            let mut batches_processed = 0;

            for mini_batch_idx in 0..num_mini_batches {
                let start_idx = mini_batch_idx * MINI_BATCH_SIZE;
                let end_idx = (start_idx + MINI_BATCH_SIZE).min(num_to_sample);
                let mini_batch = &sampled_transitions[start_idx..end_idx];
                let mini_batch_size = mini_batch.len();

                if mini_batch.is_empty() {
                    continue;
                }

                // Prepare mini-batch tensors and extract rewards
                let (batch_boards, batch_pieces, batch_targets, batch_rewards): (
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                ) = mini_batch
                    .iter()
                    .map(|(board, piece, orientation, reward)| {
                        (board.clone(), *piece, *orientation, *reward)
                    })
                    .multiunzip();

                // Convert to tensors
                let boards_u8: Vec<u8> = batch_boards
                    .iter()
                    .flat_map(|b: &TetrisBoard| b.to_binary_slice().to_vec())
                    .collect();
                let board_tensor = Tensor::from_vec(
                    boards_u8,
                    Shape::from_dims(&[mini_batch_size, TetrisBoard::SIZE]),
                    &device,
                )?;
                let board_tensor = TetrisBoardsTensor::try_from(board_tensor)?;

                let piece_tensor = TetrisPieceTensor::from_pieces(&batch_pieces, &device)?;
                let target_tensor =
                    TetrisPieceOrientationTensor::from_orientations(&batch_targets, &device)?;

                // Forward pass
                let (logits, value_pred) =
                    policy.forward_masked_with_value(&board_tensor, &piece_tensor)?; // [B,O], [B]

                // Compute probabilities and log probabilities
                let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;
                let log_probs = candle_nn::ops::log_softmax(logits.inner(), D::Minus1)?;

                let target_one_hot = target_tensor.into_dist()?;
                let sample_log_probs = (&log_probs * target_one_hot.inner())?
                    .sum(D::Minus1)?
                    .clamp(-10.0, 0.0)?; // [B] log π(a|s)

                let rewards_tensor =
                    Tensor::from_vec(batch_rewards, Shape::from_dims(&[mini_batch_size]), &device)?
                        .to_dtype(dtype)?;

                // Raw advantages using learned baseline V(s).
                // IMPORTANT: detach the baseline in the actor loss (no grads through baseline).
                let advantages = rewards_tensor.sub(&value_pred.detach())?;

                // Policy gradient loss: -E[ A(s,a) * log π(a|s) ]
                let weighted_log_probs = (&sample_log_probs * &advantages.neg()?)?;

                // Clamp weighted values to prevent extreme outliers
                let clamped_weighted =
                    weighted_log_probs.clamp(-MAX_WEIGHTED_LOSS, MAX_WEIGHTED_LOSS)?;

                let policy_loss = clamped_weighted.mean_all()?;

                // Compute entropy regularization
                let entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?;
                let avg_entropy = entropy.mean_all()?;
                let entropy_loss = avg_entropy.affine(-ENTROPY_WEIGHT as f64, 0.0)?;

                // Value loss: train value head to predict the (normalized) reward.
                let value_loss = (&value_pred - &rewards_tensor)?.sqr()?.mean_all()?;
                let value_loss_term = value_loss.affine(VALUE_LOSS_WEIGHT, 0.0)?;

                // Total loss
                let loss = ((&policy_loss + &value_loss_term)? + &entropy_loss)?;

                // Compute accuracy
                let predicted = logits.inner().argmax(D::Minus1)?;
                let target_indices: Vec<u32> =
                    batch_targets.iter().map(|o| o.index() as u32).collect();
                let target_indices_tensor = Tensor::from_vec(
                    target_indices,
                    Shape::from_dims(&[mini_batch_size]),
                    &device,
                )?;
                let correct = predicted
                    .eq(&target_indices_tensor)?
                    .to_dtype(DType::F32)?
                    .sum_all()?;
                let accuracy = correct.to_scalar::<f32>()? / mini_batch_size as f32;

                // Accumulate metrics
                let entropy_scalar = avg_entropy.to_scalar::<f32>()?;
                let loss_scalar = loss.to_scalar::<f32>()?;
                let policy_loss_scalar = policy_loss.to_scalar::<f32>()?;
                let value_loss_scalar = value_loss.to_scalar::<f32>()?;

                // Safety check: Skip batch if loss is non-finite
                if !loss_scalar.is_finite() {
                    eprintln!("  ⚠️  Non-finite loss - SKIPPING BATCH");
                    continue;
                }

                total_loss_accum += loss_scalar;
                total_policy_loss_accum += policy_loss_scalar;
                total_value_loss_accum += value_loss_scalar;
                total_accuracy_accum += accuracy;
                total_entropy_accum += entropy_scalar;
                batches_processed += 1;

                // Backward and accumulate gradients
                let grads = loss.backward()?;

                // Compute gradient norm before accumulation
                let grad_norm = get_l2_norm(&grads)?;
                total_grad_norm_accum += grad_norm;

                grad_accum.accumulate(grads, &model_params)?;
            }
            grad_accum.apply_and_reset(
                &mut optimizer,
                &model_params,
                CLIP_GRAD_MAX_NORM,
                CLIP_GRAD_MAX_VALUE,
            )?;

            if batches_processed == 0 {
                println!("  WARNING: No batches processed (all skipped), skipping epoch stats");
                continue;
            }

            // Compute epoch averages
            let avg_loss = total_loss_accum / batches_processed as f32;
            let avg_policy_loss = total_policy_loss_accum / batches_processed as f32;
            let avg_value_loss = total_value_loss_accum / batches_processed as f32;
            let avg_accuracy = total_accuracy_accum / batches_processed as f32;
            let avg_entropy = total_entropy_accum / batches_processed as f32;
            let avg_grad_norm = total_grad_norm_accum / batches_processed as f32;

            epoch_avg_loss += avg_loss;
            epoch_avg_policy_loss += avg_policy_loss;
            epoch_avg_value_loss += avg_value_loss;
            epoch_avg_accuracy += avg_accuracy;
            epoch_avg_entropy += avg_entropy;
            epoch_avg_grad_norm += avg_grad_norm;

            println!(
                "    Epoch {}/{}: Loss={:.4} (pol={:.4}, val={:.4}), Acc={:.2}%, Entropy={:.3}, GradNorm={:.4}",
                epoch + 1,
                NUM_EPOCHS_PER_BATCH,
                avg_loss,
                avg_policy_loss,
                avg_value_loss,
                avg_accuracy * 100.0,
                avg_entropy,
                avg_grad_norm
            );
        }

        // Average across all epochs
        let avg_loss = epoch_avg_loss / NUM_EPOCHS_PER_BATCH as f32;
        let avg_policy_loss = epoch_avg_policy_loss / NUM_EPOCHS_PER_BATCH as f32;
        let avg_value_loss = epoch_avg_value_loss / NUM_EPOCHS_PER_BATCH as f32;
        let avg_accuracy = epoch_avg_accuracy / NUM_EPOCHS_PER_BATCH as f32;
        let avg_entropy = epoch_avg_entropy / NUM_EPOCHS_PER_BATCH as f32;
        let avg_grad_norm = epoch_avg_grad_norm / NUM_EPOCHS_PER_BATCH as f32;

        println!(
            "  Multi-epoch training complete: Avg Loss={:.4} (pol={:.4}, val={:.4}), Avg Accuracy={:.2}%, Avg Entropy={:.3}",
            avg_loss,
            avg_policy_loss,
            avg_value_loss,
            avg_accuracy * 100.0,
            avg_entropy
        );

        // === Logging ===
        summary_writer.as_mut().map(|s| {
            s.add_scalar(
                "progress/total_games_completed",
                total_games_completed as f32,
                iteration,
            );
            s.add_scalar("loss/total", avg_loss, iteration);
            s.add_scalar("loss/policy", avg_policy_loss, iteration);
            s.add_scalar("loss/value", avg_value_loss, iteration);
            s.add_scalar("metrics/accuracy", avg_accuracy, iteration);
            s.add_scalar("metrics/entropy", avg_entropy, iteration);
            s.add_scalar("training/grad_norm", avg_grad_norm, iteration);
            s.add_scalar("training/learning_rate", LEARNING_RATE as f32, iteration);
            s.add_scalar("training/temperature", TEMPERATURE, iteration);
            s.add_scalar("curriculum/mean_game_length", mean_game_length, iteration);
            s.add_scalar(
                "curriculum/mean_lines_cleared",
                mean_lines_cleared,
                iteration,
            );
            s.add_scalar("curriculum/mean_reward", mean_reward, iteration);
            s.add_scalar("curriculum/mean_advantage", mean_advantage, iteration);
            s.add_scalar("curriculum/advantage_std", advantage_std, iteration);
            s.add_scalar("curriculum/min_advantage", min_advantage, iteration);
            s.add_scalar("curriculum/max_advantage", max_advantage, iteration);
            s.add_scalar(
                "curriculum/max_trajectory_length",
                max_trajectory_length,
                iteration,
            );
            s.add_scalar("curriculum/min_raw_reward", min_raw_reward, iteration);
            s.add_scalar("curriculum/max_raw_reward", max_raw_reward, iteration);
            s.add_scalar("curriculum/raw_reward_range", raw_range, iteration);
            s.add_scalar(
                "curriculum/total_transitions",
                total_transitions as f32,
                iteration,
            );

            // Add trajectory length distribution histogram
            let trajectory_lengths: Vec<f64> = trajectories_with_raw_rewards
                .iter()
                .map(|(t, _)| t.game_length as f64)
                .collect();
            s.add_histogram(
                "curriculum/trajectory_length_distribution",
                &trajectory_lengths,
                None,
                iteration,
            );

            // Add lines cleared distribution histogram
            let lines_cleared_dist: Vec<f64> = trajectories_with_raw_rewards
                .iter()
                .map(|(t, _)| t.lines_cleared as f64)
                .collect();
            s.add_histogram(
                "curriculum/lines_cleared_distribution",
                &lines_cleared_dist,
                None,
                iteration,
            );

            // Add advantage distribution histogram (baseline = mean_reward, monitoring only)
            let all_advantages_hist: Vec<f64> = all_advantages.iter().map(|&a| a as f64).collect();
            s.add_histogram(
                "curriculum/advantage_distribution",
                &all_advantages_hist,
                None,
                iteration,
            );
        });

        // === Checkpointing ===
        if let Some(ref checkpointer) = checkpointer {
            if iteration % CHECKPOINT_INTERVAL == 0 {
                let _ = checkpointer.checkpoint_item(iteration, &model_varmap, None, None);
            }
        }

        // === Progress logging ===
        if iteration % 10 == 0 || iteration == NUM_ITERATIONS - 1 {
            println!(
                "\n[Iteration {}] ========================================",
                iteration
            );
            println!(
                "  Progress: {}/{} iterations ({:.1}%)",
                iteration + 1,
                NUM_ITERATIONS,
                ((iteration + 1) as f32 / NUM_ITERATIONS as f32) * 100.0
            );
            println!("  Total games completed: {}", total_games_completed);
            println!(
                "  Loss: {:.4} | Accuracy: {:.2}%",
                avg_loss,
                avg_accuracy * 100.0
            );
            println!(
                "  Curriculum: mean_len={:.1} | max_len={:.0} | lines={:.1} ({:.2}/piece)",
                mean_game_length,
                max_trajectory_length,
                mean_lines_cleared,
                mean_lines_cleared / mean_game_length
            );
            println!(
                "  Rewards: mean={:.3} | Training: lr={:.1e} | temp={:.3}",
                mean_reward,
                optimizer.learning_rate(),
                TEMPERATURE,
            );
            println!("========================================\n");

            info!(
                "[Iter {}] Games: {} | Loss: {:.4} | Acc: {:.2}% | Ent: {:.3} | GradNorm: {:.4} | MeanLen: {:.1} | MaxLen: {:.0} | Lines: {:.1} | LinesPerPiece: {:.2} | MeanReward: {:.3} | lr={:.1e} | temp={:.3}",
                iteration,
                total_games_completed,
                avg_loss,
                avg_accuracy * 100.0,
                avg_entropy,
                avg_grad_norm,
                mean_game_length,
                max_trajectory_length,
                mean_lines_cleared,
                mean_lines_cleared / mean_game_length,
                mean_reward,
                optimizer.learning_rate(),
                TEMPERATURE,
            );
        }
    }

    // Final checkpoint
    if let Some(ref checkpointer) = checkpointer {
        let _ = checkpointer.force_checkpoint_item(NUM_ITERATIONS, &model_varmap, None, None);
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║           TRAINING COMPLETE!                           ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("  Total iterations: {}", NUM_ITERATIONS);
    println!("  Total games completed: {}", total_games_completed);
    println!("  Final learning rate: {:.1e}", optimizer.learning_rate());
    println!("  Final mean game length: {:.1}", mean_game_length);
    println!("════════════════════════════════════════════════════════\n");

    info!("Training complete! Total games: {}", total_games_completed);

    Ok(())
}
