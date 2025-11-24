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
    Conv2dConfig, ConvBlockSpec, ConvEncoder, ConvEncoderConfig, FiLM, FiLMConfig, Mlp, MlpConfig,
};
use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOrientationLogitsTensor, TetrisPieceOrientationTensor,
    TetrisPieceTensor,
};
use crate::tetris::{TetrisBoard, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement};
use crate::wrapped_tensor::WrappedTensor;
use crate::{device, dtype};

/// Simple policy over placements using supervised learning on high-performing trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisGoalPolicyConfig {
    pub piece_embedding_dim: usize,

    pub board_encoder_config: ConvEncoderConfig,

    pub piece_film_config: (FiLMConfig, FiLMConfig),

    pub head_mlp_config: MlpConfig, // outputs logits over orientations
}

#[derive(Debug, Clone)]
pub struct TetrisGoalPolicy {
    piece_embedding: Embedding,
    board_encoder: ConvEncoder,
    piece_film: (FiLM, FiLM),
    head_mlp: Mlp,
}

impl TetrisGoalPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisGoalPolicyConfig) -> Result<Self> {
        let piece_embedding = embedding(
            TetrisPiece::NUM_PIECES,
            cfg.piece_embedding_dim,
            vb.pp("piece_embedding"),
        )?;
        let board_encoder = ConvEncoder::init(&vb.pp("board_encoder"), &cfg.board_encoder_config)?;
        let piece_film = (
            FiLM::init(&vb.pp("piece_film"), &cfg.piece_film_config.0)?,
            FiLM::init(&vb.pp("piece_film"), &cfg.piece_film_config.1)?,
        );
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;
        Ok(Self {
            piece_embedding,
            board_encoder,
            piece_film,
            head_mlp,
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
        let x = self.piece_film.0.forward(&cur_embed, &piece_embed)?; // [B, D]
        let x = self.piece_film.1.forward(&x, &piece_embed)?; // [B, D]

        // Head -> orientation logits [B, NUM_ORIENTATIONS]
        let logits = self.head_mlp.forward(&x)?; // [B, O]
        Ok(TetrisPieceOrientationLogitsTensor::try_from(logits)?)
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

/// Train a policy using supervised learning on trajectories that exceed the mean game length
pub fn train_exceed_the_mean_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = device();
    let dtype = dtype();

    // Hyperparameters
    const NUM_ITERATIONS: usize = 500_000;
    const NUM_TRAJECTORIES_PER_BATCH: usize = 512;
    const PARALLEL_GAMES: usize = 1024;
    const SAMPLE_PERCENT_TRANSITIONS: f32 = 0.4; // Use 50% of data - don't waste it!
    const NUM_EPOCHS_PER_BATCH: usize = 1; // Multi-epoch training - reuse data better
    const MINI_BATCH_SIZE: usize = 256; // Smaller batches = more frequent updates
    const MAX_NUM_MINI_BATCHES: usize = 100; // Allow more mini-batches
    const ACCUM_STEPS: usize = 1; // Gradient accumulation for stability
    const CHECKPOINT_INTERVAL: usize = 100;
    const CLIP_GRAD_MAX_VALUE: Option<f64> = Some(10.0);
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0); // Increased from 1.0 to allow larger gradients
    const INITIAL_TEMPERATURE: f32 = 2.5; // Even higher for more initial exploration
    const FINAL_TEMPERATURE: f32 = 1.2; // Don't go too low - maintain exploration
    const TEMPERATURE_HALFLIFE_ITERS: f32 = 50_000.0; // Much slower decay for sustained exploration
    const MAX_WEIGHTED_LOSS: f64 = 10.0; // Allow larger loss values
    const ENTROPY_WEIGHT: f32 = 0.005; // Reduce entropy penalty - allow more confidence
    const REWARD_SCALE: f32 = 2.0; // Much stronger gradients for faster learning
    const LENGTH_EXPONENT: f32 = 1.5; // Exponential scaling power: length^N
    const LEARNING_RATE: f64 = 0.003; // 3× faster than default 0.001
    let model_dim = 32; // Increased from 32 for more model capacity

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

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
                intermediate_size: 2 * model_dim,
                output_size: model_dim,
                dropout: Some(0.05),
            },
        },
        piece_film_config: (
            FiLMConfig {
                cond_dim: model_dim,
                feat_dim: model_dim,
                hidden: model_dim,
                output_dim: model_dim,
            },
            FiLMConfig {
                cond_dim: model_dim,
                feat_dim: model_dim,
                hidden: model_dim,
                output_dim: model_dim,
            },
        ),
        head_mlp_config: MlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPieceOrientation::NUM_ORIENTATIONS,
            dropout: Some(0.05),
        },
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
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    let mut total_games_completed = 0;
    let mut mean_game_length: f32 = 10.0; // Initial estimate

    println!(
        "Starting training: {} iterations, batch size {} (optimized curriculum)",
        NUM_ITERATIONS, NUM_TRAJECTORIES_PER_BATCH
    );
    println!(
        "Temperature decay: {:.2} → {:.2} (exponential, halflife={:.0} iterations)",
        INITIAL_TEMPERATURE, FINAL_TEMPERATURE, TEMPERATURE_HALFLIFE_ITERS
    );
    println!("Supervised learning: Trajectory-length-weighted rewards with exponential scaling");
    println!(
        "  - Reward = (length^{:.1}) × ((T-t)/T) → z-score normalized → [0, {:.1}]",
        LENGTH_EXPONENT, REWARD_SCALE
    );
    println!(
        "  - Exponential scaling (^{:.1}): longer trajectories get much stronger signal",
        LENGTH_EXPONENT
    );
    println!("  - Z-score normalization: preserves relative improvement magnitude");
    println!(
        "  - Reward scale {:.1}: controls gradient magnitude",
        REWARD_SCALE
    );
    println!("  - Earlier actions within trajectory weighted higher");
    println!(
        "  - {} epochs × {:.0}% transitions = maximum data utilization",
        NUM_EPOCHS_PER_BATCH,
        SAMPLE_PERCENT_TRANSITIONS * 100.0
    );
    println!("Optimizations:");
    println!(
        "  - Learning rate: {:.4} (3× faster than default)",
        LEARNING_RATE
    );
    println!("  - Model capacity: {} dims (2× baseline)", model_dim);
    println!("  - Dropout: 0.05 (reduced for faster learning)");
    println!("  - Prioritized sampling: sqrt(length) trajectory weights");
    println!(
        "  - Gradient clipping: max_norm={:.1}, max_value={:.1}",
        CLIP_GRAD_MAX_NORM.unwrap_or(0.0),
        CLIP_GRAD_MAX_VALUE.unwrap_or(0.0)
    );
    println!("Gradient accumulation: {} steps", ACCUM_STEPS);

    // Main training loop - iteration-based
    for iteration in 0..NUM_ITERATIONS {
        // === Calculate exponentially decaying temperature ===
        let decay_factor = (-0.693147 * iteration as f32 / TEMPERATURE_HALFLIFE_ITERS).exp();
        let temperature = (FINAL_TEMPERATURE
            + (INITIAL_TEMPERATURE - FINAL_TEMPERATURE) * decay_factor)
            .max(FINAL_TEMPERATURE);

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

            // Sample actions for all parallel games
            let masked_logits = policy.forward_masked(&current_board, &current_piece)?;
            let sampled_orientations = masked_logits.sample(temperature)?;
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
                    let is_lost = unsafe {
                        let games = &mut *games_ptr.as_ptr();
                        games[i].apply_placement(TetrisPiecePlacement { piece, orientation })
                    };
                    bool::from(is_lost)
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

                // Exponential length bonus: reward grows as length^LENGTH_EXPONENT
                // This creates much stronger signal for longer trajectories
                // With LENGTH_EXPONENT=2.0: length=10 -> 100, length=100 -> 10000, length=200 -> 40000
                let length_score = traj_length.powf(LENGTH_EXPONENT);

                let rewards: Vec<f32> = (0..traj.game_length)
                    .map(|t| {
                        // Linear decay: earlier timesteps matter more
                        // At t=0: decay=1.0, at t=T-1: decay=1/T
                        let decay_factor = (traj_length - t as f32) / traj_length;

                        // Combined reward: exponential length bonus × linear decay
                        let reward = length_score * decay_factor;
                        reward
                    })
                    .collect();

                // Collect for global standardization
                all_raw_rewards.extend(&rewards);

                (traj, rewards)
            })
            .collect();

        // Step 2: Standardize rewards using z-score (preserves relative differences)
        let mean_raw_reward = all_raw_rewards.iter().sum::<f32>() / all_raw_rewards.len() as f32;
        let variance = all_raw_rewards
            .iter()
            .map(|r| (r - mean_raw_reward).powi(2))
            .sum::<f32>()
            / all_raw_rewards.len() as f32;
        let std_reward = variance.sqrt().max(1e-8);

        let trajectories_with_normalized_rewards: Vec<(Trajectory, Vec<f32>)> =
            trajectories_with_raw_rewards
                .into_iter()
                .map(|(traj, raw_rewards)| {
                    let normalized_rewards: Vec<f32> = raw_rewards
                        .into_iter()
                        .map(|r| {
                            // Z-score normalization: (r - μ) / σ
                            let z_score = (r - mean_raw_reward) / std_reward;
                            // Clip to [-3, 3] standard deviations (keeps ~99.7% of data)
                            // Then shift and scale to [0, REWARD_SCALE]
                            let clipped = z_score.clamp(-3.0, 3.0);
                            let normalized = (clipped + 3.0) / 6.0 * REWARD_SCALE; // Maps [-3, 3] -> [0, REWARD_SCALE]
                            // Ensure final reward is strictly in [0, REWARD_SCALE]
                            normalized.clamp(0.0, REWARD_SCALE)
                        })
                        .collect();
                    (traj, normalized_rewards)
                })
                .collect();

        // Track statistics for monitoring
        let all_normalized_rewards: Vec<f32> = trajectories_with_normalized_rewards
            .iter()
            .flat_map(|(_, rewards)| rewards.iter().copied())
            .collect();

        let mean_reward =
            all_normalized_rewards.iter().sum::<f32>() / all_normalized_rewards.len() as f32;
        let min_normalized = all_normalized_rewards
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let max_normalized = all_normalized_rewards
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(1.0);

        // Track game length and lines cleared for monitoring
        mean_game_length = trajectories_with_normalized_rewards
            .iter()
            .map(|(t, _)| t.game_length as f32)
            .sum::<f32>()
            / trajectories_with_normalized_rewards.len() as f32;

        let mean_lines_cleared = trajectories_with_normalized_rewards
            .iter()
            .map(|(t, _)| t.lines_cleared as f32)
            .sum::<f32>()
            / trajectories_with_normalized_rewards.len() as f32;

        let total_lines_cleared: u32 = trajectories_with_normalized_rewards
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
            "  Raw rewards: mean={:.1}, std={:.1}",
            mean_raw_reward, std_reward
        );
        println!(
            "  Normalized rewards [0, {:.1}]: mean={:.3}, range=[{:.3}, {:.3}]",
            REWARD_SCALE, mean_reward, min_normalized, max_normalized
        );

        // === Phase 3: Supervised learning with prioritized trajectory sampling ===
        // Step 1: Build per-trajectory transition pools with prioritization weights
        let trajectory_pools: Vec<Vec<(TetrisBoard, TetrisPiece, TetrisPieceOrientation, f32)>> =
            trajectories_with_normalized_rewards
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
        let trajectory_weights: Vec<f64> = trajectories_with_normalized_rewards
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

                // Prepare mini-batch tensors and extract weights
                let (batch_boards, batch_pieces, batch_targets, batch_weights): (
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                ) = mini_batch
                    .iter()
                    .map(|(board, piece, orientation, weight)| {
                        (board.clone(), *piece, *orientation, *weight)
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
                let logits = policy.forward_masked(&board_tensor, &piece_tensor)?;

                // Compute probabilities and log probabilities
                let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;
                let log_probs = candle_nn::ops::log_softmax(logits.inner(), D::Minus1)?;

                // Policy gradient loss: L(θ) = -∑ w_t log(π_θ(a_t|s_t))
                // Where w_t ∈ [0, REWARD_SCALE] is the normalized reward with:
                //   - Length bonus: longer trajectories weighted higher (exponential)
                //   - Linear decay: earlier actions weighted higher within trajectory
                //   - REWARD_SCALE: controls gradient magnitude for tuning learning speed
                let target_one_hot = target_tensor.into_dist()?;
                let sample_log_probs = (&log_probs * target_one_hot.inner())?.sum(D::Minus1)?; // [B] - log π(a|s) for each sample

                // Apply normalized rewards: w_t ∈ [0, 1]
                // Higher reward → increase log prob → learn to do this action more
                // Lower reward → smaller gradient → learn this action less
                let rewards_tensor =
                    Tensor::from_vec(batch_weights, Shape::from_dims(&[mini_batch_size]), &device)?;

                let weighted_log_probs = (&sample_log_probs * &rewards_tensor)?;

                // Clamp weighted values to prevent extreme outliers
                let clamped_weighted =
                    weighted_log_probs.clamp(-MAX_WEIGHTED_LOSS, MAX_WEIGHTED_LOSS)?;

                // Policy gradient loss: -mean(w_t × log π(a_t|s_t))
                // w_t ∈ [0, REWARD_SCALE]: higher reward → larger gradient → reinforce this action
                // Maximizing weighted log probs = minimizing negative weighted log probs
                let pg_loss = clamped_weighted.neg()?.mean_all()?;

                // Compute entropy regularization
                let entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?;
                let avg_entropy = entropy.mean_all()?;

                // Total loss: policy gradient + entropy penalty (negative to encourage higher entropy)
                let entropy_penalty = avg_entropy.affine(-ENTROPY_WEIGHT as f64, 0.0)?;
                let loss = (&pg_loss + &entropy_penalty)?;

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

                // Safety check: Skip batch if loss is non-finite
                if !loss_scalar.is_finite() {
                    eprintln!("  ⚠️  Non-finite loss - SKIPPING BATCH");
                    continue;
                }

                total_loss_accum += loss_scalar;
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

            // Compute epoch averages
            let avg_loss = total_loss_accum / batches_processed as f32;
            let avg_accuracy = total_accuracy_accum / batches_processed as f32;
            let avg_entropy = total_entropy_accum / batches_processed as f32;
            let avg_grad_norm = total_grad_norm_accum / batches_processed as f32;

            epoch_avg_loss += avg_loss;
            epoch_avg_accuracy += avg_accuracy;
            epoch_avg_entropy += avg_entropy;
            epoch_avg_grad_norm += avg_grad_norm;

            println!(
                "    Epoch {}/{}: Loss={:.4}, Acc={:.2}%, Entropy={:.3}, GradNorm={:.4}",
                epoch + 1,
                NUM_EPOCHS_PER_BATCH,
                avg_loss,
                avg_accuracy * 100.0,
                avg_entropy,
                avg_grad_norm
            );
        }

        // Average across all epochs
        let avg_loss = epoch_avg_loss / NUM_EPOCHS_PER_BATCH as f32;
        let avg_accuracy = epoch_avg_accuracy / NUM_EPOCHS_PER_BATCH as f32;
        let avg_entropy = epoch_avg_entropy / NUM_EPOCHS_PER_BATCH as f32;
        let avg_grad_norm = epoch_avg_grad_norm / NUM_EPOCHS_PER_BATCH as f32;

        println!(
            "  Multi-epoch training complete: Avg Loss={:.4}, Avg Accuracy={:.2}%, Avg Entropy={:.3}",
            avg_loss,
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
            s.add_scalar("loss/policy_gradient", avg_loss, iteration);
            s.add_scalar("metrics/accuracy", avg_accuracy, iteration);
            s.add_scalar("metrics/entropy", avg_entropy, iteration);
            s.add_scalar("training/grad_norm", avg_grad_norm, iteration);
            s.add_scalar("training/temperature", temperature, iteration);
            s.add_scalar("curriculum/mean_game_length", mean_game_length, iteration);
            s.add_scalar(
                "curriculum/mean_lines_cleared",
                mean_lines_cleared,
                iteration,
            );
            s.add_scalar("curriculum/mean_reward", mean_reward, iteration);
            s.add_scalar(
                "curriculum/max_trajectory_length",
                max_trajectory_length,
                iteration,
            );
            s.add_scalar("curriculum/mean_raw_reward", mean_raw_reward, iteration);
            s.add_scalar("curriculum/std_raw_reward", std_reward, iteration);
            s.add_scalar(
                "curriculum/total_transitions",
                total_transitions as f32,
                iteration,
            );

            // Add trajectory length distribution histogram
            let trajectory_lengths: Vec<f64> = trajectories_with_normalized_rewards
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
            let lines_cleared_dist: Vec<f64> = trajectories_with_normalized_rewards
                .iter()
                .map(|(t, _)| t.lines_cleared as f64)
                .collect();
            s.add_histogram(
                "curriculum/lines_cleared_distribution",
                &lines_cleared_dist,
                None,
                iteration,
            );

            // Add reward distribution histogram (flatten all timestep rewards)
            let all_rewards_hist: Vec<f64> = trajectories_with_normalized_rewards
                .iter()
                .flat_map(|(_, rewards)| rewards.iter().map(|&r| r as f64))
                .collect();
            s.add_histogram(
                "curriculum/reward_distribution",
                &all_rewards_hist,
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
                temperature,
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
                temperature,
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
