//! # Optimal Board Sharding via Neural Network
//!
//! This binary trains a neural network to optimally shard Tetris boards across N shards.
//! The goal is to learn a sharding function that:
//! 1. Balances boards evenly across shards (50/50 for 2 shards)
//! 2. Minimizes cross-shard transitions during gameplay sequences
//!
//! ## Architecture
//!
//! - Input: 200-dim binary vector (10x20 Tetris board, flattened)
//! - Output: Single scalar [0,1] representing shard assignment
//! - For N shards: partition [0,1] into N equal ranges
//!   - Shard 0: [0, 1/N)
//!   - Shard 1: [1/N, 2/N)
//!   - ...
//!   - Shard N-1: [(N-1)/N, 1]
//!
//! ## Loss Function
//!
//! ```
//! Loss = α(t) * balance_loss + β(t) * transition_loss
//!
//! balance_loss = mean(exp(5 * |count(shard_i) / total - 1/N|) - 1)
//! transition_loss = exp(3 * mean_L2_distance / sqrt(2)) - 1
//!
//! α(t) = 20.0 - 15.0 * min(t/T / 0.15, 1.0)  // Balance weight: 20.0 → 5.0 (in first 15%)
//! β(t) = 1.0 + 4.0 * min(t/T / 0.15, 1.0)    // Transition weight: 1.0 → 5.0 (in first 15%)
//! ```
//!
//! **Exponential scaling**: Both losses use exp(k * x) - 1 to dramatically reward improvements.
//! As the model gets better, small gains result in much lower loss, creating stronger gradients.
//!
//! **Curriculum learning**: Weights transition quickly (within first 15% of training), prioritizing
//! balance initially (α=20, β=1), then both objectives equally (α=5, β=5) for the remaining 85%.
//!
//! ## Training Strategy
//!
//! - Generate long sequences (100k+ boards) from the infinite Tetris player
//! - Process entire sequence as a single batch for stable loss estimates
//! - The network learns temporal locality: consecutive boards should stay in same shard
//!
//! ## Output
//!
//! - **Console**: Training progress with loss components
//! - **Checkpoints**: Saved model weights for inference

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use std::path::PathBuf;
use tensorboard::summary_writer::SummaryWriter;
use tetris_game::{TetrisBoard, TetrisPieceOrientation, TetrisPiecePlacement};
use tetris_ml::beam_search::{BeamSearch, BeamTetrisState};
use tetris_ml::grad_accum::GradientAccumulator;
use tetris_ml::modules::{Mlp, MlpConfig};
use tetris_ml::{device, set_global_threadpool};
use tetris_ml::fdtype;
use tetris_ml::ops::get_l2_norm;
use tetris_ml::tensors::TetrisBoardsTensor;

#[derive(Parser, Debug)]
#[command(name = "optimal_board_sharding")]
#[command(about = "Train a neural network to optimally shard Tetris boards", long_about = None)]
struct Args {
    /// Optional directory for tensorboard logs. If not provided, no stats will be stored.
    #[arg(long)]
    logdir: Option<PathBuf>,

    /// Number of shards to partition boards into (must be >= 2)
    #[arg(long, default_value = "2")]
    num_shards: usize,
}
use tetris_ml::wrapped_tensor::WrappedTensor;

/// Infinite Tetris player iterator using single beam search
pub struct TetrisGameBeamSearchIter<
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> {
    pub game: tetris_game::TetrisGame,
    search: BeamSearch<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameBeamSearchIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        Self {
            game: tetris_game::TetrisGame::new(),
            search: BeamSearch::new(),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            game: tetris_game::TetrisGame::new_with_seed(seed),
            search: BeamSearch::new(),
        }
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameBeamSearchIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;

        let scored = self
            .search
            .search_top_with_state(BeamTetrisState::new(self.game), MAX_DEPTH)?;
        let first_action = scored.root_action?;

        (self.game.apply_placement(first_action).is_lost != tetris_game::IsLost::LOST)
            .then_some((board_before, first_action))
    }
}

/// MLP that maps a 200-dim board to a scalar [0,1]
#[derive(Debug, Clone)]
pub struct BoardShardingNetwork {
    mlp: Mlp,
}

impl BoardShardingNetwork {
    pub fn init(vb: &VarBuilder) -> Result<Self> {
        // Deeper MLP: 200 -> 256 -> 128 -> 64 -> 1
        let mlp_cfg = MlpConfig {
            input_size: TetrisBoard::SIZE,
            hidden_sizes: vec![256, 128, 128, 64],
            output_size: 1,
            dropout: None,
        };

        let mlp = Mlp::init(&vb.pp("mlp"), &mlp_cfg)?;

        Ok(Self { mlp })
    }

    /// Forward pass: board [B, 200] -> scalar [B, 1]
    /// Apply sigmoid to constrain output to [0, 1]
    pub fn forward(&self, board: &Tensor) -> Result<Tensor> {
        let x = board.to_dtype(fdtype())?;
        let logit = self.mlp.forward(&x)?;
        // Apply sigmoid to get [0, 1] range
        Ok(candle_nn::ops::sigmoid(&logit)?)
    }
}

/// Generate a sequence of boards from the continuous infinite player
fn generate_sequence<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>(
    iter: &mut TetrisGameBeamSearchIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
    sequence_length: usize,
) -> Result<Vec<TetrisBoard>> {
    let mut boards = Vec::with_capacity(sequence_length);

    for _ in 0..sequence_length {
        match iter.next() {
            Some((board, _placement)) => boards.push(board),
            None => {
                // If game ends, reset and continue
                iter.game.reset(None);
                if let Some((board, _placement)) = iter.next() {
                    boards.push(board);
                } else {
                    break;
                }
            }
        }
    }

    Ok(boards)
}

/// Convert boards to tensor [B, 200]
fn boards_to_tensor(boards: &[TetrisBoard], device: &Device) -> Result<Tensor> {
    let tensor = TetrisBoardsTensor::from_boards(boards, device)?;
    Ok(tensor.inner().clone())
}

/// Compute SOFT shard assignments from scalar predictions (differentiable!)
/// For N shards, returns a [B, N] tensor where each row sums to 1
/// Uses softmax over distances to shard centers for smooth, differentiable assignments
fn compute_soft_shard_assignments(predictions: &Tensor, num_shards: usize) -> Result<Tensor> {
    let device = predictions.device();

    // Create shard centers: [0/(N-1), 1/(N-1), ..., (N-1)/(N-1)] = [0, 1/N, 2/N, ..., 1]
    let shard_centers: Vec<f32> = (0..num_shards)
        .map(|i| {
            if num_shards == 1 {
                0.5
            } else {
                i as f32 / (num_shards - 1) as f32
            }
        })
        .collect();
    let centers_tensor = Tensor::new(shard_centers.as_slice(), device)?.to_dtype(fdtype())?; // [N]

    // Reshape predictions to [B, 1] and centers to [1, N] for broadcasting
    let pred_expanded = predictions.unsqueeze(1)?; // [B, 1]
    let centers_expanded = centers_tensor.unsqueeze(0)?; // [1, N]

    // Compute squared distances: [B, N]
    let distances = pred_expanded.broadcast_sub(&centers_expanded)?;
    let squared_distances = (&distances * &distances)?;

    // Convert to soft assignments via softmax with temperature
    // Higher temperature = softer assignments, lower = harder
    // Using 1.0 for numerical stability with BF16
    let temperature = 1.0f32;
    let temp_tensor = Tensor::new(temperature, device)?.to_dtype(fdtype())?;
    let logits = squared_distances.broadcast_div(&temp_tensor)?.neg()?;

    // Apply softmax to get probabilities [B, N]
    // softmax is numerically stable (uses log-sum-exp trick internally)
    let soft_assignments = candle_nn::ops::softmax(&logits, 1)?;

    Ok(soft_assignments)
}

/// Compute hard shard assignments for evaluation/logging (non-differentiable)
/// For N shards: partition [0,1] into N equal ranges
/// shard_idx = floor(prediction * N), clamped to [0, N-1]
fn compute_hard_shard_assignments(predictions: &Tensor, num_shards: usize) -> Result<Tensor> {
    let num_shards_tensor = Tensor::new(num_shards as f32, predictions.device())?
        .to_dtype(fdtype())?
        .broadcast_as(predictions.shape())?;

    // Compute shard_idx = floor(prediction * N)
    let scaled = predictions.broadcast_mul(&num_shards_tensor)?;
    let shard_indices = scaled.floor()?;

    // Clamp to [0, N-1] to handle edge case where prediction = 1.0
    let max_shard = Tensor::new((num_shards - 1) as f32, predictions.device())?
        .to_dtype(fdtype())?
        .broadcast_as(predictions.shape())?;
    let zero = Tensor::new(0.0f32, predictions.device())?
        .to_dtype(fdtype())?
        .broadcast_as(predictions.shape())?;

    let clamped = shard_indices.maximum(&zero)?.minimum(&max_shard)?;
    Ok(clamped)
}

/// Compute balance loss with SOFT assignments: exponentially scaled deviation from uniform distribution
/// soft_assignments: [B, N] where each row sums to 1
/// For each shard i, sum the probabilities: count(shard_i) = sum_b(soft_assignments[b, i])
/// Then compute exp(|count(shard_i) / B - 1/N|) - 1 and average over shards
/// This exponential scaling means small improvements give dramatically better loss
fn compute_balance_loss_soft(soft_assignments: &Tensor, num_shards: usize) -> Result<Tensor> {
    let device = soft_assignments.device();
    let batch_size = soft_assignments.dims()[0] as f32;
    let target_ratio = 1.0 / num_shards as f32;

    // Sum soft assignments along batch dimension to get expected count per shard [N]
    let shard_counts = soft_assignments.sum(0)?; // [N]

    // Compute ratios: count / batch_size
    let batch_size_tensor = Tensor::new(batch_size, device)?.to_dtype(fdtype())?;
    let ratios = shard_counts.broadcast_div(&batch_size_tensor)?; // [N]

    // Create target tensor with same shape as ratios for broadcasting
    let target_tensor = Tensor::new(target_ratio, device)?
        .to_dtype(fdtype())?
        .broadcast_as(ratios.shape())?;

    // Compute deviations from target
    let deviations = ratios.broadcast_sub(&target_tensor)?.abs()?;

    // Exponential scaling: exp(k * deviation) - 1
    // where k=5 gives: 0% deviation → 0, 10% deviation → 0.65, 20% deviation → 1.72
    let scale_factor = Tensor::new(5.0f32, device)?.to_dtype(fdtype())?;
    let scaled_deviations = deviations.broadcast_mul(&scale_factor)?;
    let exp_deviations = scaled_deviations.exp()?;
    let one = Tensor::new(1.0f32, device)?.to_dtype(fdtype())?;
    let exp_loss = exp_deviations.broadcast_sub(&one)?;

    let mean_loss = exp_loss.mean_all()?;

    Ok(mean_loss)
}

/// Compute transition loss with SOFT assignments: exponentially scaled L2 distance
/// soft_assignments: [B, N] where each row is a probability distribution over shards
/// For each consecutive pair, compute L2 distance between their distributions
/// Uses exponential scaling: exp(k * distance) - 1 for stronger gradients as model improves
fn compute_transition_loss_soft(soft_assignments: &Tensor) -> Result<Tensor> {
    let batch_size = soft_assignments.dims()[0];
    if batch_size <= 1 {
        return Ok(Tensor::new(0.0f32, soft_assignments.device())?.to_dtype(fdtype())?);
    }

    // Get consecutive pairs: [:-1] and [1:]
    let current = soft_assignments.narrow(0, 0, batch_size - 1)?; // [B-1, N]
    let next = soft_assignments.narrow(0, 1, batch_size - 1)?; // [B-1, N]

    // Compute L2 distance between consecutive distributions
    let diff = current.broadcast_sub(&next)?;
    let squared = (&diff * &diff)?;
    let distances = squared.sum(1)?; // [B-1]
    let sqrt_distances = distances.sqrt()?;

    // Average over all consecutive pairs
    let mean_distance = sqrt_distances.mean_all()?;

    // Normalize by sqrt(2) since max L2 distance between two probability vectors is sqrt(2)
    let normalizer =
        Tensor::new(2.0f32.sqrt() + 1e-8, soft_assignments.device())?.to_dtype(fdtype())?;
    let normalized_distance = (mean_distance / normalizer)?;

    // Exponential scaling: exp(k * distance) - 1
    // where k=3 gives: 0% distance → 0, 10% distance → 0.35, 50% distance → 3.48
    let scale_factor = Tensor::new(3.0f32, soft_assignments.device())?.to_dtype(fdtype())?;
    let scaled_distance = normalized_distance.broadcast_mul(&scale_factor)?;
    let exp_distance = scaled_distance.exp()?;
    let one = Tensor::new(1.0f32, soft_assignments.device())?.to_dtype(fdtype())?;
    let exp_loss = exp_distance.broadcast_sub(&one)?;

    Ok(exp_loss)
}

/// Main training function
pub fn train_optimal_sharding(logdir: Option<PathBuf>, num_shards: usize) -> Result<()> {
    assert!(num_shards >= 2, "Number of shards must be at least 2");

    set_global_threadpool();
    let device = device();

    // Hyperparameters
    const SEQUENCE_LENGTH: usize = 10_000;
    const NUM_ITERATIONS: usize = 1000;
    const LEARNING_RATE: f64 = 1e-4; // Reduced from 1e-3 for stability
    const ALPHA_START: f32 = 20.0; // Balance loss weight at start
    const ALPHA_END: f32 = 5.0; // Balance loss weight at end
    const BETA_START: f32 = 1.0; // Transition loss weight at start
    const BETA_END: f32 = 5.0; // Transition loss weight at end
    const CLIP_GRAD_MAX_NORM: f64 = 1.0;
    const CURRICULUM_FRACTION: f32 = 0.15; // Reach final weights by 15% of training

    println!("Starting optimal board sharding training");
    println!("  Number of shards: {}", num_shards);
    println!("  Sequence length: {}", SEQUENCE_LENGTH);
    println!("  Iterations: {}", NUM_ITERATIONS);
    println!("  Learning rate: {}", LEARNING_RATE);
    println!(
        "  Loss weights (curriculum): α={:.1}→{:.1}, β={:.1}→{:.1} (transition at {:.0}% of training)",
        ALPHA_START, ALPHA_END, BETA_START, BETA_END, CURRICULUM_FRACTION * 100.0
    );

    // Initialize model
    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, fdtype(), &device);
    let network = BoardShardingNetwork::init(&vb)?;

    // Optimizer
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params.clone(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..ParamsAdamW::default()
        },
    )?;
    let mut grad_accum = GradientAccumulator::new(1);

    // Logging
    let mut summary_writer = logdir.map(SummaryWriter::new);

    // Create single continuous game iterator
    const BEAM_WIDTH: usize = 256;
    const MAX_DEPTH: usize = 8;
    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const SEED: u64 = 42;

    let mut game_iter =
        TetrisGameBeamSearchIter::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new_with_seed(SEED);

    println!(
        "Created continuous game iterator (beam_width={}, max_depth={})",
        BEAM_WIDTH, MAX_DEPTH
    );

    // Training loop
    for iteration in 0..NUM_ITERATIONS {
        // Compute dynamic loss weights with curriculum learning
        // Weights transition from start to end values within first 15% of training
        let raw_progress = iteration as f32 / NUM_ITERATIONS as f32;
        let progress = (raw_progress / CURRICULUM_FRACTION).min(1.0);
        let alpha = ALPHA_START - (ALPHA_START - ALPHA_END) * progress;
        let beta = BETA_START + (BETA_END - BETA_START) * progress;

        println!(
            "\n[Iteration {}] (α={:.2}, β={:.2}) Generating sequence...",
            iteration, alpha, beta
        );

        // Generate sequence from continuous game
        let boards = generate_sequence(&mut game_iter, SEQUENCE_LENGTH)?;
        let actual_length = boards.len();

        if actual_length < 2 {
            println!(
                "Warning: Sequence too short ({}), skipping iteration",
                actual_length
            );
            continue;
        }

        println!("  Generated {} boards", actual_length);

        // Convert to tensor
        let board_tensor = boards_to_tensor(&boards, &device)?;

        // Forward pass
        let predictions = network.forward(&board_tensor)?.squeeze(1)?; // [L]

        // Compute SOFT shard assignments for differentiable loss
        let soft_assignments = compute_soft_shard_assignments(&predictions, num_shards)?; // [L, N]

        // Compute losses using soft assignments (differentiable!)
        let balance_loss = compute_balance_loss_soft(&soft_assignments, num_shards)?;
        let transition_loss = compute_transition_loss_soft(&soft_assignments)?;

        // Total loss with dynamic weights
        let alpha_tensor = Tensor::new(alpha, &device)?.to_dtype(fdtype())?;
        let beta_tensor = Tensor::new(beta, &device)?.to_dtype(fdtype())?;
        let total_loss = (balance_loss.broadcast_mul(&alpha_tensor)?
            + transition_loss.broadcast_mul(&beta_tensor)?)?;

        // Extract scalars for logging (convert to F32 first)
        let balance_loss_scalar = balance_loss
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        let transition_loss_scalar = transition_loss
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;
        let total_loss_scalar = total_loss
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()?;

        // Check for NaN in losses before backprop
        if total_loss_scalar.is_nan() || total_loss_scalar.is_infinite() {
            println!(
                "  Loss: total={:.6} balance={:.6} transition={:.6}",
                total_loss_scalar, balance_loss_scalar, transition_loss_scalar
            );
            println!("  WARNING: NaN or Inf loss detected. Skipping iteration.");
            println!("  Consider using F32 dtype or adjusting hyperparameters.");
            continue;
        }

        // For evaluation/logging: compute HARD assignments (argmax of soft assignments)
        let hard_assignments = compute_hard_shard_assignments(&predictions, num_shards)?;

        // Compute shard distribution for all N shards
        let mut shard_counts = vec![0usize; num_shards];
        let hard_assignments_f32 = hard_assignments.to_dtype(candle_core::DType::F32)?;
        let hard_assignments_vec = hard_assignments_f32.to_vec1::<f32>()?;
        for &assignment in &hard_assignments_vec {
            let shard_idx = assignment as usize;
            if shard_idx < num_shards {
                shard_counts[shard_idx] += 1;
            }
        }

        // Count transitions (using hard assignments for interpretability)
        let transitions = if actual_length > 1 {
            let current = hard_assignments.narrow(0, 0, actual_length - 1)?;
            let next = hard_assignments.narrow(0, 1, actual_length - 1)?;
            let not_equal = current.ne(&next)?.to_dtype(fdtype())?;
            let transition_sum = not_equal.sum_all()?.to_dtype(candle_core::DType::F32)?;
            transition_sum.to_scalar::<f32>()? as usize
        } else {
            0
        };

        println!(
            "  Loss: total={:.6} balance={:.6} transition={:.6}",
            total_loss_scalar, balance_loss_scalar, transition_loss_scalar
        );

        // Print shard distribution
        print!("  Shard distribution: ");
        for (i, &count) in shard_counts.iter().enumerate() {
            let ratio = (count as f32 / actual_length as f32) * 100.0;
            print!("shard_{}={:.2}% ", i, ratio);
        }
        println!();

        println!(
            "  Transitions: {} / {} pairs ({:.2}%)",
            transitions,
            actual_length - 1,
            (transitions as f32 / (actual_length - 1) as f32) * 100.0
        );

        // Backward pass
        let grads = total_loss.backward()?;
        let grad_norm = get_l2_norm(&grads)?;

        // Check for NaN gradients and skip update if found
        if grad_norm.is_nan() || grad_norm.is_infinite() {
            println!(
                "  Gradient norm: {} (SKIPPING UPDATE - NaN or Inf detected)",
                grad_norm
            );
            println!("  WARNING: Numerical instability detected. Consider:");
            println!("    - Reducing learning rate");
            println!("    - Using F32 instead of BF16");
            println!("    - Increasing temperature in soft assignments");
            continue;
        }

        grad_accum.accumulate(grads, &model_params)?;
        grad_accum.apply_and_reset(
            &mut optimizer,
            &model_params,
            Some(CLIP_GRAD_MAX_NORM),
            None,
        )?;

        println!("  Gradient norm: {:.6}", grad_norm);

        // Logging
        if let Some(s) = summary_writer.as_mut() {
            s.add_scalar("loss/total", total_loss_scalar, iteration);
            s.add_scalar("loss/balance", balance_loss_scalar, iteration);
            s.add_scalar("loss/transition", transition_loss_scalar, iteration);

            // Log distribution for each shard
            for (i, &count) in shard_counts.iter().enumerate() {
                let ratio = count as f32 / actual_length as f32;
                s.add_scalar(&format!("metrics/shard_{}_ratio", i), ratio, iteration);
            }

            s.add_scalar("metrics/num_transitions", transitions as f32, iteration);
            s.add_scalar("metrics/transition_rate", transition_loss_scalar, iteration);
            s.add_scalar("training/grad_norm", grad_norm as f32, iteration);
            s.add_scalar("training/lr", optimizer.learning_rate() as f32, iteration);
            s.add_scalar("training/alpha", alpha, iteration);
            s.add_scalar("training/beta", beta, iteration);
        }
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║         TRAINING COMPLETE!                             ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("  Total iterations: {}", NUM_ITERATIONS);
    println!("  Sequence length per iteration: {}", SEQUENCE_LENGTH);

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    train_optimal_sharding(args.logdir, args.num_shards)
}
