use clap::Parser;
use std::str::FromStr;
use tetris_ml::set_global_threadpool;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::{D, DType, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
use rand::Rng;
use tensorboard::summary_writer::SummaryWriter;

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_ml::checkpointer::Checkpointer;
use tetris_ml::grad_accum::GradientAccumulator;
use tetris_ml::modules::{AttnMlp, AttnMlpConfig, Mlp, MlpConfig};
use tetris_ml::fdtype;
use tetris_ml::tensors::TetrisPieceTensor;

// ============================================================================
// Board Feature Extraction
// ============================================================================
//
// Replace raw board pixels with a compact feature vector:
// - per-column heights (0..1)
// - per-column holes (0..1)
// - total height (0..1)
// - bumpiness (0..1)
//
// Feature length: 2*WIDTH + 2
const BOARD_FEATURE_DIM: usize = 2 * TetrisBoard::WIDTH + 2;
const MAX_POLICY_TARGET_DIM: usize = TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT;

struct ProgressLine {
    label: &'static str,
    total: usize,
    start: Instant,
    last_print: Instant,
    last_i: usize,
    min_interval: Duration,
}

impl ProgressLine {
    fn new(label: &'static str, total: usize) -> Self {
        let now = Instant::now();
        Self {
            label,
            total: total.max(1),
            start: now,
            last_print: now,
            last_i: 0,
            min_interval: Duration::from_millis(250),
        }
    }

    fn tick(&mut self, i: usize) {
        let now = Instant::now();
        if i == self.last_i {
            return;
        }
        if now.duration_since(self.last_print) < self.min_interval && i + 1 < self.total {
            return;
        }
        self.last_print = now;
        self.last_i = i;

        let done = (i + 1).min(self.total);
        let frac = done as f64 / self.total as f64;
        let pct = (frac * 100.0).min(100.0);
        let elapsed = now.duration_since(self.start).as_secs_f64();
        let rate = if elapsed > 0.0 {
            done as f64 / elapsed
        } else {
            0.0
        };
        let eta = if rate > 0.0 {
            (self.total.saturating_sub(done) as f64) / rate
        } else {
            0.0
        };

        // 30-char bar.
        let bar_w = 30usize;
        let filled = ((frac * bar_w as f64).round() as usize).min(bar_w);
        let bar = format!("{}{}", "=".repeat(filled), " ".repeat(bar_w - filled));

        print!(
            "\r  {} [{}] {:6.2}%  {}/{}  {:.1}/s  ETA {:5.1}s",
            self.label, bar, pct, done, self.total, rate, eta
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    fn finish(&mut self) {
        self.tick(self.total.saturating_sub(1));
        println!();
    }
}

#[inline]
fn board_to_features(board: &TetrisBoard) -> [f32; BOARD_FEATURE_DIM] {
    let heights_u = board.heights();
    let holes_u = board.holes();

    let h_norm = TetrisBoard::HEIGHT as f32;
    let max_h_u = TetrisBoard::HEIGHT as u32;

    let mut out = [0.0f32; BOARD_FEATURE_DIM];

    // Column heights (normalized)
    let mut total_height_u: u32 = 0;
    let mut bumpiness_u: u32 = 0;
    for i in 0..TetrisBoard::WIDTH {
        let h_i = heights_u[i].min(max_h_u);
        total_height_u = total_height_u.wrapping_add(h_i);
        out[i] = h_i as f32 / h_norm;
        if i + 1 < TetrisBoard::WIDTH {
            let h_j = heights_u[i + 1].min(max_h_u);
            bumpiness_u = bumpiness_u.wrapping_add(h_i.abs_diff(h_j));
        }
    }

    // Holes per column (normalized)
    for i in 0..TetrisBoard::WIDTH {
        let holes_i = holes_u[i].min(max_h_u);
        out[TetrisBoard::WIDTH + i] = holes_i as f32 / h_norm;
    }

    // Total column height (normalized by WIDTH*HEIGHT)
    let total_height_norm = (TetrisBoard::WIDTH as f32) * h_norm;
    out[2 * TetrisBoard::WIDTH] = (total_height_u as f32) / total_height_norm;

    // Bumpiness (normalized by (WIDTH-1)*HEIGHT)
    let bumpiness_norm = ((TetrisBoard::WIDTH - 1) as f32) * h_norm;
    out[2 * TetrisBoard::WIDTH + 1] = (bumpiness_u as f32) / bumpiness_norm;

    out
}

#[inline]
fn boards_to_feature_tensor(
    boards: &[TetrisBoard],
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut feats: Vec<f32> = Vec::with_capacity(boards.len() * BOARD_FEATURE_DIM);
    for b in boards {
        feats.extend_from_slice(&board_to_features(b));
    }
    Ok(Tensor::from_vec(feats, (boards.len(), BOARD_FEATURE_DIM), device)?.to_dtype(dtype)?)
}

// ============================================================================
// AlphaZero-Style Value Function Learning for Tetris
// ============================================================================
//
// This implements AlphaZero-style batch learning with Monte Carlo returns:
//
// Key Design:
//   - Network outputs V(state) ∈ [-1, 1] - expected survival (bounded)
//   - Constant reward r=1.0 per step → discounting creates "death proximity" signal
//   - Elite filtering: Only top 50% of games added to buffer (Expert Iteration)
//   - Simple scaling: V = tanh(G / 20.0) for bounded predictions
//   - Action selection: beam search with two-stage sorting
//     1. Minimize cumulative losses (safety first)
//     2. Maximize value (optimization second)
//   - Training: Large replay buffer with uniform sampling
//
// Network:
//   - Board encoder (CNN)
//   - MLP head -> tanh activation → V ∈ [-1, 1]
//
// Value Function learns:
//   Raw: G ≈ 1 + γ + γ² + ... + γ^(steps_until_death)
//   Scaled: V = tanh(G / 20.0)
//   V ≈ +1: excellent position (far from death)
//   V ≈ 0: medium position (~10 steps remaining)
//   V ≈ -1: terrible position (near death)
//
// ============================================================================

/// Policy+Value Network configuration
#[derive(Debug, Clone)]
pub struct PolicyValueNetworkConfig {
    pub board_feature_encoder_config: AttnMlpConfig,
    pub piece_embedding_dim: usize,
    pub trunk_mlp_config: AttnMlpConfig,
    pub policy_head_config: AttnMlpConfig,
    pub value_head_config: AttnMlpConfig,
}

// ============================================================================
// Policy+Value Network: Outputs π(a|s) and V(s)
// ============================================================================

/// AlphaZero-style network with shared trunk and two heads:
/// - Policy head: logits over `TetrisPiecePlacement::NUM_PLACEMENTS`
/// - Value head: scalar in [-1, 1]
#[derive(Debug, Clone)]
pub struct PolicyValueNetwork {
    board_feature_encoder: AttnMlp,
    piece_embedding: Embedding,
    trunk_mlp: AttnMlp,
    policy_head: AttnMlp,
    value_head: AttnMlp,
}

impl PolicyValueNetwork {
    pub fn init(vb: &VarBuilder, cfg: &PolicyValueNetworkConfig) -> Result<Self> {
        let board_feature_encoder = AttnMlp::init(
            &vb.pp("board_feature_encoder"),
            &cfg.board_feature_encoder_config,
        )?;
        let piece_embedding = embedding(
            TetrisPiece::NUM_PIECES,
            cfg.piece_embedding_dim,
            vb.pp("piece_embedding"),
        )?;
        let trunk_mlp = AttnMlp::init(&vb.pp("trunk_mlp"), &cfg.trunk_mlp_config)?;
        let policy_head = AttnMlp::init(&vb.pp("policy_head"), &cfg.policy_head_config)?;
        let value_head = AttnMlp::init(&vb.pp("value_head"), &cfg.value_head_config)?;
        Ok(Self {
            board_feature_encoder,
            piece_embedding,
            trunk_mlp,
            policy_head,
            value_head,
        })
    }

    /// Forward pass producing:
    /// - policy logits [B, NUM_PLACEMENTS]
    /// - value in [-1, 1] [B, 1]
    pub fn forward(
        &self,
        board_features: &Tensor,
        pieces: &TetrisPieceTensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, f) = board_features.dims2()?;
        if f != BOARD_FEATURE_DIM {
            anyhow::bail!(
                "board_features dim mismatch: got {}, expected {}",
                f,
                BOARD_FEATURE_DIM
            );
        }

        // Encode feature vector [B, F] -> [B, D]
        let board_embed = self
            .board_feature_encoder
            .forward(&board_features.reshape(&[b, f])?)?;

        // Piece embedding [B, 1] -> [B, E]
        let piece_embed = self.piece_embedding.forward(pieces)?.squeeze(1)?;

        // Trunk on concatenated embedding [B, D+E] -> [B, H]
        let x = Tensor::cat(&[board_embed, piece_embed], D::Minus1)?;
        let h = self.trunk_mlp.forward(&x)?;

        // Heads
        let policy_logits = self.policy_head.forward(&h)?;
        let value = self.value_head.forward(&h)?.tanh()?;
        Ok((policy_logits, value))
    }
}

// ============================================================================
// Trajectory Collection for Batch Learning (Compute-Bounded Self-Play)
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct PendingTransition {
    board_features: [f32; BOARD_FEATURE_DIM],
    piece: TetrisPiece,
    policy_target: [f32; MAX_POLICY_TARGET_DIM],
    reward: f32,
}

/// A single position in the replay buffer (AlphaZero style)
#[derive(Debug, Clone, Copy)]
struct ReplayBufferEntry {
    board_features: [f32; BOARD_FEATURE_DIM],
    piece: TetrisPiece,
    raw_return: f32, // Raw MC return (before risk adjustment and normalization)
    policy_target: [f32; MAX_POLICY_TARGET_DIM],
}

// ============================================================================
// Value Function Training Loop
// ============================================================================

pub fn train_value_function_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume_from_checkpoint: bool,
) -> Result<()> {
    let device = tetris_ml::device();
    let dtype = fdtype();

    // Hyperparameters - AlphaZero-style training with replay buffer
    const NUM_ITERATIONS: usize = 100_000;
    const NUM_GAMES: usize = 64;
    // Compute-bounded self-play: we collect this many environment steps per iteration.
    // This prevents iteration wall-time from exploding as the agent improves.
    const SELF_PLAY_STEPS_PER_ITERATION: usize = 4096;
    const N_STEP: usize = 32;
    const REPLAY_BUFFER_SIZE: usize = 500_000; // Store last 20k positions (more diversity)
    const MINI_BATCH_SIZE: usize = 1024; // Smaller batches for more frequent updates
    const GRAD_ACCUM_STEPS: usize = 1; // Accumulate gradients over mini-batches
    const BUFFER_SAMPLE_FRACTION: f32 = 0.05; // Train on 50% of buffer per iteration
    const MIN_TRAINING_BATCHES: usize = 10; // Minimum number of mini-batches per iteration
    const DISCOUNT: f32 = 0.95; // Discount factor for MC returns
    const LEARNING_RATE: f64 = 0.0003; // Lower LR for stability
    const GRAD_CLIP_MAX_NORM: f64 = 1.0;
    const CHECKPOINT_INTERVAL: usize = 100;

    // Epsilon-greedy exploration
    const EPSILON_START: f32 = 0.2;
    const EPSILON_END: f32 = 0.05;
    const EPSILON_DECAY_ITERATIONS: usize = 100;

    // Network architecture (same as tetris_policy_gradients.rs)
    let model_dim = 64;

    let policy_value_network_config = PolicyValueNetworkConfig {
        board_feature_encoder_config: AttnMlpConfig {
            hidden_size: BOARD_FEATURE_DIM,
            intermediate_size: 3 * model_dim,
            output_size: model_dim,
            dropout: Some(0.01),
        },
        piece_embedding_dim: model_dim,
        trunk_mlp_config: AttnMlpConfig {
            hidden_size: model_dim + model_dim,
            intermediate_size: 2 * model_dim,
            output_size: model_dim,
            dropout: Some(0.01),
        },
        policy_head_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPiecePlacement::NUM_PLACEMENTS,
            dropout: Some(0.01),
        },
        value_head_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: 1,
            dropout: Some(0.01),
        },
    };

    // Initialize Policy+Value network
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let network = PolicyValueNetwork::init(&vb, &policy_value_network_config)?;

    let model_params = varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params.clone(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let _ = std::fs::create_dir_all(dir);
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    // Load checkpoint if requested
    let mut starting_iteration = 0;
    if resume_from_checkpoint {
        if let Some(ref cp) = checkpointer {
            match cp.load_latest_checkpoint(&mut varmap) {
                Ok(Some(iteration)) => {
                    starting_iteration = iteration;
                    println!("✅ Loaded checkpoint from iteration {}", iteration);
                }
                Ok(None) => println!("ℹ️  No checkpoint found, starting fresh"),
                Err(e) => println!("⚠️  Failed to load checkpoint: {}", e),
            }
        }
    }

    let mut rng = rand::rng();

    // Epsilon decay calculation
    let epsilon_decay = if EPSILON_DECAY_ITERATIONS > 0 {
        (EPSILON_START - EPSILON_END) / EPSILON_DECAY_ITERATIONS as f32
    } else {
        0.0
    };

    // Statistics and replay buffer
    let mut best_pieces: u32 = 0;
    let total_games_completed = AtomicU32::new(0);
    let mut replay_buffer: VecDeque<ReplayBufferEntry> =
        VecDeque::with_capacity(REPLAY_BUFFER_SIZE);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   Policy+Value + Batched MCTS (AlphaZero-style)             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Device: {:?}                                                 ║",
        device
    );
    println!("║ Network: features + (piece embed) → (policy,value) heads    ║");
    println!(
        "║ Collection: {} games/iter (parallel)                         ║",
        NUM_GAMES
    );
    println!(
        "║ Replay Buffer: {} positions (uniform sampling)              ║",
        REPLAY_BUFFER_SIZE
    );
    println!(
        "║ Training: Sample {:.0}% of buffer per iter, batch={}        ║",
        BUFFER_SAMPLE_FRACTION * 100.0,
        MINI_BATCH_SIZE
    );
    println!("║ Rewards: Constant +1.0 survival/step (death proximity)      ║");
    println!("║ Scaling: Fixed tanh(G/20) for bounded predictions           ║");
    println!(
        "║ Hyperparams: γ={}, lr={}, ε: {}→{}                ║",
        DISCOUNT, LEARNING_RATE, EPSILON_START, EPSILON_END
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Persistent games across iterations keeps compute stable and avoids reallocation.
    let mut games = tetris_game::TetrisGameSet::new(NUM_GAMES);
    let n = games.len();
    let mut pending: Vec<VecDeque<PendingTransition>> = (0..n)
        .map(|_| VecDeque::with_capacity(N_STEP + 8))
        .collect();
    let mut episode_lengths: Vec<usize> = vec![0; n];

    for iteration in starting_iteration..NUM_ITERATIONS {
        // Calculate current epsilon
        let epsilon = if iteration < EPSILON_DECAY_ITERATIONS {
            EPSILON_START - epsilon_decay * iteration as f32
        } else {
            EPSILON_END
        };

        // === Phase 1: Collect Self-Play Steps (Compute-Bounded) ===
        println!(
            "\n[Iteration {}] Self-play: collecting {} environment steps...",
            iteration, SELF_PLAY_STEPS_PER_ITERATION
        );
        let mut self_play_progress = ProgressLine::new("self_play", SELF_PLAY_STEPS_PER_ITERATION);

        let mut new_positions = 0usize;
        let mut ended_episodes = 0usize;
        let mut ended_lengths_sum = 0usize;
        let mut ended_lengths_max = 0usize;
        let mut ended_lines_sum = 0u32;

        for step in 0..SELF_PLAY_STEPS_PER_ITERATION {
            self_play_progress.tick(step);
            // One batched network evaluation for action selection + policy targets.
            let boards: Vec<TetrisBoard> = (0..n).map(|i| games[i].board).collect();
            let pieces: Vec<TetrisPiece> = (0..n).map(|i| games[i].current_piece).collect();
            let board_tensor = boards_to_feature_tensor(&boards, &device, dtype)?;
            let piece_tensor = TetrisPieceTensor::from_pieces(&pieces, &device)?;
            let (policy_logits, _v_now) = network.forward(&board_tensor, &piece_tensor)?;

            // Convert logits -> probabilities over the global placement space.
            let probs = candle_nn::ops::softmax(&policy_logits, D::Minus1)?;
            let probs = probs.to_dtype(DType::F32)?;
            // NOTE: `to_vec1` requires a rank-1 tensor; explicitly flatten for Metal/CUDA backends.
            let flat = probs.flatten_all()?.to_vec1::<f32>()?;
            let row_len = TetrisPiecePlacement::NUM_PLACEMENTS;
            debug_assert_eq!(flat.len(), n * row_len);

            // Choose one action per game (epsilon-greedy), using network policy.
            let mut actions: Vec<TetrisPiecePlacement> = Vec::with_capacity(n);
            for i in 0..n {
                let piece = games[i].current_piece;
                let placements = games[i].current_placements();
                if placements.is_empty() {
                    // Should never happen, but keep the loop robust.
                    actions.push(TetrisPiecePlacement::ALL_PLACEMENTS[0]);
                    continue;
                }

                let chosen = if rng.random::<f32>() < epsilon {
                    placements[rng.random_range(0..placements.len())]
                } else {
                    let idx_range = TetrisPiecePlacement::indices_from_piece(piece);
                    let mut best_j = 0usize;
                    let mut best_p = f32::NEG_INFINITY;
                    for j in 0..placements.len() {
                        let gi = idx_range.start + j;
                        let p = flat[i * row_len + gi];
                        if p > best_p {
                            best_p = p;
                            best_j = j;
                        }
                    }
                    placements[best_j]
                };
                actions.push(chosen);
            }

            // Add one pending transition per game (state + policy target).
            for i in 0..n {
                // Policy target = normalized network probabilities over legal actions.
                let piece = games[i].current_piece;
                let legal = TetrisPiecePlacement::indices_from_piece(piece);
                let k = legal.len().min(MAX_POLICY_TARGET_DIM);
                let mut target = [0.0f32; MAX_POLICY_TARGET_DIM];
                let mut sum = 0.0f32;
                let row_len = TetrisPiecePlacement::NUM_PLACEMENTS;
                for j in 0..k {
                    let idx = legal.start + j;
                    let p = flat[i * row_len + idx];
                    target[j] = p;
                    sum += p;
                }
                if k > 0 {
                    if sum <= 0.0 {
                        let uniform = 1.0 / (k as f32);
                        target[..k].fill(uniform);
                    } else {
                        for p in target[..k].iter_mut() {
                            *p /= sum;
                        }
                    }
                }
                pending[i].push_back(PendingTransition {
                    board_features: board_to_features(&games[i].board),
                    piece,
                    policy_target: target,
                    reward: 0.0,
                });
            }

            // Step env and advance trees.
            let lost = games.apply_placement(&actions);

            // Assign rewards for the just-added transitions and handle episode endings.
            for i in 0..n {
                let is_lost = bool::from(lost[i]);
                if let Some(last) = pending[i].back_mut() {
                    last.reward = if is_lost { 0.0 } else { 1.0 };
                }

                episode_lengths[i] += 1;

                if is_lost {
                    // Flush remaining pending transitions with terminal bootstrap V=0.
                    let len = pending[i].len();
                    if len > 0 {
                        for t in 0..len {
                            let mut g = 0.0f32;
                            let mut pow = 1.0f32;
                            for k in t..len {
                                g += pow * pending[i][k].reward;
                                pow *= DISCOUNT;
                            }
                            let tr = pending[i][t];
                            replay_buffer.push_back(ReplayBufferEntry {
                                board_features: tr.board_features,
                                piece: tr.piece,
                                raw_return: g,
                                policy_target: tr.policy_target,
                            });
                            new_positions += 1;
                            if replay_buffer.len() > REPLAY_BUFFER_SIZE {
                                replay_buffer.pop_front();
                            }
                        }
                    }
                    pending[i].clear();

                    // Episode stats.
                    ended_episodes += 1;
                    ended_lengths_sum += episode_lengths[i];
                    ended_lengths_max = ended_lengths_max.max(episode_lengths[i]);
                    ended_lines_sum = ended_lines_sum.saturating_add(games[i].lines_cleared);
                    episode_lengths[i] = 0;

                    total_games_completed.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Reset lost envs in-place for continued self-play.
            let _ = games.reset_lost_games();

            // Bootstrap n-step targets for non-terminal games using V(s_{t+N}).
            // We evaluate the current root states once per step (cheap relative to MCTS).
            // Convert V in [-1, 1] to an approximate raw-return scale in [0, R_MAX].
            let r_max = 1.0f32 / (1.0f32 - DISCOUNT);
            let boards: Vec<TetrisBoard> = (0..n).map(|i| games[i].board).collect();
            let pieces: Vec<TetrisPiece> = (0..n).map(|i| games[i].current_piece).collect();
            let board_tensor = boards_to_feature_tensor(&boards, &device, dtype)?;
            let piece_tensor = TetrisPieceTensor::from_pieces(&pieces, &device)?;
            let (_, v) = network.forward(&board_tensor, &piece_tensor)?;
            let v = v.squeeze(1)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut bootstrap_raw: Vec<f32> = Vec::with_capacity(n);
            for &vv in &v {
                let vv01 = ((vv + 1.0) * 0.5).clamp(0.0, 1.0);
                bootstrap_raw.push(vv01 * r_max);
            }

            // Finalize one n-step transition per game (steady-state) using the current bootstrap.
            for i in 0..n {
                if pending[i].len() >= N_STEP {
                    let mut g = 0.0f32;
                    let mut pow = 1.0f32;
                    for k in 0..N_STEP {
                        g += pow * pending[i][k].reward;
                        pow *= DISCOUNT;
                    }
                    g += pow * bootstrap_raw[i];

                    if let Some(tr) = pending[i].pop_front() {
                        replay_buffer.push_back(ReplayBufferEntry {
                            board_features: tr.board_features,
                            piece: tr.piece,
                            raw_return: g,
                            policy_target: tr.policy_target,
                        });
                        new_positions += 1;
                        if replay_buffer.len() > REPLAY_BUFFER_SIZE {
                            replay_buffer.pop_front();
                        }
                    }
                }
            }
        }
        self_play_progress.finish();

        println!(
            "  ✓ Self-play complete! (+{} positions, {} episodes ended)",
            new_positions, ended_episodes
        );
        if ended_episodes > 0 {
            let avg_len = ended_lengths_sum as f32 / ended_episodes as f32;
            let avg_lines = ended_lines_sum as f32 / ended_episodes as f32;
            println!(
                "  Episodes: avg_len={:.1} max_len={} avg_lines={:.1}",
                avg_len, ended_lengths_max, avg_lines
            );
        }

        // Skip training if buffer is too small
        if replay_buffer.len() < MINI_BATCH_SIZE {
            println!(
                "  Skipping training (buffer too small, need {})",
                MINI_BATCH_SIZE
            );
            continue;
        }

        // === Phase 3: Train on Replay Buffer (Dynamic Sampling) ===
        // Calculate how many positions to sample based on buffer size
        let num_positions_to_sample =
            (replay_buffer.len() as f32 * BUFFER_SAMPLE_FRACTION) as usize;
        let num_positions_to_sample = num_positions_to_sample.max(MINI_BATCH_SIZE);

        // Calculate number of mini-batches
        let num_training_batches =
            (num_positions_to_sample / MINI_BATCH_SIZE).max(MIN_TRAINING_BATCHES);

        println!(
            "  Training: {} samples in {} mini-batches ({:.1}% of buffer)...",
            num_positions_to_sample,
            num_training_batches,
            (num_positions_to_sample as f32 / replay_buffer.len() as f32) * 100.0
        );
        let mut train_progress = ProgressLine::new("train", num_training_batches);

        // === Train on Replay Buffer (Simple Normalization) ===
        let mut grad_accumulator = GradientAccumulator::new(GRAD_ACCUM_STEPS);
        let mut total_loss = 0.0f32;
        let mut num_batches = 0;
        let mut nan_batches = 0;

        for batch_idx in 0..num_training_batches {
            train_progress.tick(batch_idx);
            // Uniform random sampling from replay buffer
            let batch_size = MINI_BATCH_SIZE.min(replay_buffer.len());
            let mut sampled_indices: Vec<usize> = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                sampled_indices.push(rng.random_range(0..replay_buffer.len()));
            }

            let mut batch_board_features: Vec<f32> =
                Vec::with_capacity(sampled_indices.len() * BOARD_FEATURE_DIM);
            for &i in &sampled_indices {
                batch_board_features.extend_from_slice(&replay_buffer[i].board_features);
            }
            let batch_pieces: Vec<TetrisPiece> = sampled_indices
                .iter()
                .map(|&i| replay_buffer[i].piece)
                .collect();
            let mut batch_policy_targets: Vec<f32> =
                Vec::with_capacity(sampled_indices.len() * MAX_POLICY_TARGET_DIM);
            let mut batch_policy_indices: Vec<u32> =
                Vec::with_capacity(sampled_indices.len() * MAX_POLICY_TARGET_DIM);
            for &idx in &sampled_indices {
                let entry = replay_buffer[idx];
                batch_policy_targets.extend_from_slice(&entry.policy_target);
                let legal = TetrisPiecePlacement::indices_from_piece(entry.piece);
                let start = legal.start;
                let k = legal.len().min(MAX_POLICY_TARGET_DIM);
                for j in 0..k {
                    batch_policy_indices.push((start + j) as u32);
                }
                for _ in k..MAX_POLICY_TARGET_DIM {
                    batch_policy_indices.push(start as u32);
                }
            }

            // Get raw returns and apply simple fixed scaling
            let batch_raw_returns: Vec<f32> = sampled_indices
                .iter()
                .map(|&i| replay_buffer[i].raw_return)
                .collect();

            // Normalize survival return into [-1, 1] using R_MAX = 1/(1-γ).
            // This uses the full output range (negative = worse-than-average, positive = better).
            let r_max = 1.0f32 / (1.0f32 - DISCOUNT);
            let batch_targets: Vec<f32> = batch_raw_returns
                .iter()
                .map(|&r| (2.0 * (r / r_max) - 1.0).clamp(-1.0, 1.0))
                .collect();

            let board_tensor = Tensor::from_vec(
                batch_board_features,
                (batch_targets.len(), BOARD_FEATURE_DIM),
                &device,
            )?
            .to_dtype(dtype)?;
            let piece_tensor = TetrisPieceTensor::from_pieces(&batch_pieces, &device)?;
            let returns_tensor =
                Tensor::from_vec(batch_targets.clone(), (batch_targets.len(),), &device)?
                    .to_dtype(dtype)?;
            let policy_target_tensor = Tensor::from_vec(
                batch_policy_targets,
                (batch_targets.len(), MAX_POLICY_TARGET_DIM),
                &device,
            )?
            .to_dtype(dtype)?;
            let policy_index_tensor = Tensor::from_vec(
                batch_policy_indices,
                (batch_targets.len(), MAX_POLICY_TARGET_DIM),
                &device,
            )?
            .to_dtype(DType::U32)?;

            // Forward pass
            let (policy_logits, value) = network.forward(&board_tensor, &piece_tensor)?;
            let predictions = value.squeeze(1)?;

            // Losses: value MSE + policy cross-entropy to MCTS target distribution.
            let value_loss = (&predictions - &returns_tensor)?.sqr()?.mean_all()?;
            let log_probs = candle_nn::ops::log_softmax(
                &policy_logits
                    .reshape(&[batch_targets.len(), TetrisPiecePlacement::NUM_PLACEMENTS])?,
                D::Minus1,
            )?;
            let log_probs_legal = log_probs.gather(&policy_index_tensor, D::Minus1)?;
            let policy_loss = (&policy_target_tensor * &log_probs_legal)?
                .sum(D::Minus1)?
                .neg()?
                .mean_all()?;

            let loss = (&value_loss + &policy_loss.affine(1.0f64, 0.0)?)?;

            let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            if loss_val.is_nan() || loss_val.is_infinite() {
                nan_batches += 1;
                if nan_batches <= 3 {
                    eprintln!(
                        "  WARNING: Invalid loss detected (nan={}, inf={})",
                        loss_val.is_nan(),
                        loss_val.is_infinite()
                    );
                }
                continue;
            }

            // Backprop and accumulate gradients
            let grads = loss.backward()?;
            grad_accumulator.accumulate(grads, &model_params)?;

            // If we're doing grad accumulation, step as soon as we're ready.
            // NOTE: With GRAD_ACCUM_STEPS=1 this applies every mini-batch (intended).
            let _applied = grad_accumulator.apply_and_reset(
                &mut optimizer,
                &model_params,
                Some(GRAD_CLIP_MAX_NORM),
                None,
            )?;

            total_loss += loss_val;
            num_batches += 1;
        }
        train_progress.finish();

        // Apply any remaining accumulated gradients (when GRAD_ACCUM_STEPS > 1).
        let _ = grad_accumulator.apply_and_reset(
            &mut optimizer,
            &model_params,
            Some(GRAD_CLIP_MAX_NORM),
            None,
        )?;

        if nan_batches > 0 {
            eprintln!(
                "  WARNING: Skipped {} batches due to NaN/Inf losses",
                nan_batches
            );
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            eprintln!("  ERROR: No valid batches processed! All losses were NaN/Inf");
            0.0
        };

        // === Statistics and Logging ===
        let avg_game_length = if ended_episodes == 0 {
            0.0
        } else {
            ended_lengths_sum as f32 / ended_episodes as f32
        };
        let max_game_length = ended_lengths_max;
        let avg_lines = if ended_episodes == 0 {
            0.0
        } else {
            ended_lines_sum as f32 / ended_episodes as f32
        };

        println!(
            "  [Iter {}] ε={:.3}, Pieces: avg={:.1} max={}, Lines: {:.1}, Loss: {:.4}",
            iteration, epsilon, avg_game_length, max_game_length, avg_lines, avg_loss
        );

        // Compute standard deviation of game lengths for logging
        // Std requires a distribution; we only track per-iteration aggregates in this loop.
        let game_length_std = 0.0;

        // TensorBoard logging
        if let Some(ref mut writer) = summary_writer {
            writer.add_scalar("pieces/avg", avg_game_length, iteration);
            writer.add_scalar("pieces/max", max_game_length as f32, iteration);
            writer.add_scalar("pieces/std", game_length_std, iteration);
            writer.add_scalar("lines/avg", avg_lines, iteration);
            writer.add_scalar("metrics/epsilon", epsilon, iteration);
            writer.add_scalar("train/loss", avg_loss, iteration);
        }

        // Save best model
        if max_game_length as u32 > best_pieces {
            best_pieces = max_game_length as u32;
            println!(
                "   🏆 New best: {} pieces (iteration {})",
                best_pieces, iteration
            );
            if let Some(ref cp) = checkpointer {
                let _ = cp.checkpoint_item(iteration, &varmap, Some("best"), None);
            }
        }

        // Periodic checkpoint
        if let Some(ref cp) = checkpointer {
            if iteration % CHECKPOINT_INTERVAL == 0 && iteration > 0 {
                let _ = cp.checkpoint_item(iteration, &varmap, None, None);
            }
        }
    }

    println!("\n✅ Training complete!");
    println!(
        "   Total games: {}",
        total_games_completed.load(Ordering::Relaxed)
    );
    println!("   Best: {} pieces", best_pieces);

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

    #[arg(long, help = "Resume training from latest checkpoint")]
    resume: bool,
}

fn main() {
    info!("Starting tetris value function training");
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

    train_value_function_policy(run_name, logdir, checkpoint_dir, cli.resume).unwrap();
}
