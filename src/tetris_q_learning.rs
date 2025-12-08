use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::Result;
use candle_core::{DType, Shape, Tensor};
use candle_nn::{
    AdamW, Dropout, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear,
};
use rand::Rng;
use rayon::prelude::*;
use tensorboard::summary_writer::SummaryWriter;

use crate::checkpointer::Checkpointer;
use crate::device;
use crate::fdtype;
use crate::modules::{Conv2dConfig, ConvBlock, ConvBlockSpec};
use crate::tetris::{TetrisBoard, TetrisGame, TetrisPiecePlacement};

// ============================================================================
// State-Value DQN for Tetris (matching Python sample-tetris-ai approach)
// ============================================================================
//
// This implementation matches the approach from sample-tetris-ai:
//
// Key Design:
//   - Uses 4 HAND-CRAFTED FEATURES instead of raw board:
//     1. Lines cleared (by this move)
//     2. Number of holes (empty cells with blocks above)
//     3. Total bumpiness (sum of height differences between adjacent columns)
//     4. Sum of heights (aggregate height across all columns)
//
//   - Network outputs a SINGLE VALUE (expected future score)
//     Not Q-values per action, but V(state) for a resulting state
//
//   - Action selection: enumerate ALL possible placements,
//     compute resulting state features for each, predict values,
//     pick placement leading to highest-value state
//
// This is fundamentally different from typical DQN:
//   - Typical DQN: Q(state, action) -> value for each action
//   - This approach: V(resulting_state) -> single value
//
// ============================================================================

/// State features for the DQN - Raw board state (most general approach)
/// Uses the actual board cells as input, letting the network learn what matters
#[derive(Debug, Clone, Copy)]
pub struct BoardState {
    /// Raw board state: 200 binary values (20 rows × 10 cols)
    pub board: [u8; TetrisBoard::SIZE],
    /// Lines cleared by this move (reward signal)
    pub lines_cleared: u32,
}

impl BoardState {
    /// 200 board cells + 1 lines_cleared = 201 features
    pub const NUM_FEATURES: usize = TetrisBoard::SIZE;

    /// Extract board state features from a TetrisBoard
    pub fn from_board(board: &TetrisBoard, lines_cleared: u32) -> Self {
        Self {
            board: board.to_binary_slice(),
            lines_cleared,
        }
    }

    /// Convert to tensor for neural network input
    pub fn to_vec(&self) -> [f32; Self::NUM_FEATURES] {
        let mut features = [0.0f32; Self::NUM_FEATURES];
        for (i, &cell) in self.board.iter().enumerate() {
            features[i] = cell as f32;
        }
        features
    }
}

// ============================================================================
// Convolutional Value Network with MLP Head
// ============================================================================
//
// Architecture:
//   Conv Backbone: [B, 1, 20, 10] -> Conv blocks -> [B, C, 20, 10] -> flatten
//   MLP Head: flatten_dim -> hidden layers -> 1 (value)
//
// Conv blocks: uses ConvBlock from modules.rs (Conv2d -> GroupNorm -> SiLU)
// MLP head: Vec<Linear> layers with ReLU + Dropout (flexible architecture)
//
// ============================================================================

/// Convolutional Value Network
/// Conv backbone (spatial feature extraction) + MLP head (value prediction)
#[derive(Debug, Clone)]
pub struct ConvValueNetwork {
    conv_blocks: Vec<ConvBlock>,
    head_layers: Vec<Linear>,
    dropout: Dropout,
    #[allow(dead_code)]
    flatten_dim: usize,
}

impl ConvValueNetwork {
    /// Initialize network with conv block specs and MLP head layer sizes
    ///
    /// Example:
    ///   conv_specs: 4 blocks [1->32->32->16->8]
    ///   head_sizes: [1600, 128, 64, 1] (flatten_dim -> hidden -> ... -> 1)
    pub fn init(
        vb: &VarBuilder,
        conv_specs: &[ConvBlockSpec],
        head_sizes: &[usize],
    ) -> Result<Self> {
        // Build conv blocks using ConvBlock from modules
        let mut conv_blocks = Vec::with_capacity(conv_specs.len());
        for (i, spec) in conv_specs.iter().enumerate() {
            let (block, _out_ch) = ConvBlock::init(vb, spec, i)?;
            conv_blocks.push(block);
        }

        // Calculate flatten dim: last_channels * height * width
        let last_channels = conv_specs.last().map(|s| s.out_channels).unwrap_or(1);
        let flatten_dim = last_channels * TetrisBoard::HEIGHT * TetrisBoard::WIDTH;

        // Validate head_sizes[0] matches flatten_dim
        assert_eq!(
            head_sizes[0], flatten_dim,
            "head_sizes[0] ({}) must equal flatten_dim ({})",
            head_sizes[0], flatten_dim
        );

        // Build MLP head layers
        let mut head_layers = Vec::new();
        for i in 0..head_sizes.len() - 1 {
            let layer = linear(
                head_sizes[i],
                head_sizes[i + 1],
                vb.pp(format!("head_{}", i)),
            )?;
            head_layers.push(layer);
        }

        Ok(Self {
            conv_blocks,
            head_layers,
            dropout: Dropout::new(0.01),
            flatten_dim,
        })
    }

    /// Forward pass
    /// Input: [B, 1, H, W] board tensor
    /// Output: [B, 1] value
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // Conv backbone
        let mut y = x.clone();
        for block in &self.conv_blocks {
            y = block.forward(&y)?;
        }

        // Flatten: [B, C, H, W] -> [B, C*H*W]
        y = y.flatten_from(1)?;

        // MLP head
        for (i, layer) in self.head_layers.iter().enumerate() {
            y = layer.forward(&y)?;
            // Apply ReLU and dropout to all layers except the last
            if i < self.head_layers.len() - 1 {
                y = y.relu()?;
                y = self.dropout.forward(&y, train)?;
            }
        }

        Ok(y)
    }

    /// Predict values for a batch of board states
    /// Input: slice of BoardState
    /// Output: Vec<f32> of values
    pub fn predict_values(
        &self,
        states: &[BoardState],
        device: &candle_core::Device,
    ) -> Result<Vec<f32>> {
        if states.is_empty() {
            return Ok(vec![]);
        }
        let dtype = crate::fdtype();
        let batch_size = states.len();

        // Convert boards to [B, 1, H, W] tensor
        let board_data: Vec<f32> = states
            .iter()
            .flat_map(|s| s.board.iter().map(|&b| b as f32))
            .collect();

        let tensor = Tensor::from_vec(
            board_data,
            (batch_size, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
            device,
        )?
        .to_dtype(dtype)?;

        let values = self.forward(&tensor, false)?; // inference mode
        Ok(values.squeeze(1)?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }
}

// ============================================================================
// Experience Replay Buffer
// ============================================================================

/// Experience tuple for Q-learning
/// Unlike typical DQN, we store state features (4 floats) not raw boards
#[derive(Debug, Clone)]
struct Experience {
    /// Board features before the move
    current_state: BoardState,
    /// Board features after the move (the resulting state we chose)
    next_state: BoardState,
    /// Reward for this transition
    reward: f32,
    /// Whether the game ended
    done: bool,
    /// Episode number when this experience was collected
    episode: u32,
}

/// Priority Experience Replay Buffer
/// Samples experiences proportionally to their TD-error priority
/// Uses probabilistic eviction based on age AND reward:
/// - Older experiences are more likely to be evicted
/// - High-reward experiences are less likely to be evicted
/// - But eventually all old experiences will be evicted
struct PriorityReplayBuffer {
    buffer: VecDeque<Experience>,
    priorities: VecDeque<f32>,
    capacity: usize,
    alpha: f32,            // Priority exponent (0 = uniform, 1 = full prioritization)
    beta: f32,             // Importance sampling exponent (anneals to 1.0)
    epsilon: f32,          // Small constant to avoid zero priority
    max_age_episodes: u32, // Max age before guaranteed eviction
    max_reward: f32,       // Expected max reward (for normalizing survival bonus)
}

impl PriorityReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            alpha: 0.6,            // Standard PER value
            beta: 0.4,             // Will anneal to 1.0 during training
            epsilon: 1e-6,         // Avoid zero priority
            max_age_episodes: 100, // Max age before guaranteed eviction
            max_reward: 50.0,      // Normalize rewards (Tetris ≈ 41)
        }
    }

    /// Compute eviction probability based on age and reward
    /// High age → high eviction probability
    /// High reward → lower eviction probability (but still possible)
    fn eviction_probability(&self, exp: &Experience, current_episode: u32) -> f32 {
        let age = current_episode.saturating_sub(exp.episode) as f32;
        let max_age = self.max_age_episodes as f32;

        // Base probability increases with age (0.0 to 1.0)
        let age_factor = (age / max_age).min(1.0);

        // Reward bonus reduces eviction prob (cap at 50% reduction)
        // Higher reward = lower eviction probability
        let reward_bonus = (exp.reward / self.max_reward).clamp(0.0, 0.5);

        // Final eviction probability
        // Even high-reward experiences have at least 50% eviction prob when old
        age_factor * (1.0 - reward_bonus)
    }

    fn push(&mut self, exp: Experience, rng: &mut impl Rng) {
        let current_episode = exp.episode;

        // New experiences get max priority to ensure they're sampled at least once
        let max_priority = self.priorities.iter().cloned().fold(1.0f32, f32::max);

        // Always push the new experience first
        self.buffer.push_back(exp);
        self.priorities.push_back(max_priority);

        // Probabilistic eviction from front of buffer
        // Check oldest experiences and probabilistically evict based on age + reward
        let mut evicted = 0;
        let max_evictions = 10; // Limit evictions per push for performance

        while evicted < max_evictions && !self.buffer.is_empty() {
            let evict_prob = {
                let oldest = self.buffer.front().unwrap();
                self.eviction_probability(oldest, current_episode)
            };

            // Guaranteed eviction if over max age, otherwise probabilistic
            if evict_prob >= 1.0 || rng.random::<f32>() < evict_prob {
                self.buffer.pop_front();
                self.priorities.pop_front();
                evicted += 1;
            } else {
                break; // Front survived, stop checking
            }
        }

        // Also respect hard capacity as safety limit
        while self.buffer.len() > self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }
    }

    /// Sample batch with priority-based probabilities
    /// Returns: (experiences, indices, importance_weights)
    fn sample(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Option<(Vec<Experience>, Vec<usize>, Vec<f32>)> {
        if self.buffer.len() < batch_size {
            return None;
        }

        // Compute sampling probabilities: P(i) = priority_i^alpha / sum(priority^alpha)
        let priorities_alpha: Vec<f32> = self
            .priorities
            .iter()
            .map(|p| (p + self.epsilon).powf(self.alpha))
            .collect();
        let sum_priorities: f32 = priorities_alpha.iter().sum();
        let probabilities: Vec<f32> = priorities_alpha
            .iter()
            .map(|p| p / sum_priorities)
            .collect();

        // Sample indices based on probabilities
        let mut indices = Vec::with_capacity(batch_size);
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let r: f32 = rng.random();
            let mut cumsum = 0.0;
            let mut idx = 0;
            for (i, &prob) in probabilities.iter().enumerate() {
                cumsum += prob;
                if r <= cumsum {
                    idx = i;
                    break;
                }
            }
            indices.push(idx);
            samples.push(self.buffer[idx].clone());
        }

        // Compute importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        let n = self.buffer.len() as f32;
        let weights: Vec<f32> = indices
            .iter()
            .map(|&i| (n * probabilities[i]).powf(-self.beta))
            .collect();
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / max_weight).collect();

        Some((samples, indices, normalized_weights))
    }

    /// Update priorities for sampled experiences based on TD-errors
    fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
            if idx < self.priorities.len() {
                self.priorities[idx] = td_error.abs() + self.epsilon;
            }
        }
    }

    /// Anneal beta towards 1.0 (call each episode)
    fn anneal_beta(&mut self, current_episode: u32, total_episodes: u32) {
        // Linear annealing from initial beta to 1.0
        let progress = (current_episode as f32) / (total_episodes as f32).max(1.0);
        self.beta = 0.4 + progress * (1.0 - 0.4);
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn is_ready(&self, min_size: usize) -> bool {
        self.buffer.len() >= min_size
    }
}

// ============================================================================
// Beam Search for N-Ply Lookahead (Parallel + Batched)
// ============================================================================
//
// Optimizations:
// 1. Parallel board simulations using Rayon
// 2. Batched network inference (one call per depth instead of per-state)
//
// ============================================================================

/// Intermediate result from parallel simulation (before network evaluation)
struct SimulationResult {
    board: TetrisBoard,
    first_placement: TetrisPiecePlacement,
    first_state: BoardState,
    first_reward: f32,
    lines_cleared: u32,
    eval_state: BoardState, // State to evaluate with network
}

/// A candidate in the beam search (after network evaluation)
#[derive(Clone)]
struct BeamCandidate {
    board: TetrisBoard,
    first_placement: TetrisPiecePlacement,
    first_state: BoardState,
    first_reward: f32,
    total_lines: u32,
    value: f32,
}

/// Compute reward for a placement
fn compute_reward(lines_cleared: u32, is_lost: bool) -> f32 {
    let base_reward = 1.0 + (lines_cleared as f32).powi(2) * TetrisBoard::WIDTH as f32;
    if is_lost {
        base_reward - 2.0
    } else {
        base_reward
    }
}

/// Sort helper for beam candidates (highest value first, NaN treated as worst)
fn compare_values(a: f32, b: f32) -> std::cmp::Ordering {
    let a_safe = if a.is_nan() { f32::NEG_INFINITY } else { a };
    let b_safe = if b.is_nan() { f32::NEG_INFINITY } else { b };
    b_safe
        .partial_cmp(&a_safe)
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Beam search with parallel simulation, batched inference, and softmax selection
/// Uses softmax sampling on final beam candidates instead of argmax
/// Temperature controls exploration: high = more random, low = more greedy
/// Returns: (selected_placement, resulting_state, reward)
fn beam_search(
    game: &TetrisGame,
    beam_width: usize,
    max_depth: usize,
    network: &ConvValueNetwork,
    device: &candle_core::Device,
    temperature: f32,
    rng: &mut impl Rng,
) -> Result<(TetrisPiecePlacement, BoardState, f32)> {
    // === Depth 0: Parallel simulation ===
    let placements: Vec<_> = TetrisPiecePlacement::all_from_piece(game.current_piece).to_vec();
    let game_board = game.board;

    let depth0_results: Vec<SimulationResult> = placements
        .par_iter()
        .filter_map(|&placement| {
            let mut board = game_board;
            let result = board.apply_piece_placement(placement);
            if result.is_lost.into() {
                return None;
            }
            let state = BoardState::from_board(&board, result.lines_cleared);
            let reward = compute_reward(result.lines_cleared, false);
            Some(SimulationResult {
                board,
                first_placement: placement,
                first_state: state,
                first_reward: reward,
                lines_cleared: result.lines_cleared,
                eval_state: state,
            })
        })
        .collect();

    if depth0_results.is_empty() {
        return Err(anyhow::Error::msg("No valid placements"));
    }

    // === Depth 0: Batched network evaluation ===
    let states: Vec<BoardState> = depth0_results.iter().map(|r| r.eval_state).collect();
    let values = network.predict_values(&states, device)?;

    // Build beam candidates with values
    let mut beam: Vec<BeamCandidate> = depth0_results
        .into_iter()
        .zip(values.into_iter())
        .map(|(r, value)| BeamCandidate {
            board: r.board,
            first_placement: r.first_placement,
            first_state: r.first_state,
            first_reward: r.first_reward,
            total_lines: r.lines_cleared,
            value,
        })
        .collect();

    // Sort and truncate to beam width
    beam.sort_by(|a, b| compare_values(a.value, b.value));
    beam.truncate(beam_width);

    // === Deeper depths ===
    for depth in 1..max_depth {
        let piece = game.peek_nth_next_piece(depth);
        let piece_placements: Vec<_> = TetrisPiecePlacement::all_from_piece(piece).to_vec();

        // Parallel: simulate all (candidate, placement) pairs
        let expansion_results: Vec<SimulationResult> = beam
            .par_iter()
            .flat_map(|candidate| {
                piece_placements
                    .iter()
                    .filter_map(|&placement| {
                        let mut board = candidate.board;
                        let result = board.apply_piece_placement(placement);
                        if result.is_lost.into() {
                            return None;
                        }
                        let lines = result.lines_cleared;
                        // Use only this move's lines for evaluation (consistent with training)
                        let eval_state = BoardState::from_board(&board, lines);
                        Some(SimulationResult {
                            board,
                            first_placement: candidate.first_placement,
                            first_state: candidate.first_state,
                            first_reward: candidate.first_reward,
                            lines_cleared: candidate.total_lines + lines,
                            eval_state,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        if expansion_results.is_empty() {
            break; // No valid moves at this depth, use current beam
        }

        // Batched: evaluate all expanded states in one network call
        let states: Vec<BoardState> = expansion_results.iter().map(|r| r.eval_state).collect();
        let values = network.predict_values(&states, device)?;

        // Build next beam with values
        let mut next_beam: Vec<BeamCandidate> = expansion_results
            .into_iter()
            .zip(values.into_iter())
            .map(|(r, value)| BeamCandidate {
                board: r.board,
                first_placement: r.first_placement,
                first_state: r.first_state,
                first_reward: r.first_reward,
                total_lines: r.lines_cleared,
                value,
            })
            .collect();

        // Sort and truncate
        next_beam.sort_by(|a, b| compare_values(a.value, b.value));
        next_beam.truncate(beam_width);
        beam = next_beam;
    }

    // Softmax sample from the final beam candidates
    // This explores among top candidates proportional to their values
    if beam.is_empty() {
        return Err(anyhow::Error::msg("No valid placements found"));
    }

    // Extract values and compute softmax probabilities
    let values: Vec<f32> = beam.iter().map(|c| c.value).collect();
    let max_val = values
        .iter()
        .cloned()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp((value - max) / temperature) for numerical stability
    let exp_vals: Vec<f32> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                0.0 // NaN values get zero probability
            } else {
                ((v - max_val) / temperature.max(0.01)).exp() // Clamp temp to avoid division issues
            }
        })
        .collect();

    let sum_exp: f32 = exp_vals.iter().sum();

    // Sample from softmax distribution
    if sum_exp > 0.0 {
        let r: f32 = rng.random();
        let mut cumsum = 0.0;
        for (i, &exp_val) in exp_vals.iter().enumerate() {
            cumsum += exp_val / sum_exp;
            if r <= cumsum {
                let c = &beam[i];
                return Ok((c.first_placement, c.first_state, c.first_reward));
            }
        }
    }

    // Fallback to best candidate if something goes wrong
    let c = &beam[0];
    Ok((c.first_placement, c.first_state, c.first_reward))
}

// ============================================================================
// DQN Training Loop
// ============================================================================

pub fn train_q_learning_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume_from_checkpoint: bool,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    // Hyperparameters
    const NUM_EPISODES: u32 = 10_000_000;
    const MEM_SIZE: usize = 500_000;
    const MIN_REPLAY_SIZE: usize = 10_000;
    const BATCH_SIZE: usize = 1024;
    const EPOCHS_PER_TRAIN: usize = 1;
    const DISCOUNT: f32 = 0.99;
    const LEARNING_RATE: f64 = 0.0001;
    const LOG_EVERY: u32 = 10;
    const CHECKPOINT_INTERVAL: usize = 100;

    // Beam search parameters
    const BEAM_WIDTH: usize = 5;
    const BEAM_DEPTH: usize = 3;

    // Softmax temperature for exploration (decays over training)
    // High temp = more exploration, low temp = more greedy
    const TEMP_START: f32 = 2.0;
    const TEMP_END: f32 = 0.1;
    const TEMP_DECAY_EPISODES: u32 = 50_000;

    // Conv backbone: 1 -> 32 -> 32 -> 16 -> 8 channels (same as tetris_exceed_the_mean.rs)
    let conv_specs = vec![
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
    ];

    // MLP head: flatten_dim -> hidden layers -> 1
    // flatten_dim = 8 channels * 20 height * 10 width = 1600
    let flatten_dim = 8 * TetrisBoard::HEIGHT * TetrisBoard::WIDTH;
    let head_sizes = [flatten_dim, 128, 64, 32, 1];

    // Initialize network
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let network = ConvValueNetwork::init(&vb, &conv_specs, &head_sizes)?;

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
    let mut starting_episode = 0;
    if resume_from_checkpoint {
        if let Some(ref cp) = checkpointer {
            match cp.load_latest_checkpoint(&mut varmap) {
                Ok(Some(iteration)) => {
                    starting_episode = iteration as u32;
                    println!("✅ Loaded checkpoint from episode {}", iteration);
                }
                Ok(None) => println!("ℹ️  No checkpoint found, starting fresh"),
                Err(e) => println!("⚠️  Failed to load checkpoint: {}", e),
            }
        }
    }

    let mut replay_buffer = PriorityReplayBuffer::new(MEM_SIZE);
    let mut rng = rand::rng();

    // Temperature decay calculation (linear decay from TEMP_START to TEMP_END)
    let temp_decay = if TEMP_DECAY_EPISODES > 0 {
        (TEMP_START - TEMP_END) / TEMP_DECAY_EPISODES as f32
    } else {
        0.0
    };

    // Statistics
    let mut scores: Vec<u32> = Vec::new();
    let mut best_score: u32 = 0;
    let total_pieces = AtomicU64::new(0);
    let total_games = AtomicU32::new(0);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     State-Value DQN + Beam Search (Raw Board Input)          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ State Features:                                              ║");
    println!("║   • Raw board (200 cells) + lines_cleared = 201 features     ║");
    println!("║ Search:                                                      ║");
    println!(
        "║   • Beam search: width={}, depth={} plies                     ║",
        BEAM_WIDTH, BEAM_DEPTH
    );
    println!("║ Training:                                                    ║");
    println!(
        "║   • Discount: {}, Temp decay: {} episodes              ║",
        DISCOUNT, TEMP_DECAY_EPISODES
    );
    println!(
        "║   • Replay buffer: {}, Batch: {}                       ║",
        MEM_SIZE, BATCH_SIZE
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    for episode in starting_episode..NUM_EPISODES {
        // Calculate current temperature (decays over time for less exploration)
        let temperature = if episode < TEMP_DECAY_EPISODES {
            TEMP_START - temp_decay * episode as f32
        } else {
            TEMP_END
        };

        // Create new game
        let mut game = TetrisGame::new();
        let mut episode_score: u32 = 0;
        let mut steps: u32 = 0;

        // Get initial state features (before any move)
        let mut current_state = BoardState::from_board(&game.board, 0);

        // Play one episode using beam search
        while !game.board.is_lost() {
            // Use beam search with softmax sampling (looks BEAM_DEPTH pieces ahead)
            let search_result = beam_search(
                &game,
                BEAM_WIDTH,
                BEAM_DEPTH,
                &network,
                &device,
                temperature,
                &mut rng,
            );

            let (placement, next_state, reward) = match search_result {
                Ok(result) => result,
                Err(_) => break, // No valid placements
            };

            // Apply the chosen placement
            let is_lost = game.apply_placement(placement);
            let done = is_lost.into();

            // Store experience: (current_state, next_state, reward, done, episode)
            replay_buffer.push(
                Experience {
                    current_state,
                    next_state,
                    reward,
                    done,
                    episode,
                },
                &mut rng,
            );

            // Update score
            episode_score += reward.max(0.0) as u32;
            steps += 1;

            // Move to next state
            current_state = next_state;

            if done {
                break;
            }
        }

        scores.push(episode_score);
        total_pieces.fetch_add(steps as u64, Ordering::Relaxed);
        total_games.fetch_add(1, Ordering::Relaxed);

        // Train the network with Priority Experience Replay
        if replay_buffer.is_ready(MIN_REPLAY_SIZE) {
            for _ in 0..EPOCHS_PER_TRAIN {
                if let Some((batch, indices, weights)) = replay_buffer.sample(BATCH_SIZE, &mut rng)
                {
                    // Prepare batch tensors - reshape for conv network [B, 1, H, W]
                    let current_boards: Vec<f32> = batch
                        .iter()
                        .flat_map(|e| e.current_state.board.iter().map(|&b| b as f32))
                        .collect();
                    let next_boards: Vec<f32> = batch
                        .iter()
                        .flat_map(|e| e.next_state.board.iter().map(|&b| b as f32))
                        .collect();
                    let rewards: Vec<f32> = batch.iter().map(|e| e.reward).collect();
                    let dones: Vec<f32> = batch
                        .iter()
                        .map(|e| if e.done { 1.0 } else { 0.0 })
                        .collect();

                    let batch_len = batch.len();
                    let current_t = Tensor::from_vec(
                        current_boards,
                        (batch_len, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let next_t = Tensor::from_vec(
                        next_boards,
                        (batch_len, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let reward_t =
                        Tensor::from_vec(rewards, Shape::from_dims(&[batch_len]), &device)?
                            .to_dtype(dtype)?;
                    let done_t = Tensor::from_vec(dones, Shape::from_dims(&[batch_len]), &device)?
                        .to_dtype(dtype)?;

                    // Predict next state values (for Q-learning target)
                    // IMPORTANT: detach from gradient computation - we don't backprop through targets
                    let next_values = network.forward(&next_t, false)?.squeeze(1)?.detach();

                    // Compute targets: reward + discount * V(next_state) * (1 - done)
                    let not_done = (Tensor::ones_like(&done_t)? - &done_t)?;
                    let targets =
                        (&reward_t + (next_values * not_done)?.affine(DISCOUNT as f64, 0.0)?)?;

                    // Clamp targets to prevent extreme values
                    let targets = targets.clamp(-1000.0, 10000.0)?;

                    // Predict current state values (training mode with dropout)
                    let predictions = network.forward(&current_t, true)?.squeeze(1)?;

                    // Compute TD-errors for priority updates
                    let td_errors = (&predictions - &targets)?;
                    let td_errors_vec: Vec<f32> = td_errors.to_dtype(DType::F32)?.to_vec1()?;

                    // Weighted MSE Loss (importance sampling correction for PER)
                    let weights_t =
                        Tensor::from_vec(weights, Shape::from_dims(&[batch_len]), &device)?
                            .to_dtype(dtype)?;
                    let weighted_loss = (td_errors.sqr()? * weights_t)?.mean_all()?;

                    // Check for NaN loss and skip if detected
                    let loss_val = weighted_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                    if loss_val.is_nan() || loss_val.is_infinite() {
                        continue; // Skip this batch if loss is invalid
                    }

                    // Backprop
                    let grads = weighted_loss.backward()?;
                    optimizer.step(&grads)?;

                    // Update priorities with absolute TD-errors
                    replay_buffer.update_priorities(&indices, &td_errors_vec);
                }
            }
        }

        // Anneal beta towards 1.0 for importance sampling correction
        replay_buffer.anneal_beta(episode, TEMP_DECAY_EPISODES);

        // Logging
        if episode > 0 && episode % LOG_EVERY == 0 {
            let recent_scores: Vec<u32> = scores
                .iter()
                .rev()
                .take(LOG_EVERY as usize)
                .copied()
                .collect();
            let avg_score: f32 =
                recent_scores.iter().sum::<u32>() as f32 / recent_scores.len() as f32;
            let min_score = recent_scores.iter().min().copied().unwrap_or(0);
            let max_score = recent_scores.iter().max().copied().unwrap_or(0);

            println!(
                "Episode {}: τ={:.2}, Avg={:.1}, Min={}, Max={}, Buffer={}",
                episode,
                temperature,
                avg_score,
                min_score,
                max_score,
                replay_buffer.len()
            );

            if let Some(ref mut writer) = summary_writer {
                writer.add_scalar("score/avg", avg_score, episode as usize);
                writer.add_scalar("score/min", min_score as f32, episode as usize);
                writer.add_scalar("score/max", max_score as f32, episode as usize);
                writer.add_scalar("metrics/temperature", temperature, episode as usize);
                writer.add_scalar(
                    "metrics/buffer_size",
                    replay_buffer.len() as f32,
                    episode as usize,
                );
            }
        }

        // Save best model
        if episode_score > best_score {
            best_score = episode_score;
            println!("   🏆 New best score: {} (episode {})", best_score, episode);
            if let Some(ref cp) = checkpointer {
                let _ = cp.checkpoint_item(episode as usize, &varmap, Some("best"), None);
            }
        }

        // Periodic checkpoint
        if let Some(ref cp) = checkpointer {
            if episode as usize % CHECKPOINT_INTERVAL == 0 && episode > 0 {
                let _ = cp.checkpoint_item(episode as usize, &varmap, None, None);
            }
        }
    }

    println!("\n✅ Training complete!");
    println!(
        "   Total games: {}, Total pieces: {}",
        total_games.load(Ordering::Relaxed),
        total_pieces.load(Ordering::Relaxed)
    );
    println!("   Best score: {}", best_score);

    Ok(())
}
