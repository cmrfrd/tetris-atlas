use clap::Parser;
use std::str::FromStr;
use tetris_ml::set_global_threadpool;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::Result;
use candle_core::{DType, Shape, Tensor};
use candle_nn::{
    AdamW, Dropout, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    embedding, linear,
};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;

use tetris_game::{TetrisBoard, TetrisGame, TetrisPiecePlacement};
use tetris_ml::checkpointer::Checkpointer;
use tetris_ml::modules::{
    AttnMlp, AttnMlpConfig, CausalSelfAttentionConfig, ConvBlock, ConvBlockSpec, DynamicTanhConfig,
    Mlp, MlpConfig, TransformerBlockConfig, TransformerBody, TransformerBodyConfig,
};
use tetris_ml::fdtype;

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
        let dtype = tetris_ml::fdtype();
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
// Transformer Value Network
// ============================================================================
//
// Architecture:
//   Token + Position Embeddings: [B, 200] -> [B, 200, D]
//   Transformer Body: [B, 200, D] -> [B, 200, D]
//   Mean Pooling: [B, 200, D] -> [B, D]
//   Value Head MLP: [B, D] -> [B, 1]
//
// This treats the board as a sequence of 200 tokens (cells),
// uses self-attention to model cell relationships, and outputs a single value.
//
// ============================================================================

/// Configuration for Transformer Value Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerValueNetworkConfig {
    pub d_model: usize,
    pub num_blocks: usize,
    pub attn_config: CausalSelfAttentionConfig,
    pub block_mlp_config: AttnMlpConfig,
    pub value_head_config: AttnMlpConfig,
    pub with_causal_mask: bool,
}

impl TransformerValueNetworkConfig {
    /// Create a default config with the given model dimension and number of blocks
    pub fn default_with_dim(d_model: usize, num_blocks: usize) -> Self {
        // Ensure d_model is divisible by n_attention_heads
        let n_heads = if d_model >= 8 && d_model % 8 == 0 {
            8
        } else if d_model >= 4 && d_model % 4 == 0 {
            4
        } else {
            1
        };

        Self {
            d_model,
            num_blocks,
            attn_config: CausalSelfAttentionConfig {
                d_model,
                n_attention_heads: n_heads,
                n_kv_heads: n_heads,
                rope_theta: 10_000.0,
                max_position_embeddings: TetrisBoard::SIZE,
            },
            block_mlp_config: AttnMlpConfig {
                hidden_size: d_model,
                intermediate_size: 2 * d_model,
                output_size: d_model,
                dropout: None,
            },
            value_head_config: AttnMlpConfig {
                hidden_size: d_model,
                intermediate_size: 2 * d_model,
                output_size: 1,
                dropout: None,
            },
            with_causal_mask: false, // Bidirectional attention for value estimation
        }
    }
}

/// Transformer-based Value Network
/// Treats the Tetris board as a sequence of 200 tokens
#[derive(Debug, Clone)]
pub struct TransformerValueNetwork {
    // Embeddings
    token_embed: Embedding, // [0/1] -> D
    pos_embed: Embedding,   // [0..199] -> D

    // Transformer body
    body: TransformerBody,

    // Value head
    value_head: AttnMlp,

    // Config
    #[allow(dead_code)]
    d_model: usize,
    with_causal_mask: bool,
}

impl TransformerValueNetwork {
    pub fn init(vb: &VarBuilder, cfg: &TransformerValueNetworkConfig) -> Result<Self> {
        // Validate config
        assert_eq!(
            cfg.attn_config.d_model, cfg.d_model,
            "attn d_model must equal d_model"
        );
        assert_eq!(
            cfg.block_mlp_config.hidden_size, cfg.d_model,
            "block mlp hidden_size must equal d_model"
        );
        assert_eq!(
            cfg.block_mlp_config.output_size, cfg.d_model,
            "block mlp output_size must equal d_model"
        );
        assert_eq!(
            cfg.value_head_config.hidden_size, cfg.d_model,
            "value head hidden_size must equal d_model"
        );

        // Token embedding: 2 states (empty/filled) -> d_model
        let token_embed = embedding(
            TetrisBoard::NUM_TETRIS_CELL_STATES, // 2
            cfg.d_model,
            vb.pp("token_embed"),
        )?;

        // Positional embedding: 200 positions -> d_model
        let pos_embed = embedding(TetrisBoard::SIZE, cfg.d_model, vb.pp("pos_embed"))?;

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

        // Value head: d_model -> 1
        let value_head = AttnMlp::init(&vb.pp("value_head"), &cfg.value_head_config)?;

        Ok(Self {
            token_embed,
            pos_embed,
            body,
            value_head,
            d_model: cfg.d_model,
            with_causal_mask: cfg.with_causal_mask,
        })
    }

    /// Forward pass
    /// Input: [B, 1, H, W] board tensor (same format as ConvValueNetwork for compatibility)
    /// Output: [B, 1] value
    pub fn forward(&self, x: &Tensor, _train: bool) -> Result<Tensor> {
        let batch_size = x.dim(0)?;

        // Flatten spatial dims: [B, 1, H, W] -> [B, T] where T = H*W = 200
        let tokens = x.flatten(1, 3)?.to_dtype(DType::U32)?; // [B, 200]

        // Token embeddings
        let x_tokens = self.token_embed.forward(&tokens)?; // [B, T, D]

        // Positional embeddings
        let device = tokens.device();
        let pos_ids = Tensor::arange(0, TetrisBoard::SIZE as u32, device)?
            .to_dtype(DType::U32)?
            .reshape(&[1, TetrisBoard::SIZE])?
            .repeat(&[batch_size, 1])?; // [B, T]
        let x_pos = self.pos_embed.forward(&pos_ids)?; // [B, T, D]

        // Combine embeddings
        let x = (&x_tokens + &x_pos)?; // [B, T, D]

        // Transformer body
        let x = self.body.forward(x, self.with_causal_mask)?; // [B, T, D]

        // Per-token value contributions: each cell outputs its contribution to overall value
        let per_token_values = self.value_head.forward(&x)?; // [B, T, 1]

        // Mean over all tokens for stable, unbounded value
        let value = per_token_values.mean(1)?; // [B, 1]

        Ok(value)
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
        let dtype = tetris_ml::fdtype();
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
            alpha: 0.6,             // Standard PER value
            beta: 0.4,              // Will anneal to 1.0 during training
            epsilon: 1e-6,          // Avoid zero priority
            max_age_episodes: 1000, // Max age before guaranteed eviction
            max_reward: 50.0,       // Normalize rewards (Tetris ≈ 41)
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
    network: &TransformerValueNetwork,
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
) -> Result<()> {
    let device = tetris_ml::device();
    let dtype = fdtype();

    // Hyperparameters
    const NUM_EPISODES: u32 = 10_000_000;
    const MEM_SIZE: usize = 500_000;
    const MIN_REPLAY_SIZE: usize = 1_000;
    const BATCH_SIZE: usize = 64;
    const EPOCHS_PER_TRAIN: usize = 1;
    const DISCOUNT: f32 = 0.95;
    const LEARNING_RATE: f64 = 0.001;
    const LOG_EVERY: u32 = 10;
    const CHECKPOINT_INTERVAL: usize = 100;

    // Beam search parameters
    const BEAM_WIDTH: usize = 5;
    const BEAM_DEPTH: usize = 3;

    // Softmax temperature for exploration (decays over training)
    // High temp = more exploration, low temp = more greedy
    const TEMP_START: f32 = 2.0;
    const TEMP_END: f32 = 0.1;
    const TEMP_DECAY_EPISODES: u32 = 10_000;

    // Transformer value network configuration
    const D_MODEL: usize = 32; // Must be divisible by n_attention_heads
    const NUM_BLOCKS: usize = 4;

    let transformer_cfg = TransformerValueNetworkConfig {
        d_model: D_MODEL,
        num_blocks: NUM_BLOCKS,
        attn_config: CausalSelfAttentionConfig {
            d_model: D_MODEL,
            n_attention_heads: 8, // D_MODEL must be divisible by this
            n_kv_heads: 8,
            rope_theta: 10_000.0,
            max_position_embeddings: TetrisBoard::SIZE,
        },
        block_mlp_config: AttnMlpConfig {
            hidden_size: D_MODEL,
            intermediate_size: 2 * D_MODEL,
            output_size: D_MODEL,
            dropout: None,
        },
        value_head_config: AttnMlpConfig {
            hidden_size: D_MODEL,
            intermediate_size: 2 * D_MODEL,
            output_size: 1,
            dropout: None,
        },
        with_causal_mask: false, // Bidirectional for value estimation
    };

    // Initialize transformer network
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let network = TransformerValueNetwork::init(&vb, &transformer_cfg)?;

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
    println!("║     State-Value DQN + Beam Search (Transformer Model)        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Architecture:                                                ║");
    println!(
        "║   • Transformer: d_model={}, {} blocks, 8 heads             ║",
        D_MODEL, NUM_BLOCKS
    );
    println!("║   • Bidirectional attention + mean pooling → value          ║");
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

    for episode in 0..NUM_EPISODES {
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
            let result = game.apply_placement(placement);
            let done = result.is_lost.into();

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
    info!("Starting tetris Q-learning transformer training");
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

    train_q_learning_policy(run_name, logdir, checkpoint_dir).unwrap();
}
