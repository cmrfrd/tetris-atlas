use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Shape, Tensor};
use candle_nn::{
    AdamW, Dropout, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear,
};
use clap::Parser;
use rand::Rng;
use std::str::FromStr;
use tensorboard::summary_writer::SummaryWriter;
use tetris_game::{TetrisBoard, TetrisGame};
use tetris_ml::{
    checkpointer::Checkpointer,
    device,
    modules::{ConvBlock, ConvBlockSpec, Conv2dConfig},
    set_global_threadpool,
};
use tetris_tensor::fdtype;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

// ============================================================================
// Deep Q Network (DQN) - Viet Nguyen style (state-value for resulting states)
// ============================================================================
//
// Matches the Python reference behavior:
// - At each step, enumerate all placements for the current piece
// - Compute the resulting next state for each placement
// - Pick a placement via epsilon-greedy over V(next_state)
// - Step env, store (state, reward, next_state, done) in replay buffer
// - Train with TD(0): y = r if done else r + gamma * V(next_state)
//
// Notes:
// - This is effectively learning a state-value function V(s), not Q(s,a).
// - The policy uses one-step lookahead: choose action leading to best predicted V(s').
//
// ============================================================================

// ----------------------------
// Hyperparameters (constants)
// ----------------------------
const BATCH_SIZE: usize = 512;
const LEARNING_RATE: f64 = 1e-3;
const DISCOUNT: f32 = 0.99;

const INITIAL_EPSILON: f32 = 1.0;
const FINAL_EPSILON: f32 = 1e-3;
const NUM_DECAY_EPOCHS: u32 = 2000;

const NUM_EPOCHS: u32 = 1_000_000;
const SAVE_INTERVAL: u32 = 1000;

const REPLAY_MEMORY_SIZE: usize = 100_000;
const MIN_REPLAY_WARMUP: usize = REPLAY_MEMORY_SIZE / 100;

const DROPOUT_P: f32 = 0.01;

// Reward shaping (keep simple by default).
// Python env returns its own reward; here we approximate with lines cleared.
const LOSS_PENALTY: f32 = 0.0; // set e.g. -1.0 if you want a death penalty

// ============================================================================
// State representation
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct DqnState {
    board: [u8; TetrisBoard::SIZE],
}

impl DqnState {
    #[inline]
    fn from_game(game: &TetrisGame) -> Self {
        Self {
            board: game.board.to_binary_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Transition {
    state: DqnState,
    reward: f32,
    next_state: DqnState,
    done: bool,
}

// ============================================================================
// Conv value network: [B,1,H,W] -> [B,1]
// ============================================================================

#[derive(Debug, Clone)]
struct ConvValueNetwork {
    conv_blocks: Vec<ConvBlock>,
    head_layers: Vec<Linear>,
    dropout: Dropout,
}

impl ConvValueNetwork {
    fn init(vb: &VarBuilder, conv_specs: &[ConvBlockSpec], head_sizes: &[usize]) -> Result<Self> {
        let mut conv_blocks = Vec::with_capacity(conv_specs.len());
        for (i, spec) in conv_specs.iter().enumerate() {
            let (block, _out_ch) = ConvBlock::init(vb, spec, i)?;
            conv_blocks.push(block);
        }

        let last_channels = conv_specs.last().map(|s| s.out_channels).unwrap_or(1);
        let flatten_dim = last_channels * TetrisBoard::HEIGHT * TetrisBoard::WIDTH;
        anyhow::ensure!(
            !head_sizes.is_empty() && head_sizes[0] == flatten_dim,
            "head_sizes[0] must equal flatten_dim (expected {}, got {:?})",
            flatten_dim,
            head_sizes.get(0).copied()
        );

        let mut head_layers = Vec::new();
        for i in 0..head_sizes.len() - 1 {
            head_layers.push(linear(
                head_sizes[i],
                head_sizes[i + 1],
                vb.pp(format!("head_{i}")),
            )?);
        }

        Ok(Self {
            conv_blocks,
            head_layers,
            dropout: Dropout::new(DROPOUT_P),
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut y = x.clone();
        for block in &self.conv_blocks {
            y = block.forward(&y)?;
        }
        y = y.flatten_from(1)?;

        for (i, layer) in self.head_layers.iter().enumerate() {
            y = layer.forward(&y)?;
            if i < self.head_layers.len() - 1 {
                y = y.relu()?;
                y = self.dropout.forward(&y, train)?;
            }
        }
        Ok(y)
    }
}

// ============================================================================
// Helpers: tensor conversion
// ============================================================================

#[inline]
fn states_to_board_tensor(states: &[DqnState], device: &candle_core::Device) -> Result<Tensor> {
    let dtype = fdtype();
    let b = states.len();
    let board_data: Vec<f32> = states
        .iter()
        .flat_map(|s| s.board.iter().map(|&v| v as f32))
        .collect();
    Ok(Tensor::from_vec(
        board_data,
        (b, 1, TetrisBoard::HEIGHT, TetrisBoard::WIDTH),
        device,
    )?
    .to_dtype(dtype)?)
}

// ============================================================================
// Training loop
// ============================================================================

pub fn train_tetris_dqn(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume_from_checkpoint: bool,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    // Network architecture (conv backbone + MLP head).
    // Similar to other models in this repo (policy gradients / world model).
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
            gn_groups: 8,
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
            gn_groups: 8,
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
            gn_groups: 8,
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
    let flatten_dim =
        conv_specs.last().unwrap().out_channels * TetrisBoard::HEIGHT * TetrisBoard::WIDTH;
    let head_sizes = vec![flatten_dim, 256, 64, 1];

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let network = ConvValueNetwork::init(&vb.pp("dqn"), &conv_specs, &head_sizes)?;

    let model_params = varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params,
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..Default::default()
        },
    )?;

    let mut summary_writer = logdir.map(SummaryWriter::new);
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let _ = std::fs::create_dir_all(dir);
        Checkpointer::new(SAVE_INTERVAL as usize, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    // Load checkpoint if requested.
    let mut starting_epoch: u32 = 0;
    if resume_from_checkpoint {
        if let Some(ref cp) = checkpointer {
            match cp.load_latest_checkpoint(&mut varmap) {
                Ok(Some(iteration)) => {
                    starting_epoch = iteration as u32;
                    println!("Loaded checkpoint from epoch {}", iteration);
                }
                Ok(None) => println!("No checkpoint found, starting fresh"),
                Err(e) => println!("Failed to load checkpoint: {}", e),
            }
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   DQN (state-value) + ConvNet (Viet Nguyen style)           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Device: {:?}                                                 ║",
        device
    );
    println!("║ Input: raw board [1,20,10]                                  ║");
    println!("║ Output: V(state) scalar                                      ║");
    println!(
        "║ Hyperparams: batch={}, lr={}, γ={}, ε: {}→{} over {} epochs ║",
        BATCH_SIZE, LEARNING_RATE, DISCOUNT, INITIAL_EPSILON, FINAL_EPSILON, NUM_DECAY_EPOCHS
    );
    println!(
        "║ Replay: size={}, warmup={}                                  ║",
        REPLAY_MEMORY_SIZE, MIN_REPLAY_WARMUP
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut rng = rand::rng();
    let mut replay_memory: VecDeque<Transition> = VecDeque::with_capacity(REPLAY_MEMORY_SIZE);

    let mut epoch: u32 = starting_epoch;
    while epoch < NUM_EPOCHS {
        // Reset episode.
        let mut game = TetrisGame::new();
        let mut state = DqnState::from_game(&game);

        // Run one episode until done.
        let mut last_action_idx: u8 = 0;
        loop {
            let placements = game.current_placements();
            if placements.is_empty() {
                // Should be rare; treat as terminal.
                let done = true;
                let next_state = state;
                replay_memory.push_back(Transition {
                    state,
                    reward: LOSS_PENALTY,
                    next_state,
                    done,
                });
                break;
            }

            // Enumerate next states for all placements (one-step lookahead).
            let mut next_states: Vec<DqnState> = Vec::with_capacity(placements.len());
            for &p in placements {
                let mut g2 = game;
                let _ = g2.apply_placement(p);
                next_states.push(DqnState::from_game(&g2));
            }

            // Epsilon schedule (matches the Python reference).
            let epsilon = FINAL_EPSILON
                + ((NUM_DECAY_EPOCHS.saturating_sub(epoch)) as f32)
                    * (INITIAL_EPSILON - FINAL_EPSILON)
                    / (NUM_DECAY_EPOCHS.max(1) as f32);

            // Choose action: random w.p. epsilon else argmax V(next_state).
            let random_action = rng.random::<f32>() <= epsilon;
            let chosen_index: usize = if random_action {
                rng.random_range(0..placements.len())
            } else {
                let next_t = states_to_board_tensor(&next_states, &device)?;
                let values = network
                    .forward(&next_t, false)?
                    .squeeze(1)?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };

            let action = placements[chosen_index];
            last_action_idx = action.index();

            // Step environment.
            let result = game.apply_placement(action);
            let done = bool::from(result.is_lost);
            let reward = if done {
                LOSS_PENALTY
            } else {
                result.lines_cleared as f32
            };

            let next_state = DqnState::from_game(&game);
            replay_memory.push_back(Transition {
                state,
                reward,
                next_state,
                done,
            });
            if replay_memory.len() > REPLAY_MEMORY_SIZE {
                replay_memory.pop_front();
            }

            if done {
                break;
            }

            state = next_state;
        }

        // Wait for warmup before training (matches python behavior).
        if replay_memory.len() < MIN_REPLAY_WARMUP {
            continue;
        }

        // One training step per episode end (matches python).
        epoch += 1;

        let batch_size = BATCH_SIZE.min(replay_memory.len());
        let mut batch: Vec<Transition> = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch.push(replay_memory[rng.random_range(0..replay_memory.len())]);
        }

        let state_batch: Vec<DqnState> = batch.iter().map(|t| t.state).collect();
        let next_state_batch: Vec<DqnState> = batch.iter().map(|t| t.next_state).collect();
        let reward_batch: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let done_batch: Vec<f32> = batch
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();

        let state_t = states_to_board_tensor(&state_batch, &device)?;
        let next_state_t = states_to_board_tensor(&next_state_batch, &device)?;
        let reward_t = Tensor::from_vec(reward_batch, Shape::from_dims(&[batch_size]), &device)?
            .to_dtype(dtype)?;
        let done_t = Tensor::from_vec(done_batch, Shape::from_dims(&[batch_size]), &device)?
            .to_dtype(dtype)?;

        // TD target: y = r + gamma * V(s') * (1 - done)
        let next_v = network.forward(&next_state_t, false)?.squeeze(1)?.detach();
        let not_done = (Tensor::ones_like(&done_t)? - &done_t)?;
        let targets = (&reward_t + (next_v * not_done)?.affine(DISCOUNT as f64, 0.0)?)?;

        // Prediction: V(s)
        let preds = network.forward(&state_t, true)?.squeeze(1)?;

        // MSE loss
        let td = (&preds - &targets)?;
        let loss = td.sqr()?.mean_all()?;

        let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        if !(loss_val.is_nan() || loss_val.is_infinite()) {
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
        }

        // Episode stats (use game's tracked metrics).
        let final_score = game.lines_cleared;
        let final_tetrominoes = game.piece_count;
        let final_cleared_lines = game.lines_cleared;

        println!(
            "Epoch: {}/{}, ActionIdx: {}, Score(lines): {}, Tetrominoes: {}, Cleared lines: {}",
            epoch, NUM_EPOCHS, last_action_idx, final_score, final_tetrominoes, final_cleared_lines
        );

        if let Some(ref mut writer) = summary_writer {
            writer.add_scalar("Train/Score", final_score as f32, (epoch - 1) as usize);
            writer.add_scalar(
                "Train/Tetrominoes",
                final_tetrominoes as f32,
                (epoch - 1) as usize,
            );
            writer.add_scalar(
                "Train/Cleared lines",
                final_cleared_lines as f32,
                (epoch - 1) as usize,
            );
            writer.add_scalar("Train/Loss", loss_val, (epoch - 1) as usize);

            let epsilon = FINAL_EPSILON
                + ((NUM_DECAY_EPOCHS.saturating_sub(epoch)) as f32)
                    * (INITIAL_EPSILON - FINAL_EPSILON)
                    / (NUM_DECAY_EPOCHS.max(1) as f32);
            writer.add_scalar("metrics/epsilon", epsilon, (epoch - 1) as usize);
            writer.add_scalar(
                "metrics/replay_size",
                replay_memory.len() as f32,
                (epoch - 1) as usize,
            );
        }

        // Checkpoints (save_interval like python).
        if epoch > 0 && epoch % SAVE_INTERVAL == 0 {
            if let Some(ref cp) = checkpointer {
                let _ = cp.checkpoint_item(epoch as usize, &varmap, None, None);
            }
        }
    }

    // Final checkpoint
    if let Some(ref cp) = checkpointer {
        let _ = cp.force_checkpoint_item(NUM_EPOCHS as usize, &varmap, None, None);
    }

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
    info!("Starting tetris DQN training");
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

    train_tetris_dqn(run_name, logdir, checkpoint_dir, cli.resume).unwrap();
}
