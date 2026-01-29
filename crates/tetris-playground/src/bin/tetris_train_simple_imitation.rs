use clap::Parser;
use std::str::FromStr;
use tetris_ml::set_global_threadpool;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::Result;
use candle_core::{D, DType, IndexOp, Shape, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_nn::{Embedding, embedding};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;

use tetris_game::{
    TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_ml::checkpointer::Checkpointer;
use tetris_ml::grad_accum::GradientAccumulator;
use tetris_ml::modules::{AttnMlp, AttnMlpConfig, FiLM, FiLMConfig};
use tetris_ml::fdtype;
use tetris_ml::tensors::TetrisPieceOrientationLogitsTensor;
use tetris_ml::tensors::{TetrisPieceOrientationTensor, TetrisPieceTensor};
use tetris_ml::wrapped_tensor::WrappedTensor;

/// Simple policy over placements using supervised learning on high-performing trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisSimpleImitationPolicyConfig {
    pub piece_embedding_dim: usize,

    pub board_encoder_config: Vec<AttnMlpConfig>,

    pub piece_film_config: Vec<FiLMConfig>,

    pub head_mlp_config: AttnMlpConfig, // outputs logits over orientations

    pub value_head_config: AttnMlpConfig, // outputs value [0, 1]
}

#[derive(Debug, Clone)]
pub struct TetrisSimpleImitationPolicy {
    piece_embedding: Embedding,
    board_encoder: Vec<AttnMlp>,
    piece_film: Vec<FiLM>,
    head_mlp: AttnMlp,
    value_head: AttnMlp,
}

impl TetrisSimpleImitationPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisSimpleImitationPolicyConfig) -> Result<Self> {
        let piece_embedding = embedding(
            TetrisPiece::NUM_PIECES,
            cfg.piece_embedding_dim,
            vb.pp("piece_embedding"),
        )?;
        let board_encoder = cfg
            .board_encoder_config
            .iter()
            .enumerate()
            .map(|(i, cfg)| AttnMlp::init(&vb.pp(format!("board_encoder_{}", i)), cfg).unwrap())
            .collect();
        let piece_film = cfg
            .piece_film_config
            .iter()
            .map(|cfg| FiLM::init(&vb.pp("piece_film"), cfg).unwrap())
            .collect();
        let head_mlp = AttnMlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;
        let value_head = AttnMlp::init(&vb.pp("value_head"), &cfg.value_head_config)?;
        Ok(Self {
            piece_embedding,
            board_encoder,
            piece_film,
            head_mlp,
            value_head,
        })
    }

    /// Forward producing unmasked orientation logits [B, NUM_ORIENTATIONS] and value [B, 1]
    /// Takes board features [B, 20] (heights + holes) instead of raw board
    pub fn forward_logits(
        &self,
        board_features: &Tensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<(TetrisPieceOrientationLogitsTensor, Tensor)> {
        let dtype = tetris_ml::fdtype();

        // Board features already in [B, 20] format
        let cur_flat = board_features.to_dtype(dtype)?;

        // Pass through MLP encoder layers [B, D]
        let cur_embed = self
            .board_encoder
            .iter()
            .try_fold(cur_flat, |x, mlp| mlp.forward(&x))?; // [B, D]

        // Condition current on goal and piece
        let piece_embed = self.piece_embedding.forward(current_piece)?.squeeze(1)?; // [B, D]
        let x = self
            .piece_film
            .iter()
            .try_fold(cur_embed, |x, film| film.forward(&x, &piece_embed))?; // [B, D]

        // Head -> orientation logits [B, NUM_ORIENTATIONS]
        let logits = self.head_mlp.forward(&x)?; // [B, O]

        // Value head -> value [B, 1]
        let value_logits = self.value_head.forward(&x)?;
        let value = candle_nn::ops::sigmoid(&value_logits)?;

        Ok((TetrisPieceOrientationLogitsTensor::try_from(logits)?, value))
    }
}

/// A single element in a trajectory
#[derive(Debug, Clone)]
struct TrajectoryElem {
    orientation: TetrisPieceOrientation,

    // model outputs
    orientation_logits: Tensor,
    value: Tensor,

    // step reward computed from board transition
    step_reward: f32,
}

/// A single trajectory recording all states and actions for one game
#[derive(Debug, Clone)]
struct Trajectory {
    elems: Vec<TrajectoryElem>,
}

#[derive(Debug, Clone)]
struct Transition {
    orientation: TetrisPieceOrientation,
    orientation_logits: Tensor,
    value: Tensor,
    reward: f32,
}

pub fn train_simple_imitation_policy(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
) -> Result<()> {
    let device = tetris_ml::device();
    let dtype = fdtype();

    // Hyperparameters
    const NUM_ITERATIONS: usize = 1_000_000;
    const GAMES_PER_ITERATION: u32 = 1024;
    const PARALLEL_GAMES: usize = 256;
    const MINI_BATCH_SIZE: usize = 10_000;
    const NUM_MINI_BATCHES: usize = 1;
    const CHECKPOINT_INTERVAL: usize = 100;
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0);
    const TEMPERATURE: f32 = 1.2; // Higher = more exploration during sampling
    const ENTROPY_WEIGHT: f32 = 0.1; // Higher = more exploration (entropy bonus)
    let model_dim = 64;

    let model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    let policy_cfg = TetrisSimpleImitationPolicyConfig {
        piece_embedding_dim: model_dim,
        board_encoder_config: vec![
            AttnMlpConfig {
                hidden_size: 31, // heights(10) + holes(10) + bumpiness(9) + agg_height(1) + max_height(1)
                intermediate_size: 4 * model_dim,
                output_size: 2 * model_dim,
                dropout: Some(0.01),
            },
            AttnMlpConfig {
                hidden_size: 2 * model_dim,
                intermediate_size: 2 * model_dim,
                output_size: model_dim,
                dropout: Some(0.01),
            },
        ],
        piece_film_config: vec![
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
        ],
        head_mlp_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            dropout: Some(0.01),
        },
        value_head_config: AttnMlpConfig {
            hidden_size: model_dim,
            intermediate_size: 2 * model_dim,
            output_size: 1,
            dropout: Some(0.01),
        },
    };
    let policy = TetrisSimpleImitationPolicy::init(&vb, &policy_cfg)?;
    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(model_params.clone(), ParamsAdamW::default())?;

    let mut summary_writer = logdir.map(|s| SummaryWriter::new(s));
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let _ = std::fs::create_dir_all(dir);
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    let mut grad_accum = GradientAccumulator::new(NUM_MINI_BATCHES);

    let mut active_games = tetris_game::TetrisGameSet::new(PARALLEL_GAMES);
    let active_games_ptr = active_games.0.as_mut_ptr() as usize;

    let mut transitions: Vec<Transition> = Vec::with_capacity(MINI_BATCH_SIZE);

    let mut trajectories_in_progress: Vec<Trajectory> = (0..PARALLEL_GAMES)
        .map(|_| Trajectory { elems: Vec::new() })
        .collect();
    let trajectories_in_progress_ptr = trajectories_in_progress.as_mut_ptr() as usize;

    let mut rng = rand::rng();

    let mut total_games: u64 = 0;
    let mut total_pieces_played: u64 = 0;

    for iteration in 0..NUM_ITERATIONS {
        // Reset per-iteration counters
        let games_this_iteration = AtomicU32::new(0);
        let pieces_this_iteration = AtomicU64::new(0);

        // Play GAMES_PER_ITERATION games
        while games_this_iteration.load(Ordering::Relaxed) < GAMES_PER_ITERATION {
            // Batched forward pass
            let game_boards = active_games.boards();
            // Create feature tensor: heights (10) + holes (10) + bumpiness (9) + agg_height (1) + max_height (1) = 31 features
            const NUM_BOARD_FEATURES: usize = 31;
            let board_features: Vec<f32> = game_boards
                .iter()
                .flat_map(|board| {
                    let heights = board.heights();
                    let holes = board.holes();
                    let max_height = heights.iter().max().copied().unwrap_or(0) as f32;
                    let agg_height = heights.iter().sum::<u32>() as f32;

                    // Normalize heights by board height
                    let norm_heights = heights
                        .iter()
                        .map(|&h| h as f32 / TetrisBoard::HEIGHT as f32);
                    // Holes (typically 0-5 per column)
                    let norm_holes = holes.iter().map(|&h| h as f32 / 5.0);
                    // Bumpiness: absolute height differences between adjacent columns
                    let bumpiness = heights.windows(2).map(|w| {
                        (w[1] as i32 - w[0] as i32).abs() as f32 / TetrisBoard::HEIGHT as f32
                    });

                    norm_heights
                        .chain(norm_holes)
                        .chain(bumpiness)
                        .chain(std::iter::once(
                            agg_height / (TetrisBoard::HEIGHT * TetrisBoard::WIDTH) as f32,
                        ))
                        .chain(std::iter::once(max_height / TetrisBoard::HEIGHT as f32))
                        .collect::<Vec<_>>()
                })
                .collect();
            let board_features = Tensor::from_vec(
                board_features,
                Shape::from_dims(&[game_boards.len(), NUM_BOARD_FEATURES]),
                &device,
            )?;
            let current_pieces_vec = active_games.current_pieces();
            let current_piece =
                TetrisPieceTensor::from_pieces(current_pieces_vec.to_slice(), &device)?;

            // Sample actions with mask
            let (logits, value) = policy.forward_logits(&board_features, &current_piece)?;
            let sampled_orientations = logits.sample(TEMPERATURE, &current_piece)?;
            let orientations = sampled_orientations.into_orientations()?;

            let new_transitions = (0..PARALLEL_GAMES)
                .into_par_iter()
                .flat_map_iter(|i| {
                    let traj =
                        unsafe { &mut *(trajectories_in_progress_ptr as *mut Trajectory).add(i) };
                    let game = unsafe { &mut *(active_games_ptr as *mut TetrisGame).add(i) };

                    let piece = game.current_piece;
                    let orientation = orientations[i];

                    let orientation_logits = logits
                        .inner()
                        .i(i)
                        .expect("Failed to index logits")
                        .unsqueeze(0)
                        .expect("Failed to unsqueeze logits");
                    let value_tensor = value.i((i, ..)).expect("Failed to index value");

                    // Compute board metrics before the move
                    let board_before = game.board;
                    let holes_before = board_before.holes().iter().sum::<u32>() as f32;
                    let cells_before = board_before.count() as f32;

                    let is_lost = game.apply_placement(TetrisPiecePlacement { piece, orientation });

                    // Compute board metrics after the move
                    let board_after = game.board;
                    let holes_after = board_after.holes().iter().sum::<u32>() as f32;
                    let cells_after = board_after.count() as f32;

                    // Compute transition reward
                    // Holes created this step (positive = created holes)
                    let holes_created = (holes_after - holes_before).max(0.0);
                    // Lines cleared (count cells before + 4 new cells - cells after)
                    let lines_cleared =
                        (cells_before + 4.0 - cells_after) / TetrisBoard::WIDTH as f32;

                    // Reward: positive baseline, small hole penalty, line clear bonus
                    let step_reward = if bool::from(is_lost.is_lost) {
                        -0.1 // Small death penalty
                    } else {
                        0.1 // Base survival reward
                            - 0.02 * holes_created // Tiny hole penalty
                            + 1.0 * lines_cleared // Line clear bonus
                    };

                    traj.elems.push(TrajectoryElem {
                        orientation,
                        orientation_logits,
                        value: value_tensor,
                        step_reward,
                    });

                    if bool::from(is_lost.is_lost) {
                        let num_pieces = game.piece_count;

                        // Use immediate step rewards directly (no temporal accumulation)
                        // This is more natural for transition-based rewards where each
                        // action is judged on its immediate impact on the board
                        let episode_transitions: Vec<Transition> = traj
                            .elems
                            .iter()
                            .map(|traj_elem| Transition {
                                orientation: traj_elem.orientation,
                                orientation_logits: traj_elem.orientation_logits.clone(),
                                value: traj_elem.value.clone(),
                                reward: traj_elem.step_reward,
                            })
                            .collect();
                        traj.elems.clear();
                        game.reset(None);
                        games_this_iteration.fetch_add(1, Ordering::Relaxed);
                        pieces_this_iteration.fetch_add(num_pieces as u64, Ordering::Relaxed);
                        episode_transitions
                    } else {
                        vec![]
                    }
                })
                .collect::<Vec<_>>();
            transitions.extend(new_transitions);
        } // End of games-per-iteration loop

        // Update total counters
        let games_completed = games_this_iteration.load(Ordering::Relaxed);
        let pieces_played = pieces_this_iteration.load(Ordering::Relaxed);
        total_games += games_completed as u64;
        total_pieces_played += pieces_played;

        let avg_pieces_this_iter = if games_completed > 0 {
            pieces_played as f64 / games_completed as f64
        } else {
            0.0
        };

        // Skip training if we don't have enough transitions
        if transitions.is_empty() {
            println!(
                "[Iteration {}] No transitions collected, skipping training",
                iteration
            );
            continue;
        }

        active_games.reset_all();
        transitions.shuffle(&mut rng);

        // Normalize rewards to 0-1 scale across the batch for better training stability
        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();
        // Debug: log reward statistics
        let reward_mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        println!("  Reward stats: mean={:.3}", reward_mean);

        let logits = Tensor::cat(
            &transitions
                .iter()
                .map(|t| t.orientation_logits.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        let target_tensor = TetrisPieceOrientationTensor::from_orientations(
            &transitions
                .iter()
                .map(|t| t.orientation)
                .collect::<Vec<_>>(),
            &device,
        )?;
        let value_predictions = Tensor::cat(
            &transitions
                .iter()
                .map(|t| t.value.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        let rewards_tensor = Tensor::from_vec(
            rewards.clone(),
            Shape::from_dims(&[transitions.len()]),
            &device,
        )?
        .to_dtype(dtype)?;
        transitions.clear();

        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let value_pred = value_predictions.squeeze(D::Minus1)?;

        println!(
            "  Tensor shapes: logits={:?}, target={:?}, log_probs={:?}",
            logits.shape(),
            target_tensor.shape(),
            log_probs.shape()
        );

        // Z-score normalized advantages
        let raw_advantages = rewards_tensor.sub(&value_pred)?;
        let adv_mean = raw_advantages.mean_all()?;
        let adv_centered = raw_advantages.broadcast_sub(&adv_mean)?;
        let adv_std = adv_centered.sqr()?.mean_all()?.sqrt()?;
        let advantages = adv_centered
            .broadcast_div(&(adv_std.clone() + 1e-8)?)?
            .clamp(-2.0, 2.0)?;

        // Policy gradient loss: -advantage * log_prob (clamp log_prob to prevent explosion)
        let log_prob_selected = log_probs
            .gather(&target_tensor, D::Minus1)?
            .squeeze(1)?
            .clamp(-10.0, 0.0)?;
        let policy_loss = (&advantages.neg()? * &log_prob_selected)?.mean_all()?;

        // Debug: check values
        let mean_log_prob = log_prob_selected
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let mean_advantage = advantages
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let std_advantage = adv_std.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let mean_raw_adv = raw_advantages
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let policy_loss_value = policy_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        println!(
            "  Mean LogProb: {:.4}, Raw Adv: {:.4}, Scaled Adv: {:.4}, Std Adv: {:.4}, Policy Loss: {:.4}",
            mean_log_prob, mean_raw_adv, mean_advantage, std_advantage, policy_loss_value
        );

        // Entropy bonus (maximize entropy to encourage exploration)
        let entropy = (&log_probs.exp()? * &log_probs)?
            .sum(D::Minus1)?
            .neg()?
            .mean_all()?;
        let entropy_value = entropy.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        // Subtract entropy from loss to encourage exploration
        // Higher entropy -> Lower loss -> Better gradient
        let entropy_loss = entropy.affine(-ENTROPY_WEIGHT as f64, 0.0)?;

        // Value loss: train value function to predict rewards
        const VALUE_LOSS_WEIGHT: f64 = 0.5;
        let value_loss = (&value_pred - &rewards_tensor)?.sqr()?.mean_all()?;
        let value_loss_scaled = value_loss.affine(VALUE_LOSS_WEIGHT, 0.0)?;
        let value_loss_value = value_loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        // Total loss: policy + entropy + value (value was missing before!)
        let loss = ((policy_loss + entropy_loss)? + value_loss_scaled)?;
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        grad_accum.accumulate(grads, &model_params)?;

        let _stepped =
            grad_accum.apply_and_reset(&mut optimizer, &model_params, CLIP_GRAD_MAX_NORM, None)?;

        let overall_avg_pieces = if total_games > 0 {
            total_pieces_played as f64 / total_games as f64
        } else {
            0.0
        };

        println!(
            "[Iter {}] Games: {} (total: {}) | Avg Pieces: {:.1} (overall: {:.1}) | Policy: {:.4} | Entropy: {:.4} | Value: {:.4} | Loss: {:.4}",
            iteration,
            games_completed,
            total_games,
            avg_pieces_this_iter,
            overall_avg_pieces,
            policy_loss_value,
            entropy_value,
            value_loss_value,
            loss_value
        );

        // Logging
        if let Some(ref mut writer) = summary_writer {
            writer.add_scalar("games/per_iteration", games_completed as f32, iteration);
            writer.add_scalar("games/total", total_games as f32, iteration);
            writer.add_scalar(
                "games/avg_pieces_iter",
                avg_pieces_this_iter as f32,
                iteration,
            );
            writer.add_scalar(
                "games/avg_pieces_overall",
                overall_avg_pieces as f32,
                iteration,
            );
            writer.add_scalar("loss/policy", policy_loss_value, iteration);
            writer.add_scalar("loss/entropy", entropy_value, iteration);
            writer.add_scalar("loss/value", value_loss_value, iteration);
            writer.add_scalar("loss/total", loss_value, iteration);
        }

        // Checkpoint
        if let Some(ref cp) = checkpointer {
            if iteration % CHECKPOINT_INTERVAL == 0 {
                let _ = cp.checkpoint_item(iteration, &model_varmap, None, None);
            }
        }
    } // End of iteration loop

    println!("\nTraining complete! Total games: {}", total_games);
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
    info!("Starting tetris simple imitation training");
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

    train_simple_imitation_policy(run_name, logdir, checkpoint_dir).unwrap();
}
