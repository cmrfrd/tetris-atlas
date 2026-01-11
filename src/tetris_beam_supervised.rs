use std::path::PathBuf;

use anyhow::Result;
use candle_core::{D, DType, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, embedding};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tensorboard::summary_writer::SummaryWriter;
use tracing::info;

use crate::beam_search::{BeamSearch, BeamTetrisState};
use crate::checkpointer::Checkpointer;
use crate::grad_accum::{GradientAccumulator, get_l2_norm};
use crate::modules::{
    CausalSelfAttentionConfig, DynamicTanhConfig, Mlp, MlpConfig, TransformerBlockConfig,
    TransformerBody, TransformerBodyConfig,
};
use crate::ops::create_orientation_mask;
use crate::tensors::{
    TetrisBoardsTensor, TetrisPieceOneHotTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceOrientationTensor, TetrisPieceTensor,
};
use crate::tetris::{
    IsLost, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement,
};
use crate::wrapped_tensor::WrappedTensor;
use crate::{device, fdtype};

const NUM_GAMES_PER_ITER: usize = 16;

pub struct TetrisGameIter<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> {
    pub games: [TetrisGame; NUM_GAMES_PER_ITER],
    rng: SmallRng,
    /// Probability of taking a random action instead of the teacher action in `next()`.
    pub kink_prob: f64,
    /// Number of games reset due to loss (either already-lost when sampled, or lost after move).
    pub loss_resets: u64,
    /// Number of completed games (counted when a game is reset due to loss).
    pub completed_games: u64,
    /// Sum of piece counts over completed games.
    pub completed_game_pieces_sum: u64,
    search: BeamSearch<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        let mut rng = SmallRng::from_os_rng();
        let games = std::array::from_fn(|_| TetrisGame::new_with_seed(rng.random()));
        Self {
            games,
            rng,
            kink_prob: 0.0,
            loss_resets: 0,
            completed_games: 0,
            completed_game_pieces_sum: 0,
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let games = std::array::from_fn(|_| TetrisGame::new_with_seed(rng.random()));
        Self {
            games,
            rng,
            kink_prob: 0.0,
            loss_resets: 0,
            completed_games: 0,
            completed_game_pieces_sum: 0,
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(seed) = seed {
            self.rng = SmallRng::seed_from_u64(seed);
        }
        for g in &mut self.games {
            g.reset(Some(self.rng.random()));
        }
        self.loss_resets = 0;
        self.completed_games = 0;
        self.completed_game_pieces_sum = 0;
    }

    pub fn with_kink_prob(mut self, kink_prob: f64) -> Self {
        self.kink_prob = kink_prob;
        self
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try a few times to find a non-losing move from a randomly chosen game.
            for _ in 0..(NUM_GAMES_PER_ITER * 4) {
                let idx = self.rng.random_range(0..NUM_GAMES_PER_ITER);
                let g = &mut self.games[idx];

                if g.board.is_lost() {
                    self.completed_games += 1;
                    self.completed_game_pieces_sum += g.piece_count as u64;
                    g.reset(Some(self.rng.random()));
                    self.loss_resets += 1;
                }

                let board_before = g.board;

                // Kink: apply a random action to change the state distribution, but do NOT
                // return it as a supervised label.
                if self.kink_prob > 0.0 && self.rng.random_bool(self.kink_prob) {
                    let placements = g.current_placements();
                    let j = self.rng.random_range(0..placements.len());
                    let action = placements[j];

                    if g.apply_placement(action).is_lost == IsLost::LOST {
                        self.completed_games += 1;
                        self.completed_game_pieces_sum += g.piece_count as u64;
                        g.reset(Some(self.rng.random()));
                        self.loss_resets += 1;
                    }
                    continue;
                }

                // Teacher placement via beam search (this is what we supervise).
                let action = {
                    let Some(scored) = self
                        .search
                        .search_first_action_with_state(BeamTetrisState(*g), MAX_DEPTH)
                    else {
                        continue;
                    };
                    let Some(first_action) = scored.first_action else {
                        continue;
                    };
                    first_action
                };

                if g.apply_placement(action).is_lost == IsLost::LOST {
                    self.completed_games += 1;
                    self.completed_game_pieces_sum += g.piece_count as u64;
                    g.reset(Some(self.rng.random()));
                    self.loss_resets += 1;
                    continue;
                }

                return Some((board_before, action));
            }

            // If we fail to find a valid move after retries, reseed games and try again.
            self.reset(None);
        }
    }
}

/// Supervised policy trained on BeamSearch-generated actions.
///
/// Architecture mirrors `tetris_policy_gradients.rs` transformer policy, but **no value head**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TetrisBeamSupervisedPolicyConfig {
    pub d_model: usize,
    pub num_blocks: usize,
    pub attn_config: CausalSelfAttentionConfig,
    pub block_mlp_config: MlpConfig,
    pub head_mlp_config: MlpConfig,
    #[serde(default)]
    pub with_causal_mask: bool,
}

#[derive(Debug, Clone)]
pub struct TetrisBeamSupervisedPolicy {
    token_embed: Embedding, // [0/1] -> D
    pos_embed: Embedding,   // [0..199] -> D
    piece_embed: Embedding, // piece id -> D
    body: TransformerBody,
    head_mlp: Mlp, // pooled -> logits over orientations
    #[allow(dead_code)]
    d_model: usize,
    with_causal_mask: bool,
}

impl TetrisBeamSupervisedPolicy {
    pub fn init(vb: &VarBuilder, cfg: &TetrisBeamSupervisedPolicyConfig) -> Result<Self> {
        let token_embed = embedding(
            TetrisBoard::NUM_TETRIS_CELL_STATES,
            cfg.d_model,
            vb.pp("token_embed"),
        )?;
        let pos_embed = embedding(TetrisBoard::SIZE, cfg.d_model, vb.pp("pos_embed"))?;
        let piece_embed = embedding(TetrisPiece::NUM_PIECES, cfg.d_model, vb.pp("piece_embed"))?;

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
        let head_mlp = Mlp::init(&vb.pp("head_mlp"), &cfg.head_mlp_config)?;

        Ok(Self {
            token_embed,
            pos_embed,
            piece_embed,
            body,
            head_mlp,
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
        let tokens = current_board.inner().to_dtype(DType::U32)?;

        // Token + pos embeddings: [B, 200, D]
        let x_tokens = self.token_embed.forward(&tokens)?;
        let device = tokens.device();
        let pos_ids = Tensor::arange(0, TetrisBoard::SIZE as u32, device)?
            .to_dtype(DType::U32)?
            .reshape(&[1, TetrisBoard::SIZE])?
            .repeat(&[b, 1])?;
        let x_pos = self.pos_embed.forward(&pos_ids)?;

        // Piece conditioning: [B, D] -> [B, 200, D]
        let piece_embed = self.piece_embed.forward(current_piece)?.squeeze(1)?;
        let piece_b = piece_embed.unsqueeze(1)?.broadcast_as(x_tokens.dims())?;

        let x = ((&x_tokens + &x_pos)? + &piece_b)?;
        let x = self.body.forward(x, self.with_causal_mask)?;
        Ok(x.mean(1)?)
    }

    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let pooled = self.forward_pooled(current_board, current_piece)?;
        let logits = self.head_mlp.forward(&pooled)?;
        TetrisPieceOrientationLogitsTensor::try_from(logits)
    }

    /// Mask invalid orientations for the current piece.
    pub fn forward_masked(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(current_board, current_piece)?;
        let mask = create_orientation_mask(current_piece)?;
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        TetrisPieceOrientationLogitsTensor::try_from(masked)
    }
}

/// An MLP policy trained on BeamSearch-generated actions.
///
/// This is a `Vec<MlpConfig>` to mirror config patterns used elsewhere (see `src/modules.rs`).
///
/// For this policy, each `MlpConfig` entry is used to instantiate a `crate::modules::Mlp`,
/// and the list is applied sequentially.
pub type TetrisBeamSupervisedPolicyMLPConfig = Vec<MlpConfig>;

#[derive(Debug, Clone)]
pub struct TetrisBeamSupervisedPolicyMLP {
    mlps: Vec<Mlp>,
    #[allow(dead_code)]
    input_size: usize,
}

impl TetrisBeamSupervisedPolicyMLP {
    pub fn init(vb: &VarBuilder, cfg: &TetrisBeamSupervisedPolicyMLPConfig) -> Result<Self> {
        // Input features:
        // - board: 200 binary cells (as floats)
        // - piece: one-hot NUM_PIECES
        let input_size = TetrisBoard::SIZE + TetrisPiece::NUM_PIECES;
        let output_size = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

        anyhow::ensure!(
            !cfg.is_empty(),
            "TetrisBeamSupervisedPolicyMLP expects at least 1 MlpConfig"
        );
        anyhow::ensure!(
            cfg[0].hidden_size == input_size,
            "TetrisBeamSupervisedPolicyMLP mlp_0.hidden_size must equal input_size={}, got {}",
            input_size,
            cfg[0].hidden_size
        );
        anyhow::ensure!(
            cfg[cfg.len() - 1].output_size == output_size,
            "TetrisBeamSupervisedPolicyMLP last mlp.output_size must equal output_size={}, got {}",
            output_size,
            cfg[cfg.len() - 1].output_size
        );
        for i in 0..(cfg.len() - 1) {
            anyhow::ensure!(
                cfg[i].output_size == cfg[i + 1].hidden_size,
                "TetrisBeamSupervisedPolicyMLP mlp_{i}.output_size ({}) must equal mlp_{}.hidden_size ({})",
                cfg[i].output_size,
                i + 1,
                cfg[i + 1].hidden_size
            );
        }

        let mlp_vb = vb.pp("mlp");
        let mut mlps = Vec::with_capacity(cfg.len());
        for (i, mlp_cfg) in cfg.iter().enumerate() {
            mlps.push(Mlp::init(&mlp_vb.pp(format!("mlp_{i}")), mlp_cfg)?);
        }

        Ok(Self { mlps, input_size })
    }

    fn forward_features(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let board = current_board.inner().to_dtype(fdtype())?;
        let piece_oh = TetrisPieceOneHotTensor::from_piece_tensor(current_piece.clone())?;
        Ok(Tensor::cat(&[&board, piece_oh.inner()], 1)?)
    }

    pub fn forward_logits(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let mut x = self.forward_features(current_board, current_piece)?;

        for mlp in &self.mlps {
            x = mlp.forward(&x)?;
        }

        TetrisPieceOrientationLogitsTensor::try_from(x)
    }

    /// Mask invalid orientations for the current piece.
    pub fn forward_masked(
        &self,
        current_board: &TetrisBoardsTensor,
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(current_board, current_piece)?;
        let mask = create_orientation_mask(current_piece)?;
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        TetrisPieceOrientationLogitsTensor::try_from(masked)
    }
}

/// Train a transformer policy with supervised learning on BeamSearch-generated labels.
pub fn train_tetris_beam_supervised(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume: bool,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    // --- Data generation (teacher) ---
    const TEACHER_BEAM_WIDTH: usize = 16;
    const TEACHER_DEPTH: usize = 16;
    const TEACHER_MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS; // safe upper bound for placements
    const SEED: u64 = 123;
    /// Probability of injecting a "kink" where we play a random move instead of the teacher move.
    const KINK_PROB: f64 = 0.03;

    // --- Training hyperparams ---
    const NUM_ITERATIONS: usize = 10_000_000;
    const BATCH_SIZE: usize = 1024;
    const CHECKPOINT_INTERVAL: usize = 100;
    const LEARNING_RATE: f64 = 1e-4;
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0);
    const CLIP_GRAD_MAX_VALUE: Option<f64> = None;

    // Model hyperparams
    let model_dim = 32;
    // (transformer hyperparams kept here for reference; currently using the MLP policy)
    // let num_blocks = 4;
    // let num_attention_heads = 8;
    // let num_kv_heads = 8;

    let mut model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    // let policy_cfg = TetrisBeamSupervisedPolicyConfig {
    //     d_model: model_dim,
    //     num_blocks,
    //     attn_config: CausalSelfAttentionConfig {
    //         d_model: model_dim,
    //         n_attention_heads: num_attention_heads,
    //         n_kv_heads: num_kv_heads,
    //         rope_theta: 10_000.0,
    //         max_position_embeddings: TetrisBoard::SIZE,
    //     },
    //     block_mlp_config: MlpConfig {
    //         hidden_size: model_dim,
    //         intermediate_size: 2 * model_dim,
    //         output_size: model_dim,
    //         dropout: None,
    //     },
    //     head_mlp_config: MlpConfig {
    //         hidden_size: model_dim,
    //         intermediate_size: 2 * model_dim,
    //         output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
    //         dropout: None,
    //     },
    //     with_causal_mask: false,
    // };
    // let policy = TetrisBeamSupervisedPolicy::init(&vb, &policy_cfg)?;

    // For the Vec<MlpConfig> policy, every stage must chain:
    // mlp_i.output_size == mlp_{i+1}.hidden_size, and only the last mlp outputs orientations.
    let in_size = TetrisBoard::SIZE + TetrisPiece::NUM_PIECES;
    let h = 2 * model_dim;
    let policy_cfg = vec![
        MlpConfig {
            hidden_size: in_size,
            intermediate_size: h,
            output_size: h,
            dropout: None,
        },
        MlpConfig {
            hidden_size: h,
            intermediate_size: h,
            output_size: h,
            dropout: None,
        },
        MlpConfig {
            hidden_size: h,
            intermediate_size: h,
            output_size: h,
            dropout: None,
        },
        MlpConfig {
            hidden_size: h,
            intermediate_size: h,
            output_size: h,
            dropout: None,
        },
        MlpConfig {
            hidden_size: h,
            intermediate_size: h,
            output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
            dropout: None,
        },
    ];
    let policy = TetrisBeamSupervisedPolicyMLP::init(&vb, &policy_cfg)?;

    let model_params = model_varmap.all_vars();
    let mut optimizer = AdamW::new(
        model_params.clone(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            ..ParamsAdamW::default()
        },
    )?;
    let mut grad_accum = GradientAccumulator::new(1);

    let mut summary_writer = logdir.map(SummaryWriter::new);
    let checkpointer = checkpoint_dir.as_ref().map(|dir| {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            dir.join("beam_supervised_policy_config.json"),
            serde_json::to_string_pretty(&policy_cfg).unwrap(),
        );
        Checkpointer::new(CHECKPOINT_INTERVAL, dir.clone(), run_name.clone())
            .expect("Failed to create checkpointer")
    });

    if resume {
        if let Some(cp) = checkpointer.as_ref() {
            let _ = cp.load_latest_checkpoint(&mut model_varmap)?;
        }
    }

    // Teacher iterator: maintains 16 parallel games and samples one uniformly each step.
    // `kink_prob` injects random actions to broaden state distribution.
    let mut teacher =
        TetrisGameIter::<TEACHER_BEAM_WIDTH, TEACHER_DEPTH, TEACHER_MAX_MOVES>::new_with_seed(SEED)
            .with_kink_prob(KINK_PROB);

    info!("Starting BeamSupervised training: iterations={NUM_ITERATIONS} batch={BATCH_SIZE}");

    let mut rng = rand::rng();

    // Track basic teacher-game stats (stored on the iterator).

    let mut batch = Vec::with_capacity(BATCH_SIZE);
    for iteration in 0..NUM_ITERATIONS {
        let loss_resets_before_iter = teacher.loss_resets;

        // Build a full batch; `teacher.next()` handles game selection, kink injection,
        // and loss resets internally.
        batch.clear();
        batch.extend(teacher.by_ref().take(BATCH_SIZE));
        batch.shuffle(&mut rng);

        let boards = batch.iter().map(|(b, _p)| *b).collect::<Vec<_>>();
        let board_tensor = TetrisBoardsTensor::from_boards(&boards, &device)?;
        let pieces = batch.iter().map(|(_, p)| p.piece).collect::<Vec<_>>();
        let piece_tensor = TetrisPieceTensor::from_pieces(&pieces, &device)?;
        let targets = batch.iter().map(|(_, p)| p.orientation).collect::<Vec<_>>();
        let target_tensor = TetrisPieceOrientationTensor::from_orientations(&targets, &device)?;

        // Forward
        let logits = policy.forward_masked(&board_tensor, &piece_tensor)?;
        let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;
        let log_probs = candle_nn::ops::log_softmax(logits.inner(), D::Minus1)?;

        // Cross-entropy via one-hot targets: -E[log π(a|s)]
        let target_one_hot = target_tensor.into_dist()?;
        let nll = (&log_probs * target_one_hot.inner())?
            .sum(D::Minus1)?
            .neg()?; // [B]
        let loss = nll.mean_all()?;

        // Metrics
        let entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?;
        let avg_entropy = entropy.mean_all()?;
        let predicted = logits.inner().argmax(D::Minus1)?;
        let target_indices: Vec<u32> = targets.iter().map(|o| o.index() as u32).collect();
        let target_indices_tensor =
            Tensor::from_vec(target_indices, Shape::from_dims(&[BATCH_SIZE]), &device)?;
        let correct = predicted
            .eq(&target_indices_tensor)?
            .to_dtype(DType::F32)?
            .sum_all()?;
        let accuracy = correct.to_scalar::<f32>()? / BATCH_SIZE as f32;

        let loss_scalar = loss.to_scalar::<f32>()?;
        let entropy_scalar = avg_entropy.to_scalar::<f32>()?;
        let pieces_sum: u64 = teacher.games.iter().map(|g| g.piece_count as u64).sum();
        let current_game_pieces = teacher
            .games
            .iter()
            .map(|g| g.piece_count as u64)
            .max()
            .unwrap_or(0);
        let mean_completed_game_pieces = if teacher.completed_games > 0 {
            teacher.completed_game_pieces_sum as f64 / teacher.completed_games as f64
        } else {
            0.0
        };
        let loss_resets_this_iter = teacher.loss_resets - loss_resets_before_iter;
        let loss_resets_total = teacher.loss_resets;

        if !loss_scalar.is_finite() {
            eprintln!("⚠️  Non-finite loss at iter {iteration}, skipping");
            continue;
        }

        // Backward
        let grads = loss.backward()?;
        let grad_norm = get_l2_norm(&grads)?;
        grad_accum.accumulate(grads, &model_params)?;
        grad_accum.apply_and_reset(
            &mut optimizer,
            &model_params,
            CLIP_GRAD_MAX_NORM,
            CLIP_GRAD_MAX_VALUE,
        )?;

        // Log
        if let Some(s) = summary_writer.as_mut() {
            s.add_scalar("loss/supervised", loss_scalar, iteration);
            s.add_scalar("metrics/accuracy", accuracy, iteration);
            s.add_scalar("metrics/entropy", entropy_scalar, iteration);
            s.add_scalar("metrics/grad_norm", grad_norm as f32, iteration);
            s.add_scalar(
                "data/teacher/game_pieces_current",
                current_game_pieces as f32,
                iteration,
            );
            s.add_scalar("data/teacher/game_pieces_sum", pieces_sum as f32, iteration);
            s.add_scalar(
                "data/teacher/game_pieces_mean_completed",
                mean_completed_game_pieces as f32,
                iteration,
            );
            s.add_scalar(
                "data/teacher/loss_resets_total",
                loss_resets_total as f32,
                iteration,
            );
            s.add_scalar(
                "data/teacher/loss_resets_iter",
                loss_resets_this_iter as f32,
                iteration,
            );
        }

        // Checkpoint
        if let Some(cp) = checkpointer.as_ref() {
            let _ = cp.checkpoint_item(iteration, &model_varmap, None, None)?;
        }

        println!(
            "iter={iteration} loss={loss_scalar:.4} acc={accuracy:.3} ent={entropy_scalar:.3} grad_norm={grad_norm:.3} loss_resets_iter={loss_resets_this_iter} loss_resets_total={loss_resets_total} pieces_sum={pieces_sum} pieces_current={current_game_pieces} pieces_mean_done={mean_completed_game_pieces:.1}"
        );
    }

    Ok(())
}

/// Load a BeamSupervised checkpoint and play a Tetris game with the policy in inference mode.
///
/// `checkpoint` should point to a `.safetensors` file produced by training.
/// The policy config is expected at `checkpoint.parent()/beam_supervised_policy_config.json`.
pub fn inference_tetris_beam_supervised(checkpoint: PathBuf) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    let checkpoint_dir = checkpoint
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Checkpoint path has no parent dir: {checkpoint:?}"))?;
    let cfg_path = checkpoint_dir.join("beam_supervised_policy_config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read policy config at {:?} (expected alongside checkpoint): {e}",
            cfg_path
        )
    })?;
    let policy_cfg: TetrisBeamSupervisedPolicyMLPConfig =
        serde_json::from_str(&cfg_str).map_err(|e| {
            anyhow::anyhow!("Failed to parse policy config JSON at {:?}: {e}", cfg_path)
        })?;

    let mut model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);
    // let policy = TetrisBeamSupervisedPolicy::init(&vb, &policy_cfg)?;
    let policy = TetrisBeamSupervisedPolicyMLP::init(&vb, &policy_cfg)?;

    model_varmap
        .load(&checkpoint)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint {:?}: {e}", checkpoint))?;

    info!("Loaded BeamSupervised checkpoint {:?}", checkpoint);

    let mut game = TetrisGame::new();
    let mut steps: u64 = 0;

    loop {
        // Print the tetris board and wait for the user to hit enter to proceed
        println!("{}", game.current_piece);
        println!("{}", game.board);
        let mut input = String::new();
        println!("Press Enter to continue...");
        let _ = std::io::stdin().read_line(&mut input);

        if game.board.is_lost() {
            break;
        }

        let board_tensor = TetrisBoardsTensor::from_boards(&[game.board], &device)?;
        let piece_tensor = TetrisPieceTensor::from_pieces(&[game.current_piece], &device)?;
        let logits = policy.forward_masked(&board_tensor, &piece_tensor)?;

        // print distribution of logits
        let dist_tensor = logits.clone().inner().flatten_all()?;
        let dist_values = dist_tensor.to_vec1::<f32>()?;
        let rounded: Vec<f32> = dist_values
            .iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect();
        for (i, value) in rounded.iter().enumerate() {
            println!(
                "{}: {}",
                TetrisPieceOrientation::from_index(i as u8).to_string(),
                value
            );
        }

        let orientations = logits.sample(0.0, &piece_tensor)?.into_orientations()?;
        let placement = TetrisPiecePlacement {
            piece: game.current_piece,
            orientation: orientations[0],
        };

        let r = game.apply_placement(placement);
        steps += 1;

        if r.is_lost == IsLost::LOST {
            break;
        }
    }

    println!(
        "inference done: pieces={} lines_cleared={} board_height={}",
        steps,
        game.lines_cleared,
        game.board.height()
    );

    Ok(())
}
