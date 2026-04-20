use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use candle_core::{D, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use tensorboard::summary_writer::SummaryWriter;
use tetris_ml::Mlp;
use tetris_ml::MlpConfig;
use tracing::{Level, info};
use tracing_subscriber::prelude::*;

use tetris_game::{
    IsLost, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_ml::fdtype;
use tetris_ml::ops::{create_orientation_mask, get_l2_norm};
use tetris_ml::tensors::{
    TetrisPieceOneHotTensor, TetrisPieceOrientationDistTensor, TetrisPieceOrientationLogitsTensor,
    TetrisPieceTensor,
};
use tetris_ml::wrapped_tensor::WrappedTensor;
use tetris_ml::{
    checkpointer::{Checkpointer, sorted_named_vars},
    device,
    grad_accum::GradientAccumulator,
    optim::{AdamW, ParamsAdamW},
    set_global_threadpool,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch, OrientationCounts};

fn board_health(board: &TetrisBoard) -> f64 {
    let max_height = board.height() as f64;
    let total_holes = board.total_holes() as f64;
    let heights = board.heights();
    let mut bumpiness: f64 = 0.0;
    let mut i = 0;
    while i < heights.len() - 1 {
        bumpiness += (heights[i] as f64 - heights[i + 1] as f64).abs();
        i += 1;
    }
    let height_score = 1.0 - (max_height / 20.0);
    let hole_score = 1.0 - (total_holes / 20.0).min(1.0);
    let bump_score = 1.0 - (bumpiness / 30.0).min(1.0);
    (0.5 * height_score + 0.35 * hole_score + 0.15 * bump_score).clamp(0.0, 1.0)
}

fn adaptive_kink_prob(board: &TetrisBoard, max_kink_prob: f64) -> f64 {
    let health = board_health(board);
    let p = if health < 0.4 {
        0.0
    } else if health > 0.7 {
        max_kink_prob
    } else {
        max_kink_prob * (health - 0.4) / 0.3
    };
    p.min(0.1)
}

pub struct TetrisGameIter<
    const NUM_GAMES: usize,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> {
    pub games: [TetrisGame; NUM_GAMES],
    rng: SmallRng,
    pub kink_prob: f64,
    pub loss_resets: u64,
    pub completed_games: u64,
    pub completed_game_pieces_sum: u64,
    search: MultiBeamSearch<
        BeamTetrisState,
        NUM_BEAMS,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    >,
    step_counter: u64,
    current_game_idx: usize, // Round-robin index for batch diversity
}

impl<
    const NUM_GAMES: usize,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> TetrisGameIter<NUM_GAMES, NUM_BEAMS, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
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
            search: MultiBeamSearch::new(),
            step_counter: 0,
            current_game_idx: 0,
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
            search: MultiBeamSearch::new(),
            step_counter: seed,
            current_game_idx: 0,
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(seed) = seed {
            self.rng = SmallRng::seed_from_u64(seed);
            self.step_counter = seed;
        }
        for g in self.games.iter_mut() {
            g.reset(Some(self.rng.random::<u64>()));
        }
        self.loss_resets = 0;
        self.completed_games = 0;
        self.completed_game_pieces_sum = 0;
        self.current_game_idx = 0;
    }

    pub fn with_kink_prob(mut self, kink_prob: f64) -> Self {
        self.kink_prob = kink_prob;
        self
    }
}

impl<
    const NUM_GAMES: usize,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> Iterator
    for TetrisGameIter<NUM_GAMES, NUM_BEAMS, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiece, OrientationCounts);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to get a sample from one of the games
            if let Some(sample) = self.try_next_sample() {
                return Some(sample);
            }
            // All games failed - reset all and retry
            self.reset(None);
        }
    }
}

impl<
    const NUM_GAMES: usize,
    const NUM_BEAMS: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> TetrisGameIter<NUM_GAMES, NUM_BEAMS, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    /// Try to generate one training sample from the game pool.
    /// Uses round-robin sampling to ensure batch diversity.
    /// Returns None if all games are lost or beam search fails for all games.
    fn try_next_sample(&mut self) -> Option<(TetrisBoard, TetrisPiece, OrientationCounts)> {
        // Try all games starting from current round-robin position
        for _ in 0..(3 * NUM_GAMES) {
            let game_idx = self.current_game_idx;
            // Advance round-robin index for next call
            self.current_game_idx = (self.current_game_idx + 1) % NUM_GAMES;

            let g = &mut self.games[game_idx];

            // Reset lost games and move to next
            if g.board.is_lost() {
                self.completed_games += 1;
                self.completed_game_pieces_sum += g.piece_count as u64;
                g.reset(Some(self.rng.random()));
                self.loss_resets += 1;
                continue;
            }

            let effective_kink = adaptive_kink_prob(&g.board, self.kink_prob);
            if effective_kink > 0.0 && self.rng.random_bool(effective_kink) {
                let placements = g.current_placements();
                let random_action = placements[self.rng.random_range(0..placements.len())];

                let result = g.apply_placement(random_action);
                if result.is_lost == IsLost::LOST {
                    self.completed_games += 1;
                    self.completed_game_pieces_sum += g.piece_count as u64;
                    g.reset(Some(self.rng.random()));
                    self.loss_resets += 1;
                }
                // Continue to next iteration - will try beam search on perturbed state
                continue;
            }

            let board_before = g.board;
            let piece = g.current_piece;

            // Use multi-beam search to get vote counts
            let counts = self.search.search_count_actions_with_seeds(
                BeamTetrisState::new(*g),
                self.step_counter,
                MAX_DEPTH,
                BEAM_WIDTH,
            );

            self.step_counter += 1;

            // Pick the action with most votes to actually play
            let (orientation, vote_count) = counts.top_orientation();
            if vote_count == 0 {
                continue; // All beam searches failed
            }

            let action = TetrisPiecePlacement { piece, orientation };

            // Apply action to advance the game
            let result = g.apply_placement(action);
            if result.is_lost == IsLost::LOST {
                self.completed_games += 1;
                self.completed_game_pieces_sum += g.piece_count as u64;
                g.reset(Some(self.rng.random()));
                self.loss_resets += 1;
                continue;
            }

            // Success! Return training sample (board, piece, vote counts)
            return Some((board_before, piece, counts));
        }

        // All games failed in this cycle
        None
    }
}

#[derive(Debug, Clone)]
pub struct TetrisBeamSupervisedPolicyMLP {
    mlp: Mlp,
}

impl TetrisBeamSupervisedPolicyMLP {
    pub fn init(vb: &VarBuilder, cfg: &MlpConfig) -> Result<Self> {
        let mlp = Mlp::init(vb, cfg)?;
        Ok(Self { mlp })
    }

    /// Extract features from board state
    /// Features: [binary_board(200), piece_onehot(7)]
    fn forward_features(
        &self,
        boards: &[TetrisBoard],
        current_piece: &TetrisPieceTensor,
    ) -> Result<Tensor> {
        let batch_size = boards.len();
        let device = current_piece.device();
        let dtype = fdtype();

        let mut features_vec = Vec::with_capacity(batch_size * TetrisBoard::SIZE);

        for board in boards {
            let binary = board.to_binary_slice();
            for &cell in binary.iter() {
                features_vec.push(cell as f32);
            }
        }

        let board_features =
            Tensor::from_vec(features_vec, (batch_size, TetrisBoard::SIZE), device)?
                .to_dtype(dtype)?;
        let piece_oh = TetrisPieceOneHotTensor::from_piece_tensor(current_piece.clone())?;
        Ok(Tensor::cat(&[&board_features, piece_oh.inner()], 1)?)
    }

    pub fn forward_logits(
        &self,
        boards: &[TetrisBoard],
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let mut x = self.forward_features(boards, current_piece)?;

        x = self.mlp.forward(&x)?;

        TetrisPieceOrientationLogitsTensor::try_from(x)
    }

    pub fn forward_masked(
        &self,
        boards: &[TetrisBoard],
        current_piece: &TetrisPieceTensor,
    ) -> Result<TetrisPieceOrientationLogitsTensor> {
        let logits = self.forward_logits(boards, current_piece)?;
        let mask = create_orientation_mask(current_piece)?;
        let device = logits.device();
        let zero = mask.zeros_like()?;
        let keep = mask.gt(&zero)?;
        let neg_inf = Tensor::new(-1e9f32, &device)?.broadcast_as(logits.dims())?;
        let masked = keep.where_cond(logits.inner(), &neg_inf)?;
        TetrisPieceOrientationLogitsTensor::try_from(masked)
    }
}

pub fn train_tetris_beam_supervised(
    run_name: String,
    logdir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    resume: bool,
) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    const TEACHER_NUM_GAMES: usize = 32;
    const TEACHER_NUM_BEAMS: usize = 8;
    const TEACHER_TOP_N_PER_BEAM: usize = 128;
    const TEACHER_BEAM_WIDTH: usize = 128;
    const TEACHER_DEPTH: usize = 7;
    const TEACHER_MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const SEED: u64 = 123;
    const KINK_PROB: f64 = 0.2;

    const NUM_ITERATIONS: usize = 10_000_000;
    const BATCH_SIZE: usize = 4096;
    const CHECKPOINT_INTERVAL: usize = 100;
    const CLIP_GRAD_MAX_NORM: Option<f64> = Some(1.0);
    const CLIP_GRAD_MAX_VALUE: Option<f64> = None;

    let model_dim = 128;

    let mut model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);

    // Binary board (200) + piece one-hot (7)
    let in_size = TetrisBoard::SIZE + TetrisPiece::NUM_PIECES;
    let h = model_dim;
    let policy_cfg = MlpConfig {
        input_size: in_size,
        hidden_sizes: vec![h, h, h, h, h, h],
        residual: true,
        output_size: TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS,
        leaky_relu_negative_slope: 0.1,
        dropout: Some(0.05),
    };
    let policy = TetrisBeamSupervisedPolicyMLP::init(&vb, &policy_cfg)?;

    let named_params = sorted_named_vars(&model_varmap);
    let model_params: Vec<_> = named_params.iter().map(|(_, v)| v.clone()).collect();
    let mut optimizer = AdamW::new_named(
        named_params,
        ParamsAdamW {
            ..Default::default()
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

    let mut starting_iteration = 0;
    if resume {
        if let Some(cp) = checkpointer.as_ref() {
            match cp.load_latest_checkpoint(&mut model_varmap) {
                Ok(Some(iteration)) => {
                    starting_iteration = iteration;
                    println!("✅ Loaded model checkpoint from iteration {}", iteration);
                }
                Ok(None) => println!("ℹ️  No model checkpoint found, starting fresh"),
                Err(e) => println!("⚠️  Failed to load model checkpoint: {}", e),
            }
            match cp.load_item_latest(&mut optimizer, Some("optimizer")) {
                Ok(Some((path, iteration))) => {
                    println!(
                        "✅ Loaded optimizer checkpoint from iteration {} at {:?}",
                        iteration, path
                    );
                }
                Ok(None) => println!("ℹ️  No optimizer checkpoint found, using fresh optimizer"),
                Err(e) => println!("⚠️  Failed to load optimizer checkpoint: {}", e),
            }
        }
    }

    let mut teacher = TetrisGameIter::<
        TEACHER_NUM_GAMES,
        TEACHER_NUM_BEAMS,
        TEACHER_TOP_N_PER_BEAM,
        TEACHER_BEAM_WIDTH,
        TEACHER_DEPTH,
        TEACHER_MAX_MOVES,
    >::new_with_seed(SEED)
    .with_kink_prob(KINK_PROB);

    info!("Starting BeamSupervised training: iterations={NUM_ITERATIONS} batch={BATCH_SIZE}");

    let mut rng = rand::rng();

    let mut batch = Vec::with_capacity(BATCH_SIZE);
    for iteration in starting_iteration..NUM_ITERATIONS {
        let loss_resets_before_iter = teacher.loss_resets;

        batch.clear();
        batch.extend(teacher.by_ref().take(BATCH_SIZE));
        batch.shuffle(&mut rng);

        let boards = batch.iter().map(|(b, _p, _c)| *b).collect::<Vec<_>>();
        let pieces = batch.iter().map(|(_, p, _c)| *p).collect::<Vec<_>>();
        let piece_tensor = TetrisPieceTensor::from_pieces(&pieces, &device)?;
        let counts = batch.iter().map(|(_, _, c)| *c).collect::<Vec<_>>();

        // Convert orientation counts to soft label distribution
        let target_dist =
            TetrisPieceOrientationDistTensor::from_orientation_counts(&counts, &device)?;

        // Apply the same masking to target distribution as we do to logits
        // This prevents NaN from -inf * small_probability on invalid orientations
        let mask = create_orientation_mask(&piece_tensor)?;
        let zero = mask.zeros_like()?;
        let keep_mask = mask.gt(&zero)?;
        let masked_target = keep_mask
            .to_dtype(fdtype())?
            .broadcast_mul(target_dist.inner())?;

        // Re-normalize after masking to ensure it's a valid distribution
        // Add epsilon to prevent division by zero
        let target_sum_raw = masked_target.sum(D::Minus1)?;
        let eps = Tensor::new(1e-8f32, &device)?
            .to_dtype(fdtype())?
            .broadcast_as(target_sum_raw.shape())?;
        let target_sum = (target_sum_raw + eps)?.unsqueeze(D::Minus1)?;
        let target_dist_normalized = masked_target.broadcast_div(&target_sum)?;

        let logits = policy.forward_masked(&boards, &piece_tensor)?;

        // Debug: Check logits before softmax
        let logits_check = logits.inner().flatten_all()?.to_vec1::<f32>()?;
        let has_nan_logits = logits_check.iter().any(|x| !x.is_finite());
        if has_nan_logits {
            eprintln!("⚠️  Logits contain NaN/Inf before softmax at iter {iteration}");
            let nan_count = logits_check.iter().filter(|x| !x.is_finite()).count();
            eprintln!("  NaN/Inf count: {}/{}", nan_count, logits_check.len());
            continue;
        }

        let probs = candle_nn::ops::softmax(logits.inner(), D::Minus1)?;

        // MSE loss: L2 distance between predicted and target distributions
        // Both distributions are already masked and normalized
        let diff = (&probs - &target_dist_normalized)?;
        let squared_diff = diff.sqr()?;
        let mse_per_sample = squared_diff.sum(D::Minus1)?;
        let loss = mse_per_sample.mean_all()?;

        // Compute entropy for monitoring (even though not used in loss)
        // Add epsilon for numerical stability in log
        let eps_broadcast = Tensor::new(1e-8f32, &device)?
            .to_dtype(fdtype())?
            .broadcast_as(probs.dims())?;
        let log_probs = (&probs + eps_broadcast)?.log()?;
        let keep_mask_float = keep_mask.to_dtype(fdtype())?;
        let log_probs_masked = log_probs.mul(&keep_mask_float)?;
        let entropy = (&probs).mul(&log_probs_masked)?.sum(D::Minus1)?.neg()?;
        let avg_entropy = entropy.mean_all()?;

        // Compute accuracy: argmax(predictions) == argmax(masked_target_distribution)
        let predicted = logits.inner().argmax(D::Minus1)?;
        let target_argmax = target_dist_normalized.argmax(D::Minus1)?;
        let correct = predicted
            .eq(&target_argmax)?
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

            // Debug: check what's causing the non-finite loss
            let target_sum_check = target_sum.to_vec2::<f32>()?;
            let has_zero_sum = target_sum_check.iter().flatten().any(|&x| x < 1e-6);
            eprintln!("  Debug: target_sum has near-zero? {}", has_zero_sum);

            let probs_check = probs.flatten_all()?.to_vec1::<f32>()?;
            let has_nan_probs = probs_check.iter().any(|x| !x.is_finite());
            eprintln!("  Debug: probs has NaN/Inf? {}", has_nan_probs);

            let log_probs_check = log_probs_masked.flatten_all()?.to_vec1::<f32>()?;
            let has_nan_log = log_probs_check.iter().any(|x| !x.is_finite());
            eprintln!("  Debug: log_probs_masked has NaN/Inf? {}", has_nan_log);

            let target_check = target_dist_normalized.flatten_all()?.to_vec1::<f32>()?;
            let has_nan_target = target_check.iter().any(|x| !x.is_finite());
            eprintln!("  Debug: target_dist has NaN/Inf? {}", has_nan_target);

            continue;
        }

        let grads = loss.backward()?;
        let grad_norm = get_l2_norm(&grads)?;
        grad_accum.accumulate(grads, &model_params)?;
        grad_accum.apply_and_reset(
            &mut optimizer,
            &model_params,
            CLIP_GRAD_MAX_NORM,
            CLIP_GRAD_MAX_VALUE,
        )?;

        if let Some(s) = summary_writer.as_mut() {
            s.add_scalar("loss/mse", loss_scalar, iteration);
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

        if let Some(cp) = checkpointer.as_ref() {
            let _ = cp.checkpoint_item(iteration, &model_varmap, None, None)?;
            let _ = cp.checkpoint_item(iteration, &optimizer, None, Some("optimizer"))?;
        }

        println!(
            "iter={iteration} loss={loss_scalar:.4} acc={accuracy:.3} ent={entropy_scalar:.3} grad_norm={grad_norm:.3} loss_resets_iter={loss_resets_this_iter} loss_resets_total={loss_resets_total} pieces_sum={pieces_sum} pieces_current={current_game_pieces} pieces_mean_done={mean_completed_game_pieces:.1}"
        );
    }

    Ok(())
}

pub fn inference_tetris_beam_supervised(checkpoint_dir: PathBuf) -> Result<()> {
    let device = device();
    let dtype = fdtype();

    let cfg_path = checkpoint_dir.join("beam_supervised_policy_config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read policy config at {:?} (expected alongside checkpoint): {e}",
            cfg_path
        )
    })?;
    let policy_cfg: MlpConfig = serde_json::from_str(&cfg_str).map_err(|e| {
        anyhow::anyhow!("Failed to parse policy config JSON at {:?}: {e}", cfg_path)
    })?;

    // Find the latest .safetensors checkpoint (excluding optimizer files) in the directory
    let checkpoint_path = std::fs::read_dir(&checkpoint_dir)
        .map_err(|e| anyhow::anyhow!("Failed to read checkpoint dir {:?}: {e}", checkpoint_dir))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".safetensors") && !name.contains(".optimizer.")
        })
        .max_by_key(|entry| entry.file_name())
        .map(|entry| entry.path())
        .ok_or_else(|| {
            anyhow::anyhow!("No .safetensors checkpoint found in {:?}", checkpoint_dir)
        })?;

    info!("Loading policy config from {:?}", cfg_path);
    info!("Loading checkpoint from {:?}", checkpoint_path);

    let mut model_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&model_varmap, dtype, &device);
    let policy = TetrisBeamSupervisedPolicyMLP::init(&vb, &policy_cfg)?;

    model_varmap
        .load(&checkpoint_path)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint {:?}: {e}", checkpoint_path))?;

    info!("Loaded BeamSupervised checkpoint successfully");

    const NUM_GAMES: usize = 10;
    const INFERENCE_BATCH_SIZE: usize = 1;
    const HISTOGRAM_BINS: usize = 10;
    const MAX_STEPS_PER_GAME: u64 = 10_000_000;
    const BAR_WIDTH: usize = 40;

    let mut pieces_per_game = Vec::with_capacity(NUM_GAMES);
    let mut lines_per_game = Vec::with_capacity(NUM_GAMES);

    for batch_start in (0..NUM_GAMES).step_by(INFERENCE_BATCH_SIZE) {
        let batch_end = (batch_start + INFERENCE_BATCH_SIZE).min(NUM_GAMES);
        let batch_len = batch_end - batch_start;

        let mut seeds = Vec::with_capacity(batch_len);
        let mut games = Vec::with_capacity(batch_len);
        let mut steps = vec![0u64; batch_len];
        let mut finished = vec![false; batch_len];

        for _ in 0..batch_len {
            let seed: u64 = rand::random();
            seeds.push(seed);
            games.push(TetrisGame::new_with_seed(seed));
        }

        while finished.iter().any(|&x| !x) {
            let mut active_local_indices = Vec::with_capacity(batch_len);
            let mut active_boards = Vec::with_capacity(batch_len);
            let mut active_pieces = Vec::with_capacity(batch_len);

            for local_idx in 0..batch_len {
                if finished[local_idx] {
                    continue;
                }
                if steps[local_idx] >= MAX_STEPS_PER_GAME || games[local_idx].board.is_lost() {
                    finished[local_idx] = true;
                    continue;
                }

                active_local_indices.push(local_idx);
                active_boards.push(games[local_idx].board);
                active_pieces.push(games[local_idx].current_piece);
            }

            if active_local_indices.is_empty() {
                break;
            }

            let piece_tensor = TetrisPieceTensor::from_pieces(&active_pieces, &device)?;
            let logits = policy.forward_masked(&active_boards, &piece_tensor)?;
            let orientations = logits.sample(0.0, &piece_tensor)?.into_orientations()?;

            for (active_idx, &local_idx) in active_local_indices.iter().enumerate() {
                let game = &mut games[local_idx];
                let placement = TetrisPiecePlacement {
                    piece: game.current_piece,
                    orientation: orientations[active_idx],
                };

                let r = game.apply_placement(placement);
                steps[local_idx] += 1;
                if r.is_lost == IsLost::LOST || steps[local_idx] >= MAX_STEPS_PER_GAME {
                    finished[local_idx] = true;
                }
            }
        }

        for local_idx in 0..batch_len {
            pieces_per_game.push(steps[local_idx]);
            lines_per_game.push(games[local_idx].lines_cleared as u64);
            info!(
                "game={}/{} seed={} pieces={} lines_cleared={} board_height={}",
                batch_start + local_idx + 1,
                NUM_GAMES,
                seeds[local_idx],
                steps[local_idx],
                games[local_idx].lines_cleared,
                games[local_idx].board.height()
            );
        }
    }

    let mut pieces_sorted = pieces_per_game.clone();
    pieces_sorted.sort_unstable();

    let sum_pieces: u64 = pieces_per_game.iter().sum();
    let mean_pieces = sum_pieces as f64 / NUM_GAMES as f64;
    let median_pieces = if NUM_GAMES % 2 == 0 {
        (pieces_sorted[NUM_GAMES / 2 - 1] + pieces_sorted[NUM_GAMES / 2]) as f64 / 2.0
    } else {
        pieces_sorted[NUM_GAMES / 2] as f64
    };
    let p90_idx = ((NUM_GAMES as f64 * 0.90).ceil() as usize).saturating_sub(1);
    let p90_pieces = pieces_sorted[p90_idx.min(NUM_GAMES - 1)];
    let min_pieces = *pieces_sorted.first().unwrap_or(&0);
    let max_pieces = *pieces_sorted.last().unwrap_or(&0);
    let sum_lines: u64 = lines_per_game.iter().sum();

    println!();
    println!(
        "=== Beam-supervised inference over {NUM_GAMES} games (batch_size={INFERENCE_BATCH_SIZE}) ==="
    );
    println!("pieces mean   : {:.2}", mean_pieces);
    println!("pieces median : {:.2}", median_pieces);
    println!("pieces p90    : {}", p90_pieces);
    println!("pieces min/max: {min_pieces}/{max_pieces}");
    println!(
        "lines mean    : {:.2}",
        sum_lines as f64 / lines_per_game.len() as f64
    );
    println!();
    println!("Distribution of pieces played:");

    let range = max_pieces.saturating_sub(min_pieces);
    let bin_width = ((range + HISTOGRAM_BINS as u64) / HISTOGRAM_BINS as u64).max(1);
    let mut counts = vec![0usize; HISTOGRAM_BINS];
    for &value in &pieces_per_game {
        let mut idx = ((value.saturating_sub(min_pieces)) / bin_width) as usize;
        if idx >= HISTOGRAM_BINS {
            idx = HISTOGRAM_BINS - 1;
        }
        counts[idx] += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(1);
    for (i, &count) in counts.iter().enumerate() {
        let start = min_pieces + i as u64 * bin_width;
        let end = start + bin_width - 1;
        let filled = if max_count == 0 {
            0
        } else {
            (count * BAR_WIDTH + max_count - 1) / max_count
        };
        let bar = "#".repeat(filled);
        println!("[{start:>8}..{end:>8}] {count:>3} {bar}");
    }

    Ok(())
}

#[derive(Debug, Subcommand)]
enum Commands {
    Train {
        #[arg(long, help = "Path to save the tensorboard logs")]
        logdir: Option<String>,

        #[arg(long, help = "Path to load/save model checkpoints")]
        checkpoint_dir: Option<String>,

        #[arg(long, help = "Name of the training run")]
        run_name: String,

        #[arg(
            long,
            help = "Resume training from a ULID checkpoint (e.g. --resume 01JMABCD...)"
        )]
        resume: Option<String>,
    },
    Inference {
        #[arg(long, help = "Path to a checkpoint directory")]
        checkpoint: PathBuf,
    },
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity level (-v = ERROR, -vv = WARN, -vvv = INFO, -vvvv = DEBUG, -vvvvv = TRACE)")]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

fn main() {
    info!("Starting tetris beam supervised training");
    set_global_threadpool();

    let cli = Cli::parse();

    let verbosity = cli.verbose.saturating_add(2).clamp(0, 5);
    let level = Level::from_str(verbosity.to_string().as_str()).unwrap();

    let registry = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(tracing_subscriber::filter::LevelFilter::from_level(level));

    registry.init();
    info!("Logging initialized at level: {}", level);

    match &cli.command {
        Commands::Train {
            logdir,
            checkpoint_dir,
            run_name,
            resume,
        } => {
            let is_resume = resume.is_some();
            let ulid = match resume {
                Some(ulid) => ulid.clone(),
                None => ulid::Ulid::new().to_string(),
            };
            println!("Run: {run_name}/{ulid} (resume={is_resume})");
            let logdir = logdir.as_ref().map(|s| {
                let path = std::path::Path::new(s).join(run_name).join(&ulid);
                std::fs::create_dir_all(&path).expect("Failed to create log directory");
                path
            });
            let checkpoint_dir = checkpoint_dir.as_ref().map(|s| {
                let path = std::path::Path::new(s).join(run_name).join(&ulid);
                std::fs::create_dir_all(&path).expect("Failed to create checkpoint directory");
                path
            });
            train_tetris_beam_supervised(
                run_name.clone(),
                logdir.clone(),
                checkpoint_dir.clone(),
                is_resume,
            )
            .unwrap();
        }
        Commands::Inference { checkpoint } => {
            inference_tetris_beam_supervised(checkpoint.clone()).unwrap();
        }
    }
}
