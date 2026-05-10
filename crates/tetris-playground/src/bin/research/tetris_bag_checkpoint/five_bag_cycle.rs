//! Five-bag cycle feasibility checker via beam search.
//!
//! From the empty board, given 5 bag permutations (35 pieces), can we return to empty?
//! Uses beam search with cell-count pruning at bag boundaries.
//!
//! "Best effort" strategy: within each bag, use beam search to find good placements.
//! At bag boundaries, filter to valid cell counts. If the beam dies, widen and retry.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use super::common::{generate_permutations, has_too_many_holes};
use super::five_bag_mcts::{
    BoardConstraints, MctsPolicy, MctsTargetKind, MctsTerminalMode, PuctConfig, PuctSearchResult,
    TargetSpec, run_puct_search_with_progress,
};

fn fmt_bags(bags: &[usize]) -> String {
    bags.iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_bag_perms(input: &str, perm_count: usize) -> Result<Vec<usize>> {
    let mut bag_perms = Vec::new();
    for raw in input.split(',') {
        let token = raw.trim();
        if token.is_empty() {
            bail!("empty bag permutation index in `{input}`");
        }
        let perm = token
            .parse::<usize>()
            .with_context(|| format!("invalid bag permutation index `{token}`"))?;
        if perm >= perm_count {
            bail!(
                "bag permutation index {perm} is out of range 0..{}",
                perm_count - 1
            );
        }
        bag_perms.push(perm);
    }
    Ok(bag_perms)
}

fn validate_random_trials(args: &CycleRunConfig) -> Result<()> {
    if args.trials == 0 {
        bail!("--trials must be greater than zero");
    }
    if args.num_bags == 0 {
        bail!("--num-bags must be greater than zero");
    }
    Ok(())
}

fn fmt_optional_cells(cells: u32) -> String {
    if cells == u32::MAX {
        "-".to_owned()
    } else {
        cells.to_string()
    }
}

fn empty_puct_result() -> PuctSearchResult {
    PuctSearchResult {
        success: false,
        best_cells: u32::MAX,
        best_score: f32::NEG_INFINITY,
        iterations: 0,
        tree_nodes: 0,
        terminal_evaluations: 0,
    }
}

fn merge_puct_result(total: &mut PuctSearchResult, next: PuctSearchResult) {
    total.success |= next.success;
    total.best_cells = total.best_cells.min(next.best_cells);
    total.best_score = total.best_score.max(next.best_score);
    total.iterations += next.iterations;
    total.tree_nodes += next.tree_nodes;
    total.terminal_evaluations += next.terminal_evaluations;
}

#[derive(Args)]
pub struct FiveBagCycleArgs {
    #[command(subcommand)]
    command: FiveBagCycleCommand,
}

#[derive(Subcommand)]
enum FiveBagCycleCommand {
    /// Beam-search random N-bag sequences for return-to-empty feasibility.
    Beam(BeamCommandArgs),
    /// Run PUCT tree search or the rollout baseline over random N-bag sequences.
    Mcts(MctsCommandArgs),
    /// Beam the first N-1 bags, then rollout the last bag from each beam state.
    Split(SplitCommandArgs),
    /// Exact bag-by-bag DFS for one explicit bag-permutation sequence.
    Exact(SequenceCommandArgs),
    /// Exhaustive memoized DFS for one explicit bag-permutation sequence.
    Prove(SequenceCommandArgs),
}

#[derive(Args, Clone, Copy)]
struct BoardConstraintArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 20)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = u32::MAX)]
    max_holes: u32,
}

#[derive(Args, Clone, Copy)]
struct TrialSamplingArgs {
    /// Number of bags per trial.
    #[arg(long, default_value_t = 5)]
    num_bags: usize,

    /// Number of random N-bag sequences to test.
    #[arg(long, default_value_t = 10_000)]
    trials: u64,

    /// RNG seed for selecting bag permutations.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Args)]
struct BeamCommandArgs {
    #[command(flatten)]
    constraints: BoardConstraintArgs,

    #[command(flatten)]
    trials: TrialSamplingArgs,

    /// Beam width per piece step.
    #[arg(long, default_value_t = 1000)]
    beam_width: usize,

    /// Use best-effort: retry failed sequences with noisy beam search.
    #[arg(long, default_value_t = false)]
    best_effort: bool,

    /// Number of noisy retries per failed sequence (best-effort mode).
    #[arg(long, default_value_t = 50)]
    max_retries: usize,

    /// Noise magnitude for score perturbation (0 = no noise).
    /// Higher values explore more diverse states.
    #[arg(long, default_value_t = 0.0)]
    noise: f32,
}

#[derive(Args)]
struct MctsCommandArgs {
    #[command(flatten)]
    constraints: BoardConstraintArgs,

    #[command(flatten)]
    trials: TrialSamplingArgs,

    /// MCTS implementation to use.
    #[arg(long, value_enum, default_value = "puct")]
    mcts_policy: MctsPolicy,

    /// Number of PUCT iterations per sequence.
    #[arg(long, default_value_t = 50_000)]
    mcts_iterations: u64,

    /// Number of independent PUCT trees per trial.
    #[arg(long, default_value_t = 1)]
    mcts_restarts: u64,

    /// PUCT exploration constant.
    #[arg(long, default_value_t = 1.4)]
    mcts_exploration: f32,

    /// PUCT prior softmax temperature. Higher values flatten placement priors.
    #[arg(long, default_value_t = 2.0)]
    mcts_prior_temperature: f32,

    /// Deterministic per-restart prior perturbation scale.
    #[arg(long, default_value_t = 0.0)]
    mcts_prior_noise: f32,

    /// Maximum children expanded per PUCT node.
    #[arg(long, default_value_t = 48)]
    mcts_max_children: usize,

    /// Terminal miss scoring mode.
    #[arg(long, value_enum, default_value = "shaped")]
    mcts_terminal_mode: MctsTerminalMode,

    /// Extra penalty for repeatedly hitting the same failed terminal board in one tree.
    #[arg(long, default_value_t = 0.05)]
    mcts_terminal_repeat_penalty: f32,

    /// Print per-trial PUCT progress every N iterations. Use 0 to disable.
    #[arg(long, default_value_t = 25_000)]
    mcts_progress_every: u64,

    /// Terminal target for MCTS experiments.
    #[arg(long, value_enum, default_value = "empty")]
    target: MctsTargetKind,

    /// Accepted terminal cell counts for `--target cell-count`.
    #[arg(long, value_delimiter = ',')]
    target_cells: Vec<u32>,

    /// Optional terminal height bound for `--target cell-count` or `--target low-board`.
    #[arg(long)]
    target_max_height: Option<u32>,

    /// Optional terminal hole bound for `--target cell-count` or `--target low-board`.
    #[arg(long)]
    target_max_holes: Option<u32>,

    /// Number of random rollouts per sequence in rollout policy mode.
    #[arg(long, default_value_t = 100_000)]
    rollouts: u64,

    /// Base temperature for rollout sampling (lower = greedier).
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Explicit rollout temperatures. If omitted, a schedule is derived from `--temperature`.
    #[arg(long, value_delimiter = ',')]
    rollout_temperatures: Vec<f32>,

    /// Parallel rollout chunk size.
    #[arg(long, default_value_t = 5_000)]
    rollout_chunk_size: u64,
}

#[derive(Args)]
struct SplitCommandArgs {
    #[command(flatten)]
    constraints: BoardConstraintArgs,

    #[command(flatten)]
    trials: TrialSamplingArgs,

    /// Beam width for the first N-1 bags.
    #[arg(long, default_value_t = 1000)]
    beam_width: usize,

    /// Number of weighted rollouts per beam state for the last bag.
    #[arg(long, default_value_t = 100_000)]
    rollouts: u64,

    /// Base temperature for last-bag rollout sampling.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
}

#[derive(Args)]
struct SequenceCommandArgs {
    #[command(flatten)]
    constraints: BoardConstraintArgs,

    /// Comma-separated bag permutation indices.
    #[arg(long, value_name = "INDICES")]
    bags: String,
}

struct CycleRunConfig {
    max_height: u32,
    max_holes: u32,
    beam_width: usize,
    num_bags: usize,
    trials: u64,
    seed: u64,
    best_effort: bool,
    max_retries: usize,
    noise: f32,
    mcts_policy: MctsPolicy,
    mcts_iterations: u64,
    mcts_restarts: u64,
    mcts_exploration: f32,
    mcts_prior_temperature: f32,
    mcts_prior_noise: f32,
    mcts_max_children: usize,
    mcts_terminal_mode: MctsTerminalMode,
    mcts_terminal_repeat_penalty: f32,
    mcts_progress_every: u64,
    target: MctsTargetKind,
    target_cells: Vec<u32>,
    target_max_height: Option<u32>,
    target_max_holes: Option<u32>,
    rollouts: u64,
    temperature: f32,
    rollout_temperatures: Vec<f32>,
    rollout_chunk_size: u64,
}

impl CycleRunConfig {
    fn base(constraints: BoardConstraintArgs, trials: TrialSamplingArgs) -> Self {
        Self {
            max_height: constraints.max_height,
            max_holes: constraints.max_holes,
            beam_width: 1000,
            num_bags: trials.num_bags,
            trials: trials.trials,
            seed: trials.seed,
            best_effort: false,
            max_retries: 50,
            noise: 0.0,
            mcts_policy: MctsPolicy::Puct,
            mcts_iterations: 50_000,
            mcts_restarts: 1,
            mcts_exploration: 1.4,
            mcts_prior_temperature: 2.0,
            mcts_prior_noise: 0.0,
            mcts_max_children: 48,
            mcts_terminal_mode: MctsTerminalMode::Shaped,
            mcts_terminal_repeat_penalty: 0.05,
            mcts_progress_every: 25_000,
            target: MctsTargetKind::Empty,
            target_cells: Vec::new(),
            target_max_height: None,
            target_max_holes: None,
            rollouts: 100_000,
            temperature: 1.0,
            rollout_temperatures: Vec::new(),
            rollout_chunk_size: 5_000,
        }
    }

    fn sequence(constraints: BoardConstraintArgs) -> Self {
        Self::base(
            constraints,
            TrialSamplingArgs {
                num_bags: 5,
                trials: 1,
                seed: 42,
            },
        )
    }

    fn from_beam(args: &BeamCommandArgs) -> Self {
        let mut config = Self::base(args.constraints, args.trials);
        config.beam_width = args.beam_width;
        config.best_effort = args.best_effort;
        config.max_retries = args.max_retries;
        config.noise = args.noise;
        config
    }

    fn from_mcts(args: &MctsCommandArgs) -> Self {
        let mut config = Self::base(args.constraints, args.trials);
        config.mcts_policy = args.mcts_policy;
        config.mcts_iterations = args.mcts_iterations;
        config.mcts_restarts = args.mcts_restarts;
        config.mcts_exploration = args.mcts_exploration;
        config.mcts_prior_temperature = args.mcts_prior_temperature;
        config.mcts_prior_noise = args.mcts_prior_noise;
        config.mcts_max_children = args.mcts_max_children;
        config.mcts_terminal_mode = args.mcts_terminal_mode;
        config.mcts_terminal_repeat_penalty = args.mcts_terminal_repeat_penalty;
        config.mcts_progress_every = args.mcts_progress_every;
        config.target = args.target;
        config.target_cells = args.target_cells.clone();
        config.target_max_height = args.target_max_height;
        config.target_max_holes = args.target_max_holes;
        config.rollouts = args.rollouts;
        config.temperature = args.temperature;
        config.rollout_temperatures = args.rollout_temperatures.clone();
        config.rollout_chunk_size = args.rollout_chunk_size;
        config
    }

    fn from_split(args: &SplitCommandArgs) -> Self {
        let mut config = Self::base(args.constraints, args.trials);
        config.beam_width = args.beam_width;
        config.rollouts = args.rollouts;
        config.temperature = args.temperature;
        config
    }
}

pub fn run(args: FiveBagCycleArgs) -> Result<()> {
    let perms = generate_permutations();
    match args.command {
        FiveBagCycleCommand::Beam(command) => {
            run_beam(&CycleRunConfig::from_beam(&command), &perms)
        }
        FiveBagCycleCommand::Mcts(command) => {
            run_mcts(&CycleRunConfig::from_mcts(&command), &perms)
        }
        FiveBagCycleCommand::Split(command) => {
            run_split(&CycleRunConfig::from_split(&command), &perms)
        }
        FiveBagCycleCommand::Exact(command) => {
            let bag_perms = parse_bag_perms(&command.bags, perms.len())?;
            run_exact(
                &CycleRunConfig::sequence(command.constraints),
                &perms,
                bag_perms,
            )
        }
        FiveBagCycleCommand::Prove(command) => {
            let bag_perms = parse_bag_perms(&command.bags, perms.len())?;
            run_prove(
                &CycleRunConfig::sequence(command.constraints),
                &perms,
                &bag_perms,
            )
        }
    }
}

fn run_beam(args: &CycleRunConfig, perms: &[[TetrisPiece; 7]]) -> Result<()> {
    validate_random_trials(args)?;

    println!("five_bag_cycle (beam search, parallel)");
    println!("max_height    = {}", args.max_height);
    println!("max_holes     = {}", args.max_holes);
    println!("beam_width    = {}", args.beam_width);
    println!("best_effort   = {}", args.best_effort);
    if args.best_effort {
        println!("max_retries   = {}", args.max_retries);
    }
    println!("trials        = {}", args.trials);
    println!("seed          = {}", args.seed);
    println!("threads       = {}", rayon::current_num_threads());
    println!();

    let t0 = Instant::now();
    let successes = AtomicU64::new(0);
    let completed = AtomicU64::new(0);

    let num_bags = args.num_bags;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let trial_bags: Vec<Vec<usize>> = (0..args.trials)
        .map(|_| {
            (0..num_bags)
                .map(|_| rng.random_range(0..5040usize))
                .collect()
        })
        .collect();

    let results: Vec<(u64, Vec<usize>, u32, usize)> = trial_bags
        .par_iter()
        .enumerate()
        .filter_map(|(trial_idx, bag_perms)| {
            let pieces: Vec<TetrisPiece> = bag_perms
                .iter()
                .flat_map(|&pi| perms[pi].iter().copied())
                .collect();

            let noise = args.noise;
            let (min_cells, used_width) = if args.best_effort {
                best_effort_search(
                    &pieces,
                    args.beam_width,
                    args.max_retries,
                    args.max_height,
                    args.max_holes,
                    noise,
                )
            } else {
                let seed = if noise > 0.0 { 1 } else { 0 };
                let r = beam_search_to_empty_noisy(
                    &pieces,
                    args.beam_width,
                    args.max_height,
                    args.max_holes,
                    seed,
                    noise,
                );
                (r.min_cells, 0)
            };

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let succ = successes.load(Ordering::Relaxed);
                eprintln!(
                    "[{done:>6}/{}] successes={succ} rate={rate:.0}/s time={elapsed:.1}s",
                    args.trials,
                );
            }

            if min_cells == 0 {
                successes.fetch_add(1, Ordering::Relaxed);
            }
            Some((trial_idx as u64, bag_perms.clone(), min_cells, used_width))
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f64();
    let succ = successes.load(Ordering::Relaxed);
    let failures = args.trials - succ;

    // Separate successes and failures
    let success_results: Vec<_> = results.iter().filter(|r| r.2 == 0).collect();
    let fail_results: Vec<_> = results.iter().filter(|r| r.2 > 0).collect();

    println!();
    for (trial, bags, cells, width) in success_results.iter().take(3) {
        println!(
            "  SUCCESS trial={trial} bags=[{}] cells={cells} beam_w={width}",
            fmt_bags(&bags),
        );
    }
    if success_results.len() > 3 {
        println!("  ... ({} total successes) ...", success_results.len());
    }

    if !fail_results.is_empty() {
        let mut fail_cells: Vec<u32> = fail_results.iter().map(|r| r.2).collect();
        fail_cells.sort_unstable();
        let min_fail = fail_cells[0];
        let max_fail = fail_cells[fail_cells.len() - 1];
        let median_fail = fail_cells[fail_cells.len() / 2];
        let mean_fail = fail_cells.iter().map(|&c| c as f64).sum::<f64>() / fail_cells.len() as f64;
        println!();
        println!("failure min_cells distribution:");
        println!(
            "  min={min_fail} median={median_fail} mean={mean_fail:.1} max={max_fail} count={}",
            fail_cells.len()
        );
        for threshold in [2, 4, 6, 8, 10, 14, 18, 20, 28] {
            let count = fail_cells.iter().filter(|&&c| c <= threshold).count();
            if count > 0 {
                println!("  cells<={threshold}: {count}");
            }
        }
        println!();
        println!("failed bag sequences:");
        for (trial, bags, cells, width) in &fail_results {
            println!(
                "  FAIL trial={trial} bags=[{}] cells={cells} beam_w={width}",
                fmt_bags(&bags),
            );
        }
    }

    if !fail_results.is_empty() {
        println!();
        println!("still-failing bag sequences:");
        for (trial, bags, cells, retries) in &fail_results {
            println!(
                "  FAIL trial={trial} bags=[{}] cells={cells} retries={retries}",
                fmt_bags(&bags),
            );
        }
    }

    println!();
    println!("--- done ---");
    println!("trials        = {}", args.trials);
    println!("successes     = {succ}");
    println!("failures      = {failures}");
    println!(
        "success_rate  = {:.4}%",
        succ as f64 / args.trials as f64 * 100.0
    );
    println!("total_time    = {elapsed:.1}s");
    println!(
        "rate          = {:.0} trials/s",
        args.trials as f64 / elapsed
    );

    Ok(())
}

/// Best-effort: first try clean beam, then retry with increasing noise.
/// Each noisy retry explores a different part of the search space
/// at the same beam width — much cheaper than widening the beam.
fn best_effort_search(
    pieces: &[TetrisPiece],
    beam_width: usize,
    max_retries: usize,
    max_height: u32,
    max_holes: u32,
    base_noise: f32,
) -> (u32, usize) {
    // First: clean run (no noise unless user specified)
    let seed0 = if base_noise > 0.0 { 1 } else { 0 };
    let result =
        beam_search_to_empty_noisy(pieces, beam_width, max_height, max_holes, seed0, base_noise);
    if result.min_cells == 0 {
        return (0, 0);
    }

    // Retry with different noise seeds
    for retry in 1..=max_retries {
        let noise = if base_noise > 0.0 {
            base_noise
        } else {
            0.5 + (retry as f32) * 0.3
        };
        let result = beam_search_to_empty_noisy(
            pieces,
            beam_width,
            max_height,
            max_holes,
            retry as u64 + 1,
            noise,
        );
        if result.min_cells == 0 {
            return (0, retry);
        }
    }

    (result.min_cells, max_retries)
}

struct BeamResult {
    min_cells: u32,
}

/// Score a board for the empty-targeting objective. Lower = better.
///
/// Key insight: a row with 9/10 cells is about to clear — it's much better
/// than 9 scattered cells. We heavily discount cells in near-complete rows
/// and heavily penalize holes (which block line clears).
#[inline]
fn score_board(board: &TetrisBoard) -> f32 {
    let h = board.height() as usize;
    if h == 0 {
        return 0.0;
    }

    let limbs = board.as_limbs();

    // Per-column stats
    let mut holes: u32 = 0;
    let mut col_heights = [0u32; 10];
    for (i, col) in limbs.iter().enumerate() {
        col_heights[i] = u32::BITS - col.leading_zeros();
        holes += col_heights[i] - col.count_ones();
    }

    // Roughness
    let mut roughness: u32 = 0;
    for i in 0..9 {
        roughness += col_heights[i].abs_diff(col_heights[i + 1]);
    }

    // Per-row analysis: classify cells as "clearable" vs "stubborn"
    let mut clearable: u32 = 0;
    let mut stubborn: u32 = 0;
    for row in 0..h {
        let mut row_cells = 0u32;
        for col in 0..10 {
            if board.get_bit(col, row) {
                row_cells += 1;
            }
        }
        if row_cells >= 8 {
            // Near-complete row: these cells will likely clear
            clearable += row_cells;
        } else {
            stubborn += row_cells;
        }
    }

    // Stubborn cells are bad (hard to clear), clearable cells are OK
    stubborn as f32 * 2.0 + clearable as f32 * 0.2 + holes as f32 * 5.0 + roughness as f32 * 1.0
}

/// Beam search with temperature-based sampling for diversity.
/// - temperature = 0: deterministic top-K (standard beam search)
/// - temperature > 0: sample K states with softmax weighting
/// - noise_seed: different seeds explore different paths
fn beam_search_to_empty_noisy(
    pieces: &[TetrisPiece],
    beam_width: usize,
    max_height: u32,
    max_holes: u32,
    noise_seed: u64,
    temperature: f32,
) -> BeamResult {
    let empty = TetrisBoard::new();
    let mut beam: Vec<(f32, TetrisBoard)> = vec![(0.0, empty)];
    let mut rng = StdRng::seed_from_u64(noise_seed.wrapping_add(12345));

    for &piece in pieces {
        let mut next_beam: Vec<(f32, TetrisBoard)> = Vec::with_capacity(beam.len() * 17);

        for &(_, board) in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = board;
                let result = next.apply_piece_placement(placement);

                if result.is_lost == IsLost::LOST {
                    continue;
                }
                if next.height() > max_height {
                    continue;
                }
                if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
                    continue;
                }

                let score = score_board(&next);
                next_beam.push((score, next));
            }
        }

        if next_beam.is_empty() {
            return BeamResult {
                min_cells: u32::MAX,
            };
        }

        // Deduplicate first
        let mut seen = FxHashSet::default();
        next_beam.retain(|(_, board)| {
            let h = super::common::board_hash(board);
            seen.insert(h)
        });

        if temperature > 0.0 && next_beam.len() > beam_width {
            // Temperature-based sampling: sample beam_width states
            // using Gumbel-max trick (equivalent to softmax sampling)
            // score is a cost (lower=better), so negate for logits
            for entry in next_beam.iter_mut() {
                let logit = -entry.0 / temperature;
                // Gumbel noise: -ln(-ln(U)) where U ~ Uniform(0,1)
                let u: f64 = rng.random_range(1e-10f64..1.0f64);
                let gumbel = -((-u.ln()).ln()) as f32;
                entry.0 = -(logit + gumbel); // negate back so lower = better for sort
            }
            next_beam.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Deterministic top-K
            next_beam.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        next_beam.truncate(beam_width);

        // Recompute actual scores after sampling (undo Gumbel perturbation)
        if temperature > 0.0 {
            for entry in next_beam.iter_mut() {
                entry.0 = score_board(&entry.1);
            }
        }

        beam = next_beam;
    }

    let min_cells = beam
        .iter()
        .map(|(_, b)| b.count())
        .min()
        .unwrap_or(u32::MAX);

    BeamResult { min_cells }
}

/// Split mode: beam search bags 1-(N-1), then MCTS last bag from each beam state.
fn run_split(args: &CycleRunConfig, perms: &[[TetrisPiece; 7]]) -> Result<()> {
    validate_random_trials(args)?;

    let num_bags = args.num_bags;

    println!("five_bag_cycle SPLIT (beam + MCTS)");
    println!("max_height   = {}", args.max_height);
    println!("max_holes    = {}", args.max_holes);
    println!("beam_width   = {}", args.beam_width);
    println!(
        "rollouts     = {} (per beam state for last bag)",
        args.rollouts
    );
    println!("temperature  = {}", args.temperature);
    println!("num_bags     = {num_bags}");
    println!("trials       = {}", args.trials);
    println!("seed         = {}", args.seed);
    println!("threads      = {}", rayon::current_num_threads());
    println!();

    let mut rng = StdRng::seed_from_u64(args.seed);
    let trial_bags: Vec<Vec<usize>> = (0..args.trials)
        .map(|_| {
            (0..num_bags)
                .map(|_| rng.random_range(0..5040usize))
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let successes = AtomicU64::new(0);
    let completed = AtomicU64::new(0);

    let results: Vec<(u64, Vec<usize>, u32, usize)> = trial_bags
        .par_iter()
        .enumerate()
        .map(|(trial_idx, bag_perms)| {
            let all_pieces: Vec<TetrisPiece> = bag_perms
                .iter()
                .flat_map(|&pi| perms[pi].iter().copied())
                .collect();

            // Split: first (N-1) bags via beam, last bag via MCTS
            let split_at = (num_bags - 1) * 7;
            let beam_pieces = &all_pieces[..split_at];
            let last_bag_pieces = &all_pieces[split_at..];

            // Phase 1: beam search to get diverse boards after N-1 bags
            let beam = beam_search_collect(
                beam_pieces,
                args.beam_width,
                args.max_height,
                args.max_holes,
            );

            // Phase 2: from each beam state, MCTS the last bag targeting empty
            let mut found = false;
            let mut best_cells = u32::MAX;
            let mut boards_tried = 0usize;

            for &(_, board) in &beam {
                // Only try boards where reaching 0 is residue-compatible
                let c = board.count();
                if (c + 28) % 10 != 0 {
                    continue;
                }

                boards_tried += 1;
                let mut board_rng = StdRng::seed_from_u64(
                    args.seed
                        .wrapping_add(trial_idx as u64)
                        .wrapping_mul(0x517cc1b727220a95)
                        .wrapping_add(boards_tried as u64),
                );

                // MCTS rollouts for just the last 7 pieces
                for _rollout in 0..args.rollouts {
                    let cells = single_rollout_from(
                        board,
                        last_bag_pieces,
                        args.temperature,
                        args.max_height,
                        args.max_holes,
                        &mut board_rng,
                    );
                    if cells < best_cells {
                        best_cells = cells;
                    }
                    if cells == 0 {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if found {
                let s = successes.fetch_add(1, Ordering::Relaxed) + 1;
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  SUCCESS trial={trial_idx} bags=[{}] boards_tried={boards_tried} successes={s} time={elapsed:.1}s",
                    fmt_bags(bag_perms),
                );
            } else if done % 10 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let succ = successes.load(Ordering::Relaxed);
                eprintln!(
                    "[{done:>6}/{}] successes={succ} time={elapsed:.1}s",
                    args.trials,
                );
            }

            (trial_idx as u64, bag_perms.clone(), best_cells, boards_tried)
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f64();
    let succ = successes.load(Ordering::Relaxed);

    let success_results: Vec<_> = results.iter().filter(|r| r.2 == 0).collect();
    let fail_results: Vec<_> = results.iter().filter(|r| r.2 > 0).collect();

    println!();
    for (trial, bags, _, boards) in success_results.iter().take(5) {
        println!(
            "  SUCCESS trial={trial} bags=[{}] boards_tried={boards}",
            fmt_bags(bags),
        );
    }
    if success_results.len() > 5 {
        println!("  ... ({} total successes) ...", success_results.len());
    }

    if !fail_results.is_empty() {
        let mut fail_cells: Vec<u32> = fail_results.iter().map(|r| r.2).collect();
        fail_cells.sort_unstable();
        println!();
        println!(
            "failure cells: min={} median={} max={} count={}",
            fail_cells[0],
            fail_cells[fail_cells.len() / 2],
            fail_cells[fail_cells.len() - 1],
            fail_cells.len(),
        );
    }

    println!();
    println!("--- done ---");
    println!("trials       = {}", args.trials);
    println!("successes    = {succ}");
    println!("failures     = {}", args.trials as u64 - succ);
    println!(
        "success_rate = {:.4}%",
        succ as f64 / args.trials as f64 * 100.0
    );
    println!("total_time   = {elapsed:.1}s");

    Ok(())
}

/// Beam search that returns the full beam (not just min cells).
fn beam_search_collect(
    pieces: &[TetrisPiece],
    beam_width: usize,
    max_height: u32,
    max_holes: u32,
) -> Vec<(f32, TetrisBoard)> {
    let empty = TetrisBoard::new();
    let mut beam: Vec<(f32, TetrisBoard)> = vec![(0.0, empty)];

    for (step, &piece) in pieces.iter().enumerate() {
        let mut next_beam: Vec<(f32, TetrisBoard)> = Vec::with_capacity(beam.len() * 17);

        for &(_, board) in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = board;
                let result = next.apply_piece_placement(placement);

                if result.is_lost == IsLost::LOST {
                    continue;
                }
                if next.height() > max_height {
                    continue;
                }
                if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
                    continue;
                }

                // Use pacing-aware score
                let n_pieces = pieces.len();
                let lines_approx = ((step + 1) as u32 * 4 - next.count()) / 10;
                let score =
                    score_placement(&next, result.lines_cleared, lines_approx, step, n_pieces);
                next_beam.push((score, next));
            }
        }

        if next_beam.is_empty() {
            return beam;
        }

        next_beam
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut seen = FxHashSet::default();
        next_beam.retain(|(_, board)| {
            let h = super::common::board_hash(board);
            seen.insert(h)
        });

        next_beam.truncate(beam_width);
        beam = next_beam;
    }

    beam
}

/// MCTS rollout starting from a given board (not empty).
#[inline]
fn single_rollout_from(
    start: TetrisBoard,
    pieces: &[TetrisPiece],
    temperature: f32,
    max_height: u32,
    max_holes: u32,
    rng: &mut StdRng,
) -> u32 {
    let mut board = start;
    let n = pieces.len();

    for (step, &piece) in pieces.iter().enumerate() {
        let placements = TetrisPiecePlacement::all_from_piece(piece);

        let progress = step as f32 / n as f32;
        let temp = temperature * (1.0 + (1.0 - progress));

        let mut best_val = f32::NEG_INFINITY;
        let mut chosen_idx: Option<usize> = None;

        for (i, &placement) in placements.iter().enumerate() {
            let mut next = board;
            let result = next.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST {
                continue;
            }
            if next.height() > max_height {
                continue;
            }
            if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
                continue;
            }

            let score = score_board(&next);
            let logit = -score / temp;
            let u: f64 = rng.random_range(1e-10f64..1.0f64);
            let gumbel = -((-u.ln()).ln()) as f32;
            let val = logit + gumbel;
            if val > best_val {
                best_val = val;
                chosen_idx = Some(i);
            }
        }

        match chosen_idx {
            Some(idx) => {
                let _result = board.apply_piece_placement(placements[idx]);
            }
            None => return u32::MAX,
        }
    }

    board.count()
}

/// Exact bag-by-bag DFS for one specific 5-bag sequence.
/// No heuristic — enumerates all reachable boards at valid cell counts.
fn run_exact(
    args: &CycleRunConfig,
    perms: &[[TetrisPiece; 7]],
    bag_perms: Vec<usize>,
) -> Result<()> {
    if bag_perms.len() != 5 {
        bail!("exact mode expects exactly 5 bag permutation indices");
    }

    let empty = TetrisBoard::new();
    let empty_hash = super::common::board_hash(&empty);

    println!("five_bag_cycle EXACT DFS");
    println!("max_height  = {}", args.max_height);
    println!("max_holes   = {}", args.max_holes);
    println!("bags        = [{}]", fmt_bags(&bag_perms));
    println!();

    let t0 = Instant::now();
    let mut boards: Vec<TetrisBoard> = vec![empty];
    let mut total_nodes: u64 = 0;

    for bag_idx in 0..5 {
        let perm = &perms[bag_perms[bag_idx]];

        // Valid cell counts after this bag
        let target_cells: Vec<u32> = if bag_idx == 4 {
            vec![0]
        } else {
            let residue = ((bag_idx + 1) as u32 * 8) % 10;
            let max_c = 28 * (bag_idx as u32 + 1);
            (0..=max_c.min(60))
                .step_by(10)
                .map(|i| i + residue)
                .filter(|&c| c <= 60)
                .collect()
        };

        let n_sources = boards.len();
        let bag_start = Instant::now();
        let mut next_set: FxHashSet<u64> = FxHashSet::default();
        let mut next_boards: Vec<TetrisBoard> = Vec::new();

        for (src_idx, board) in boards.iter().enumerate() {
            // DFS targeting specific cell counts (prunes early)
            for &tc in &target_cells {
                let mut out: Vec<(u64, TetrisBoard)> = Vec::new();
                let mut nodes: u64 = 0;
                super::common::collect_reachable_targeted(
                    *board,
                    perm,
                    0,
                    args.max_height,
                    args.max_holes,
                    tc,
                    &mut out,
                    &mut nodes,
                );
                total_nodes += nodes;

                for (h, b) in out {
                    if next_set.insert(h) {
                        next_boards.push(b);
                    }
                }
            }

            if (src_idx + 1) % 1000 == 0 || src_idx + 1 == n_sources {
                let elapsed = bag_start.elapsed().as_secs_f64();
                println!(
                    "  bag {} [{}/{}] next={} nodes={} time={elapsed:.1}s",
                    bag_idx + 1,
                    src_idx + 1,
                    n_sources,
                    next_set.len(),
                    total_nodes,
                );
            }
        }

        let elapsed = bag_start.elapsed().as_secs_f64();
        let has_empty = next_set.contains(&empty_hash);
        println!(
            "bag {}: {} sources -> {} boards, target_cells={:?}, time={elapsed:.1}s, has_empty={}",
            bag_idx + 1,
            n_sources,
            next_set.len(),
            target_cells,
            has_empty,
        );

        if next_set.is_empty() {
            println!("DEAD at bag {}", bag_idx + 1);
            break;
        }

        boards = next_boards;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let has_empty = boards.iter().any(|b| b.count() == 0);
    println!();
    println!("--- result ---");
    println!("final_boards  = {}", boards.len());
    println!("has_empty     = {has_empty}");
    println!("total_nodes   = {total_nodes}");
    println!("total_time    = {elapsed:.1}s");

    if has_empty {
        println!("VERDICT: this 5-bag sequence CAN return to empty.");
    } else {
        println!(
            "VERDICT: this 5-bag sequence CANNOT return to empty (within h<={}/holes<={}).",
            args.max_height, args.max_holes
        );
    }

    Ok(())
}

fn run_mcts(args: &CycleRunConfig, perms: &[[TetrisPiece; 7]]) -> Result<()> {
    let target = TargetSpec::from_cli(
        args.target,
        args.target_cells.clone(),
        args.target_max_height,
        args.target_max_holes,
    )?;

    match args.mcts_policy {
        MctsPolicy::Puct => run_puct_mcts(args, perms, target),
        MctsPolicy::Rollout => {
            if target.kind() != MctsTargetKind::Empty {
                bail!("--mcts-policy rollout currently supports only --target empty");
            }
            run_rollout_mcts(args, perms)
        }
    }
}

/// PUCT tree search with a transposition table.
///
/// For each fixed N-bag sequence, this runs a single MCTS tree from the empty
/// board. The tree policy uses board-scoring priors to bias expansion while
/// still backing up terminal target scores through repeated visits.
fn run_puct_mcts(
    args: &CycleRunConfig,
    perms: &[[TetrisPiece; 7]],
    target: TargetSpec,
) -> Result<()> {
    validate_random_trials(args)?;
    if args.mcts_restarts == 0 {
        bail!("--mcts-restarts must be greater than zero");
    }

    let constraints = BoardConstraints::new(args.max_height, args.max_holes);
    let base_config = PuctConfig {
        start_board: TetrisBoard::new(),
        constraints,
        target,
        iterations: args.mcts_iterations,
        exploration: args.mcts_exploration,
        prior_temperature: args.mcts_prior_temperature,
        prior_noise: args.mcts_prior_noise,
        max_children: args.mcts_max_children,
        terminal_mode: args.mcts_terminal_mode,
        terminal_repeat_penalty: args.mcts_terminal_repeat_penalty,
        noise_seed: args.seed,
    };
    base_config.validate(args.num_bags * 7)?;

    println!("five_bag_cycle MCTS (PUCT tree search)");
    println!("max_height            = {}", constraints.max_height());
    println!("max_holes             = {}", constraints.max_holes());
    println!("target                = {}", base_config.target.describe());
    println!("mcts_iterations       = {}", args.mcts_iterations);
    println!("mcts_restarts         = {}", args.mcts_restarts);
    println!("mcts_exploration      = {}", args.mcts_exploration);
    println!("mcts_prior_temperature= {}", args.mcts_prior_temperature);
    println!("mcts_prior_noise      = {}", args.mcts_prior_noise);
    println!("mcts_max_children     = {}", args.mcts_max_children);
    println!("mcts_terminal_mode    = {:?}", args.mcts_terminal_mode);
    println!(
        "mcts_terminal_repeat_penalty = {}",
        args.mcts_terminal_repeat_penalty
    );
    println!("mcts_progress_every   = {}", args.mcts_progress_every);
    println!("num_bags              = {}", args.num_bags);
    println!("trials                = {}", args.trials);
    println!("seed                  = {}", args.seed);
    println!("threads               = {}", rayon::current_num_threads());
    println!();

    let mut rng = StdRng::seed_from_u64(args.seed);
    let trial_bags: Vec<Vec<usize>> = (0..args.trials)
        .map(|_| {
            (0..args.num_bags)
                .map(|_| rng.random_range(0..perms.len()))
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let successes = AtomicU64::new(0);
    let completed = AtomicU64::new(0);

    let results: Vec<(u64, Vec<usize>, super::five_bag_mcts::PuctSearchResult)> = trial_bags
        .par_iter()
        .enumerate()
        .map(|(trial_idx, bag_perms)| {
            let pieces: Vec<TetrisPiece> = bag_perms
                .iter()
                .flat_map(|&pi| perms[pi].iter().copied())
                .collect();
            let trial_start = Instant::now();
            let bag_label = fmt_bags(bag_perms);
            let mut result = empty_puct_result();

            for restart in 0..args.mcts_restarts {
                let mut restart_config = base_config.clone();
                restart_config.noise_seed = args
                    .seed
                    .wrapping_add((trial_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    .wrapping_add(restart.wrapping_mul(0xbf58_476d_1ce4_e5b9));

                let restart_result = run_puct_search_with_progress(
                    &pieces,
                    &restart_config,
                    args.mcts_progress_every,
                    |progress| {
                        let elapsed = t0.elapsed().as_secs_f64();
                        let trial_elapsed = trial_start.elapsed().as_secs_f64();
                        let done = completed.load(Ordering::Relaxed);
                        let succ = successes.load(Ordering::Relaxed);
                        eprintln!(
                            "  progress trial={trial_idx} restart={}/{} bags=[{bag_label}] iter={}/{} nodes={} terminal_evals={} best_cells={} best_score={:.3} done={}/{} successes={} trial_time={trial_elapsed:.1}s total_time={elapsed:.1}s",
                            restart + 1,
                            args.mcts_restarts,
                            progress.iterations,
                            args.mcts_iterations,
                            progress.tree_nodes,
                            progress.terminal_evaluations,
                            fmt_optional_cells(progress.best_cells),
                            progress.best_score,
                            done,
                            args.trials,
                            succ,
                        );
                    },
                );
                let restart_success = restart_result.success;
                merge_puct_result(&mut result, restart_result);
                if restart_success {
                    break;
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;

            if result.success {
                let s = successes.fetch_add(1, Ordering::Relaxed) + 1;
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  SUCCESS trial={trial_idx} bags=[{}] iterations={} nodes={} successes={s} time={elapsed:.1}s",
                    bag_label,
                    result.iterations,
                    result.tree_nodes,
                );
            } else if done % 10 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let succ = successes.load(Ordering::Relaxed);
                let rate = done as f64 / elapsed;
                eprintln!(
                    "[{done:>6}/{}] successes={succ} best_cells={} nodes={} rate={rate:.1}/s time={elapsed:.1}s",
                    args.trials,
                    result.best_cells,
                    result.tree_nodes,
                );
            }

            (trial_idx as u64, bag_perms.clone(), result)
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f64();
    let succ = successes.load(Ordering::Relaxed);
    let success_results: Vec<_> = results.iter().filter(|r| r.2.success).collect();
    let fail_results: Vec<_> = results.iter().filter(|r| !r.2.success).collect();

    println!();
    for (trial, bags, result) in success_results.iter().take(5) {
        println!(
            "  SUCCESS trial={trial} bags=[{}] iterations={} nodes={} terminal_evals={}",
            fmt_bags(bags),
            result.iterations,
            result.tree_nodes,
            result.terminal_evaluations,
        );
    }
    if success_results.len() > 5 {
        println!("  ... ({} total successes) ...", success_results.len());
    }

    if !fail_results.is_empty() {
        let mut fail_cells: Vec<u32> = fail_results.iter().map(|r| r.2.best_cells).collect();
        fail_cells.sort_unstable();
        let min_f = fail_cells[0];
        let max_f = fail_cells[fail_cells.len() - 1];
        let med_f = fail_cells[fail_cells.len() / 2];
        let mean_nodes = fail_results
            .iter()
            .map(|r| r.2.tree_nodes as f64)
            .sum::<f64>()
            / fail_results.len() as f64;
        println!();
        println!(
            "failure cells: min={min_f} median={med_f} max={max_f} count={}",
            fail_cells.len()
        );
        println!("failure mean_tree_nodes={mean_nodes:.1}");
    }

    let total_nodes: usize = results.iter().map(|r| r.2.tree_nodes).sum();
    let total_terminal_evals: u64 = results.iter().map(|r| r.2.terminal_evaluations).sum();

    println!();
    println!("--- done ---");
    println!("trials              = {}", args.trials);
    println!("successes           = {succ}");
    println!("failures            = {}", args.trials - succ);
    println!(
        "success_rate        = {:.4}%",
        succ as f64 / args.trials as f64 * 100.0
    );
    println!("total_tree_nodes    = {total_nodes}");
    println!("terminal_evals      = {total_terminal_evals}");
    println!("total_time          = {elapsed:.1}s");

    Ok(())
}

/// Legacy MCTS-style parallel weighted random rollouts.
///
/// For each N-bag sequence, run independent rollouts where each piece placement
/// is sampled with probability proportional to `exp(-score / temperature)`.
fn run_rollout_mcts(args: &CycleRunConfig, perms: &[[TetrisPiece; 7]]) -> Result<()> {
    validate_random_trials(args)?;
    let temperatures = rollout_temperatures(args)?;

    println!("five_bag_cycle MCTS (parallel rollouts baseline)");
    println!("max_height   = {}", args.max_height);
    println!("max_holes    = {}", args.max_holes);
    println!("rollouts     = {}", args.rollouts);
    println!("temperatures = {:?}", temperatures);
    println!("chunk_size   = {}", args.rollout_chunk_size);
    println!("num_bags     = {}", args.num_bags);
    println!("trials       = {}", args.trials);
    println!("seed         = {}", args.seed);
    println!("threads      = {}", rayon::current_num_threads());
    println!();

    let num_bags = args.num_bags;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let trial_bags: Vec<Vec<usize>> = (0..args.trials)
        .map(|_| {
            (0..num_bags)
                .map(|_| rng.random_range(0..5040usize))
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let successes = AtomicU64::new(0);
    let completed = AtomicU64::new(0);

    let results: Vec<(u64, Vec<usize>, u32, u64)> = trial_bags
        .par_iter()
        .enumerate()
        .map(|(trial_idx, bag_perms)| {
            let pieces: Vec<TetrisPiece> = bag_perms
                .iter()
                .flat_map(|&pi| perms[pi].iter().copied())
                .collect();

            let (min_cells, winning_rollout) = mcts_rollouts(
                &pieces,
                args.rollouts,
                &temperatures,
                args.rollout_chunk_size,
                args.max_height,
                args.max_holes,
                args.seed.wrapping_add(trial_idx as u64),
            );

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;

            if min_cells == 0 {
                let s = successes.fetch_add(1, Ordering::Relaxed) + 1;
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  SUCCESS trial={trial_idx} bags=[{}] rollout={winning_rollout} successes={s} time={elapsed:.1}s",
                    fmt_bags(bag_perms),
                );
            } else if done % 10 == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let succ = successes.load(Ordering::Relaxed);
                let rate = done as f64 / elapsed;
                eprintln!(
                    "[{done:>6}/{}] successes={succ} best_fail_cells={min_cells} rate={rate:.1}/s time={elapsed:.1}s",
                    args.trials,
                );
            }

            (trial_idx as u64, bag_perms.clone(), min_cells, winning_rollout)
        })
        .collect();

    let elapsed = t0.elapsed().as_secs_f64();
    let succ = successes.load(Ordering::Relaxed);

    // Report
    let success_results: Vec<_> = results.iter().filter(|r| r.2 == 0).collect();
    let fail_results: Vec<_> = results.iter().filter(|r| r.2 > 0).collect();

    println!();
    for (trial, bags, _, rollout) in success_results.iter().take(5) {
        println!(
            "  SUCCESS trial={trial} bags=[{}] found_at_rollout={rollout}",
            fmt_bags(&bags),
        );
    }
    if success_results.len() > 5 {
        println!("  ... ({} total successes) ...", success_results.len());
    }

    if !fail_results.is_empty() {
        let mut fail_cells: Vec<u32> = fail_results.iter().map(|r| r.2).collect();
        fail_cells.sort_unstable();
        let min_f = fail_cells[0];
        let max_f = fail_cells[fail_cells.len() - 1];
        let med_f = fail_cells[fail_cells.len() / 2];
        println!();
        println!(
            "failure cells: min={min_f} median={med_f} max={max_f} count={}",
            fail_cells.len()
        );
    }

    println!();
    println!("--- done ---");
    println!("trials       = {}", args.trials);
    println!("successes    = {succ}");
    println!("failures     = {}", args.trials as u64 - succ);
    println!(
        "success_rate = {:.4}%",
        succ as f64 / args.trials as f64 * 100.0
    );
    println!("total_time   = {elapsed:.1}s");

    Ok(())
}

fn rollout_temperatures(args: &CycleRunConfig) -> Result<Vec<f32>> {
    if args.rollouts == 0 {
        bail!("--rollouts must be greater than zero");
    }
    if args.rollout_chunk_size == 0 {
        bail!("--rollout-chunk-size must be greater than zero");
    }

    let temperatures = if args.rollout_temperatures.is_empty() {
        vec![
            args.temperature * 0.3,
            args.temperature * 0.5,
            args.temperature * 0.7,
            args.temperature,
            args.temperature * 1.5,
            args.temperature * 2.0,
            args.temperature * 3.0,
        ]
    } else {
        args.rollout_temperatures.clone()
    };

    for temperature in &temperatures {
        if !temperature.is_finite() || *temperature <= 0.0 {
            bail!("rollout temperatures must be finite and positive");
        }
    }

    Ok(temperatures)
}

/// Run `n_rollouts` weighted random rollouts in parallel chunks.
/// Splits budget across multiple temperatures for diversity.
/// Returns (min_cells_achieved, rollout_index_of_first_success).
fn mcts_rollouts(
    pieces: &[TetrisPiece],
    n_rollouts: u64,
    temperatures: &[f32],
    chunk_size: u64,
    max_height: u32,
    max_holes: u32,
    seed: u64,
) -> (u32, u64) {
    let found = AtomicU64::new(u64::MAX);
    let best_cells = std::sync::atomic::AtomicU32::new(u32::MAX);

    for (temp_idx, &temp) in temperatures.iter().enumerate() {
        if found.load(Ordering::Relaxed) < u64::MAX {
            break;
        }

        let temp_start = n_rollouts * temp_idx as u64 / temperatures.len() as u64;
        let temp_end = n_rollouts * (temp_idx as u64 + 1) / temperatures.len() as u64;
        let n_for_temp = temp_end - temp_start;
        if n_for_temp == 0 {
            continue;
        }
        let n_chunks = n_for_temp.div_ceil(chunk_size);

        (0..n_chunks).into_par_iter().for_each(|chunk_idx| {
            if found.load(Ordering::Relaxed) < u64::MAX {
                return;
            }

            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n_for_temp);
            let mut rng = StdRng::seed_from_u64(
                seed.wrapping_add(temp_start)
                    .wrapping_add(chunk_idx)
                    .wrapping_mul(0x517cc1b727220a95),
            );

            for rollout_in_chunk in chunk_start..chunk_end {
                if found.load(Ordering::Relaxed) < u64::MAX {
                    return;
                }

                let cells = single_rollout(pieces, temp, max_height, max_holes, &mut rng);

                if cells < best_cells.load(Ordering::Relaxed) {
                    best_cells.fetch_min(cells, Ordering::Relaxed);
                }
                if cells == 0 {
                    let global_rollout = temp_start + rollout_in_chunk;
                    found.fetch_min(global_rollout, Ordering::Relaxed);
                    return;
                }
            }
        });
    }

    let f = found.load(Ordering::Relaxed);
    let b = best_cells.load(Ordering::Relaxed);
    if f < u64::MAX {
        (0, f)
    } else {
        (b, n_rollouts)
    }
}

/// Score a placement considering pacing toward the line-clear target.
///
/// `lines_cleared`: lines cleared so far in this rollout
/// `step`: which piece we're placing (0..n_pieces)
/// `n_pieces`: total pieces in the sequence
#[inline]
fn score_placement(
    board: &TetrisBoard,
    lines_just_cleared: u32,
    total_lines_cleared: u32,
    step: usize,
    n_pieces: usize,
) -> f32 {
    let cells = board.count();
    if cells == 0 && step == n_pieces {
        return -1000.0; // perfect — lowest possible score
    }

    let h = board.height() as usize;
    let limbs = board.as_limbs();

    // Holes
    let mut holes: u32 = 0;
    let mut col_heights = [0u32; 10];
    for (i, col) in limbs.iter().enumerate() {
        col_heights[i] = u32::BITS - col.leading_zeros();
        holes += col_heights[i] - col.count_ones();
    }

    // Roughness
    let mut roughness: u32 = 0;
    for i in 0..9 {
        roughness += col_heights[i].abs_diff(col_heights[i + 1]);
    }

    // Row completeness analysis
    let mut clearable: u32 = 0;
    let mut stubborn: u32 = 0;
    for row in 0..h {
        let mut row_cells = 0u32;
        for col in 0..10 {
            if board.get_bit(col, row) {
                row_cells += 1;
            }
        }
        if row_cells >= 8 {
            clearable += row_cells;
        } else {
            stubborn += row_cells;
        }
    }

    // Pacing: are we on track for the required line clears?
    let target_lines = (n_pieces as u32 * 4) / 10; // total lines needed
    let progress = (step + 1) as f32 / n_pieces as f32;
    let expected_lines = target_lines as f32 * progress;
    let line_deficit = expected_lines - total_lines_cleared as f32;

    // Line clear bonus: reward placements that clear lines, especially when behind
    let clear_bonus = if lines_just_cleared > 0 {
        let urgency = if line_deficit > 2.0 {
            3.0
        } else if line_deficit > 0.0 {
            1.5
        } else {
            0.5
        };
        -(lines_just_cleared as f32) * urgency * 5.0
    } else {
        0.0
    };

    // Deficit penalty: if we're behind on line clears, increase urgency
    let deficit_penalty = if line_deficit > 0.0 {
        line_deficit * 2.0
    } else {
        0.0
    };

    stubborn as f32 * 2.0
        + clearable as f32 * 0.2
        + holes as f32 * 5.0
        + roughness as f32 * 1.0
        + deficit_penalty
        + clear_bonus
}

/// One weighted random rollout with pacing-aware scoring and temperature annealing.
#[inline]
fn single_rollout(
    pieces: &[TetrisPiece],
    temperature: f32,
    max_height: u32,
    max_holes: u32,
    rng: &mut StdRng,
) -> u32 {
    let mut board = TetrisBoard::new();
    let n_pieces = pieces.len();
    let mut total_lines_cleared: u32 = 0;

    for (step, &piece) in pieces.iter().enumerate() {
        let placements = TetrisPiecePlacement::all_from_piece(piece);

        // Temperature annealing: warm early (explore), cool late (exploit)
        let progress = step as f32 / n_pieces as f32;
        let temp = temperature * (1.0 + 2.0 * (1.0 - progress)); // 3x at start, 1x at end

        // Gumbel-max sampling
        let mut best_val = f32::NEG_INFINITY;
        let mut chosen_idx: Option<usize> = None;
        let mut chosen_lines: u32 = 0;

        for (i, &placement) in placements.iter().enumerate() {
            let mut next = board;
            let result = next.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST {
                continue;
            }
            if next.height() > max_height {
                continue;
            }
            if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
                continue;
            }

            let lines = result.lines_cleared;
            let score = score_placement(&next, lines, total_lines_cleared + lines, step, n_pieces);
            let logit = -score / temp;
            let u: f64 = rng.random_range(1e-10f64..1.0f64);
            let gumbel = -((-u.ln()).ln()) as f32;
            let val = logit + gumbel;
            if val > best_val {
                best_val = val;
                chosen_idx = Some(i);
                chosen_lines = lines;
            }
        }

        match chosen_idx {
            Some(idx) => {
                let _result = board.apply_piece_placement(placements[idx]);
                total_lines_cleared += chosen_lines;
            }
            None => return u32::MAX, // dead
        }
    }

    board.count()
}

// ---------------------------------------------------------------------------
// Prove mode: exhaustive memoized DFS
// ---------------------------------------------------------------------------

/// Exhaustive DFS over the full N*7 piece sequence.
/// Proves whether ANY placement sequence can reach the empty board.
fn run_prove(args: &CycleRunConfig, perms: &[[TetrisPiece; 7]], bag_perms: &[usize]) -> Result<()> {
    let pieces: Vec<TetrisPiece> = bag_perms
        .iter()
        .flat_map(|&pi| perms[pi].iter().copied())
        .collect();
    let n_pieces = pieces.len();

    println!("five_bag_cycle PROVE (exhaustive DFS)");
    println!("max_height  = {}", args.max_height);
    println!("max_holes   = {}", args.max_holes);
    println!("bags        = [{}]", fmt_bags(bag_perms));
    println!("pieces      = {n_pieces}");
    println!("lines needed= {}", n_pieces as u32 * 4 / 10);
    println!();

    let t0 = Instant::now();
    let nodes = AtomicU64::new(0);

    // Use a packed key: board bytes (40 bytes) + step (1 byte)
    let mut memo: FxHashSet<(super::common::BoardKey, u8)> = FxHashSet::default();

    let found = prove_dfs(
        TetrisBoard::new(),
        &pieces,
        0,
        args.max_height,
        args.max_holes,
        &mut memo,
        &nodes,
        &t0,
    );

    let elapsed = t0.elapsed().as_secs_f64();
    let total_nodes = nodes.load(Ordering::Relaxed);
    let total_memo = memo.len();

    println!();
    println!("--- result ---");
    println!("nodes         = {total_nodes}");
    println!("memo_entries  = {total_memo}");
    println!("total_time    = {elapsed:.1}s");

    if found {
        println!("PROVED POSSIBLE: a placement sequence exists that returns to empty.");
    } else {
        println!("PROVED IMPOSSIBLE: NO placement sequence can return to empty");
        println!(
            "  (within h<={}, holes<={})",
            args.max_height, args.max_holes
        );
    }

    Ok(())
}

/// Recursive DFS: can we reach the empty board from `board` by placing
/// pieces[step..] in order?
fn prove_dfs(
    board: TetrisBoard,
    pieces: &[TetrisPiece],
    step: usize,
    max_height: u32,
    max_holes: u32,
    memo: &mut FxHashSet<(super::common::BoardKey, u8)>,
    nodes: &AtomicU64,
    t0: &Instant,
) -> bool {
    let n = pieces.len();

    // Terminal: all pieces placed
    if step == n {
        return board.count() == 0;
    }

    // Cell-count feasibility pruning:
    // At step k with cells c, remaining pieces = n-k.
    // Final cells = c + 4*(n-k) - 10*future_lines
    // For final=0: future_lines = (c + 4*(n-k)) / 10
    // This must be a non-negative integer.
    let c = board.count();
    let remaining_cells = 4 * (n - step) as u32;
    let total_future = c + remaining_cells;
    if total_future % 10 != 0 {
        return false; // residue mismatch, impossible
    }
    let lines_needed = total_future / 10;
    // Upper bound on lines clearable: at most (n-step) lines (one per piece, very generous)
    // More realistic: can't clear more than remaining_cells/10 + current_height lines
    // But the generous bound is fine for pruning
    if lines_needed > (n - step) as u32 {
        return false; // need more lines than pieces remaining
    }

    // Memo check
    let key = (super::common::board_key(&board), step as u8);
    if memo.contains(&key) {
        return false; // already explored, no solution from here
    }

    // Progress logging
    let n_so_far = nodes.fetch_add(1, Ordering::Relaxed) + 1;
    if n_so_far % 10_000_000 == 0 {
        let elapsed = t0.elapsed().as_secs_f64();
        let rate = n_so_far as f64 / elapsed;
        eprintln!(
            "  step={step}/{n} cells={c} nodes={n_so_far} memo={} rate={rate:.0}/s time={elapsed:.1}s",
            memo.len(),
        );
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        if prove_dfs(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            memo,
            nodes,
            t0,
        ) {
            return true; // found a solution!
        }
    }

    // No placement worked — memoize failure
    memo.insert(key);
    false
}
