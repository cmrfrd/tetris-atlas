//! # Tetris Controlled-Invariant Synthesis (MVP)
//!
//! Builds an abstract safety game from sampled concrete states and solves for a
//! controlled invariant set via fixed-point elimination.
//!
//! This is an approximation of the game-theoretic condition:
//! for every piece draw, there exists a placement that keeps the state safe.

use clap::{Parser, ValueEnum};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

#[derive(Parser, Debug, Clone)]
#[command(name = "tetris-invariant-synthesis")]
#[command(about = "Approximate controlled-invariant synthesis for Tetris")]
struct Cli {
    /// Search mode: one-shot (existing behavior) or incremental round-based search.
    #[arg(long, value_enum, default_value_t = SearchMode::OneShot)]
    mode: SearchMode,

    /// Maximum concrete (board,bag) states to sample.
    #[arg(long, default_value_t = 150_000)]
    max_concrete_states: usize,

    /// Maximum BFS expansion depth while sampling concrete states.
    #[arg(long, default_value_t = 22)]
    max_sampling_depth: usize,

    /// Number of representatives kept per abstract bucket.
    #[arg(long, default_value_t = 16)]
    max_representatives_per_abstract: usize,

    /// Number of refinement rounds (bucket tightening on failure).
    #[arg(long, default_value_t = 3)]
    max_refinement_rounds: usize,

    /// Number of rollout seeds for falsification.
    #[arg(long, default_value_t = 64)]
    falsify_rollouts: usize,

    /// Horizon per rollout.
    #[arg(long, default_value_t = 30_000)]
    falsify_horizon: usize,

    /// Initial max-height bucket size.
    #[arg(long, default_value_t = 3)]
    height_bucket: u32,

    /// Initial total-holes bucket size.
    #[arg(long, default_value_t = 2)]
    holes_bucket: u32,

    /// Initial roughness bucket size.
    #[arg(long, default_value_t = 3)]
    roughness_bucket: u32,

    /// How to treat transitions that land in unseen abstract buckets.
    #[arg(long, value_enum, default_value_t = UnknownSuccessorPolicy::Ignore)]
    unknown_successor_policy: UnknownSuccessorPolicy,

    /// How representative checks are quantified per abstract bucket.
    #[arg(long, value_enum, default_value_t = RepresentativeQuantifier::Exists)]
    representative_quantifier: RepresentativeQuantifier,

    /// Output CSV path for the synthesized abstract policy map.
    #[arg(long, default_value = "tetris_invariant_policy_map.csv")]
    policy_map_csv: String,

    /// Number of incremental rounds.
    #[arg(long, default_value_t = 20)]
    max_rounds: usize,

    /// Number of frontier states to expand per incremental round.
    #[arg(long, default_value_t = 50_000)]
    expand_steps_per_round: usize,

    /// Stop incremental search if this many kept abstract states are reached.
    #[arg(long)]
    target_kept_states: Option<usize>,

    /// Stop incremental search if an SCC of at least this size is found.
    #[arg(long, default_value_t = 32)]
    target_min_cycle_size: usize,

    /// Directory where incremental artifacts are written.
    #[arg(long, default_value = "/tmp")]
    artifact_dir: String,

    /// In-memory frontier scheduling policy.
    #[arg(long, value_enum, default_value_t = FrontierOrder::Fifo)]
    frontier_order: FrontierOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum UnknownSuccessorPolicy {
    Unsafe,
    Ignore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RepresentativeQuantifier {
    Forall,
    Exists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SearchMode {
    OneShot,
    Incremental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FrontierOrder {
    Fifo,
    Height,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConcreteState {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AbstractState {
    bag_mask: u8,
    max_height_bucket: u8,
    holes_bucket: u8,
    roughness_bucket: u8,
}

#[derive(Debug, Clone)]
struct AbstractionConfig {
    height_bucket: u32,
    holes_bucket: u32,
    roughness_bucket: u32,
}

impl AbstractionConfig {
    fn refined(&self) -> Self {
        Self {
            height_bucket: self.height_bucket.saturating_sub(1).max(1),
            holes_bucket: self.holes_bucket.saturating_sub(1).max(1),
            roughness_bucket: self.roughness_bucket.saturating_sub(1).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CachedTransition {
    next: ConcreteState,
}

#[derive(Debug)]
struct IncrementalState {
    visited_concrete: HashSet<ConcreteState>,
    frontier: VecDeque<ConcreteState>,
    transition_cache: HashMap<ConcreteState, Vec<CachedTransition>>,
}

impl IncrementalState {
    fn new() -> Self {
        let root = ConcreteState {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::new(),
        };
        let mut visited = HashSet::new();
        visited.insert(root);
        let mut frontier = VecDeque::new();
        frontier.push_back(root);
        Self {
            visited_concrete: visited,
            frontier,
            transition_cache: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct TransitionCoverage {
    known_successors: usize,
    unknown_successors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoverageKey {
    state: AbstractState,
    piece: TetrisPiece,
}

#[derive(Debug, Clone)]
struct RoundStats {
    round: usize,
    newly_expanded: usize,
    total_concrete: usize,
    total_abstract: usize,
    kept_abstract: usize,
    complete_policy_states: usize,
    incomplete_policy_states: usize,
    scc_count: usize,
    largest_scc: usize,
    reachable_cycle_count: usize,
    falsify_successes: usize,
    falsify_failures: usize,
    falsify_earliest_failure_step: Option<usize>,
}

#[derive(Debug, Clone)]
struct CycleCandidate {
    id: usize,
    size: usize,
    all_piece_closed_states: usize,
    all_piece_closure_ratio: f64,
    reachable_from_empty: bool,
}

fn main() {
    let cli = Cli::parse();
    match cli.mode {
        SearchMode::OneShot => run_tetris_invariant_synthesis(cli),
        SearchMode::Incremental => run_tetris_invariant_incremental(cli),
    }
}

fn run_tetris_invariant_incremental(cli: Cli) {
    let start = Instant::now();
    println!("=== Tetris Controlled-Invariant Synthesis (Incremental) ===");
    println!(
        "Rounds={} expand_steps_per_round={} max_concrete_states={} frontier_order={:?}",
        cli.max_rounds, cli.expand_steps_per_round, cli.max_concrete_states, cli.frontier_order
    );
    println!(
        "Solver: unknown_successor={:?} representative_quantifier={:?}",
        cli.unknown_successor_policy, cli.representative_quantifier
    );
    println!("Artifacts: {}", cli.artifact_dir);

    let mut incremental = IncrementalState::new();
    let mut abstraction = AbstractionConfig {
        height_bucket: cli.height_bucket.max(1),
        holes_bucket: cli.holes_bucket.max(1),
        roughness_bucket: cli.roughness_bucket.max(1),
    };
    let mut best_kept = 0usize;

    ensure_artifact_dir(&cli.artifact_dir);
    let stats_csv_path = format!("{}/tetris_invariant_round_stats.csv", cli.artifact_dir);
    initialize_round_stats_csv(&stats_csv_path);

    for round in 1..=cli.max_rounds {
        let newly_expanded = expand_incremental_round(
            &mut incremental,
            cli.expand_steps_per_round,
            cli.max_concrete_states,
            cli.frontier_order,
        );

        let concrete_states: Vec<ConcreteState> =
            incremental.visited_concrete.iter().copied().collect();
        let mut best_round_cfg = abstraction.clone();
        let mut best_round_solved: Option<SolveResult> = None;
        let mut best_round_falsify: Option<FalsifyResult> = None;
        let mut best_round_kept = 0usize;

        let mut refinement_cfg = abstraction.clone();
        for _ in 0..cli.max_refinement_rounds {
            let model = build_abstract_model(
                &concrete_states,
                &refinement_cfg,
                cli.max_representatives_per_abstract,
            );
            let solved = solve_controlled_invariant(
                &model,
                &refinement_cfg,
                cli.unknown_successor_policy,
                cli.representative_quantifier,
            );
            let falsify = falsify_with_rollouts(
                &solved,
                &refinement_cfg,
                cli.falsify_rollouts,
                cli.falsify_horizon,
                42,
            );
            if solved.kept_states.len() >= best_round_kept {
                best_round_kept = solved.kept_states.len();
                best_round_cfg = refinement_cfg.clone();
                best_round_solved = Some(solved);
                best_round_falsify = Some(falsify);
            }
            refinement_cfg = refinement_cfg.refined();
        }

        let Some(solved) = best_round_solved else {
            println!("Round {}: no solve result, stopping", round);
            break;
        };
        let falsify = best_round_falsify.expect("falsify exists when solved exists");
        let model = build_abstract_model(
            &concrete_states,
            &best_round_cfg,
            cli.max_representatives_per_abstract,
        );
        let policy = build_abstract_policy_map(
            &model,
            &solved,
            &best_round_cfg,
            cli.unknown_successor_policy,
        );

        let coverage =
            compute_transition_coverage(&model, &best_round_cfg, cli.unknown_successor_policy);
        let graph = build_policy_graph(&model, &solved, &policy, &best_round_cfg);
        let sccs = strongly_connected_components(&graph, solved.kept_states.len());
        let empty_abs = abstract_state(
            ConcreteState {
                board: TetrisBoard::new(),
                bag: TetrisPieceBagState::new(),
            },
            &best_round_cfg,
        );
        let cycles =
            rank_cycle_candidates(&sccs, &graph, &model, &policy, &best_round_cfg, empty_abs);
        let reachable_cycle_count = cycles.iter().filter(|c| c.reachable_from_empty).count();

        let largest_scc = sccs.iter().map(std::vec::Vec::len).max().unwrap_or(0);
        let round_stats = RoundStats {
            round,
            newly_expanded,
            total_concrete: concrete_states.len(),
            total_abstract: solved.total_abstract_states,
            kept_abstract: solved.kept_states.len(),
            complete_policy_states: policy.complete_state_count,
            incomplete_policy_states: policy.incomplete_state_count,
            scc_count: sccs.len(),
            largest_scc,
            reachable_cycle_count,
            falsify_successes: falsify.successes,
            falsify_failures: falsify.failures,
            falsify_earliest_failure_step: falsify.earliest_failure_step,
        };

        println!(
            "round={} expanded={} concrete={} abstract={} kept={} complete={} scc={} largest_scc={} reachable_cycles={} falsify={}/{} earliest_fail={}",
            round_stats.round,
            round_stats.newly_expanded,
            round_stats.total_concrete,
            round_stats.total_abstract,
            round_stats.kept_abstract,
            round_stats.complete_policy_states,
            round_stats.scc_count,
            round_stats.largest_scc,
            round_stats.reachable_cycle_count,
            round_stats.falsify_successes,
            cli.falsify_rollouts,
            round_stats
                .falsify_earliest_failure_step
                .map_or_else(|| "none".to_string(), |v| v.to_string()),
        );
        println!(
            "coverage_keys={} known_edges_total={} unknown_edges_total={}",
            coverage.len(),
            coverage.values().map(|x| x.known_successors).sum::<usize>(),
            coverage
                .values()
                .map(|x| x.unknown_successors)
                .sum::<usize>()
        );

        append_round_stats_csv(&stats_csv_path, &round_stats);
        let round_policy_path = format!(
            "{}/tetris_invariant_policy_map_round_{}.csv",
            cli.artifact_dir, round
        );
        if let Err(error) = write_policy_map_csv(&round_policy_path, &policy) {
            eprintln!("Failed to write round policy map: {error}");
        }
        let round_cycles_path = format!(
            "{}/tetris_invariant_cycles_round_{}.csv",
            cli.artifact_dir, round
        );
        if let Err(error) = write_cycle_candidates_csv(&round_cycles_path, &cycles) {
            eprintln!("Failed to write round cycle candidates: {error}");
        }

        if solved.kept_states.len() > best_kept {
            best_kept = solved.kept_states.len();
            if let Err(error) = write_policy_map_csv(&cli.policy_map_csv, &policy) {
                eprintln!("Failed to write best policy map: {error}");
            }
        }

        let reached_kept_target = cli
            .target_kept_states
            .is_some_and(|target| solved.kept_states.len() >= target);
        let reached_cycle_target = cycles
            .iter()
            .any(|c| c.reachable_from_empty && c.size >= cli.target_min_cycle_size);
        if reached_kept_target || reached_cycle_target {
            println!(
                "Stopping early at round {} (kept_target={} cycle_target={})",
                round, reached_kept_target, reached_cycle_target
            );
            break;
        }

        if incremental.frontier.is_empty() {
            println!("Frontier exhausted at round {}, stopping.", round);
            break;
        }

        abstraction = best_round_cfg;
    }

    println!(
        "Incremental run complete in {:.2}s. best_kept={}",
        start.elapsed().as_secs_f64(),
        best_kept
    );
}

fn expand_incremental_round(
    incremental: &mut IncrementalState,
    budget: usize,
    max_states: usize,
    frontier_order: FrontierOrder,
) -> usize {
    let mut expanded = 0usize;
    while expanded < budget && incremental.visited_concrete.len() < max_states {
        let Some(state) = pop_frontier(&mut incremental.frontier, frontier_order) else {
            break;
        };
        let transitions = incremental
            .transition_cache
            .entry(state)
            .or_insert_with(|| enumerate_transitions(state));
        for transition in transitions.iter().copied() {
            if incremental.visited_concrete.insert(transition.next) {
                incremental.frontier.push_back(transition.next);
            }
            if incremental.visited_concrete.len() >= max_states {
                break;
            }
        }
        expanded += 1;
    }
    expanded
}

fn pop_frontier(
    frontier: &mut VecDeque<ConcreteState>,
    order: FrontierOrder,
) -> Option<ConcreteState> {
    match order {
        FrontierOrder::Fifo => frontier.pop_front(),
        FrontierOrder::Height => {
            let mut best_idx = None;
            let mut best_priority = u32::MAX;
            for (idx, state) in frontier.iter().enumerate() {
                let priority = state.board.count();
                if priority < best_priority {
                    best_priority = priority;
                    best_idx = Some(idx);
                }
            }
            best_idx.and_then(|idx| frontier.remove(idx))
        }
    }
}

fn enumerate_transitions(state: ConcreteState) -> Vec<CachedTransition> {
    let mut out = Vec::new();
    for (piece, next_bag) in state.bag.iter_next_states() {
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next_board = state.board;
            let result = next_board.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST {
                continue;
            }
            out.push(CachedTransition {
                next: ConcreteState {
                    board: next_board,
                    bag: next_bag,
                },
            });
        }
    }
    out
}

fn run_tetris_invariant_synthesis(cli: Cli) {
    let start = Instant::now();
    println!("=== Tetris Controlled-Invariant Synthesis (MVP) ===");
    println!(
        "Sampling: max_states={} max_depth={} reps_per_abstract={}",
        cli.max_concrete_states, cli.max_sampling_depth, cli.max_representatives_per_abstract
    );
    println!(
        "Falsification: rollouts={} horizon={}",
        cli.falsify_rollouts, cli.falsify_horizon
    );
    println!(
        "Solver: unknown_successor={:?} representative_quantifier={:?}",
        cli.unknown_successor_policy, cli.representative_quantifier
    );

    let concrete_states =
        sample_reachable_concrete_states(cli.max_concrete_states, cli.max_sampling_depth);

    if concrete_states.is_empty() {
        println!("No concrete states sampled.");
        return;
    }

    println!(
        "Sampled {} concrete states in {:.2}s",
        concrete_states.len(),
        start.elapsed().as_secs_f64()
    );

    let mut abstraction = AbstractionConfig {
        height_bucket: cli.height_bucket.max(1),
        holes_bucket: cli.holes_bucket.max(1),
        roughness_bucket: cli.roughness_bucket.max(1),
    };

    let mut best_kept = 0usize;
    let mut best_result = None;

    for refinement_round in 0..cli.max_refinement_rounds {
        println!(
            "\n--- Refinement round {} (height={}, holes={}, roughness={}) ---",
            refinement_round + 1,
            abstraction.height_bucket,
            abstraction.holes_bucket,
            abstraction.roughness_bucket
        );

        let model = build_abstract_model(
            &concrete_states,
            &abstraction,
            cli.max_representatives_per_abstract,
        );

        let solve_start = Instant::now();
        let solved = solve_controlled_invariant(
            &model,
            &abstraction,
            cli.unknown_successor_policy,
            cli.representative_quantifier,
        );
        println!(
            "Abstract solve finished in {:.2}s",
            solve_start.elapsed().as_secs_f64()
        );
        println!(
            "Abstract states total={} kept={} ({:.2}%)",
            solved.total_abstract_states,
            solved.kept_states.len(),
            (solved.kept_states.len() as f64 / solved.total_abstract_states.max(1) as f64) * 100.0
        );

        let empty = ConcreteState {
            board: TetrisBoard::new(),
            bag: TetrisPieceBagState::new(),
        };
        let empty_abs = abstract_state(empty, &abstraction);
        let empty_invariant = solved.kept_states.contains(&empty_abs);
        println!("Empty-state abstract bucket in invariant: {empty_invariant}");

        let falsify = falsify_with_rollouts(
            &solved,
            &abstraction,
            cli.falsify_rollouts,
            cli.falsify_horizon,
            42,
        );
        println!(
            "Falsification: successes={}/{} failures={} earliest_failure_step={}",
            falsify.successes,
            cli.falsify_rollouts,
            falsify.failures,
            falsify
                .earliest_failure_step
                .map_or_else(|| "none".to_string(), |v| v.to_string())
        );

        if solved.kept_states.len() > best_kept {
            best_kept = solved.kept_states.len();
            best_result = Some((abstraction.clone(), solved.clone(), falsify.clone()));
        }

        if empty_invariant && falsify.failures == 0 {
            println!("Found empirically stable invariant bucket for the empty state.");
            best_result = Some((abstraction.clone(), solved, falsify));
            break;
        }

        abstraction = abstraction.refined();
    }

    println!("\n=== Final Result ===");
    if let Some((cfg, solved, falsify)) = best_result {
        let empty_abs = abstract_state(
            ConcreteState {
                board: TetrisBoard::new(),
                bag: TetrisPieceBagState::new(),
            },
            &cfg,
        );
        println!(
            "Best abstraction: height={} holes={} roughness={}",
            cfg.height_bucket, cfg.holes_bucket, cfg.roughness_bucket
        );
        println!(
            "Kept abstract states: {} / {}",
            solved.kept_states.len(),
            solved.total_abstract_states
        );
        println!(
            "Empty-state bucket kept: {}",
            solved.kept_states.contains(&empty_abs)
        );
        println!(
            "Rollout falsification successes={} failures={} earliest_failure_step={}",
            falsify.successes,
            falsify.failures,
            falsify
                .earliest_failure_step
                .map_or_else(|| "none".to_string(), |v| v.to_string())
        );

        let model =
            build_abstract_model(&concrete_states, &cfg, cli.max_representatives_per_abstract);
        let policy = build_abstract_policy_map(&model, &solved, &cfg, cli.unknown_successor_policy);
        println!(
            "Policy map stats: complete_states={} incomplete_states={} entries={}",
            policy.complete_state_count,
            policy.incomplete_state_count,
            policy.entries.len()
        );
        if let Err(error) = write_policy_map_csv(&cli.policy_map_csv, &policy) {
            eprintln!("Failed to write policy map CSV: {error}");
        } else {
            println!("Wrote policy map CSV: {}", cli.policy_map_csv);
        }
    } else {
        println!("No successful synthesis result produced.");
    }

    println!("Total runtime: {:.2}s", start.elapsed().as_secs_f64());
}

fn sample_reachable_concrete_states(max_states: usize, max_depth: usize) -> Vec<ConcreteState> {
    let root = ConcreteState {
        board: TetrisBoard::new(),
        bag: TetrisPieceBagState::new(),
    };
    let mut visited: HashSet<ConcreteState> = HashSet::with_capacity(max_states.min(1_000_000));
    let mut out = Vec::with_capacity(max_states.min(1_000_000));
    let mut q = VecDeque::new();
    q.push_back((root, 0usize));
    visited.insert(root);
    out.push(root);

    while let Some((state, depth)) = q.pop_front() {
        if out.len() >= max_states || depth >= max_depth {
            continue;
        }

        for (piece, next_bag) in state.bag.iter_next_states() {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next_board = state.board;
                let result = next_board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }

                let next_state = ConcreteState {
                    board: next_board,
                    bag: next_bag,
                };

                if visited.insert(next_state) {
                    out.push(next_state);
                    q.push_back((next_state, depth + 1));
                    if out.len() >= max_states {
                        break;
                    }
                }
            }
            if out.len() >= max_states {
                break;
            }
        }
    }

    // Keep deterministic ordering and cap pathological growth from huge frontiers.
    if out.len() > max_states {
        out.truncate(max_states);
    }

    out
}

#[derive(Debug, Clone)]
struct AbstractModel {
    representatives: HashMap<AbstractState, Vec<ConcreteState>>,
    all_states: Vec<AbstractState>,
}

fn build_abstract_model(
    concrete_states: &[ConcreteState],
    abstraction: &AbstractionConfig,
    max_representatives_per_abstract: usize,
) -> AbstractModel {
    let mut representatives: HashMap<AbstractState, Vec<ConcreteState>> = HashMap::new();

    for &state in concrete_states {
        let abs = abstract_state(state, abstraction);
        let reps = representatives.entry(abs).or_default();
        if reps.len() < max_representatives_per_abstract.max(1) {
            reps.push(state);
        }
    }

    let mut all_states: Vec<AbstractState> = representatives.keys().copied().collect();
    all_states.sort_by_key(|k| {
        (
            k.bag_mask,
            k.max_height_bucket,
            k.holes_bucket,
            k.roughness_bucket,
        )
    });

    AbstractModel {
        representatives,
        all_states,
    }
}

fn abstract_state(state: ConcreteState, cfg: &AbstractionConfig) -> AbstractState {
    let heights = state.board.heights();
    let max_height = heights.into_iter().max().unwrap_or(0);
    let holes = state.board.total_holes();

    let mut roughness = 0u32;
    let mut i = 0usize;
    while i + 1 < TetrisBoard::WIDTH {
        roughness += heights[i].abs_diff(heights[i + 1]);
        i += 1;
    }

    AbstractState {
        bag_mask: state.bag.into(),
        max_height_bucket: (max_height / cfg.height_bucket.max(1)) as u8,
        holes_bucket: (holes / cfg.holes_bucket.max(1)) as u8,
        roughness_bucket: (roughness / cfg.roughness_bucket.max(1)) as u8,
    }
}

#[derive(Debug, Clone)]
struct SolveResult {
    kept_states: HashSet<AbstractState>,
    total_abstract_states: usize,
}

fn solve_controlled_invariant(
    model: &AbstractModel,
    abstraction: &AbstractionConfig,
    unknown_successor_policy: UnknownSuccessorPolicy,
    representative_quantifier: RepresentativeQuantifier,
) -> SolveResult {
    let mut kept: HashSet<AbstractState> = model.all_states.iter().copied().collect();
    if kept.is_empty() {
        return SolveResult {
            kept_states: kept,
            total_abstract_states: 0,
        };
    }

    loop {
        let mut removed_any = false;
        let snapshot = kept.clone();

        for &abs_state in &model.all_states {
            if !snapshot.contains(&abs_state) {
                continue;
            }

            let Some(reps) = model.representatives.get(&abs_state) else {
                kept.remove(&abs_state);
                removed_any = true;
                continue;
            };

            let bag = TetrisPieceBagState::from(abs_state.bag_mask);
            let mut is_valid = true;

            for (piece, next_bag) in bag.iter_next_states() {
                let rep_outcomes: Vec<bool> = reps
                    .iter()
                    .map(|rep| {
                        has_safe_action_for_piece(
                            rep.board,
                            piece,
                            next_bag,
                            abstraction,
                            &snapshot,
                            &model.representatives,
                            unknown_successor_policy,
                        )
                    })
                    .collect();

                let piece_ok = match representative_quantifier {
                    RepresentativeQuantifier::Forall => rep_outcomes.iter().all(|v| *v),
                    RepresentativeQuantifier::Exists => rep_outcomes.iter().any(|v| *v),
                };

                if !piece_ok {
                    is_valid = false;
                    break;
                }
            }

            if !is_valid {
                kept.remove(&abs_state);
                removed_any = true;
            }
        }

        if !removed_any {
            break;
        }
    }

    SolveResult {
        kept_states: kept,
        total_abstract_states: model.all_states.len(),
    }
}

fn has_safe_action_for_piece(
    board: TetrisBoard,
    piece: TetrisPiece,
    next_bag: TetrisPieceBagState,
    abstraction: &AbstractionConfig,
    kept: &HashSet<AbstractState>,
    known_abstract_states: &HashMap<AbstractState, Vec<ConcreteState>>,
    unknown_successor_policy: UnknownSuccessorPolicy,
) -> bool {
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next_board = board;
        let result = next_board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            continue;
        }
        let next_abs = abstract_state(
            ConcreteState {
                board: next_board,
                bag: next_bag,
            },
            abstraction,
        );
        if kept.contains(&next_abs) {
            return true;
        }
        if !known_abstract_states.contains_key(&next_abs)
            && unknown_successor_policy == UnknownSuccessorPolicy::Ignore
        {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PolicyKey {
    state: AbstractState,
    piece: TetrisPiece,
}

#[derive(Debug, Clone, Copy)]
struct PolicyValue {
    placement_index: u8,
}

#[derive(Debug, Clone)]
struct PolicyMap {
    entries: HashMap<PolicyKey, PolicyValue>,
    complete_state_count: usize,
    incomplete_state_count: usize,
}

#[derive(Debug, Clone)]
struct PolicyGraph {
    nodes: Vec<AbstractState>,
    edges: Vec<Vec<usize>>,
    index: HashMap<AbstractState, usize>,
}

fn build_abstract_policy_map(
    model: &AbstractModel,
    solved: &SolveResult,
    abstraction: &AbstractionConfig,
    unknown_successor_policy: UnknownSuccessorPolicy,
) -> PolicyMap {
    let mut entries: HashMap<PolicyKey, PolicyValue> = HashMap::new();
    let mut complete_state_count = 0usize;
    let mut incomplete_state_count = 0usize;

    for &state in &solved.kept_states {
        let Some(reps) = model.representatives.get(&state) else {
            incomplete_state_count += 1;
            continue;
        };
        let rep = reps[0];
        let bag = TetrisPieceBagState::from(state.bag_mask);
        let mut state_complete = true;

        for (piece, next_bag) in bag.iter_next_states() {
            let mut selected = None;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next_board = rep.board;
                let result = next_board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }
                let next_abs = abstract_state(
                    ConcreteState {
                        board: next_board,
                        bag: next_bag,
                    },
                    abstraction,
                );

                if solved.kept_states.contains(&next_abs)
                    || (unknown_successor_policy == UnknownSuccessorPolicy::Ignore
                        && !model.representatives.contains_key(&next_abs))
                {
                    selected = Some(placement.index());
                    break;
                }
            }

            if let Some(placement_index) = selected {
                entries.insert(PolicyKey { state, piece }, PolicyValue { placement_index });
            } else {
                state_complete = false;
            }
        }

        if state_complete {
            complete_state_count += 1;
        } else {
            incomplete_state_count += 1;
        }
    }

    PolicyMap {
        entries,
        complete_state_count,
        incomplete_state_count,
    }
}

fn write_policy_map_csv(path: &str, policy: &PolicyMap) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "bag_mask,max_height_bucket,holes_bucket,roughness_bucket,piece,placement_index"
    )?;

    let mut rows: Vec<(PolicyKey, PolicyValue)> =
        policy.entries.iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by_key(|(k, _)| {
        (
            k.state.bag_mask,
            k.state.max_height_bucket,
            k.state.holes_bucket,
            k.state.roughness_bucket,
            u8::from(k.piece),
        )
    });

    for (k, v) in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{}",
            k.state.bag_mask,
            k.state.max_height_bucket,
            k.state.holes_bucket,
            k.state.roughness_bucket,
            u8::from(k.piece),
            v.placement_index
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn compute_transition_coverage(
    model: &AbstractModel,
    abstraction: &AbstractionConfig,
    unknown_successor_policy: UnknownSuccessorPolicy,
) -> HashMap<CoverageKey, TransitionCoverage> {
    let mut coverage: HashMap<CoverageKey, TransitionCoverage> = HashMap::new();

    for (&abs_state, reps) in &model.representatives {
        let bag = TetrisPieceBagState::from(abs_state.bag_mask);
        for (piece, next_bag) in bag.iter_next_states() {
            let mut known = 0usize;
            let mut unknown = 0usize;
            for rep in reps {
                for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                    let mut next_board = rep.board;
                    let result = next_board.apply_piece_placement(placement);
                    if result.is_lost == IsLost::LOST {
                        continue;
                    }
                    let next_abs = abstract_state(
                        ConcreteState {
                            board: next_board,
                            bag: next_bag,
                        },
                        abstraction,
                    );
                    if model.representatives.contains_key(&next_abs) {
                        known += 1;
                    } else if unknown_successor_policy == UnknownSuccessorPolicy::Ignore {
                        unknown += 1;
                    }
                }
            }
            coverage.insert(
                CoverageKey {
                    state: abs_state,
                    piece,
                },
                TransitionCoverage {
                    known_successors: known,
                    unknown_successors: unknown,
                },
            );
        }
    }

    coverage
}

fn build_policy_graph(
    model: &AbstractModel,
    solved: &SolveResult,
    policy: &PolicyMap,
    abstraction: &AbstractionConfig,
) -> PolicyGraph {
    let mut nodes: Vec<AbstractState> = solved.kept_states.iter().copied().collect();
    nodes.sort_by_key(|k| {
        (
            k.bag_mask,
            k.max_height_bucket,
            k.holes_bucket,
            k.roughness_bucket,
        )
    });
    let index: HashMap<AbstractState, usize> =
        nodes.iter().enumerate().map(|(idx, &s)| (s, idx)).collect();
    let mut edges = vec![Vec::<usize>::new(); nodes.len()];

    for (key, value) in &policy.entries {
        let Some(&from_idx) = index.get(&key.state) else {
            continue;
        };
        let Some(reps) = model.representatives.get(&key.state) else {
            continue;
        };
        let rep = reps[0];
        let mut next_bag = TetrisPieceBagState::from(key.state.bag_mask);
        next_bag.remove(key.piece);
        if next_bag.is_empty() {
            next_bag.fill();
        }
        let placement = TetrisPiecePlacement::from_index(value.placement_index);
        let mut next_board = rep.board;
        let result = next_board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            continue;
        }
        let next_abs = abstract_state(
            ConcreteState {
                board: next_board,
                bag: next_bag,
            },
            abstraction,
        );
        if let Some(&to_idx) = index.get(&next_abs) {
            edges[from_idx].push(to_idx);
        }
    }

    for neighbors in &mut edges {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    PolicyGraph {
        nodes,
        edges,
        index,
    }
}

fn strongly_connected_components(graph: &PolicyGraph, kept_len: usize) -> Vec<Vec<usize>> {
    if kept_len == 0 {
        return Vec::new();
    }
    struct TarjanCtx {
        index_counter: usize,
        indices: Vec<Option<usize>>,
        lowlink: Vec<usize>,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        sccs: Vec<Vec<usize>>,
    }
    fn strongconnect(v: usize, graph: &PolicyGraph, ctx: &mut TarjanCtx) {
        ctx.indices[v] = Some(ctx.index_counter);
        ctx.lowlink[v] = ctx.index_counter;
        ctx.index_counter += 1;
        ctx.stack.push(v);
        ctx.on_stack[v] = true;

        for &w in &graph.edges[v] {
            if ctx.indices[w].is_none() {
                strongconnect(w, graph, ctx);
                ctx.lowlink[v] = ctx.lowlink[v].min(ctx.lowlink[w]);
            } else if ctx.on_stack[w] {
                let w_idx = ctx.indices[w].expect("index exists");
                ctx.lowlink[v] = ctx.lowlink[v].min(w_idx);
            }
        }

        let v_idx = ctx.indices[v].expect("index exists");
        if ctx.lowlink[v] == v_idx {
            let mut component = Vec::new();
            loop {
                let w = ctx.stack.pop().expect("non-empty stack");
                ctx.on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            if component.len() > 1 || graph.edges[component[0]].contains(&component[0]) {
                ctx.sccs.push(component);
            }
        }
    }

    let n = graph.nodes.len();
    let mut ctx = TarjanCtx {
        index_counter: 0,
        indices: vec![None; n],
        lowlink: vec![0; n],
        stack: Vec::new(),
        on_stack: vec![false; n],
        sccs: Vec::new(),
    };

    for v in 0..n {
        if ctx.indices[v].is_none() {
            strongconnect(v, graph, &mut ctx);
        }
    }
    ctx.sccs
}

fn rank_cycle_candidates(
    sccs: &[Vec<usize>],
    graph: &PolicyGraph,
    model: &AbstractModel,
    policy: &PolicyMap,
    abstraction: &AbstractionConfig,
    empty_state: AbstractState,
) -> Vec<CycleCandidate> {
    let reachable = reachable_nodes_from_empty(graph, empty_state);
    let mut out = Vec::new();
    for (id, scc) in sccs.iter().enumerate() {
        let set: HashSet<usize> = scc.iter().copied().collect();
        let mut closed_states = 0usize;
        for &node_idx in scc {
            let state = graph.nodes[node_idx];
            let bag = TetrisPieceBagState::from(state.bag_mask);
            let mut all_closed = true;
            for (piece, _) in bag.iter_next_states() {
                let Some(value) = policy.entries.get(&PolicyKey { state, piece }) else {
                    all_closed = false;
                    break;
                };
                let Some(reps) = model.representatives.get(&state) else {
                    all_closed = false;
                    break;
                };
                let rep = reps[0];
                let mut next_bag = TetrisPieceBagState::from(state.bag_mask);
                next_bag.remove(piece);
                if next_bag.is_empty() {
                    next_bag.fill();
                }
                let placement = TetrisPiecePlacement::from_index(value.placement_index);
                let mut next_board = rep.board;
                let result = next_board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    all_closed = false;
                    break;
                }
                let next_state = abstract_state(
                    ConcreteState {
                        board: next_board,
                        bag: next_bag,
                    },
                    abstraction,
                );
                let next_idx = graph.index.get(&next_state).copied();
                if !next_idx.is_some_and(|idx| set.contains(&idx)) {
                    all_closed = false;
                    break;
                }
            }
            if all_closed {
                closed_states += 1;
            }
        }
        let reachable_from_empty = scc.iter().any(|idx| reachable.contains(idx));
        out.push(CycleCandidate {
            id,
            size: scc.len(),
            all_piece_closed_states: closed_states,
            all_piece_closure_ratio: closed_states as f64 / scc.len().max(1) as f64,
            reachable_from_empty,
        });
    }
    out.sort_by(|a, b| {
        b.reachable_from_empty
            .cmp(&a.reachable_from_empty)
            .then_with(|| b.all_piece_closed_states.cmp(&a.all_piece_closed_states))
            .then_with(|| b.size.cmp(&a.size))
    });
    out
}

fn reachable_nodes_from_empty(graph: &PolicyGraph, empty_state: AbstractState) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let Some(&start) = graph.index.get(&empty_state) else {
        return reachable;
    };
    let mut q = VecDeque::new();
    q.push_back(start);
    reachable.insert(start);
    while let Some(v) = q.pop_front() {
        for &nxt in &graph.edges[v] {
            if reachable.insert(nxt) {
                q.push_back(nxt);
            }
        }
    }
    reachable
}

fn ensure_artifact_dir(path: &str) {
    if let Err(error) = std::fs::create_dir_all(path) {
        eprintln!("Failed to create artifact dir {}: {}", path, error);
    }
}

fn initialize_round_stats_csv(path: &str) {
    if Path::new(path).exists() {
        return;
    }
    if let Ok(file) = File::create(path) {
        let mut writer = BufWriter::new(file);
        let _ = writeln!(
            writer,
            "round,newly_expanded,total_concrete,total_abstract,kept_abstract,complete_policy_states,incomplete_policy_states,scc_count,largest_scc,reachable_cycle_count,falsify_successes,falsify_failures,falsify_earliest_failure_step"
        );
        let _ = writer.flush();
    }
}

fn append_round_stats_csv(path: &str, stats: &RoundStats) {
    let Ok(file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let mut writer = BufWriter::new(file);
    let _ = writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{}",
        stats.round,
        stats.newly_expanded,
        stats.total_concrete,
        stats.total_abstract,
        stats.kept_abstract,
        stats.complete_policy_states,
        stats.incomplete_policy_states,
        stats.scc_count,
        stats.largest_scc,
        stats.reachable_cycle_count,
        stats.falsify_successes,
        stats.falsify_failures,
        stats
            .falsify_earliest_failure_step
            .map_or_else(|| "none".to_string(), |v| v.to_string())
    );
    let _ = writer.flush();
}

fn write_cycle_candidates_csv(path: &str, cycles: &[CycleCandidate]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "cycle_id,size,all_piece_closed_states,all_piece_closure_ratio,reachable_from_empty"
    )?;
    for c in cycles {
        writeln!(
            writer,
            "{},{},{},{:.4},{}",
            c.id,
            c.size,
            c.all_piece_closed_states,
            c.all_piece_closure_ratio,
            c.reachable_from_empty
        )?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone)]
struct FalsifyResult {
    successes: usize,
    failures: usize,
    earliest_failure_step: Option<usize>,
}

fn falsify_with_rollouts(
    solved: &SolveResult,
    abstraction: &AbstractionConfig,
    rollouts: usize,
    horizon: usize,
    base_seed: u64,
) -> FalsifyResult {
    let mut successes = 0usize;
    let mut failures = 0usize;
    let mut earliest_failure_step: Option<usize> = None;

    for r in 0..rollouts {
        let mut game = tetris_game::TetrisGame::new_with_seed(base_seed.wrapping_add(r as u64));
        let mut bag = TetrisPieceBagState::new();

        let mut survived = true;
        for step in 0..horizon {
            let board = game.board;
            let state = ConcreteState { board, bag };
            let abs = abstract_state(state, abstraction);

            if !solved.kept_states.contains(&abs) {
                survived = false;
                earliest_failure_step = Some(match earliest_failure_step {
                    Some(prev) => prev.min(step),
                    None => step,
                });
                break;
            }

            let piece = game.current_piece;
            let mut chosen = None;

            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next_board = board;
                let result = next_board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }

                let mut next_bag = bag;
                next_bag.remove(piece);
                if next_bag.is_empty() {
                    next_bag.fill();
                }
                let next_abs = abstract_state(
                    ConcreteState {
                        board: next_board,
                        bag: next_bag,
                    },
                    abstraction,
                );
                if solved.kept_states.contains(&next_abs) {
                    chosen = Some((placement, next_bag));
                    break;
                }
            }

            let Some((placement, next_bag)) = chosen else {
                survived = false;
                earliest_failure_step = Some(match earliest_failure_step {
                    Some(prev) => prev.min(step),
                    None => step,
                });
                break;
            };

            let result = game.apply_placement(placement);
            if result.is_lost == IsLost::LOST {
                survived = false;
                earliest_failure_step = Some(match earliest_failure_step {
                    Some(prev) => prev.min(step),
                    None => step,
                });
                break;
            }
            bag = next_bag;
        }

        if survived {
            successes += 1;
        } else {
            failures += 1;
        }
    }

    FalsifyResult {
        successes,
        failures,
        earliest_failure_step,
    }
}
