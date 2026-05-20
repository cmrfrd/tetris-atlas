#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! 5-bag empty-board witness search.
//!
//! This binary enumerates fixed 5-bag sequences and runs a direct PUCT/MCTS search
//! over the full 35-piece sequence. The policy prior and non-terminal value both
//! use the shared height-MSE scorer from `tetris-search/src/scoring.rs`; terminal
//! success is still exact and replay-verified: after all 35 pieces, the board must
//! be empty and exactly 14 lines must have cleared.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rustc_hash::{FxHashMap, FxHasher};
use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPiecePlacement,
    constants,
};
use tetris_search::{
    TetrisBoardScoreState, height_mse_board_score, height_mse_distance_from_empty,
};

const BAG_COUNT: usize = 5;
const PIECES_PER_BAG: usize = 7;
const TOTAL_PIECES: usize = BAG_COUNT * PIECES_PER_BAG;

const BAG_PERMUTATION_COUNT: usize = 5_040;
const TOTAL_FIVE_BAG_SEQUENCES: u128 = pow_usize_as_u128(BAG_PERMUTATION_COUNT, BAG_COUNT);

const DEFAULT_MCTS_RESTARTS: u64 = 1;
const DEFAULT_MCTS_EXPLORATION: f32 = 1.4;
const DEFAULT_MCTS_PRIOR_TEMPERATURE: f32 = 64.0;
const DEFAULT_MCTS_PRIOR_NOISE: f32 = 0.0;
const DEFAULT_MCTS_MAX_CHILDREN: usize = TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT;
const DEFAULT_MCTS_PROGRESS_EVERY: u64 = 25_000;
const DEFAULT_MAX_HEIGHT: u32 = StandardTetris::ROWS as u32;

type ForcedBag = [TetrisPiece; PIECES_PER_BAG];

static ALL_BAG_PERMUTATIONS: [ForcedBag; BAG_PERMUTATION_COUNT] =
    generate_forced_bag_permutations();

const DEFAULT_PLACEMENT: TetrisPiecePlacement = TetrisPiecePlacement::from_index(0);
const DEFAULT_CANDIDATE: Candidate = Candidate {
    placement: DEFAULT_PLACEMENT,
    board: TetrisBoard::EMPTY_BOARD,
    prior_score: f32::NEG_INFINITY,
};

#[derive(Parser, Debug)]
#[command(name = "tetris_five_bag_empty")]
#[command(about = "Scan 5-bag tuples with PUCT for empty-to-empty placement witnesses")]
struct Cli {
    /// Optional explicit 5-bag tuple as comma-separated bag permutation indices.
    #[arg(long)]
    bags: Option<String>,

    /// First 5-bag ordinal to scan when --bags is omitted.
    #[arg(long, default_value_t = 0)]
    start: u128,

    /// Maximum number of 5-bag tuples to scan. Omit to scan until success or exhaustion.
    #[arg(long)]
    limit: Option<u64>,

    /// Reject intermediate boards above this height.
    #[arg(long, default_value_t = DEFAULT_MAX_HEIGHT)]
    max_height: u32,

    /// PUCT iterations per restart and fixed 5-bag tuple. Omit to run until a witness is found.
    #[arg(long)]
    mcts_iterations: Option<u64>,

    /// Number of independent deterministic PUCT restarts per fixed 5-bag tuple.
    #[arg(long, default_value_t = DEFAULT_MCTS_RESTARTS)]
    mcts_restarts: u64,

    /// PUCT exploration constant.
    #[arg(long, default_value_t = DEFAULT_MCTS_EXPLORATION)]
    mcts_exploration: f32,

    /// Softmax temperature for scorer-derived move priors.
    #[arg(long, default_value_t = DEFAULT_MCTS_PRIOR_TEMPERATURE)]
    mcts_prior_temperature: f32,

    /// Deterministic signed noise added to scorer priors before sorting.
    #[arg(long, default_value_t = DEFAULT_MCTS_PRIOR_NOISE)]
    mcts_prior_noise: f32,

    /// Maximum children retained per expanded node after scorer-prior sorting.
    #[arg(long, default_value_t = DEFAULT_MCTS_MAX_CHILDREN)]
    mcts_max_children: usize,

    /// Print per-restart MCTS progress every N iterations. Use 0 to disable.
    #[arg(long, default_value_t = DEFAULT_MCTS_PROGRESS_EVERY)]
    mcts_progress_every: u64,

    /// Print scan progress every N scanned tuples. Use 0 to disable.
    #[arg(long, default_value_t = 10_000)]
    progress_every: u64,

    /// Continue after the first witness instead of stopping immediately.
    #[arg(long, default_value_t = false)]
    keep_going: bool,

    /// Print each scanned tuple and the best placement prefix found before failure.
    #[arg(long, default_value_t = false)]
    print_every_attempt: bool,

    /// Print board states while replaying a found witness.
    #[arg(long, default_value_t = false)]
    print_boards: bool,
}

#[derive(Debug, Clone, Copy)]
struct FiveBagWitness {
    placements: [TetrisPiecePlacement; TOTAL_PIECES],
    lines_cleared: u32,
    max_height: u32,
    iterations: u64,
    tree_nodes: usize,
    terminal_evaluations: u64,
    restart: u64,
    best_score: f32,
}

#[derive(Debug, Clone, Copy)]
struct PuctFailure {
    placements: [TetrisPiecePlacement; TOTAL_PIECES],
    prefix_len: usize,
    best_cells: u32,
    best_score: f32,
    iterations: u64,
    tree_nodes: usize,
    terminal_evaluations: u64,
}

#[derive(Debug, Clone, Copy)]
enum PuctAttempt {
    Witness(FiveBagWitness),
    Failure(PuctFailure),
}

#[derive(Debug, Clone, Copy)]
struct PuctConfig {
    max_height: u32,
    iterations: Option<u64>,
    exploration: f32,
    prior_temperature: f32,
    prior_noise: f32,
    max_children: usize,
    noise_seed: u64,
}

#[derive(Debug, Clone, Copy)]
struct PuctProgress {
    iterations: u64,
    tree_nodes: usize,
    terminal_evaluations: u64,
    best_cells: u32,
    best_score: f32,
    best_prefix_len: usize,
}

#[derive(Debug, Clone, Copy)]
struct ScanPosition {
    item: u128,
    total: u128,
    ordinal: Option<u128>,
}

impl ScanPosition {
    const fn single() -> Self {
        Self {
            item: 1,
            total: 1,
            ordinal: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PuctRunResult {
    witness_placements: Option<[TetrisPiecePlacement; TOTAL_PIECES]>,
    best_placements: [TetrisPiecePlacement; TOTAL_PIECES],
    best_prefix_len: usize,
    best_cells: u32,
    best_score: f32,
    iterations: u64,
    tree_nodes: usize,
    terminal_evaluations: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ScanStats {
    scanned: u64,
    found: u64,
    mcts_iterations: u64,
    terminal_evaluations: u64,
    tree_nodes: usize,
    best_cells: u32,
    best_score: f32,
}

impl ScanStats {
    fn record_attempt(&mut self, attempt: &PuctAttempt) {
        let failure = match attempt {
            PuctAttempt::Witness(witness) => {
                self.found += 1;
                self.mcts_iterations += witness.iterations;
                self.terminal_evaluations += witness.terminal_evaluations;
                self.tree_nodes += witness.tree_nodes;
                self.best_cells = 0;
                self.best_score = self.best_score.max(witness.best_score);
                return;
            }
            PuctAttempt::Failure(failure) => failure,
        };

        self.mcts_iterations += failure.iterations;
        self.terminal_evaluations += failure.terminal_evaluations;
        self.tree_nodes += failure.tree_nodes;
        self.best_cells = self.best_cells.min(failure.best_cells);
        self.best_score = self.best_score.max(failure.best_score);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StateKey {
    board: TetrisBoard,
    step: u8,
}

#[derive(Clone, Copy, Debug)]
struct SearchState {
    board: TetrisBoard,
    step: usize,
}

#[derive(Clone, Debug)]
struct SearchNode {
    state: SearchState,
    visits: u32,
    value_sum: f32,
    expanded: bool,
    children: Vec<ActionEdge>,
}

impl SearchNode {
    fn new(state: SearchState) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActionEdge {
    placement: TetrisPiecePlacement,
    child: usize,
    visits: u32,
    value_sum: f32,
    prior: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    placement: TetrisPiecePlacement,
    board: TetrisBoard,
    prior_score: f32,
}

struct PuctSearch<'a> {
    pieces: &'a [TetrisPiece],
    config: PuctConfig,
    nodes: Vec<SearchNode>,
    index: FxHashMap<StateKey, usize>,
    best_placements: [TetrisPiecePlacement; TOTAL_PIECES],
    best_prefix_len: usize,
    best_cells: u32,
    best_score: f32,
    witness_placements: Option<[TetrisPiecePlacement; TOTAL_PIECES]>,
    terminal_evaluations: u64,
}

impl<'a> PuctSearch<'a> {
    fn new(pieces: &'a [TetrisPiece], config: PuctConfig) -> Self {
        let root_state = SearchState {
            board: TetrisBoard::EMPTY_BOARD,
            step: 0,
        };
        let mut index = FxHashMap::default();
        index.insert(state_key(root_state), 0);

        Self {
            pieces,
            config,
            nodes: vec![SearchNode::new(root_state)],
            index,
            best_placements: [DEFAULT_PLACEMENT; TOTAL_PIECES],
            best_prefix_len: 0,
            best_cells: u32::MAX,
            best_score: f32::NEG_INFINITY,
            witness_placements: None,
            terminal_evaluations: 0,
        }
    }

    fn search(
        mut self,
        progress_every: u64,
        mut on_progress: impl FnMut(PuctProgress),
    ) -> PuctRunResult {
        let mut iterations = 0u64;

        while self
            .config
            .iterations
            .is_none_or(|limit| iterations < limit)
        {
            iterations += 1;
            self.run_iteration();

            if progress_every > 0 && iterations % progress_every == 0 {
                on_progress(self.progress(iterations));
            }
            if self.witness_placements.is_some() {
                break;
            }
        }

        PuctRunResult {
            witness_placements: self.witness_placements,
            best_placements: self.best_placements,
            best_prefix_len: self.best_prefix_len,
            best_cells: self.best_cells,
            best_score: self.best_score,
            iterations,
            tree_nodes: self.nodes.len(),
            terminal_evaluations: self.terminal_evaluations,
        }
    }

    fn progress(&self, iterations: u64) -> PuctProgress {
        PuctProgress {
            iterations,
            tree_nodes: self.nodes.len(),
            terminal_evaluations: self.terminal_evaluations,
            best_cells: self.best_cells,
            best_score: self.best_score,
            best_prefix_len: self.best_prefix_len,
        }
    }

    fn run_iteration(&mut self) {
        let mut node_id = 0usize;
        let mut path_edges = [(usize::MAX, usize::MAX); TOTAL_PIECES];
        let mut path_placements = [DEFAULT_PLACEMENT; TOTAL_PIECES];
        let mut path_len = 0usize;

        let value = loop {
            if self.is_terminal(node_id) {
                break self.evaluate_terminal(node_id, &path_placements, path_len);
            }

            if !self.nodes[node_id].expanded {
                if self.expand_node(node_id) {
                    break self.evaluate_leaf(node_id, &path_placements, path_len);
                }
                break self.evaluate_dead_end(&path_placements, path_len);
            }

            if self.nodes[node_id].children.is_empty() {
                break self.evaluate_dead_end(&path_placements, path_len);
            }

            let edge_id = self.select_child(node_id);
            let edge = self.nodes[node_id].children[edge_id];
            if path_len == TOTAL_PIECES {
                break self.evaluate_dead_end(&path_placements, path_len);
            }

            path_edges[path_len] = (node_id, edge_id);
            path_placements[path_len] = edge.placement;
            path_len += 1;
            node_id = edge.child;
        };

        self.nodes[node_id].visits = self.nodes[node_id].visits.saturating_add(1);
        self.nodes[node_id].value_sum += value;

        for &(parent_id, edge_id) in path_edges.iter().take(path_len) {
            self.nodes[parent_id].visits = self.nodes[parent_id].visits.saturating_add(1);
            self.nodes[parent_id].value_sum += value;

            let edge = &mut self.nodes[parent_id].children[edge_id];
            edge.visits = edge.visits.saturating_add(1);
            edge.value_sum += value;
        }
    }

    fn is_terminal(&self, node_id: usize) -> bool {
        self.nodes[node_id].state.step == self.pieces.len()
    }

    fn evaluate_terminal(
        &mut self,
        node_id: usize,
        placements: &[TetrisPiecePlacement; TOTAL_PIECES],
        prefix_len: usize,
    ) -> f32 {
        self.terminal_evaluations += 1;

        let board = self.nodes[node_id].state.board;
        let is_success = board.count() == 0;
        let value = if is_success {
            1.0
        } else {
            terminal_failure_value(board)
        };

        self.record_best(board, value, placements, prefix_len);

        if is_success && prefix_len == self.pieces.len() {
            self.witness_placements = Some(*placements);
        }

        value
    }

    fn evaluate_leaf(
        &mut self,
        node_id: usize,
        placements: &[TetrisPiecePlacement; TOTAL_PIECES],
        prefix_len: usize,
    ) -> f32 {
        let state = self.nodes[node_id].state;
        let value = nonterminal_value(state.board, state.step, self.pieces.len());
        self.record_best(state.board, value, placements, prefix_len);
        value
    }

    fn evaluate_dead_end(
        &mut self,
        placements: &[TetrisPiecePlacement; TOTAL_PIECES],
        prefix_len: usize,
    ) -> f32 {
        let value = -1.0;
        if prefix_len > self.best_prefix_len {
            self.best_placements = *placements;
            self.best_prefix_len = prefix_len;
            self.best_score = self.best_score.max(value);
        }
        value
    }

    fn record_best(
        &mut self,
        board: TetrisBoard,
        value: f32,
        placements: &[TetrisPiecePlacement; TOTAL_PIECES],
        prefix_len: usize,
    ) {
        let terminal_bonus = usize::from(prefix_len == self.pieces.len());
        let best_terminal_bonus = usize::from(self.best_prefix_len == self.pieces.len());
        let better = terminal_bonus > best_terminal_bonus
            || (terminal_bonus == best_terminal_bonus
                && (prefix_len > self.best_prefix_len
                    || (prefix_len == self.best_prefix_len && value > self.best_score)));

        if better {
            self.best_placements = *placements;
            self.best_prefix_len = prefix_len;
            self.best_score = value;
        }

        if prefix_len == self.pieces.len() {
            self.best_cells = self.best_cells.min(board.count());
        }
    }

    fn expand_node(&mut self, node_id: usize) -> bool {
        let state = self.nodes[node_id].state;
        let piece = self.pieces[state.step];
        let mut candidates = [DEFAULT_CANDIDATE; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT];
        let mut candidate_len = 0usize;

        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next = state.board;
            let result = next.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST || next.height() > self.config.max_height {
                continue;
            }

            let mut prior_score = scorer_prior(next, result.lines_cleared);
            if self.config.prior_noise > 0.0 {
                prior_score +=
                    self.config.prior_noise * prior_noise(state, placement, self.config.noise_seed);
            }

            candidates[candidate_len] = Candidate {
                placement,
                board: next,
                prior_score,
            };
            candidate_len += 1;
        }

        candidates[..candidate_len].sort_unstable_by(|left, right| {
            right
                .prior_score
                .partial_cmp(&left.prior_score)
                .unwrap_or(Ordering::Equal)
        });

        let child_len = candidate_len
            .min(self.config.max_children)
            .min(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT);
        let priors = normalized_priors(&candidates, child_len, self.config.prior_temperature);
        let mut children = Vec::with_capacity(child_len);

        for idx in 0..child_len {
            let candidate = candidates[idx];
            let child_state = SearchState {
                board: candidate.board,
                step: state.step + 1,
            };
            let child_key = state_key(child_state);
            let child = if let Some(&child_id) = self.index.get(&child_key) {
                child_id
            } else {
                let child_id = self.nodes.len();
                self.nodes.push(SearchNode::new(child_state));
                self.index.insert(child_key, child_id);
                child_id
            };

            children.push(ActionEdge {
                placement: candidate.placement,
                child,
                visits: 0,
                value_sum: 0.0,
                prior: priors[idx],
            });
        }

        self.nodes[node_id].children = children;
        self.nodes[node_id].expanded = true;
        !self.nodes[node_id].children.is_empty()
    }

    fn select_child(&self, node_id: usize) -> usize {
        let node = &self.nodes[node_id];
        let parent_visits = node.visits.max(1) as f32;
        let exploration_scale = self.config.exploration * parent_visits.sqrt();
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, edge) in node.children.iter().enumerate() {
            let q = if edge.visits == 0 {
                0.0
            } else {
                edge.value_sum / edge.visits as f32
            };
            let u = exploration_scale * edge.prior / (1.0 + edge.visits as f32);
            let score = q + u;

            if score > best_score {
                best_idx = idx;
                best_score = score;
            }
        }

        best_idx
    }
}

const fn pow_usize_as_u128(base: usize, exponent: usize) -> u128 {
    let mut out = 1u128;
    let mut i = 0usize;
    while i < exponent {
        out *= base as u128;
        i += 1;
    }
    out
}

const fn advance_permutation(indices: &mut [u8; PIECES_PER_BAG]) -> bool {
    let mut pivot = PIECES_PER_BAG - 2;
    loop {
        if indices[pivot] < indices[pivot + 1] {
            break;
        }
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }

    let mut successor = PIECES_PER_BAG - 1;
    while indices[pivot] >= indices[successor] {
        successor -= 1;
    }

    let tmp = indices[pivot];
    indices[pivot] = indices[successor];
    indices[successor] = tmp;

    let mut left = pivot + 1;
    let mut right = PIECES_PER_BAG - 1;
    while left < right {
        let tmp = indices[left];
        indices[left] = indices[right];
        indices[right] = tmp;
        left += 1;
        right -= 1;
    }

    true
}

const fn generate_forced_bag_permutations() -> [ForcedBag; BAG_PERMUTATION_COUNT] {
    let mut permutations = [[TetrisPiece::O_PIECE; PIECES_PER_BAG]; BAG_PERMUTATION_COUNT];
    let mut indices = [0u8, 1, 2, 3, 4, 5, 6];
    let mut out_idx = 0usize;

    loop {
        let mut bag = [TetrisPiece::O_PIECE; PIECES_PER_BAG];
        let mut i = 0usize;
        while i < PIECES_PER_BAG {
            bag[i] = TetrisPiece::from_index(indices[i]);
            i += 1;
        }
        permutations[out_idx] = bag;
        out_idx += 1;

        if !advance_permutation(&mut indices) {
            break;
        }
    }

    permutations
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_cli(&cli)?;

    println!("=== tetris_five_bag_empty ===");
    println!("bag_count                = {BAG_COUNT}");
    println!("pieces_per_bag           = {PIECES_PER_BAG}");
    println!("total_pieces             = {TOTAL_PIECES}");
    println!("bag_permutation_count    = {BAG_PERMUTATION_COUNT}");
    println!("total_5bag_sequences     = {TOTAL_FIVE_BAG_SEQUENCES}");
    println!("max_height               = {}", cli.max_height);
    println!(
        "mcts_iterations          = {}",
        display_optional_iterations(cli.mcts_iterations)
    );
    println!("mcts_restarts            = {}", cli.mcts_restarts);
    println!("mcts_exploration         = {}", cli.mcts_exploration);
    println!("mcts_prior_temperature   = {}", cli.mcts_prior_temperature);
    println!("mcts_prior_noise         = {}", cli.mcts_prior_noise);
    println!(
        "mcts_max_children        = {}",
        cli.mcts_max_children
            .min(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT)
    );
    println!();

    if let Some(raw_bags) = cli.bags.as_deref() {
        let tuple = parse_bag_tuple(raw_bags)?;
        run_one_tuple(&cli, tuple)?;
        return Ok(());
    }

    scan_all_tuples(&cli)
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.start >= TOTAL_FIVE_BAG_SEQUENCES {
        bail!(
            "--start={} is outside 0..{}",
            cli.start,
            TOTAL_FIVE_BAG_SEQUENCES - 1
        );
    }
    if cli.mcts_iterations == Some(0) {
        bail!("--mcts-iterations must be greater than zero");
    }
    if cli.mcts_restarts == 0 {
        bail!("--mcts-restarts must be greater than zero");
    }
    if cli.mcts_max_children == 0 {
        bail!("--mcts-max-children must be greater than zero");
    }
    if !cli.mcts_exploration.is_finite() || cli.mcts_exploration < 0.0 {
        bail!("--mcts-exploration must be finite and non-negative");
    }
    if !cli.mcts_prior_temperature.is_finite() || cli.mcts_prior_temperature <= 0.0 {
        bail!("--mcts-prior-temperature must be finite and positive");
    }
    if !cli.mcts_prior_noise.is_finite() || cli.mcts_prior_noise < 0.0 {
        bail!("--mcts-prior-noise must be finite and non-negative");
    }
    if cli.max_height > StandardTetris::ROWS as u32 {
        bail!("--max-height must be <= {}", StandardTetris::ROWS);
    }
    Ok(())
}

fn scan_all_tuples(cli: &Cli) -> Result<()> {
    let started = Instant::now();
    let mut stats = ScanStats {
        best_cells: u32::MAX,
        best_score: f32::NEG_INFINITY,
        ..ScanStats::default()
    };
    let mut ordinal = cli.start;
    let scan_limit = cli
        .limit
        .map(u128::from)
        .unwrap_or(TOTAL_FIVE_BAG_SEQUENCES - cli.start);
    let end = cli
        .start
        .saturating_add(scan_limit)
        .min(TOTAL_FIVE_BAG_SEQUENCES);
    let scan_total = end - cli.start;

    println!("scan_start               = {}", cli.start);
    println!("scan_end_exclusive       = {end}");
    println!("scan_limit               = {scan_total}");
    println!();

    while ordinal < end {
        let scan_position = ScanPosition {
            item: u128::from(stats.scanned) + 1,
            total: scan_total,
            ordinal: Some(ordinal),
        };
        let tuple = bag_tuple_from_ordinal(ordinal);
        let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
        fill_five_bag_sequence(tuple, &mut pieces);

        let attempt = run_puct_for_sequence(&pieces, cli, ordinal_seed(ordinal), scan_position);
        stats.record_attempt(&attempt);

        match attempt {
            PuctAttempt::Witness(witness) => {
                println!();
                println!("FOUND witness");
                println!(
                    "scan_item    = {}/{}",
                    scan_position.item, scan_position.total
                );
                println!("ordinal      = {ordinal}");
                print_bag_tuple(tuple);
                print_piece_sequence(&pieces);
                print_witness(&pieces, &witness, cli.print_boards);
                if !cli.keep_going {
                    stats.scanned += 1;
                    print_summary(&stats, started);
                    return Ok(());
                }
            }
            PuctAttempt::Failure(failure) => {
                if cli.print_every_attempt {
                    println!();
                    println!("ATTEMPT failed");
                    println!(
                        "scan_item    = {}/{}",
                        scan_position.item, scan_position.total
                    );
                    println!("ordinal      = {ordinal}");
                    print_bag_tuple(tuple);
                    print_piece_sequence(&pieces);
                    print_failure_prefix(&pieces, &failure);
                }
            }
        }

        stats.scanned += 1;
        ordinal += 1;

        if cli.progress_every > 0 && stats.scanned % cli.progress_every == 0 {
            print_progress(&stats, scan_total, ordinal, started);
        }
    }

    print_summary(&stats, started);
    Ok(())
}

fn run_one_tuple(cli: &Cli, tuple: [usize; BAG_COUNT]) -> Result<()> {
    let started = Instant::now();
    let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
    fill_five_bag_sequence(tuple, &mut pieces);

    print_bag_tuple(tuple);
    print_piece_sequence(&pieces);

    match run_puct_for_sequence(&pieces, cli, tuple_seed(tuple), ScanPosition::single()) {
        PuctAttempt::Witness(witness) => {
            println!("FOUND witness");
            println!("scan_item    = 1/1");
            print_witness(&pieces, &witness, cli.print_boards);
        }
        PuctAttempt::Failure(failure) => {
            println!("NO witness found with configured PUCT budget");
            println!("scan_item    = 1/1");
            print_failure_prefix(&pieces, &failure);
        }
    }

    println!("time = {:.2?}", started.elapsed());
    Ok(())
}

fn run_puct_for_sequence(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    cli: &Cli,
    sequence_seed: u64,
    scan_position: ScanPosition,
) -> PuctAttempt {
    let mut aggregate = PuctFailure {
        placements: [DEFAULT_PLACEMENT; TOTAL_PIECES],
        prefix_len: 0,
        best_cells: u32::MAX,
        best_score: f32::NEG_INFINITY,
        iterations: 0,
        tree_nodes: 0,
        terminal_evaluations: 0,
    };

    for restart in 0..cli.mcts_restarts {
        let config = PuctConfig {
            max_height: cli.max_height,
            iterations: cli.mcts_iterations,
            exploration: cli.mcts_exploration,
            prior_temperature: cli.mcts_prior_temperature,
            prior_noise: cli.mcts_prior_noise,
            max_children: cli
                .mcts_max_children
                .min(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT),
            noise_seed: mix_seed(sequence_seed, restart),
        };

        let result = PuctSearch::new(pieces, config).search(cli.mcts_progress_every, |progress| {
            print_mcts_progress(scan_position, restart, cli.mcts_restarts, progress);
        });

        aggregate.iterations += result.iterations;
        aggregate.tree_nodes += result.tree_nodes;
        aggregate.terminal_evaluations += result.terminal_evaluations;
        aggregate.best_cells = aggregate.best_cells.min(result.best_cells);
        if result.best_prefix_len > aggregate.prefix_len
            || (result.best_prefix_len == aggregate.prefix_len
                && result.best_score > aggregate.best_score)
        {
            aggregate.placements = result.best_placements;
            aggregate.prefix_len = result.best_prefix_len;
            aggregate.best_score = result.best_score;
        }

        if let Some(placements) = result.witness_placements {
            if let Some(mut witness) = replay_witness(pieces, &placements, cli.max_height) {
                witness.iterations = aggregate.iterations;
                witness.tree_nodes = aggregate.tree_nodes;
                witness.terminal_evaluations = aggregate.terminal_evaluations;
                witness.restart = restart + 1;
                witness.best_score = result.best_score;
                return PuctAttempt::Witness(witness);
            }
        }
    }

    PuctAttempt::Failure(aggregate)
}

fn parse_bag_tuple(input: &str) -> Result<[usize; BAG_COUNT]> {
    let mut tuple = [0usize; BAG_COUNT];
    let mut count = 0usize;

    for raw in input.split(',') {
        if count == BAG_COUNT {
            bail!("--bags expects exactly {BAG_COUNT} comma-separated indices");
        }

        let token = raw.trim();
        if token.is_empty() {
            bail!("empty bag index in --bags={input}");
        }

        let idx = token
            .parse::<usize>()
            .with_context(|| format!("invalid bag permutation index `{token}`"))?;
        if idx >= BAG_PERMUTATION_COUNT {
            bail!(
                "bag permutation index {idx} is outside 0..{}",
                BAG_PERMUTATION_COUNT - 1
            );
        }

        tuple[count] = idx;
        count += 1;
    }

    if count != BAG_COUNT {
        bail!("--bags expects exactly {BAG_COUNT} comma-separated indices");
    }

    Ok(tuple)
}

fn bag_tuple_from_ordinal(mut ordinal: u128) -> [usize; BAG_COUNT] {
    let mut tuple = [0usize; BAG_COUNT];
    let base = BAG_PERMUTATION_COUNT as u128;
    let mut idx = 0usize;
    while idx < BAG_COUNT {
        tuple[idx] = (ordinal % base) as usize;
        ordinal /= base;
        idx += 1;
    }
    tuple
}

fn fill_five_bag_sequence(tuple: [usize; BAG_COUNT], out: &mut [TetrisPiece; TOTAL_PIECES]) {
    for (bag_idx, &perm_idx) in tuple.iter().enumerate() {
        let bag = ALL_BAG_PERMUTATIONS[perm_idx];
        let start = bag_idx * PIECES_PER_BAG;
        out[start..start + PIECES_PER_BAG].copy_from_slice(&bag);
    }
}

fn state_key(state: SearchState) -> StateKey {
    StateKey {
        board: state.board,
        step: state.step as u8,
    }
}

fn scorer_prior(board: TetrisBoard, recent_lines_cleared: u32) -> f32 {
    height_mse_board_score(&TetrisBoardScoreState {
        board,
        recent_lines_cleared,
    })
}

fn nonterminal_value(board: TetrisBoard, step: usize, total_steps: usize) -> f32 {
    let scorer_value = score_value_from_empty_distance(board);
    let depth_value = step as f32 / total_steps.max(1) as f32;
    (scorer_value * 0.85 + depth_value * 0.15).clamp(-1.0, 0.99)
}

fn terminal_failure_value(board: TetrisBoard) -> f32 {
    let distance = height_mse_distance_from_empty(board);
    let cells = board.count();
    let height = board.height();
    let holes = board.total_holes();
    (-1.0 - cells as f32 / 40.0 - distance / 400.0 - height as f32 / 80.0 - holes as f32 / 80.0)
        .max(-8.0)
}

fn score_value_from_empty_distance(board: TetrisBoard) -> f32 {
    let distance = height_mse_distance_from_empty(board);
    (1.0 - distance / 200.0).clamp(-1.0, 1.0)
}

fn normalized_priors(
    candidates: &[Candidate; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT],
    len: usize,
    temperature: f32,
) -> [f32; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT] {
    let mut priors = [0.0f32; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT];
    if len == 0 {
        return priors;
    }

    let max_logit = candidates[..len]
        .iter()
        .map(|candidate| candidate.prior_score / temperature)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut total = 0.0f32;
    for idx in 0..len {
        let weight = (candidates[idx].prior_score / temperature - max_logit).exp();
        let finite_weight = if weight.is_finite() { weight } else { 0.0 };
        priors[idx] = finite_weight;
        total += finite_weight;
    }

    if total <= f32::EPSILON {
        let uniform = 1.0 / len as f32;
        for prior in priors.iter_mut().take(len) {
            *prior = uniform;
        }
        return priors;
    }

    for prior in priors.iter_mut().take(len) {
        *prior /= total;
    }

    priors
}

fn prior_noise(state: SearchState, placement: TetrisPiecePlacement, seed: u64) -> f32 {
    let mut hasher = FxHasher::default();
    seed.hash(&mut hasher);
    state.board.hash(&mut hasher);
    state.step.hash(&mut hasher);
    placement.index().hash(&mut hasher);
    let bits = hasher.finish();
    let unit = (bits as f64 / u64::MAX as f64) as f32;
    unit * 2.0 - 1.0
}

fn ordinal_seed(ordinal: u128) -> u64 {
    let lo = ordinal as u64;
    let hi = (ordinal >> 64) as u64;
    mix_seed(lo, hi)
}

fn tuple_seed(tuple: [usize; BAG_COUNT]) -> u64 {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for bag_idx in tuple {
        seed = mix_seed(seed, bag_idx as u64);
    }
    seed
}

fn mix_seed(left: u64, right: u64) -> u64 {
    let mut x = left ^ right.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn replay_witness(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    placements: &[TetrisPiecePlacement; TOTAL_PIECES],
    max_height: u32,
) -> Option<FiveBagWitness> {
    let mut board: TetrisBoard = TetrisBoard::EMPTY_BOARD;
    let mut lines_cleared = 0u32;
    let mut observed_max_height = 0u32;

    for (idx, placement) in placements.iter().copied().enumerate() {
        if placement.piece != pieces[idx] {
            return None;
        }

        let result = board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            return None;
        }

        let height = board.height();
        if height > max_height {
            return None;
        }

        observed_max_height = observed_max_height.max(height);
        lines_cleared += result.lines_cleared;
    }

    if board.count() != 0 || lines_cleared != 14 {
        return None;
    }

    Some(FiveBagWitness {
        placements: *placements,
        lines_cleared,
        max_height: observed_max_height,
        iterations: 0,
        tree_nodes: 0,
        terminal_evaluations: 0,
        restart: 0,
        best_score: f32::NEG_INFINITY,
    })
}

fn print_progress(stats: &ScanStats, scan_total: u128, next_ordinal: u128, started: Instant) {
    let elapsed = started.elapsed().as_secs_f64();
    let rate = stats.scanned as f64 / elapsed.max(1e-9);
    let next_item = (u128::from(stats.scanned) + 1).min(scan_total);
    println!(
        "[scan] elapsed={elapsed:.1}s completed={}/{} next_item={}/{} next_ordinal={} found={} rate={rate:.2}/s mcts_iters={} terminals={} tree_nodes={} best_cells={} best_score={:.4}",
        stats.scanned,
        scan_total,
        next_item,
        scan_total,
        next_ordinal,
        stats.found,
        stats.mcts_iterations,
        stats.terminal_evaluations,
        stats.tree_nodes,
        display_best_cells(stats.best_cells),
        stats.best_score,
    );
}

fn print_mcts_progress(
    scan_position: ScanPosition,
    restart: u64,
    total_restarts: u64,
    progress: PuctProgress,
) {
    if let Some(ordinal) = scan_position.ordinal {
        println!(
            "[mcts] item={}/{} ordinal={} restart={}/{} iter={} nodes={} terminals={} best_prefix={} best_cells={} best_score={:.4}",
            scan_position.item,
            scan_position.total,
            ordinal,
            restart + 1,
            total_restarts,
            progress.iterations,
            progress.tree_nodes,
            progress.terminal_evaluations,
            progress.best_prefix_len,
            display_best_cells(progress.best_cells),
            progress.best_score,
        );
    } else {
        println!(
            "[mcts] item={}/{} restart={}/{} iter={} nodes={} terminals={} best_prefix={} best_cells={} best_score={:.4}",
            scan_position.item,
            scan_position.total,
            restart + 1,
            total_restarts,
            progress.iterations,
            progress.tree_nodes,
            progress.terminal_evaluations,
            progress.best_prefix_len,
            display_best_cells(progress.best_cells),
            progress.best_score,
        );
    }
}

fn print_summary(stats: &ScanStats, started: Instant) {
    let elapsed = started.elapsed().as_secs_f64();
    let rate = stats.scanned as f64 / elapsed.max(1e-9);
    println!();
    println!("--- summary ---");
    println!("scanned               = {}", stats.scanned);
    println!("found                 = {}", stats.found);
    println!("mcts_iterations       = {}", stats.mcts_iterations);
    println!("terminal_evaluations  = {}", stats.terminal_evaluations);
    println!("tree_nodes            = {}", stats.tree_nodes);
    println!(
        "best_cells            = {}",
        display_best_cells(stats.best_cells)
    );
    println!("best_score            = {:.4}", stats.best_score);
    println!("time                  = {elapsed:.2}s");
    println!("rate                  = {rate:.2}/s");
}

fn print_bag_tuple(tuple: [usize; BAG_COUNT]) {
    print!("bags         = ");
    for (idx, bag) in tuple.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{bag}");
    }
    println!();
}

fn print_piece_sequence(pieces: &[TetrisPiece; TOTAL_PIECES]) {
    print!("pieces       = ");
    for (idx, piece) in pieces.iter().enumerate() {
        if idx > 0 && idx % PIECES_PER_BAG == 0 {
            print!("| ");
        }
        print!("{piece} ");
    }
    println!();
}

fn print_witness(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    witness: &FiveBagWitness,
    print_boards: bool,
) {
    println!("lines_cleared       = {}", witness.lines_cleared);
    println!("max_height          = {}", witness.max_height);
    println!("restart             = {}", witness.restart);
    println!("mcts_iterations     = {}", witness.iterations);
    println!("tree_nodes          = {}", witness.tree_nodes);
    println!("terminal_evals      = {}", witness.terminal_evaluations);
    println!("best_score          = {:.4}", witness.best_score);
    print_placement_indices("placement_indices", &witness.placements, TOTAL_PIECES);
    println!("placements:");

    let mut board = TetrisBoard::EMPTY_BOARD;
    for (idx, placement) in witness.placements.iter().copied().enumerate() {
        let result = board.apply_piece_placement(placement);
        println!(
            "  step={idx:02} piece={} placement_idx={} placement={} lines={} cells={} height={} score={:.3}",
            pieces[idx],
            placement.index(),
            placement,
            result.lines_cleared,
            board.count(),
            board.height(),
            scorer_prior(board, result.lines_cleared),
        );
        if print_boards {
            println!("{board}");
        }
    }
}

fn print_failure_prefix(pieces: &[TetrisPiece; TOTAL_PIECES], failure: &PuctFailure) {
    println!("best_prefix_len     = {}", failure.prefix_len);
    println!(
        "best_terminal_cells = {}",
        display_best_cells(failure.best_cells)
    );
    println!("best_score          = {:.4}", failure.best_score);
    println!("mcts_iterations     = {}", failure.iterations);
    println!("tree_nodes          = {}", failure.tree_nodes);
    println!("terminal_evals      = {}", failure.terminal_evaluations);
    print_placement_indices(
        "placement_prefix_indices",
        &failure.placements,
        failure.prefix_len,
    );
    if failure.prefix_len == 0 {
        return;
    }

    println!("placement_prefix:");
    let mut board = TetrisBoard::EMPTY_BOARD;
    for (idx, placement) in failure
        .placements
        .iter()
        .copied()
        .take(failure.prefix_len)
        .enumerate()
    {
        let result = board.apply_piece_placement(placement);
        println!(
            "  step={idx:02} piece={} placement_idx={} placement={} lines={} cells={} height={} score={:.3}",
            pieces[idx],
            placement.index(),
            placement,
            result.lines_cleared,
            board.count(),
            board.height(),
            scorer_prior(board, result.lines_cleared),
        );
    }
}

fn print_placement_indices(
    label: &str,
    placements: &[TetrisPiecePlacement; TOTAL_PIECES],
    len: usize,
) {
    print!("{label} = ");
    if len == 0 {
        println!("[]");
        return;
    }

    for (idx, placement) in placements.iter().take(len).enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!("{}", placement.index());
    }
    println!();
}

fn display_best_cells(best_cells: u32) -> String {
    if best_cells == u32::MAX {
        "none".to_owned()
    } else {
        best_cells.to_string()
    }
}

fn display_optional_iterations(iterations: Option<u64>) -> String {
    match iterations {
        Some(iterations) => iterations.to_string(),
        None => "unbounded".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PuctConfig {
        PuctConfig {
            max_height: StandardTetris::ROWS as u32,
            iterations: Some(8),
            exploration: 1.4,
            prior_temperature: 64.0,
            prior_noise: 0.0,
            max_children: TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT,
            noise_seed: 1,
        }
    }

    #[test]
    fn permutation_table_has_expected_edges() {
        assert_eq!(ALL_BAG_PERMUTATIONS.len(), BAG_PERMUTATION_COUNT);
        assert_eq!(
            ALL_BAG_PERMUTATIONS[0],
            [
                TetrisPiece::O_PIECE,
                TetrisPiece::I_PIECE,
                TetrisPiece::S_PIECE,
                TetrisPiece::Z_PIECE,
                TetrisPiece::T_PIECE,
                TetrisPiece::L_PIECE,
                TetrisPiece::J_PIECE,
            ]
        );
        assert_eq!(
            ALL_BAG_PERMUTATIONS[BAG_PERMUTATION_COUNT - 1],
            [
                TetrisPiece::J_PIECE,
                TetrisPiece::L_PIECE,
                TetrisPiece::T_PIECE,
                TetrisPiece::Z_PIECE,
                TetrisPiece::S_PIECE,
                TetrisPiece::I_PIECE,
                TetrisPiece::O_PIECE,
            ]
        );
    }

    #[test]
    fn ordinal_mapping_is_base_5040() {
        assert_eq!(bag_tuple_from_ordinal(0), [0, 0, 0, 0, 0]);
        assert_eq!(bag_tuple_from_ordinal(1), [1, 0, 0, 0, 0]);
        assert_eq!(
            bag_tuple_from_ordinal(BAG_PERMUTATION_COUNT as u128),
            [0, 1, 0, 0, 0]
        );
    }

    #[test]
    fn explicit_tuple_fills_sequence() {
        let tuple = [0, 1, 2, 3, 4];
        let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
        fill_five_bag_sequence(tuple, &mut pieces);
        assert_eq!(&pieces[0..PIECES_PER_BAG], &ALL_BAG_PERMUTATIONS[0]);
        assert_eq!(
            &pieces[PIECES_PER_BAG..2 * PIECES_PER_BAG],
            &ALL_BAG_PERMUTATIONS[1]
        );
    }

    #[test]
    fn scorer_prefers_empty_board() {
        let empty = TetrisBoard::EMPTY_BOARD;
        let mut occupied = TetrisBoard::EMPTY_BOARD;
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let result = occupied.apply_piece_placement(placement);
        assert_eq!(result.is_lost, IsLost::NOT_LOST);

        assert!(scorer_prior(empty, 0) > scorer_prior(occupied, result.lines_cleared));
        assert!(score_value_from_empty_distance(empty) > score_value_from_empty_distance(occupied));
    }

    #[test]
    fn normalized_priors_prefer_higher_scores() {
        let mut candidates = [DEFAULT_CANDIDATE; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT];
        candidates[0].prior_score = 0.0;
        candidates[1].prior_score = -100.0;

        let priors = normalized_priors(&candidates, 2, 64.0);
        assert!(priors[0] > priors[1]);
        assert!((priors[0] + priors[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn puct_accepts_zero_piece_empty_target() {
        let result = PuctSearch::new(&[], test_config()).search(0, |_| {});
        assert!(result.witness_placements.is_some());
        assert_eq!(result.best_cells, 0);
    }
}
