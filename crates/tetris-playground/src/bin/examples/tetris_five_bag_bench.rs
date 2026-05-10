//! Benchmark: solve 10 fixed random 5-bag sequences and report total time.
//!
//! This binary is the autoresearch verification target.
//! It picks 10 deterministic random ordinals (seeded from 0xCAFE_BABE),
//! const-computes the full piece sequences at compile time, runs the
//! PUCT/MCTS search on each (single-threaded, serial), and prints the
//! total elapsed seconds as the last line of output.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rustc_hash::FxHasher;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_search::{TetrisBoardScoreState, height_mse_board_score, height_mse_distance_from_empty};

const BAG_COUNT: usize = 5;
const PIECES_PER_BAG: usize = 7;
const TOTAL_PIECES: usize = BAG_COUNT * PIECES_PER_BAG;
const BAG_PERMUTATION_COUNT: usize = 5_040;
const NUM_SEQUENCES: usize = 10;

const fn pow_usize_as_u128(base: usize, exponent: usize) -> u128 {
    let mut out = 1u128;
    let mut i = 0usize;
    while i < exponent {
        out *= base as u128;
        i += 1;
    }
    out
}

const TOTAL_FIVE_BAG_SEQUENCES: u128 = pow_usize_as_u128(BAG_PERMUTATION_COUNT, BAG_COUNT);

const fn splitmix64(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (s, z)
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

type ForcedBag = [TetrisPiece; PIECES_PER_BAG];

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

const fn const_bag_tuple_from_ordinal(mut ordinal: u128) -> [usize; BAG_COUNT] {
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

const fn const_pieces_from_ordinal(ordinal: u128) -> [TetrisPiece; TOTAL_PIECES] {
    let tuple = const_bag_tuple_from_ordinal(ordinal);
    let perms = generate_forced_bag_permutations();
    let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
    let mut bag_idx = 0;
    while bag_idx < BAG_COUNT {
        let bag = perms[tuple[bag_idx]];
        let mut p = 0;
        while p < PIECES_PER_BAG {
            pieces[bag_idx * PIECES_PER_BAG + p] = bag[p];
            p += 1;
        }
        bag_idx += 1;
    }
    pieces
}

/// 10 random piece sequences, const-computed from seed 0xCAFE_BABE.
/// Never change this — it defines the benchmark workload.
const BENCH_SEQUENCES: [[TetrisPiece; TOTAL_PIECES]; NUM_SEQUENCES] = {
    let mut sequences = [[TetrisPiece::O_PIECE; TOTAL_PIECES]; NUM_SEQUENCES];
    let mut state = 0xCAFE_BABE_u64;
    let mut i = 0;
    while i < NUM_SEQUENCES {
        let (s1, lo) = splitmix64(state);
        let (s2, hi) = splitmix64(s1);
        state = s2;
        let raw = ((hi as u128) << 64) | (lo as u128);
        let ordinal = raw % TOTAL_FIVE_BAG_SEQUENCES;
        sequences[i] = const_pieces_from_ordinal(ordinal);
        i += 1;
    }
    sequences
};

// --- PUCT/MCTS search ---

const BEAM_ROLLOUT_WIDTH: usize = 8;
const BEAM_ROLLOUT_MIN_DEPTH: u8 = 28; // only do rollout for last bag (step >= 28)

const MCTS_EXPLORATION: f32 = 1.0;
const MAX_HEIGHT: u32 = TetrisBoard::HEIGHT as u32;
const MAX_CHILDREN: usize = 20;
const PRIOR_NOISE_AMPLITUDE: f32 = 1.0;

const RANK_PRIORS: [f32; MAX_CHILDREN] = {
    let mut raw = [0.0f32; MAX_CHILDREN];
    let mut sum = 0.0f32;
    let mut i = 0;
    while i < MAX_CHILDREN {
        raw[i] = 1.0 / (i as f32 + 1.0);
        sum += raw[i];
        i += 1;
    }
    let mut j = 0;
    while j < MAX_CHILDREN {
        raw[j] /= sum;
        j += 1;
    }
    raw
};

const DEFAULT_PLACEMENT: TetrisPiecePlacement = TetrisPiecePlacement::from_index(0);
const DEFAULT_CANDIDATE: Candidate = Candidate {
    placement: DEFAULT_PLACEMENT,
    board: TetrisBoard::new(),
    prior_score: f32::NEG_INFINITY,
};

#[derive(Clone, Copy, Debug)]
struct SearchState {
    board: TetrisBoard,
    step: u8,
}

#[derive(Clone, Copy, Debug)]
struct SearchNode {
    state: SearchState,
    visits: u32,
    value_sum: f32,
    expanded: bool,
    children_start: u32,  // index into edge arena
    children_len: u16,
}

impl SearchNode {
    fn new(state: SearchState) -> Self {
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            children_start: 0,
            children_len: 0,
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

struct SearchResult {
    placements: Option<[TetrisPiecePlacement; TOTAL_PIECES]>,
    iterations: u64,
    restarts: u32,
    nodes: usize,
}

struct PuctSearch<'a> {
    pieces: &'a [TetrisPiece],
    nodes: Vec<SearchNode>,
    edges: Vec<ActionEdge>,  // flat arena for all edges
    witness_placements: Option<[TetrisPiecePlacement; TOTAL_PIECES]>,
    noise_seed: u64,
    // Reusable scratch space (avoids re-init each call)
    path_edges: [(usize, usize); TOTAL_PIECES],
    path_placements: [TetrisPiecePlacement; TOTAL_PIECES],
    expand_candidates: [Candidate; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT],
    beam_scratch: BeamRolloutScratch,
}

impl<'a> PuctSearch<'a> {
    fn new(pieces: &'a [TetrisPiece]) -> Self {
        let root_state = SearchState {
            board: TetrisBoard::new(),
            step: 0u8,
        };
        let mut nodes = Vec::with_capacity(64 * 1024);
        nodes.push(SearchNode::new(root_state));
        Self {
            pieces,
            nodes,
            edges: Vec::with_capacity(64 * 1024),
            witness_placements: None,
            noise_seed: 0,
            path_edges: [(usize::MAX, usize::MAX); TOTAL_PIECES],
            path_placements: [DEFAULT_PLACEMENT; TOTAL_PIECES],
            expand_candidates: [DEFAULT_CANDIDATE; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT],
            beam_scratch: BeamRolloutScratch::new(),
        }
    }

    /// Clear the tree for a new restart, reusing allocated memory.
    fn reset(&mut self, noise_seed: u64) {
        self.nodes.clear();
        self.edges.clear();
        self.nodes.push(SearchNode::new(SearchState {
            board: TetrisBoard::new(),
            step: 0u8,
        }));
        self.witness_placements = None;
        self.noise_seed = noise_seed;
    }

    /// Run MCTS with noise-based restarts until deadline or solution found.
    fn search_until_deadline(
        &mut self,
        deadline: Instant,
        max_iters_per_restart: u64,
        max_restarts: u32,
    ) -> SearchResult {
        let mut total_iters = 0u64;
        let mut total_nodes = 0usize;
        for restart in 0..max_restarts {
            let noise_seed = if restart == 0 { 0 } else { restart as u64 };
            self.reset(noise_seed);
            for iter in 0..max_iters_per_restart {
                self.run_iteration();
                if self.witness_placements.is_some() {
                    total_iters += iter + 1;
                    total_nodes += self.nodes.len();
                    return SearchResult {
                        placements: self.witness_placements,
                        iterations: total_iters,
                        restarts: restart + 1,
                        nodes: total_nodes,
                    };
                }
                if iter & 0x1FFF == 0x1FFF && Instant::now() >= deadline {
                    total_iters += iter + 1;
                    total_nodes += self.nodes.len();
                    return SearchResult {
                        placements: None,
                        iterations: total_iters,
                        restarts: restart + 1,
                        nodes: total_nodes,
                    };
                }
            }
            total_iters += max_iters_per_restart;
            total_nodes += self.nodes.len();
            if Instant::now() >= deadline {
                return SearchResult {
                    placements: None,
                    iterations: total_iters,
                    restarts: restart + 1,
                    nodes: total_nodes,
                };
            }
        }
        SearchResult {
            placements: None,
            iterations: total_iters,
            restarts: max_restarts,
            nodes: total_nodes,
        }
    }

    #[inline(always)]
    fn node_edge(&self, node_id: usize, edge_idx: usize) -> &ActionEdge {
        &self.edges[self.nodes[node_id].children_start as usize + edge_idx]
    }

    #[inline(always)]
    fn node_edge_mut(&mut self, node_id: usize, edge_idx: usize) -> &mut ActionEdge {
        &mut self.edges[self.nodes[node_id].children_start as usize + edge_idx]
    }

    fn run_iteration(&mut self) {
        let mut node_id = 0usize;
        let mut path_len = 0usize;

        let value = loop {
            if self.is_terminal(node_id) {
                break self.evaluate_terminal(node_id, path_len);
            }

            if !self.nodes[node_id].expanded {
                if self.expand_node(node_id) {
                    break self.evaluate_leaf(node_id, path_len);
                }
                break -1.0;
            }

            if self.nodes[node_id].children_len == 0 {
                break -1.0;
            }

            let edge_id = self.select_child(node_id);
            let edge = *self.node_edge(node_id, edge_id);
            if path_len == TOTAL_PIECES {
                break -1.0;
            }

            self.path_edges[path_len] = (node_id, edge_id);
            self.path_placements[path_len] = edge.placement;
            path_len += 1;
            node_id = edge.child;
        };

        self.nodes[node_id].visits = self.nodes[node_id].visits.saturating_add(1);
        self.nodes[node_id].value_sum += value;

        for i in 0..path_len {
            let (parent_id, edge_id) = self.path_edges[i];
            self.nodes[parent_id].visits = self.nodes[parent_id].visits.saturating_add(1);
            self.nodes[parent_id].value_sum += value;

            let edge = self.node_edge_mut(parent_id, edge_id);
            edge.visits = edge.visits.saturating_add(1);
            edge.value_sum += value;
        }
    }

    fn is_terminal(&self, node_id: usize) -> bool {
        self.nodes[node_id].state.step as usize == self.pieces.len()
    }

    fn evaluate_terminal(&mut self, node_id: usize, prefix_len: usize) -> f32 {
        let board = self.nodes[node_id].state.board;
        if board.count() == 0 && prefix_len == self.pieces.len() {
            self.witness_placements = Some(self.path_placements);
            return 1.0;
        }
        if board.count() == 0 { 1.0 } else { terminal_failure_value(board) }
    }

    fn evaluate_leaf(&mut self, node_id: usize, path_len: usize) -> f32 {
        let state = self.nodes[node_id].state;

        // For deep nodes, use beam rollout for better value estimate
        if state.step >= BEAM_ROLLOUT_MIN_DEPTH {
            let (value, witness) = self.beam_scratch.rollout(
                state.board,
                state.step as usize,
                self.pieces,
                &self.path_placements,
            );
            if let Some(placements) = witness {
                if path_len == state.step as usize {
                    self.witness_placements = Some(placements);
                }
            }
            return value;
        }

        nonterminal_value(state.board, state.step as usize, self.pieces.len())
    }

    fn expand_node(&mut self, node_id: usize) -> bool {
        let state = self.nodes[node_id].state;
        let piece = self.pieces[state.step as usize];
        let mut candidate_len = 0usize;

        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next = state.board;
            let result = next.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST || next.height() > MAX_HEIGHT {
                continue;
            }

            let mut prior_score = scorer_prior(next, result.lines_cleared);

            // Add deterministic noise to break ties differently per restart.
            if self.noise_seed != 0 {
                prior_score +=
                    PRIOR_NOISE_AMPLITUDE * prior_noise(state, placement, self.noise_seed);
            }

            self.expand_candidates[candidate_len] = Candidate {
                placement,
                board: next,
                prior_score,
            };
            candidate_len += 1;
        }

        self.expand_candidates[..candidate_len].sort_unstable_by(|left, right| {
            right
                .prior_score
                .partial_cmp(&left.prior_score)
                .unwrap_or(Ordering::Equal)
        });

        let child_len = candidate_len
            .min(MAX_CHILDREN)
            .min(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT);

        let edges_start = self.edges.len() as u32;
        for idx in 0..child_len {
            let candidate = self.expand_candidates[idx];
            let child_state = SearchState {
                board: candidate.board,
                step: state.step + 1u8,
            };
            let child = self.nodes.len();
            self.nodes.push(SearchNode::new(child_state));

            self.edges.push(ActionEdge {
                placement: candidate.placement,
                child,
                visits: 0,
                value_sum: 0.0,
                prior: RANK_PRIORS[idx],
            });
        }

        self.nodes[node_id].children_start = edges_start;
        self.nodes[node_id].children_len = child_len as u16;
        self.nodes[node_id].expanded = true;
        child_len > 0
    }

    fn select_child(&self, node_id: usize) -> usize {
        let node = &self.nodes[node_id];
        let parent_visits = node.visits.max(1) as f32;
        let exploration_scale = MCTS_EXPLORATION * parent_visits.sqrt();
        let start = node.children_start as usize;
        let len = node.children_len as usize;
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for idx in 0..len {
            let edge = &self.edges[start + idx];
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

/// Pre-allocated scratch memory for beam rollout.
const MAX_BEAM_CANDS: usize = BEAM_ROLLOUT_WIDTH * TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT;

struct BeamRolloutScratch {
    beam_boards: [TetrisBoard; BEAM_ROLLOUT_WIDTH],
    beam_parents: [[u8; TOTAL_PIECES]; BEAM_ROLLOUT_WIDTH],  // parent index at each step
    beam_actions: [[TetrisPiecePlacement; TOTAL_PIECES]; BEAM_ROLLOUT_WIDTH],
    cand_boards: [TetrisBoard; MAX_BEAM_CANDS],
    cand_scores: [f32; MAX_BEAM_CANDS],
    cand_parent: [u8; MAX_BEAM_CANDS],
    cand_action: [TetrisPiecePlacement; MAX_BEAM_CANDS],
}

impl BeamRolloutScratch {
    fn new() -> Self {
        Self {
            beam_boards: [TetrisBoard::new(); BEAM_ROLLOUT_WIDTH],
            beam_parents: [[0u8; TOTAL_PIECES]; BEAM_ROLLOUT_WIDTH],
            beam_actions: [[DEFAULT_PLACEMENT; TOTAL_PIECES]; BEAM_ROLLOUT_WIDTH],
            cand_boards: [TetrisBoard::new(); MAX_BEAM_CANDS],
            cand_scores: [f32::NEG_INFINITY; MAX_BEAM_CANDS],
            cand_parent: [0u8; MAX_BEAM_CANDS],
            cand_action: [DEFAULT_PLACEMENT; MAX_BEAM_CANDS],
        }
    }

    /// Run beam rollout from (board, start_step) over remaining pieces.
    /// Returns (value, optional witness placements).
    fn rollout(
        &mut self,
        board: TetrisBoard,
        start_step: usize,
        pieces: &[TetrisPiece],
        path_so_far: &[TetrisPiecePlacement; TOTAL_PIECES],
    ) -> (f32, Option<[TetrisPiecePlacement; TOTAL_PIECES]>) {
        const K: usize = BEAM_ROLLOUT_WIDTH;
        let mut beam_len = 1usize;

        self.beam_boards[0] = board;
        // Copy the PUCT path for steps 0..start_step into beam element 0
        self.beam_actions[0] = *path_so_far;

        for step in start_step..pieces.len() {
            let piece = pieces[step];
            let mut cand_len = 0usize;

            for bi in 0..beam_len {
                for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                    let mut next = self.beam_boards[bi];
                    let result = next.apply_piece_placement(placement);
                    if result.is_lost == IsLost::LOST || next.height() > MAX_HEIGHT {
                        continue;
                    }
                    if cand_len < MAX_BEAM_CANDS {
                        self.cand_boards[cand_len] = next;
                        self.cand_scores[cand_len] = scorer_prior(next, result.lines_cleared);
                        self.cand_parent[cand_len] = bi as u8;
                        self.cand_action[cand_len] = placement;
                        cand_len += 1;
                    }
                }
            }

            if cand_len == 0 {
                return (-1.0, None);
            }

            let keep = cand_len.min(K);
            for i in 0..keep {
                let mut best = i;
                for j in (i + 1)..cand_len {
                    if self.cand_scores[j] > self.cand_scores[best] {
                        best = j;
                    }
                }
                if best != i {
                    self.cand_boards.swap(i, best);
                    self.cand_scores.swap(i, best);
                    self.cand_parent.swap(i, best);
                    self.cand_action.swap(i, best);
                }
            }

            // Promote top-K candidates into beam
            // Copy parent's action history, then append this step's action
            beam_len = keep;
            for i in (0..keep).rev() {
                let pi = self.cand_parent[i] as usize;
                if i != pi {
                    self.beam_actions[i] = self.beam_actions[pi];
                }
                self.beam_actions[i][step] = self.cand_action[i];
                self.beam_boards[i] = self.cand_boards[i];
            }
        }

        // Check for empty board
        for bi in 0..beam_len {
            if self.beam_boards[bi].count() == 0 {
                return (1.0, Some(self.beam_actions[bi]));
            }
        }

        let mut best_cells = u32::MAX;
        for bi in 0..beam_len {
            best_cells = best_cells.min(self.beam_boards[bi].count());
        }
        ((-(best_cells as f32) / 20.0).max(-1.0), None)
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

fn prior_noise(state: SearchState, placement: TetrisPiecePlacement, seed: u64) -> f32 {
    let mut hasher = FxHasher::default();
    seed.hash(&mut hasher);
    state.board.hash(&mut hasher);
    state.step.hash(&mut hasher);
    placement.index().hash(&mut hasher);
    let bits = hasher.finish();
    (bits as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
}

fn replay_witness(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    placements: &[TetrisPiecePlacement; TOTAL_PIECES],
) -> bool {
    let mut board = TetrisBoard::new();
    let mut lines_cleared = 0u32;
    for (idx, placement) in placements.iter().copied().enumerate() {
        if placement.piece != pieces[idx] {
            return false;
        }
        let result = board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            return false;
        }
        lines_cleared += result.lines_cleared;
    }
    board.count() == 0 && lines_cleared == 14
}

const PER_SEQ_TIMEOUT_SECS: u64 = 10;
const GLOBAL_TIMEOUT_SECS: u64 = 360;
const DEFAULT_MAX_ITERS_PER_RESTART: u64 = u64::MAX;
const DEFAULT_MAX_RESTARTS: u32 = u32::MAX;

fn main() {
    let overall_start = Instant::now();
    let global_deadline = overall_start + std::time::Duration::from_secs(GLOBAL_TIMEOUT_SECS);
    let mut solved = 0usize;
    let mut total_solve_time = 0.0f64;

    let mut search = PuctSearch::new(&BENCH_SEQUENCES[0]);

    for (seq_idx, pieces) in BENCH_SEQUENCES.iter().enumerate() {
        if Instant::now() >= global_deadline {
            eprintln!("seq={seq_idx} skipped (global deadline)");
            total_solve_time += PER_SEQ_TIMEOUT_SECS as f64;
            continue;
        }

        search.pieces = pieces;
        let seq_start = Instant::now();
        let seq_deadline = (seq_start + std::time::Duration::from_secs(PER_SEQ_TIMEOUT_SECS))
            .min(global_deadline);
        let result = search.search_until_deadline(
            seq_deadline,
            DEFAULT_MAX_ITERS_PER_RESTART,
            DEFAULT_MAX_RESTARTS,
        );
        let seq_elapsed = seq_start.elapsed().as_secs_f64();

        match result.placements {
            Some(placements) if replay_witness(pieces, &placements) => {
                solved += 1;
                total_solve_time += seq_elapsed;
                eprintln!(
                    "seq={seq_idx} solved=true time={seq_elapsed:.3}s iters={} restarts={} nodes={}",
                    result.iterations, result.restarts, result.nodes,
                );
            }
            _ => {
                total_solve_time += PER_SEQ_TIMEOUT_SECS as f64;
                eprintln!(
                    "seq={seq_idx} solved=false time={seq_elapsed:.3}s iters={} restarts={} nodes={}",
                    result.iterations, result.restarts, result.nodes,
                );
            }
        }
    }

    let total_elapsed = overall_start.elapsed().as_secs_f64();
    eprintln!("solved={solved}/{NUM_SEQUENCES} wall={total_elapsed:.3}s metric={total_solve_time:.3}s");
    println!("{total_solve_time:.3}");
}
