//! 5-bag chain solver: solve one 5-bag, then swap the last bag and reuse
//! the MCTS tree to solve the next sequence, chaining through as many
//! sequences as possible.
//!
//! Strategy:
//! 1. Start with a random 5-bag sequence [B0, B1, B2, B3, B4]
//! 2. Solve it with PUCT/MCTS (empty board → empty board)
//! 3. Swap bag 4 with a new random bag → [B0, B1, B2, B3, B5]
//!    - Invalidate all nodes at step >= 28 (bag 5 territory)
//!    - Keep the tree for steps 0–27 (bags 0–3 are unchanged)
//! 4. Resume MCTS on the modified tree
//! 5. Repeat: after solving, swap bag 4 again
//!
//! The metric is: how many 5-bag sequences can we solve in 6 minutes?

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use clap::{Parser, Subcommand};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal,
};
use rustc_hash::FxHasher;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_search::{
    TetrisBoardScoreState, height_mse_board_score, height_mse_distance_from_empty,
};

const BAG_COUNT: usize = 5;
const PIECES_PER_BAG: usize = 7;
const TOTAL_PIECES: usize = BAG_COUNT * PIECES_PER_BAG;
const BAG_PERMUTATION_COUNT: usize = 5_040;

// --- Bag generation ---

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

static ALL_BAG_PERMUTATIONS: [ForcedBag; BAG_PERMUTATION_COUNT] =
    generate_forced_bag_permutations();

/// Simple splitmix64 PRNG for deterministic bag selection.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_bag_index(&mut self) -> usize {
        (self.next_u64() as usize) % BAG_PERMUTATION_COUNT
    }
}

fn build_sequence(bag_indices: &[usize; BAG_COUNT]) -> [TetrisPiece; TOTAL_PIECES] {
    let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
    for (bag_idx, &perm_idx) in bag_indices.iter().enumerate() {
        let bag = ALL_BAG_PERMUTATIONS[perm_idx];
        let start = bag_idx * PIECES_PER_BAG;
        pieces[start..start + PIECES_PER_BAG].copy_from_slice(&bag);
    }
    pieces
}

// --- PUCT/MCTS search (same as bench, but with tree-pruning support) ---

const MCTS_EXPLORATION: f32 = 0.8;
const MAX_HEIGHT: u32 = TetrisBoard::HEIGHT as u32;
const MAX_CHILDREN: usize = 64;
const PRIOR_NOISE_AMPLITUDE: f32 = 0.2;

/// Max nodes in the arena. Keeps tree cache-friendly.
/// ~2M nodes × ~24 bytes ≈ 48MB — fits comfortably in L3.
const NODE_BUDGET: usize = 2_000_000;

/// What to do when the arena fills up.
/// - `Restart`: clear the tree and start fresh with a new noise seed
/// - `Evict(pct)`: reclaim ~pct% of nodes by evicting random leaves, keep searching
#[derive(Clone, Copy)]
enum FullPolicy {
    Restart,
    Evict(f32), // fraction of NODE_BUDGET to reclaim, e.g. 0.10 = 10%
}

const ARENA_FULL_POLICY: FullPolicy = FullPolicy::Restart;

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
    children_start: u32,
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

/// Filter function: given a board and the step it was reached at,
/// return `true` to keep the board or `false` to prune it.
type FilterFn = fn(board: TetrisBoard, step: u8) -> bool;

struct PuctSearch {
    pieces: [TetrisPiece; TOTAL_PIECES],
    nodes: Vec<SearchNode>,
    edges: Vec<ActionEdge>,
    free_list: Vec<usize>, // recycled node indices from eviction
    witness_placements: Option<[TetrisPiecePlacement; TOTAL_PIECES]>,
    noise_seed: u64,
    filter_fn: Option<FilterFn>,
    // Scratch space
    path_edges: [(usize, usize); TOTAL_PIECES],
    path_placements: [TetrisPiecePlacement; TOTAL_PIECES],
    expand_candidates: [Candidate; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT],
}

impl PuctSearch {
    fn new(pieces: [TetrisPiece; TOTAL_PIECES]) -> Self {
        let root_state = SearchState {
            board: TetrisBoard::new(),
            step: 0u8,
        };
        let mut nodes = Vec::with_capacity(NODE_BUDGET);
        nodes.push(SearchNode::new(root_state));
        Self {
            pieces,
            nodes,
            edges: Vec::with_capacity(NODE_BUDGET * MAX_CHILDREN / 2),
            free_list: Vec::new(),
            witness_placements: None,
            noise_seed: 0,
            filter_fn: None,
            path_edges: [(usize::MAX, usize::MAX); TOTAL_PIECES],
            path_placements: [DEFAULT_PLACEMENT; TOTAL_PIECES],
            expand_candidates: [DEFAULT_CANDIDATE; TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT],
        }
    }

    fn set_filter(&mut self, f: FilterFn) {
        self.filter_fn = Some(f);
    }

    /// Reset for a new sequence, reusing allocated memory.
    fn reset(&mut self, pieces: [TetrisPiece; TOTAL_PIECES]) {
        self.nodes.clear();
        self.edges.clear();
        self.nodes.push(SearchNode::new(SearchState {
            board: TetrisBoard::new(),
            step: 0u8,
        }));
        self.pieces = pieces;
        self.witness_placements = None;
        self.noise_seed = 0;
    }

    /// Swap the last bag in the piece sequence and invalidate all nodes
    /// at step >= `invalidate_from` (i.e., bag index × PIECES_PER_BAG).
    /// Nodes at earlier steps keep their structure and visit stats.
    fn swap_bag_and_prune(&mut self, bag_slot: usize, new_bag_index: usize) {
        let start_piece = bag_slot * PIECES_PER_BAG;
        let bag = ALL_BAG_PERMUTATIONS[new_bag_index];
        self.pieces[start_piece..start_piece + PIECES_PER_BAG].copy_from_slice(&bag);

        // Invalidate: mark all nodes at step >= start_piece as unexpanded
        // and zero out their children. We can't easily reclaim edge arena
        // memory, but we can rebuild from scratch by truncating.
        //
        // Strategy: walk all nodes. For nodes at step >= invalidate_from,
        // mark them dead. For nodes at step < invalidate_from that have
        // children pointing to dead nodes, remove those children.
        let invalidate_from = start_piece as u8;

        // Phase 1: Mark nodes at >= invalidate_from as dead
        let mut dead = vec![false; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if node.state.step >= invalidate_from {
                dead[i] = true;
            }
        }

        // Phase 2: For surviving nodes, remove edges pointing to dead children.
        // We rebuild the edge arena from scratch for simplicity.
        let old_edges = std::mem::take(&mut self.edges);
        self.edges = Vec::with_capacity(old_edges.len() / 2);

        for node in &mut self.nodes {
            if dead[node.state.step as usize / PIECES_PER_BAG * PIECES_PER_BAG]
                && node.state.step >= invalidate_from
            {
                // Dead node — clear it
                node.expanded = false;
                node.children_start = 0;
                node.children_len = 0;
                node.visits = 0;
                node.value_sum = 0.0;
                continue;
            }

            if node.children_len == 0 {
                continue;
            }

            // Copy surviving edges
            let old_start = node.children_start as usize;
            let old_len = node.children_len as usize;
            let new_start = self.edges.len() as u32;
            let mut new_len = 0u16;

            for ei in 0..old_len {
                let edge = old_edges[old_start + ei];
                if edge.child < dead.len() && !dead[edge.child] {
                    self.edges.push(edge);
                    new_len += 1;
                }
            }

            node.children_start = new_start;
            node.children_len = new_len;

            // If all children were pruned, mark as unexpanded so MCTS re-expands
            if new_len == 0 {
                node.expanded = false;
            }
        }

        // Clear witness from previous solve
        self.witness_placements = None;
    }

    /// Returns true if node arena is full.
    fn is_arena_full(&self) -> bool {
        self.nodes.len() >= NODE_BUDGET && self.free_list.is_empty()
    }

    /// Allocate a node, reusing a free slot if available.
    fn alloc_node(&mut self, node: SearchNode) -> Option<usize> {
        if let Some(slot) = self.free_list.pop() {
            self.nodes[slot] = node;
            Some(slot)
        } else if self.nodes.len() < NODE_BUDGET {
            let slot = self.nodes.len();
            self.nodes.push(node);
            Some(slot)
        } else {
            None // arena full, no free slots
        }
    }

    /// Restart the tree with a new noise seed, reusing allocated memory.
    fn restart_with_noise(&mut self, noise_seed: u64) {
        self.nodes.clear();
        self.edges.clear();
        self.free_list.clear();
        self.nodes.push(SearchNode::new(SearchState {
            board: TetrisBoard::new(),
            step: 0u8,
        }));
        self.witness_placements = None;
        self.noise_seed = noise_seed;
    }

    /// Evict leaf nodes to free `target_count` slots.
    /// Only zeros out the nodes and adds to free list — no edge rebuild.
    /// Dead children are harmless: PUCT won't select them (visits=0,
    /// Q=0, no exploration bonus since parent edge visits stay).
    fn evict_leaves(&mut self, target_count: usize) {
        let num_nodes = self.nodes.len();
        let mut evicted = 0usize;

        // Scan backwards — deeper nodes are more likely to be leaves
        let mut i = num_nodes;
        while i > 1 && evicted < target_count {
            i -= 1;
            let node = self.nodes[i];
            if node.visits == 0 && !node.expanded {
                continue; // already dead
            }
            // Only evict true leaves (no children or unexpanded)
            if node.children_len > 0 && node.expanded {
                continue;
            }

            self.nodes[i].visits = 0;
            self.nodes[i].value_sum = 0.0;
            self.nodes[i].expanded = false;
            self.nodes[i].children_start = 0;
            self.nodes[i].children_len = 0;
            self.free_list.push(i);
            evicted += 1;
        }
    }

    /// Handle arena-full condition based on ARENA_FULL_POLICY.
    fn handle_arena_full(&mut self, restart_count: &mut u32) {
        match ARENA_FULL_POLICY {
            FullPolicy::Restart => {
                *restart_count += 1;
                self.restart_with_noise(*restart_count as u64);
            }
            FullPolicy::Evict(frac) => {
                let target = ((NODE_BUDGET as f32) * frac) as usize;
                self.evict_leaves(target.max(1));
            }
        }
    }

    /// Run iterations until deadline or solution found. Returns (solved, iterations).
    fn run_until_deadline(
        &mut self,
        deadline: Instant,
        progress_every: u64,
        mut on_progress: impl FnMut(u64, &Self),
    ) -> (bool, u64) {
        let mut iterations = 0u64;
        let mut restart_count = 0u32;
        loop {
            if self.is_arena_full() {
                self.handle_arena_full(&mut restart_count);
            }

            self.run_iteration();
            iterations += 1;
            if self.witness_placements.is_some() {
                return (true, iterations);
            }
            if progress_every > 0 && iterations % progress_every == 0 {
                on_progress(iterations, self);
            }
            if iterations & 0x1FFF == 0x1FFF && Instant::now() >= deadline {
                return (false, iterations);
            }
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
                    break self.evaluate_leaf(node_id);
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
        if board.count() == 0 {
            1.0
        } else {
            terminal_failure_value(board)
        }
    }

    fn evaluate_leaf(&self, node_id: usize) -> f32 {
        let state = self.nodes[node_id].state;
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

            // Apply filter — prune undesirable boards early
            if let Some(f) = self.filter_fn {
                if !f(next, state.step + 1) {
                    continue;
                }
            }

            let mut prior_score = scorer_prior(next, result.lines_cleared);
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
            let child = match self.alloc_node(SearchNode::new(child_state)) {
                Some(slot) => slot,
                None => break, // arena full, no free slots
            };

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

// --- Scoring functions ---

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

// --- Filters ---

/// Default filter for empty-to-empty 5-bag search.
/// Prunes boards that can't possibly reach 0 cells with the remaining pieces.
fn empty_cycle_filter(board: TetrisBoard, step: u8) -> bool {
    let cells = board.count();
    let remaining = TOTAL_PIECES as u32 - step as u32;
    let cells_to_add = remaining * 4;

    // Total cells that need to be cleared via line clears.
    // Each line clear removes 10 cells. Need total / 10 lines.
    // Each piece can clear at most 4 lines.
    let total_cells_to_clear = cells + cells_to_add;
    let lines_needed = (total_cells_to_clear + 9) / 10;
    let max_lines_possible = remaining * 4;
    if lines_needed > max_lines_possible {
        return false;
    }

    // Too many holes makes clearing very unlikely.
    // Each hole typically needs at least 1 piece to uncover.
    let holes = board.total_holes();
    if holes > remaining / 2 {
        return false;
    }

    true
}

// --- CLI ---

#[derive(Parser)]
#[command(name = "tetris_five_bag_chain")]
#[command(about = "5-bag chain solver with PUCT/MCTS")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the chain solver (default). Solve random 5-bags until global timeout.
    Run {
        /// Per-sequence timeout in seconds.
        #[arg(long, default_value_t = 60)]
        timeout: u64,
        /// Global timeout in seconds.
        #[arg(long, default_value_t = 360)]
        global_timeout: u64,
        /// RNG seed for bag generation.
        #[arg(long, default_value_t = 0xCAFE_BABE)]
        seed: u64,
    },
    /// Spot-check a single 5-bag sequence. Runs until solved (no time limit).
    Spotcheck {
        /// Five comma-separated bag permutation indices (0-5039).
        #[arg(value_name = "BAGS")]
        bags: String,
        /// Log progress every N iterations (0 to disable).
        #[arg(long, default_value_t = 500_000)]
        log_every: u64,
    },
}

fn parse_bag_indices(s: &str) -> Result<[usize; BAG_COUNT], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != BAG_COUNT {
        return Err(format!(
            "expected {BAG_COUNT} comma-separated bag indices, got {}",
            parts.len()
        ));
    }
    let mut indices = [0usize; BAG_COUNT];
    for (i, part) in parts.iter().enumerate() {
        let idx: usize = part
            .trim()
            .parse()
            .map_err(|e| format!("bad index `{}`: {e}", part.trim()))?;
        if idx >= BAG_PERMUTATION_COUNT {
            return Err(format!(
                "bag index {idx} out of range 0..{BAG_PERMUTATION_COUNT}"
            ));
        }
        indices[i] = idx;
    }
    Ok(indices)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None | Some(Command::Run { .. }) => {
            let (timeout, global_timeout, seed) = match cli.command {
                Some(Command::Run {
                    timeout,
                    global_timeout,
                    seed,
                }) => (timeout, global_timeout, seed),
                _ => (60, 360, 0xCAFE_BABE),
            };
            cmd_run(timeout, global_timeout, seed);
        }
        Some(Command::Spotcheck { bags, log_every }) => {
            let bag_indices = match parse_bag_indices(&bags) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            };
            cmd_spotcheck(bag_indices, log_every);
        }
    }
}

fn cmd_run(per_seq_timeout: u64, global_timeout: u64, seed: u64) {
    let overall_start = Instant::now();
    let global_deadline = overall_start + std::time::Duration::from_secs(global_timeout);
    let mut rng = Rng::new(seed);
    let mut solved = 0usize;
    let mut attempted = 0usize;

    let mut bag_indices = [0usize; BAG_COUNT];
    for b in &mut bag_indices {
        *b = rng.next_bag_index();
    }
    let mut search = PuctSearch::new(build_sequence(&bag_indices));
    search.set_filter(empty_cycle_filter);

    loop {
        if Instant::now() >= global_deadline {
            break;
        }
        attempted += 1;

        for b in &mut bag_indices {
            *b = rng.next_bag_index();
        }
        search.reset(build_sequence(&bag_indices));

        let seq_start = Instant::now();
        let seq_deadline =
            (seq_start + std::time::Duration::from_secs(per_seq_timeout)).min(global_deadline);

        let (found, iterations) = search.run_until_deadline(seq_deadline, 0, |_, _| {});
        let seq_elapsed = seq_start.elapsed().as_secs_f64();

        let status = if found {
            match search.witness_placements {
                Some(ref p) if replay_witness(&search.pieces, p) => {
                    solved += 1;
                    " true"
                }
                _ => "INVAL",
            }
        } else {
            "false"
        };

        eprintln!(
            "chain={:<4} solved={:<5} time={:>7.3}s iters={:>12} bags=[{:>4},{:>4},{:>4},{:>4},{:>4}] nodes={:>12} edges={:>12}",
            attempted,
            status,
            seq_elapsed,
            iterations,
            bag_indices[0],
            bag_indices[1],
            bag_indices[2],
            bag_indices[3],
            bag_indices[4],
            search.nodes.len(),
            search.edges.len(),
        );
    }

    let total_elapsed = overall_start.elapsed().as_secs_f64();
    eprintln!("=== chain complete ===");
    eprintln!("solved={solved}/{attempted} wall={total_elapsed:.1}s");
    eprintln!(
        "solve_rate={:.1}%",
        100.0 * solved as f64 / attempted.max(1) as f64
    );
    println!("{solved}");
}

fn print_raw(s: &str) {
    use std::io::Write;
    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(s.as_bytes());
    let _ = stderr.flush();
}

fn interactive_viewer(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    placements: &[TetrisPiecePlacement; TOTAL_PIECES],
) {
    // Pre-compute all board states
    let mut boards = [TetrisBoard::new(); TOTAL_PIECES + 1];
    let mut lines_at_step = [0u32; TOTAL_PIECES];
    for (step, placement) in placements.iter().copied().enumerate() {
        let mut board = boards[step];
        let result = board.apply_piece_placement(placement);
        boards[step + 1] = board;
        lines_at_step[step] = result.lines_cleared;
    }

    let mut cursor: usize = 0; // 0 = empty board, 1..=35 = after step 0..34

    terminal::enable_raw_mode().ok();
    eprintln!("\x1b[2J\x1b[H"); // clear screen

    loop {
        // Clear and redraw
        eprint!("\x1b[H\x1b[2J");

        let board = boards[cursor];

        if cursor == 0 {
            print_raw("=== Step: START (empty board) ===\r\n");
        } else {
            let step = cursor - 1;
            print_raw(&format!(
                "=== Step {step:>2}/{TOTAL_PIECES}: piece={} placement={:>3} lines={} ===\r\n",
                pieces[step],
                placements[step].index(),
                lines_at_step[step],
            ));
        }

        print_raw(&format!(
            "cells={:>2} height={:>2}\r\n",
            board.count(),
            board.height()
        ));

        // Render board with \r\n line endings for raw mode
        let board_str = format!("{board}");
        for line in board_str.lines() {
            print_raw(line);
            print_raw("\r\n");
        }

        if cursor == TOTAL_PIECES {
            print_raw("\r\n  ** EMPTY BOARD — witness complete! **\r\n");
        }

        print_raw(&format!(
            "\r\n[→/Enter] next  [←/Backspace] prev  [q/Ctrl-C] quit  [{cursor}/{}]\r\n",
            TOTAL_PIECES
        ));

        // Wait for key
        match event::read() {
            Ok(Event::Key(KeyEvent {
                code, modifiers, ..
            })) => match code {
                KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                    if cursor < TOTAL_PIECES {
                        cursor += 1;
                    }
                }
                KeyCode::Left | KeyCode::Backspace => {
                    if cursor > 0 {
                        cursor -= 1;
                    }
                }
                KeyCode::Home => cursor = 0,
                KeyCode::End => cursor = TOTAL_PIECES,
                KeyCode::Char('q') => break,
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => break,
                _ => {}
            },
            Ok(_) => {}
            Err(_) => break,
        }
    }

    terminal::disable_raw_mode().ok();
    eprintln!();
}

fn cmd_spotcheck(bag_indices: [usize; BAG_COUNT], log_every: u64) {
    let pieces = build_sequence(&bag_indices);
    let mut search = PuctSearch::new(pieces);
    search.set_filter(empty_cycle_filter);

    eprintln!(
        "spotcheck bags=[{},{},{},{},{}] — running until solved (no time limit)...",
        bag_indices[0], bag_indices[1], bag_indices[2], bag_indices[3], bag_indices[4],
    );

    let start = Instant::now();
    // Effectively no deadline — 1 year
    let deadline = Instant::now() + std::time::Duration::from_secs(365 * 24 * 3600);
    let (found, iterations) = search.run_until_deadline(deadline, log_every, |iters, s| {
        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "  progress iters={:>12} time={:>7.1}s nodes={:>12} edges={:>12} iters/s={:.0}",
            iters,
            elapsed,
            s.nodes.len(),
            s.edges.len(),
            iters as f64 / elapsed.max(1e-9),
        );
    });
    let elapsed = start.elapsed().as_secs_f64();

    if found {
        if let Some(ref p) = search.witness_placements {
            if replay_witness(&search.pieces, p) {
                eprintln!(
                    "SOLVED time={elapsed:.3}s iters={iterations} nodes={} edges={}",
                    search.nodes.len(),
                    search.edges.len()
                );
                eprintln!();
                // Print summary, then launch interactive viewer
                let mut board = TetrisBoard::new();
                for (step, placement) in p.iter().copied().enumerate() {
                    let result = board.apply_piece_placement(placement);
                    eprintln!(
                        "  step={step:>2} piece={} placement={:>3} lines={} cells={:>2} height={:>2}",
                        pieces[step],
                        placement.index(),
                        result.lines_cleared,
                        board.count(),
                        board.height(),
                    );
                }
                eprintln!();
                interactive_viewer(&pieces, p);
                return;
            }
        }
        eprintln!("INVALID WITNESS");
    } else {
        eprintln!("NOT SOLVED (should not happen with no deadline) iters={iterations}");
    }
}
