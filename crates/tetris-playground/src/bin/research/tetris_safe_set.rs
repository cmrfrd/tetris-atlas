//! Minimal safe-set certifier built around an allocation-free best-first backtracker.
//!
//! Goal:
//! - start from one board, typically the empty board
//! - for each forced first bag, certify it under a bounded recursive recovery proof
//! - the base case is a direct 7-piece script back into the safe set
//! - deeper recovery levels choose one fixed 7-piece script and then require every
//!   next bag to certify recursively at one smaller remaining depth
//!
//! The hot search core is intentionally allocation-free:
//! - fixed-capacity node arena
//! - fixed-capacity frontier heap
//! - fixed-capacity first-bag candidate buffer
//! - permutation iterators instead of materialized `Vec<[TetrisPiece; 7]>`
//!
//! The outer adapter layer still allows normal allocations for CLI parsing, logging,
//! and optional caches.

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::HashSet;
use std::sync::{
    Arc, Condvar, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser;
use crossbeam_queue::SegQueue;
use dashmap::{DashMap, mapref::entry::Entry};
use itertools::Itertools;
use tetris_game::{
    FixedBinMinHeap, HeaplessVec, IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement,
    repeat_idx_unroll,
};

type ForcedBag = [TetrisPiece; PIECES_PER_BAG];
type PlacementScript = [TetrisPiecePlacement; PIECES_PER_BAG];
const PIECES_PER_BAG: usize = 7;
const RECOVERY_BAG_DEPTH: usize = 2;
const RECOVERY_CANDIDATE_LIMITS: [usize; RECOVERY_BAG_DEPTH - 1] = [512];
const CANDIDATE_BUFFER_CAPACITY: usize = RECOVERY_CANDIDATE_LIMITS[0];
type CandidateBuffer = HeaplessVec<SearchWitness, CANDIDATE_BUFFER_CAPACITY>;

const PLANNER_WIDTH: usize = 8;
const SAFE_HEIGHT_CAP: u8 = 3;
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(2);

const BAG_PERMUTATION_COUNT: usize = 5_040;
static ALL_FORCED_BAG_PERMUTATIONS: [ForcedBag; BAG_PERMUTATION_COUNT] =
    generate_forced_bag_permutations();
const DEPTH_COUNTER_LEN: usize = RECOVERY_BAG_DEPTH + 1;
type DepthAtomicCounters = [AtomicUsize; DEPTH_COUNTER_LEN];
type DepthCounterSnapshot = [usize; DEPTH_COUNTER_LEN];

/// Allocates one zeroed atomic counter per possible remaining-depth slot.
fn new_depth_counters() -> DepthAtomicCounters {
    std::array::from_fn(|_| AtomicUsize::new(0))
}

/// Loads depth-indexed counters for progress reporting.
fn snapshot_depth_counters(counters: &DepthAtomicCounters) -> DepthCounterSnapshot {
    std::array::from_fn(|idx| counters[idx].load(AtomicOrdering::Relaxed))
}

/// Maps the recursive remaining depth into the matching metrics array slot.
fn depth_slot(remaining_depth: usize) -> usize {
    debug_assert!((1..=RECOVERY_BAG_DEPTH).contains(&remaining_depth));
    remaining_depth.min(RECOVERY_BAG_DEPTH)
}

/// Advances a lexicographic permutation in-place.
const fn advance_permutation(indices: &mut [u8; PIECES_PER_BAG]) -> bool {
    if PIECES_PER_BAG < 2 {
        return false;
    }

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

/// Generates all 7-piece bag permutations at compile time.
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

/// Returns true if the candidate board is below the safe height cap.
#[inline(always)]
const fn terminal_fn(candidate: &TetrisBoard) -> bool {
    candidate.height() <= SAFE_HEIGHT_CAP as u32 && candidate.total_holes() <= 2
}

/// Scores a board so the frontier prefers cleaner, lower, line-clearing continuations.
#[inline(always)]
const fn default_score_fn(state: &PlannerScoreState) -> f32 {
    let lines = state.recent_lines_cleared as f32;
    let holes = state.board.total_holes() as f32;
    let heights = state.board.heights();

    let mut aggregate_height = 0.0;
    repeat_idx_unroll!(TetrisBoard::WIDTH, I, {
        aggregate_height += heights[I] as f32;
    });

    let mut bumpiness = 0.0;
    repeat_idx_unroll!(TetrisBoard::WIDTH - 1, I, {
        bumpiness += (heights[I] as f32 - heights[I + 1] as f32).abs();
    });

    0.760666 * lines + (-0.510066) * aggregate_height + (-0.35663) * holes + (-0.184483) * bumpiness
}

/// Computes the full width-limited best-first tree capacity for a fixed depth.
const fn best_first_capacity(width: usize, depth: usize) -> usize {
    let mut total = 1usize;
    let mut layer = 1usize;
    let mut level = 0usize;
    while level < depth {
        layer *= width;
        total += layer;
        level += 1;
    }
    total
}

const BEST_FIRST_NODE_CAPACITY: usize = best_first_capacity(PLANNER_WIDTH, PIECES_PER_BAG);

/// Maps a remaining recursive depth to its configured fixed-script candidate budget.
const fn recovery_candidate_limit_for_remaining_depth(remaining_depth: usize) -> usize {
    RECOVERY_CANDIDATE_LIMITS[RECOVERY_BAG_DEPTH - remaining_depth]
}

#[derive(Debug, Parser)]
#[command(name = "tetris_safe_set")]
#[command(
    about = "Minimal best-first safe-set certifier with fixed-first-bag universal second-bag recovery"
)]
struct Cli {
    #[arg(long, default_value_t = 5)]
    print_witness_limit: usize,
    #[arg(long, default_value_t = 1)]
    workers: usize,
}

#[derive(Debug, Clone, Copy)]
struct PlannerScoreState {
    board: TetrisBoard,
    recent_lines_cleared: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CandidateNode {
    board: TetrisBoard,
    placement: u8,
    score: f32,
}

impl Default for CandidateNode {
    /// Creates an empty sentinel candidate used in fixed-size top-k buffers.
    fn default() -> Self {
        Self {
            board: TetrisBoard::new(),
            placement: TetrisPiecePlacement::default().index(),
            score: f32::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SearchNode {
    board: TetrisBoard,
    depth: u8,
    parent_slot: u32,
    placement: u8,
}

impl Default for SearchNode {
    /// Creates an empty sentinel node used to initialize the planner arena.
    fn default() -> Self {
        Self {
            board: TetrisBoard::new(),
            depth: 0,
            parent_slot: u32::MAX,
            placement: TetrisPiecePlacement::default().index(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SearchWitness {
    board: TetrisBoard,
    placements: PlacementScript,
    backtracks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchOutcome {
    Success(SearchWitness),
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    /// Delegates to the total ordering used by the heap entry wrapper.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    /// Provides a total order for `f32` priorities in fixed-capacity heaps.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrontierEntry {
    node_idx: u32,
    priority: OrderedF32,
    depth: u8,
    board_limbs: [u32; TetrisBoard::WIDTH],
}

impl FrontierEntry {
    /// Packs all ordering data needed to rank a planner node in the frontier heap.
    fn new(node_idx: u32, priority: f32, depth: u8, board: TetrisBoard) -> Self {
        Self {
            node_idx,
            priority: OrderedF32(priority),
            depth,
            board_limbs: board.as_limbs(),
        }
    }
}

impl PartialOrd for FrontierEntry {
    /// Defers to the full `Ord` implementation used by the frontier heap.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierEntry {
    /// Orders frontier entries by score, then depth, then board tie-breakers.
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| other.board_limbs.cmp(&self.board_limbs))
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

struct BacktrackingPlanner {
    nodes: HeaplessVec<SearchNode, BEST_FIRST_NODE_CAPACITY>,
    frontier: FixedBinMinHeap<Reverse<FrontierEntry>, BEST_FIRST_NODE_CAPACITY>,
    backtracks: usize,
}

impl BacktrackingPlanner {
    /// Builds a reusable fixed-capacity planner with no runtime allocation in the search loop.
    const fn new() -> Self {
        Self {
            nodes: HeaplessVec::new(),
            frontier: FixedBinMinHeap::new(),
            backtracks: 0,
        }
    }

    /// Resets logical lengths so the next query reuses the same fixed-capacity buffers.
    fn reset(&mut self) {
        self.nodes.clear();
        self.frontier.clear();
        self.backtracks = 0;
    }

    /// Finds one best-first 7-piece script, optionally requiring the final board to satisfy a terminal predicate.
    fn search<TerminalFnT>(
        &mut self,
        start_board: TetrisBoard,
        forced_sequence: &ForcedBag,
        terminal_fn: Option<&TerminalFnT>,
    ) -> SearchOutcome
    where
        TerminalFnT: Fn(TetrisBoard) -> bool,
    {
        // Reset the planner to its initial state.
        self.reset();
        let root = SearchNode {
            board: start_board,
            depth: 0,
            parent_slot: u32::MAX,
            placement: TetrisPiecePlacement::default().index(),
        };
        let pushed = self.nodes.try_push(root);
        debug_assert!(pushed, "planner root must fit in node arena");
        self.frontier.push(Reverse(FrontierEntry::new(
            0,
            f32::INFINITY,
            0,
            start_board,
        )));

        // Start search process by popping from the frontier.
        while let Some(Reverse(entry)) = self.frontier.pop_min() {
            let node = self.nodes[entry.node_idx as usize];
            if node.depth as usize == PIECES_PER_BAG {
                if terminal_fn.is_none_or(|terminal| terminal(node.board)) {
                    self.backtracks = self.backtracks.saturating_add(1);
                    return SearchOutcome::Success(self.build_witness(entry.node_idx));
                }
                continue;
            }

            let piece = forced_sequence[node.depth as usize];
            let mut top_candidates = [CandidateNode::default(); PLANNER_WIDTH];
            let mut count = 0usize;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child_board = node.board;
                let outcome = child_board.apply_piece_placement(placement);
                if outcome.is_lost == IsLost::LOST {
                    continue;
                }
                let score = default_score_fn(&PlannerScoreState {
                    board: child_board,
                    recent_lines_cleared: outcome.lines_cleared,
                });
                insert_top_candidate(
                    &mut top_candidates,
                    &mut count,
                    CandidateNode {
                        board: child_board,
                        placement: placement.index(),
                        score,
                    },
                );
            }

            for child in top_candidates.into_iter().take(count.min(PLANNER_WIDTH)) {
                let child_depth = node.depth + 1;
                let child_idx = self.nodes.len();
                if child_idx >= BEST_FIRST_NODE_CAPACITY {
                    return SearchOutcome::Exhausted;
                }
                if !self.nodes.try_push(SearchNode {
                    board: child.board,
                    depth: child_depth,
                    parent_slot: entry.node_idx,
                    placement: child.placement,
                }) {
                    return SearchOutcome::Exhausted;
                }
                self.frontier.push(Reverse(FrontierEntry::new(
                    child_idx as u32,
                    child.score,
                    child_depth,
                    child.board,
                )));
            }
        }

        SearchOutcome::Exhausted
    }

    /// Collects the top complete 7-piece scripts in best-first order into a fixed-capacity buffer.
    fn collect_witnesses(
        &mut self,
        start_board: TetrisBoard,
        forced_sequence: &ForcedBag,
        limit: usize,
    ) -> CandidateBuffer {
        self.reset();
        let root = SearchNode {
            board: start_board,
            depth: 0,
            parent_slot: u32::MAX,
            placement: TetrisPiecePlacement::default().index(),
        };
        let pushed = self.nodes.try_push(root);
        debug_assert!(pushed, "planner root must fit in node arena");
        self.frontier.push(Reverse(FrontierEntry::new(
            0,
            f32::INFINITY,
            0,
            start_board,
        )));

        let mut witnesses = CandidateBuffer::new();
        while let Some(Reverse(entry)) = self.frontier.pop_min() {
            let node = self.nodes[entry.node_idx as usize];
            if node.depth as usize == PIECES_PER_BAG {
                let pushed = witnesses.try_push(self.build_witness(entry.node_idx));
                debug_assert!(pushed, "candidate buffer capacity must cover its own limit");
                if witnesses.len() >= limit {
                    break;
                }
                continue;
            }

            let piece = forced_sequence[node.depth as usize];
            let mut top_candidates = [CandidateNode::default(); PLANNER_WIDTH];
            let mut count = 0usize;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child_board = node.board;
                let outcome = child_board.apply_piece_placement(placement);
                if outcome.is_lost == IsLost::LOST {
                    continue;
                }
                let score = default_score_fn(&PlannerScoreState {
                    board: child_board,
                    recent_lines_cleared: outcome.lines_cleared,
                });
                insert_top_candidate(
                    &mut top_candidates,
                    &mut count,
                    CandidateNode {
                        board: child_board,
                        placement: placement.index(),
                        score,
                    },
                );
            }

            for child in top_candidates.into_iter().take(count.min(PLANNER_WIDTH)) {
                let child_depth = node.depth + 1;
                let child_idx = self.nodes.len();
                if child_idx >= BEST_FIRST_NODE_CAPACITY {
                    return witnesses;
                }
                if !self.nodes.try_push(SearchNode {
                    board: child.board,
                    depth: child_depth,
                    parent_slot: entry.node_idx,
                    placement: child.placement,
                }) {
                    return witnesses;
                }
                self.frontier.push(Reverse(FrontierEntry::new(
                    child_idx as u32,
                    child.score,
                    child_depth,
                    child.board,
                )));
            }
        }

        witnesses
    }

    /// Reconstructs the placement script by following parent pointers back from a frontier node.
    fn build_witness(&self, final_slot: u32) -> SearchWitness {
        let mut placements = [TetrisPiecePlacement::default(); PIECES_PER_BAG];
        let mut cursor = final_slot;
        let mut remaining = PIECES_PER_BAG;
        while cursor != 0 && cursor != u32::MAX {
            let node = self.nodes[cursor as usize];
            remaining -= 1;
            placements[remaining] = TetrisPiecePlacement::from_index(node.placement);
            cursor = node.parent_slot;
        }
        debug_assert_eq!(remaining, 0);
        SearchWitness {
            board: self.nodes[final_slot as usize].board,
            placements,
            backtracks: self.backtracks,
        }
    }
}

thread_local! {
    static THREAD_LOCAL_PLANNER: RefCell<BacktrackingPlanner> =
        const { RefCell::new(BacktrackingPlanner::new()) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DirectSearchCacheKey {
    board: TetrisBoard,
    bag: ForcedBag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RecursiveBoardCacheKey {
    board: TetrisBoard,
    remaining_depth: u8,
}

#[derive(Debug, Default)]
struct WorkSignal {
    ready: Mutex<bool>,
    condvar: Condvar,
}

impl WorkSignal {
    /// Creates an unresolved signal used to coordinate one in-progress computation.
    fn new() -> Self {
        Self::default()
    }

    /// Waits until the owner thread publishes a terminal cached value.
    fn wait(&self) {
        let mut ready = self
            .ready
            .lock()
            .expect("work signal mutex poisoned while waiting");
        while !*ready {
            ready = self
                .condvar
                .wait(ready)
                .expect("work signal mutex poisoned while blocked");
        }
    }

    /// Wakes all waiters after the cache entry is resolved.
    fn notify_ready(&self) {
        let mut ready = self
            .ready
            .lock()
            .expect("work signal mutex poisoned while notifying");
        *ready = true;
        self.condvar.notify_all();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoardState {
    Pending,
    InFlight,
    Solved,
    Failed,
    Abandoned,
}

#[derive(Debug, Clone)]
enum DirectSearchCacheValue {
    InProgress(Arc<WorkSignal>),
    Success(SearchWitness),
    Failure,
}

#[derive(Debug)]
struct ProgressCounters {
    direct_cache_hits: AtomicUsize,
    direct_cache_misses: AtomicUsize,
    direct_cache_waits: AtomicUsize,
    recursive_cache_hits: AtomicUsize,
    recursive_cache_misses: AtomicUsize,
    recursive_cache_waits: AtomicUsize,
    forced_bag_direct_by_depth: DepthAtomicCounters,
    forced_bag_recovered_by_depth: DepthAtomicCounters,
    forced_bag_failed_by_depth: DepthAtomicCounters,
    candidate_scripts_examined_by_depth: DepthAtomicCounters,
    recursive_cache_hits_by_depth: DepthAtomicCounters,
    recursive_cache_misses_by_depth: DepthAtomicCounters,
    recursive_cache_waits_by_depth: DepthAtomicCounters,
    recursive_cache_complete_hits_by_depth: DepthAtomicCounters,
    recursive_cache_impossible_hits_by_depth: DepthAtomicCounters,
    recursive_board_complete_by_depth: DepthAtomicCounters,
    recursive_board_impossible_by_depth: DepthAtomicCounters,
    recovered_safe_boards_discovered: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
struct ProgressSnapshot {
    direct_cache_hits: usize,
    direct_cache_misses: usize,
    direct_cache_waits: usize,
    recursive_cache_hits: usize,
    recursive_cache_misses: usize,
    recursive_cache_waits: usize,
    forced_bag_direct_by_depth: DepthCounterSnapshot,
    forced_bag_recovered_by_depth: DepthCounterSnapshot,
    forced_bag_failed_by_depth: DepthCounterSnapshot,
    candidate_scripts_examined_by_depth: DepthCounterSnapshot,
    recursive_cache_hits_by_depth: DepthCounterSnapshot,
    recursive_cache_misses_by_depth: DepthCounterSnapshot,
    recursive_cache_waits_by_depth: DepthCounterSnapshot,
    recursive_cache_complete_hits_by_depth: DepthCounterSnapshot,
    recursive_cache_impossible_hits_by_depth: DepthCounterSnapshot,
    recursive_board_complete_by_depth: DepthCounterSnapshot,
    recursive_board_impossible_by_depth: DepthCounterSnapshot,
    recovered_safe_boards_discovered: usize,
}

impl ProgressCounters {
    /// Initializes all runtime cache counters to zero.
    fn new() -> Self {
        Self {
            direct_cache_hits: AtomicUsize::new(0),
            direct_cache_misses: AtomicUsize::new(0),
            direct_cache_waits: AtomicUsize::new(0),
            recursive_cache_hits: AtomicUsize::new(0),
            recursive_cache_misses: AtomicUsize::new(0),
            recursive_cache_waits: AtomicUsize::new(0),
            forced_bag_direct_by_depth: new_depth_counters(),
            forced_bag_recovered_by_depth: new_depth_counters(),
            forced_bag_failed_by_depth: new_depth_counters(),
            candidate_scripts_examined_by_depth: new_depth_counters(),
            recursive_cache_hits_by_depth: new_depth_counters(),
            recursive_cache_misses_by_depth: new_depth_counters(),
            recursive_cache_waits_by_depth: new_depth_counters(),
            recursive_cache_complete_hits_by_depth: new_depth_counters(),
            recursive_cache_impossible_hits_by_depth: new_depth_counters(),
            recursive_board_complete_by_depth: new_depth_counters(),
            recursive_board_impossible_by_depth: new_depth_counters(),
            recovered_safe_boards_discovered: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug, Clone)]
enum RecursiveBoardStatus {
    InProgress(Arc<WorkSignal>),
    Complete { safe_boards: Arc<[TetrisBoard]> },
    Impossible { failure: FailureExample },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RecursiveBoardCertification {
    Complete { safe_boards: Arc<[TetrisBoard]> },
    Impossible { failure: FailureExample },
}

#[derive(Debug)]
struct SharedSearchContext {
    direct_cache: DashMap<DirectSearchCacheKey, DirectSearchCacheValue>,
    recursive_board_cache: DashMap<RecursiveBoardCacheKey, RecursiveBoardStatus>,
    progress: ProgressCounters,
}

impl SharedSearchContext {
    /// Creates the outer cache layer that sits on top of the allocation-free search core.
    fn new() -> Self {
        Self {
            direct_cache: DashMap::new(),
            recursive_board_cache: DashMap::new(),
            progress: ProgressCounters::new(),
        }
    }

    /// Increments direct-cache hit counters.
    fn record_direct_cache_hit(&self) {
        self.progress
            .direct_cache_hits
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments direct-cache miss counters.
    fn record_direct_cache_miss(&self) {
        self.progress
            .direct_cache_misses
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments direct-cache wait counters when a worker reuses an in-progress search.
    fn record_direct_cache_wait(&self) {
        self.progress
            .direct_cache_waits
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments recursive-board cache counters for a complete hit.
    fn record_recursive_cache_complete_hit(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress
            .recursive_cache_hits
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_hits_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_complete_hits_by_depth[slot]
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments recursive-board cache counters for an impossible hit.
    fn record_recursive_cache_impossible_hit(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress
            .recursive_cache_hits
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_hits_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_impossible_hits_by_depth[slot]
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments recursive-board cache miss counters.
    fn record_recursive_cache_miss(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress
            .recursive_cache_misses
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_misses_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Increments recursive-board cache wait counters when a worker reuses an in-progress sweep.
    fn record_recursive_cache_wait(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress
            .recursive_cache_waits
            .fetch_add(1, AtomicOrdering::Relaxed);
        self.progress.recursive_cache_waits_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Records one forced bag that certified directly at the given remaining depth.
    fn record_forced_bag_direct(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.forced_bag_direct_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Records one forced bag that certified through recursive recovery.
    fn record_forced_bag_recovered(&self, remaining_depth: usize, discovered_safe_boards: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.forced_bag_recovered_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
        self.progress
            .recovered_safe_boards_discovered
            .fetch_add(discovered_safe_boards, AtomicOrdering::Relaxed);
    }

    /// Records one forced bag that failed after exhausting its direct and recovery searches.
    fn record_forced_bag_failed(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.forced_bag_failed_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Records how many fixed-script recovery candidates were considered at a depth.
    fn record_candidate_scripts_examined(&self, remaining_depth: usize, candidate_count: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.candidate_scripts_examined_by_depth[slot]
            .fetch_add(candidate_count, AtomicOrdering::Relaxed);
    }

    /// Records a full recursive board sweep that proved all forced bags complete.
    fn record_recursive_board_complete(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.recursive_board_complete_by_depth[slot].fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Records a full recursive board sweep that found an impossible forced bag.
    fn record_recursive_board_impossible(&self, remaining_depth: usize) {
        let slot = depth_slot(remaining_depth);
        self.progress.recursive_board_impossible_by_depth[slot]
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Returns a snapshot of cache hit/miss counters.
    fn progress_snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            direct_cache_hits: self
                .progress
                .direct_cache_hits
                .load(AtomicOrdering::Relaxed),
            direct_cache_misses: self
                .progress
                .direct_cache_misses
                .load(AtomicOrdering::Relaxed),
            direct_cache_waits: self
                .progress
                .direct_cache_waits
                .load(AtomicOrdering::Relaxed),
            recursive_cache_hits: self
                .progress
                .recursive_cache_hits
                .load(AtomicOrdering::Relaxed),
            recursive_cache_misses: self
                .progress
                .recursive_cache_misses
                .load(AtomicOrdering::Relaxed),
            recursive_cache_waits: self
                .progress
                .recursive_cache_waits
                .load(AtomicOrdering::Relaxed),
            forced_bag_direct_by_depth: snapshot_depth_counters(
                &self.progress.forced_bag_direct_by_depth,
            ),
            forced_bag_recovered_by_depth: snapshot_depth_counters(
                &self.progress.forced_bag_recovered_by_depth,
            ),
            forced_bag_failed_by_depth: snapshot_depth_counters(
                &self.progress.forced_bag_failed_by_depth,
            ),
            candidate_scripts_examined_by_depth: snapshot_depth_counters(
                &self.progress.candidate_scripts_examined_by_depth,
            ),
            recursive_cache_hits_by_depth: snapshot_depth_counters(
                &self.progress.recursive_cache_hits_by_depth,
            ),
            recursive_cache_misses_by_depth: snapshot_depth_counters(
                &self.progress.recursive_cache_misses_by_depth,
            ),
            recursive_cache_waits_by_depth: snapshot_depth_counters(
                &self.progress.recursive_cache_waits_by_depth,
            ),
            recursive_cache_complete_hits_by_depth: snapshot_depth_counters(
                &self.progress.recursive_cache_complete_hits_by_depth,
            ),
            recursive_cache_impossible_hits_by_depth: snapshot_depth_counters(
                &self.progress.recursive_cache_impossible_hits_by_depth,
            ),
            recursive_board_complete_by_depth: snapshot_depth_counters(
                &self.progress.recursive_board_complete_by_depth,
            ),
            recursive_board_impossible_by_depth: snapshot_depth_counters(
                &self.progress.recursive_board_impossible_by_depth,
            ),
            recovered_safe_boards_discovered: self
                .progress
                .recovered_safe_boards_discovered
                .load(AtomicOrdering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CertificationKind {
    Direct,
    Recovered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CertifiedFirstBag {
    first_bag: ForcedBag,
    placements: PlacementScript,
    final_board: TetrisBoard,
    kind: CertificationKind,
    backtracks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RecursiveForcedBagProof {
    Direct {
        safe_boards: Arc<[TetrisBoard]>,
    },
    Recovered {
        intermediate_board: TetrisBoard,
        child_remaining_depth: u8,
        safe_boards: Arc<[TetrisBoard]>,
    },
}

impl RecursiveForcedBagProof {
    /// Returns the safe-set boards proved reachable by this forced-bag certification.
    fn discovered_safe_boards(&self) -> Arc<[TetrisBoard]> {
        match self {
            Self::Direct { safe_boards } | Self::Recovered { safe_boards, .. } => {
                Arc::clone(safe_boards)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FailureExample {
    board: TetrisBoard,
    forced_bag: ForcedBag,
    chosen_placements: Option<PlacementScript>,
    intermediate_board: Option<TetrisBoard>,
    child_failure: Option<Box<FailureExample>>,
}

impl FailureExample {
    /// Creates a leaf failure after all direct and recovery options were exhausted.
    fn leaf(board: TetrisBoard, forced_bag: ForcedBag) -> Self {
        Self {
            board,
            forced_bag,
            chosen_placements: None,
            intermediate_board: None,
            child_failure: None,
        }
    }

    /// Records that a chosen fixed-bag script failed because a deeper recursive certification failed.
    fn with_child(
        board: TetrisBoard,
        forced_bag: ForcedBag,
        chosen_placements: PlacementScript,
        intermediate_board: TetrisBoard,
        child_failure: FailureExample,
    ) -> Self {
        Self {
            board,
            forced_bag,
            chosen_placements: Some(chosen_placements),
            intermediate_board: Some(intermediate_board),
            child_failure: Some(Box::new(child_failure)),
        }
    }

    /// Returns the forced bag at the current recursion level.
    fn top_level_bag(&self) -> ForcedBag {
        self.forced_bag
    }

    /// Returns the chosen placements from the current recursion level, if any.
    fn top_level_placements(&self) -> Option<PlacementScript> {
        self.chosen_placements
    }

    /// Returns the intermediate board chosen at the current recursion level, if any.
    fn top_level_intermediate_board(&self) -> Option<TetrisBoard> {
        self.intermediate_board
    }

    /// Returns the deepest failing forced bag in the recursive failure chain.
    fn leaf_bag(&self) -> ForcedBag {
        self.child_failure
            .as_deref()
            .map_or(self.forced_bag, FailureExample::leaf_bag)
    }

    /// Returns the next failing recovery bag if the failure occurred below the current level.
    fn failing_recovery_bag(&self) -> Option<ForcedBag> {
        self.child_failure.as_deref().map(FailureExample::leaf_bag)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FirstBagCertification {
    Certified {
        entry: CertifiedFirstBag,
        proof: RecursiveForcedBagProof,
    },
    Failed(FailureExample),
}

/// Aggregates one full first-bag certification sweep for a single board.
struct BoardProcessResult {
    board: TetrisBoard,
    failure: Option<FailureExample>,
    cancelled: bool,
    processed_first_bags: usize,
    direct_first_bags: usize,
    recovered_first_bags: usize,
    failed_first_bags: usize,
    sample_entries: Vec<CertifiedFirstBag>,
}

#[derive(Debug, Clone, Copy)]
struct BoardProcessProgressUpdate {
    board: TetrisBoard,
    current_first_bag_idx: usize,
    completed_first_bags: usize,
    direct_first_bags: usize,
    recovered_first_bags: usize,
    failed_first_bags: usize,
    unique_safe_boards: usize,
}

#[derive(Debug, Clone, Copy)]
struct ActiveBoardProgress {
    board: Option<TetrisBoard>,
    started_at: Option<Instant>,
    current_first_bag_idx: usize,
    completed_first_bags: usize,
    direct_first_bags: usize,
    recovered_first_bags: usize,
    failed_first_bags: usize,
    unique_safe_boards: usize,
}

impl ActiveBoardProgress {
    /// Creates an inactive worker-progress slot.
    fn inactive() -> Self {
        Self {
            board: None,
            started_at: None,
            current_first_bag_idx: 0,
            completed_first_bags: 0,
            direct_first_bags: 0,
            recovered_first_bags: 0,
            failed_first_bags: 0,
            unique_safe_boards: 0,
        }
    }

    /// Marks a worker as actively certifying one board.
    fn start(board: TetrisBoard, started_at: Instant) -> Self {
        Self {
            board: Some(board),
            started_at: Some(started_at),
            ..Self::inactive()
        }
    }

    /// Updates the active worker slot after a top-level first-bag progress event.
    fn apply_update(&mut self, update: BoardProcessProgressUpdate) {
        debug_assert_eq!(self.board, Some(update.board));
        self.current_first_bag_idx = update.current_first_bag_idx;
        self.completed_first_bags = update.completed_first_bags;
        self.direct_first_bags = update.direct_first_bags;
        self.recovered_first_bags = update.recovered_first_bags;
        self.failed_first_bags = update.failed_first_bags;
        self.unique_safe_boards = update.unique_safe_boards;
    }
}

#[derive(Debug, Clone)]
struct ActiveWorkerSnapshot {
    worker_idx: usize,
    board: TetrisBoard,
    elapsed_secs: f64,
    current_first_bag_idx: usize,
    completed_first_bags: usize,
    remaining_first_bags: usize,
    direct_first_bags: usize,
    recovered_first_bags: usize,
    failed_first_bags: usize,
    unique_safe_boards: usize,
    first_bags_per_sec: f64,
}

struct ActiveWorkerProgress {
    slots: Vec<Mutex<ActiveBoardProgress>>,
}

impl ActiveWorkerProgress {
    /// Creates one inactive progress slot per worker.
    fn new(worker_count: usize) -> Self {
        let slots = (0..worker_count.max(1))
            .map(|_| Mutex::new(ActiveBoardProgress::inactive()))
            .collect();
        Self { slots }
    }

    /// Marks a worker as certifying a board.
    fn start_board(&self, worker_idx: usize, board: TetrisBoard) {
        let Some(slot) = self.slots.get(worker_idx) else {
            debug_assert!(false, "worker progress slot must exist");
            return;
        };
        let mut progress = slot
            .lock()
            .expect("active worker progress mutex poisoned while starting board");
        *progress = ActiveBoardProgress::start(board, Instant::now());
    }

    /// Updates a worker's active board counters.
    fn update_board(&self, worker_idx: usize, update: BoardProcessProgressUpdate) {
        let Some(slot) = self.slots.get(worker_idx) else {
            debug_assert!(false, "worker progress slot must exist");
            return;
        };
        let mut progress = slot
            .lock()
            .expect("active worker progress mutex poisoned while updating board");
        progress.apply_update(update);
    }

    /// Clears a worker slot after its board is solved, failed, or abandoned on cancellation.
    fn clear_board(&self, worker_idx: usize) {
        let Some(slot) = self.slots.get(worker_idx) else {
            debug_assert!(false, "worker progress slot must exist");
            return;
        };
        let mut progress = slot
            .lock()
            .expect("active worker progress mutex poisoned while clearing board");
        *progress = ActiveBoardProgress::inactive();
    }

    /// Returns active worker snapshots for progress reporting.
    fn snapshot(&self, now: Instant) -> Vec<ActiveWorkerSnapshot> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(worker_idx, slot)| {
                let progress = slot
                    .lock()
                    .expect("active worker progress mutex poisoned while reading board");
                let board = progress.board?;
                let started_at = progress.started_at?;
                let elapsed_secs = now.duration_since(started_at).as_secs_f64();
                let first_bags_per_sec = if elapsed_secs > 0.0 {
                    progress.completed_first_bags as f64 / elapsed_secs
                } else {
                    0.0
                };
                Some(ActiveWorkerSnapshot {
                    worker_idx,
                    board,
                    elapsed_secs,
                    current_first_bag_idx: progress.current_first_bag_idx,
                    completed_first_bags: progress.completed_first_bags,
                    remaining_first_bags: BAG_PERMUTATION_COUNT
                        .saturating_sub(progress.completed_first_bags),
                    direct_first_bags: progress.direct_first_bags,
                    recovered_first_bags: progress.recovered_first_bags,
                    failed_first_bags: progress.failed_first_bags,
                    unique_safe_boards: progress.unique_safe_boards,
                    first_bags_per_sec,
                })
            })
            .collect()
    }
}

/// Low-contention scheduler for safe-set boards discovered during closure expansion.
struct Scheduler {
    queue: SegQueue<TetrisBoard>,
    board_states: DashMap<TetrisBoard, BoardState>,
    in_flight_count: AtomicUsize,
    solved_board_count: AtomicUsize,
    failed_board_count: AtomicUsize,
    abandoned_board_count: AtomicUsize,
    queue_pushes: AtomicUsize,
    queue_claims: AtomicUsize,
    queue_dedup_rejections: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
struct SchedulerSnapshot {
    tracked_boards: usize,
    pending_boards: usize,
    in_flight_boards: usize,
    solved_boards: usize,
    failed_boards: usize,
    abandoned_boards: usize,
    queue_pushes: usize,
    queue_claims: usize,
    queue_dedup_rejections: usize,
}

impl Scheduler {
    /// Seeds the concurrent queue and state map from one initial safe-set board.
    fn new(start_board: TetrisBoard) -> Self {
        let board_states = DashMap::new();
        board_states.insert(start_board, BoardState::Pending);

        let queue = SegQueue::new();
        queue.push(start_board);

        Self {
            queue,
            board_states,
            in_flight_count: AtomicUsize::new(0),
            solved_board_count: AtomicUsize::new(0),
            failed_board_count: AtomicUsize::new(0),
            abandoned_board_count: AtomicUsize::new(0),
            queue_pushes: AtomicUsize::new(1),
            queue_claims: AtomicUsize::new(0),
            queue_dedup_rejections: AtomicUsize::new(0),
        }
    }

    /// Attempts to enqueue a newly discovered safe-set board exactly once.
    fn enqueue_discovered_board(&self, board: TetrisBoard) -> bool {
        match self.board_states.entry(board) {
            Entry::Occupied(_) => {
                self.queue_dedup_rejections
                    .fetch_add(1, AtomicOrdering::Relaxed);
                false
            }
            Entry::Vacant(entry) => {
                entry.insert(BoardState::Pending);
                self.queue.push(board);
                self.queue_pushes.fetch_add(1, AtomicOrdering::Relaxed);
                true
            }
        }
    }

    /// Claims the next pending board for a worker, skipping stale duplicate queue entries.
    fn claim_next_board(&self) -> ClaimDecision {
        loop {
            match self.queue.pop() {
                Some(board) => {
                    let Some(mut state) = self.board_states.get_mut(&board) else {
                        continue;
                    };
                    if *state != BoardState::Pending {
                        continue;
                    }
                    *state = BoardState::InFlight;
                    drop(state);
                    self.in_flight_count.fetch_add(1, AtomicOrdering::Relaxed);
                    self.queue_claims.fetch_add(1, AtomicOrdering::Relaxed);
                    return ClaimDecision::Claimed(board);
                }
                None => {
                    if self.in_flight_count.load(AtomicOrdering::Acquire) == 0 {
                        return ClaimDecision::Exhausted;
                    }
                    return ClaimDecision::WaitForInFlight;
                }
            }
        }
    }

    /// Marks one claimed board as solved and releases its in-flight slot.
    fn record_board_solved(&self, board: TetrisBoard) {
        let Some(mut state) = self.board_states.get_mut(&board) else {
            debug_assert!(false, "solved board must still exist in scheduler state");
            return;
        };
        debug_assert_eq!(*state, BoardState::InFlight);
        *state = BoardState::Solved;
        drop(state);
        self.in_flight_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.solved_board_count
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Marks one claimed board as failed and releases its in-flight slot.
    fn record_board_failed(&self, board: TetrisBoard) {
        let Some(mut state) = self.board_states.get_mut(&board) else {
            debug_assert!(false, "failed board must still exist in scheduler state");
            return;
        };
        debug_assert_eq!(*state, BoardState::InFlight);
        *state = BoardState::Failed;
        drop(state);
        self.in_flight_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.failed_board_count
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Releases one claimed board without marking it solved after global cancellation.
    fn record_board_abandoned(&self, board: TetrisBoard) {
        let Some(mut state) = self.board_states.get_mut(&board) else {
            debug_assert!(false, "abandoned board must still exist in scheduler state");
            return;
        };
        debug_assert_eq!(*state, BoardState::InFlight);
        *state = BoardState::Abandoned;
        drop(state);
        self.in_flight_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.abandoned_board_count
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Returns the total number of tracked boards across all states.
    fn tracked_board_count(&self) -> usize {
        self.board_states.len()
    }

    /// Returns the number of in-flight boards.
    fn in_flight_board_count(&self) -> usize {
        self.in_flight_count.load(AtomicOrdering::Relaxed)
    }

    /// Returns the number of solved boards.
    fn solved_board_count(&self) -> usize {
        self.solved_board_count.load(AtomicOrdering::Relaxed)
    }

    /// Returns the number of failed boards.
    fn failed_board_count(&self) -> usize {
        self.failed_board_count.load(AtomicOrdering::Relaxed)
    }

    /// Returns the number of boards abandoned after another worker recorded a fatal failure.
    fn abandoned_board_count(&self) -> usize {
        self.abandoned_board_count.load(AtomicOrdering::Relaxed)
    }

    /// Returns the number of pending boards implied by tracked state.
    fn pending_board_count(&self) -> usize {
        self.tracked_board_count()
            .saturating_sub(self.in_flight_board_count())
            .saturating_sub(self.solved_board_count())
            .saturating_sub(self.failed_board_count())
            .saturating_sub(self.abandoned_board_count())
    }

    /// Returns a point-in-time scheduler snapshot for reporting.
    fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            tracked_boards: self.tracked_board_count(),
            pending_boards: self.pending_board_count(),
            in_flight_boards: self.in_flight_board_count(),
            solved_boards: self.solved_board_count(),
            failed_boards: self.failed_board_count(),
            abandoned_boards: self.abandoned_board_count(),
            queue_pushes: self.queue_pushes.load(AtomicOrdering::Relaxed),
            queue_claims: self.queue_claims.load(AtomicOrdering::Relaxed),
            queue_dedup_rejections: self.queue_dedup_rejections.load(AtomicOrdering::Relaxed),
        }
    }
}

/// Global runtime metrics updated by workers with low-contention atomics.
struct RuntimeMetrics {
    processed_boards: AtomicUsize,
    processed_first_bags: AtomicUsize,
    direct_first_bags: AtomicUsize,
    recovered_first_bags: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeMetricsSnapshot {
    processed_boards: usize,
    processed_first_bags: usize,
    direct_first_bags: usize,
    recovered_first_bags: usize,
}

impl RuntimeMetrics {
    /// Creates zeroed runtime counters.
    fn new() -> Self {
        Self {
            processed_boards: AtomicUsize::new(0),
            processed_first_bags: AtomicUsize::new(0),
            direct_first_bags: AtomicUsize::new(0),
            recovered_first_bags: AtomicUsize::new(0),
        }
    }

    /// Returns a point-in-time runtime metrics snapshot.
    fn snapshot(&self) -> RuntimeMetricsSnapshot {
        RuntimeMetricsSnapshot {
            processed_boards: self.processed_boards.load(AtomicOrdering::Relaxed),
            processed_first_bags: self.processed_first_bags.load(AtomicOrdering::Relaxed),
            direct_first_bags: self.direct_first_bags.load(AtomicOrdering::Relaxed),
            recovered_first_bags: self.recovered_first_bags.load(AtomicOrdering::Relaxed),
        }
    }
}

/// Keeps a bounded sample of certified first-bag witnesses for end-of-run reporting.
struct SampleCollector {
    limit: usize,
    entries: Mutex<Vec<CertifiedFirstBag>>,
}

impl SampleCollector {
    /// Creates a bounded sample collector.
    fn new(limit: usize) -> Self {
        Self {
            limit,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Merges one worker-local sample batch into the global bounded sample.
    fn merge_entries(&self, mut incoming: Vec<CertifiedFirstBag>) {
        if self.limit == 0 || incoming.is_empty() {
            return;
        }
        trim_sample_entries(&mut incoming, self.limit);
        let mut entries = self
            .entries
            .lock()
            .expect("sample collector mutex poisoned while merging");
        entries.extend(incoming);
        trim_sample_entries(&mut entries, self.limit);
    }

    /// Returns the final bounded sample set.
    fn snapshot(&self) -> Vec<CertifiedFirstBag> {
        self.entries
            .lock()
            .expect("sample collector mutex poisoned while reading")
            .clone()
    }
}

/// Shared runtime state used by worker threads.
struct SharedRuntimeContext {
    search: SharedSearchContext,
    scheduler: Scheduler,
    metrics: RuntimeMetrics,
    active_workers: ActiveWorkerProgress,
    samples: SampleCollector,
    failure: OnceLock<(TetrisBoard, FailureExample)>,
    cancel_requested: AtomicBool,
    shutdown_requested: AtomicBool,
}

impl SharedRuntimeContext {
    /// Creates runtime state for one closure run from an initial board.
    fn new(start_board: TetrisBoard, sample_limit: usize) -> Self {
        Self::with_worker_count(start_board, sample_limit, 1)
    }

    /// Creates runtime state with one active-progress slot per worker.
    fn with_worker_count(
        start_board: TetrisBoard,
        sample_limit: usize,
        worker_count: usize,
    ) -> Self {
        Self {
            search: SharedSearchContext::new(),
            scheduler: Scheduler::new(start_board),
            metrics: RuntimeMetrics::new(),
            active_workers: ActiveWorkerProgress::new(worker_count),
            samples: SampleCollector::new(sample_limit),
            failure: OnceLock::new(),
            cancel_requested: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
        }
    }
}

/// Compares two candidate child boards for the local top-k beam at one depth.
fn compare_candidate_nodes(left: &CandidateNode, right: &CandidateNode) -> Ordering {
    left.score
        .total_cmp(&right.score)
        .then_with(|| right.board.as_limbs().cmp(&left.board.as_limbs()))
}

/// Orders certified witnesses so reporting keeps the hardest examples first.
fn compare_sample_entries(left: &CertifiedFirstBag, right: &CertifiedFirstBag) -> Ordering {
    left.backtracks
        .cmp(&right.backtracks)
        .then_with(|| left.first_bag.cmp(&right.first_bag))
        .reverse()
}

/// Trims a witness vector down to the globally preferred sample ordering.
fn trim_sample_entries(entries: &mut Vec<CertifiedFirstBag>, limit: usize) {
    if entries.len() <= limit {
        entries.sort_by(compare_sample_entries);
        return;
    }
    entries.sort_by(compare_sample_entries);
    entries.truncate(limit);
}

/// Inserts a candidate into a fixed-size descending top-k buffer if it is competitive.
fn insert_top_candidate(
    top_candidates: &mut [CandidateNode; PLANNER_WIDTH],
    count: &mut usize,
    candidate: CandidateNode,
) {
    let mut insert_at = (*count).min(PLANNER_WIDTH);
    while insert_at > 0
        && compare_candidate_nodes(&candidate, &top_candidates[insert_at.saturating_sub(1)]).is_gt()
    {
        insert_at -= 1;
    }

    if *count < PLANNER_WIDTH {
        for idx in (insert_at..*count).rev() {
            top_candidates[idx + 1] = top_candidates[idx];
        }
        top_candidates[insert_at] = candidate;
        *count += 1;
        return;
    }

    if insert_at == PLANNER_WIDTH {
        return;
    }

    for idx in (insert_at..(PLANNER_WIDTH - 1)).rev() {
        top_candidates[idx + 1] = top_candidates[idx];
    }
    top_candidates[insert_at] = candidate;
}

/// Thin wrapper around the allocation-free planner for a single forced sequence query.
#[inline(always)]
fn search_forced_sequence_core<TerminalFnT>(
    start_board: TetrisBoard,
    forced_sequence: &ForcedBag,
    terminal_fn: Option<&TerminalFnT>,
) -> SearchOutcome
where
    TerminalFnT: Fn(TetrisBoard) -> bool,
{
    THREAD_LOCAL_PLANNER.with(|planner| {
        planner
            .borrow_mut()
            .search(start_board, forced_sequence, terminal_fn)
    })
}

/// Collects first-bag candidate scripts from the allocation-free planner.
#[inline(always)]
fn collect_first_bag_candidates_core(
    start_board: TetrisBoard,
    forced_sequence: &ForcedBag,
    limit: usize,
) -> CandidateBuffer {
    THREAD_LOCAL_PLANNER.with(|planner| {
        planner
            .borrow_mut()
            .collect_witnesses(start_board, forced_sequence, limit)
    })
}

/// Runs one direct 7-piece search back into the safe set without consulting outer caches.
#[inline(always)]
fn search_safe_set_script_core(start_board: TetrisBoard, bag: &ForcedBag) -> Option<SearchWitness> {
    let terminal = |candidate: TetrisBoard| terminal_fn(&candidate);
    match search_forced_sequence_core(start_board, bag, Some(&terminal)) {
        SearchOutcome::Success(witness) => Some(witness),
        SearchOutcome::Exhausted => None,
    }
}

/// Memoizes direct 7-piece safe-set queries keyed by `(board, bag)` with per-key ownership.
fn query_direct_search_cached_with<F>(
    context: &SharedSearchContext,
    start_board: TetrisBoard,
    bag: &ForcedBag,
    compute: F,
) -> Option<SearchWitness>
where
    F: FnOnce() -> Option<SearchWitness>,
{
    let key = DirectSearchCacheKey {
        board: start_board,
        bag: *bag,
    };
    let signal = loop {
        match context.direct_cache.entry(key) {
            Entry::Occupied(entry) => {
                let cached = entry.get().clone();
                drop(entry);
                match cached {
                    DirectSearchCacheValue::Success(witness) => {
                        context.record_direct_cache_hit();
                        return Some(witness);
                    }
                    DirectSearchCacheValue::Failure => {
                        context.record_direct_cache_hit();
                        return None;
                    }
                    DirectSearchCacheValue::InProgress(signal) => {
                        context.record_direct_cache_wait();
                        signal.wait();
                    }
                }
            }
            Entry::Vacant(entry) => {
                context.record_direct_cache_miss();
                let signal = Arc::new(WorkSignal::new());
                entry.insert(DirectSearchCacheValue::InProgress(Arc::clone(&signal)));
                break signal;
            }
        }
    };

    let result = compute();
    let cached = match result {
        Some(witness) => DirectSearchCacheValue::Success(witness),
        None => DirectSearchCacheValue::Failure,
    };
    context.direct_cache.insert(key, cached);
    signal.notify_ready();
    result
}

/// Memoizes direct 7-piece safe-set queries keyed by `(board, bag)`.
fn query_direct_search_cached(
    context: &SharedSearchContext,
    start_board: TetrisBoard,
    bag: &ForcedBag,
) -> Option<SearchWitness> {
    query_direct_search_cached_with(context, start_board, bag, || {
        search_safe_set_script_core(start_board, bag)
    })
}

/// Memoizes direct 7-piece safe-set queries keyed by `(board, bag)`.
fn search_safe_set_script_cached(
    context: &SharedSearchContext,
    start_board: TetrisBoard,
    bag: &ForcedBag,
) -> Option<SearchWitness> {
    query_direct_search_cached(context, start_board, bag)
}

/// Certifies one forced bag recursively, using direct search as the base case.
fn certify_forced_bag_at_depth(
    context: &SharedSearchContext,
    board: TetrisBoard,
    forced_bag: &ForcedBag,
    remaining_depth: usize,
) -> FirstBagCertification {
    debug_assert!(remaining_depth >= 1);

    if let Some(witness) = search_safe_set_script_cached(context, board, forced_bag) {
        let safe_boards = Arc::from([witness.board]);
        context.record_forced_bag_direct(remaining_depth);
        return FirstBagCertification::Certified {
            entry: CertifiedFirstBag {
                first_bag: *forced_bag,
                placements: witness.placements,
                final_board: witness.board,
                kind: CertificationKind::Direct,
                backtracks: witness.backtracks,
            },
            proof: RecursiveForcedBagProof::Direct { safe_boards },
        };
    }

    if remaining_depth == 1 {
        context.record_forced_bag_failed(remaining_depth);
        return FirstBagCertification::Failed(FailureExample::leaf(board, *forced_bag));
    }

    let candidates = collect_first_bag_candidates_core(
        board,
        forced_bag,
        recovery_candidate_limit_for_remaining_depth(remaining_depth),
    );
    let mut representative_failure = FailureExample::leaf(board, *forced_bag);
    let mut candidates_examined = 0usize;

    for candidate in candidates.into_iter() {
        let candidate = *candidate;
        candidates_examined += 1;
        if terminal_fn(&candidate.board) {
            let safe_boards = Arc::from([candidate.board]);
            context.record_candidate_scripts_examined(remaining_depth, candidates_examined);
            context.record_forced_bag_direct(remaining_depth);
            return FirstBagCertification::Certified {
                entry: CertifiedFirstBag {
                    first_bag: *forced_bag,
                    placements: candidate.placements,
                    final_board: candidate.board,
                    kind: CertificationKind::Direct,
                    backtracks: candidate.backtracks,
                },
                proof: RecursiveForcedBagProof::Direct { safe_boards },
            };
        }

        match certify_board_at_depth(context, candidate.board, remaining_depth - 1) {
            RecursiveBoardCertification::Impossible { failure } => {
                representative_failure = FailureExample::with_child(
                    board,
                    *forced_bag,
                    candidate.placements,
                    candidate.board,
                    failure,
                );
                continue;
            }
            RecursiveBoardCertification::Complete { safe_boards } => {
                context.record_candidate_scripts_examined(remaining_depth, candidates_examined);
                context.record_forced_bag_recovered(remaining_depth, safe_boards.len());
                return FirstBagCertification::Certified {
                    entry: CertifiedFirstBag {
                        first_bag: *forced_bag,
                        placements: candidate.placements,
                        final_board: candidate.board,
                        kind: CertificationKind::Recovered,
                        backtracks: candidate.backtracks,
                    },
                    proof: RecursiveForcedBagProof::Recovered {
                        intermediate_board: candidate.board,
                        child_remaining_depth: (remaining_depth - 1) as u8,
                        safe_boards,
                    },
                };
            }
        }
    }

    context.record_candidate_scripts_examined(remaining_depth, candidates_examined);
    context.record_forced_bag_failed(remaining_depth);
    FirstBagCertification::Failed(representative_failure)
}

/// Memoizes recursive board certification keyed by board, depth, and tier with per-key ownership.
fn certify_board_at_depth_cached_with<F>(
    context: &SharedSearchContext,
    board: TetrisBoard,
    remaining_depth: usize,
    compute: F,
) -> RecursiveBoardCertification
where
    F: FnOnce() -> RecursiveBoardCertification,
{
    let key = RecursiveBoardCacheKey {
        board,
        remaining_depth: remaining_depth as u8,
    };
    let signal = loop {
        match context.recursive_board_cache.entry(key) {
            Entry::Occupied(entry) => {
                let cached = entry.get().clone();
                drop(entry);
                match cached {
                    RecursiveBoardStatus::Complete { safe_boards } => {
                        context.record_recursive_cache_complete_hit(remaining_depth);
                        return RecursiveBoardCertification::Complete { safe_boards };
                    }
                    RecursiveBoardStatus::Impossible { failure } => {
                        context.record_recursive_cache_impossible_hit(remaining_depth);
                        return RecursiveBoardCertification::Impossible { failure };
                    }
                    RecursiveBoardStatus::InProgress(signal) => {
                        context.record_recursive_cache_wait(remaining_depth);
                        signal.wait();
                    }
                }
            }
            Entry::Vacant(entry) => {
                context.record_recursive_cache_miss(remaining_depth);
                let signal = Arc::new(WorkSignal::new());
                entry.insert(RecursiveBoardStatus::InProgress(Arc::clone(&signal)));
                break signal;
            }
        }
    };

    let result = compute();
    match &result {
        RecursiveBoardCertification::Complete { .. } => {
            context.record_recursive_board_complete(remaining_depth);
        }
        RecursiveBoardCertification::Impossible { .. } => {
            context.record_recursive_board_impossible(remaining_depth);
        }
    }
    let cached = match &result {
        RecursiveBoardCertification::Complete { safe_boards } => RecursiveBoardStatus::Complete {
            safe_boards: Arc::clone(safe_boards),
        },
        RecursiveBoardCertification::Impossible { failure } => RecursiveBoardStatus::Impossible {
            failure: failure.clone(),
        },
    };
    context.recursive_board_cache.insert(key, cached);
    signal.notify_ready();
    result
}

/// Certifies one board under all forced bags at the requested recursive depth.
fn certify_board_at_depth(
    context: &SharedSearchContext,
    board: TetrisBoard,
    remaining_depth: usize,
) -> RecursiveBoardCertification {
    debug_assert!(remaining_depth >= 1);
    certify_board_at_depth_cached_with(context, board, remaining_depth, || {
        let mut discovered_safe_boards = HashSet::new();
        for &forced_bag in ALL_FORCED_BAG_PERMUTATIONS.iter() {
            match certify_forced_bag_at_depth(context, board, &forced_bag, remaining_depth) {
                FirstBagCertification::Certified { proof, .. } => {
                    discovered_safe_boards.extend(proof.discovered_safe_boards().iter().copied());
                }
                FirstBagCertification::Failed(failure) => {
                    return RecursiveBoardCertification::Impossible { failure };
                }
            }
        }

        let safe_boards = Arc::<[TetrisBoard]>::from(
            discovered_safe_boards
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        RecursiveBoardCertification::Complete { safe_boards }
    })
}

/// Compatibility wrapper for the old "intermediate board" notion: all next bags must succeed directly.
fn certify_intermediate_board_cached(
    context: &SharedSearchContext,
    board: TetrisBoard,
) -> RecursiveBoardCertification {
    certify_board_at_depth(context, board, 1)
}

/// Compatibility wrapper that preserves the old per-key ownership tests for direct second-bag sweeps.
fn certify_intermediate_board_cached_with<F>(
    context: &SharedSearchContext,
    board: TetrisBoard,
    compute: F,
) -> RecursiveBoardCertification
where
    F: FnOnce() -> RecursiveBoardCertification,
{
    certify_board_at_depth_cached_with(context, board, 1, compute)
}

/// Public wrapper for top-level first-bag certification under the configured recovery depth.
fn certify_first_bag(
    context: &SharedSearchContext,
    board: TetrisBoard,
    first_bag: &ForcedBag,
) -> FirstBagCertification {
    certify_forced_bag_at_depth(context, board, first_bag, RECOVERY_BAG_DEPTH)
}

/// Prints a small sample of certified first-bag witnesses for debugging and inspection.
fn print_sample_certifications(results: &[CertifiedFirstBag], limit: usize) {
    let mut samples = results.to_vec();
    trim_sample_entries(&mut samples, limit);

    for entry in samples.into_iter().take(limit) {
        println!("sample_first_bag={:?}", entry.first_bag);
        println!("sample_kind={:?}", entry.kind);
        println!("sample_final_board={:?}", entry.final_board);
        println!("sample_backtracks={}", entry.backtracks);
        println!(
            "sample_placements={}",
            entry
                .placements
                .into_iter()
                .map(|placement| placement.to_string())
                .join(" | ")
        );
    }
}

/// Processes one board by certifying all 5040 first-bag permutations.
fn process_board_with_hooks<FProgress, FDiscovered, FCancelled>(
    context: &SharedSearchContext,
    board: TetrisBoard,
    sample_limit: usize,
    mut is_cancelled: FCancelled,
    mut on_progress: FProgress,
    mut on_new_safe_board: FDiscovered,
) -> BoardProcessResult
where
    FProgress: FnMut(BoardProcessProgressUpdate),
    FDiscovered: FnMut(TetrisBoard),
    FCancelled: FnMut() -> bool,
{
    let mut discovered_safe_boards = HashSet::new();
    let mut failure = None;
    let mut cancelled = false;
    let mut processed_first_bags = 0usize;
    let mut direct_first_bags = 0usize;
    let mut recovered_first_bags = 0usize;
    let mut failed_first_bags = 0usize;
    let mut sample_entries = Vec::new();

    for (bag_idx, &first_bag) in ALL_FORCED_BAG_PERMUTATIONS.iter().enumerate() {
        if is_cancelled() {
            cancelled = true;
            break;
        }
        on_progress(BoardProcessProgressUpdate {
            board,
            current_first_bag_idx: bag_idx + 1,
            completed_first_bags: processed_first_bags,
            direct_first_bags,
            recovered_first_bags,
            failed_first_bags,
            unique_safe_boards: discovered_safe_boards.len(),
        });
        processed_first_bags += 1;
        match certify_first_bag(context, board, &first_bag) {
            FirstBagCertification::Certified { entry, proof } => {
                match entry.kind {
                    CertificationKind::Direct => direct_first_bags += 1,
                    CertificationKind::Recovered => recovered_first_bags += 1,
                }
                publish_new_safe_boards(
                    &mut discovered_safe_boards,
                    proof.discovered_safe_boards().iter().copied(),
                    &mut on_new_safe_board,
                );
                sample_entries.push(entry);
                if sample_entries.len() > sample_limit.saturating_mul(4).max(4) {
                    trim_sample_entries(&mut sample_entries, sample_limit);
                }
            }
            FirstBagCertification::Failed(example) => {
                failed_first_bags += 1;
                failure = Some(example);
            }
        }
        on_progress(BoardProcessProgressUpdate {
            board,
            current_first_bag_idx: bag_idx + 1,
            completed_first_bags: processed_first_bags,
            direct_first_bags,
            recovered_first_bags,
            failed_first_bags,
            unique_safe_boards: discovered_safe_boards.len(),
        });
        if failure.is_some() {
            break;
        }
    }

    BoardProcessResult {
        board,
        failure,
        cancelled,
        processed_first_bags,
        direct_first_bags,
        recovered_first_bags,
        failed_first_bags,
        sample_entries,
    }
}

/// Publishes each board that is new to the current parent-board proof.
fn publish_new_safe_boards<I, F>(
    discovered_safe_boards: &mut HashSet<TetrisBoard>,
    safe_boards: I,
    on_new_safe_board: &mut F,
) where
    I: IntoIterator<Item = TetrisBoard>,
    F: FnMut(TetrisBoard),
{
    for safe_board in safe_boards {
        if discovered_safe_boards.insert(safe_board) {
            on_new_safe_board(safe_board);
        }
    }
}

/// Formats per-depth proof and recursive-cache counters into a compact progress field.
fn format_depth_metrics(search: &ProgressSnapshot) -> String {
    (1..=RECOVERY_BAG_DEPTH)
        .map(|depth| {
            format!(
                "d{depth}:{{direct:{},recovered:{},failed:{},candidates:{},cache:{}/{}/{},cache_complete:{},cache_impossible:{},board_complete:{},board_impossible:{}}}",
                search.forced_bag_direct_by_depth[depth],
                search.forced_bag_recovered_by_depth[depth],
                search.forced_bag_failed_by_depth[depth],
                search.candidate_scripts_examined_by_depth[depth],
                search.recursive_cache_hits_by_depth[depth],
                search.recursive_cache_misses_by_depth[depth],
                search.recursive_cache_waits_by_depth[depth],
                search.recursive_cache_complete_hits_by_depth[depth],
                search.recursive_cache_impossible_hits_by_depth[depth],
                search.recursive_board_complete_by_depth[depth],
                search.recursive_board_impossible_by_depth[depth],
            )
        })
        .join(",")
}

/// Counts all top-level forced-bag outcomes recorded so far.
fn top_level_forced_bags_completed(search: &ProgressSnapshot) -> usize {
    let depth = RECOVERY_BAG_DEPTH;
    search.forced_bag_direct_by_depth[depth]
        + search.forced_bag_recovered_by_depth[depth]
        + search.forced_bag_failed_by_depth[depth]
}

/// Formats active worker progress in a compact single-line form.
fn format_active_workers(active_workers: &[ActiveWorkerSnapshot]) -> String {
    if active_workers.is_empty() {
        return "none".to_string();
    }

    active_workers
        .iter()
        .map(|worker| {
            format!(
                "w{}:{{board:{:?},elapsed:{:.1}s,current:{}/{},completed:{},remaining:{},direct:{},recovered:{},failed:{},unique_safe:{},rate:{:.2}/s}}",
                worker.worker_idx,
                worker.board,
                worker.elapsed_secs,
                worker.current_first_bag_idx,
                BAG_PERMUTATION_COUNT,
                worker.completed_first_bags,
                worker.remaining_first_bags,
                worker.direct_first_bags,
                worker.recovered_first_bags,
                worker.failed_first_bags,
                worker.unique_safe_boards,
                worker.first_bags_per_sec,
            )
        })
        .join(",")
}

/// Prints a low-contention runtime snapshot for long-running closure runs.
fn print_runtime_progress(shared: &SharedRuntimeContext, started_at: Instant, completed: bool) {
    let elapsed = started_at.elapsed().as_secs_f64();
    let now = Instant::now();
    let scheduler = shared.scheduler.snapshot();
    let metrics = shared.metrics.snapshot();
    let search = shared.search.progress_snapshot();
    let depth_metrics = format_depth_metrics(&search);
    let active_workers = shared.active_workers.snapshot(now);
    let active_top_level_completed = active_workers
        .iter()
        .map(|worker| worker.completed_first_bags)
        .sum::<usize>();
    let active_top_level_remaining = active_workers
        .iter()
        .map(|worker| worker.remaining_first_bags)
        .sum::<usize>();
    let top_level_completed_total = top_level_forced_bags_completed(&search);
    let top_level_rate = if elapsed > 0.0 {
        top_level_completed_total as f64 / elapsed
    } else {
        0.0
    };
    let queue_push_rate = if elapsed > 0.0 {
        scheduler.queue_pushes as f64 / elapsed
    } else {
        0.0
    };
    let active_worker_metrics = format_active_workers(&active_workers);
    let failure_recorded = shared.failure.get().is_some();

    let prefix = if completed {
        "[safe-set] completed"
    } else {
        "[safe-set] progress"
    };

    println!(
        "{prefix} elapsed={elapsed:.1}s tracked_boards={} global_unique_frontier_boards={} pending_boards={} in_flight_boards={} solved_boards={} failed_boards={} abandoned_boards={} processed_boards={} processed_first_bags={} direct_first_bags={} recovered_first_bags={} recovered_safe_board_outputs={} top_level=[completed_total:{},active_completed:{},active_remaining:{},rate:{:.2}/s] queue=[pushes:{},push_rate:{:.2}/s,claims:{},dedup:{}] direct_cache=[hit:{},miss:{},wait:{}] recursive_cache=[hit:{},miss:{},wait:{}] proof_depths=[{}] active_workers=[{}] cancel_requested={} failure_recorded={}",
        scheduler.tracked_boards,
        scheduler.tracked_boards,
        scheduler.pending_boards,
        scheduler.in_flight_boards,
        scheduler.solved_boards,
        scheduler.failed_boards,
        scheduler.abandoned_boards,
        metrics.processed_boards,
        metrics.processed_first_bags,
        metrics.direct_first_bags,
        metrics.recovered_first_bags,
        search.recovered_safe_boards_discovered,
        top_level_completed_total,
        active_top_level_completed,
        active_top_level_remaining,
        top_level_rate,
        scheduler.queue_pushes,
        queue_push_rate,
        scheduler.queue_claims,
        scheduler.queue_dedup_rejections,
        search.direct_cache_hits,
        search.direct_cache_misses,
        search.direct_cache_waits,
        search.recursive_cache_hits,
        search.recursive_cache_misses,
        search.recursive_cache_waits,
        depth_metrics,
        active_worker_metrics,
        shared.cancel_requested.load(AtomicOrdering::Acquire),
        failure_recorded,
    );
}

/// Background reporter that periodically snapshots the concurrent runtime state.
fn run_progress_reporter(shared: &Arc<SharedRuntimeContext>, started_at: Instant) {
    let mut last_report_at = Instant::now();
    loop {
        if shared.shutdown_requested.load(AtomicOrdering::Acquire) {
            break;
        }
        if last_report_at.elapsed() >= PROGRESS_LOG_INTERVAL {
            print_runtime_progress(shared, started_at, false);
            last_report_at = Instant::now();
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimDecision {
    Claimed(TetrisBoard),
    WaitForInFlight,
    Exhausted,
}

/// Worker loop that repeatedly claims boards, processes them, and commits outputs.
fn run_worker(worker_idx: usize, shared: &Arc<SharedRuntimeContext>) {
    let mut local_samples = Vec::new();
    loop {
        if shared.cancel_requested.load(AtomicOrdering::Acquire) {
            break;
        }

        let board = match shared.scheduler.claim_next_board() {
            ClaimDecision::Claimed(board) => board,
            ClaimDecision::WaitForInFlight => {
                thread::yield_now();
                continue;
            }
            ClaimDecision::Exhausted => break,
        };

        shared.active_workers.start_board(worker_idx, board);
        let BoardProcessResult {
            board: processed_board,
            failure,
            cancelled,
            processed_first_bags: board_processed_first_bags,
            direct_first_bags,
            recovered_first_bags,
            failed_first_bags: _failed_first_bags,
            sample_entries: board_samples,
            ..
        } = process_board_with_hooks(
            &shared.search,
            board,
            shared.samples.limit,
            || shared.cancel_requested.load(AtomicOrdering::Acquire),
            |update| shared.active_workers.update_board(worker_idx, update),
            |discovered| {
                debug_assert!(
                    terminal_fn(&discovered),
                    "only terminal-safe boards should enter the closure frontier"
                );
                if !shared.cancel_requested.load(AtomicOrdering::Acquire) {
                    shared.scheduler.enqueue_discovered_board(discovered);
                }
            },
        );

        if let Some(example) = failure {
            shared.scheduler.record_board_failed(processed_board);
            let _ = shared.failure.set((processed_board, example));
            shared.cancel_requested.store(true, AtomicOrdering::Release);
            shared.active_workers.clear_board(worker_idx);
            break;
        }

        if cancelled {
            shared.scheduler.record_board_abandoned(processed_board);
            shared.active_workers.clear_board(worker_idx);
            break;
        }

        shared
            .metrics
            .processed_boards
            .fetch_add(1, AtomicOrdering::Relaxed);
        shared
            .metrics
            .processed_first_bags
            .fetch_add(board_processed_first_bags, AtomicOrdering::Relaxed);
        shared
            .metrics
            .direct_first_bags
            .fetch_add(direct_first_bags, AtomicOrdering::Relaxed);
        shared
            .metrics
            .recovered_first_bags
            .fetch_add(recovered_first_bags, AtomicOrdering::Relaxed);
        shared.scheduler.record_board_solved(processed_board);
        shared.active_workers.clear_board(worker_idx);

        local_samples.extend(board_samples);
        if local_samples.len() > shared.samples.limit.saturating_mul(8).max(8) {
            trim_sample_entries(&mut local_samples, shared.samples.limit);
        }
    }

    shared.samples.merge_entries(local_samples);
}

/// CLI entrypoint that certifies all 5040 first bags from the empty start board.
fn main() -> Result<()> {
    let cli = Cli::parse();
    let start_board = TetrisBoard::new();
    let worker_count = cli.workers.max(1);

    println!("tetris_safe_set");
    println!("planner_width={}", PLANNER_WIDTH);
    println!("best_first_node_capacity={}", BEST_FIRST_NODE_CAPACITY);
    println!("recovery_bag_depth={}", RECOVERY_BAG_DEPTH);
    println!("recovery_candidate_limits={:?}", RECOVERY_CANDIDATE_LIMITS);
    println!("workers={worker_count}");
    println!("start_board={start_board:?}");
    println!("first_bag_permutation_count={BAG_PERMUTATION_COUNT}");
    println!("second_bag_permutation_count={BAG_PERMUTATION_COUNT}");

    let context = Arc::new(SharedRuntimeContext::with_worker_count(
        start_board,
        cli.print_witness_limit,
        worker_count,
    ));

    let started_at = Instant::now();
    let reporter_shared = Arc::clone(&context);
    let reporter_handle =
        thread::spawn(move || run_progress_reporter(&reporter_shared, started_at));
    let mut worker_handles = Vec::with_capacity(worker_count);
    for worker_idx in 0..worker_count {
        let shared = Arc::clone(&context);
        worker_handles.push(thread::spawn(move || run_worker(worker_idx, &shared)));
    }
    for handle in worker_handles {
        handle.join().expect("worker thread panicked");
    }
    context
        .shutdown_requested
        .store(true, AtomicOrdering::Release);
    reporter_handle
        .join()
        .expect("progress reporter thread panicked");

    if let Some((board, failure)) = context.failure.get() {
        let mut message = format!(
            "safe-set certification failed for board={board:?} first_bag={:?}",
            failure.top_level_bag()
        );
        if let Some(placements) = failure.top_level_placements() {
            message.push_str(&format!(
                " first_bag_placements={}",
                placements
                    .into_iter()
                    .map(|placement| placement.to_string())
                    .join(" | ")
            ));
        }
        if let Some(intermediate_board) = failure.top_level_intermediate_board() {
            message.push_str(&format!(" intermediate_board={intermediate_board:?}"));
        }
        if failure.child_failure.is_some() {
            message.push_str(&format!(" failing_recovery_bag={:?}", failure.leaf_bag()));
        }
        bail!("{message}");
    }

    let elapsed = started_at.elapsed();
    print_runtime_progress(&context, started_at, true);
    let search_progress = context.search.progress_snapshot();
    let direct_count = context
        .metrics
        .direct_first_bags
        .load(AtomicOrdering::Relaxed);
    let recovered_count = context
        .metrics
        .recovered_first_bags
        .load(AtomicOrdering::Relaxed);
    let pending_board_count = context.scheduler.pending_board_count();
    let in_flight_board_count = context.scheduler.in_flight_board_count();
    let solved_board_count = context.scheduler.solved_board_count();
    let failed_board_count = context.scheduler.failed_board_count();
    let abandoned_board_count = context.scheduler.abandoned_board_count();
    let tracked_board_count = context.scheduler.tracked_board_count();
    let samples = context.samples.snapshot();

    println!("elapsed_secs={:.3}", elapsed.as_secs_f64());
    println!(
        "processed_boards={}",
        context
            .metrics
            .processed_boards
            .load(AtomicOrdering::Relaxed)
    );
    println!(
        "processed_first_bags={}",
        context
            .metrics
            .processed_first_bags
            .load(AtomicOrdering::Relaxed)
    );
    println!("tracked_boards={tracked_board_count}");
    println!("global_unique_frontier_boards={tracked_board_count}");
    println!("pending_boards={pending_board_count}");
    println!("in_flight_boards={in_flight_board_count}");
    println!("solved_boards={solved_board_count}");
    println!("failed_boards={failed_board_count}");
    println!("abandoned_boards={abandoned_board_count}");
    println!("certified_first_bags={}", direct_count + recovered_count);
    println!("direct_first_bags={direct_count}");
    println!("recovered_first_bags={recovered_count}");
    println!(
        "queue_pushes={}",
        context.scheduler.queue_pushes.load(AtomicOrdering::Relaxed)
    );
    println!(
        "queue_claims={}",
        context.scheduler.queue_claims.load(AtomicOrdering::Relaxed)
    );
    println!(
        "queue_dedup_rejections={}",
        context
            .scheduler
            .queue_dedup_rejections
            .load(AtomicOrdering::Relaxed)
    );
    println!("direct_cache_hits={}", search_progress.direct_cache_hits);
    println!(
        "direct_cache_misses={}",
        search_progress.direct_cache_misses
    );
    println!("direct_cache_waits={}", search_progress.direct_cache_waits);
    println!(
        "recursive_cache_hits={}",
        search_progress.recursive_cache_hits
    );
    println!(
        "recursive_cache_misses={}",
        search_progress.recursive_cache_misses
    );
    println!(
        "recursive_cache_waits={}",
        search_progress.recursive_cache_waits
    );
    println!(
        "recovered_safe_board_outputs={}",
        search_progress.recovered_safe_boards_discovered
    );
    println!(
        "top_level_forced_bags_completed_total={}",
        top_level_forced_bags_completed(&search_progress)
    );
    println!("proof_depths={}", format_depth_metrics(&search_progress));

    if !samples.is_empty() {
        print_sample_certifications(&samples, cli.print_witness_limit);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Converts compact placement indices into the fixed-size script type used by the core.
    fn script_from_indices(indices: [u8; PIECES_PER_BAG]) -> PlacementScript {
        indices.map(TetrisPiecePlacement::from_index)
    }

    /// Builds a test witness with a fixed script and chosen final board.
    fn test_witness(indices: [u8; PIECES_PER_BAG], board: TetrisBoard) -> SearchWitness {
        SearchWitness {
            board,
            placements: script_from_indices(indices),
            backtracks: 0,
        }
    }

    /// Loads a small list of witnesses into the fixed-capacity candidate buffer used by the core.
    fn candidate_buffer(items: &[SearchWitness]) -> CandidateBuffer {
        let mut buffer = CandidateBuffer::new();
        for &item in items {
            let pushed = buffer.try_push(item);
            assert!(pushed, "test candidate buffer overflow");
        }
        buffer
    }

    /// Constructs a board with exactly one occupied cell.
    fn board_with_bit(col: usize, row: usize) -> TetrisBoard {
        let mut board = TetrisBoard::new();
        board.set_bit(col, row);
        board
    }

    /// Constructs a board directly from its column-major limb representation.
    fn board_from_limbs(limbs: [u32; TetrisBoard::WIDTH]) -> TetrisBoard {
        let mut bytes = [0u8; 40];
        for (idx, limb) in limbs.into_iter().enumerate() {
            let start = idx * std::mem::size_of::<u32>();
            bytes[start..start + std::mem::size_of::<u32>()].copy_from_slice(&limb.to_ne_bytes());
        }
        TetrisBoard::from(bytes)
    }

    /// Renders a fixed placement script in the same format used by runtime error reporting.
    fn placement_script_string(placements: PlacementScript) -> String {
        placements
            .into_iter()
            .map(|placement| placement.to_string())
            .join(" | ")
    }

    /// Waits for a concurrent test condition that should become true after a worker blocks.
    fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
        let started_at = Instant::now();
        while !condition() {
            if started_at.elapsed() >= Duration::from_secs(1) {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// Sweeps a board through all first bags and fails on the first uncertified bag.
    fn assert_board_certifies_all_first_bags(label: &str, board: TetrisBoard) {
        let context = SharedSearchContext::new();
        let mut discovered_intermediate_boards = HashSet::new();

        for (bag_idx, first_bag) in ALL_FORCED_BAG_PERMUTATIONS.iter().copied().enumerate() {
            if search_safe_set_script_cached(&context, board, &first_bag).is_none() {
                let candidates = collect_first_bag_candidates_core(
                    board,
                    &first_bag,
                    recovery_candidate_limit_for_remaining_depth(RECOVERY_BAG_DEPTH),
                );
                for candidate in candidates.into_iter() {
                    let candidate = *candidate;
                    if !terminal_fn(&candidate.board) {
                        discovered_intermediate_boards.insert(candidate.board);
                    }
                }
            }

            let result = certify_first_bag(&context, board, &first_bag);
            match result {
                FirstBagCertification::Certified { .. } => {
                    eprintln!(
                        "[{label}] {}/{} first_bag={:?} certified",
                        bag_idx + 1,
                        BAG_PERMUTATION_COUNT,
                        first_bag
                    );
                }
                FirstBagCertification::Failed(example) => {
                    panic!(
                        "{label} failed certification for first bag {:?} intermediate={:?} recovery_bag={:?}",
                        example.top_level_bag(),
                        example
                            .top_level_intermediate_board()
                            .map(|candidate_board| format!("{candidate_board:?}"))
                            .unwrap_or_else(|| "<none>".to_string()),
                        example
                            .failing_recovery_bag()
                            .map(|bag| format!("{bag:?}"))
                            .unwrap_or_else(|| "<none>".to_string())
                    );
                }
            }
        }

        eprintln!(
            "[{label}] discovered_intermediate_boards={}",
            discovered_intermediate_boards.len()
        );
    }

    /// Test-only helper that runs first-bag certification against injected search oracles.
    fn certify_first_bag_with<FDirect, FCandidates>(
        board: TetrisBoard,
        first_bag: &ForcedBag,
        second_bags: &[ForcedBag],
        mut direct_search: FDirect,
        mut candidates_fn: FCandidates,
    ) -> FirstBagCertification
    where
        FDirect: FnMut(TetrisBoard, &ForcedBag) -> Option<SearchWitness>,
        FCandidates: FnMut(TetrisBoard, &ForcedBag) -> CandidateBuffer,
    {
        if let Some(witness) = direct_search(board, first_bag) {
            let safe_boards = Arc::from([witness.board]);
            return FirstBagCertification::Certified {
                entry: CertifiedFirstBag {
                    first_bag: *first_bag,
                    placements: witness.placements,
                    final_board: witness.board,
                    kind: CertificationKind::Direct,
                    backtracks: witness.backtracks,
                },
                proof: RecursiveForcedBagProof::Direct { safe_boards },
            };
        }

        let candidates = candidates_fn(board, first_bag);
        let mut representative_failure = FailureExample::leaf(board, *first_bag);

        for candidate in candidates.into_iter() {
            let candidate = *candidate;
            if terminal_fn(&candidate.board) {
                let safe_boards = Arc::from([candidate.board]);
                return FirstBagCertification::Certified {
                    entry: CertifiedFirstBag {
                        first_bag: *first_bag,
                        placements: candidate.placements,
                        final_board: candidate.board,
                        kind: CertificationKind::Direct,
                        backtracks: candidate.backtracks,
                    },
                    proof: RecursiveForcedBagProof::Direct { safe_boards },
                };
            }

            let mut failing_bag = None;
            let mut safe_boards = HashSet::new();
            for second_bag in second_bags {
                let Some(witness) = direct_search(candidate.board, second_bag) else {
                    failing_bag = Some(*second_bag);
                    break;
                };
                safe_boards.insert(witness.board);
            }

            if let Some(failing_second_bag) = failing_bag {
                representative_failure = FailureExample::with_child(
                    board,
                    *first_bag,
                    candidate.placements,
                    candidate.board,
                    FailureExample::leaf(candidate.board, failing_second_bag),
                );
                continue;
            }

            let safe_boards = Arc::from(
                safe_boards
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            return FirstBagCertification::Certified {
                entry: CertifiedFirstBag {
                    first_bag: *first_bag,
                    placements: candidate.placements,
                    final_board: candidate.board,
                    kind: CertificationKind::Recovered,
                    backtracks: candidate.backtracks,
                },
                proof: RecursiveForcedBagProof::Recovered {
                    intermediate_board: candidate.board,
                    child_remaining_depth: 1,
                    safe_boards,
                },
            };
        }

        FirstBagCertification::Failed(representative_failure)
    }

    #[test]
    /// Ensures the permutation iterator covers all 5040 unique bags exactly once.
    fn forced_bag_permutation_iter_yields_all_permutations() {
        let bags = ALL_FORCED_BAG_PERMUTATIONS.to_vec();
        assert_eq!(bags.len(), BAG_PERMUTATION_COUNT);

        let mut unique = std::collections::HashSet::new();
        for bag in bags {
            assert!(unique.insert(bag));
        }
    }

    #[test]
    /// Verifies each generated bag contains exactly the 7 unique tetrominoes.
    fn all_forced_bag_permutations_contain_each_piece_once() {
        let mut canonical = [false; TetrisPiece::NUM_PIECES];
        for piece_idx in 0..TetrisPiece::NUM_PIECES {
            canonical[piece_idx] = true;
        }

        for bag in ALL_FORCED_BAG_PERMUTATIONS {
            let mut seen = [false; TetrisPiece::NUM_PIECES];
            for piece in bag {
                seen[piece.index() as usize] = true;
            }
            assert_eq!(seen, canonical);
        }
    }

    #[test]
    /// Confirms the planner uses the fixed-capacity storage chosen for the allocation-free core.
    fn planner_uses_fixed_capacity_buffers() {
        THREAD_LOCAL_PLANNER.with(|planner| {
            let planner = planner.borrow();
            assert_eq!(
                std::mem::size_of_val(&planner.nodes),
                std::mem::size_of::<HeaplessVec<SearchNode, BEST_FIRST_NODE_CAPACITY>>()
            );
            assert_eq!(planner.nodes.len(), 0);
            assert_eq!(planner.frontier.len(), 0);
        });
        assert!(BEST_FIRST_NODE_CAPACITY >= 21_845);
    }

    #[test]
    /// Smoke test that the best-first planner can produce a full 7-piece script from empty.
    fn best_first_search_returns_full_depth_plan() {
        let board = TetrisBoard::new();
        let sequence = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];

        let outcome =
            search_forced_sequence_core(board, &sequence, None::<&fn(TetrisBoard) -> bool>);
        let SearchOutcome::Success(witness) = outcome else {
            panic!("best-first search should produce a 7-piece script");
        };
        assert_eq!(witness.placements.len(), PIECES_PER_BAG);
    }

    #[test]
    /// Direct safe-set returns should short-circuit before any two-bag recovery logic.
    fn certify_first_bag_prefers_direct_solution() {
        let first_bag = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let direct = test_witness([0, 1, 2, 3, 4, 5, 6], TetrisBoard::new());

        let result = certify_first_bag_with(
            TetrisBoard::new(),
            &first_bag,
            &ALL_FORCED_BAG_PERMUTATIONS,
            |_, bag| {
                if bag == &first_bag {
                    Some(direct)
                } else {
                    None
                }
            },
            |_, _| panic!("candidate enumeration should not be used when direct succeeds"),
        );

        let FirstBagCertification::Certified {
            entry: certified, ..
        } = result
        else {
            panic!("direct witness should certify the first bag");
        };
        assert!(matches!(certified.kind, CertificationKind::Direct));
        assert_eq!(certified.placements, direct.placements);
    }

    #[test]
    /// A single fixed first-bag witness is valid only if every second bag succeeds from its intermediate board.
    fn certify_first_bag_uses_one_fixed_first_bag_for_all_second_bags() {
        let first_bag = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let second_a = [
            TetrisPiece::J_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
        ];
        let second_b = [
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];

        let mut intermediate = TetrisBoard::new();
        intermediate.set_bit(0, 3);
        intermediate.set_bit(1, 2);
        intermediate.set_bit(2, 1);

        let candidate = test_witness([0, 1, 2, 3, 4, 5, 6], intermediate);
        let recovered_terminal = test_witness([6, 5, 4, 3, 2, 1, 0], TetrisBoard::new());

        let result = certify_first_bag_with(
            TetrisBoard::new(),
            &first_bag,
            &[second_a, second_b],
            |board, bag| {
                if board == intermediate && (bag == &second_a || bag == &second_b) {
                    Some(recovered_terminal)
                } else {
                    None
                }
            },
            |_, _| candidate_buffer(&[candidate]),
        );

        let FirstBagCertification::Certified {
            entry: certified,
            proof,
        } = result
        else {
            panic!("candidate should certify when all second bags succeed");
        };
        assert!(matches!(certified.kind, CertificationKind::Recovered));
        assert_eq!(certified.final_board, intermediate);
        let mut safe_boards = proof
            .discovered_safe_boards()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        safe_boards.sort();
        assert_eq!(safe_boards, vec![TetrisBoard::new()]);
    }

    #[test]
    /// A failing second bag must reject the current first-bag candidate and allow later candidates to try.
    fn certify_first_bag_rejects_candidate_when_any_second_bag_fails() {
        let first_bag = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let second_a = [
            TetrisPiece::J_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
        ];
        let second_b = [
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];

        let mut bad_intermediate = TetrisBoard::new();
        bad_intermediate.set_bit(0, 3);
        let mut good_intermediate = TetrisBoard::new();
        good_intermediate.set_bit(1, 3);

        let bad_candidate = test_witness([0, 1, 2, 3, 4, 5, 6], bad_intermediate);
        let good_candidate = test_witness([6, 5, 4, 3, 2, 1, 0], good_intermediate);
        let direct_recovery = test_witness([1, 1, 1, 1, 1, 1, 1], TetrisBoard::new());

        let result = certify_first_bag_with(
            TetrisBoard::new(),
            &first_bag,
            &[second_a, second_b],
            |board, bag| {
                if board == good_intermediate {
                    Some(direct_recovery)
                } else if board == bad_intermediate && bag == &second_a {
                    Some(direct_recovery)
                } else {
                    None
                }
            },
            |_, _| candidate_buffer(&[bad_candidate, good_candidate]),
        );

        let FirstBagCertification::Certified {
            entry: certified, ..
        } = result
        else {
            panic!("second candidate should recover after first candidate is rejected");
        };
        assert_eq!(certified.final_board, good_intermediate);
    }

    #[test]
    /// A cached `Complete` intermediate board should immediately certify future hits on that board.
    fn certify_first_bag_reuses_complete_intermediate_board() {
        let mut intermediate = TetrisBoard::new();
        intermediate.set_bit(0, 3);

        let context = SharedSearchContext::new();
        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board: intermediate,
                remaining_depth: 1,
            },
            RecursiveBoardStatus::Complete {
                safe_boards: Arc::from([TetrisBoard::new()]),
            },
        );

        let result = certify_intermediate_board_cached(&context, intermediate);
        assert_eq!(
            result,
            RecursiveBoardCertification::Complete {
                safe_boards: Arc::from([TetrisBoard::new()]),
            }
        );
        let progress = context.progress_snapshot();
        assert_eq!(progress.recursive_cache_hits, 1);
        assert_eq!(progress.recursive_cache_complete_hits_by_depth[1], 1);
    }

    #[test]
    /// A cached `Impossible` intermediate board should immediately fail future hits on that board.
    fn certify_first_bag_reuses_impossible_intermediate_board() {
        let failing_second_bag = [
            TetrisPiece::J_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::O_PIECE,
        ];
        let mut intermediate = TetrisBoard::new();
        intermediate.set_bit(0, 3);

        let context = SharedSearchContext::new();
        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board: intermediate,
                remaining_depth: 1,
            },
            RecursiveBoardStatus::Impossible {
                failure: FailureExample::leaf(intermediate, failing_second_bag),
            },
        );

        let result = certify_intermediate_board_cached(&context, intermediate);
        assert_eq!(
            result,
            RecursiveBoardCertification::Impossible {
                failure: FailureExample::leaf(intermediate, failing_second_bag)
            }
        );
        let progress = context.progress_snapshot();
        assert_eq!(progress.recursive_cache_hits, 1);
        assert_eq!(progress.recursive_cache_impossible_hits_by_depth[1], 1);
    }

    #[test]
    /// The production recursive base case should be direct-only and ignore deeper proof caches.
    fn production_depth_one_certification_is_direct_only() {
        let context = SharedSearchContext::new();
        let board = board_with_bit(0, SAFE_HEIGHT_CAP as usize + 1);
        let forced_bag = ALL_FORCED_BAG_PERMUTATIONS[0];

        context.direct_cache.insert(
            DirectSearchCacheKey {
                board,
                bag: forced_bag,
            },
            DirectSearchCacheValue::Failure,
        );

        let result = certify_forced_bag_at_depth(&context, board, &forced_bag, 1);
        let FirstBagCertification::Failed(failure) = result else {
            panic!("depth-one certification should fail after a cached direct miss");
        };

        assert_eq!(failure.board, board);
        assert_eq!(failure.forced_bag, forced_bag);
        assert!(failure.chosen_placements.is_none());
        assert!(failure.intermediate_board.is_none());
        assert!(failure.child_failure.is_none());
        assert!(context.recursive_board_cache.is_empty());
        let progress = context.progress_snapshot();
        assert_eq!(progress.forced_bag_failed_by_depth[1], 1);
        assert_eq!(progress.forced_bag_direct_by_depth[1], 0);
        assert_eq!(progress.forced_bag_recovered_by_depth[1], 0);
    }

    #[test]
    /// Recursive board status must be isolated by remaining depth.
    fn recursive_board_cache_isolates_remaining_depth() {
        let context = SharedSearchContext::new();
        let board = board_with_bit(0, SAFE_HEIGHT_CAP as usize + 1);
        let failing_bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let safe_board = TetrisBoard::new();

        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board,
                remaining_depth: 1,
            },
            RecursiveBoardStatus::Impossible {
                failure: FailureExample::leaf(board, failing_bag),
            },
        );
        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board,
                remaining_depth: 2,
            },
            RecursiveBoardStatus::Complete {
                safe_boards: Arc::from([safe_board]),
            },
        );

        assert_eq!(
            certify_board_at_depth(&context, board, 1),
            RecursiveBoardCertification::Impossible {
                failure: FailureExample::leaf(board, failing_bag),
            }
        );
        assert_eq!(
            certify_board_at_depth(&context, board, 2),
            RecursiveBoardCertification::Complete {
                safe_boards: Arc::from([safe_board]),
            }
        );

        let progress = context.progress_snapshot();
        assert_eq!(progress.recursive_cache_hits, 2);
        assert_eq!(progress.recursive_cache_impossible_hits_by_depth[1], 1);
        assert_eq!(progress.recursive_cache_complete_hits_by_depth[2], 1);
    }

    #[test]
    /// Production recovery should surface terminal safe boards from cached child certification.
    fn production_recovered_proof_propagates_child_safe_boards() {
        let context = SharedSearchContext::new();
        let board = board_with_bit(0, SAFE_HEIGHT_CAP as usize + 1);
        let forced_bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let safe_a = TetrisBoard::new();
        let safe_b = board_with_bit(1, 0);

        context.direct_cache.insert(
            DirectSearchCacheKey {
                board,
                bag: forced_bag,
            },
            DirectSearchCacheValue::Failure,
        );

        let candidates = collect_first_bag_candidates_core(
            board,
            &forced_bag,
            recovery_candidate_limit_for_remaining_depth(2),
        );
        let candidate_idx = candidates
            .into_iter()
            .position(|candidate| !terminal_fn(&candidate.board))
            .expect("high test board should produce a non-terminal first-bag candidate");
        let expected_candidate_scripts_examined = candidate_idx + 1;
        let candidate = *candidates
            .into_iter()
            .nth(candidate_idx)
            .expect("candidate index should still be present");

        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board: candidate.board,
                remaining_depth: 1,
            },
            RecursiveBoardStatus::Complete {
                safe_boards: Arc::from([safe_a, safe_b]),
            },
        );

        let result = certify_forced_bag_at_depth(&context, board, &forced_bag, 2);
        let FirstBagCertification::Certified { entry, proof } = result else {
            panic!("cached child completion should recover the forced bag");
        };

        assert_eq!(entry.kind, CertificationKind::Recovered);
        assert_eq!(entry.final_board, candidate.board);
        let RecursiveForcedBagProof::Recovered {
            intermediate_board,
            child_remaining_depth,
            safe_boards,
        } = proof
        else {
            panic!("expected a recovered recursive proof");
        };
        assert_eq!(intermediate_board, candidate.board);
        assert_eq!(child_remaining_depth, 1);
        let mut discovered = safe_boards.iter().copied().collect::<Vec<_>>();
        discovered.sort();
        let mut expected = vec![safe_a, safe_b];
        expected.sort();
        assert_eq!(discovered, expected);

        let progress = context.progress_snapshot();
        assert_eq!(progress.forced_bag_recovered_by_depth[2], 1);
        assert_eq!(
            progress.candidate_scripts_examined_by_depth[2],
            expected_candidate_scripts_examined
        );
        assert_eq!(progress.recursive_cache_complete_hits_by_depth[1], 1);
        assert_eq!(progress.recovered_safe_boards_discovered, 2);
    }

    #[test]
    /// Production recovery should query the child board at exactly one smaller depth.
    fn production_recovery_uses_one_smaller_child_cache() {
        let context = SharedSearchContext::new();
        let board = board_with_bit(0, SAFE_HEIGHT_CAP as usize + 1);
        let forced_bag = ALL_FORCED_BAG_PERMUTATIONS[1];
        let safe_board = board_with_bit(2, 0);
        let parent_depth = RECOVERY_BAG_DEPTH;
        let child_depth = parent_depth - 1;
        let parent_depth_u8 = parent_depth as u8;
        let child_depth_u8 = child_depth as u8;

        context.direct_cache.insert(
            DirectSearchCacheKey {
                board,
                bag: forced_bag,
            },
            DirectSearchCacheValue::Failure,
        );

        let candidates = collect_first_bag_candidates_core(
            board,
            &forced_bag,
            recovery_candidate_limit_for_remaining_depth(parent_depth),
        );
        let candidate_idx = candidates
            .into_iter()
            .position(|candidate| !terminal_fn(&candidate.board))
            .expect("high test board should produce a non-terminal first-bag candidate");
        let expected_candidate_scripts_examined = candidate_idx + 1;
        let candidate = *candidates
            .into_iter()
            .nth(candidate_idx)
            .expect("candidate index should still be present");

        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board: candidate.board,
                remaining_depth: parent_depth_u8,
            },
            RecursiveBoardStatus::Impossible {
                failure: FailureExample::leaf(candidate.board, ALL_FORCED_BAG_PERMUTATIONS[2]),
            },
        );
        context.recursive_board_cache.insert(
            RecursiveBoardCacheKey {
                board: candidate.board,
                remaining_depth: child_depth_u8,
            },
            RecursiveBoardStatus::Complete {
                safe_boards: Arc::from([safe_board]),
            },
        );

        let result = certify_forced_bag_at_depth(&context, board, &forced_bag, parent_depth);
        let FirstBagCertification::Certified { proof, .. } = result else {
            panic!("recovery proof should use the cached child completion");
        };
        let RecursiveForcedBagProof::Recovered {
            intermediate_board,
            child_remaining_depth,
            safe_boards,
        } = proof
        else {
            panic!("expected recovered recursive proof");
        };

        assert_eq!(intermediate_board, candidate.board);
        assert_eq!(usize::from(child_remaining_depth), child_depth);
        assert_eq!(safe_boards.as_ref(), &[safe_board]);

        let progress = context.progress_snapshot();
        assert_eq!(progress.forced_bag_recovered_by_depth[parent_depth], 1);
        assert_eq!(
            progress.candidate_scripts_examined_by_depth[parent_depth],
            expected_candidate_scripts_examined
        );
        assert_eq!(
            progress.recursive_cache_complete_hits_by_depth[child_depth],
            1
        );
        assert_eq!(progress.recovered_safe_boards_discovered, 1);
    }

    #[test]
    /// Recursive board-cache misses should record both the cache miss depth and computed result kind.
    fn recursive_board_cache_metrics_record_misses_and_results_by_depth() {
        let context = SharedSearchContext::new();
        let complete_board = board_with_bit(0, 4);
        let impossible_board = board_with_bit(1, 4);
        let safe_board = TetrisBoard::new();
        let failing_bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let complete_depth = RECOVERY_BAG_DEPTH;
        let impossible_depth = complete_depth - 1;

        assert_eq!(
            certify_board_at_depth_cached_with(&context, complete_board, complete_depth, || {
                RecursiveBoardCertification::Complete {
                    safe_boards: Arc::from([safe_board]),
                }
            }),
            RecursiveBoardCertification::Complete {
                safe_boards: Arc::from([safe_board]),
            }
        );
        assert_eq!(
            certify_board_at_depth_cached_with(
                &context,
                impossible_board,
                impossible_depth,
                || {
                    RecursiveBoardCertification::Impossible {
                        failure: FailureExample::leaf(impossible_board, failing_bag),
                    }
                }
            ),
            RecursiveBoardCertification::Impossible {
                failure: FailureExample::leaf(impossible_board, failing_bag),
            }
        );

        let progress = context.progress_snapshot();
        assert_eq!(progress.recursive_cache_misses, 2);
        assert_eq!(progress.recursive_cache_misses_by_depth[complete_depth], 1);
        assert_eq!(
            progress.recursive_cache_misses_by_depth[impossible_depth],
            1
        );
        assert_eq!(
            progress.recursive_board_complete_by_depth[complete_depth],
            1
        );
        assert_eq!(
            progress.recursive_board_impossible_by_depth[impossible_depth],
            1
        );
    }

    #[test]
    /// Production recovery failures should preserve the real candidate script and child failure.
    fn production_recovery_failure_chain_uses_real_candidates_and_cached_child_failure() {
        let context = SharedSearchContext::new();
        let board = board_with_bit(0, SAFE_HEIGHT_CAP as usize + 8);
        let remaining_depth = 2;
        let child_failure_bag = ALL_FORCED_BAG_PERMUTATIONS[17];
        let mut selected_case = None;

        for forced_bag in ALL_FORCED_BAG_PERMUTATIONS.iter().copied().take(64) {
            let candidates = collect_first_bag_candidates_core(
                board,
                &forced_bag,
                recovery_candidate_limit_for_remaining_depth(remaining_depth),
            )
            .into_iter()
            .map(|candidate| *candidate)
            .collect::<Vec<_>>();
            if !candidates.is_empty()
                && candidates
                    .iter()
                    .all(|candidate| !terminal_fn(&candidate.board))
            {
                selected_case = Some((forced_bag, candidates));
                break;
            }
        }

        let Some((forced_bag, candidates)) = selected_case else {
            panic!("expected to find a non-terminal real-candidate test case");
        };
        let expected_candidate = *candidates
            .last()
            .expect("selected candidate list should not be empty");

        context.direct_cache.insert(
            DirectSearchCacheKey {
                board,
                bag: forced_bag,
            },
            DirectSearchCacheValue::Failure,
        );
        for candidate in &candidates {
            context.recursive_board_cache.insert(
                RecursiveBoardCacheKey {
                    board: candidate.board,
                    remaining_depth: 1,
                },
                RecursiveBoardStatus::Impossible {
                    failure: FailureExample::leaf(candidate.board, child_failure_bag),
                },
            );
        }

        let result = certify_forced_bag_at_depth(&context, board, &forced_bag, remaining_depth);
        let FirstBagCertification::Failed(example) = result else {
            panic!("all cached child failures should reject every real candidate");
        };

        assert_eq!(example.top_level_bag(), forced_bag);
        assert_eq!(
            example.top_level_placements(),
            Some(expected_candidate.placements)
        );
        assert_eq!(
            example.top_level_intermediate_board(),
            Some(expected_candidate.board)
        );
        assert_eq!(example.failing_recovery_bag(), Some(child_failure_bag));

        let child = example
            .child_failure
            .as_deref()
            .expect("top-level failure should retain child failure details");
        assert_eq!(child.board, expected_candidate.board);
        assert_eq!(child.forced_bag, child_failure_bag);

        let progress = context.progress_snapshot();
        assert_eq!(progress.forced_bag_failed_by_depth[remaining_depth], 1);
        assert_eq!(
            progress.candidate_scripts_examined_by_depth[remaining_depth],
            candidates.len()
        );
        assert_eq!(
            progress.recursive_cache_impossible_hits_by_depth[remaining_depth - 1],
            candidates.len()
        );
    }

    #[test]
    /// Recursive failure examples should retain each board and bag in a deeper failure chain.
    fn recursive_failure_chain_retains_all_levels() {
        let root_board = board_with_bit(0, 4);
        let middle_board = board_with_bit(1, 5);
        let leaf_board = board_with_bit(2, 6);
        let root_bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let middle_bag = ALL_FORCED_BAG_PERMUTATIONS[1];
        let leaf_bag = ALL_FORCED_BAG_PERMUTATIONS[2];
        let root_script = script_from_indices([0, 1, 2, 3, 4, 5, 6]);
        let middle_script = script_from_indices([6, 5, 4, 3, 2, 1, 0]);

        let leaf = FailureExample::leaf(leaf_board, leaf_bag);
        let middle =
            FailureExample::with_child(middle_board, middle_bag, middle_script, leaf_board, leaf);
        let root =
            FailureExample::with_child(root_board, root_bag, root_script, middle_board, middle);

        assert_eq!(root.top_level_bag(), root_bag);
        assert_eq!(root.top_level_placements(), Some(root_script));
        assert_eq!(root.top_level_intermediate_board(), Some(middle_board));
        assert_eq!(root.failing_recovery_bag(), Some(leaf_bag));
        assert_eq!(root.leaf_bag(), leaf_bag);

        let middle = root
            .child_failure
            .as_deref()
            .expect("root failure should retain the middle failure");
        assert_eq!(middle.board, middle_board);
        assert_eq!(middle.forced_bag, middle_bag);
        assert_eq!(middle.chosen_placements, Some(middle_script));
        assert_eq!(middle.intermediate_board, Some(leaf_board));

        let leaf = middle
            .child_failure
            .as_deref()
            .expect("middle failure should retain the leaf failure");
        assert_eq!(leaf.board, leaf_board);
        assert_eq!(leaf.forced_bag, leaf_bag);
        assert!(leaf.child_failure.is_none());
    }

    #[test]
    /// The scheduler should claim pending work once and track transitions to solved.
    fn scheduler_claim_lifecycle_and_metrics() {
        let start_board = TetrisBoard::new();
        let scheduler = Scheduler::new(start_board);

        assert_eq!(scheduler.pending_board_count(), 1);
        assert_eq!(scheduler.in_flight_board_count(), 0);
        assert_eq!(scheduler.solved_board_count(), 0);

        assert_eq!(
            scheduler.claim_next_board(),
            ClaimDecision::Claimed(start_board)
        );
        assert_eq!(scheduler.pending_board_count(), 0);
        assert_eq!(scheduler.in_flight_board_count(), 1);
        assert_eq!(scheduler.solved_board_count(), 0);

        assert_eq!(scheduler.claim_next_board(), ClaimDecision::WaitForInFlight);

        scheduler.record_board_solved(start_board);
        assert_eq!(scheduler.pending_board_count(), 0);
        assert_eq!(scheduler.in_flight_board_count(), 0);
        assert_eq!(scheduler.solved_board_count(), 1);
    }

    #[test]
    /// Newly discovered boards should deduplicate across repeated enqueue attempts.
    fn scheduler_enqueue_discovered_board_deduplicates() {
        let start_board = TetrisBoard::new();
        let scheduler = Scheduler::new(start_board);
        let discovered_a = board_with_bit(0, 0);
        let discovered_b = board_with_bit(1, 0);

        assert!(!scheduler.enqueue_discovered_board(start_board));
        assert_eq!(scheduler.pending_board_count(), 1);

        assert!(scheduler.enqueue_discovered_board(discovered_a));
        assert!(!scheduler.enqueue_discovered_board(discovered_a));
        assert!(scheduler.enqueue_discovered_board(discovered_b));
        assert!(!scheduler.enqueue_discovered_board(discovered_b));

        assert_eq!(scheduler.tracked_board_count(), 3);
        assert_eq!(scheduler.pending_board_count(), 3);
    }

    #[test]
    /// First fatal certification failure should win the once-set race.
    fn failure_slot_records_first_failure_only() {
        let failure_slot = OnceLock::new();
        let board_a = board_with_bit(0, 0);
        let board_b = board_with_bit(1, 0);
        let first_failure = FailureExample::with_child(
            board_a,
            ALL_FORCED_BAG_PERMUTATIONS[0],
            script_from_indices([0, 1, 2, 3, 4, 5, 6]),
            board_a,
            FailureExample::leaf(board_a, ALL_FORCED_BAG_PERMUTATIONS[1]),
        );
        let second_failure = FailureExample::with_child(
            board_b,
            ALL_FORCED_BAG_PERMUTATIONS[2],
            script_from_indices([6, 5, 4, 3, 2, 1, 0]),
            board_b,
            FailureExample::leaf(board_b, ALL_FORCED_BAG_PERMUTATIONS[3]),
        );

        let _ = failure_slot.set((board_a, first_failure.clone()));
        let _ = failure_slot.set((board_b, second_failure));

        assert_eq!(failure_slot.get().cloned(), Some((board_a, first_failure)));
    }

    #[test]
    /// Scheduler accounting should maintain tracked = pending + in_flight + solved + failed + abandoned.
    fn scheduler_accounting_invariant_holds() {
        let start_board = TetrisBoard::new();
        let scheduler = Scheduler::new(start_board);
        let discovered_a = board_with_bit(0, 0);
        let discovered_b = board_with_bit(1, 0);

        scheduler.enqueue_discovered_board(discovered_a);
        scheduler.enqueue_discovered_board(discovered_b);

        let ClaimDecision::Claimed(first) = scheduler.claim_next_board() else {
            panic!("expected first claimed board");
        };
        scheduler.record_board_solved(first);

        let ClaimDecision::Claimed(second) = scheduler.claim_next_board() else {
            panic!("expected second claimed board");
        };

        let tracked = scheduler.tracked_board_count();
        let pending = scheduler.pending_board_count();
        let in_flight = scheduler.in_flight_board_count();
        let solved = scheduler.solved_board_count();
        let failed = scheduler.failed_board_count();
        assert_eq!(tracked, pending + in_flight + solved + failed);
        assert_eq!(in_flight, 1);

        scheduler.record_board_failed(second);
        let tracked_after = scheduler.tracked_board_count();
        let pending_after = scheduler.pending_board_count();
        let in_flight_after = scheduler.in_flight_board_count();
        let solved_after = scheduler.solved_board_count();
        let failed_after = scheduler.failed_board_count();
        let abandoned_after = scheduler.abandoned_board_count();
        assert_eq!(
            tracked_after,
            pending_after + in_flight_after + solved_after + failed_after + abandoned_after
        );
        assert_eq!(failed_after, 1);
    }

    #[test]
    /// Abandoned boards should release in-flight accounting without being marked solved or failed.
    fn scheduler_abandoned_board_releases_inflight_slot() {
        let start_board = TetrisBoard::new();
        let scheduler = Scheduler::new(start_board);

        let ClaimDecision::Claimed(board) = scheduler.claim_next_board() else {
            panic!("expected the seeded board to be claimed");
        };
        scheduler.record_board_abandoned(board);

        assert_eq!(scheduler.in_flight_board_count(), 0);
        assert_eq!(scheduler.solved_board_count(), 0);
        assert_eq!(scheduler.failed_board_count(), 0);
        assert_eq!(scheduler.abandoned_board_count(), 1);
        assert_eq!(scheduler.pending_board_count(), 0);
    }

    #[test]
    /// Once all work is drained and nothing is in flight, the scheduler should report exhaustion.
    fn scheduler_reports_exhausted_after_draining_work() {
        let start_board = TetrisBoard::new();
        let scheduler = Scheduler::new(start_board);

        let ClaimDecision::Claimed(board) = scheduler.claim_next_board() else {
            panic!("expected the seeded board to be claimed");
        };
        scheduler.record_board_solved(board);

        assert_eq!(scheduler.claim_next_board(), ClaimDecision::Exhausted);
        assert_eq!(scheduler.pending_board_count(), 0);
        assert_eq!(scheduler.in_flight_board_count(), 0);
        assert_eq!(scheduler.solved_board_count(), 1);
    }

    #[test]
    /// Newly discovered boards must be published before the last in-flight board is released.
    fn scheduler_can_claim_published_work_before_last_inflight_finishes() {
        let start_board = TetrisBoard::new();
        let discovered = board_with_bit(2, 0);
        let scheduler = Scheduler::new(start_board);

        let ClaimDecision::Claimed(board) = scheduler.claim_next_board() else {
            panic!("expected the seeded board to be claimed");
        };
        assert_eq!(board, start_board);
        assert_eq!(scheduler.claim_next_board(), ClaimDecision::WaitForInFlight);

        assert!(scheduler.enqueue_discovered_board(discovered));
        assert_eq!(
            scheduler.claim_next_board(),
            ClaimDecision::Claimed(discovered)
        );

        scheduler.record_board_solved(start_board);
        scheduler.record_board_solved(discovered);
        assert_eq!(scheduler.claim_next_board(), ClaimDecision::Exhausted);
    }

    #[test]
    /// Active worker progress should expose partial first-bag progress and clear when finished.
    fn active_worker_progress_tracks_and_clears_active_board() {
        let active_workers = ActiveWorkerProgress::new(2);
        let board = board_with_bit(0, 0);

        assert!(active_workers.snapshot(Instant::now()).is_empty());
        active_workers.start_board(1, board);
        active_workers.update_board(
            1,
            BoardProcessProgressUpdate {
                board,
                current_first_bag_idx: 17,
                completed_first_bags: 16,
                direct_first_bags: 14,
                recovered_first_bags: 2,
                failed_first_bags: 0,
                unique_safe_boards: 9,
            },
        );

        let snapshot = active_workers.snapshot(Instant::now());
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].worker_idx, 1);
        assert_eq!(snapshot[0].board, board);
        assert_eq!(snapshot[0].current_first_bag_idx, 17);
        assert_eq!(snapshot[0].completed_first_bags, 16);
        assert_eq!(snapshot[0].remaining_first_bags, BAG_PERMUTATION_COUNT - 16);
        assert_eq!(snapshot[0].direct_first_bags, 14);
        assert_eq!(snapshot[0].recovered_first_bags, 2);
        assert_eq!(snapshot[0].unique_safe_boards, 9);

        active_workers.clear_board(1);
        assert!(active_workers.snapshot(Instant::now()).is_empty());
    }

    #[test]
    /// Incremental publishing should emit each active-board safe board at most once.
    fn publish_new_safe_boards_deduplicates_within_active_board() {
        let safe_a = board_with_bit(0, 0);
        let safe_b = board_with_bit(1, 0);
        let mut discovered_safe_boards = HashSet::new();
        let mut published = Vec::new();

        publish_new_safe_boards(
            &mut discovered_safe_boards,
            [safe_a, safe_a, safe_b, safe_a],
            &mut |board| published.push(board),
        );
        publish_new_safe_boards(
            &mut discovered_safe_boards,
            [safe_b, safe_a],
            &mut |board| published.push(board),
        );

        assert_eq!(published, vec![safe_a, safe_b]);
        assert_eq!(discovered_safe_boards.len(), 2);
    }

    #[test]
    /// Board processing should stop before more first-bag work once global cancellation is observed.
    fn process_board_stops_before_work_when_cancelled() {
        let context = SharedSearchContext::new();
        let board = TetrisBoard::new();
        let mut progress_updates = 0usize;
        let mut published_boards = 0usize;

        let result = process_board_with_hooks(
            &context,
            board,
            0,
            || true,
            |_| progress_updates += 1,
            |_| published_boards += 1,
        );

        assert_eq!(result.board, board);
        assert!(result.cancelled);
        assert!(result.failure.is_none());
        assert_eq!(result.processed_first_bags, 0);
        assert_eq!(result.direct_first_bags, 0);
        assert_eq!(result.recovered_first_bags, 0);
        assert_eq!(result.failed_first_bags, 0);
        assert_eq!(progress_updates, 0);
        assert_eq!(published_boards, 0);
    }

    #[test]
    /// Top-level summary metrics should be derived from the configured recovery depth only.
    fn top_level_forced_bag_summary_uses_configured_recovery_depth() {
        let context = SharedSearchContext::new();

        context.record_forced_bag_direct(RECOVERY_BAG_DEPTH);
        context.record_forced_bag_recovered(RECOVERY_BAG_DEPTH, 3);
        context.record_forced_bag_failed(RECOVERY_BAG_DEPTH);
        context.record_forced_bag_direct(RECOVERY_BAG_DEPTH - 1);

        let progress = context.progress_snapshot();
        assert_eq!(top_level_forced_bags_completed(&progress), 3);
        assert_eq!(progress.recovered_safe_boards_discovered, 3);
    }

    #[test]
    /// Sample collector should keep only the globally strongest bounded witness set.
    fn sample_collector_keeps_bounded_best_entries() {
        let collector = SampleCollector::new(2);
        let board = TetrisBoard::new();

        collector.merge_entries(vec![
            CertifiedFirstBag {
                first_bag: ALL_FORCED_BAG_PERMUTATIONS[0],
                placements: script_from_indices([0, 1, 2, 3, 4, 5, 6]),
                final_board: board,
                kind: CertificationKind::Direct,
                backtracks: 1,
            },
            CertifiedFirstBag {
                first_bag: ALL_FORCED_BAG_PERMUTATIONS[1],
                placements: script_from_indices([6, 5, 4, 3, 2, 1, 0]),
                final_board: board,
                kind: CertificationKind::Recovered,
                backtracks: 5,
            },
            CertifiedFirstBag {
                first_bag: ALL_FORCED_BAG_PERMUTATIONS[2],
                placements: script_from_indices([1, 2, 3, 4, 5, 6, 0]),
                final_board: board,
                kind: CertificationKind::Direct,
                backtracks: 3,
            },
        ]);

        let samples = collector.snapshot();
        assert_eq!(samples.len(), 2);
        assert!(samples[0].backtracks >= samples[1].backtracks);
        assert_eq!(samples[0].backtracks, 5);
        assert_eq!(samples[1].backtracks, 3);
    }

    #[test]
    /// Workers should exit immediately when cancellation is already requested.
    fn worker_exits_immediately_when_cancel_requested() {
        let shared = Arc::new(SharedRuntimeContext::new(TetrisBoard::new(), 4));
        shared.cancel_requested.store(true, AtomicOrdering::Release);

        let worker_shared = Arc::clone(&shared);
        let handle = thread::spawn(move || run_worker(0, &worker_shared));
        handle
            .join()
            .expect("worker should exit without panic when cancelled");

        assert_eq!(
            shared
                .metrics
                .processed_boards
                .load(AtomicOrdering::Relaxed),
            0
        );
        assert_eq!(shared.scheduler.solved_board_count(), 0);
    }

    #[test]
    #[ignore = "targeted repro for a discovered frontier counterexample"]
    /// Reproduces the reported safe-set frontier failure for one specific discovered board.
    fn reported_frontier_failure_reproduces() {
        let context = SharedSearchContext::new();
        let board = board_from_limbs([3, 3, 2, 2, 3, 3, 0, 3, 3, 3]);
        let first_bag = [
            TetrisPiece::S_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::J_PIECE,
            TetrisPiece::O_PIECE,
        ];
        let expected_intermediate = board_from_limbs([5, 3, 7, 6, 7, 7, 2, 7, 7, 5]);
        let expected_failing_second_bag = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let expected_placements = "TetrisPiecePlacement(piece: S, orientation: TetrisPieceOrientation(rotation: 1, column: Column(0))) | TetrisPiecePlacement(piece: T, orientation: TetrisPieceOrientation(rotation: 1, column: Column(5))) | TetrisPiecePlacement(piece: L, orientation: TetrisPieceOrientation(rotation: 1, column: Column(7))) | TetrisPiecePlacement(piece: Z, orientation: TetrisPieceOrientation(rotation: 1, column: Column(4))) | TetrisPiecePlacement(piece: I, orientation: TetrisPieceOrientation(rotation: 1, column: Column(2))) | TetrisPiecePlacement(piece: J, orientation: TetrisPieceOrientation(rotation: 1, column: Column(3))) | TetrisPiecePlacement(piece: O, orientation: TetrisPieceOrientation(rotation: 0, column: Column(8)))";

        let result = certify_first_bag(&context, board, &first_bag);
        let FirstBagCertification::Failed(example) = result else {
            panic!("expected the reported board/first-bag pair to fail certification");
        };

        assert_eq!(example.top_level_bag(), first_bag);
        assert_eq!(
            example.top_level_intermediate_board(),
            Some(expected_intermediate)
        );
        assert_eq!(
            example.failing_recovery_bag(),
            Some(expected_failing_second_bag)
        );
        let placement_string = placement_script_string(
            example
                .top_level_placements()
                .expect("reported failure should include representative placements"),
        );
        assert_eq!(placement_string, expected_placements);
    }

    #[test]
    #[ignore = "expensive full first-bag sweep for a known frontier counterexample board"]
    /// Sweeps all 5040 first bags from the reported frontier board and fails on the first uncertified bag.
    fn reported_frontier_board_certifies_all_first_bags() {
        let board = board_from_limbs([3, 3, 2, 2, 3, 3, 0, 3, 3, 3]);
        assert_board_certifies_all_first_bags(
            "reported_frontier_board_certifies_all_first_bags",
            board,
        );
    }

    #[test]
    #[ignore = "expensive full first-bag sweep for a later discovered frontier counterexample board"]
    /// Sweeps the ZJITSL0 frontier counterexample board across all first bags.
    fn reported_frontier_counterexample_zjit_sl_o_board_certifies_all_first_bags() {
        let board = board_from_limbs([3, 0, 3, 3, 3, 2, 3, 3, 3, 1]);
        assert_board_certifies_all_first_bags(
            "reported_frontier_counterexample_zjit_sl_o_board_certifies_all_first_bags",
            board,
        );
    }

    #[test]
    #[ignore = "expensive full first-bag sweep for a later discovered frontier counterexample board"]
    /// Sweeps the OSTLZJI frontier counterexample board across all first bags.
    fn reported_frontier_counterexample_ostlzji_board_certifies_all_first_bags() {
        let board = board_from_limbs([1, 2, 3, 3, 3, 3, 2, 3, 3, 2]);
        assert_board_certifies_all_first_bags(
            "reported_frontier_counterexample_ostlzji_board_certifies_all_first_bags",
            board,
        );
    }

    #[test]
    #[ignore = "expensive full first-bag sweep for a known frontier counterexample board"]
    /// Sweeps all 5040 first bags from the reported frontier board and expects the known failure.
    fn reported_frontier_board_certification_matches_runtime_failure() {
        let context = SharedSearchContext::new();
        let board = board_from_limbs([3, 3, 2, 2, 3, 3, 0, 3, 3, 3]);
        let expected_failing_first_bag = [
            TetrisPiece::S_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::J_PIECE,
            TetrisPiece::O_PIECE,
        ];
        let expected_intermediate = board_from_limbs([5, 3, 7, 6, 7, 7, 2, 7, 7, 5]);
        let expected_failing_second_bag = [
            TetrisPiece::O_PIECE,
            TetrisPiece::I_PIECE,
            TetrisPiece::S_PIECE,
            TetrisPiece::Z_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::J_PIECE,
        ];
        let expected_placements = "TetrisPiecePlacement(piece: S, orientation: TetrisPieceOrientation(rotation: 1, column: Column(0))) | TetrisPiecePlacement(piece: T, orientation: TetrisPieceOrientation(rotation: 1, column: Column(5))) | TetrisPiecePlacement(piece: L, orientation: TetrisPieceOrientation(rotation: 1, column: Column(7))) | TetrisPiecePlacement(piece: Z, orientation: TetrisPieceOrientation(rotation: 1, column: Column(4))) | TetrisPiecePlacement(piece: I, orientation: TetrisPieceOrientation(rotation: 1, column: Column(2))) | TetrisPiecePlacement(piece: J, orientation: TetrisPieceOrientation(rotation: 1, column: Column(3))) | TetrisPiecePlacement(piece: O, orientation: TetrisPieceOrientation(rotation: 0, column: Column(8)))";

        let mut processed = 0usize;
        for (bag_idx, first_bag) in ALL_FORCED_BAG_PERMUTATIONS.iter().copied().enumerate() {
            let result = certify_first_bag(&context, board, &first_bag);
            processed += 1;
            match result {
                FirstBagCertification::Certified { .. } => {
                    eprintln!(
                        "[reported_frontier_board_certification_matches_runtime_failure] {}/{} first_bag={:?} certified",
                        bag_idx + 1,
                        BAG_PERMUTATION_COUNT,
                        first_bag
                    );
                }
                FirstBagCertification::Failed(example) => {
                    eprintln!(
                        "[reported_frontier_board_certification_matches_runtime_failure] {}/{} first_bag={:?} failed",
                        bag_idx + 1,
                        BAG_PERMUTATION_COUNT,
                        first_bag
                    );
                    assert_eq!(example.top_level_bag(), expected_failing_first_bag);
                    assert_eq!(
                        example.top_level_intermediate_board(),
                        Some(expected_intermediate)
                    );
                    assert_eq!(
                        example.failing_recovery_bag(),
                        Some(expected_failing_second_bag)
                    );
                    let placement_string = placement_script_string(
                        example
                            .top_level_placements()
                            .expect("reported failure should include representative placements"),
                    );
                    assert_eq!(placement_string, expected_placements);
                    eprintln!(
                        "[reported_frontier_board_certification_matches_runtime_failure] processed_first_bags_before_failure={processed}"
                    );
                    return;
                }
            }
        }

        panic!("expected the reported frontier board to fail certification for some first bag");
    }

    #[test]
    /// Concurrent direct-cache lookups should execute the underlying search only once.
    fn direct_cache_deduplicates_in_progress_searches() {
        let context = Arc::new(SharedSearchContext::new());
        let board = TetrisBoard::new();
        let bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let witness = test_witness([0, 1, 2, 3, 4, 5, 6], TetrisBoard::new());
        let calls = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for _ in 0..2 {
                let context = Arc::clone(&context);
                let calls = Arc::clone(&calls);
                scope.spawn(move || {
                    let result = query_direct_search_cached_with(&context, board, &bag, || {
                        calls.fetch_add(1, AtomicOrdering::Relaxed);
                        thread::sleep(std::time::Duration::from_millis(10));
                        Some(witness)
                    });
                    assert_eq!(result, Some(witness));
                });
            }
        });

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    /// Concurrent direct-cache failures should also be computed only once and shared across waiters.
    fn direct_cache_deduplicates_in_progress_failures() {
        let context = Arc::new(SharedSearchContext::new());
        let board = TetrisBoard::new();
        let bag = ALL_FORCED_BAG_PERMUTATIONS[3];
        let calls = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for _ in 0..2 {
                let context = Arc::clone(&context);
                let calls = Arc::clone(&calls);
                scope.spawn(move || {
                    let result = query_direct_search_cached_with(&context, board, &bag, || {
                        calls.fetch_add(1, AtomicOrdering::Relaxed);
                        thread::sleep(std::time::Duration::from_millis(10));
                        None
                    });
                    assert_eq!(result, None);
                });
            }
        });

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    /// Direct-cache wait metrics should increment when a second worker joins in-progress work.
    fn direct_cache_wait_metrics_record_in_progress_waiter() {
        let context = Arc::new(SharedSearchContext::new());
        let board = TetrisBoard::new();
        let bag = ALL_FORCED_BAG_PERMUTATIONS[0];
        let witness = test_witness([0, 1, 2, 3, 4, 5, 6], TetrisBoard::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        thread::scope(|scope| {
            let owner_context = Arc::clone(&context);
            let owner_calls = Arc::clone(&calls);
            scope.spawn(move || {
                let result = query_direct_search_cached_with(&owner_context, board, &bag, || {
                    owner_calls.fetch_add(1, AtomicOrdering::Relaxed);
                    started_tx
                        .send(())
                        .expect("owner should signal after claiming direct cache work");
                    release_rx
                        .recv()
                        .expect("test should release the direct cache owner");
                    Some(witness)
                });
                assert_eq!(result, Some(witness));
            });

            started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("direct cache owner should start");

            let waiter_context = Arc::clone(&context);
            scope.spawn(move || {
                let result = query_direct_search_cached_with(&waiter_context, board, &bag, || {
                    panic!("waiter should not compute an already in-progress direct cache key");
                });
                assert_eq!(result, Some(witness));
            });

            let observed_wait = wait_until(|| context.progress_snapshot().direct_cache_waits == 1);
            release_tx
                .send(())
                .expect("direct cache owner should still be waiting");
            assert!(
                observed_wait,
                "timed out waiting for direct-cache wait metric"
            );
        });

        let progress = context.progress_snapshot();
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(progress.direct_cache_misses, 1);
        assert_eq!(progress.direct_cache_waits, 1);
        assert_eq!(progress.direct_cache_hits, 1);
    }

    #[test]
    /// Concurrent intermediate sweeps should execute the underlying certification only once.
    fn intermediate_cache_deduplicates_in_progress_sweeps() {
        let context = Arc::new(SharedSearchContext::new());
        let board = board_with_bit(0, 3);
        let safe_boards = Arc::from([TetrisBoard::new()]);
        let calls = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for _ in 0..2 {
                let context = Arc::clone(&context);
                let calls = Arc::clone(&calls);
                let safe_boards = Arc::clone(&safe_boards);
                scope.spawn(move || {
                    let result = certify_intermediate_board_cached_with(&context, board, || {
                        calls.fetch_add(1, AtomicOrdering::Relaxed);
                        thread::sleep(std::time::Duration::from_millis(10));
                        RecursiveBoardCertification::Complete {
                            safe_boards: Arc::clone(&safe_boards),
                        }
                    });
                    assert_eq!(
                        result,
                        RecursiveBoardCertification::Complete {
                            safe_boards: Arc::from([TetrisBoard::new()])
                        }
                    );
                });
            }
        });

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    /// Concurrent impossible intermediate sweeps should also be computed once and shared.
    fn intermediate_cache_deduplicates_in_progress_failures() {
        let context = Arc::new(SharedSearchContext::new());
        let board = board_with_bit(1, 4);
        let calls = Arc::new(AtomicUsize::new(0));
        let failing_second_bag = ALL_FORCED_BAG_PERMUTATIONS[4];

        thread::scope(|scope| {
            for _ in 0..2 {
                let context = Arc::clone(&context);
                let calls = Arc::clone(&calls);
                scope.spawn(move || {
                    let result = certify_intermediate_board_cached_with(&context, board, || {
                        calls.fetch_add(1, AtomicOrdering::Relaxed);
                        thread::sleep(std::time::Duration::from_millis(10));
                        RecursiveBoardCertification::Impossible {
                            failure: FailureExample::leaf(board, failing_second_bag),
                        }
                    });
                    assert_eq!(
                        result,
                        RecursiveBoardCertification::Impossible {
                            failure: FailureExample::leaf(board, failing_second_bag)
                        }
                    );
                });
            }
        });

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    /// Recursive-cache wait metrics should be attributed to the blocked key's remaining depth.
    fn recursive_board_cache_wait_metrics_record_in_progress_depth() {
        let context = Arc::new(SharedSearchContext::new());
        let board = board_with_bit(3, 4);
        let safe_boards = Arc::from([TetrisBoard::new()]);
        let calls = Arc::new(AtomicUsize::new(0));
        let remaining_depth = RECOVERY_BAG_DEPTH;
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        thread::scope(|scope| {
            let owner_context = Arc::clone(&context);
            let owner_calls = Arc::clone(&calls);
            let owner_safe_boards = Arc::clone(&safe_boards);
            scope.spawn(move || {
                let result = certify_board_at_depth_cached_with(
                    &owner_context,
                    board,
                    remaining_depth,
                    || {
                        owner_calls.fetch_add(1, AtomicOrdering::Relaxed);
                        started_tx
                            .send(())
                            .expect("owner should signal after claiming recursive cache work");
                        release_rx
                            .recv()
                            .expect("test should release the recursive cache owner");
                        RecursiveBoardCertification::Complete {
                            safe_boards: Arc::clone(&owner_safe_boards),
                        }
                    },
                );
                assert_eq!(
                    result,
                    RecursiveBoardCertification::Complete {
                        safe_boards: Arc::from([TetrisBoard::new()])
                    }
                );
            });

            started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("recursive cache owner should start");

            let waiter_context = Arc::clone(&context);
            scope.spawn(move || {
                let result = certify_board_at_depth_cached_with(
                    &waiter_context,
                    board,
                    remaining_depth,
                    || {
                        panic!(
                            "waiter should not compute an already in-progress recursive cache key"
                        );
                    },
                );
                assert_eq!(
                    result,
                    RecursiveBoardCertification::Complete {
                        safe_boards: Arc::from([TetrisBoard::new()])
                    }
                );
            });

            let observed_wait = wait_until(|| {
                context.progress_snapshot().recursive_cache_waits_by_depth[remaining_depth] == 1
            });
            release_tx
                .send(())
                .expect("recursive cache owner should still be waiting");
            assert!(
                observed_wait,
                "timed out waiting for recursive-cache wait metric"
            );
        });

        let progress = context.progress_snapshot();
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(progress.recursive_cache_misses, 1);
        assert_eq!(progress.recursive_cache_waits, 1);
        assert_eq!(progress.recursive_cache_waits_by_depth[remaining_depth], 1);
        assert_eq!(progress.recursive_cache_hits, 1);
        assert_eq!(
            progress.recursive_cache_complete_hits_by_depth[remaining_depth],
            1
        );
        assert_eq!(
            progress.recursive_board_complete_by_depth[remaining_depth],
            1
        );
    }

    #[test]
    /// Recursive board-cache ownership should deduplicate in-progress work at non-base depths.
    fn recursive_board_cache_deduplicates_in_progress_max_depth_work() {
        let context = Arc::new(SharedSearchContext::new());
        let board = board_with_bit(2, 4);
        let safe_boards = Arc::from([TetrisBoard::new()]);
        let calls = Arc::new(AtomicUsize::new(0));
        let remaining_depth = RECOVERY_BAG_DEPTH;

        thread::scope(|scope| {
            for _ in 0..2 {
                let context = Arc::clone(&context);
                let calls = Arc::clone(&calls);
                let safe_boards = Arc::clone(&safe_boards);
                scope.spawn(move || {
                    let result = certify_board_at_depth_cached_with(
                        &context,
                        board,
                        remaining_depth,
                        || {
                            calls.fetch_add(1, AtomicOrdering::Relaxed);
                            thread::sleep(std::time::Duration::from_millis(10));
                            RecursiveBoardCertification::Complete {
                                safe_boards: Arc::clone(&safe_boards),
                            }
                        },
                    );
                    assert_eq!(
                        result,
                        RecursiveBoardCertification::Complete {
                            safe_boards: Arc::from([TetrisBoard::new()])
                        }
                    );
                });
            }
        });

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    #[ignore = "expensive end-to-end certification over all 5040 first bags from the empty board"]
    /// End-to-end regression that checks empty-board certification across all first bags and reports frontier growth.
    fn empty_board_certifies_all_first_bags() {
        let start_board = TetrisBoard::new();
        let context = SharedSearchContext::new();
        let mut discovered_intermediate_boards = HashSet::new();

        for (bag_idx, first_bag) in ALL_FORCED_BAG_PERMUTATIONS.iter().copied().enumerate() {
            if search_safe_set_script_cached(&context, start_board, &first_bag).is_none() {
                let candidates = collect_first_bag_candidates_core(
                    start_board,
                    &first_bag,
                    recovery_candidate_limit_for_remaining_depth(RECOVERY_BAG_DEPTH),
                );
                for candidate in candidates.into_iter() {
                    let candidate = *candidate;
                    if !terminal_fn(&candidate.board) {
                        discovered_intermediate_boards.insert(candidate.board);
                    }
                }
            }

            let result = certify_first_bag(&context, start_board, &first_bag);
            if let FirstBagCertification::Failed(example) = result {
                panic!(
                    "empty-board certification failed for first bag {:?} intermediate={:?} recovery_bag={:?}",
                    example.top_level_bag(),
                    example
                        .top_level_intermediate_board()
                        .map(|board| format!("{board:?}"))
                        .unwrap_or_else(|| "<none>".to_string()),
                    example
                        .failing_recovery_bag()
                        .map(|bag| format!("{bag:?}"))
                        .unwrap_or_else(|| "<none>".to_string())
                );
            } else {
                eprintln!(
                    "[empty_board_certifies_all_first_bags] {}/{} first_bag={:?}",
                    bag_idx + 1,
                    BAG_PERMUTATION_COUNT,
                    first_bag
                );
            }
        }

        eprintln!(
            "[empty_board_certifies_all_first_bags] discovered_intermediate_boards={}",
            discovered_intermediate_boards.len()
        );

        let mut discovered_safe_set_boards = HashSet::new();
        for entry in context.direct_cache.iter() {
            if let DirectSearchCacheValue::Success(witness) = entry.value() {
                if terminal_fn(&witness.board) {
                    discovered_safe_set_boards.insert(witness.board);
                }
            }
        }
        let new_safe_set_board_count = discovered_safe_set_boards
            .iter()
            .filter(|&&board| board != start_board)
            .count();
        eprintln!(
            "[empty_board_certifies_all_first_bags] discovered_safe_set_boards_total={} new_safe_set_boards={}",
            discovered_safe_set_boards.len(),
            new_safe_set_board_count
        );
    }
}
