use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::{BoardAdmissibility, SolverConfig};
use crate::graph::{EdgeRange, FlatEdge, PredecessorRef, StateIndex};
use crate::state::{
    BoardId, PackedPlacement, StateId, StateKey, StateKeyPacked, pack_placement, piece_branches,
};

/// A precomputed board-level successor: the resulting board and one representative placement.
#[derive(Clone, Copy)]
struct BoardSuccEntry {
    board_id: BoardId,
    placement: PackedPlacement,
}

/// Wrapper for concurrent writes to disjoint indices of a slice.
///
/// SAFETY: The caller must ensure no two threads write to the same index.
struct UnsafeSlice<T> {
    ptr: *mut T,
    #[cfg(debug_assertions)]
    len: usize,
}

unsafe impl<T: Send> Send for UnsafeSlice<T> {}
unsafe impl<T: Send> Sync for UnsafeSlice<T> {}

impl<T> UnsafeSlice<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            #[cfg(debug_assertions)]
            len: slice.len(),
        }
    }

    /// Write a value at `index`.
    ///
    /// # Safety
    /// - `index` must be in bounds.
    /// - No other thread may write to the same `index` concurrently.
    #[inline]
    unsafe fn write(&self, index: usize, value: T) {
        #[cfg(debug_assertions)]
        debug_assert!(
            index < self.len,
            "UnsafeSlice::write out of bounds: {} >= {}",
            index,
            self.len
        );
        unsafe {
            self.ptr.add(index).write(value);
        }
    }
}

/// The complete state graph for a full 7-bag adversarial Tetris game.
///
/// Built via forward BFS from `(empty_board, full_bag)`, then augmented with
/// predecessor links for backward propagation.
pub struct Universe {
    pub config: SolverConfig,
    // --- Board interning ---
    pub boards: Vec<TetrisBoard>,
    pub board_to_id: FxHashMap<TetrisBoard, BoardId>,
    // --- Board-level precomputed successors (with edge dedup) ---
    board_succ_ranges: Vec<[EdgeRange; 7]>,
    board_succs: Vec<BoardSuccEntry>,
    // --- State interning ---
    pub states: Vec<StateKey>,
    pub state_to_id: FxHashMap<StateKeyPacked, StateId>,
    // --- Forward edges ---
    pub state_indices: Vec<StateIndex>,
    pub edges: Vec<FlatEdge>,
    // --- Backward edges ---
    pub pred_ranges: Vec<EdgeRange>,
    pub predecessors: Vec<PredecessorRef>,
    // --- Notable states ---
    pub root_state_id: StateId,
}

impl Universe {
    /// Build the universe via two-phase BFS:
    /// 1. **Board BFS**: Discover all reachable boards, precompute per-(board, piece)
    ///    successors with edge deduplication (one entry per unique successor board).
    /// 2. **State BFS** (parallel): Walk (board, bag) states using precomputed successors.
    ///    Uses DashMap for concurrent state interning and atomic edge allocation for
    ///    lock-free parallel writes to the edge array.
    pub fn build(config: &SolverConfig) -> Self {
        let total_start = Instant::now();

        let estimated_boards = 100_000;
        let estimated_states = estimated_boards * 50;

        let mut boards: Vec<TetrisBoard> = Vec::with_capacity(estimated_boards);
        let mut board_to_id: FxHashMap<TetrisBoard, BoardId> =
            FxHashMap::with_capacity_and_hasher(estimated_boards, Default::default());

        // Board-level precomputed successors (flat storage).
        let mut board_succ_ranges: Vec<[EdgeRange; 7]> = Vec::with_capacity(estimated_boards);
        let mut board_succs: Vec<BoardSuccEntry> = Vec::new();

        // ================================================================
        // Phase 1: Board BFS — discover boards, precompute successors
        // ================================================================
        let board_start = Instant::now();

        let empty_board = TetrisBoard::EMPTY_BOARD;
        let empty_board_id = intern_board(&mut boards, &mut board_to_id, &empty_board);
        board_succ_ranges.push([EdgeRange::EMPTY; 7]);

        let mut board_frontier: Vec<BoardId> = vec![empty_board_id];
        let mut board_frontier_idx: usize = 0;
        let mut last_report = Instant::now();

        while board_frontier_idx < board_frontier.len() {
            let bid = board_frontier[board_frontier_idx];
            board_frontier_idx += 1;

            let board = boards[bid as usize];
            let mut ranges = [EdgeRange::EMPTY; 7];

            for piece in TetrisPiece::all() {
                let pidx = piece.index() as usize;
                let succ_start = board_succs.len() as u32;

                for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                    let mut new_board = board;
                    let result = new_board.apply_piece_placement(placement);

                    if result.is_lost.into() {
                        continue;
                    }
                    if !board_is_admissible(&new_board, &config.admissibility) {
                        continue;
                    }

                    let prev_board_count = boards.len();
                    let new_board_id = intern_board(&mut boards, &mut board_to_id, &new_board);

                    // New board? Extend successor storage and add to frontier.
                    if boards.len() > prev_board_count {
                        board_succ_ranges.push([EdgeRange::EMPTY; 7]);
                        board_frontier.push(new_board_id);
                    }

                    // Edge dedup: skip if this successor board is already recorded
                    // for this (board, piece) pair. Linear scan is fine — at most
                    // ~34 entries per piece.
                    let already = board_succs[succ_start as usize..]
                        .iter()
                        .any(|e| e.board_id == new_board_id);
                    if already {
                        continue;
                    }

                    board_succs.push(BoardSuccEntry {
                        board_id: new_board_id,
                        placement: pack_placement(placement),
                    });
                }

                let succ_len = board_succs.len() as u32 - succ_start;
                ranges[pidx] = EdgeRange {
                    start: succ_start,
                    len: succ_len,
                };
            }

            board_succ_ranges[bid as usize] = ranges;

            if last_report.elapsed().as_secs() >= 2 {
                let elapsed = board_start.elapsed().as_secs_f64();
                let rate = board_frontier_idx as f64 / elapsed;
                eprintln!(
                    "[board-bfs] {:.1}s | boards={} succs={} frontier={} ({:.0} boards/s)",
                    elapsed,
                    boards.len(),
                    board_succs.len(),
                    board_frontier.len() - board_frontier_idx,
                    rate,
                );
                last_report = Instant::now();
            }
        }

        let board_time = board_start.elapsed().as_secs_f64();
        eprintln!(
            "[board-bfs] done in {:.2}s | boards={} board_succs={}",
            board_time,
            boards.len(),
            board_succs.len(),
        );

        // ================================================================
        // Phase 2: Parallel State BFS — walk (board, bag) states using
        // precomputed board successors, DashMap for concurrent interning,
        // and atomic edge allocation for lock-free parallel writes.
        // ================================================================
        let state_start = Instant::now();

        let state_to_id_par: DashMap<StateKeyPacked, StateId, BuildHasherDefault<FxHasher>> =
            DashMap::with_capacity_and_hasher(estimated_states, BuildHasherDefault::default());
        let next_state_id = AtomicU32::new(0);

        let mut states: Vec<StateKey> = Vec::with_capacity(estimated_states);
        let mut state_indices: Vec<StateIndex> = Vec::with_capacity(estimated_states);
        let mut edges: Vec<FlatEdge> = Vec::with_capacity(estimated_states * 3);

        // Intern root state
        let root_key = StateKey::new(empty_board_id, config.root_bag);
        let root_state_id = next_state_id.fetch_add(1, Ordering::Relaxed);
        state_to_id_par.insert(root_key.pack(), root_state_id);
        states.push(root_key);
        state_indices.push(StateIndex::default());

        /// Maximum states to expand per parallel batch. Smaller batches give more
        /// frequent progress reports; larger batches amortize per-batch overhead.
        const BATCH_CAP: usize = 1_000_000;

        let mut frontier: Vec<StateId> = vec![root_state_id];
        let mut frontier_idx: usize = 0;

        while frontier_idx < frontier.len() {
            let current_count = next_state_id.load(Ordering::Relaxed) as usize;
            if current_count >= config.max_states {
                eprintln!(
                    "[state-bfs] hit max_states cap ({}) -- stopping expansion",
                    config.max_states,
                );
                break;
            }

            // Take up to BATCH_CAP unprocessed states
            let batch_end = (frontier_idx + BATCH_CAP).min(frontier.len());
            let batch: Vec<StateId> = frontier[frontier_idx..batch_end].to_vec();
            frontier_idx = batch_end;

            // Step A: Compute per-state edge counts (parallel, read-only)
            let edge_counts: Vec<u32> = batch
                .par_iter()
                .map(|&sid| {
                    let key = &states[sid as usize];
                    piece_branches(key.bag)
                        .map(|branch| {
                            board_succ_ranges[key.board_id as usize][branch.piece.index() as usize]
                                .len
                        })
                        .sum()
                })
                .collect();

            // Step B: Prefix sum for edge start offsets (sequential)
            let edge_base = edges.len() as u32;
            let mut edge_starts: Vec<u32> = Vec::with_capacity(batch.len());
            let mut offset = edge_base;
            for &count in &edge_counts {
                edge_starts.push(offset);
                offset += count;
            }
            let total_new_edges = (offset - edge_base) as usize;

            // Step C: Pre-allocate edge slots
            edges.resize(edges.len() + total_new_edges, FlatEdge::default());

            // Step D: Parallel expand — write edges and state_indices, intern new states
            let new_states_per_thread: Vec<Vec<(StateKey, StateId)>> = {
                let edges_unsafe = UnsafeSlice::new(&mut edges);
                let indices_unsafe = UnsafeSlice::new(&mut state_indices);

                batch
                    .par_iter()
                    .enumerate()
                    .map(|(i, &sid)| {
                        let key = states[sid as usize];
                        let mut local_new: Vec<(StateKey, StateId)> = Vec::new();
                        let mut write_pos = edge_starts[i] as usize;
                        let mut index = StateIndex {
                            bag: key.bag,
                            piece_ranges: [EdgeRange::EMPTY; 7],
                        };

                        for branch in piece_branches(key.bag) {
                            let pidx = branch.piece.index() as usize;
                            let range_start = write_pos as u32;

                            let board_range = board_succ_ranges[key.board_id as usize][pidx];
                            for idx in 0..board_range.len {
                                let bse = board_succs[(board_range.start + idx) as usize];

                                let succ_key = StateKey::new(bse.board_id, branch.next_bag);
                                let packed = succ_key.pack();

                                let succ_id = match state_to_id_par.entry(packed) {
                                    dashmap::Entry::Occupied(e) => *e.get(),
                                    dashmap::Entry::Vacant(e) => {
                                        let id = next_state_id.fetch_add(1, Ordering::Relaxed);
                                        e.insert(id);
                                        local_new.push((succ_key, id));
                                        id
                                    }
                                };

                                // SAFETY: Each thread writes to its own pre-allocated slice
                                // (disjoint ranges computed via prefix sum in Step B).
                                unsafe {
                                    edges_unsafe.write(
                                        write_pos,
                                        FlatEdge {
                                            succ: succ_id,
                                            placement: bse.placement,
                                        },
                                    );
                                }
                                write_pos += 1;
                            }

                            let range_len = write_pos as u32 - range_start;
                            index.piece_ranges[pidx] = EdgeRange {
                                start: range_start,
                                len: range_len,
                            };
                        }

                        // SAFETY: Each batch entry has a unique sid, so writes are disjoint.
                        unsafe {
                            indices_unsafe.write(sid as usize, index);
                        }

                        local_new
                    })
                    .collect()
            }; // UnsafeSlice borrows end here

            // Step E: Register new states, extend frontier (sequential)
            let new_total = next_state_id.load(Ordering::Relaxed) as usize;
            let default_key = StateKey::new(0, TetrisPieceBagState::from(0u8));
            states.resize(new_total, default_key);
            state_indices.resize(new_total, StateIndex::default());

            for batch_new in new_states_per_thread {
                for (key, id) in batch_new {
                    states[id as usize] = key;
                    frontier.push(id);
                }
            }

            if last_report.elapsed().as_secs() >= 2 {
                let elapsed = state_start.elapsed().as_secs_f64();
                let rate = frontier_idx as f64 / elapsed;
                eprintln!(
                    "[state-bfs] {:.1}s | states={} edges={} frontier={} ({:.0} states/s)",
                    elapsed,
                    states.len(),
                    edges.len(),
                    frontier.len() - frontier_idx,
                    rate,
                );
                last_report = Instant::now();
            }
        }

        let state_time = state_start.elapsed().as_secs_f64();
        eprintln!(
            "[state-bfs] done in {:.2}s | states={} edges={}",
            state_time,
            states.len(),
            edges.len(),
        );

        // Convert DashMap to FxHashMap for the struct (used by expand_states)
        let state_to_id: FxHashMap<StateKeyPacked, StateId> = state_to_id_par.into_iter().collect();

        // Build predecessor arrays
        let (pred_ranges, predecessors) = build_predecessors(states.len(), &state_indices, &edges);

        let total_time = total_start.elapsed().as_secs_f64();
        eprintln!("[build] total {:.2}s", total_time);

        Universe {
            config: *config,
            boards,
            board_to_id,
            board_succ_ranges,
            board_succs,
            states,
            state_to_id,
            state_indices,
            edges,
            pred_ranges,
            predecessors,
            root_state_id,
        }
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the edge slice for a given range.
    pub fn edge_slice(&self, range: EdgeRange) -> &[FlatEdge] {
        &self.edges[range.start as usize..(range.start + range.len) as usize]
    }

    /// Iterate predecessors of a state.
    pub fn predecessors_of(&self, sid: StateId) -> &[PredecessorRef] {
        let range = &self.pred_ranges[sid as usize];
        &self.predecessors[range.start as usize..(range.start + range.len) as usize]
    }

    /// Check whether a state has been expanded (has a real StateIndex, not default).
    pub fn is_expanded(&self, sid: StateId) -> bool {
        let key = &self.states[sid as usize];
        let index = &self.state_indices[sid as usize];
        index.bag == key.bag
    }

    /// Expand a set of previously-unexpanded states on demand.
    /// Uses precomputed board successors — no board mutations needed.
    /// Returns the number of new states discovered.
    pub fn expand_states(&mut self, state_ids: &[StateId]) -> usize {
        let initial_state_count = self.states.len();

        for &sid in state_ids {
            let key = self.states[sid as usize];
            let mut index = StateIndex {
                bag: key.bag,
                piece_ranges: [EdgeRange::EMPTY; 7],
            };

            for branch in piece_branches(key.bag) {
                let pidx = branch.piece.index() as usize;
                let piece_edge_start = self.edges.len() as u32;

                // Look up precomputed board successors
                let board_range = self.board_succ_ranges[key.board_id as usize][pidx];
                for i in 0..board_range.len {
                    let bse = self.board_succs[(board_range.start + i) as usize];

                    let succ_key = StateKey::new(bse.board_id, branch.next_bag);
                    let packed = succ_key.pack();

                    let succ_id = if let Some(&id) = self.state_to_id.get(&packed) {
                        id
                    } else {
                        let id = intern_state(&mut self.states, &mut self.state_to_id, succ_key);
                        self.state_indices.push(StateIndex::default());
                        id
                    };

                    self.edges.push(FlatEdge {
                        succ: succ_id,
                        placement: bse.placement,
                    });
                }

                let piece_edge_len = self.edges.len() as u32 - piece_edge_start;
                index.piece_ranges[pidx] = EdgeRange {
                    start: piece_edge_start,
                    len: piece_edge_len,
                };
            }

            self.state_indices[sid as usize] = index;
        }

        self.states.len() - initial_state_count
    }

    /// Rebuild predecessor arrays from the current state_indices and edges.
    pub fn rebuild_predecessors(&mut self) {
        let (pred_ranges, predecessors) =
            build_predecessors(self.states.len(), &self.state_indices, &self.edges);
        self.pred_ranges = pred_ranges;
        self.predecessors = predecessors;
    }
}

// --- Helpers ---

fn intern_board(
    boards: &mut Vec<TetrisBoard>,
    board_to_id: &mut FxHashMap<TetrisBoard, BoardId>,
    board: &TetrisBoard,
) -> BoardId {
    if let Some(&id) = board_to_id.get(board) {
        return id;
    }
    let id = boards.len() as BoardId;
    boards.push(*board);
    board_to_id.insert(*board, id);
    id
}

fn intern_state(
    states: &mut Vec<StateKey>,
    state_to_id: &mut FxHashMap<StateKeyPacked, StateId>,
    key: StateKey,
) -> StateId {
    let id = states.len() as StateId;
    states.push(key);
    state_to_id.insert(key.pack(), id);
    id
}

/// Admissibility check with early exits ordered cheapest-first.
#[inline]
fn board_is_admissible(board: &TetrisBoard, adm: &BoardAdmissibility) -> bool {
    if adm.max_height != u32::MAX && board.height() > adm.max_height {
        return false;
    }
    if adm.max_holes != u32::MAX && board.total_holes() > adm.max_holes {
        return false;
    }
    if adm.max_roughness != u32::MAX && board.roughness() > adm.max_roughness {
        return false;
    }
    if adm.max_count != u32::MAX && board.count() > adm.max_count {
        return false;
    }
    true
}

/// Two-pass predecessor building with piece_idx tracking.
///
/// Each predecessor ref records which piece index was played, so the AND-OR
/// retrograde knows which counter to decrement.
fn build_predecessors(
    state_count: usize,
    state_indices: &[StateIndex],
    edges: &[FlatEdge],
) -> (Vec<EdgeRange>, Vec<PredecessorRef>) {
    let start = Instant::now();

    // Pass 1: count inbound edges per state
    let mut in_count = vec![0u32; state_count];
    for edge in edges {
        in_count[edge.succ as usize] += 1;
    }

    // Compute ranges
    let mut pred_ranges = Vec::with_capacity(state_count);
    let mut offset = 0u32;
    for &count in &in_count {
        pred_ranges.push(EdgeRange {
            start: offset,
            len: count,
        });
        offset += count;
    }

    // Pass 2: fill predecessor array
    let mut predecessors = vec![
        PredecessorRef {
            parent: 0,
            piece_idx: 0,
        };
        offset as usize
    ];
    let mut write_pos = vec![0u32; state_count];

    for sid in 0..state_count {
        let index = &state_indices[sid];
        for pidx in 0..7u8 {
            let range = &index.piece_ranges[pidx as usize];
            for edge in &edges[range.start as usize..(range.start + range.len) as usize] {
                let succ = edge.succ as usize;
                let pred_range = &pred_ranges[succ];
                let pos = pred_range.start + write_pos[succ];
                predecessors[pos as usize] = PredecessorRef {
                    parent: sid as StateId,
                    piece_idx: pidx,
                };
                write_pos[succ] += 1;
            }
        }
    }

    eprintln!(
        "[build] predecessors built in {:.2}s ({} refs)",
        start.elapsed().as_secs_f64(),
        predecessors.len(),
    );

    (pred_ranges, predecessors)
}
