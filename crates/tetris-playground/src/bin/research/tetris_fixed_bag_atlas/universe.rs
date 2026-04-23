use std::time::Instant;

use rustc_hash::FxHashMap;
use tetris_game::{TetrisBoard, TetrisPiecePlacement};

use crate::config::{BoardAdmissibility, SolverConfig};
use crate::state::{
    BoardId, EdgeRange, FlatEdge, PredecessorRef, StateId, StateKey, pack_placement,
};

/// The complete state graph for a fixed-cycle Tetris game.
///
/// Built via forward BFS from the empty board, then augmented with predecessor
/// links for backward propagation.
pub struct Universe {
    pub config: SolverConfig,
    // --- Board interning ---
    pub boards: Vec<TetrisBoard>,
    pub board_to_id: FxHashMap<TetrisBoard, BoardId>,
    // --- State interning ---
    pub states: Vec<StateKey>,
    pub state_to_id: FxHashMap<StateKey, StateId>,
    // --- Forward edges ---
    pub edge_ranges: Vec<EdgeRange>,
    pub edges: Vec<FlatEdge>,
    // --- Backward edges ---
    pub pred_ranges: Vec<EdgeRange>,
    pub predecessors: Vec<PredecessorRef>,
    // --- Notable states ---
    pub root_state_id: StateId,
}

impl Universe {
    /// Build the universe via forward BFS from the empty board at cycle position 0.
    pub fn build(config: &SolverConfig) -> Self {
        let start = Instant::now();

        let mut boards: Vec<TetrisBoard> = Vec::new();
        let mut board_to_id: FxHashMap<TetrisBoard, BoardId> = FxHashMap::default();
        let mut states: Vec<StateKey> = Vec::new();
        let mut state_to_id: FxHashMap<StateKey, StateId> = FxHashMap::default();
        let mut edge_ranges: Vec<EdgeRange> = Vec::new();
        let mut edges: Vec<FlatEdge> = Vec::new();

        // Intern the empty board
        let empty_board = TetrisBoard::new();
        let empty_board_id = intern_board(&mut boards, &mut board_to_id, &empty_board);

        // Intern the root state
        let root_key = StateKey::new(empty_board_id, 0);
        let root_state_id = intern_state(&mut states, &mut state_to_id, root_key);

        // BFS frontier: Vec + read cursor (better cache locality than VecDeque)
        let mut frontier: Vec<StateId> = Vec::new();
        let mut frontier_idx: usize = 0;
        frontier.push(root_state_id);

        // Reserve edge_ranges slot for root (and all future states as they're interned)
        edge_ranges.push(EdgeRange::default());

        let mut last_report = Instant::now();

        while frontier_idx < frontier.len() {
            let sid = frontier[frontier_idx];
            frontier_idx += 1;
            if states.len() >= config.max_states {
                eprintln!(
                    "[build] hit max_states cap ({}) — stopping expansion",
                    config.max_states,
                );
                // Drain remaining frontier states so they have edge_ranges (empty)
                break;
            }

            let key = states[sid as usize];
            let board = boards[key.board_id as usize];
            let piece = config.cycle[key.cycle_pos as usize];
            let next_pos = (key.cycle_pos + 1) % 7;

            let edge_start = edges.len() as u32;

            // Expand all valid placements of this piece
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut new_board = board;
                let result = new_board.apply_piece_placement(placement);

                if result.is_lost.into() {
                    continue;
                }
                if !board_is_admissible(&new_board, &config.admissibility) {
                    continue;
                }

                let new_board_id = intern_board(&mut boards, &mut board_to_id, &new_board);
                let succ_key = StateKey::new(new_board_id, next_pos);

                let succ_id = if let Some(&id) = state_to_id.get(&succ_key) {
                    id
                } else {
                    let id = intern_state(&mut states, &mut state_to_id, succ_key);
                    edge_ranges.push(EdgeRange::default());
                    frontier.push(id);
                    id
                };

                edges.push(FlatEdge {
                    succ: succ_id,
                    placement: pack_placement(placement),
                });
            }

            let edge_len = edges.len() as u32 - edge_start;
            edge_ranges[sid as usize] = EdgeRange {
                start: edge_start,
                len: edge_len,
            };

            // Progress reporting
            if last_report.elapsed().as_secs() >= 2 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = states.len() as f64 / elapsed;
                eprintln!(
                    "[build] {:.1}s | states={} boards={} edges={} frontier={} ({:.0} states/s)",
                    elapsed,
                    states.len(),
                    boards.len(),
                    edges.len(),
                    frontier.len() - frontier_idx,
                    rate,
                );
                last_report = Instant::now();
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        eprintln!(
            "[build] done in {:.2}s | states={} boards={} edges={}",
            elapsed,
            states.len(),
            boards.len(),
            edges.len(),
        );

        // Build predecessor arrays
        let (pred_ranges, predecessors) = build_predecessors(states.len(), &edge_ranges, &edges);

        Universe {
            config: *config,
            boards,
            board_to_id,
            states,
            state_to_id,
            edge_ranges,
            edges,
            pred_ranges,
            predecessors,
            root_state_id,
        }
    }

    /// Number of states in the universe.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of edges (total placements across all states).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate successors of a state.
    pub fn successors(&self, sid: StateId) -> &[FlatEdge] {
        let range = &self.edge_ranges[sid as usize];
        &self.edges[range.start as usize..(range.start + range.len) as usize]
    }

    /// Iterate predecessors of a state.
    pub fn predecessors_of(&self, sid: StateId) -> &[PredecessorRef] {
        let range = &self.pred_ranges[sid as usize];
        &self.predecessors[range.start as usize..(range.start + range.len) as usize]
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
    state_to_id: &mut FxHashMap<StateKey, StateId>,
    key: StateKey,
) -> StateId {
    let id = states.len() as StateId;
    states.push(key);
    state_to_id.insert(key, id);
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

/// Two-pass predecessor building: count inbound edges, then fill.
fn build_predecessors(
    state_count: usize,
    edge_ranges: &[EdgeRange],
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
    let mut predecessors = vec![PredecessorRef { parent: 0 }; offset as usize];
    let mut write_pos = vec![0u32; state_count];

    for sid in 0..state_count {
        let range = &edge_ranges[sid];
        for edge in &edges[range.start as usize..(range.start + range.len) as usize] {
            let succ = edge.succ as usize;
            let pred_range = &pred_ranges[succ];
            let pos = pred_range.start + write_pos[succ];
            predecessors[pos as usize] = PredecessorRef {
                parent: sid as StateId,
            };
            write_pos[succ] += 1;
        }
    }

    eprintln!(
        "[build] predecessors built in {:.2}s ({} refs)",
        start.elapsed().as_secs_f64(),
        predecessors.len(),
    );

    (pred_ranges, predecessors)
}
