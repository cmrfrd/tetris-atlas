use std::time::Instant;

use crate::state::{PackedPlacement, StateId, piece_branches};
use crate::universe::Universe;

/// Result of the AND-OR retrograde fixed-point computation.
pub struct SolveResult {
    /// Whether each state is in the winning set (survives forever against
    /// adversarial piece draws).
    pub alive: Vec<bool>,
    /// For each winning state and each piece in its bag, the best placement.
    /// Indexed as `best_action[state_id][piece.index()]`.
    pub best_action: Vec<[Option<PackedPlacement>; 7]>,
}

impl SolveResult {
    pub fn winning_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    pub fn losing_count(&self) -> usize {
        self.alive.iter().filter(|&&a| !a).count()
    }
}

/// AND-OR greatest-fixed-point retrograde elimination.
///
/// Starts with all states alive. A state dies when ANY piece in its bag has
/// zero alive successors (the adversary can pick that piece and the player
/// has no surviving placement). Death propagates backward through predecessors,
/// decrementing per-(state, piece) counters.
pub fn solve(universe: &Universe) -> SolveResult {
    let start = Instant::now();
    let n = universe.state_count();

    // Per-(state, piece_idx) alive placement count.
    // Pieces IN the bag: initialized to edge count for that piece branch.
    // Pieces NOT in the bag: u16::MAX sentinel (never triggers death).
    // Max placements per piece is 34 (L/J), so u16 is more than sufficient.
    let mut alive_placement_count: Vec<u16> = vec![u16::MAX; n * 7];
    let mut alive = vec![true; n];

    // Initialize counters from the graph
    for sid in 0..n {
        let index = &universe.state_indices[sid];
        for branch in piece_branches(index.bag) {
            let pidx = branch.piece.index() as usize;
            alive_placement_count[sid * 7 + pidx] = index.piece_ranges[pidx].len as u16;
        }
    }

    // Seed dead queue: states where any bag-present piece has 0 valid placements.
    // Note: unexpanded states (hit max_states cap) have default StateIndex with
    // bag=0, so piece_branches returns nothing and they stay alive. This is
    // intentional — it gives an optimistic result for capped runs.
    let mut dead_queue: Vec<StateId> = Vec::new();
    for sid in 0..n {
        let index = &universe.state_indices[sid];
        for branch in piece_branches(index.bag) {
            let pidx = branch.piece.index() as usize;
            if alive_placement_count[sid * 7 + pidx] == 0 {
                alive[sid] = false;
                dead_queue.push(sid as StateId);
                break;
            }
        }
    }

    let initial_dead = dead_queue.len();
    eprintln!(
        "[retrograde] initial dead (piece branch with no placements): {}",
        initial_dead,
    );

    // Propagate death backward using Vec + read cursor (better cache locality
    // than VecDeque for this pattern).
    let mut propagated = 0u64;
    let mut dead_idx: usize = 0;
    let mut last_report = Instant::now();

    while dead_idx < dead_queue.len() {
        let dead_state = dead_queue[dead_idx];
        dead_idx += 1;

        for pred_ref in universe.predecessors_of(dead_state) {
            let parent = pred_ref.parent as usize;
            if !alive[parent] {
                continue;
            }

            let pidx = pred_ref.piece_idx as usize;
            let counter = &mut alive_placement_count[parent * 7 + pidx];

            // Sentinel check (piece not in parent's bag — shouldn't happen
            // with a correct graph, but defensive)
            if *counter == u16::MAX {
                continue;
            }

            *counter = counter.saturating_sub(1);
            if *counter == 0 {
                // All placements for this piece lead to dead states.
                // Adversary can pick this piece -> parent is dead.
                alive[parent] = false;
                dead_queue.push(parent as StateId);
            }
        }

        propagated += 1;
        if last_report.elapsed().as_secs() >= 2 {
            let elapsed = start.elapsed().as_secs_f64();
            let current_dead = alive.iter().filter(|&&a| !a).count();
            eprintln!(
                "[retrograde] {:.1}s | propagated={} dead={}/{} queue={}",
                elapsed,
                propagated,
                current_dead,
                n,
                dead_queue.len() - dead_idx,
            );
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let winning = alive.iter().filter(|&&a| a).count();
    eprintln!(
        "[retrograde] done in {:.2}s | winning={} losing={} total={}",
        elapsed,
        winning,
        n - winning,
        n,
    );

    let best_action = extract_best_actions(universe, &alive);

    SolveResult { alive, best_action }
}

/// For each winning state and each piece in its bag, pick the best placement
/// among alive successors.
///
/// Heuristic: prefer the successor with the lowest `height * 16 + total_holes`.
fn extract_best_actions(universe: &Universe, alive: &[bool]) -> Vec<[Option<PackedPlacement>; 7]> {
    let n = universe.state_count();
    let mut best_action: Vec<[Option<PackedPlacement>; 7]> = vec![[None; 7]; n];

    for sid in 0..n {
        if !alive[sid] {
            continue;
        }

        let index = &universe.state_indices[sid];

        for branch in piece_branches(index.bag) {
            let pidx = branch.piece.index() as usize;
            let range = index.piece_ranges[pidx];

            let mut best_placement: Option<PackedPlacement> = None;
            let mut best_score: u32 = u32::MAX;

            for edge in universe.edge_slice(range) {
                if !alive[edge.succ as usize] {
                    continue;
                }

                let succ_key = &universe.states[edge.succ as usize];
                let succ_board = &universe.boards[succ_key.board_id as usize];
                let score = succ_board.height() * 16 + succ_board.total_holes();

                if score < best_score {
                    best_score = score;
                    best_placement = Some(edge.placement);
                }
            }

            best_action[sid][pidx] = best_placement;
        }
    }

    best_action
}
