use std::collections::VecDeque;
use std::time::Instant;

use crate::state::{PackedPlacement, StateId};
use crate::universe::Universe;

/// Result of the retrograde fixed-point computation.
pub struct SolveResult {
    /// Whether each state is in the winning set (can survive forever).
    pub alive: Vec<bool>,
    /// For each winning state, the best placement to make.
    pub best_action: Vec<Option<PackedPlacement>>,
}

impl SolveResult {
    pub fn winning_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    pub fn losing_count(&self) -> usize {
        self.alive.iter().filter(|&&a| !a).count()
    }
}

/// Greatest-fixed-point retrograde elimination.
///
/// Starts with all states alive. A state dies when it has zero alive successors
/// (no valid placement leads to a surviving state). Death propagates backward
/// through predecessors until the winning set stabilizes.
pub fn solve(universe: &Universe) -> SolveResult {
    let start = Instant::now();
    let n = universe.state_count();

    let mut alive = vec![true; n];
    let mut alive_succ_count: Vec<u32> = universe.edge_ranges.iter().map(|r| r.len).collect();

    // Seed the dead queue with states that have no successors at all
    let mut dead_queue: VecDeque<StateId> = VecDeque::new();
    for (sid, &count) in alive_succ_count.iter().enumerate() {
        if count == 0 {
            dead_queue.push_back(sid as StateId);
            alive[sid] = false;
        }
    }

    let initial_dead = dead_queue.len();
    eprintln!(
        "[retrograde] initial dead (no successors): {}",
        initial_dead,
    );

    // Propagate death backward
    let mut propagated = 0u64;
    let mut last_report = Instant::now();

    while let Some(dead_state) = dead_queue.pop_front() {
        for pred_ref in universe.predecessors_of(dead_state) {
            let parent = pred_ref.parent as usize;
            if !alive[parent] {
                continue;
            }
            alive_succ_count[parent] -= 1;
            if alive_succ_count[parent] == 0 {
                alive[parent] = false;
                dead_queue.push_back(parent as StateId);
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
                dead_queue.len(),
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

    // Extract best action for each winning state
    let best_action = extract_best_actions(universe, &alive);

    SolveResult { alive, best_action }
}

/// For each winning state, pick the best placement among alive successors.
///
/// Heuristic: prefer the successor with the lowest board height, breaking
/// ties by fewest holes.
fn extract_best_actions(universe: &Universe, alive: &[bool]) -> Vec<Option<PackedPlacement>> {
    let n = universe.state_count();
    let mut best_action: Vec<Option<PackedPlacement>> = vec![None; n];

    for sid in 0..n {
        if !alive[sid] {
            continue;
        }

        let mut best_placement: Option<PackedPlacement> = None;
        let mut best_score: u32 = u32::MAX;

        for edge in universe.successors(sid as StateId) {
            if !alive[edge.succ as usize] {
                continue;
            }

            let succ_key = &universe.states[edge.succ as usize];
            let succ_board = &universe.boards[succ_key.board_id as usize];
            // Score: lower is better (height * 16 + holes gives height priority)
            let score = succ_board.height() * 16 + succ_board.total_holes();

            if score < best_score {
                best_score = score;
                best_placement = Some(edge.placement);
            }
        }

        best_action[sid] = best_placement;
    }

    best_action
}
