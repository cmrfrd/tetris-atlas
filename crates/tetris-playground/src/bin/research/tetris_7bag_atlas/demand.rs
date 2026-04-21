use std::collections::VecDeque;
use std::time::Instant;

use crate::retrograde::{self, SolveResult};
use crate::state::{StateId, piece_branches};
use crate::universe::Universe;

/// Result of demand-driven verification.
pub enum DemandResult {
    /// The strategy is fully verified — every state it touches is expanded
    /// and confirmed winning. This constitutes a proof of infinite play.
    Verified {
        rounds: u32,
        verified_states: usize,
    },
    /// The root was disproved — after expanding needed states, retrograde
    /// killed the root. No winning strategy exists in this state space.
    Disproved {
        rounds: u32,
    },
    /// The initial solve didn't find root winning (nothing to verify).
    RootNotWinning,
}

/// Demand-driven verification loop.
///
/// Starting from a capped solve where unexpanded states are optimistically alive,
/// iteratively:
/// 1. Trace the winning strategy from root, collecting unexpanded states
/// 2. Expand those states (compute their edges)
/// 3. Rebuild predecessors and re-run retrograde
/// 4. Repeat until the strategy only touches expanded states (verified) or root dies
pub fn demand_verify(universe: &mut Universe, result: &mut SolveResult) -> DemandResult {
    let start = Instant::now();

    if !result.alive[universe.root_state_id as usize] {
        return DemandResult::RootNotWinning;
    }

    for round in 1u32.. {
        // Trace the strategy and find unexpanded states it depends on
        let (traced, unexpanded) = trace_strategy(universe, result);

        eprintln!(
            "[demand] round {} | traced={} unexpanded={} universe={}",
            round,
            traced,
            unexpanded.len(),
            universe.state_count(),
        );

        if unexpanded.is_empty() {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "[demand] VERIFIED in {:.2}s ({} rounds, {} strategy states)",
                elapsed, round, traced,
            );
            return DemandResult::Verified {
                rounds: round,
                verified_states: traced,
            };
        }

        // Expand the unexpanded states
        let new_states = universe.expand_states(&unexpanded);
        eprintln!(
            "[demand] expanded {} states, {} new states discovered",
            unexpanded.len(),
            new_states,
        );

        // Rebuild predecessors for the updated graph
        universe.rebuild_predecessors();

        // Re-run retrograde
        *result = retrograde::solve(universe);

        if !result.alive[universe.root_state_id as usize] {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "[demand] DISPROVED in {:.2}s ({} rounds)",
                elapsed, round,
            );
            return DemandResult::Disproved { rounds: round };
        }
    }

    // Unreachable (loop is infinite with early returns)
    DemandResult::Disproved { rounds: 0 }
}

/// Trace the winning strategy from root via BFS.
///
/// For each alive state, follows `best_action[state][piece]` for every piece
/// in the bag. Returns (total traced count, list of unexpanded state IDs).
fn trace_strategy(
    universe: &Universe,
    result: &SolveResult,
) -> (usize, Vec<StateId>) {
    let mut visited = vec![false; universe.state_count()];
    let mut queue: VecDeque<StateId> = VecDeque::new();
    let mut unexpanded: Vec<StateId> = Vec::new();

    let root = universe.root_state_id;
    if !result.alive[root as usize] {
        return (0, unexpanded);
    }

    visited[root as usize] = true;
    queue.push_back(root);

    while let Some(sid) = queue.pop_front() {
        if !universe.is_expanded(sid) {
            unexpanded.push(sid);
            continue;
        }

        let index = &universe.state_indices[sid as usize];

        for branch in piece_branches(index.bag) {
            let pidx = branch.piece.index() as usize;

            let Some(packed) = result.best_action[sid as usize][pidx] else {
                continue;
            };

            // Find the successor for this action
            let range = index.piece_ranges[pidx];
            let Some(edge) = universe
                .edge_slice(range)
                .iter()
                .find(|e| e.placement == packed)
            else {
                continue;
            };

            let succ = edge.succ as usize;
            if !visited[succ] && result.alive[succ] {
                visited[succ] = true;
                queue.push_back(edge.succ);
            }
        }
    }

    let traced = visited.iter().filter(|&&v| v).count();
    (traced, unexpanded)
}
