use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::transition::TransitionCache;
use crate::universe::{BoardId, BoardUniverse};

const GFP_PROGRESS_INTERVAL_SECS: u64 = 2;
const INNER_PROGRESS_CALL_INTERVAL: u64 = 65_536;

#[derive(Debug, Default, Clone, Copy)]
struct GfpStats {
    inner_calls: u64,
    memo_hits: u64,
    memo_true: u64,
    memo_false: u64,
    piece_branches: u64,
    placement_branches: u64,
    terminal_checks: u64,
    terminal_successes: u64,
    registry_cap_prunes: u64,
    deepest_ply: u32,
}

struct GfpProgress {
    iteration: u32,
    current_root_board: BoardId,
    total_plies: u32,
    root_start: Instant,
    root_start_calls: u64,
    last_report: Instant,
}

/// Memo key for the inner AND-OR search: (board_id, bag_state, cycles_remaining).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MemoKey {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
}

/// Result of the GFP computation.
pub struct GfpResult {
    /// The set of board IDs that survived the GFP.
    pub safe_set: FxHashSet<BoardId>,
    /// Number of GFP iterations performed.
    pub iterations: u32,
    /// Total boards removed across all iterations.
    pub total_removed: usize,
    /// Transition cache stats.
    pub transition_cache_entries: usize,
    /// Number of branches pruned because the intermediate registry cap was hit.
    pub registry_cap_hits: u64,
}

/// Extended board registry that includes both universe boards (safe-set candidates)
/// and intermediate boards (used as stepping stones during AND-OR search).
struct BoardRegistry {
    boards: Vec<TetrisBoard>,
    board_to_id: FxHashMap<TetrisBoard, BoardId>,
    /// Number of boards from the original universe (IDs 0..universe_count are safe-set candidates).
    universe_count: u32,
    max_boards: usize,
    cap_hits: u64,
}

impl BoardRegistry {
    fn from_universe(universe: &BoardUniverse, max_boards: usize) -> Self {
        Self {
            boards: universe.boards.clone(),
            board_to_id: universe.board_to_id.clone(),
            universe_count: universe.board_count() as u32,
            max_boards,
            cap_hits: 0,
        }
    }

    /// Get or insert a board, returning its ID. Boards beyond universe_count are intermediates.
    fn get_or_insert(&mut self, board: TetrisBoard) -> Option<BoardId> {
        if let Some(&id) = self.board_to_id.get(&board) {
            return Some(id);
        }
        if self.boards.len() >= self.max_boards {
            self.cap_hits += 1;
            return None;
        }
        let id = self.boards.len() as BoardId;
        self.boards.push(board);
        self.board_to_id.insert(board, id);
        Some(id)
    }

    /// Check if a board ID is from the original universe (safe-set candidate).
    #[inline]
    fn is_universe_board(&self, id: BoardId) -> bool {
        id < self.universe_count
    }
}

/// Run the greatest-fixed-point iteration.
///
/// 1. Initialize safe_set = all discovered boards
/// 2. For each board, run inner AND-OR check (over `bag_cycles` cycles). If fails, mark for removal.
/// 3. Remove marked boards, clear memo, repeat.
/// 4. Stop when no boards are removed (fixed point).
pub fn run_gfp(universe: &BoardUniverse, config: &SolverConfig) -> GfpResult {
    let start = Instant::now();

    let mut safe_set: FxHashSet<BoardId> = (0..universe.board_count() as BoardId).collect();
    let mut transitions = TransitionCache::new(config.admissibility);
    let mut iterations: u32 = 0;
    let mut total_removed: usize = 0;

    // Extended registry: universe boards + intermediate boards discovered during search
    let mut registry = BoardRegistry::from_universe(universe, config.max_registry_boards);

    eprintln!(
        "[gfp] starting with {} boards in safe set, bag_cycles={}",
        safe_set.len(),
        config.bag_cycles,
    );

    loop {
        iterations += 1;
        let iter_start = Instant::now();

        // Memo is valid only within one GFP iteration (safe_set changes between iterations)
        let mut memo: FxHashMap<MemoKey, bool> = FxHashMap::default();
        let mut stats = GfpStats::default();
        let total_plies = config.bag_cycles * 7;

        let mut to_remove: Vec<BoardId> = Vec::new();
        let safe_snapshot: Vec<BoardId> = safe_set.iter().copied().collect();
        let total_boards = safe_snapshot.len();
        let mut last_progress = Instant::now();
        let mut checked: usize = 0;
        let mut surviving: usize = 0;

        for &board_id in &safe_snapshot {
            let now = Instant::now();
            let mut inner_progress = GfpProgress {
                iteration: iterations,
                current_root_board: board_id,
                total_plies,
                root_start: now,
                root_start_calls: stats.inner_calls,
                last_report: now,
            };
            let survives = inner_check(
                board_id,
                TetrisPieceBagState::new(),
                config.bag_cycles,
                &mut registry,
                &safe_set,
                &mut transitions,
                &mut memo,
                &mut stats,
                &mut inner_progress,
            );
            checked += 1;
            if survives {
                surviving += 1;
            } else {
                to_remove.push(board_id);
            }

            if last_progress.elapsed().as_secs() >= GFP_PROGRESS_INTERVAL_SECS {
                let elapsed = iter_start.elapsed().as_secs_f64();
                let rate = checked as f64 / elapsed;
                let transition_stats = transitions.stats();
                eprintln!(
                    "[gfp] iter {} progress: {}/{} checked current_board={} surviving={} removed={} elapsed={:.1}s rate={:.1}/s memo={} memo_hits={} memo_true={} memo_false={} calls={} branches={} placements={} terminals={}/{} depth={}/{} registry={}/{} reg_cap_hits={} reg_prunes={} tcache={} cache_hits={} cache_misses={} gen={} lost_prunes={} adm_prunes={}",
                    iterations,
                    checked,
                    total_boards,
                    board_id,
                    surviving,
                    to_remove.len(),
                    elapsed,
                    rate,
                    memo.len(),
                    stats.memo_hits,
                    stats.memo_true,
                    stats.memo_false,
                    stats.inner_calls,
                    stats.piece_branches,
                    stats.placement_branches,
                    stats.terminal_successes,
                    stats.terminal_checks,
                    stats.deepest_ply,
                    total_plies,
                    registry.boards.len(),
                    registry.max_boards,
                    registry.cap_hits,
                    stats.registry_cap_prunes,
                    transitions.len(),
                    transition_stats.cache_hits,
                    transition_stats.cache_misses,
                    transition_stats.successors_generated,
                    transition_stats.lost_prunes,
                    transition_stats.admissibility_prunes,
                );
                last_progress = Instant::now();
            }
        }

        let removed = to_remove.len();
        for bid in &to_remove {
            safe_set.remove(bid);
        }
        total_removed += removed;

        let iter_elapsed = iter_start.elapsed().as_secs_f64();
        let transition_stats = transitions.stats();
        eprintln!(
            "[gfp] iteration {} | removed={} remaining={} memo_size={} memo_hits={} memo_true={} memo_false={} calls={} branches={} placements={} terminals={}/{} depth={}/{} registry={}/{} cap_hits={} reg_prunes={} transition_cache={} cache_hits={} cache_misses={} gen={} lost_prunes={} adm_prunes={} ({:.2}s)",
            iterations,
            removed,
            safe_set.len(),
            memo.len(),
            stats.memo_hits,
            stats.memo_true,
            stats.memo_false,
            stats.inner_calls,
            stats.piece_branches,
            stats.placement_branches,
            stats.terminal_successes,
            stats.terminal_checks,
            stats.deepest_ply,
            total_plies,
            registry.boards.len(),
            registry.max_boards,
            registry.cap_hits,
            stats.registry_cap_prunes,
            transitions.len(),
            transition_stats.cache_hits,
            transition_stats.cache_misses,
            transition_stats.successors_generated,
            transition_stats.lost_prunes,
            transition_stats.admissibility_prunes,
            iter_elapsed,
        );

        if removed == 0 {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[gfp] converged in {} iterations, {:.2}s | safe_set={} total_removed={}",
        iterations,
        elapsed,
        safe_set.len(),
        total_removed,
    );

    GfpResult {
        safe_set,
        iterations,
        total_removed,
        transition_cache_entries: transitions.len(),
        registry_cap_hits: registry.cap_hits,
    }
}

/// Entry point: AND-OR check over `cycles_remaining` bag cycles.
///
/// A board is "safe" if the player can survive `cycles_remaining` complete bag
/// cycles (each with adversarial piece ordering) and end on a board in `safe_set`.
fn inner_check(
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
    registry: &mut BoardRegistry,
    safe_set: &FxHashSet<BoardId>,
    transitions: &mut TransitionCache,
    memo: &mut FxHashMap<MemoKey, bool>,
    stats: &mut GfpStats,
    progress: &mut GfpProgress,
) -> bool {
    stats.inner_calls += 1;
    let plies_remaining = if cycles_remaining == 0 {
        0
    } else {
        bag.count() as u32 + (cycles_remaining - 1) * 7
    };
    stats.deepest_ply = stats
        .deepest_ply
        .max(progress.total_plies.saturating_sub(plies_remaining));
    maybe_report_inner_progress(progress, stats, registry, transitions, memo);

    // Terminal: all cycles exhausted, check safe set
    // Only universe boards can be in the safe set.
    if cycles_remaining == 0 {
        stats.terminal_checks += 1;
        let result = registry.is_universe_board(board_id) && safe_set.contains(&board_id);
        if result {
            stats.terminal_successes += 1;
        }
        return result;
    }

    // Memo lookup
    let key = MemoKey {
        board_id,
        bag,
        cycles_remaining,
    };
    if let Some(&result) = memo.get(&key) {
        stats.memo_hits += 1;
        return result;
    }

    let board = registry.boards[board_id as usize];

    // AND node: adversary picks piece (ALL must be survivable)
    // Sort by fewest placements first for fail-fast
    let pieces: Vec<(TetrisPiece, TetrisPieceBagState)> = bag.iter_next_states().collect();
    let mut piece_order: Vec<usize> = (0..pieces.len()).collect();
    piece_order.sort_unstable_by_key(|&i| {
        let (piece, _) = pieces[i];
        TetrisPiecePlacement::all_from_piece(piece).len()
    });

    let mut result = true;
    for &i in &piece_order {
        stats.piece_branches += 1;
        let (piece, next_bag) = pieces[i];

        // Determine next cycles_remaining: if the bag just refilled (last piece
        // was popped), we've completed a cycle.
        let next_cycles = if next_bag == TetrisPieceBagState::new() {
            cycles_remaining - 1
        } else {
            cycles_remaining
        };

        // OR node: player picks placement (ONE must work)
        let mut transitions_vec = transitions.successors(board, piece).to_vec();
        // Sort by lines cleared descending (succeed fast)
        transitions_vec.sort_unstable_by(|a, b| b.lines_cleared.cmp(&a.lines_cleared));

        let mut found = false;
        for transition in &transitions_vec {
            stats.placement_branches += 1;
            // Intern the successor board — this includes intermediate boards
            // that pass admissibility but not safe_set_admissibility. They get
            // IDs beyond universe_count and can be used as stepping stones.
            let Some(next_board_id) = registry.get_or_insert(transition.board) else {
                stats.registry_cap_prunes += 1;
                continue;
            };

            if inner_check(
                next_board_id,
                next_bag,
                next_cycles,
                registry,
                safe_set,
                transitions,
                memo,
                stats,
                progress,
            ) {
                found = true;
                break;
            }
        }

        if !found {
            result = false;
            break;
        }
    }

    memo.insert(key, result);
    if result {
        stats.memo_true += 1;
    } else {
        stats.memo_false += 1;
    }
    result
}

fn maybe_report_inner_progress(
    progress: &mut GfpProgress,
    stats: &GfpStats,
    registry: &BoardRegistry,
    transitions: &TransitionCache,
    memo: &FxHashMap<MemoKey, bool>,
) {
    if stats.inner_calls % INNER_PROGRESS_CALL_INTERVAL != 0 {
        return;
    }
    if progress.last_report.elapsed().as_secs() < GFP_PROGRESS_INTERVAL_SECS {
        return;
    }

    let transition_stats = transitions.stats();
    let root_elapsed = progress.root_start.elapsed().as_secs_f64();
    let root_calls = stats.inner_calls - progress.root_start_calls;
    let root_rate = root_calls as f64 / root_elapsed.max(f64::EPSILON);
    eprintln!(
        "[gfp:inner] iter={} root_board={} root_elapsed={:.1}s root_calls={} root_rate={:.1}/s calls={} memo={} memo_hits={} memo_true={} memo_false={} branches={} placements={} terminals={}/{} depth={}/{} registry={}/{} cap_hits={} reg_prunes={} tcache={} cache_hits={} cache_misses={} gen={} lost_prunes={} adm_prunes={}",
        progress.iteration,
        progress.current_root_board,
        root_elapsed,
        root_calls,
        root_rate,
        stats.inner_calls,
        memo.len(),
        stats.memo_hits,
        stats.memo_true,
        stats.memo_false,
        stats.piece_branches,
        stats.placement_branches,
        stats.terminal_successes,
        stats.terminal_checks,
        stats.deepest_ply,
        progress.total_plies,
        registry.boards.len(),
        registry.max_boards,
        registry.cap_hits,
        stats.registry_cap_prunes,
        transitions.len(),
        transition_stats.cache_hits,
        transition_stats.cache_misses,
        transition_stats.successors_generated,
        transition_stats.lost_prunes,
        transition_stats.admissibility_prunes,
    );
    progress.last_report = Instant::now();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BoardAdmissibility;

    #[test]
    fn gfp_with_tiny_constraints_converges() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 0,
                max_roughness: 2,
                max_count: u32::MAX,
            },
            safe_set_admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 0,
                max_roughness: 2,
                max_count: u32::MAX,
            },
            bag_cycles: 1,
            max_boards: 10_000,
            max_registry_boards: 50_000,
        };
        let universe = BoardUniverse::discover(&config);
        let result = run_gfp(&universe, &config);
        // GFP must converge (iterations > 0)
        assert!(result.iterations >= 1);
        // Safe set size <= initial board count
        assert!(result.safe_set.len() <= universe.board_count());
    }

    #[test]
    fn gfp_multi_cycle_converges() {
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 0,
                max_roughness: 2,
                max_count: u32::MAX,
            },
            safe_set_admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 0,
                max_roughness: 2,
                max_count: u32::MAX,
            },
            bag_cycles: 2,
            max_boards: 10_000,
            max_registry_boards: 50_000,
        };
        let universe = BoardUniverse::discover(&config);
        let result = run_gfp(&universe, &config);
        assert!(result.iterations >= 1);
        assert!(result.safe_set.len() <= universe.board_count());
    }

    #[test]
    fn gfp_with_loose_intermediates_uses_stepping_stones() {
        // Intermediate admissibility is looser than safe-set admissibility.
        // This tests that the search can use intermediate boards as stepping stones.
        let config = SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 4,
                max_holes: 1,
                max_roughness: 6,
                max_count: u32::MAX,
            },
            safe_set_admissibility: BoardAdmissibility {
                max_height: 2,
                max_holes: 0,
                max_roughness: 2,
                max_count: u32::MAX,
            },
            bag_cycles: 1,
            max_boards: 10_000,
            max_registry_boards: 50_000,
        };
        let universe = BoardUniverse::discover(&config);
        let _result = run_gfp(&universe, &config);
        // Just verify it runs without panicking — the stepping stones
        // should be discoverable via the registry.
    }
}
