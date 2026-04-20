use std::collections::VecDeque;
use std::time::{Duration, Instant};

use tetris_game::TetrisPiece;

use crate::graph::EdgeRange;
use crate::state::{PackedPlacement, StateId, piece_branches};
use crate::universe::Universe;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SolveResult {
    pub winning: Vec<bool>,
    pub depth: Vec<u32>,
    pub best_action: Vec<[Option<PackedPlacement>; 7]>,
    pub pass_stats: SolvePassStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolvePassStats {
    pub passes: u32,
    pub states_marked_winning: u32,
}

pub fn solve(universe: &Universe) -> SolveResult {
    const LOG_EVERY: Duration = Duration::from_secs(2);
    let state_count = universe.states.len();
    let mut result = SolveResult {
        winning: vec![false; state_count],
        depth: vec![0; state_count],
        best_action: vec![[None; 7]; state_count],
        pass_stats: SolvePassStats::default(),
    };
    let mut winning_queue = VecDeque::new();
    let mut satisfied_mask = vec![0u8; state_count];
    let mut winning_count = 0usize;
    let mut processed_winning = 0usize;
    let winning_start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_processed_winning = 0usize;
    let mut last_winning_count = 0usize;

    for state_id in universe.target_states() {
        result.winning[state_id as usize] = true;
        winning_queue.push_back(state_id);
        winning_count += 1;
    }

    while let Some(child_state) = winning_queue.pop_front() {
        processed_winning += 1;
        for predecessor in universe.predecessor_slice(child_state) {
            let parent_idx = predecessor.parent as usize;
            if result.winning[parent_idx] {
                continue;
            }

            let piece_bit = 1_u8 << predecessor.piece_idx;
            if (satisfied_mask[parent_idx] & piece_bit) != 0 {
                continue;
            }

            satisfied_mask[parent_idx] |= piece_bit;
            let required_mask = u8::from(universe.state_key(predecessor.parent).bag);
            if satisfied_mask[parent_idx] != required_mask {
                continue;
            }

            result.winning[parent_idx] = true;
            winning_queue.push_back(predecessor.parent);
            result.pass_stats.states_marked_winning += 1;
            winning_count += 1;
        }

        if last_log.elapsed() >= LOG_EVERY {
            let log_elapsed = last_log.elapsed().as_secs_f64().max(1e-9);
            let total_elapsed = winning_start.elapsed().as_secs_f64().max(1e-9);
            let delta_processed = processed_winning.saturating_sub(last_processed_winning);
            let delta_winning = winning_count.saturating_sub(last_winning_count);
            eprintln!(
                "[success-set:winning] elapsed={:.1}s processed={} queue={} winning={} losing={} new_winning={} process_rate={:.1}/s winning_rate={:.1}/s coverage={:.2}%",
                total_elapsed,
                processed_winning,
                winning_queue.len(),
                winning_count,
                state_count.saturating_sub(winning_count),
                delta_winning,
                delta_processed as f64 / log_elapsed,
                delta_winning as f64 / log_elapsed,
                100.0 * winning_count as f64 / state_count.max(1) as f64,
            );
            last_log = Instant::now();
            last_processed_winning = processed_winning;
            last_winning_count = winning_count;
        }
    }

    result.pass_stats.passes = 1;
    recompute_exact_depths(universe, &mut result);
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WinningEvaluation {
    pub depth: u32,
    pub best_action: [Option<PackedPlacement>; 7],
}

pub fn evaluate_root(universe: &Universe, result: &SolveResult) -> Option<WinningEvaluation> {
    evaluate_state(universe, result, universe.root_state_id)
}

pub fn evaluate_state(
    universe: &Universe,
    result: &SolveResult,
    state_id: StateId,
) -> Option<WinningEvaluation> {
    let state = universe.state_key(state_id);
    let state_index = universe.state_index(state_id);
    let mut best_action = [None; 7];
    let mut worst_case_depth = 0;

    for branch in piece_branches(state.bag) {
        let piece_idx = branch.piece.index() as usize;
        let range = state_index.piece_ranges[piece_idx];
        let (branch_depth, placement) =
            best_winning_successor(universe, result, range, branch.piece)?;

        best_action[piece_idx] = Some(placement);
        worst_case_depth = worst_case_depth.max(branch_depth);
    }

    Some(WinningEvaluation {
        depth: worst_case_depth + 1,
        best_action,
    })
}

pub fn best_winning_successor(
    universe: &Universe,
    result: &SolveResult,
    range: EdgeRange,
    piece: TetrisPiece,
) -> Option<(u32, PackedPlacement)> {
    let mut best: Option<(u32, PackedPlacement)> = None;

    for edge in universe.edge_slice(range) {
        if !result.winning[edge.succ as usize] {
            continue;
        }

        let succ_depth = result.depth[edge.succ as usize];
        match best {
            Some((best_depth, best_placement))
                if succ_depth > best_depth
                    || (succ_depth == best_depth && edge.placement >= best_placement) => {}
            _ => best = Some((succ_depth, edge.placement)),
        }
    }

    debug_assert!(
        best.is_none_or(
            |(_, placement)| TetrisPiece::from_index(piece.index()) == piece
                && tetris_game::TetrisPiecePlacement::from_index(placement).piece == piece
        )
    );

    best
}

fn recompute_exact_depths(universe: &Universe, result: &mut SolveResult) {
    const LOG_EVERY: Duration = Duration::from_secs(2);
    let state_count = universe.states.len();
    let mut exact_depth = vec![u32::MAX; state_count];
    let mut exact_action = vec![[None; 7]; state_count];
    let mut depth_queue = VecDeque::new();
    let mut queued = vec![false; state_count];
    let mut processed = 0usize;
    let mut known_depths = 0usize;
    let total_winning = result
        .winning
        .iter()
        .filter(|&&is_winning| is_winning)
        .count();
    let depth_start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_processed = 0usize;
    let mut last_known_depths = 0usize;

    for state_id in universe.target_states() {
        exact_depth[state_id as usize] = 0;
        depth_queue.push_back(state_id);
        queued[state_id as usize] = true;
        known_depths += 1;
    }

    while let Some(child_state) = depth_queue.pop_front() {
        queued[child_state as usize] = false;
        processed += 1;

        for predecessor in universe.predecessor_slice(child_state) {
            let parent_state = predecessor.parent;
            if !result.winning[parent_state as usize] || exact_depth[parent_state as usize] == 0 {
                continue;
            }

            let Some(evaluation) =
                evaluate_state_with_depths(universe, result, &exact_depth, parent_state)
            else {
                continue;
            };

            if evaluation.depth < exact_depth[parent_state as usize]
                || (evaluation.depth == exact_depth[parent_state as usize]
                    && evaluation.best_action != exact_action[parent_state as usize])
            {
                if exact_depth[parent_state as usize] == u32::MAX {
                    known_depths += 1;
                }
                exact_depth[parent_state as usize] = evaluation.depth;
                exact_action[parent_state as usize] = evaluation.best_action;
                if !queued[parent_state as usize] {
                    depth_queue.push_back(parent_state);
                    queued[parent_state as usize] = true;
                }
            }
        }

        if last_log.elapsed() >= LOG_EVERY {
            let log_elapsed = last_log.elapsed().as_secs_f64().max(1e-9);
            let total_elapsed = depth_start.elapsed().as_secs_f64().max(1e-9);
            let delta_processed = processed.saturating_sub(last_processed);
            let delta_known_depths = known_depths.saturating_sub(last_known_depths);
            eprintln!(
                "[success-set:depth] elapsed={:.1}s processed={} queue={} known_depths={} unresolved={} new_known_depths={} process_rate={:.1}/s depth_rate={:.1}/s resolved={:.2}%",
                total_elapsed,
                processed,
                depth_queue.len(),
                known_depths,
                total_winning.saturating_sub(known_depths),
                delta_known_depths,
                delta_processed as f64 / log_elapsed,
                delta_known_depths as f64 / log_elapsed,
                100.0 * known_depths as f64 / total_winning.max(1) as f64,
            );
            last_log = Instant::now();
            last_processed = processed;
            last_known_depths = known_depths;
        }
    }

    for state_id in 0..state_count as StateId {
        if !result.winning[state_id as usize] {
            continue;
        }

        debug_assert_ne!(
            exact_depth[state_id as usize],
            u32::MAX,
            "winning state must have finite exact depth"
        );
        result.depth[state_id as usize] = exact_depth[state_id as usize];
        result.best_action[state_id as usize] = exact_action[state_id as usize];
    }
}

fn evaluate_state_with_depths(
    universe: &Universe,
    result: &SolveResult,
    depth: &[u32],
    state_id: StateId,
) -> Option<WinningEvaluation> {
    let state = universe.state_key(state_id);
    let state_index = universe.state_index(state_id);
    let mut best_action = [None; 7];
    let mut worst_case_depth = 0;

    for branch in piece_branches(state.bag) {
        let piece_idx = branch.piece.index() as usize;
        let range = state_index.piece_ranges[piece_idx];
        let (branch_depth, placement) =
            best_winning_successor_with_depths(universe, result, depth, range, branch.piece)?;
        best_action[piece_idx] = Some(placement);
        worst_case_depth = worst_case_depth.max(branch_depth);
    }

    Some(WinningEvaluation {
        depth: worst_case_depth + 1,
        best_action,
    })
}

fn best_winning_successor_with_depths(
    universe: &Universe,
    result: &SolveResult,
    depth: &[u32],
    range: EdgeRange,
    piece: TetrisPiece,
) -> Option<(u32, PackedPlacement)> {
    let mut best: Option<(u32, PackedPlacement)> = None;

    for edge in universe.edge_slice(range) {
        if !result.winning[edge.succ as usize] {
            continue;
        }

        let succ_depth = depth[edge.succ as usize];
        if succ_depth == u32::MAX {
            continue;
        }

        match best {
            Some((best_depth, best_placement))
                if succ_depth > best_depth
                    || (succ_depth == best_depth && edge.placement >= best_placement) => {}
            _ => best = Some((succ_depth, edge.placement)),
        }
    }

    debug_assert!(
        best.is_none_or(
            |(_, placement)| TetrisPiece::from_index(piece.index()) == piece
                && tetris_game::TetrisPiecePlacement::from_index(placement).piece == piece
        )
    );

    best
}

#[cfg(test)]
mod tests {
    use tetris_game::{TetrisBoard, TetrisPiece, TetrisPieceBagState};

    use crate::config::SolverConfig;
    use crate::config::{BoardAdmissibility, RootStateConfig};
    use crate::policy::verify_policy;
    use crate::state::{PackedPlacement, StateId};
    use crate::universe::UniverseBuilder;

    use super::*;

    #[test]
    fn empty_states_are_winning_with_depth_zero() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");
        let result = solve(&universe);

        for state_id in universe.states_with_empty_board() {
            if state_id == universe.root_state_id {
                continue;
            }
            assert!(result.winning[state_id as usize]);
            assert_eq!(result.depth[state_id as usize], 0);
        }
    }

    #[test]
    fn policy_verification_passes_on_small_universe() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: u32::MAX,
                max_cells: u32::MAX,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");
        let result = solve(&universe);
        let verification = verify_policy(&universe, &result);

        assert!(verification.is_clean());
    }

    #[test]
    fn root_query_is_yes_for_immediate_line_clear() {
        let mut board = TetrisBoard::new();
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 10,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig {
                board,
                bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
            },
        })
        .expect("universe should build");
        let result = solve(&universe);
        let root = evaluate_root(&universe, &result).expect("root should be winning");

        assert_eq!(root.depth, 1);
        assert!(root.best_action[TetrisPiece::I_PIECE.index() as usize].is_some());
    }

    #[test]
    fn root_query_is_no_when_no_admissible_initial_move_exists() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");
        let result = solve(&universe);

        assert!(evaluate_root(&universe, &result).is_none());
        assert!(!result.winning[universe.root_state_id as usize]);
    }

    #[test]
    fn empty_root_is_not_trivially_marked_winning() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");
        let result = solve(&universe);

        assert!(!result.winning[universe.root_state_id as usize]);
        assert!(evaluate_root(&universe, &result).is_none());
    }

    fn oracle_solve_root(universe: &Universe) -> Option<WinningEvaluation> {
        let oracle = oracle_solve_full(universe);
        evaluate_root(universe, &oracle)
    }

    fn oracle_solve_full(universe: &Universe) -> SolveResult {
        let state_count = universe.states.len();
        let mut result = SolveResult {
            winning: vec![false; state_count],
            depth: vec![0; state_count],
            best_action: vec![[None; 7]; state_count],
            pass_stats: SolvePassStats::default(),
        };

        for state_id in universe.target_states() {
            result.winning[state_id as usize] = true;
        }

        loop {
            let mut changed = false;
            for state_id in 0..state_count as StateId {
                if result.winning[state_id as usize] {
                    continue;
                }

                let Some(evaluation) =
                    oracle_evaluate_state(universe, &result.winning, &result.depth, state_id)
                else {
                    continue;
                };

                result.winning[state_id as usize] = true;
                result.depth[state_id as usize] = evaluation.depth;
                result.best_action[state_id as usize] = evaluation.best_action;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        loop {
            let mut changed = false;
            for state_id in 0..state_count as StateId {
                if !result.winning[state_id as usize] {
                    continue;
                }

                let state = universe.state_key(state_id);
                if state_id != universe.root_state_id && state.board_id == universe.empty_board_id {
                    continue;
                }

                let Some(evaluation) =
                    oracle_evaluate_state(universe, &result.winning, &result.depth, state_id)
                else {
                    continue;
                };

                if result.depth[state_id as usize] != evaluation.depth
                    || result.best_action[state_id as usize] != evaluation.best_action
                {
                    result.depth[state_id as usize] = evaluation.depth;
                    result.best_action[state_id as usize] = evaluation.best_action;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        result
    }

    fn oracle_evaluate_state(
        universe: &Universe,
        winning: &[bool],
        depth: &[u32],
        state_id: StateId,
    ) -> Option<WinningEvaluation> {
        let state = universe.state_key(state_id);
        if state_id != universe.root_state_id && state.board_id == universe.empty_board_id {
            return Some(WinningEvaluation {
                depth: 0,
                best_action: [None; 7],
            });
        }

        let state_index = universe.state_index(state_id);
        let mut best_action = [None; 7];
        let mut worst_case_depth = 0;
        for branch in piece_branches(state.bag) {
            let piece_idx = branch.piece.index() as usize;
            let range = state_index.piece_ranges[piece_idx];
            let mut best_branch: Option<(u32, PackedPlacement)> = None;

            for edge in universe.edge_slice(range) {
                if !winning[edge.succ as usize] {
                    continue;
                }
                let succ_depth = depth[edge.succ as usize];
                match best_branch {
                    Some((best_depth, best_placement))
                        if succ_depth > best_depth
                            || (succ_depth == best_depth && edge.placement >= best_placement) => {}
                    _ => best_branch = Some((succ_depth, edge.placement)),
                }
            }

            let Some((branch_depth, placement)) = best_branch else {
                return None;
            };

            best_action[piece_idx] = Some(placement);
            worst_case_depth = worst_case_depth.max(branch_depth);
        }

        Some(WinningEvaluation {
            depth: worst_case_depth + 1,
            best_action,
        })
    }

    #[test]
    fn oracle_cross_check_matches_yes_root() {
        let mut board = TetrisBoard::new();
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 10,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig {
                board,
                bag: TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE)),
            },
        })
        .expect("universe should build");
        let fast = evaluate_root(&universe, &solve(&universe));
        let oracle = oracle_solve_root(&universe);

        assert_eq!(fast.is_some(), oracle.is_some());
        assert_eq!(
            fast.map(|value| value.depth),
            oracle.map(|value| value.depth)
        );
    }

    #[test]
    fn oracle_cross_check_matches_no_root() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");
        let fast = evaluate_root(&universe, &solve(&universe));
        let oracle = oracle_solve_root(&universe);

        assert_eq!(fast.is_none(), oracle.is_none());
    }

    #[test]
    fn oracle_cross_check_matches_full_small_universe() {
        let universe = UniverseBuilder::build(SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_cells: 8,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        })
        .expect("universe should build");

        let fast = solve(&universe);
        let oracle = oracle_solve_full(&universe);

        assert_eq!(fast.winning, oracle.winning);
        assert_eq!(fast.depth, oracle.depth);
        assert_eq!(
            evaluate_root(&universe, &fast),
            evaluate_root(&universe, &oracle)
        );
    }
}
