use rustc_hash::FxHashMap;
use tetris_game::{TetrisBoard, TetrisPieceBagState};

use crate::config::SolverConfig;
use crate::solver::{
    RootDiagnostics, RootPieceDiagnostic, RootPieceStatus, RootProgress, SolveResult, SolveStats,
    WitnessEntry,
};
use crate::state::{BoardCache, BoardId, WitnessKey};
use crate::transition::TransitionCache;

type StateId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TimedStateKey {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    plies_remaining: u32,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    placement_index: u8,
    succ: StateId,
}

#[derive(Debug, Default)]
struct StateEdges {
    piece_edges: [Vec<Edge>; 7],
}

impl StateEdges {
    fn new() -> Self {
        Self {
            piece_edges: std::array::from_fn(|_| Vec::new()),
        }
    }
}

pub fn solve_with_retrograde(config: SolverConfig) -> SolveResult {
    solve_state_with_retrograde(
        config,
        TetrisBoard::EMPTY_BOARD,
        TetrisPieceBagState::new(),
        config.total_pieces(),
    )
}

pub fn solve_state_with_retrograde(
    config: SolverConfig,
    initial_board: TetrisBoard,
    initial_bag: TetrisPieceBagState,
    plies_remaining: u32,
) -> SolveResult {
    let mut solver = RetrogradeSolver::new(config);
    solver.solve(initial_board, initial_bag, plies_remaining)
}

struct RetrogradeSolver {
    config: SolverConfig,
    boards: BoardCache,
    state_to_id: FxHashMap<TimedStateKey, StateId>,
    states: Vec<TimedStateKey>,
    edges: Vec<StateEdges>,
    transitions: TransitionCache,
}

impl RetrogradeSolver {
    fn new(config: SolverConfig) -> Self {
        Self {
            config,
            boards: BoardCache::new(),
            state_to_id: FxHashMap::default(),
            states: Vec::new(),
            edges: Vec::new(),
            transitions: TransitionCache::new(config.admissibility),
        }
    }

    fn solve(
        mut self,
        initial_board: TetrisBoard,
        initial_bag: TetrisPieceBagState,
        plies_remaining: u32,
    ) -> SolveResult {
        let root_board_id = self.boards.get_or_insert(initial_board);
        let root_state = TimedStateKey {
            board_id: root_board_id,
            bag: initial_bag,
            plies_remaining,
        };
        let root_state_id = self.get_or_insert_state(root_state);
        self.expand_universe();

        let mut winning = vec![false; self.states.len()];
        let mut depth = vec![u32::MAX; self.states.len()];
        let mut policy = FxHashMap::default();
        let mut target_hits = 0;
        let mut root_progress = RootProgress::default();

        let mut order: Vec<StateId> = (0..self.states.len() as StateId).collect();
        order.sort_unstable_by_key(|&id| self.states[id as usize].plies_remaining);

        for state_id in order {
            let state = self.states[state_id as usize];
            let board = self.boards.boards[state.board_id as usize];
            if state.plies_remaining == 0 {
                if self.config.target_contains(&board) {
                    winning[state_id as usize] = true;
                    depth[state_id as usize] = 0;
                }
                continue;
            }

            let mut witnesses = Vec::new();
            let mut worst_child_depth = 0;
            let mut state_is_winning = true;

            for piece in state.bag.iter_pieces() {
                let piece_edges =
                    &self.edges[state_id as usize].piece_edges[piece.index() as usize];
                let best = piece_edges
                    .iter()
                    .filter(|edge| winning[edge.succ as usize])
                    .min_by_key(|edge| depth[edge.succ as usize])
                    .copied();

                let Some(best) = best else {
                    state_is_winning = false;
                    break;
                };

                let child_state = self.states[best.succ as usize];
                let terminal = child_state.plies_remaining == 0;
                witnesses.push((
                    piece,
                    WitnessEntry {
                        placement_index: best.placement_index,
                        child_board_id: (!terminal).then_some(child_state.board_id),
                        terminal,
                    },
                ));
                worst_child_depth = worst_child_depth.max(depth[best.succ as usize]);
            }

            if state_is_winning {
                winning[state_id as usize] = true;
                depth[state_id as usize] = worst_child_depth.saturating_add(1);
                if let Some(cycles_remaining) =
                    cycles_remaining_from_plies(state.plies_remaining, state.bag)
                {
                    for (piece, witness) in witnesses {
                        if witness.terminal {
                            target_hits += 1;
                        }
                        policy.insert(
                            WitnessKey::new(state.board_id, state.bag, cycles_remaining, piece),
                            witness,
                        );
                    }
                }
                if state_id == root_state_id {
                    for piece in state.bag.iter_pieces() {
                        root_progress.statuses[piece.index() as usize] = RootPieceStatus::Proven;
                    }
                }
            } else if state_id == root_state_id {
                for piece in state.bag.iter_pieces() {
                    let piece_edges =
                        &self.edges[state_id as usize].piece_edges[piece.index() as usize];
                    let piece_is_proven =
                        piece_edges.iter().any(|edge| winning[edge.succ as usize]);
                    root_progress.statuses[piece.index() as usize] = if piece_is_proven {
                        RootPieceStatus::Proven
                    } else {
                        RootPieceStatus::Failed
                    };
                }
            }
        }

        let mut stats = SolveStats {
            nodes_explored: self.states.len() as u64,
            target_hits,
            deepest_ply_reached: plies_remaining,
            total_plies: plies_remaining,
            retrograde_states: self.states.len(),
            retrograde_edges: self
                .edges
                .iter()
                .flat_map(|state_edges| state_edges.piece_edges.iter())
                .map(Vec::len)
                .sum(),
            retrograde_winning_states: winning.iter().filter(|&&is_winning| is_winning).count(),
            transition_cache_entries: self.transitions.len(),
            transition: self.transitions.stats(),
            ..SolveStats::default()
        };
        stats.transition_cache_entries = self.transitions.len();

        let root_diagnostics = Some(root_diagnostics_from_policy(
            root_board_id,
            initial_bag,
            plies_remaining,
            &policy,
        ));

        SolveResult {
            winning: winning[root_state_id as usize],
            stats,
            boards: self.boards,
            memo_entries: self.state_to_id.len(),
            policy,
            root_diagnostics,
            root_progress,
        }
    }

    fn expand_universe(&mut self) {
        let mut cursor = 0;
        while cursor < self.states.len() {
            let state = self.states[cursor];
            if state.plies_remaining == 0 {
                cursor += 1;
                continue;
            }

            let board = self.boards.boards[state.board_id as usize];
            for (piece, next_bag) in state.bag.iter_next_states() {
                let successors = self.transitions.successors(board, piece).to_vec();
                for transition in successors {
                    let child_board_id = self.boards.get_or_insert(transition.board);
                    let child_key = TimedStateKey {
                        board_id: child_board_id,
                        bag: next_bag,
                        plies_remaining: state.plies_remaining - 1,
                    };
                    let child_state_id = self.get_or_insert_state(child_key);
                    self.edges[cursor].piece_edges[piece.index() as usize].push(Edge {
                        placement_index: transition.placement.index(),
                        succ: child_state_id,
                    });
                }
            }

            cursor += 1;
        }
    }

    fn get_or_insert_state(&mut self, key: TimedStateKey) -> StateId {
        if let Some(&id) = self.state_to_id.get(&key) {
            return id;
        }

        let id = self.states.len() as StateId;
        self.state_to_id.insert(key, id);
        self.states.push(key);
        self.edges.push(StateEdges::new());
        id
    }
}

fn cycles_remaining_from_plies(plies_remaining: u32, bag: TetrisPieceBagState) -> Option<u32> {
    if plies_remaining == 0 {
        return Some(0);
    }

    let pieces_in_bag = bag.count() as u32;
    if pieces_in_bag == 0 || plies_remaining < pieces_in_bag {
        return None;
    }
    let completed_full_bags = plies_remaining - pieces_in_bag;
    if completed_full_bags % 7 != 0 {
        return None;
    }
    Some(completed_full_bags / 7 + 1)
}

fn root_diagnostics_from_policy(
    root_board_id: BoardId,
    root_bag: TetrisPieceBagState,
    root_plies_remaining: u32,
    policy: &FxHashMap<WitnessKey, WitnessEntry>,
) -> RootDiagnostics {
    let cycles_remaining = cycles_remaining_from_plies(root_plies_remaining, root_bag).unwrap_or(0);
    let mut pieces = Vec::new();
    let mut proven_pieces = 0;

    for piece in root_bag.iter_pieces() {
        let witness = policy.get(&WitnessKey::new(
            root_board_id,
            root_bag,
            cycles_remaining,
            piece,
        ));
        if witness.is_some() {
            proven_pieces += 1;
        }
        pieces.push(RootPieceDiagnostic {
            piece_index: piece.index(),
            piece_name: piece.to_string(),
            proven: witness.is_some(),
            placement_index: witness.map(|entry| entry.placement_index),
        });
    }

    let total_pieces = pieces.len();
    let piece_coverage = if total_pieces == 0 {
        1.0
    } else {
        proven_pieces as f64 / total_pieces as f64
    };

    RootDiagnostics {
        proven_pieces,
        total_pieces,
        piece_coverage,
        pieces,
    }
}

#[cfg(test)]
mod tests {
    use tetris_game::{TetrisPiece, TetrisPiecePlacement};

    use crate::config::{BoardAdmissibility, TargetMode};
    use crate::verifier::verify_witness_policy;

    use super::*;

    fn single_i_clear_board() -> TetrisBoard {
        let mut board = TetrisBoard::EMPTY_BOARD;
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        board
    }

    fn single_i_bag() -> TetrisPieceBagState {
        TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE))
    }

    #[test]
    fn cycles_remaining_from_plies_tracks_partial_bag_state() {
        assert_eq!(
            cycles_remaining_from_plies(35, TetrisPieceBagState::new()),
            Some(5)
        );
        assert_eq!(cycles_remaining_from_plies(1, single_i_bag()), Some(1));
        assert_eq!(cycles_remaining_from_plies(2, single_i_bag()), None);
    }

    #[test]
    fn retrograde_solves_one_piece_empty_reset() {
        let config = SolverConfig {
            bag_cycles: 1,
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_roughness: u32::MAX,
                max_count: 6,
            },
            target_mode: TargetMode::Empty,
            closed_set_admissibility: BoardAdmissibility::default(),
        };

        let result = solve_state_with_retrograde(config, single_i_clear_board(), single_i_bag(), 1);

        assert!(result.winning);
        let verification =
            verify_witness_policy(config, &result.boards, &result.policy, 0, single_i_bag(), 1);
        assert!(verification.is_clean());
    }

    #[test]
    fn expected_one_piece_witness_is_legal() {
        let placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)[0];
        let mut board = single_i_clear_board();
        let result = board.apply_piece_placement(placement);

        assert_eq!(result.lines_cleared, 1);
        assert_eq!(board, TetrisBoard::EMPTY_BOARD);
    }
}
