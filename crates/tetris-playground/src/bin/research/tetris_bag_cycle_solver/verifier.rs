use rustc_hash::FxHashSet;
use serde::Serialize;
use tetris_game::{IsLost, TetrisPieceBagState, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::solver::WitnessPolicy;
use crate::state::{BoardCache, BoardId, WitnessKey};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PolicyVerification {
    pub states_checked: usize,
    pub piece_branches_checked: usize,
    pub terminal_branches_checked: usize,
    pub terminal_target_hits: usize,
    pub witness_failures: usize,
    pub replay_failures: usize,
    pub target_failures: usize,
    pub missing_child_failures: usize,
}

impl PolicyVerification {
    #[inline]
    pub const fn is_clean(self) -> bool {
        self.witness_failures == 0
            && self.replay_failures == 0
            && self.target_failures == 0
            && self.missing_child_failures == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReplayState {
    board_id: BoardId,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
}

pub fn verify_witness_policy(
    config: SolverConfig,
    boards: &BoardCache,
    policy: &WitnessPolicy,
    root_board_id: BoardId,
    root_bag: TetrisPieceBagState,
    root_cycles: u32,
) -> PolicyVerification {
    let mut verification = PolicyVerification::default();
    let mut stack = vec![ReplayState {
        board_id: root_board_id,
        bag: root_bag,
        cycles_remaining: root_cycles,
    }];
    let mut seen = FxHashSet::default();

    while let Some(state) = stack.pop() {
        if !seen.insert(state) {
            continue;
        }

        verification.states_checked += 1;
        if state.cycles_remaining == 0 {
            let Some(board) = boards.boards.get(state.board_id as usize) else {
                verification.missing_child_failures += 1;
                continue;
            };
            if !config.target_contains(board) {
                verification.target_failures += 1;
            }
            continue;
        }

        let Some(&board) = boards.boards.get(state.board_id as usize) else {
            verification.missing_child_failures += 1;
            continue;
        };

        for (piece, next_bag) in state.bag.iter_next_states() {
            verification.piece_branches_checked += 1;
            let key = WitnessKey::new(state.board_id, state.bag, state.cycles_remaining, piece);
            let Some(witness) = policy.get(&key) else {
                verification.witness_failures += 1;
                continue;
            };
            if witness.placement_index as usize >= TetrisPiecePlacement::NUM_PLACEMENTS {
                verification.witness_failures += 1;
                continue;
            }

            let placement = TetrisPiecePlacement::from_index(witness.placement_index);
            if placement.piece != piece {
                verification.witness_failures += 1;
                continue;
            }

            let mut next_board = board;
            let result = next_board.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST || !config.admissibility.contains(&next_board) {
                verification.replay_failures += 1;
                continue;
            }

            if state.bag.count() == 1 && state.cycles_remaining <= 1 {
                verification.terminal_branches_checked += 1;
                if !witness.terminal || witness.child_board_id.is_some() {
                    verification.witness_failures += 1;
                    continue;
                }
                if config.target_contains(&next_board) {
                    verification.terminal_target_hits += 1;
                } else {
                    verification.target_failures += 1;
                }
                continue;
            }

            if witness.terminal {
                verification.witness_failures += 1;
                continue;
            }
            let Some(actual_child_id) = boards.get_id(&next_board) else {
                verification.missing_child_failures += 1;
                continue;
            };
            if witness.child_board_id != Some(actual_child_id) {
                verification.missing_child_failures += 1;
                continue;
            }

            let child_cycles = if state.bag.count() == 1 {
                state.cycles_remaining.saturating_sub(1)
            } else {
                state.cycles_remaining
            };
            stack.push(ReplayState {
                board_id: actual_child_id,
                bag: next_bag,
                cycles_remaining: child_cycles,
            });
        }
    }

    verification
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use tetris_game::{TetrisBoard, TetrisPiece};

    use crate::config::{BoardAdmissibility, TargetMode};
    use crate::solver::WitnessEntry;

    use super::*;

    fn single_i_clear_board() -> TetrisBoard {
        let mut board = TetrisBoard::new();
        for col in 4..10 {
            board.set_bit(col, 0);
        }
        board
    }

    fn single_i_bag() -> TetrisPieceBagState {
        TetrisPieceBagState::from(u8::from(TetrisPiece::I_PIECE))
    }

    fn single_i_horizontal_left() -> TetrisPiecePlacement {
        TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)[0]
    }

    fn empty_target_config() -> SolverConfig {
        SolverConfig {
            bag_cycles: 1,
            admissibility: BoardAdmissibility {
                max_height: 1,
                max_holes: 0,
                max_roughness: u32::MAX,
                max_count: 6,
            },
            target_mode: TargetMode::Empty,
            closed_set_admissibility: BoardAdmissibility::default(),
        }
    }

    #[test]
    fn verifier_accepts_terminal_empty_reset_witness() {
        let config = empty_target_config();
        let mut boards = BoardCache::new();
        let root_id = boards.get_or_insert(single_i_clear_board());
        let bag = single_i_bag();
        let placement = single_i_horizontal_left();
        let mut policy = FxHashMap::default();
        policy.insert(
            WitnessKey::new(root_id, bag, 1, TetrisPiece::I_PIECE),
            WitnessEntry {
                placement_index: placement.index(),
                child_board_id: None,
                terminal: true,
            },
        );

        let verification = verify_witness_policy(config, &boards, &policy, root_id, bag, 1);

        assert!(verification.is_clean());
        assert_eq!(verification.terminal_branches_checked, 1);
        assert_eq!(verification.terminal_target_hits, 1);
    }

    #[test]
    fn verifier_rejects_witness_for_wrong_piece() {
        let config = empty_target_config();
        let mut boards = BoardCache::new();
        let root_id = boards.get_or_insert(single_i_clear_board());
        let bag = single_i_bag();
        let bad_placement = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        let mut policy = FxHashMap::default();
        policy.insert(
            WitnessKey::new(root_id, bag, 1, TetrisPiece::I_PIECE),
            WitnessEntry {
                placement_index: bad_placement.index(),
                child_board_id: None,
                terminal: true,
            },
        );

        let verification = verify_witness_policy(config, &boards, &policy, root_id, bag, 1);

        assert!(!verification.is_clean());
        assert_eq!(verification.witness_failures, 1);
    }
}
