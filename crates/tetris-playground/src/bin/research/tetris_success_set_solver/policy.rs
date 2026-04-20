use crate::retrograde::{SolveResult, evaluate_state};
use crate::state::{StateId, unpack_placement};
use crate::universe::{Universe, edge_contains_placement};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PolicyVerification {
    pub states_checked: usize,
    pub witness_failures: usize,
    pub depth_failures: usize,
}

impl PolicyVerification {
    pub const fn is_clean(self) -> bool {
        self.witness_failures == 0 && self.depth_failures == 0
    }
}

pub fn verify_policy(universe: &Universe, result: &SolveResult) -> PolicyVerification {
    let mut verification = PolicyVerification::default();

    for state_id in 0..universe.states.len() as StateId {
        if !result.winning[state_id as usize] {
            continue;
        }

        verification.states_checked += 1;

        let state = universe.state_key(state_id);
        let state_index = universe.state_index(state_id);
        if state_id != universe.root_state_id && state.board_id == universe.empty_board_id {
            if result.depth[state_id as usize] != 0 {
                verification.depth_failures += 1;
            }
            continue;
        }

        for piece in state.bag.iter_pieces() {
            let piece_idx = piece.index() as usize;
            let Some(placement) = result.best_action[state_id as usize][piece_idx] else {
                verification.witness_failures += 1;
                continue;
            };

            if unpack_placement(placement).piece != piece {
                verification.witness_failures += 1;
                continue;
            }

            let range = state_index.piece_ranges[piece_idx];
            if !edge_contains_placement(universe.edge_slice(range), placement) {
                verification.witness_failures += 1;
                continue;
            }

            let witness_is_winning = universe
                .edge_slice(range)
                .iter()
                .filter(|edge| edge.placement == placement)
                .any(|edge| result.winning[edge.succ as usize]);
            if !witness_is_winning {
                verification.witness_failures += 1;
            }
        }

        let Some(expected) = evaluate_state(universe, result, state_id) else {
            verification.depth_failures += 1;
            continue;
        };

        if expected.depth != result.depth[state_id as usize] {
            verification.depth_failures += 1;
        }
    }

    verification
}
