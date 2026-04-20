use std::collections::{HashMap, HashSet};

use serde::Serialize;
use tetris_game::{IsLost, TetrisPiecePlacement};

use crate::config::SolverConfig;
use crate::proof::{BagKey, BagNodeId, NodeResolution, PieceNode, ProofSolveResult};
use crate::state::{next_bag_state, piece_branches};
use crate::universe::board_is_admissible;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct ReplayVerification {
    pub won_bags_checked: usize,
    pub won_pieces_checked: usize,
    pub lost_pieces_checked: usize,
    pub witness_replays: usize,
    pub lost_piece_placements_checked: usize,
    pub replay_failures: usize,
    pub missing_node_failures: usize,
    pub bag_transition_failures: usize,
    pub resolution_failures: usize,
}

impl ReplayVerification {
    pub const fn is_clean(self) -> bool {
        self.replay_failures == 0
            && self.missing_node_failures == 0
            && self.bag_transition_failures == 0
            && self.resolution_failures == 0
    }
}

pub fn verify_replay(config: SolverConfig, result: &ProofSolveResult) -> ReplayVerification {
    let mut verification = ReplayVerification::default();
    let mut bag_by_key = HashMap::<BagKey, BagNodeId>::with_capacity(result.bag_nodes.len());
    let mut won_bag_keys = HashSet::<BagKey>::new();

    for (bag_id, bag) in result.bag_nodes.iter().enumerate() {
        if bag_by_key.insert(bag.key, bag_id as BagNodeId).is_some() {
            verification.resolution_failures += 1;
        }
        if bag.resolution == NodeResolution::Won {
            won_bag_keys.insert(bag.key);
        }
    }

    for (bag_id, bag) in result.bag_nodes.iter().enumerate() {
        if bag.resolution != NodeResolution::Won {
            continue;
        }
        verification.won_bags_checked += 1;
        for branch in piece_branches(bag.key.bag) {
            let Some(piece_id) = bag.piece_nodes[branch.piece.index() as usize] else {
                verification.missing_node_failures += 1;
                continue;
            };
            let Some(piece) = result.piece_nodes.get(piece_id as usize) else {
                verification.missing_node_failures += 1;
                continue;
            };
            if piece.resolution != NodeResolution::Won
                || piece.parent_bag as usize != bag_id
                || piece.key.board != bag.key.board
                || piece.key.bag != bag.key.bag
                || piece.key.piece != branch.piece
            {
                verification.resolution_failures += 1;
                continue;
            }

            verification.won_pieces_checked += 1;
            verify_winning_piece(
                config,
                result,
                &bag_by_key,
                bag.key,
                piece,
                &mut verification,
            );
        }
    }

    for piece in &result.piece_nodes {
        if piece.resolution != NodeResolution::Lost {
            continue;
        }
        verification.lost_pieces_checked += 1;
        verify_lost_piece(config, &won_bag_keys, piece, &mut verification);
    }

    verification
}

fn verify_winning_piece(
    config: SolverConfig,
    result: &ProofSolveResult,
    bag_key: &HashMap<BagKey, BagNodeId>,
    parent_key: BagKey,
    piece: &PieceNode,
    verification: &mut ReplayVerification,
) {
    let Some(best_action) = piece.best_action else {
        verification.missing_node_failures += 1;
        return;
    };
    if best_action as usize >= TetrisPiecePlacement::NUM_PLACEMENTS {
        verification.replay_failures += 1;
        return;
    }
    let placement = TetrisPiecePlacement::from_index(best_action);
    if placement.piece != piece.key.piece {
        verification.replay_failures += 1;
        return;
    }
    let Some(expected_next_bag) = next_bag_state(parent_key.bag, piece.key.piece) else {
        verification.bag_transition_failures += 1;
        return;
    };

    let mut replayed_board = parent_key.board;
    let outcome = replayed_board.apply_piece_placement(placement);
    verification.witness_replays += 1;
    if outcome.is_lost == IsLost::LOST || !board_is_admissible(replayed_board, config.admissibility)
    {
        verification.replay_failures += 1;
        return;
    }

    let expected_key = BagKey {
        board: replayed_board,
        bag: expected_next_bag,
    };

    if let Some(best_child) = piece.best_child {
        let Some(child) = result.bag_nodes.get(best_child as usize) else {
            verification.missing_node_failures += 1;
            return;
        };
        if child.key.board != expected_key.board {
            verification.replay_failures += 1;
            return;
        }
        if child.key.bag != expected_key.bag {
            verification.bag_transition_failures += 1;
            return;
        }
        if child.resolution != NodeResolution::Won {
            verification.resolution_failures += 1;
            return;
        }
        return;
    }

    let Some(&target_id) = bag_key.get(&expected_key) else {
        verification.missing_node_failures += 1;
        return;
    };
    if result.bag_nodes[target_id as usize].resolution != NodeResolution::Won {
        verification.resolution_failures += 1;
    }
}

fn verify_lost_piece(
    config: SolverConfig,
    won_bag_keys: &HashSet<BagKey>,
    piece: &PieceNode,
    verification: &mut ReplayVerification,
) {
    let Some(next_bag) = next_bag_state(piece.key.bag, piece.key.piece) else {
        verification.bag_transition_failures += 1;
        return;
    };

    for &placement in TetrisPiecePlacement::all_from_piece(piece.key.piece) {
        let mut next_board = piece.key.board;
        let outcome = next_board.apply_piece_placement(placement);
        if outcome.is_lost == IsLost::LOST || !board_is_admissible(next_board, config.admissibility)
        {
            continue;
        }
        verification.lost_piece_placements_checked += 1;
        if won_bag_keys.contains(&BagKey {
            board: next_board,
            bag: next_bag,
        }) {
            verification.resolution_failures += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{BoardAdmissibility, RootStateConfig};
    use crate::proof::solve_root;

    use super::*;

    fn no_move_config() -> SolverConfig {
        SolverConfig {
            admissibility: BoardAdmissibility {
                max_height: 0,
                max_holes: 0,
                max_cells: 0,
                max_roughness: u32::MAX,
                max_height_spread: u32::MAX,
            },
            root: RootStateConfig::default(),
        }
    }

    #[test]
    fn replay_verifier_accepts_small_no_proof() {
        let config = no_move_config();
        let result = solve_root(config).expect("solver should produce a proof result");
        let verification = verify_replay(config, &result);

        assert!(verification.is_clean());
        assert!(verification.lost_pieces_checked > 0);
    }
}
