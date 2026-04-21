use std::time::Instant;

use crate::retrograde::SolveResult;
use crate::state::{PackedPlacement, StateId, piece_branches, unpack_placement};
use crate::universe::Universe;

/// Verify the AND-OR retrograde solution.
///
/// For winning states: confirms that for EVERY piece in the bag, `best_action`
/// exists and leads to a winning successor.
///
/// For losing states: confirms there EXISTS at least one piece in the bag where
/// ALL placements lead to dead/losing successors.
///
/// Returns `true` if the solution is consistent.
pub fn verify(universe: &Universe, result: &SolveResult) -> bool {
    let start = Instant::now();
    let n = universe.state_count();
    let mut errors = 0u64;

    let mut unexpanded_alive = 0u64;

    for sid in 0..n {
        let state_key = &universe.states[sid];
        let index = &universe.state_indices[sid];

        // Detect unexpanded states: their StateIndex has the default bag (0)
        // but the real state bag is non-empty. These were never expanded due
        // to max_states cap and are treated as "unknown" — skip verification.
        let is_unexpanded = index.bag != state_key.bag;

        if is_unexpanded {
            if result.alive[sid] {
                unexpanded_alive += 1;
            }
            continue;
        }

        if result.alive[sid] {
            // Winning state: every piece branch must have a valid best_action
            // leading to a winning successor.
            for branch in piece_branches(index.bag) {
                let pidx = branch.piece.index() as usize;

                let Some(packed) = result.best_action[sid][pidx] else {
                    eprintln!(
                        "[verify] ERROR: winning state {} has no best_action for piece idx {}",
                        sid, pidx,
                    );
                    errors += 1;
                    continue;
                };

                // Verify the placement is for the correct piece
                let placement = unpack_placement(packed);
                if placement.piece != branch.piece {
                    eprintln!(
                        "[verify] ERROR: state {} piece mismatch for idx {} (got {:?}, expected {:?})",
                        sid, pidx, placement.piece, branch.piece,
                    );
                    errors += 1;
                    continue;
                }

                // Verify the placement leads to a winning successor
                let range = index.piece_ranges[pidx];
                let found = universe
                    .edge_slice(range)
                    .iter()
                    .any(|edge| edge.placement == packed && result.alive[edge.succ as usize]);

                if !found {
                    eprintln!(
                        "[verify] ERROR: winning state {} piece idx {} best_action does not lead to winning successor",
                        sid, pidx,
                    );
                    errors += 1;
                }
            }
        } else {
            // Losing state: must have at least one piece where ALL placements
            // lead to dead/losing successors.
            let has_killing_piece = piece_branches(index.bag).any(|branch| {
                let pidx = branch.piece.index() as usize;
                let range = index.piece_ranges[pidx];
                // All edges for this piece lead to non-alive successors
                universe
                    .edge_slice(range)
                    .iter()
                    .all(|edge| !result.alive[edge.succ as usize])
            });

            if !has_killing_piece {
                eprintln!(
                    "[verify] ERROR: losing state {} has no piece that kills it",
                    sid,
                );
                errors += 1;
            }
        }
    }

    if unexpanded_alive > 0 {
        eprintln!(
            "[verify] note: {} unexpanded states treated as alive (BFS cap hit)",
            unexpanded_alive,
        );
    }

    let elapsed = start.elapsed().as_secs_f64();
    if errors == 0 {
        eprintln!("[verify] passed in {:.2}s ({} states checked)", elapsed, n);
        true
    } else {
        eprintln!(
            "[verify] FAILED in {:.2}s ({} errors in {} states)",
            elapsed, errors, n,
        );
        false
    }
}

/// Adversarial replay from the root state.
///
/// At each step the adversary picks the hardest piece (highest successor score
/// after the player's best response) and the player plays the atlas action.
pub fn replay_from_root(universe: &Universe, result: &SolveResult, max_steps: usize) {
    let mut current = universe.root_state_id;

    for step in 0..max_steps {
        if !result.alive[current as usize] {
            println!("step {}: DEAD (state {})", step, current);
            return;
        }

        let key = &universe.states[current as usize];
        let index = &universe.state_indices[current as usize];
        let board = &universe.boards[key.board_id as usize];

        // Skip unexpanded states (BFS cap)
        if index.bag != key.bag {
            println!(
                "step {:>4}: UNEXPANDED (state {}, bag={}, height={} holes={})",
                step, current, key.bag, board.height(), board.total_holes(),
            );
            return;
        }

        // Adversary picks the piece that maximizes post-move score
        let mut worst_piece_name = "?";
        let mut worst_score: u32 = 0;
        let mut worst_succ: StateId = 0;
        let mut worst_packed: PackedPlacement = 0;

        for branch in piece_branches(index.bag) {
            let pidx = branch.piece.index() as usize;
            let Some(packed) = result.best_action[current as usize][pidx] else {
                continue;
            };

            let range = index.piece_ranges[pidx];
            let Some(edge) = universe
                .edge_slice(range)
                .iter()
                .find(|e| e.placement == packed)
            else {
                continue;
            };

            let succ_key = &universe.states[edge.succ as usize];
            let succ_board = &universe.boards[succ_key.board_id as usize];
            let score = succ_board.height() * 16 + succ_board.total_holes();

            if score >= worst_score {
                worst_score = score;
                worst_succ = edge.succ;
                worst_packed = packed;
                worst_piece_name = piece_name(branch.piece);
            }
        }

        let placement = unpack_placement(worst_packed);
        println!(
            "step {:>4}: bag={} piece={} placement=({:?}, col={}) height={} holes={}",
            step,
            key.bag,
            worst_piece_name,
            placement.orientation.rotation,
            placement.orientation.column,
            board.height(),
            board.total_holes(),
        );

        current = worst_succ;
    }

    println!("... (stopped after {} steps, still alive)", max_steps);
}

fn piece_name(piece: tetris_game::TetrisPiece) -> &'static str {
    use tetris_game::TetrisPiece;
    match piece {
        p if p == TetrisPiece::O_PIECE => "O",
        p if p == TetrisPiece::I_PIECE => "I",
        p if p == TetrisPiece::S_PIECE => "S",
        p if p == TetrisPiece::Z_PIECE => "Z",
        p if p == TetrisPiece::T_PIECE => "T",
        p if p == TetrisPiece::L_PIECE => "L",
        p if p == TetrisPiece::J_PIECE => "J",
        _ => "?",
    }
}
