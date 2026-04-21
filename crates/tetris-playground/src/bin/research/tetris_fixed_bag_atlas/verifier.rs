use std::time::Instant;

use tetris_game::TetrisPiecePlacement;

use crate::retrograde::SolveResult;
use crate::state::{unpack_placement, StateId};
use crate::universe::Universe;

/// Verify the retrograde solution by replaying every state.
///
/// For winning states: confirms the best_action leads to a winning successor.
/// For losing states: confirms all placements lead to losing/dead successors.
///
/// Returns `true` if the solution is consistent, `false` otherwise.
pub fn verify(universe: &Universe, result: &SolveResult) -> bool {
    let start = Instant::now();
    let n = universe.state_count();
    let mut errors = 0u64;

    for sid in 0..n {
        if result.alive[sid] {
            // Winning state: best_action must exist and lead to a winning successor
            let Some(packed) = result.best_action[sid] else {
                eprintln!("[verify] ERROR: winning state {} has no best_action", sid);
                errors += 1;
                continue;
            };

            let placement = unpack_placement(packed);
            let key = &universe.states[sid];
            let expected_piece = universe.config.cycle[key.cycle_pos as usize];

            // Verify the placement is for the correct piece
            if placement.piece != expected_piece {
                eprintln!(
                    "[verify] ERROR: state {} placement piece mismatch (got {:?}, expected {:?})",
                    sid, placement.piece, expected_piece,
                );
                errors += 1;
                continue;
            }

            // Verify the placement leads to a winning successor
            let found = universe
                .successors(sid as StateId)
                .iter()
                .any(|edge| edge.placement == packed && result.alive[edge.succ as usize]);

            if !found {
                eprintln!(
                    "[verify] ERROR: winning state {} best_action does not lead to winning successor",
                    sid,
                );
                errors += 1;
            }
        } else {
            // Losing state: all successors must be dead
            for edge in universe.successors(sid as StateId) {
                if result.alive[edge.succ as usize] {
                    eprintln!(
                        "[verify] ERROR: losing state {} has alive successor {}",
                        sid, edge.succ,
                    );
                    errors += 1;
                    break;
                }
            }
        }
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

/// Simulate playing from the root using the atlas, printing each step.
///
/// Useful for manual inspection. Plays up to `max_steps` before stopping.
pub fn replay_from_root(universe: &Universe, result: &SolveResult, max_steps: usize) {
    let mut current = universe.root_state_id;

    for step in 0..max_steps {
        if !result.alive[current as usize] {
            println!("step {}: DEAD (state {})", step, current);
            return;
        }

        let key = &universe.states[current as usize];
        let board = &universe.boards[key.board_id as usize];
        let piece = universe.config.cycle[key.cycle_pos as usize];

        let packed = result.best_action[current as usize]
            .expect("winning state must have best_action");
        let placement = unpack_placement(packed);

        println!(
            "step {:>4}: pos={} piece={:?} placement=({:?}, col={}) height={} holes={}",
            step,
            key.cycle_pos,
            piece,
            placement.orientation.rotation,
            placement.orientation.column,
            board.height(),
            board.total_holes(),
        );

        // Find the successor state
        let succ = universe
            .successors(current)
            .iter()
            .find(|edge| edge.placement == packed)
            .expect("best_action must correspond to an edge")
            .succ;

        current = succ;
    }

    println!("... (stopped after {} steps, still alive)", max_steps);
}
