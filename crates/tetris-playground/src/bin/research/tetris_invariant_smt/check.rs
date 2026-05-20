use std::time::Instant;

use tracing::{debug, info, warn};
use z3::ast::{Ast, BV, Bool};
use z3::{SatResult, Solver};

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::config::*;
use crate::encoding::*;

/// Result of checking one piece type for inductiveness.
pub enum PieceCheckResult {
    /// All boards satisfying P have at least one safe placement for this piece.
    Safe,
    /// Found a board satisfying P where this piece has no safe placement.
    Counterexample { board: TetrisBoard },
    /// Solver returned unknown.
    Unknown,
}

/// Check whether the invariant is inductive for a single piece type.
///
/// Query: does there exist a board satisfying P(board) such that for EVERY
/// placement of the given piece, the result violates P or causes loss?
///
/// UNSAT → piece is safe (every board has at least one good placement).
/// SAT → counterexample board found.
pub fn check_piece_inductive(
    piece: TetrisPiece,
    all_placements: &[PlacementInfo],
) -> PieceCheckResult {
    let solver = Solver::new();
    let board = make_symbolic_board("b");

    // Assert: board satisfies the invariant P
    solver.assert(encode_invariant(&board));

    // Get all placements for this piece type
    let piece_placements: Vec<&PlacementInfo> =
        all_placements.iter().filter(|p| p.piece == piece).collect();

    // For each placement: encode the transition and assert it fails.
    // "fails" = result does NOT satisfy P, OR the placement causes loss.
    let mut fail_constraints: Vec<Bool> = Vec::new();
    for info in &piece_placements {
        let enc = encode_placement(&board, info);
        let result_ok = encode_invariant(&enc.result_board);
        // This placement fails if: result violates P OR placement causes loss
        let fails = Bool::or(&[result_ok.not(), enc.is_lost]);
        fail_constraints.push(fails);
    }

    // Assert ALL placements fail (conjunction):
    // If there's a board where every placement fails, the invariant isn't inductive for this piece.
    solver.assert(Bool::and(&fail_constraints));

    match solver.check() {
        SatResult::Unsat => PieceCheckResult::Safe,
        SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let mut tb = TetrisBoard::EMPTY_BOARD;
            for (i, col_bv) in board.iter().enumerate() {
                if let Some(val) = model.eval(col_bv, true) {
                    if let Some(bits) = val.as_u64() {
                        for r in 0..BOARD_ROWS {
                            if bits & (1u64 << r) != 0 {
                                tb.set_bit(i, r as usize);
                            }
                        }
                    }
                }
            }
            PieceCheckResult::Counterexample { board: tb }
        }
        SatResult::Unknown => PieceCheckResult::Unknown,
    }
}

/// Run the full inductiveness check across all 7 piece types.
/// Returns true if the invariant is inductive (all pieces safe).
pub fn check_all_pieces(all_placements: &[PlacementInfo]) -> bool {
    let pieces = TetrisPiece::all();
    let mut all_safe = true;

    for &piece in &pieces {
        let piece_count = all_placements.iter().filter(|p| p.piece == piece).count();
        info!("checking {:?} ({} placements)...", piece, piece_count);

        let start = Instant::now();
        let result = check_piece_inductive(piece, all_placements);
        let elapsed = start.elapsed().as_secs_f64();

        match result {
            PieceCheckResult::Safe => {
                info!("  {:?} → SAFE ({:.3}s)", piece, elapsed);
            }
            PieceCheckResult::Counterexample { board } => {
                warn!("  {:?} → COUNTEREXAMPLE ({:.3}s)", piece, elapsed);
                print_board_counterexample(piece, &board, all_placements);
                all_safe = false;
            }
            PieceCheckResult::Unknown => {
                warn!("  {:?} → UNKNOWN ({:.3}s)", piece, elapsed);
                all_safe = false;
            }
        }
    }

    all_safe
}

fn print_board_counterexample(
    piece: TetrisPiece,
    board: &TetrisBoard,
    all_placements: &[PlacementInfo],
) {
    // Print the board
    let heights: Vec<u32> = (0..BOARD_COLS)
        .map(|c| {
            let mut h = 0u32;
            for r in 0..BOARD_ROWS {
                if board.get_bit(c as usize, r as usize) {
                    h = r + 1;
                }
            }
            h
        })
        .collect();

    let holes: Vec<u32> = (0..BOARD_COLS)
        .map(|c| {
            let h = heights[c as usize];
            let filled = (0..h)
                .filter(|&r| board.get_bit(c as usize, r as usize))
                .count() as u32;
            h - filled
        })
        .collect();
    let total_holes: u32 = holes.iter().sum();
    let max_roughness = heights
        .windows(2)
        .map(|w| w[0].abs_diff(w[1]))
        .max()
        .unwrap_or(0);

    warn!("    board heights:  {:?}", heights);
    warn!("    per-col holes:  {:?}", holes);
    warn!(
        "    total_holes={}, max_roughness={}",
        total_holes, max_roughness
    );

    // Print the board visually (top rows first)
    let max_h = *heights.iter().max().unwrap_or(&0);
    for r in (0..max_h).rev() {
        let mut row_str = String::from("    ");
        for c in 0..BOARD_COLS {
            if board.get_bit(c as usize, r as usize) {
                row_str.push('#');
            } else {
                row_str.push('.');
            }
        }
        warn!("{}", row_str);
    }

    // Try all placements and show why each fails
    let piece_placements: Vec<&PlacementInfo> =
        all_placements.iter().filter(|p| p.piece == piece).collect();
    debug!("    placement results:");
    for info in piece_placements.iter().take(5) {
        let mut b = *board;
        let placement = TetrisPiecePlacement::from_index(info.index);
        let pr = b.apply_piece_placement(placement);
        if bool::from(pr.is_lost) {
            debug!("      placement {} → LOST", info.index);
        } else {
            let ok = board_satisfies_invariant(&b);
            let h = b.height();
            debug!(
                "      placement {} → height={}, invariant={}, lines={}",
                info.index, h, ok, pr.lines_cleared
            );
        }
    }
    if piece_placements.len() > 5 {
        debug!("      ... and {} more", piece_placements.len() - 5);
    }
}
