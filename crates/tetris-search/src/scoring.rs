use proc_macros::inline_conditioned;
use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig};
use tetris_utils::repeat_idx_unroll;

use crate::BeamTetrisState;

#[derive(Debug, Clone, Copy)]
pub struct TetrisBoardScoreState {
    pub board: TetrisBoard,
    pub recent_lines_cleared: u32,
}

pub const HEIGHT_MSE_ROW_PENALTY_BASE: f32 = 1.5;

const fn pow_f32(base: f32, exponent: usize) -> f32 {
    let mut result = 1.0;
    let mut remaining = exponent;
    while remaining > 0 {
        result *= base;
        remaining -= 1;
    }
    result
}

/// Byte-level LUT for weighted popcount: `WPOP_LUT[byte]` = Σ (bit_i × 1.25^i) for i in 0..8.
///
/// Higher byte lanes reuse this table with a constant scale factor:
///   mid-byte score = WPOP_LUT[byte] × 1.25^8
///   hi-nibble score = WPOP_LUT[nibble] × 1.25^16
const WPOP_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut v = 0usize;
    while v < 256 {
        let mut sum = 0.0f32;
        let mut bit = 0usize;
        while bit < 8 {
            if (v >> bit) & 1 != 0 {
                sum += pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, bit);
            }
            bit += 1;
        }
        lut[v] = sum;
        v += 1;
    }
    lut
};

const WPOP_SCALE_MID: f32 = pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, 8);
const WPOP_SCALE_HI: f32 = pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, 16);

/// Compare a board against empty by summing per-cell weights `1.25^row` for every filled cell.
///
/// Uses a single byte-level LUT for branchless weighted popcount per column.
#[inline_conditioned(always)]
pub const fn height_mse_distance_from_empty(board: TetrisBoard) -> f32 {
    const PLAYABLE_MASK: u32 = (1u32 << StandardTetris::ROWS) - 1;

    let limbs = board.as_limbs();
    let mut total = 0.0f32;

    repeat_idx_unroll!(StandardTetris::COLS, COL, {
        let v = limbs[COL] & PLAYABLE_MASK;
        total += WPOP_LUT[(v & 0xFF) as usize]
            + WPOP_LUT[((v >> 8) & 0xFF) as usize] * WPOP_SCALE_MID
            + WPOP_LUT[((v >> 16) & 0xF) as usize] * WPOP_SCALE_HI;
    });
    total
}

/// Default Tetris scorer for beam search.
///
/// Beam search maximizes scores, so this returns the negative height-weighted distance from empty.
#[inline_conditioned(always)]
pub const fn height_mse_board_score(state: &TetrisBoardScoreState) -> f32 {
    if state.board.is_lost() {
        return f32::NEG_INFINITY;
    }

    -height_mse_distance_from_empty(state.board)
}

#[inline_conditioned(always)]
pub fn height_mse_beam_tetris_score(state: &BeamTetrisState) -> f32 {
    height_mse_board_score(&TetrisBoardScoreState {
        board: state.0.board,
        recent_lines_cleared: state.0.recent_lines_cleared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_height_mse_distance_from_empty_empty_board() {
        let board: TetrisBoard = TetrisBoard::new();
        assert_eq!(height_mse_distance_from_empty(board), 0.0);
    }

    #[test]
    fn test_height_mse_distance_from_empty_random_boards() {
        fn reference_score(board: TetrisBoard) -> f32 {
            let limbs = board.as_limbs();
            let mask = (1u32 << StandardTetris::ROWS) - 1;
            let mut row_diffs = [0u8; StandardTetris::ROWS];
            for col in 0..StandardTetris::COLS {
                let mut bits = limbs[col] & mask;
                while bits != 0 {
                    let row = bits.trailing_zeros() as usize;
                    row_diffs[row] += 1;
                    bits &= bits - 1;
                }
            }
            let mut total = 0.0f32;
            for row in 0..StandardTetris::ROWS {
                total += row_diffs[row] as f32 * pow_f32(HEIGHT_MSE_ROW_PENALTY_BASE, row);
            }
            total
        }

        let mut rng = rand::rng();
        for _ in 0..1000 {
            let mut board: TetrisBoard = TetrisBoard::new();
            board.set_random_bits(50, &mut rng);
            let expected = reference_score(board);
            let actual = height_mse_distance_from_empty(board);
            let tol = expected.abs() * 1e-5 + 1e-3;
            assert!(
                (expected - actual).abs() < tol,
                "mismatch: expected={expected}, actual={actual}, tol={tol}"
            );
        }
    }
}
