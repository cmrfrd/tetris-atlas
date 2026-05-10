use z3::ast::{Ast, BV, Bool};

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::config::*;

// ---------------------------------------------------------------------------
// Placement precomputation (from game engine)
// ---------------------------------------------------------------------------

/// Precomputed data for one concrete placement (piece + rotation + column).
#[derive(Debug, Clone)]
pub struct PlacementInfo {
    pub piece: TetrisPiece,
    pub index: u8,
    pub start_col: usize,
    pub width: usize,
    pub piece_consts: Vec<u32>,
    pub trailing: Vec<u32>,
}

/// Extract placement data from the game engine.
pub fn precompute_placements() -> Vec<PlacementInfo> {
    let mut result = Vec::new();
    for &placement in &TetrisPiecePlacement::ALL_PLACEMENTS {
        let mut board = TetrisBoard::new();
        let pr = board.apply_piece_placement(placement);
        if bool::from(pr.is_lost) {
            continue;
        }

        let limbs = board.as_limbs();
        let start_col = limbs.iter().position(|&c| c != 0).unwrap_or(0);
        let end_col = limbs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
        let width = end_col - start_col;
        let piece_cols = &limbs[start_col..end_col];

        let max_placed_height = piece_cols
            .iter()
            .map(|col: &u32| u32::BITS - col.leading_zeros())
            .max()
            .unwrap_or(0);
        let shift_on_empty = u32::BITS - max_placed_height;

        let piece_consts: Vec<u32> = piece_cols
            .iter()
            .map(|&col: &u32| col << shift_on_empty)
            .collect();

        let trailing: Vec<u32> = piece_consts
            .iter()
            .map(|c: &u32| c.trailing_zeros())
            .collect();

        result.push(PlacementInfo {
            piece: placement.piece,
            index: placement.index(),
            start_col,
            width,
            piece_consts,
            trailing,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Symbolic board construction
// ---------------------------------------------------------------------------

pub fn make_symbolic_board(prefix: &str) -> Vec<BV> {
    (0..BOARD_COLS)
        .map(|i| BV::new_const(format!("{prefix}_c{i}"), BV_WIDTH))
        .collect()
}

// ---------------------------------------------------------------------------
// Symbolic primitives
// ---------------------------------------------------------------------------

fn test_bit(val: &BV, r: u32) -> Bool {
    let mask = BV::from_u64(1u64 << r, BV_WIDTH);
    let zero = BV::from_u64(0, BV_WIDTH);
    val.bvand(&mask).eq(&zero).not()
}

pub fn col_height(col: &BV) -> BV {
    let mut h = BV::from_u64(0, BV_WIDTH);
    for r in 0..BV_WIDTH {
        let has_bit = test_bit(col, r);
        let height_val = BV::from_u64((r + 1) as u64, BV_WIDTH);
        h = has_bit.ite(&height_val, &h);
    }
    h
}

fn bv_abs_diff(a: &BV, b: &BV) -> BV {
    let a_ge_b = a.bvuge(b);
    a_ge_b.ite(&a.bvsub(b), &b.bvsub(a))
}

fn ctz(val: &BV) -> BV {
    let mut result = BV::from_u64(BV_WIDTH as u64, BV_WIDTH);
    for i in (0..BV_WIDTH).rev() {
        let has_bit = test_bit(val, i);
        let pos = BV::from_u64(i as u64, BV_WIDTH);
        result = has_bit.ite(&pos, &result);
    }
    result
}

pub fn symbolic_popcount(val: &BV, max_bits: u32) -> BV {
    let mut count = BV::from_u64(0, BV_WIDTH);
    let one = BV::from_u64(1, BV_WIDTH);
    let zero = BV::from_u64(0, BV_WIDTH);
    for r in 0..max_bits {
        let bit = test_bit(val, r);
        count = count.bvadd(bit.ite(&one, &zero));
    }
    count
}

// ---------------------------------------------------------------------------
// Placement encoding
// ---------------------------------------------------------------------------

pub struct EncodedPlacement {
    pub result_board: Vec<BV>,
    pub is_lost: Bool,
}

pub fn encode_placement(board: &[BV], info: &PlacementInfo) -> EncodedPlacement {
    let mut shift = BV::from_u64(u32::MAX as u64, BV_WIDTH);
    for i in 0..info.width {
        let board_col = &board[info.start_col + i];
        let h = col_height(board_col);
        let trailing_bv = BV::from_u64(info.trailing[i] as u64, BV_WIDTH);
        let diff = bv_abs_diff(&h, &trailing_bv);
        let lt = diff.bvult(&shift);
        shift = lt.ite(&diff, &shift);
    }

    let mut new_board: Vec<BV> = board.to_vec();
    for i in 0..info.width {
        let piece_bv = BV::from_u64(info.piece_consts[i] as u64, BV_WIDTH);
        let shifted = piece_bv.bvlshr(&shift);
        new_board[info.start_col + i] = new_board[info.start_col + i].bvor(&shifted);
    }

    let cleared = encode_line_clearing(&new_board);

    let loss_mask = BV::from_u64(u64::MAX << BOARD_ROWS, BV_WIDTH);
    let bv_zero = BV::from_u64(0, BV_WIDTH);
    let loss_flags: Vec<Bool> = cleared
        .iter()
        .map(|col| col.bvand(&loss_mask).eq(&bv_zero).not())
        .collect();
    let is_lost = Bool::or(&loss_flags);

    EncodedPlacement {
        result_board: cleared,
        is_lost,
    }
}

fn encode_line_clearing(board: &[BV]) -> Vec<BV> {
    let valid_mask = BV::from_u64((1u64 << BOARD_ROWS) - 1, BV_WIDTH);
    let mut filled = valid_mask;
    for col in board {
        filled = filled.bvand(col);
    }

    let mut result: Vec<BV> = board.to_vec();
    let mut mask = filled;
    let bv_zero = BV::from_u64(0, BV_WIDTH);
    let bv_one = BV::from_u64(1, BV_WIDTH);

    for _ in 0..MAX_LINE_CLEARS {
        let has_row = mask.eq(&bv_zero).not();

        let pos = ctz(&mask);
        let pivot = bv_one.bvshl(&pos);
        let keep_below = pivot.bvsub(&bv_one);
        let above = pivot.bvnot().bvand(keep_below.bvnot());

        for col in &mut result {
            let below = col.bvand(&keep_below);
            let shifted_above = col.bvand(&above).bvlshr(&bv_one);
            let new_col = below.bvor(&shifted_above);
            *col = has_row.ite(&new_col, col);
        }

        let new_mask = mask.bvand(pivot.bvnot()).bvlshr(&bv_one);
        mask = has_row.ite(&new_mask, &mask);
    }

    result
}

// ---------------------------------------------------------------------------
// Invariant encoding
// ---------------------------------------------------------------------------

pub fn encode_invariant_with_params(board: &[BV], params: &InvariantParams) -> Bool {
    let mut constraints: Vec<Bool> = Vec::new();
    let bv_zero = BV::from_u64(0, BV_WIDTH);

    let mut heights: Vec<BV> = Vec::new();
    let mut total_holes = BV::from_u64(0, BV_WIDTH);

    for col in board {
        let height_mask = BV::from_u64(u64::MAX << params.max_height, BV_WIDTH);
        constraints.push(col.bvand(&height_mask).eq(&bv_zero));

        let h = col_height(col);

        if params.max_total_holes == 0 {
            for r in 1..params.max_height {
                let curr = test_bit(col, r);
                let prev = test_bit(col, r - 1);
                constraints.push(curr.implies(&prev));
            }
        } else {
            let pop = symbolic_popcount(col, params.max_height);
            let col_holes = h.bvsub(&pop);
            total_holes = total_holes.bvadd(&col_holes);
        }

        heights.push(h);
    }

    if params.max_total_holes > 0 {
        let max_holes_bv = BV::from_u64(params.max_total_holes as u64, BV_WIDTH);
        constraints.push(total_holes.bvule(&max_holes_bv));
    }

    if params.max_roughness < u32::MAX {
        let max_rough_bv = BV::from_u64(params.max_roughness as u64, BV_WIDTH);
        for c in 0..(BOARD_COLS as usize - 1) {
            let diff = bv_abs_diff(&heights[c], &heights[c + 1]);
            constraints.push(diff.bvule(&max_rough_bv));
        }
    }

    // No filled rows
    for r in 0..params.max_height {
        let mut col_bits: Vec<BV> = Vec::new();
        for col in board {
            let mask = BV::from_u64(1u64 << r, BV_WIDTH);
            col_bits.push(col.bvand(&mask));
        }
        let mut row_and = col_bits[0].clone();
        for cb in &col_bits[1..] {
            row_and = row_and.bvand(cb);
        }
        constraints.push(row_and.eq(&bv_zero));
    }

    if params.min_flat_block >= 2 {
        let w = params.min_flat_block as usize;
        let n = BOARD_COLS as usize;
        let mut window_options: Vec<Bool> = Vec::new();
        for start in 0..=(n - w) {
            let mut equal_pairs: Vec<Bool> = Vec::new();
            for i in start..(start + w - 1) {
                equal_pairs.push(heights[i].eq(&heights[i + 1]));
            }
            window_options.push(Bool::and(&equal_pairs));
        }
        constraints.push(Bool::or(&window_options));
    }

    Bool::and(&constraints)
}

// ---------------------------------------------------------------------------
// Concrete validation
// ---------------------------------------------------------------------------

pub fn board_satisfies_invariant_with_params(
    board: &TetrisBoard,
    params: &InvariantParams,
) -> bool {
    let mut heights = [0u32; BOARD_COLS as usize];
    let mut total_holes = 0u32;

    for col in 0..BOARD_COLS as usize {
        let h = {
            let mut height = 0u32;
            for r in 0..BOARD_ROWS {
                if board.get_bit(col, r as usize) {
                    height = r + 1;
                }
            }
            height
        };
        heights[col] = h;

        if h > params.max_height {
            return false;
        }

        if params.max_total_holes == 0 {
            for r in 0..h {
                if !board.get_bit(col, r as usize) {
                    return false;
                }
            }
        } else {
            let filled = (0..h).filter(|&r| board.get_bit(col, r as usize)).count() as u32;
            total_holes += h - filled;
        }
    }

    if params.max_total_holes > 0 && total_holes > params.max_total_holes {
        return false;
    }

    if params.max_roughness < u32::MAX {
        for c in 0..(BOARD_COLS as usize - 1) {
            if heights[c].abs_diff(heights[c + 1]) > params.max_roughness {
                return false;
            }
        }
    }

    // No filled rows
    let max_h = *heights.iter().max().unwrap_or(&0);
    for r in 0..max_h {
        if (0..BOARD_COLS as usize).all(|c| board.get_bit(c, r as usize)) {
            return false;
        }
    }

    if params.min_flat_block >= 2 {
        let w = params.min_flat_block as usize;
        let n = BOARD_COLS as usize;
        let mut found = false;
        for start in 0..=(n - w) {
            if (start..(start + w - 1)).all(|i| heights[i] == heights[i + 1]) {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    true
}

/// Helper: extract a column's bits as u64 from a TetrisBoard.
pub fn board_col_bits(board: &TetrisBoard, col: usize) -> u64 {
    let mut bits = 0u64;
    for r in 0..BV_WIDTH {
        if board.get_bit(col, r as usize) {
            bits |= 1u64 << r;
        }
    }
    bits
}

/// Helper: build a TetrisBoard from column heights (no holes, contiguous from row 0).
pub fn board_from_heights(heights: &[u32]) -> TetrisBoard {
    let mut board = TetrisBoard::new();
    for (col, &h) in heights.iter().enumerate() {
        for r in 0..h {
            board.set_bit(col, r as usize);
        }
    }
    board
}

/// Extract column heights from a TetrisBoard.
pub fn extract_heights(board: &TetrisBoard) -> [u8; 10] {
    let mut heights = [0u8; 10];
    for col in 0..10 {
        let mut h = 0u32;
        for r in 0..BOARD_ROWS {
            if board.get_bit(col, r as usize) {
                h = r + 1;
            }
        }
        heights[col] = h as u8;
    }
    heights
}

/// Extract column words from a TetrisBoard, masked to the lower 8 bits.
/// This matches VM_VALUE_BITS = 8, supporting max_height up to 8.
pub fn extract_col_words(board: &TetrisBoard) -> [u32; 10] {
    let limbs = board.as_limbs();
    std::array::from_fn(|i| limbs[i] & 0xFF)
}

/// Get the global placement index range for a piece in ALL_PLACEMENTS.
pub fn piece_global_placement_range(piece: TetrisPiece) -> std::ops::Range<u8> {
    let range = TetrisPiecePlacement::indices_from_piece(piece);
    range.start as u8..range.end as u8
}

/// Get placements for a specific piece.
pub fn placements_for_piece(all: &[PlacementInfo], piece: TetrisPiece) -> Vec<PlacementInfo> {
    all.iter().filter(|p| p.piece == piece).cloned().collect()
}

/// Find all valid global placement indices for a given board and piece.
/// A placement is valid if it doesn't cause loss and the result satisfies the invariant.
/// Returns global indices into `TetrisPiecePlacement::ALL_PLACEMENTS`.
pub fn find_safe_placements(
    board: &TetrisBoard,
    piece: TetrisPiece,
    params: &InvariantParams,
) -> Vec<u8> {
    let range = TetrisPiecePlacement::indices_from_piece(piece);
    let mut safe = Vec::new();

    for idx in range {
        let placement = TetrisPiecePlacement::from_index(idx as u8);
        let mut test_board = *board;
        let result = test_board.apply_piece_placement(placement);

        if bool::from(result.is_lost) {
            continue;
        }

        if board_satisfies_invariant_with_params(&test_board, params) {
            safe.push(idx as u8);
        }
    }

    safe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_col_words_empty() {
        let board = TetrisBoard::new();
        assert_eq!(extract_col_words(&board), [0; 10]);
    }

    #[test]
    fn test_extract_col_words_with_bits() {
        let mut board = TetrisBoard::new();
        board.set_bit(0, 0); // col 0, row 0
        board.set_bit(0, 1); // col 0, row 1
        board.set_bit(0, 7); // col 0, row 7 (highest bit in 8-bit mask)
        board.set_bit(0, 12); // col 0, row 12 (above 8-bit mask, should be masked out)
        board.set_bit(3, 0); // col 3, row 0
        let cw = extract_col_words(&board);
        assert_eq!(cw[0], 0b1000_0011); // rows 0, 1, 7 (row 12 masked out)
        assert_eq!(cw[3], 0b1); // row 0 set
        assert_eq!(cw[1], 0);
    }

    #[test]
    fn test_piece_global_placement_range() {
        for piece in TetrisPiece::all() {
            let range = piece_global_placement_range(piece);
            assert!(range.start < range.end, "piece {:?} has empty range", piece);
            assert!(range.end <= TetrisPiecePlacement::NUM_PLACEMENTS as u8);
        }
    }

    #[test]
    fn test_find_safe_placements_empty_board() {
        let board = TetrisBoard::new();
        let params = InvariantParams::new(4, 2, 4);

        for piece in TetrisPiece::all() {
            let safe = find_safe_placements(&board, piece, &params);
            assert!(
                !safe.is_empty(),
                "empty board should have safe placements for {:?}",
                piece
            );
            // All returned indices must be within the piece's range
            let range = piece_global_placement_range(piece);
            for &idx in &safe {
                assert!(range.contains(&idx), "index {idx} not in range {range:?}");
            }
        }
    }

    #[test]
    fn test_find_safe_placements_tight_invariant() {
        let board = TetrisBoard::new();
        let params = InvariantParams::new(2, 0, 2);

        let mut total_safe = 0;
        for piece in TetrisPiece::all() {
            let safe = find_safe_placements(&board, piece, &params);
            total_safe += safe.len();
        }
        assert!(total_safe > 0, "at least some pieces should work");
    }
}
