use z3::ast::{Ast, BV, Bool};

use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::config::*;

// ---------------------------------------------------------------------------
// Placement precomputation (from game engine)
// ---------------------------------------------------------------------------

/// Precomputed data for one concrete placement (piece + rotation + column).
///
/// Stores the raw piece column constants and trailing zeros, mirroring
/// the game engine's `place_piece_with_consts!` macro representation.
#[derive(Debug, Clone)]
pub struct PlacementInfo {
    pub piece: TetrisPiece,
    pub index: u8,
    /// Starting column index on the board (from placement column).
    pub start_col: usize,
    /// Number of board columns this piece spans.
    pub width: usize,
    /// Raw piece column bitmasks (bits at high end of u32, matching game engine's
    /// `tetris_piece_data`). Length = width.
    pub piece_consts: Vec<u32>,
    /// Trailing zeros of each piece column constant. Length = width.
    pub trailing: Vec<u32>,
}

/// Extract placement data from the game engine by placing each piece on an
/// empty board and reconstructing the raw piece column constants.
///
/// The game engine stores pieces with bits at the HIGH end of u32 and shifts
/// them right to drop. We reconstruct these constants from the placed result
/// on an empty board using: `piece_const[c] = placed[c] << shift_on_empty`
/// where `shift_on_empty = BV_WIDTH - max_placed_height`.
pub fn precompute_placements() -> Vec<PlacementInfo> {
    let mut result = Vec::new();
    for &placement in &TetrisPiecePlacement::ALL_PLACEMENTS {
        // Place on empty board to get placed column values
        let mut board = TetrisBoard::new();
        let pr = board.apply_piece_placement(placement);
        if bool::from(pr.is_lost) {
            continue;
        }

        let limbs = board.as_limbs();

        // Find the contiguous range of non-zero columns (piece span)
        let start_col = limbs.iter().position(|&c| c != 0).unwrap_or(0);
        let end_col = limbs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
        let width = end_col - start_col;
        let piece_cols = &limbs[start_col..end_col];

        // Reconstruct piece constants.
        // On an empty board, the piece drops to the bottom. The highest set bit
        // across piece columns gives max_placed_height. The original piece
        // constants had their highest bits at position 31, so:
        //   shift_on_empty = BV_WIDTH - max_placed_height
        //   piece_const[c] = placed[c] << shift_on_empty
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

/// Create a symbolic board: BOARD_COLS bitvectors of width BV_WIDTH.
pub fn make_symbolic_board(prefix: &str) -> Vec<BV> {
    (0..BOARD_COLS)
        .map(|i| BV::new_const(format!("{prefix}_c{i}"), BV_WIDTH))
        .collect()
}

// ---------------------------------------------------------------------------
// Symbolic primitives (mirroring game engine bitwise ops)
// ---------------------------------------------------------------------------

/// Test whether bit `r` of bitvector `val` is set.
fn test_bit(val: &BV, r: u32) -> Bool {
    let mask = BV::from_u64(1u64 << r, BV_WIDTH);
    let zero = BV::from_u64(0, BV_WIDTH);
    val.bvand(&mask).eq(&zero).not()
}

/// Symbolic column height: `BV_WIDTH - leading_zeros(col)`.
///
/// Matches the game engine's `u32::BITS - col.leading_zeros()`.
/// Returns 0 for an empty column, or (highest set bit + 1) otherwise.
pub fn col_height(col: &BV) -> BV {
    let mut h = BV::from_u64(0, BV_WIDTH);
    for r in 0..BV_WIDTH {
        let has_bit = test_bit(col, r);
        let height_val = BV::from_u64((r + 1) as u64, BV_WIDTH);
        h = has_bit.ite(&height_val, &h);
    }
    h
}

/// Symbolic unsigned absolute difference: `|a - b|`.
///
/// Matches the game engine's `u32::abs_diff(col_height, TRAILING)`.
fn bv_abs_diff(a: &BV, b: &BV) -> BV {
    let a_ge_b = a.bvuge(b);
    a_ge_b.ite(&a.bvsub(b), &b.bvsub(a))
}

/// Symbolic count-trailing-zeros. Returns BV_WIDTH if val == 0.
fn ctz(val: &BV) -> BV {
    let mut result = BV::from_u64(BV_WIDTH as u64, BV_WIDTH);
    // Iterate from high bit to low bit so lowest set bit wins
    for i in (0..BV_WIDTH).rev() {
        let has_bit = test_bit(val, i);
        let pos = BV::from_u64(i as u64, BV_WIDTH);
        result = has_bit.ite(&pos, &result);
    }
    result
}

/// Symbolic popcount (number of set bits) within the first `max_bits` positions.
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
// Placement encoding (mirrors game engine's place_piece_with_consts!)
// ---------------------------------------------------------------------------

/// Result of encoding a single placement in z3.
pub struct EncodedPlacement {
    /// Board state after placement and line clearing.
    pub result_board: Vec<BV>,
    /// Whether this placement causes loss (height > BOARD_ROWS after clearing).
    pub is_lost: Bool,
}

/// Encode the effect of a placement on a symbolic board.
///
/// Directly mirrors the game engine's `place_piece_with_consts!` macro:
///
/// ```text
/// // Game engine (concrete):
/// let shift = min_c(abs_diff(col_height[c], TRAILING[c]));
/// board_cols[c] |= piece_cols[c] >> shift;
///
/// // z3 encoding (symbolic):
/// shift = bv_umin(bv_abs_diff(col_height(board[c]), TRAILING[c]))
/// result[c] = board[c] | (piece_const[c] >> shift)
/// ```
pub fn encode_placement(board: &[BV], info: &PlacementInfo) -> EncodedPlacement {
    // 1. Compute shift = min(abs_diff(col_height, TRAILING[c]))
    //    Mirrors: min_diff = u32::MAX; diff = col_height.abs_diff(TRAILING[I]);
    //            min_diff = min(min_diff, diff);
    let mut shift = BV::from_u64(u32::MAX as u64, BV_WIDTH);
    for i in 0..info.width {
        let board_col = &board[info.start_col + i];
        let h = col_height(board_col);
        let trailing_bv = BV::from_u64(info.trailing[i] as u64, BV_WIDTH);
        let diff = bv_abs_diff(&h, &trailing_bv);
        let lt = diff.bvult(&shift);
        shift = lt.ite(&diff, &shift);
    }

    // 2. Place piece: board_col |= piece_col >> shift
    let mut new_board: Vec<BV> = board.to_vec();
    for i in 0..info.width {
        let piece_bv = BV::from_u64(info.piece_consts[i] as u64, BV_WIDTH);
        let shifted = piece_bv.bvlshr(&shift);
        new_board[info.start_col + i] = new_board[info.start_col + i].bvor(&shifted);
    }

    // 3. Clear filled lines (mirrors rshift_slice_from_mask_u32)
    let cleared = encode_line_clearing(&new_board);

    // 4. Check loss: any bit >= BOARD_ROWS set after clearing
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

// ---------------------------------------------------------------------------
// Line clearing (mirrors rshift_slice_from_mask_u32)
// ---------------------------------------------------------------------------

/// Encode line clearing on a symbolic board.
///
/// Mirrors the game engine's `rshift_slice_from_mask_u32`:
/// 1. Compute filled rows mask = AND of all columns (restricted to BOARD_ROWS).
/// 2. Iteratively remove the lowest filled row and compact (up to MAX_LINE_CLEARS times).
fn encode_line_clearing(board: &[BV]) -> Vec<BV> {
    // Compute filled rows mask (only within valid board rows)
    let valid_mask = BV::from_u64((1u64 << BOARD_ROWS) - 1, BV_WIDTH);
    let mut filled = valid_mask;
    for col in board {
        filled = filled.bvand(col);
    }

    let mut result: Vec<BV> = board.to_vec();
    let mut mask = filled;
    let bv_zero = BV::from_u64(0, BV_WIDTH);
    let bv_one = BV::from_u64(1, BV_WIDTH);

    // Unrolled loop: remove up to MAX_LINE_CLEARS rows
    // Mirrors: while remaining_mask != 0 { ... }
    for _ in 0..MAX_LINE_CLEARS {
        let has_row = mask.eq(&bv_zero).not();

        // Find position of lowest set bit in mask (mirrors mask.trailing_zeros())
        let pos = ctz(&mask);

        // pivot = 1 << pos
        let pivot = bv_one.bvshl(&pos);
        // keep_below = pivot - 1 (bits below the row to remove)
        let keep_below = pivot.bvsub(&bv_one);
        // above = ~pivot & ~keep_below (bits above the row to remove)
        let above = pivot.bvnot().bvand(keep_below.bvnot());

        // For each column: keep bits below, shift bits above down by 1
        // Mirrors: xs[I] = (xs[I] & keep) | ((xs[I] & above) >> 1);
        for col in &mut result {
            let below = col.bvand(&keep_below);
            let shifted_above = col.bvand(&above).bvlshr(&bv_one);
            let new_col = below.bvor(&shifted_above);
            *col = has_row.ite(&new_col, col);
        }

        // Update mask: clear the processed bit and shift right
        // Mirrors: *m = (*m & !pivot) >> 1;
        let new_mask = mask.bvand(pivot.bvnot()).bvlshr(&bv_one);
        mask = has_row.ite(&new_mask, &mask);
    }

    result
}

// ---------------------------------------------------------------------------
// Invariant encoding
// ---------------------------------------------------------------------------

/// Encode the invariant P(board) with explicit parameters:
///   - Height <= params.max_height for each column
///   - Total board holes <= params.max_total_holes (sum of per-column holes)
///   - Adjacent column height difference <= params.max_roughness
///   - No bits above BOARD_ROWS
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

    // No filled rows: any reachable board has already had filled rows cleared.
    // For each row r, at least one column must have bit r unset.
    for r in 0..params.max_height {
        let mut col_bits: Vec<BV> = Vec::new();
        for col in board {
            let mask = BV::from_u64(1u64 << r, BV_WIDTH);
            col_bits.push(col.bvand(&mask));
        }
        // AND all columns at row r — if result is nonzero, all columns have bit r set
        let mut row_and = col_bits[0].clone();
        for cb in &col_bits[1..] {
            row_and = row_and.bvand(cb);
        }
        let bv_zero = BV::from_u64(0, BV_WIDTH);
        constraints.push(row_and.eq(&bv_zero));
    }

    // min_flat_block: require at least N adjacent columns of equal height
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

/// Convenience wrapper using global config constants.
pub fn encode_invariant(board: &[BV]) -> Bool {
    encode_invariant_with_params(board, &InvariantParams::from_config())
}

// ---------------------------------------------------------------------------
// Cross-validation with game engine
// ---------------------------------------------------------------------------

/// Verify a concrete board satisfies the invariant with explicit parameters.
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

/// Convenience wrapper using global config constants.
pub fn board_satisfies_invariant(board: &TetrisBoard) -> bool {
    board_satisfies_invariant_with_params(board, &InvariantParams::from_config())
}

/// Helper: extract a column's bits as u64 from a TetrisBoard (reads up to BV_WIDTH rows).
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

/// Helper: evaluate the z3 invariant on concrete column values.
pub fn eval_z3_invariant_concrete(col_values: &[u64]) -> bool {
    use z3::{SatResult, Solver};

    let solver = Solver::new();
    let board = make_symbolic_board("test");

    for (bv, &val) in board.iter().zip(col_values.iter()) {
        let concrete = BV::from_u64(val, BV_WIDTH);
        solver.assert(bv.eq(&concrete));
    }

    let inv = encode_invariant(&board);
    solver.push();
    solver.assert(inv);
    let result = solver.check();
    solver.pop(1);

    matches!(result, SatResult::Sat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::TetrisPiecePlacement;
    use z3::{SatResult, Solver};

    // -----------------------------------------------------------------------
    // precompute_placements
    // -----------------------------------------------------------------------

    #[test]
    fn test_precompute_placements_count() {
        let placements = precompute_placements();
        assert_eq!(placements.len(), TetrisPiecePlacement::NUM_PLACEMENTS);
    }

    #[test]
    fn test_precompute_placements_piece_consts_valid() {
        let placements = precompute_placements();
        for info in &placements {
            assert_eq!(
                info.piece_consts.len(),
                info.width,
                "placement {}: piece_consts len mismatch",
                info.index
            );
            assert_eq!(
                info.trailing.len(),
                info.width,
                "placement {}: trailing len mismatch",
                info.index
            );
            // Each piece constant must have exactly the right trailing zeros
            for (i, &pc) in info.piece_consts.iter().enumerate() {
                assert!(
                    pc != 0,
                    "placement {}: piece_const[{}] is zero",
                    info.index,
                    i
                );
                assert_eq!(
                    pc.trailing_zeros(),
                    info.trailing[i],
                    "placement {}: trailing[{}] mismatch",
                    info.index,
                    i
                );
            }
            // start_col + width must be within board
            assert!(
                info.start_col + info.width <= BOARD_COLS as usize,
                "placement {}: start_col + width out of bounds",
                info.index
            );
        }
    }

    #[test]
    fn test_precompute_placements_total_bits() {
        // Each tetromino has exactly 4 cells. Verify that each placement's
        // piece constants encode exactly 4 set bits total.
        let placements = precompute_placements();
        for info in &placements {
            let total_bits: u32 = info.piece_consts.iter().map(|c| c.count_ones()).sum();
            assert_eq!(
                total_bits, 4,
                "placement {} has {} bits, expected 4",
                info.index, total_bits
            );
        }
    }

    // -----------------------------------------------------------------------
    // board_satisfies_invariant (concrete)
    // -----------------------------------------------------------------------

    #[test]
    fn test_invariant_empty_board() {
        let board = TetrisBoard::new();
        assert!(board_satisfies_invariant(&board));
    }

    #[test]
    fn test_invariant_max_height_board() {
        // All columns at MAX_HEIGHT would have filled rows — unreachable.
        // Use a valid board ramping up to MAX_HEIGHT with empty edges (no filled rows).
        let board = board_from_heights(&[0, 2, 4, 4, 4, 4, 4, 4, 2, 0]);
        assert!(board_satisfies_invariant(&board));
    }

    #[test]
    fn test_invariant_exceeds_height() {
        let mut heights = [0u32; BOARD_COLS as usize];
        heights[0] = MAX_HEIGHT + 1;
        let board = board_from_heights(&heights);
        assert!(!board_satisfies_invariant(&board));
    }

    #[test]
    fn test_invariant_with_holes() {
        // Column 0: bit 1 set but not bit 0 → 1 hole, height 2
        let mut board = TetrisBoard::new();
        board.set_bit(0, 1);
        if MAX_TOTAL_HOLES >= 1 && MAX_HEIGHT >= 2 {
            assert!(board_satisfies_invariant(&board));
        } else {
            assert!(!board_satisfies_invariant(&board));
        }
    }

    // -----------------------------------------------------------------------
    // z3 encode_invariant vs concrete board_satisfies_invariant
    // -----------------------------------------------------------------------

    #[test]
    fn test_z3_invariant_matches_concrete_empty() {
        let col_values = vec![0u64; BOARD_COLS as usize];
        assert!(eval_z3_invariant_concrete(&col_values));
    }

    #[test]
    fn test_z3_invariant_matches_concrete_max_height() {
        // Board ramping up to MAX_HEIGHT with empty edges (no filled rows).
        let heights: [u32; 10] = [0, 2, 4, 4, 4, 4, 4, 4, 2, 0];
        let col_values: Vec<u64> = heights
            .iter()
            .map(|&h| if h == 0 { 0 } else { (1u64 << h) - 1 })
            .collect();
        assert!(eval_z3_invariant_concrete(&col_values));
    }

    #[test]
    fn test_z3_invariant_matches_concrete_over_height() {
        let mut col_values = vec![0u64; BOARD_COLS as usize];
        col_values[0] = (1u64 << (MAX_HEIGHT + 1)) - 1;
        assert!(!eval_z3_invariant_concrete(&col_values));
    }

    #[test]
    fn test_z3_invariant_matches_concrete_with_hole() {
        // Column 0: bit 1 set, bit 0 clear → height 2, holes 1
        let mut col_values = vec![0u64; BOARD_COLS as usize];
        col_values[0] = 0b10;
        let expected = MAX_TOTAL_HOLES >= 1 && MAX_HEIGHT >= 2;
        assert_eq!(eval_z3_invariant_concrete(&col_values), expected);
    }

    #[test]
    fn test_z3_invariant_matches_concrete_various_boards() {
        // Test boards and verify z3 matches concrete check
        let test_boards: Vec<Vec<u64>> = vec![
            vec![0; BOARD_COLS as usize],
            // All columns at MAX_HEIGHT, no holes
            vec![(1u64 << MAX_HEIGHT) - 1; BOARD_COLS as usize],
            // Various heights, no holes
            {
                let heights = [1u32, 2, 3, 4, 3, 2, 1, 0, 0, 0];
                heights
                    .iter()
                    .map(|&h| if h == 0 { 0 } else { (1u64 << h) - 1 })
                    .collect()
            },
            // Board with holes (bit pattern with gaps)
            {
                let mut v = vec![0u64; BOARD_COLS as usize];
                v[0] = 0b1010; // height 4, 2 holes
                v[1] = 0b0110; // height 3, 1 hole
                v
            },
        ];

        for col_values in &test_boards {
            // Build a TetrisBoard from raw column values
            let mut board = TetrisBoard::new();
            for (col, &val) in col_values.iter().enumerate() {
                for r in 0..BV_WIDTH {
                    if val & (1u64 << r) != 0 {
                        board.set_bit(col, r as usize);
                    }
                }
            }

            let concrete_result = board_satisfies_invariant(&board);
            let z3_result = eval_z3_invariant_concrete(col_values);

            assert_eq!(
                z3_result, concrete_result,
                "mismatch for cols {:?}: z3={}, concrete={}",
                col_values, z3_result, concrete_result
            );
        }
    }

    // -----------------------------------------------------------------------
    // z3 encode_placement vs game engine
    // -----------------------------------------------------------------------

    #[test]
    fn test_z3_placement_matches_engine_empty_board() {
        let placements = precompute_placements();
        let col_values = vec![0u64; BOARD_COLS as usize];

        for info in &placements {
            let mut engine_board = TetrisBoard::new();
            let placement = TetrisPiecePlacement::from_index(info.index);
            let engine_result = engine_board.apply_piece_placement(placement);

            let solver = Solver::new();
            let board = make_symbolic_board("b");
            for (bv, &val) in board.iter().zip(col_values.iter()) {
                solver.assert(bv.eq(&BV::from_u64(val, BV_WIDTH)));
            }

            let enc = encode_placement(&board, info);

            solver.assert(enc.is_lost.not());
            assert_eq!(solver.check(), SatResult::Sat);

            let model = solver.get_model().unwrap();
            for (c, result_col) in enc.result_board.iter().enumerate() {
                if let Some(val) = model.eval(result_col, true) {
                    if let Some(bits) = val.as_u64() {
                        let engine_col = board_col_bits(&engine_board, c);
                        assert_eq!(
                            bits, engine_col,
                            "column {} mismatch for placement {}: z3={:#x}, engine={:#x}",
                            c, info.index, bits, engine_col
                        );
                    }
                }
            }

            assert!(
                !bool::from(engine_result.is_lost),
                "engine says lost for placement {} on empty board",
                info.index
            );
        }
    }

    #[test]
    fn test_z3_placement_matches_engine_nonempty_boards() {
        let placements = precompute_placements();

        let test_heights: Vec<Vec<u32>> = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            vec![3, 1, 2, 0, 4, 2, 1, 3, 0, 2],
        ];

        for heights in &test_heights {
            let base_board = board_from_heights(heights);
            let col_values: Vec<u64> = heights
                .iter()
                .map(|&h| if h == 0 { 0 } else { (1u64 << h) - 1 })
                .collect();

            for info in &placements {
                let mut engine_board = base_board;
                let placement = TetrisPiecePlacement::from_index(info.index);
                let engine_result = engine_board.apply_piece_placement(placement);
                let engine_lost = bool::from(engine_result.is_lost);

                let solver = Solver::new();
                let board = make_symbolic_board("b");
                for (bv, &val) in board.iter().zip(col_values.iter()) {
                    solver.assert(bv.eq(&BV::from_u64(val, BV_WIDTH)));
                }

                let enc = encode_placement(&board, info);

                solver.push();
                solver.assert(enc.is_lost.clone());
                let z3_lost = matches!(solver.check(), SatResult::Sat);
                solver.pop(1);

                assert_eq!(
                    z3_lost, engine_lost,
                    "loss mismatch for placement {} on heights {:?}: z3={}, engine={}",
                    info.index, heights, z3_lost, engine_lost
                );

                if !engine_lost {
                    solver.push();
                    solver.assert(enc.is_lost.not());
                    assert_eq!(solver.check(), SatResult::Sat);
                    let model = solver.get_model().unwrap();

                    for (c, result_col) in enc.result_board.iter().enumerate() {
                        if let Some(val) = model.eval(result_col, true) {
                            if let Some(bits) = val.as_u64() {
                                let engine_col = board_col_bits(&engine_board, c);
                                assert_eq!(
                                    bits, engine_col,
                                    "column {} mismatch for placement {} on heights {:?}: z3={:#x}, engine={:#x}",
                                    c, info.index, heights, bits, engine_col
                                );
                            }
                        }
                    }
                    solver.pop(1);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Line clearing (direct validation)
    // -----------------------------------------------------------------------

    /// Helper: build a concrete board from raw column u32 values, apply game
    /// engine's clear_filled_rows, and return (cleared_limbs, lines_cleared).
    fn engine_clear_lines(col_vals: &[u32; 10]) -> ([u32; 10], u32) {
        let mut board = TetrisBoard::new();
        for (c, &val) in col_vals.iter().enumerate() {
            for r in 0..BV_WIDTH {
                if val & (1 << r) != 0 {
                    board.set_bit(c, r as usize);
                }
            }
        }
        let lines = board.clear_filled_rows();
        (board.as_limbs(), lines)
    }

    /// Helper: build a symbolic board from concrete values, run encode_line_clearing,
    /// evaluate the result, and return the concrete column values.
    fn z3_clear_lines(col_vals: &[u32; 10]) -> [u64; 10] {
        let solver = Solver::new();
        let board: Vec<BV> = (0..10)
            .map(|i| {
                let bv = BV::new_const(format!("c{i}"), BV_WIDTH);
                solver.assert(bv.eq(&BV::from_u64(col_vals[i] as u64, BV_WIDTH)));
                bv
            })
            .collect();

        let cleared = encode_line_clearing(&board);
        assert_eq!(solver.check(), SatResult::Sat);
        let model = solver.get_model().unwrap();

        let mut result = [0u64; 10];
        for (i, bv) in cleared.iter().enumerate() {
            if let Some(val) = model.eval(bv, true) {
                result[i] = val.as_u64().unwrap_or(0);
            }
        }
        result
    }

    #[test]
    fn test_line_clearing_no_clears() {
        // No filled rows → nothing changes
        let board: [u32; 10] = [0b01, 0b11, 0b01, 0, 0, 0, 0, 0, 0, 0];
        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 0);
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (no clears): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
    }

    #[test]
    fn test_line_clearing_one_row() {
        // Row 0 filled across all 10 columns, row 1 has some cells
        let mut board = [0u32; 10];
        for col in &mut board {
            *col = 0b01; // bit 0 set in all columns
        }
        board[0] = 0b11; // col 0 also has bit 1
        board[3] = 0b101; // col 3 has bits 0 and 2

        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 1, "expected 1 line clear");
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (1 clear): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
        // After clearing row 0: col 0 should have bit 0 (shifted from bit 1)
        assert_eq!(engine[0], 0b01);
        // col 3: bit 0 removed, bit 2 shifts to bit 1
        assert_eq!(engine[3], 0b10);
    }

    #[test]
    fn test_line_clearing_two_rows() {
        // Rows 0 and 1 filled across all columns, row 2 partial
        let mut board = [0u32; 10];
        for col in &mut board {
            *col = 0b11; // bits 0,1 set
        }
        board[0] = 0b111; // col 0 also has bit 2
        board[5] = 0b1011; // col 5 has bits 0,1,3

        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 2, "expected 2 line clears");
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (2 clears): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
        // After clearing rows 0,1: col 0 bit 2 → bit 0
        assert_eq!(engine[0], 0b01);
        // col 5: bits 0,1 cleared, bit 3 → bit 1
        assert_eq!(engine[5], 0b10);
    }

    #[test]
    fn test_line_clearing_three_rows() {
        // Rows 0, 1, 2 filled across all columns, row 3 partial
        let mut board = [0u32; 10];
        for col in &mut board {
            *col = 0b111; // bits 0,1,2
        }
        board[2] = 0b1111; // col 2 also has bit 3

        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 3, "expected 3 line clears");
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (3 clears): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
        // All columns should be empty except col 2
        assert_eq!(engine[0], 0);
        assert_eq!(engine[2], 0b01); // bit 3 → bit 0
    }

    #[test]
    fn test_line_clearing_four_rows() {
        // Rows 0-3 filled across all columns, row 4 partial
        let mut board = [0u32; 10];
        for col in &mut board {
            *col = 0b1111; // bits 0-3
        }
        board[0] = 0b11111; // col 0 also has bit 4
        board[9] = 0b101111; // col 9 has bits 0-3 and bit 5

        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 4, "expected 4 line clears");
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (4 clears): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
        // col 0: bit 4 → bit 0
        assert_eq!(engine[0], 0b01);
        // col 9: bits 0-3 cleared, bit 5 → bit 1
        assert_eq!(engine[9], 0b10);
    }

    #[test]
    fn test_line_clearing_non_contiguous_rows() {
        // Rows 0 and 2 filled (row 1 NOT filled) — only 2 clears but non-adjacent
        let mut board = [0u32; 10];
        for col in &mut board {
            *col = 0b101; // bits 0 and 2 set (row 1 empty)
        }
        board[0] = 0b10111; // col 0 has bits 0,1,2,4
        board[4] = 0b1101; // col 4 has bits 0,2,3

        let (engine, lines) = engine_clear_lines(&board);
        let z3 = z3_clear_lines(&board);
        assert_eq!(lines, 2, "expected 2 line clears (non-contiguous)");
        for i in 0..10 {
            assert_eq!(
                z3[i], engine[i] as u64,
                "col {} mismatch (non-contiguous): z3={:#x}, engine={:#x}",
                i, z3[i], engine[i]
            );
        }
    }

    #[test]
    fn test_line_clearing_with_placement_triggers_clears() {
        // Board with rows 0-2 filled. Place I-piece horizontal → row 3 fills → 4 clears total?
        // Actually rows 0-2 are already filled, placing I-piece adds row 3 in 4 cols.
        // Row 3 won't be filled (only 4 of 10 cols), so only 3 clears.
        let placements = precompute_placements();
        let heights = vec![3u32; 10]; // all columns height 3, rows 0-2 filled
        let base_board = board_from_heights(&heights);
        let col_values: Vec<u64> = vec![0b111; 10]; // rows 0,1,2

        let mut total_clears = 0u32;
        for info in &placements {
            let mut engine_board = base_board;
            let placement = TetrisPiecePlacement::from_index(info.index);
            let engine_result = engine_board.apply_piece_placement(placement);
            if bool::from(engine_result.is_lost) {
                continue;
            }

            if engine_result.lines_cleared > 0 {
                total_clears += 1;

                // Cross-validate with z3
                let solver = Solver::new();
                let board = make_symbolic_board("b");
                for (bv, &val) in board.iter().zip(col_values.iter()) {
                    solver.assert(bv.eq(&BV::from_u64(val, BV_WIDTH)));
                }
                let enc = encode_placement(&board, info);
                solver.assert(enc.is_lost.not());
                assert_eq!(solver.check(), SatResult::Sat);
                let model = solver.get_model().unwrap();

                for (c, result_col) in enc.result_board.iter().enumerate() {
                    if let Some(val) = model.eval(result_col, true) {
                        if let Some(bits) = val.as_u64() {
                            let engine_col = board_col_bits(&engine_board, c);
                            assert_eq!(
                                bits, engine_col,
                                "col {} mismatch for placement {} ({} clears): z3={:#x}, engine={:#x}",
                                c, info.index, engine_result.lines_cleared, bits, engine_col
                            );
                        }
                    }
                }
            }
        }
        assert!(
            total_clears > 0,
            "expected at least one placement to trigger line clears on height-3 board"
        );
    }
}
