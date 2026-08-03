use tetris_game::config::StandardTetris;
use tetris_game::tetris::{Major, TetrisGame, TetrisPiece};

/// Board occupancy bits (10 × 20).
pub const NUM_CELLS: usize = 200;
/// Number of distinct tetrominoes.
pub const NUM_PIECES: usize = 7;
/// Total binary inputs each tree sees: board + bag-remaining + current-piece.
pub const NUM_INPUTS: usize = NUM_CELLS + NUM_PIECES + NUM_PIECES;

/// Encode the full game as a fixed binary vector the trees read:
/// - `0..200`   board occupancy (`to_cell_array(Major::Col)`, index `col*20+row`),
/// - `200..207` bag-remaining (`bag.contains(piece)` for each of the 7 pieces),
/// - `207..214` current-piece one-hot (`current_piece == piece`).
pub fn encode_game(game: &TetrisGame<StandardTetris>) -> [u8; NUM_INPUTS] {
    let mut bits = [0u8; NUM_INPUTS];

    // Board occupancy, straight from the engine (BOARD_SIZE == 200 for standard).
    let cells = game.board.to_cell_array(Major::Col);
    bits[..NUM_CELLS].copy_from_slice(&cells[..NUM_CELLS]);

    // Bag-remaining and current-piece one-hot, one bit per piece.
    for (i, piece) in TetrisPiece::all().iter().enumerate() {
        if game.bag.contains(*piece) {
            bits[NUM_CELLS + i] = 1;
        }
        if game.current_piece == *piece {
            bits[NUM_CELLS + NUM_PIECES + i] = 1;
        }
    }

    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_game_encoding_has_expected_shape() {
        let game = TetrisGame::<StandardTetris>::new_with_seed(1);
        let bits = encode_game(&game);
        assert_eq!(bits.len(), NUM_INPUTS);
        // Fresh board is empty.
        assert!(bits[..NUM_CELLS].iter().all(|&b| b == 0), "empty board");
        // Exactly one current-piece bit is set (one-hot).
        let piece_bits: u32 = bits[NUM_CELLS + NUM_PIECES..]
            .iter()
            .map(|&b| u32::from(b))
            .sum();
        assert_eq!(piece_bits, 1, "exactly one current piece");
        // The bag has at least one remaining piece at game start.
        let bag_bits: u32 = bits[NUM_CELLS..NUM_CELLS + NUM_PIECES]
            .iter()
            .map(|&b| u32::from(b))
            .sum();
        assert!(bag_bits >= 1, "bag not empty at start");
        // All inputs are 0/1.
        assert!(bits.iter().all(|&b| b <= 1));
    }

    #[test]
    fn placing_a_piece_marks_board_bits() {
        let mut game = TetrisGame::<StandardTetris>::new_with_seed(1);
        let placement = game.current_placements()[0];
        game.apply_placement(placement);
        let bits = encode_game(&game);
        let board_on: u32 = bits[..NUM_CELLS].iter().map(|&b| u32::from(b)).sum();
        assert_eq!(board_on, 4, "one tetromino fills exactly 4 board bits");
    }
}
