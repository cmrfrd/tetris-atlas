use std::fmt::Display;

const ROWS: usize = 20;
const COLS: usize = 10;
const BOARD_SIZE: usize = ROWS * COLS;
const NUM_BYTES_FOR_BOARD: usize = BOARD_SIZE / 8;
const ROW_CHUNK: usize = 4;
const BYTES_PER_ROW_CHUNK: usize = ROW_CHUNK * COLS / 8;

pub const ASSERT_LEVEL: u32 = 1;

/// This custom assert is so we can can disable it at compile
/// time, to remove runtime impact.
macro_rules! assert_level {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            if $crate::ASSERT_LEVEL >= 1 {
                assert!($($arg)*);
            }
        }
    };
}

#[inline]
fn row_to_bit_indices(row: usize) -> (usize, usize) {
    let first_byte = row + (row / 4);
    let second_byte = first_byte + 1;
    (first_byte, second_byte)
}

/// A board is a 20x10 grid of cells. Each cell is a bit.
/// Indexing the tetris board we goes left to right, top to bottom.
///
/// We operate on a byte by byte level.
/// This is the layout of the board:
///
///  0 00000000 00 0 1
///  1 000000 0000 1 2
///  2 0000 000000 2 3
///  3 00 00000000 3 4
///  4 00000000 00 5 6
///  5 000000 0000 6 7
///  6 0000 000000 7 8
///  7 00 00000000 8 9
///  8 00000000 00 10 11
///  9 000000 0000 11 12
/// 10 0000 000000 12 13
/// 11 00 00000000 13 14
/// 12 00000000 00 15 16
/// 13 000000 0000 17 18
/// 14 0000 000000 18 19
/// 15 00 00000000 19 20
/// 16 00000000 00 21 22
/// 17 000000 0000 22 23
/// 18 0000 000000 23 24
/// 19 00 00000000 24 25
type BoardRaw = [u8; NUM_BYTES_FOR_BOARD];

// A rotation is a u8 that represents a respective 90 degree rotation.
// 0 = 0 degrees
// 1 = 90 degrees
// 2 = 180 degrees
// 3 = 270 degrees
struct Rotation(u8);

impl Rotation {
    // When we iterate over the rotations
    fn next(&mut self) {
        *self = unsafe { std::mem::transmute((self.0.wrapping_add(1)) % 4) };
    }
}

/// A tetris piece is a u8 that represents a respective piece.
/// 0 = O -> 1 rotation
/// 1 = I -> 2 rotations
/// 2 = S -> 2 rotations
/// 3 = Z -> 2 rotations
/// 4 = T -> 4 rotations
/// 5 = L -> 4 rotations
/// 6 = J -> 4 rotations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TetrisPiece(u8);

impl TetrisPiece {
    const fn num_rotations(&self) -> u8 {
        // 1_u8.wrapping_add((!(self.0 & !self.0)) & 3).wrapping_sub(
        //     (!(((self.0 & 0b0000_0100).wrapping_add(!6))
        //         | ((self.0 & 0b0000_0100).wrapping_add(!6).wrapping_add(1))))
        //         & 2,
        // )
        match self.0 {
            0 => 1,
            1 | 2 | 3 => 2,
            4 | 5 | 6 => 4,
            _ => panic!("Invalid piece"),
        }
    }

    const fn dimensions(&self, rotation: Rotation) -> (u8, u8) {
        // return (width, height)
        match (self.0, rotation.0) {
            (0, 0 | 2) => (4, 1), // I (flat)
            (0, 1 | 3) => (1, 4), // I (tall)

            (1, 0) => (3, 2), // J (flat)
            (1, 1) => (2, 3), // J (tall)
            (1, 2) => (3, 2), // J (flat)
            (1, 3) => (2, 3), // J (tall)

            (2, 0) => (3, 2), // L (flat)
            (2, 1) => (2, 3), // L (tall)
            (2, 2) => (3, 2), // L (flat)
            (2, 3) => (2, 3), // L (tall)

            (3, _) => (2, 2), // O (no different rotations)

            (4, 0 | 2) => (3, 2), // S (flat)
            (4, 1 | 3) => (2, 3), // S (tall)

            (5, 0 | 2) => (3, 2), // T (flat)
            (5, 1 | 3) => (2, 3), // T (tall)

            _ => panic!("Invalid piece or rotation"),
        }
    }
}

pub struct TetrisBoard {
    board: BoardRaw,

    /// Shift buffers used to store intermediate results of shifts.
    shift_buf_a: u8,
    shift_buf_b: u8,
}

pub trait Shiftable {
    /// Shift all the rows up by 1.
    /// After we 'drop' a piece and find a collision, we need to shift up
    /// the piece before adding it to the board.
    fn shift_up(&mut self);

    /// Shift all the rows down by 1.
    /// When placing a piece, we progressively shift it down until it
    /// either hits the bottom of the board or collides with another space.
    fn shift_down(&mut self);

    /// Shift all the columns left by 1.
    /// When placing a piece at a particular position, we may need to shift
    /// it left to adjust it to the correct column.
    fn shift_left(&mut self);

    /// Shift all the columns right by 1.
    /// When placing a piece at a particular position, we may need to shift
    /// it right to adjust it to the correct column.
    fn shift_right(&mut self);
}

impl Shiftable for TetrisBoard {
    fn shift_up(&mut self) {
        // Shift all bytes up by 1
        self.board.copy_within(1..NUM_BYTES_FOR_BOARD, 0);
        assert_level!((NUM_BYTES_FOR_BOARD - 1) < self.board.len());
        self.board[NUM_BYTES_FOR_BOARD - 1] = 0;

        // Shift the bits up and carry the bottom 2 bits to the next byte
        for i in (0..(NUM_BYTES_FOR_BOARD - 1)).rev() {
            assert_level!(i < self.board.len());
            unsafe {
                self.shift_buf_b = (self.board.get_unchecked(i) & 0b1100_0000) >> 6; // are we at a byte boundary?
                *self.board.get_unchecked_mut(i) =
                    (self.board.get_unchecked(i) << 2) | self.shift_buf_a;
            }
            self.shift_buf_a = self.shift_buf_b;
        }

        // Clear the shift buffer
        self.shift_buf_a = 0;
    }

    fn shift_down(&mut self) {
        // Shift all bytes down by 1
        self.board.copy_within(0..(NUM_BYTES_FOR_BOARD - 1), 1);
        assert_level!(0 < self.board.len());
        self.board[0] = 0;

        // Shift the bits down and carry the top 2 bits to the next byte
        for i in 0..NUM_BYTES_FOR_BOARD {
            assert_level!(i < self.board.len());
            unsafe {
                self.shift_buf_b = (self.board.get_unchecked(i) & 0b0000_0011) << 6;
                *self.board.get_unchecked_mut(i) =
                    (self.board.get_unchecked(i) >> 2) | self.shift_buf_a;
            }
            self.shift_buf_a = self.shift_buf_b;
        }

        // Clear the shift buffer
        self.shift_buf_a = 0;
    }

    fn shift_left(&mut self) {
        self.shift_buf_a = (self.board[NUM_BYTES_FOR_BOARD - 1] & 0b1000_0000) >> 7;
        self.board[NUM_BYTES_FOR_BOARD - 1] = self.board[NUM_BYTES_FOR_BOARD - 1] << 1;
        for i in (0..NUM_BYTES_FOR_BOARD - 1).rev() {
            self.board[i] = (self.board[i] << 1) | self.shift_buf_a;
            self.shift_buf_a = (self.board[i] & 0b1000_0000) >> 7;
        }
    }

    fn shift_right(&mut self) {
        self.shift_buf_a = (self.board[0] & 0b0000_0001) << 7;
        self.board[0] = self.board[0] >> 1;
        for i in 1..NUM_BYTES_FOR_BOARD {
            self.board[i] = (self.board[i] >> 1) | self.shift_buf_a;
            let mask = !(0b1000_0000_u8.unbounded_shr((2 * (i % 5)) as u32));
            println!("i: {}, mask: {:#010b}", i, mask);
            self.board[i] &= mask;
            self.shift_buf_a = (self.board[i] & 0b0000_0001) << 7;

            // 0 -> 0b0111_1111
            // 1 -> 0b1101_1111
            // 2 -> 0b1111_0111
            // 3 -> 0b1111_1101
            // 4 -> 0b1111_1111
        }
        println!("{}", self);
    }
}

pub trait BitSetter {
    fn set_bit(&mut self, col: usize, row: usize, value: bool);
    fn get_bit(&self, col: usize, row: usize) -> bool;
}

impl BitSetter for TetrisBoard {
    fn set_bit(&mut self, col: usize, row: usize, value: bool) {
        // Set the bit at the given column and row
        if value {
            self.board[(row * COLS + col) / 8] |= 1 << (7 - ((row * COLS + col) % 8));
        } else {
            self.board[(row * COLS + col) / 8] &= !(1 << (7 - ((row * COLS + col) % 8)));
        }
    }

    fn get_bit(&self, col: usize, row: usize) -> bool {
        self.board[(row.wrapping_mul(COLS).wrapping_add(col)) >> 3]
            & (1_u8.wrapping_shl(
                7_u32.wrapping_sub(((row.wrapping_mul(COLS).wrapping_add(col)) & 7) as u32),
            ))
            != 0
    }
}

pub trait RowClearer {
    /// Whenever a row is full, clear it and settle the rows above it.
    fn clear_rows(&mut self);
}

// impl RowClearer for TetrisBoard {
//     fn clear_row(&mut self, row: usize) {
//         self.board[row] = 0;
//     }
// }

impl TetrisBoard {
    pub fn add_piece_top(&mut self, piece: TetrisPiece, rotation: Rotation) {
        match (piece.0, rotation.0) {
            (0, 0 | 2) => self.board[0] |= 0b1111_0000, // I (flat)
            (0, 1 | 3) => {
                self.board[0] |= 0b1000_0000;
                self.board[1] |= 0b0010_0000;
                self.board[2] |= 0b0000_1000;
                self.board[3] |= 0b0000_0010;
            }
            _ => todo!(),
        }
    }

    pub fn collides(&self, other: &Self) -> bool {
        self.board[0] & other.board[0] != 0
            || self.board[1] & other.board[1] != 0
            || self.board[2] & other.board[2] != 0
            || self.board[3] & other.board[3] != 0
            || self.board[4] & other.board[4] != 0
            || self.board[5] & other.board[5] != 0
            || self.board[6] & other.board[6] != 0
            || self.board[7] & other.board[7] != 0
            || self.board[8] & other.board[8] != 0
            || self.board[9] & other.board[9] != 0
            || self.board[10] & other.board[10] != 0
            || self.board[11] & other.board[11] != 0
            || self.board[12] & other.board[12] != 0
            || self.board[13] & other.board[13] != 0
            || self.board[14] & other.board[14] != 0
            || self.board[15] & other.board[15] != 0
            || self.board[16] & other.board[16] != 0
            || self.board[17] & other.board[17] != 0
            || self.board[18] & other.board[18] != 0
            || self.board[19] & other.board[19] != 0
            || self.board[20] & other.board[20] != 0
            || self.board[21] & other.board[21] != 0
            || self.board[22] & other.board[22] != 0
            || self.board[23] & other.board[23] != 0
            || self.board[24] & other.board[24] != 0
    }

    pub fn clear(&mut self) {
        for byte in 0..NUM_BYTES_FOR_BOARD {
            self.board[byte] = 0;
        }
    }

    /// Count all the bits in the board.
    pub fn count(&self) -> u8 {
        self.board
            .iter()
            .fold(0, |acc, &byte| acc + byte.count_ones() as u8)
    }

    /// A loss is defined when we have any cell filled in the
    /// top 4 rows. 4 rows = 40 cells = 40 bits = 5 bytes.
    /// Check if the top 5 bytes are non-zero.
    pub fn loss(&self) -> bool {
        (self.board[0] != 0)
            || (self.board[1] != 0)
            || (self.board[2] != 0)
            || (self.board[3] != 0)
            || (self.board[4] != 0)
    }

    pub fn from_bytes(bytes: BoardRaw) -> Self {
        Self {
            board: bytes,
            shift_buf_a: 0,
            shift_buf_b: 0,
        }
    }

    pub fn from_bytes_mut(mut self, bytes: BoardRaw) -> Self {
        self.board = bytes;
        self
    }
}

impl Default for TetrisBoard {
    fn default() -> Self {
        Self {
            board: [0; NUM_BYTES_FOR_BOARD],
            shift_buf_a: 0,
            shift_buf_b: 0,
        }
    }
}

impl Display for TetrisBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        write!(f, " {:2} {:2} {:2}\n", "r", "a", "b")?;
        for row in 0..ROWS {
            let (first_byte, second_byte) = row_to_bit_indices(row);
            write!(f, "{:2} {:2} {:2} | ", row, first_byte, second_byte)?;
            for col in 0..COLS {
                write!(f, "{}", self.get_bit(col, row) as u8)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_bit() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0, true);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_left() {
        let mut board = TetrisBoard::default();
        board.set_bit(9, 0, true); // Rightmost bit in first row
        assert!(board.get_bit(9, 0));

        board.shift_left();
        assert!(!board.get_bit(9, 0));
        assert!(board.get_bit(8, 0));
    }

    #[test]
    fn test_shift_down() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0, true); // Top-left bit
        assert!(board.get_bit(0, 0));

        board.shift_down();
        assert!(!board.get_bit(0, 0));
        assert!(board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_up() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 1, true); // Second row, first column
        assert!(board.get_bit(0, 1));

        board.shift_up();
        assert!(!board.get_bit(0, 1));
        assert!(board.get_bit(0, 0));
    }

    #[test]
    fn test_clear() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0, true);
        board.set_bit(9, 19, true);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(9, 19));

        board.clear();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));
    }

    #[test]
    fn test_loss_condition() {
        let mut board = TetrisBoard::default();
        assert!(!board.loss());

        // Set a bit in the top 4 rows
        board.set_bit(0, 3, true);
        assert!(board.loss());
    }

    #[test]
    fn test_collision() {
        let mut board1 = TetrisBoard::default();
        let mut board2 = TetrisBoard::default();

        board1.set_bit(0, 0, true);
        assert!(!board1.collides(&board2));

        board2.set_bit(0, 0, true);
        assert!(board1.collides(&board2));

        board2.set_bit(0, 0, false);
        assert!(!board1.collides(&board2));
    }

    #[test]
    fn test_traverse_down() {
        let mut board = TetrisBoard::default();
        for col in 0..COLS {
            board.set_bit(col, 0, true);
            for row in 0..ROWS {
                assert!(
                    board.get_bit(col, row),
                    "Bit at ({}, {}) is false",
                    col,
                    row
                );
                assert!(board.count() == 1);
                board.shift_down();
            }

            // test if the shift goes off the board
            assert!(
                !board.get_bit(col, ROWS - 1),
                "Bit at ({}, {}) is true",
                col,
                ROWS - 1
            );
            assert!(board.count() == 0);

            board.clear();
        }
    }

    #[test]
    fn test_traverse_up() {
        let mut board = TetrisBoard::default();
        for col in 0..COLS {
            board.set_bit(col, ROWS - 1, true);
            for row in (0..ROWS).rev() {
                assert!(
                    board.get_bit(col, row),
                    "Bit at ({}, {}) is false",
                    col,
                    row
                );
                assert!(board.count() == 1);
                board.shift_up();
            }

            // test if the shift goes off the board
            assert!(!board.get_bit(col, 0), "Bit at ({}, {}) is true", col, 0);
            assert!(board.count() == 0);

            board.clear();
        }
    }

    #[test]
    fn test_traverse_right() {
        let mut board = TetrisBoard::default();
        for row in 0..ROWS {
            board.set_bit(0, row, true);
            for col in 0..COLS {
                assert!(
                    board.get_bit(col, row),
                    "Bit at ({}, {}) is false",
                    col,
                    row
                );
                assert!(board.count() == 1);
                board.shift_right();
            }

            // test if the shift goes off the board
            assert!(
                !board.get_bit(COLS - 1, row),
                "Bit at ({}, {}) is true",
                COLS - 1,
                row
            );
            assert!(board.count() == 0);

            board.clear();
        }
    }

    // /// Traverse a bit in the (0, 0) position snaking down the board.
    // #[test]
    // fn test_bit_traversal() {
    //     let mut board = TetrisBoard::default();
    //     board.set_bit(0, 0, true);

    //     for row in 0..(ROWS - 1) {
    //         for col in 0..(COLS - 1) {
    //             if row % 2 == 0 {
    //                 board.row_shift_right();
    //                 println!("{}", board);
    //                 assert!(board.get_bit(col + 1, row));
    //             } else {
    //                 board.row_shift_left();
    //                 println!("{}", board);
    //                 assert!(board.get_bit(COLS - 1 - col, row));
    //             }
    //             assert!(board.count() == 1);
    //         }
    //         board.row_shift_down();
    //     }
    // }
}
