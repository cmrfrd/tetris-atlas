//! This module implements a bit-packed Tetris game board representation.
//!
//! The board is represented as a 20x10 grid of cells, where each cell is a single bit.
//! The bits are packed into a 25 byte array for efficient storage and manipulation.
//! The board is indexed from left to right, top to bottom.

use crate::assert_level;
use std::fmt::Display;

const ROWS: usize = 20;
const COLS: usize = 10;
const BOARD_SIZE: usize = ROWS * COLS;
const NUM_BYTES_FOR_BOARD: usize = BOARD_SIZE / 8;
const ROW_CHUNK: usize = 4;
const BYTES_PER_ROW_CHUNK: usize = ROW_CHUNK * COLS / 8;
const BYTE_OVERLAP_PER_ROW: usize = COLS - 8;

const NUM_TETRIS_PIECES: usize = 7;

/// Macro to create a tetris piece from a visual representation.
/// Each line must be exactly COLS characters of 1s and 0s.
/// The number of rows must be exactly ROW_CHUNK.
///
/// # Example
/// ```ignore
/// let t_piece = piece_bytes! {
///     1110000000  // ███░░░░░░░
///     0100000000  // ░█░░░░░░░░
///     0000000000  // ░░░░░░░░░░
///     0000000000  // ░░░░░░░░░░
/// };
/// ```
#[macro_export]
macro_rules! piece_bytes {
    (
        $line1:literal
        $line2:literal
        $line3:literal
        $line4:literal
    ) => {{
        const _: () = assert!(
            stringify!($line1).len() == COLS
                && stringify!($line2).len() == COLS
                && stringify!($line3).len() == COLS
                && stringify!($line4).len() == COLS,
            "Each line must be exactly COLS (10) characters"
        );
        [
            u8::from_str_radix(&stringify!($line1)[0..8], 2).expect("Invalid binary string"),
            u8::from_str_radix(
                &(stringify!($line1)[8..10].to_owned() + &stringify!($line2)[0..6]),
                2,
            )
            .expect("Invalid binary string"),
            u8::from_str_radix(
                &(stringify!($line2)[6..10].to_owned() + &stringify!($line3)[0..4]),
                2,
            )
            .expect("Invalid binary string"),
            u8::from_str_radix(
                &(stringify!($line3)[4..10].to_owned() + &stringify!($line4)[0..2]),
                2,
            )
            .expect("Invalid binary string"),
            u8::from_str_radix(&stringify!($line4)[2..10], 2).expect("Invalid binary string"),
        ]
    }};
}

/// A board is a 20x10 grid of cells. Each cell is a bit.
///
/// For conciseness, we operate on a byte by byte level.
/// The memory layout of the board is as follows:
///
/// row | bit layout  | byte a | byte b
///  0  | 00000000 00 |   0    |   1
///  1  | 000000 0000 |   1    |   2
///  2  | 0000 000000 |   2    |   3
///  3  | 00 00000000 |   3    |   4
///  4  | 00000000 00 |   5    |   6
///  5  | 000000 0000 |   6    |   7
///  6  | 0000 000000 |   7    |   8
///  7  | 00 00000000 |   8    |   9
///  8  | 00000000 00 |   10   |   11
///  9  | 000000 0000 |   11   |   12
/// 10  | 0000 000000 |   12   |   13
/// 11  | 00 00000000 |   13   |   14
/// 12  | 00000000 00 |   15   |   16
/// 13  | 000000 0000 |   17   |   18
/// 14  | 0000 000000 |   18   |   19
/// 15  | 00 00000000 |   19   |   20
/// 16  | 00000000 00 |   21   |   22
/// 17  | 000000 0000 |   22   |   23
/// 18  | 0000 000000 |   23   |   24
/// 19  | 00 00000000 |   24   |   25
type BoardRaw = [u8; NUM_BYTES_FOR_BOARD];

// A rotation is represented as a u8.
// 0 = 0 degrees
// 1 = 90 degrees
// 2 = 180 degrees
// 3 = 270 degrees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rotation(pub u8);

impl Rotation {
    // When we iterate over the rotations
    pub fn next(&mut self) {
        *self = unsafe { std::mem::transmute((self.0.wrapping_add(1)) % 4) };
    }
}

/// A tetris piece is represented as a u8.
/// 0 = O -> 1 rotation
/// 1 = I -> 2 rotations
/// 2 = S -> 2 rotations
/// 3 = Z -> 2 rotations
/// 4 = T -> 4 rotations
/// 5 = L -> 4 rotations
/// 6 = J -> 4 rotations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TetrisPiece(pub u8);
impl TetrisPiece {
    /// Calculate the number of rotations for a tetris piece.
    ///
    /// The mapping for each piece is:
    /// ```text
    /// 0 -> 1
    /// 1 | 2 | 3 -> 2
    /// 4 | 5 | 6 -> 4
    /// ```
    ///
    /// In pseudo-code:
    /// ```text
    /// b = 1 if piece != 0 else 0
    /// c = 1 if piece >= 4 else 0
    /// return (1 + b) << c
    /// ```
    ///
    /// NOTE: I originally did 1 << (b + c), with the idea rotations is just '1' but shifted,
    ///       but this is slower ¯\_(ツ)_/¯. (1 + b) << c is 'maginally' faster.
    ///
    /// NOTE: I also tried doing a lookup from `const ROTATION_MAP: [u8; 7] = [1, 2, 2, 2, 4, 4, 4];`
    ///       but this was slower than the above!
    ///
    /// More fine grained:
    /// ```text
    /// b = ((x | -x) >> 7) & 1
    /// c = (x >> 2) & 1
    /// return (1 + b) << c
    /// ```
    ///
    pub const fn num_rotations(&self) -> u8 {
        (1_u8.wrapping_add((self.0 != 0) as u8)) << ((self.0 >> 2) & 1)

        // // much slower ...
        // match self.0 {
        //     0 => 1,
        //     1..=3 => 2,
        //     4..=6 => 4,
        //     _ => panic!("Invalid piece"),
        // }
    }

    /// Calculate the width of a tetris piece.
    ///
    /// The width is the number of columns the piece occupies. This is useful to know
    /// when determining the number of possible columns to place a piece.
    ///
    /// The mapping for each piece is:
    /// ```text
    /// (0, _)     => 2, // O (no different rotations)
    /// (1, 0 | 2) => 4, // I (flat)
    /// (1, 1 | 3) => 1, // I (tall)
    /// (2, 0 | 2) => 3, // S (flat)
    /// (2, 1 | 3) => 2, // S (tall)
    /// (3, 0 | 2) => 3, // Z (flat)
    /// (3, 1 | 3) => 2, // Z (tall)
    /// (4, 0 | 2) => 3, // T (flat)
    /// (4, 1 | 3) => 2, // T (tall)
    /// (5, 0 | 2) => 3, // L (flat)
    /// (5, 1 | 3) => 2, // L (tall)
    /// (6, 0 | 2) => 3, // J (flat)
    /// (6, 1 | 3) => 2, // J (tall)
    /// ```
    ///
    /// We could implement this as a lookup table, but we can get more perf gains
    /// if we think of the lookup table as follows:
    ///
    /// ```text
    ///   rotation
    ///   |
    /// 3 | 2  1  2  2  2  2  2
    ///   |
    /// 2 | 2  4  3  3  3  3  3
    ///   |
    /// 1 | 2  1  2  2  2  2  2
    ///   |
    /// 0 | 2  4  3  3  3  3  3
    ///   |
    ///   |______________________
    ///     0  1  2  3  4  5  6
    ///           piece
    /// ```
    ///
    /// With this table in mind, we can think of computing width as follows:
    ///
    /// 1. calc b if the piece is the line piece
    /// 2. branch if the rotation is odd or the square piece
    /// 3. use b to compute the width
    ///
    /// ```text
    /// b = 1 if piece=1 else 0
    /// if (rotation is odd | piece=0)
    ///     return 2 - b
    /// else
    ///     return 3 + b
    /// ```
    ///
    /// Instead of fetching an entry in a lookup table, leverage that certain
    /// widths can be grouped together based on common properties. This results
    /// in orders of magnitude faster code.
    pub const fn width(&self, rotation: Rotation) -> u8 {
        let b = (self.0 == 1) as u8;
        if ((rotation.0 & 1_u8) | (self.0 == 0) as u8) == 1 {
            2_u8.wrapping_sub(b)
        } else {
            3_u8.wrapping_add(b)
        }

        // let b0 = ((self.0 | self.0.wrapping_neg()) >> 7) & 1; // 1 if piece != 0 else 0
        // let b1 = ((self.0.wrapping_sub(2)) >> 7) ^ 1; // 1 if piece >= 2 else 0
        // let b2 = b0 & (1_u8 ^ b1); // 1 if piece=0, 0 if piece=1, 0 if piece>=2
        // // println!(
        // //     "p: {}, r: {}, b0: {:3}, b1: {:3}, b2: {:3}",
        // //     self, rotation.0, b0, b1, b2
        // // );
        // if (rotation.0 & 1) == 1 || (self.0 == 0) {
        //     2_u8.wrapping_sub(b2)
        // } else {
        //     3_u8.wrapping_add(b2)
        // }

        // let b0 = ((self.0 | self.0.wrapping_neg()) >> 7) & 1; // 1 if piece != 0 else 0
        // let b1 = ((self.0.wrapping_sub(2)) >> 7) ^ 1; // 1 if piece >= 2 else 0
        // let b2 = b0 & (1_u8 ^ b1); // 1 if piece=0, 0 if piece=1, 0 if piece>=2
        // if (!rotation.0 & 1) == 0 {
        //     2_u8.wrapping_sub(b2)
        // } else {
        //     2_u8.wrapping_add(3_u8.wrapping_mul(b2).wrapping_add(b0 & b1))
        //         .wrapping_sub(b2)
        // }

        // let b0 = ((self.0 | self.0.wrapping_neg()) >> 7) & 1; // 1 if piece != 0 else 0
        // let b1 = ((self.0.wrapping_sub(2)) >> 7) ^ 1; // 1 if piece >= 2 else 0
        // let b2 = b0 & (1_u8 ^ b1); // 1 if piece=0, 0 if piece=1, 0 if piece>=2
        // // a = 1 if rotation is even, 0 if rotation is odd
        // // re = a * (b2 * 3 + b0 & b1)
        // let re = (!rotation.0 & 1).wrapping_mul(b2.wrapping_mul(3_u8).wrapping_add(b0 & b1));
        // 2_u8.wrapping_add(re).wrapping_sub(b2)

        // let b0 = ((self.0 | self.0.wrapping_neg()) >> 7) & 1;
        // let b1 = ((self.0.wrapping_sub(2)) >> 7) ^ 1;
        // let eq0 = (1_u8 ^ b0) & (1_u8 ^ b1); // (piece=0)
        // let eq1 = b0 & (b1 ^ 1); // (piece=1)
        // let eq2 = b0 & b1; // (piece>=2)
        // let r_even = 1_u8 ^ (rotation.0 & 1);

        // let eq0_shifted = eq0 << 1; // Used in both width and height
        // let r_even_mul3 = (r_even << 1).wrapping_add(r_even); // 3 * r_even, used in both
        // let eq1_term = eq1.wrapping_mul(1_u8.wrapping_add(r_even_mul3)); // First eq1 term
        // let eq2_term = eq2.wrapping_mul(2_u8.wrapping_add(r_even)); // First eq2 term
        // (
        //     (eq0 << 1)
        //         .wrapping_add(eq1.wrapping_mul(1_u8.wrapping_add(3_u8.wrapping_mul(r_even))))
        //         .wrapping_add(eq2.wrapping_mul(2_u8.wrapping_add(r_even))),
        //     (eq0 << 1)
        //         .wrapping_add(eq1.wrapping_mul(4_u8.wrapping_sub(3_u8.wrapping_mul(r_even))))
        //         .wrapping_add(eq2.wrapping_mul(3_u8.wrapping_sub(r_even))),
        // )

        // // b0 = 1 if piece != 0 else 0
        // // Trick: (x | -x) has top bit set if x != 0. Shift >>7, &1.
        // let b0 = ((self.0 | self.0.wrapping_neg()) >> 7) & 1;

        // // b1 = 1 if piece >= 2 else 0
        // // We'll do (piece as i8).wrapping_sub(2) and take its sign bit.
        // // Then invert, so that sign=1 => piece<2 => b1=0, else => b1=1.
        // let sign = ((((self.0 as i8).wrapping_sub(2)) >> 7) as u8) & 1;
        // let b1 = 1 - sign;

        // // eq0=1 if piece=0, eq1=1 if piece=1, eq2=1 if piece>=2
        // let nb0 = 1 - b0;
        // let nb1 = 1 - b1;
        // let eq0 = nb0 & nb1; // (piece=0)
        // let eq1 = b0 & nb1; // (piece=1)
        // let eq2 = b0 & b1; // (piece>=2)

        // // r_even = 1 if rotation is even (0 or 2), else 0 (1 or 3).
        // let r_even = 1 - (rotation.0 & 1);

        // // Now compute width, height based on eq0, eq1, eq2, r_even
        // // piece=0 => always 2×2
        // // piece=1 => (4,1) if even, (1,4) if odd
        // // piece in [2..6] => (3,2) if even, (2,3) if odd
        // let width = eq0 * 2 + eq1 * (1 + 3 * r_even) + eq2 * (2 + r_even);
        // let height = eq0 * 2 + eq1 * (4 - 3 * r_even) + eq2 * (3 - r_even);
        // (width, height)

        // // return (width, height)
        // match (self.0, rotation.0) {
        //     (0, _) => 2, // O (no different rotations)

        //     (1, 0 | 2) => 4, // I (flat)
        //     (1, 1 | 3) => 1, // I (tall)

        //     (2, 0 | 2) => 3, // S (flat)
        //     (2, 1 | 3) => 2, // S (tall)

        //     (3, 0 | 2) => 3, // Z (flat)
        //     (3, 1 | 3) => 2, // Z (tall)

        //     (4, 0 | 2) => 3, // T (flat)
        //     (4, 1 | 3) => 2, // T (tall)

        //     (5, 0 | 2) => 3, // L (flat)
        //     (5, 1 | 3) => 2, // L (tall)

        //     (6, 0 | 2) => 3, // J (flat)
        //     (6, 1 | 3) => 2, // J (tall)

        //     _ => panic!("Invalid piece or rotation"),
        // }
    }
}

impl Display for TetrisPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0 => write!(f, "O"),
            1 => write!(f, "I"),
            2 => write!(f, "S"),
            3 => write!(f, "Z"),
            4 => write!(f, "T"),
            5 => write!(f, "L"),
            6 => write!(f, "J"),
            _ => panic!("Invalid piece"),
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
        // self.shift_buf_a = (self.board[NUM_BYTES_FOR_BOARD - 1] & 0b1000_0000) >> 7;
        // self.board[NUM_BYTES_FOR_BOARD - 1] = self.board[NUM_BYTES_FOR_BOARD - 1] << 1;
        // for i in (0..NUM_BYTES_FOR_BOARD - 1).rev() {
        //     self.board[i] = (self.board[i] << 1) | self.shift_buf_a;
        //     self.shift_buf_a = (self.board[i] & 0b1000_0000) >> 7;
        // }
        todo!()
    }

    fn shift_right(&mut self) {
        // self.shift_buf_a = (self.board[0] & 0b0000_0001) << 7;
        // self.board[0] = self.board[0] >> 1;
        // for i in 1..NUM_BYTES_FOR_BOARD {
        //     self.board[i] = (self.board[i] >> 1) | self.shift_buf_a;
        //     let mask = !(0b1000_0000_u8.unbounded_shr((2 * (i % 5)) as u32));
        //     println!("i: {}, mask: {:#010b}", i, mask);
        //     self.board[i] &= mask;
        //     self.shift_buf_a = (self.board[i] & 0b0000_0001) << 7;

        //     // 0 -> 0b0111_1111
        //     // 1 -> 0b1101_1111
        //     // 2 -> 0b1111_0111
        //     // 3 -> 0b1111_1101
        //     // 4 -> 0b1111_1111
        // }
        // println!("{}", self);
        todo!();
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

pub trait Clearer {
    /// Whenever a row is full, clear it and settle the rows above it.
    fn clear_filled_rows(&mut self);

    /// Clear the entire board.
    fn clear_all(&mut self);
}

impl Clearer for TetrisBoard {
    fn clear_filled_rows(&mut self) {
        for row in 0..ROWS {
            let lbi = ROW_CHUNK
                .wrapping_mul(row.wrapping_div(ROW_CHUNK))
                .wrapping_add(row % ROW_CHUNK);
            let rbi = lbi.wrapping_add(1);
            let lmask =
                !(0b1111_1111_u8.wrapping_shl(lbi.wrapping_mul(BYTE_OVERLAP_PER_ROW) as u32));
            let rmask =
                !(0b1111_1111_u8.wrapping_shr(rbi.wrapping_mul(BYTE_OVERLAP_PER_ROW) as u32));
            let lfilled = (self.board[lbi] & lmask) == lmask;
            let rfilled = (self.board[rbi] & rmask) == rmask;
            if lfilled || rfilled {
                self.board[lbi] &= lmask;
                self.board[rbi] &= rmask;
            }
        }
    }

    fn clear_all(&mut self) {
        self.board = [0; NUM_BYTES_FOR_BOARD];
    }
}

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
        self.board
            .iter()
            .zip(other.board.iter())
            .any(|(&a, &b)| a & b != 0)
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
            let first_byte = row + (row / 4);
            let second_byte = first_byte + 1;
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
    use rand::Rng;

    use super::*;

    #[test]
    fn test_piece_bytes_macro() {
        let t_piece = piece_bytes! {
            1110000000
            0100000000
            0000000000
            0000000000
        };
        assert_eq!(
            t_piece,
            [0b11100000, 0b00010000, 0b00000000, 0b00000000, 0b00000000]
        );
    }

    #[test]
    fn test_piece_rotations() {
        fn rotations_reference_slow(i: u8) -> u8 {
            match i {
                0 => 1,
                1 | 2 | 3 => 2,
                4 | 5 | 6 => 4,
                _ => panic!("Invalid piece"),
            }
        }
        for i in 0..7 {
            let piece = TetrisPiece(i);
            assert_eq!(
                piece.num_rotations(),
                rotations_reference_slow(i),
                "Piece: {}, Rotations: {}",
                piece,
                piece.num_rotations()
            );
        }
    }

    #[test]
    fn test_piece_dimensions() {
        fn dimensions_reference_slow(i: u8, r: u8) -> (u8, u8) {
            // return (width, height)
            match (i, r) {
                (0, _) => (2, 2), // O (no different rotations)

                (1, 0 | 2) => (4, 1), // I (flat)
                (1, 1 | 3) => (1, 4), // I (tall)

                (2, 0 | 2) => (3, 2), // S (flat)
                (2, 1 | 3) => (2, 3), // S (tall)

                (3, 0 | 2) => (3, 2), // Z (flat)
                (3, 1 | 3) => (2, 3), // Z (tall)

                (4, 0 | 2) => (3, 2), // T (flat)
                (4, 1 | 3) => (2, 3), // T (tall)

                (5, 0 | 2) => (3, 2), // L (flat)
                (5, 1 | 3) => (2, 3), // L (tall)

                (6, 0 | 2) => (3, 2), // J (flat)
                (6, 1 | 3) => (2, 3), // J (tall)

                _ => panic!("Invalid piece or rotation"),
            }
        }

        for i in 0..7 {
            for r in 0..4 {
                let created = TetrisPiece(i).width(Rotation(r));
                let reference = dimensions_reference_slow(i, r);
                assert_eq!(
                    created,
                    reference.0,
                    "Piece: {}, Rotation: {} should be {}x{}, not {}x{}",
                    TetrisPiece(i),
                    r,
                    reference.0,
                    reference.1,
                    created,
                    ""
                );
            }
        }
    }

    #[test]
    fn test_set_and_get_bit() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0, true);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
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

        board.clear_all();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));

        // small fuzz test
        for _ in 0..100 {
            board.set_bit(
                rand::rng().random_range(0..COLS),
                rand::rng().random_range(0..ROWS),
                rand::rng().random(),
            );
        }
        board.clear_all();
        assert!(board.count() == 0);
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

            board.clear_all();
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

            board.clear_all();
        }
    }

    // #[test]
    // fn test_traverse_right() {
    //     let mut board = TetrisBoard::default();
    //     for row in 0..ROWS {
    //         board.set_bit(0, row, true);
    //         for col in 0..COLS {
    //             assert!(
    //                 board.get_bit(col, row),
    //                 "Bit at ({}, {}) is false",
    //                 col,
    //                 row
    //             );
    //             assert!(board.count() == 1);
    //             board.shift_right();
    //         }

    //         // test if the shift goes off the board
    //         assert!(
    //             !board.get_bit(COLS - 1, row),
    //             "Bit at ({}, {}) is true",
    //             COLS - 1,
    //             row
    //         );
    //         assert!(board.count() == 0);

    //         board.clear_all();
    //     }
    // }

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
