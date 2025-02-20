//! This module implements a bit-packed Tetris game board representation.
//!
//! The board is represented as a 20x10 grid of cells, where each cell is a single bit.
//! The bits are packed into a 25 byte array for efficient storage and manipulation.
//! The board is indexed from left to right, top to bottom.

use macros::{pack_bytes_u64, piece_u64};
use rand::Rng;

use std::fmt::Display;

const ROWS: usize = 20;
const COLS: usize = 10;
const BOARD_SIZE: usize = ROWS * COLS;
const NUM_BYTES_FOR_BOARD: usize = BOARD_SIZE / 8;
const ROW_CHUNK: usize = 4;
const BYTES_PER_ROW_CHUNK: usize = ROW_CHUNK * COLS / 8;
const BYTE_OVERLAP_PER_ROW: usize = COLS - 8;

const NUM_TETRIS_PIECES: usize = 7;

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
/// 13  | 000000 0000 |   16   |   17
/// 14  | 0000 000000 |   17   |   18
/// 15  | 00 00000000 |   18   |   19
/// 16  | 00000000 00 |   20   |   21
/// 17  | 000000 0000 |   21   |   22
/// 18  | 0000 000000 |   22   |   23
/// 19  | 00 00000000 |   23   |   24
type BoardRaw = [u8; NUM_BYTES_FOR_BOARD];

// A rotation is represented as a u8.
// 0 = 0 degrees
// 1 = 90 degrees
// 2 = 180 degrees
// 3 = 270 degrees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rotation(pub u8);

impl Rotation {
    pub fn next(&mut self) {
        *self = unsafe { std::mem::transmute((self.0.wrapping_add(1)) % 4) };
    }

    pub fn is_last(&self) -> bool {
        self.0 == 3
    }
}

/// A tetris piece is represented as a single byte,
/// where only one bit is set.
///
/// 0000_0001 = O -> 1 rotation
/// 0000_0010 = I -> 2 rotations
/// 0000_0100 = S -> 2 rotations
/// 0000_1000 = Z -> 2 rotations
/// 0001_0000 = T -> 4 rotations
/// 0010_0000 = L -> 4 rotations
/// 0100_0000 = J -> 4 rotations
/// 1000_0000 = "Empty piece"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TetrisPiece(pub u8);
impl TetrisPiece {
    pub fn new(piece: u8) -> Self {
        Self(0b0000_0001 << piece)
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0 == 0b1000_0000
    }

    #[inline(always)]
    pub fn to_empty(&mut self) {
        *self = unsafe { std::mem::transmute(0b1000_0000u8) };
    }

    #[inline(always)]
    pub fn next(&mut self) {
        *self = unsafe { std::mem::transmute(1u8 << (((self.0.trailing_zeros() as u8) + 1) % 7)) };
    }

    #[inline(always)]
    pub fn is_last(&self) -> bool {
        self.0 == 0b0100_0000
    }

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
    #[inline(always)]
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
    #[inline(always)]
    pub const fn width(&self, rotation: Rotation) -> u8 {
        let b = (self.0 == 0b0000_0010) as u8;
        if ((rotation.0 & 1_u8) | (self.0 == 0b0000_0001) as u8) == 1 {
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

        // // return (width, height)
        // match (self.0, rotation.0) {
        //     (0, _) => 2, // O piece: always 2x2

        //     (1, 0 | 2) => 4, // I piece: 4x1 when flat
        //     (1, 1 | 3) => 1, // I piece: 1x4 when tall

        //     (2..=6, 0 | 2) => 3, // S,Z,T,L,J: 3x2 when flat
        //     (2..=6, 1 | 3) => 2, // S,Z,T,L,J: 2x3 when tall

        //     _ => panic!("Invalid piece or rotation"),
        // }
    }

    /// Calculate the height of a tetris piece.
    ///
    /// See [`width`](TetrisPiece::width) for more details.
    ///
    /// The height is the number of rows the piece occupies. This is useful to know
    /// when determining the number of possible rows to place a piece.
    ///
    /// ```text
    ///   rotation
    ///   |
    /// 3 | 2  4  3  3  3  3  3
    ///   |
    /// 2 | 2  1  2  2  2  2  2
    ///   |
    /// 1 | 2  4  3  3  3  3  3
    ///   |
    /// 0 | 2  1  2  2  2  2  2
    ///   |
    ///   |______________________
    ///     0  1  2  3  4  5  6
    ///           piece
    /// ```
    #[inline(always)]
    pub const fn height(&self, rotation: Rotation) -> u8 {
        let b = (self.0 == 0b0000_0010) as u8;
        if (((rotation.0 & 1) == 0) as u8 | (self.0 == 0b0000_0001) as u8) == 1 {
            2_u8.wrapping_sub(b)
        } else {
            3_u8.wrapping_add(b)
        }
    }
}

impl Display for TetrisPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0b0000_0001 => write!(f, "O"),
            0b0000_0010 => write!(f, "I"),
            0b0000_0100 => write!(f, "S"),
            0b0000_1000 => write!(f, "Z"),
            0b0001_0000 => write!(f, "T"),
            0b0010_0000 => write!(f, "L"),
            0b0100_0000 => write!(f, "J"),
            0b1000_0000 => write!(f, "Empty"),
            _ => panic!("Invalid piece"),
        }
    }
}

/// A tetris piece bag is a random selection algorithm to prevent
/// sequences of pieces that 'garuntee' losses (like consistent
/// S & Z pieces).
///
/// A bag starts with 7 pieces, as a tetris game is played, pieces
/// are randomly removed from the bag one by one. Once the bag is
/// 'empty', a new bag is created with all the pieces.
///
/// A single bag of pieces is represented by a single byte where
/// mulitple bits are set to represent what's left in the bag.
///
/// ```text
/// +---+---+---+---+---+---+---+---+
/// | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
/// +---+---+---+---+---+---+---+---+
/// | - | J | L | T | Z | S | I | O |
/// +---+---+---+---+---+---+---+---+
/// ```
///
/// The '-' bit is reserved so when getting the 'next' possible
/// bags (represented as a u64), we can indicate which 'next'
/// bags are invalid.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TetrisPieceBag(pub u8);

impl Display for TetrisPieceBag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pieces = Vec::new();
        for i in 0..7 {
            if (self.0 & (1 << i)) != 0 {
                pieces.push(TetrisPiece::new(i).to_string());
            }
        }
        write!(f, "{:?}", pieces)
    }
}

/// Default to a full bag, and zeroed next bags
impl Default for TetrisPieceBag {
    fn default() -> Self {
        Self(0b0111_1111)
    }
}

impl TetrisPieceBag {
    const BASE_BAG: u64 = pack_bytes_u64!(
        0b0011_1111, // next bag 6
        0b0101_1111, // next bag 5
        0b0110_1111, // next bag 4
        0b0111_0111, // next bag 3
        0b0111_1011, // next bag 2
        0b0111_1101, // next bag 1
        0b0111_1110, // next bag 0
        0b0111_1111  // current bag
    );
    const FULL_BAG: [(TetrisPieceBag, TetrisPiece); 7] = [
        (
            TetrisPieceBag((Self::BASE_BAG >> 8) as u8),
            TetrisPiece(0b0000_0001),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 16) as u8),
            TetrisPiece(0b0000_0010),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 24) as u8),
            TetrisPiece(0b0000_0100),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 32) as u8),
            TetrisPiece(0b0000_1000),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 40) as u8),
            TetrisPiece(0b0001_0000),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 48) as u8),
            TetrisPiece(0b0010_0000),
        ),
        (
            TetrisPieceBag((Self::BASE_BAG >> 56) as u8),
            TetrisPiece(0b0100_0000),
        ),
    ];

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn next_bags(&self) -> NextBagsIter {
        // Check if there is exactly one piece left in the bag
        // Count bits set to 1 in self.0 using population count
        if self.0 == 0 {
            return NextBagsIter {
                next_bags: Self::FULL_BAG,
                current_bag: Self::default(),
            };
        }

        // broadcast the current bag to a u64
        // then use a mask to remove the bit that represents each
        // piece from it's respective future bag
        let next_bags =
            ((self.0 as u64) * 0x0101_0101_0101_0100_u64) & !(0x40_20_10_08_04_02_01_FF_u64);
        NextBagsIter {
            next_bags: [
                (
                    TetrisPieceBag((next_bags >> 8) as u8),
                    TetrisPiece(
                        // Assume the piece is empty (already been removed)
                        // but if the bag has the piece, then include the piece,
                        // by bitshifting 'empty' to the appropriate position.
                        0b1000_0000 >> (7 * (((self.0 & 0b0000_0001) != 0) as u8)),
                    ),
                ),
                (
                    TetrisPieceBag((next_bags >> 16) as u8),
                    TetrisPiece(0b1000_0000 >> (6 * (((self.0 & 0b0000_0010) != 0) as u8))),
                ),
                (
                    TetrisPieceBag((next_bags >> 24) as u8),
                    TetrisPiece(0b1000_0000 >> (5 * (((self.0 & 0b0000_0100) != 0) as u8))),
                ),
                (
                    TetrisPieceBag((next_bags >> 32) as u8),
                    TetrisPiece(0b1000_0000 >> (4 * (((self.0 & 0b0000_1000) != 0) as u8))),
                ),
                (
                    TetrisPieceBag((next_bags >> 40) as u8),
                    TetrisPiece(0b1000_0000 >> (3 * (((self.0 & 0b0001_0000) != 0) as u8))),
                ),
                (
                    TetrisPieceBag((next_bags >> 48) as u8),
                    TetrisPiece(0b1000_0000 >> (2 * (((self.0 & 0b0010_0000) != 0) as u8))),
                ),
                (
                    TetrisPieceBag((next_bags >> 56) as u8),
                    TetrisPiece(0b1000_0000 >> (1 * (((self.0 & 0b0100_0000) != 0) as u8))),
                ),
            ],
            current_bag: *self,
        }
    }
}

#[derive(Debug)]
pub struct NextBagsIter {
    next_bags: [(TetrisPieceBag, TetrisPiece); 7],
    current_bag: TetrisPieceBag,
}

impl Iterator for NextBagsIter {
    type Item = (TetrisPieceBag, TetrisPiece);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_bag.0 == 0 {
            return None;
        }

        let pos = self.current_bag.0.trailing_zeros() as usize;
        self.current_bag.0 &= !(1 << pos);
        Some(self.next_bags[pos])
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.current_bag.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NextBagsIter {}

/// To save on compute, we use a u64 to represent 8 bags.
/// 1 byte to represent the current bag, and 7 bytes to represent
/// the atmost 7 possible next bags.
///
/// nb - 'next bag' is the bag that will be selected next.
///
/// ```text
/// +------------------------------------------------+-----+
/// | J    |  L   |  T   |  Z   |  S   |  I   |  O   |     |
/// +------------------------------------------------+-----+
/// | nb6  | nb5  | nb4  | nb3  | nb2  | nb1  | nb0  | bag |
/// +------------------------------------------------+-----+
/// |  b7  |  b6  |  b5  |  b4  |  b3  |  b2  |  b1  |  b0 |
/// +------------------------------------------------+-----+
/// ```
pub struct BAGS(u64);

/// A bag of tetris pieces. This is used to randomly select pieces.
///
/// ```text
///   ┌────────────bag number bits──────────────────────┐┌─pieceidx─┐
///   +----+----+----+----+----+----+---+---+---+---+---+---+---+---+---+---+
///   | 15 | 14 | 13 | 12 | 11 | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
///   +----+----+----+----+----+----+---+---+---+---+---+---+---+---+---+---+
/// ```
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
// pub struct TetrisPieceBag(pub u16);

// impl TetrisPieceBag {
//     const BAG_SIZE: usize = 7;
//     const NUM_BAGS: usize = 7 * 6 * 5 * 4 * 3 * 2 * 1;

//     #[inline]
//     pub fn last_bag(&self) -> bool {
//         (self.0 >> 3) == (Self::NUM_BAGS as u16)
//     }

//     #[inline]
//     pub fn next_bag(&mut self) {
//         *self = unsafe { std::mem::transmute(self.0 + 0b0000_0000_0000_1000) };
//     }

//     #[inline]
//     pub fn last_piece(&self) -> bool {
//         (self.0 & 0b0000_0000_0000_0111) == (Self::BAG_SIZE as u16)
//     }

//     #[inline]
//     pub fn next_piece(&mut self) {
//         *self = unsafe { std::mem::transmute(self.0 + 0b0000_0000_0000_0001) };
//     }
// }

pub struct TetrisBoard {
    play_board: BoardRaw,
    piece_board: BoardRaw,

    /// Cleared rows buffer.
    cleared_rows: [u8; 3],
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
}

impl Shiftable for TetrisBoard {
    fn shift_up(&mut self) {
        // // Shift all bytes up by 1
        // self.play_board.copy_within(1..NUM_BYTES_FOR_BOARD, 0);
        // assert_level!((NUM_BYTES_FOR_BOARD - 1) < self.play_board.len());
        // self.play_board[NUM_BYTES_FOR_BOARD - 1] = 0;

        // // Shift the bits up and carry the bottom 2 bits to the next byte
        // for i in (0..(NUM_BYTES_FOR_BOARD - 1)).rev() {
        //     assert_level!(i < self.play_board.len());
        //     unsafe {
        //         self.shift_buf_b = (self.play_board.get_unchecked(i) & 0b1100_0000) >> 6; // are we at a byte boundary?
        //         *self.play_board.get_unchecked_mut(i) =
        //             (self.play_board.get_unchecked(i) << 2) | self.shift_buf_a;
        //     }
        //     self.shift_buf_a = self.shift_buf_b;
        // }

        // // Clear the shift buffer
        // self.shift_buf_a = 0;

        for i in 0..(NUM_BYTES_FOR_BOARD - 2) {
            self.play_board[i] =
                (self.play_board[i + 1] << 2) | ((self.play_board[i + 2] & 0b1100_0000) >> 6);
        }
        self.play_board[NUM_BYTES_FOR_BOARD - 2] = self.play_board[NUM_BYTES_FOR_BOARD - 1] << 2;
        self.play_board[NUM_BYTES_FOR_BOARD - 1] = 0;
    }

    fn shift_down(&mut self) {
        // // Shift all bytes down by 1
        // self.play_board.copy_within(0..(NUM_BYTES_FOR_BOARD - 1), 1);
        // // assert_level!(0 < self.play_board.len());
        // self.play_board[0] = 0;

        // // Shift the bits down and carry the top 2 bits to the next byte
        // for i in 0..NUM_BYTES_FOR_BOARD {
        //     // assert_level!(i < self.play_board.len());
        //     self.shift_buf_b = (self.play_board[i] & 0b0000_0011) << 6;
        //     self.play_board[i] = (self.play_board[i] >> 2) | self.shift_buf_a;
        //     self.shift_buf_a = self.shift_buf_b;
        // }
        // self.shift_buf_a = 0;

        // iterator pattern
        let old_board = self.play_board;
        old_board[..NUM_BYTES_FOR_BOARD - 1]
            .array_windows::<2>()
            .rev()
            .enumerate()
            .for_each(|(i, &[prev, curr])| {
                self.play_board[NUM_BYTES_FOR_BOARD - 1 - i] =
                    (curr >> 2) | ((prev & 0b0000_0011) << 6);
            });
        self.play_board[1] = old_board[0] >> 2;
        self.play_board[0] = 0;
    }
}

pub trait BitSetter {
    fn flip_bit(&mut self, col: usize, row: usize);
    fn set_bit(&mut self, col: usize, row: usize);
    fn unset_bit(&mut self, col: usize, row: usize);
    fn flip_random_bits(&mut self, num_bits: usize);
    fn get_bit(&self, col: usize, row: usize) -> bool;
}

impl BitSetter for TetrisBoard {
    fn flip_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.play_board[byte_idx] ^= 1 << bit_idx;
    }

    fn set_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.play_board[byte_idx] |= 1 << bit_idx;
    }

    fn unset_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.play_board[byte_idx] &= !(1 << bit_idx);
    }

    fn flip_random_bits(&mut self, num_bits: usize) {
        let mut rng = rand::rng();
        for _ in 0..num_bits {
            let col = rng.random_range(0..COLS);
            let row = rng.random_range(0..ROWS);
            self.flip_bit(col, row);
        }
    }

    fn get_bit(&self, col: usize, row: usize) -> bool {
        self.play_board[(row.wrapping_mul(COLS).wrapping_add(col)) >> 3]
            & (1_u8.wrapping_shl(
                7_u32.wrapping_sub(((row.wrapping_mul(COLS).wrapping_add(col)) & 7) as u32),
            ))
            != 0
    }
}

pub trait Merge {
    fn merge(&mut self, other: &Self);
}

impl Merge for TetrisBoard {
    fn merge(&mut self, other: &Self) {
        // for i in 0..NUM_BYTES_FOR_BOARD {
        //     self.play_board[i] |= other.play_board[i];
        // }

        // Use unaligned reads/writes with decreasing sizes to exactly match 25 bytes
        unsafe {
            let src = other.play_board.as_ptr();
            let dst = self.play_board.as_mut_ptr();

            // First 16 bytes (128-bit)
            let chunk128 = (src as *const u128).read_unaligned();
            let existing128 = (dst as *const u128).read_unaligned();
            (dst as *mut u128).write_unaligned(chunk128 | existing128);

            // Next 8 bytes (64-bit)
            let offset64 = 16;
            let chunk64 = (src.add(offset64) as *const u64).read_unaligned();
            let existing64 = (dst.add(offset64) as *const u64).read_unaligned();
            (dst.add(offset64) as *mut u64).write_unaligned(chunk64 | existing64);

            // Final byte (8-bit)
            let offset8 = 24;
            self.play_board[offset8] |= other.play_board[offset8];
        }
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
            let lfilled = ((self.play_board[lbi] & lmask) == lmask) as u8;
            let rfilled = ((self.play_board[rbi] & rmask) == rmask) as u8;
            if lfilled & rfilled == 0 {
                self.play_board[lbi] &= lmask;
                self.play_board[rbi] &= rmask;
                self.cleared_rows[row / 8] |= 0b1000_0000 >> (row % 8);
            }
        }
    }

    fn clear_all(&mut self) {
        self.play_board = [0_u8; NUM_BYTES_FOR_BOARD];
    }
}

impl TetrisBoard {
    pub fn add_piece_top(&mut self, piece: TetrisPiece, rotation: Rotation, col: usize) {
        match (piece.0, rotation.0, col) {
            (0, _, 0) => unsafe {
                *(self.piece_board.as_mut_ptr() as *mut u64) |= piece_u64! {
                    1100000000
                    1100000000
                    0000000000
                    0000000000
                };
            },
            _ => todo!(),
        }
    }

    pub fn collides(&self, other: &Self) -> bool {
        unsafe {
            let ptr_a = self.play_board.as_ptr();
            let ptr_b = other.play_board.as_ptr();
            let chunk1_a = std::ptr::read_unaligned(ptr_a as *const u128);
            let chunk1_b = std::ptr::read_unaligned(ptr_b as *const u128);
            let chunk2_a = std::ptr::read_unaligned(ptr_a.add(16) as *const u128);
            let chunk2_b = std::ptr::read_unaligned(ptr_b.add(16) as *const u128);
            if (chunk1_a & chunk1_b).wrapping_add(chunk2_a & chunk2_b) != 0 {
                return true;
            }
        }
        false
    }

    /// Count all the bits in the board.
    pub fn count(&self) -> u8 {
        self.play_board
            .iter()
            .fold(0, |acc, &byte| acc + byte.count_ones() as u8)
    }

    /// A loss is defined when we have any cell in the
    /// top 4 rows is filled.
    ///
    /// To compute this quickly, we read the first 8 bytes of the board
    /// as a u64, mask out the last 3 bytes, and check if the result is non-zero.
    /// Any bits in the first 5 bytes make the value positive (a loss).
    #[inline]
    pub fn loss(&self) -> bool {
        unsafe {
            (std::ptr::read_unaligned(self.play_board.as_ptr() as *const u64) & 0x000000FFFFFFFFFF)
                != 0
        }
    }

    pub fn from_bytes(bytes: BoardRaw) -> Self {
        Self {
            play_board: bytes,
            piece_board: [0; NUM_BYTES_FOR_BOARD],
            cleared_rows: [0; 3],
        }
    }

    pub fn from_bytes_mut(&mut self, bytes: BoardRaw) {
        self.play_board = bytes;
    }
}

impl Default for TetrisBoard {
    fn default() -> Self {
        Self {
            play_board: [0; NUM_BYTES_FOR_BOARD],
            piece_board: [0; NUM_BYTES_FOR_BOARD],
            cleared_rows: [0; 3],
        }
    }
}

impl TetrisBoard {
    pub fn new() -> Self {
        Self::default()
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
    use std::collections::HashSet;

    use rand::{
        Rng,
        seq::{IndexedRandom, IteratorRandom},
    };

    use super::*;

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
                let width = TetrisPiece::new(i).width(Rotation(r));
                let height = TetrisPiece::new(i).height(Rotation(r));
                let reference = dimensions_reference_slow(i, r);
                assert_eq!(
                    width,
                    reference.0,
                    "Piece: {}, Rotation: {} | width should be {}, not {}",
                    TetrisPiece(i),
                    r,
                    reference.0,
                    width,
                );
                assert_eq!(
                    height,
                    reference.1,
                    "Piece: {}, Rotation: {} | height should be {}, not {}",
                    TetrisPiece(i),
                    r,
                    reference.1,
                    height,
                );
            }
        }
    }

    #[test]
    fn test_set_and_get_bit() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_down() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0); // Top-left bit
        assert!(board.get_bit(0, 0));

        board.shift_down();
        assert!(!board.get_bit(0, 0));
        assert!(board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_up() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 1); // Second row, first column
        assert!(board.get_bit(0, 1));

        board.shift_up();
        assert!(!board.get_bit(0, 1));
        assert!(board.get_bit(0, 0));
    }

    #[test]
    fn test_merge() {
        let mut boarda = TetrisBoard::default();
        let mut boardb = TetrisBoard::default();
        boarda.set_bit(0, 0);
        boarda.set_bit(1, 1);
        boardb.set_bit(0, 0);
        boardb.set_bit(1, 1);
        boarda.merge(&boardb);
        assert!(boarda.get_bit(0, 0));
        assert!(boarda.get_bit(1, 1));

        boarda.clear_all();
        boardb.clear_all();
        assert!(!boarda.get_bit(0, 0));
        assert!(!boardb.get_bit(1, 1));
        boarda.merge(&boardb);
        assert!(!boarda.get_bit(0, 0));
        assert!(!boarda.get_bit(1, 1));

        boarda.clear_all();
        boardb.clear_all();
        // fill boardb with 1s
        for i in 0..ROWS {
            for j in 0..COLS {
                boardb.set_bit(j, i);
            }
        }
        assert!(boarda.count() == 0);
        assert!(boardb.count() == (ROWS * COLS) as u8);
        boarda.merge(&boardb);
        assert!(boarda.count() == (ROWS * COLS) as u8);
    }

    #[test]
    fn test_clear() {
        let mut board = TetrisBoard::default();
        board.set_bit(0, 0);
        board.set_bit(9, 19);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(9, 19));

        board.clear_all();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));

        // small fuzz test
        board.flip_random_bits(100);
        assert!(board.count() > 0);
        board.clear_all();
        assert!(board.count() == 0);
    }

    #[test]
    fn test_loss_condition() {
        let mut board = TetrisBoard::default();
        assert!(!board.loss());

        // Set a bit in the top 4 rows
        board.set_bit(0, 1);
        assert!(board.loss());
        board.clear_all();
        assert!(!board.loss());

        // Set a bit in the bottom row
        board.set_bit(0, ROWS - 1);
        assert!(!board.loss());
    }

    #[test]
    fn test_collision() {
        let mut board1 = TetrisBoard::default();
        let mut board2 = TetrisBoard::default();

        board1.flip_bit(0, 0);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 1);
        assert!(board2.count() == 0);

        board2.flip_bit(0, 0);
        assert!(board1.collides(&board2));
        assert!(board1.count() == 1);
        assert!(board2.count() == 1);

        board2.flip_bit(0, 0);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 1);
        assert!(board2.count() == 0);

        board1.flip_bit(4, 4);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 2);
        assert!(board2.count() == 0);

        board2.flip_bit(4, 4);
        assert!(board1.collides(&board2));
        assert!(board1.count() == 2);
        assert!(board2.count() == 1);

        board2.flip_bit(4, 4);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 2);
        assert!(board2.count() == 0);

        board1.flip_bit(5, 4);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 3);
        assert!(board2.count() == 0);

        board1.flip_bit(4, 5);
        assert!(!board1.collides(&board2));
        assert!(board1.count() == 4);
        assert!(board2.count() == 0);

        // test end of board
        board1.flip_bit(COLS - 1, ROWS - 1);
        assert!(!board1.collides(&board2));

        board2.flip_bit(COLS - 1, ROWS - 1);
        assert!(board1.collides(&board2));

        // another method
        board1.clear_all();
        board2.clear_all();
        assert!(
            !board1.collides(&board2),
            "Cleared boards should not collide"
        );

        // set the board to all ones
        board1.from_bytes_mut([255; NUM_BYTES_FOR_BOARD]);
        assert!(
            !board1.collides(&board2),
            "All ones and empty board should not collide"
        );

        for i in 0..COLS {
            for j in 0..ROWS {
                board2.flip_bit(i, j);
                assert!(board1.collides(&board2));
                board1.flip_bit(i, j);
                assert!(!board1.collides(&board2));
            }
        }
    }

    #[test]
    fn test_traverse_down() {
        let mut board = TetrisBoard::default();
        for col in 0..COLS {
            board.set_bit(col, 0);
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
            board.set_bit(col, ROWS - 1);
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

    #[test]
    fn test_bag() {
        // take 6 pieces and check that the bag has reset
        let mut bag = TetrisPieceBag::new();
        assert_eq!(bag.next_bags().count(), 7);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 6);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 5);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 4);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 3);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 2);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 1);

        bag = bag.next_bags().next().unwrap().0;
        assert_eq!(bag.next_bags().count(), 7);

        // fuzz test sampling a bunch from the bag
        let num_bags = 10_000;
        let mut all_collected_pieces = Vec::new();
        let mut bag = TetrisPieceBag::new();
        let mut rng = rand::rng();
        for _ in 0..(num_bags * NUM_TETRIS_PIECES) {
            let selected = bag.next_bags().choose(&mut rng).unwrap();
            bag = selected.0;
            all_collected_pieces.push(selected.1);
        }

        // check that all pieces of the correct are present
        assert_eq!(all_collected_pieces.len(), num_bags * NUM_TETRIS_PIECES);
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(0))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(1))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(2))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(3))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(4))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(5))
                .count(),
            num_bags
        );
        assert_eq!(
            all_collected_pieces
                .iter()
                .filter(|piece| **piece == TetrisPiece::new(6))
                .count(),
            num_bags
        );
    }

    #[test]
    fn ____test_bag() {
        #[inline(always)]
        fn permutation_7(i: u64) -> Option<u64> {
            if i >= 5040 {
                return None;
            }

            let mut used: u8 = 0;
            let mut result: u64 = 0;

            // Unroll and inline everything aggressively
            {
                let mut c = 0;
                for d in 0..7 {
                    if c == i / 720 {
                        used |= 1 << d;
                        result = d as u64 + 1;
                        break;
                    }
                    c += 1;
                }
            }

            {
                let mut c = 0;
                for d in 0..7 {
                    if (used & (1 << d)) == 0 {
                        if c == (i % 720) / 120 {
                            used |= 1 << d;
                            result = result * 10 + (d as u64 + 1);
                            break;
                        }
                        c += 1;
                    }
                }
            }

            {
                let mut c = 0;
                for d in 0..7 {
                    if (used & (1 << d)) == 0 {
                        if c == (i % 120) / 24 {
                            used |= 1 << d;
                            result = result * 10 + (d as u64 + 1);
                            break;
                        }
                        c += 1;
                    }
                }
            }

            {
                let mut c = 0;
                for d in 0..7 {
                    if (used & (1 << d)) == 0 {
                        if c == (i % 24) / 6 {
                            used |= 1 << d;
                            result = result * 10 + (d as u64 + 1);
                            break;
                        }
                        c += 1;
                    }
                }
            }

            {
                let mut c = 0;
                for d in 0..7 {
                    if (used & (1 << d)) == 0 {
                        if c == (i % 6) / 2 {
                            used |= 1 << d;
                            result = result * 10 + (d as u64 + 1);
                            break;
                        }
                        c += 1;
                    }
                }
            }

            {
                let mut c = 0;
                for d in 0..7 {
                    if (used & (1 << d)) == 0 {
                        if c == i % 2 {
                            used |= 1 << d;
                            result = result * 10 + (d as u64 + 1);
                            break;
                        }
                        c += 1;
                    }
                }
            }

            // Last digit
            for d in 0..7 {
                if (used & (1 << d)) == 0 {
                    result = result * 10 + (d as u64 + 1);
                    break;
                }
            }

            Some(result)
        }

        #[inline(always)]
        fn ith_permutation_7(i: u64) -> Option<u64> {
            if i >= 5040 {
                return None;
            }

            let fact = [720, 120, 24, 6, 2, 1];
            let mut perm = [0u8; 7];
            let mut idx = i;

            // compute factorial code
            for k in 0..6 {
                perm[k] = (idx / fact[k]) as u8;
                idx %= fact[k];
            }
            perm[6] = idx as u8;

            // readjust values to obtain the permutation
            for k in (1..7).rev() {
                for j in (0..k).rev() {
                    if perm[j] <= perm[k] {
                        perm[k] += 1;
                    }
                }
            }

            // Convert to decimal number (1-based)
            let mut result: u64 = 0;
            for &d in &perm {
                result = result * 10 + (d as u64 + 1);
            }

            Some(result)
        }

        fn swap_digits(mut n: u32, i: u32, j: u32) -> u32 {
            // Get the digit at position `i`
            let pow10_i = 1u32 << (i * 3 + i); // i * 3 + i is equivalent to i * 4, representing multiplication by log2(10) ≈ 3.32
            let digit_i = (n / pow10_i) % 10;

            // Get the digit at position `j`
            let pow10_j = 10u32.pow(j);
            let digit_j = (n / pow10_j) % 10;

            // Remove the original digits from the number
            n -= digit_i * pow10_i;
            n -= digit_j * pow10_j;

            // Insert the swapped digits
            n += digit_i * pow10_j;
            n += digit_j * pow10_i;

            n
        }
        assert_eq!(swap_digits(123456, 0, 1), 123465);

        #[inline(always)]
        fn elt(mut k: usize) -> u64 {
            // Initialize array with 0..7 directly
            let mut perm = [0, 1, 2, 3, 4, 5, 6];

            // only generate the swapings
            let mut swapings = [(0u8, 0u8); 6];
            for i in (1..7).rev() {
                swapings[i - 1] = ((k % (i + 1)) as u8, i as u8);
                k /= i + 1;
            }

            let mut final_num = 0123456_u32;
            for (i, j) in swapings {
                let i = i as u32;
                let j = j as u32;
                // Get bits at positions i and j
                let x = (final_num >> i) & 1;
                let y = (final_num >> j) & 1;
                // Clear bits at i and j, then set them to swapped values
                final_num = (final_num & !(1 << i) & !(1 << j)) | (x << j) | (y << i);
            }

            // Generate permutation using Fisher-Yates shuffle with XOR swap
            for i in (1..7).rev() {
                let j = k.wrapping_rem(i + 1);
                if i != j {
                    // Only swap if indices are different
                    unsafe {
                        *perm.get_unchecked_mut(i) ^= *perm.get_unchecked(j);
                        *perm.get_unchecked_mut(j) ^= *perm.get_unchecked(i);
                        *perm.get_unchecked_mut(i) ^= *perm.get_unchecked(j);
                    }
                }
                k /= i + 1;
            }

            // Convert to decimal number (1-based)
            let mut result: u64 = 0;
            for &d in &perm {
                result = result * 10 + (d as u64 + 1);
            }

            result
        }

        let mut set = HashSet::new();
        for i in 0..5040 {
            let perm = elt(i);
            set.insert(perm);
        }
        // assert!(
        //     set.len() == 5040,
        //     "Set length is not 5040, it is {}",
        //     set.len()
        // );
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
