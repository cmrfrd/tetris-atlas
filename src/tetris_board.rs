use borsh::{BorshDeserialize, BorshSerialize};
use macros::{piece_bytes, piece_u64};
use rand::{Rng, SeedableRng};

use std::hash::Hash;
use std::{fmt::Display, hash::Hasher};

pub const ROWS: usize = 20;
pub const COLS: usize = 10;
pub const ROWS_U8: u8 = ROWS as u8;
pub const COLS_U8: u8 = COLS as u8;
pub const BOARD_SIZE: usize = ROWS * COLS;
pub const NUM_BYTES_FOR_BOARD: usize = BOARD_SIZE / 8;
pub const ROW_CHUNK: usize = 4;
pub const BYTES_PER_ROW_CHUNK: usize = ROW_CHUNK * COLS / 8;
pub const BYTE_OVERLAP_PER_ROW: usize = COLS - 8;

pub const NUM_TETRIS_PIECES: usize = 7;

/// A rotation is represented as a u8.
///
/// ```text
/// 0 = 0 degrees
/// 1 = 90 degrees
/// 2 = 180 degrees
/// 3 = 270 degrees
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, BorshSerialize, BorshDeserialize,
)]
pub struct Rotation(pub u8);

impl Default for Rotation {
    fn default() -> Self {
        Self(0)
    }
}

impl Display for Rotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

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
/// ```text
/// 0000_0001 = 1 = O -> 1 rotation
/// 0000_0010 = 2 = I -> 2 rotations
/// 0000_0100 = 4 = S -> 2 rotations
/// 0000_1000 = 8 = Z -> 2 rotations
/// 0001_0000 = 16 = T -> 4 rotations
/// 0010_0000 = 32 = L -> 4 rotations
/// 0100_0000 = 64 = J -> 4 rotations
/// 1000_0000 = 128 = "Empty piece"
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, BorshSerialize, BorshDeserialize,
)]
pub struct TetrisPiece(pub u8);

impl Default for TetrisPiece {
    fn default() -> Self {
        Self(0b0000_0001)
    }
}

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
    pub fn is_last(&self) -> bool {
        self.0 == 0b0100_0000
    }

    /// Calculate the number of rotations for a tetris piece.
    ///
    /// The mapping for each piece is:
    /// ```text
    /// 1 -> 1
    /// 2 | 4 | 8 -> 2
    /// 16 | 32 | 64 -> 4
    /// ```
    ///
    /// NOTE: I also tried doing a lookup from `const ROTATION_MAP: [u8; 7] = [1, 2, 2, 2, 4, 4, 4];`
    ///       but this was slow!
    ///
    /// More fine grained:
    /// ```text
    /// b = 1 + (piece >= 16 && piece < 128)
    /// c = piece == 1
    /// return 2 * b - c
    /// ```
    ///
    #[inline(always)]
    pub const fn num_rotations(&self) -> u8 {
        2u8.wrapping_mul(1u8.wrapping_add(((self.0 & 0b0111_0000) != 0) as u8))
            .wrapping_sub((self.0 == 0b0000_0001) as u8)

        // // slightly faster ...
        // let tz = self.0.trailing_zeros();
        // (1_u8.wrapping_add((tz != 0) as u8)) << ((tz >> 2) & 1)

        // much slower ...
        // match self.0 {
        //     // O piece
        //     0b0000_0001 => 1,
        //     // I, S, Z pieces
        //     0b0000_0010 | 0b0000_0100 | 0b0000_1000 => 2,
        //     // T, L, J pieces
        //     0b0001_0000 | 0b0010_0000 | 0b0100_0000 => 4,
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

impl Iterator for TetrisPiece {
    type Item = TetrisPiece;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            self.0 <<= 1;
            Some(*self)
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
/// 'empty', a new bag is created with all the pieces. This process
/// is continually repeated.
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
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, BorshSerialize, BorshDeserialize,
)]
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

/// Default to a full bag
impl Default for TetrisPieceBag {
    fn default() -> Self {
        Self(0b0111_1111)
    }
}

impl TetrisPieceBag {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn inc(&mut self) {
        self.0 += 1;
    }

    #[inline(always)]
    pub fn pieces_remaining(&self) -> u8 {
        self.0.count_ones() as u8
    }

    #[inline(always)]
    pub fn merge(&mut self, other: &Self) {
        self.0 |= other.0;
    }

    #[inline(always)]
    pub fn next_bags(&self) -> NextBagsIter {
        // Check if there are no pieces left in the bag
        // If so, return the full bag
        if self.0 == 0 {
            return NextBagsIter::new(Self::default());
        }
        NextBagsIter::new(*self)
    }
}

#[derive(Debug)]
pub struct NextBagsIter {
    current_bag: TetrisPieceBag,
    next_bags_mask: u64,
}

impl NextBagsIter {
    fn new(current_bag: TetrisPieceBag) -> Self {
        Self {
            current_bag,
            // If there are pieces left in the bag, we broadcast the
            // current bag to a u64 (all possible next future bags)
            // then use a mask to remove the bit that represents each
            // piece from it's respective future bag
            next_bags_mask: ((current_bag.0 as u64) * 0x0101_0101_0101_0100_u64)
                & !0x40_20_10_08_04_02_01_FF_u64,
        }
    }
}

impl Iterator for NextBagsIter {
    type Item = (TetrisPieceBag, TetrisPiece);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_bag.0 == 0 {
            return None;
        }

        let pos = self.current_bag.0.trailing_zeros();
        self.current_bag.0 &= !(1 << pos);
        Some((
            TetrisPieceBag((self.next_bags_mask >> ((pos + 1) * 8)) as u8),
            TetrisPiece::new(pos as u8),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.current_bag.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NextBagsIter {}

/// A board is a 20x10 grid of cells. Each cell is a bit.
///
/// For conciseness, we operate on a byte by byte level.
/// The memory layout of the board is as follows:
///
/// ```text
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
/// ```
#[derive(
    PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd, BorshSerialize, BorshDeserialize,
)]
pub struct BoardRaw(pub [u8; NUM_BYTES_FOR_BOARD]);

impl Default for BoardRaw {
    fn default() -> Self {
        Self([0; NUM_BYTES_FOR_BOARD])
    }
}

impl Display for BoardRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        write!(f, " {:2} {:2} {:2}| game\n", "r", "a", "b")?;
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

/// A trait for types that can be shifted up and down,
/// such as a tetris board..
pub trait Shiftable {
    /// Shift all the rows up by 1.
    /// This is used after we 'drop' a piece and find a collision,
    /// Once a collision is found, we need to shift up the piece
    /// before adding it to the board.
    fn shift_up(&mut self);

    /// Shift all the rows down by 1.
    /// When placing a piece, we progressively shift it down until it
    /// either hits the bottom of the board or collides with another space.
    fn shift_down(&mut self);

    /// Shift all the rows down by 1 w.r.t. a given row.
    /// This is used when we want to 'clear' a row.
    /// All the row bits above the given row are shifted down
    /// by 1 row.
    fn shift_down_from(&mut self, row: usize);
}

impl Shiftable for BoardRaw {
    #[inline(always)]
    fn shift_up(&mut self) {
        for i in 0..(NUM_BYTES_FOR_BOARD - 2) {
            self.0[i] = (self.0[i + 1] << 2) | ((self.0[i + 2] & 0b1100_0000) >> 6);
        }
        self.0[NUM_BYTES_FOR_BOARD - 2] = self.0[NUM_BYTES_FOR_BOARD - 1] << 2;
        self.0[NUM_BYTES_FOR_BOARD - 1] = 0;
    }

    fn shift_down(&mut self) {
        // unsafe {
        //     std::ptr::copy(
        //         self.play_board.as_ptr(),
        //         self.play_board.as_mut_ptr().add(1),
        //         24,
        //     );
        // }
        // self.play_board[0] = 0;
        // for i in (1..NUM_BYTES_FOR_BOARD).rev() {
        //     self.play_board[i] =
        //         (self.play_board[i] >> 2) | ((self.play_board[i - 1] & 0b0000_0011) << 6);
        // }
        // self.play_board[0] >>= 2;

        let old_board = self.0;
        old_board[..NUM_BYTES_FOR_BOARD - 1]
            .array_windows::<2>()
            .rev()
            .enumerate()
            .for_each(|(i, &[prev, curr])| {
                self.0[NUM_BYTES_FOR_BOARD - 1 - i] = (curr >> 2) | ((prev & 0b0000_0011) << 6);
            });
        self.0[1] = old_board[0] >> 2;
        self.0[0] = 0;
    }

    fn shift_down_from(&mut self, row: usize) {
        if row == 0 {
            self.0[0] = 0;
            self.0[1] &= 0b0011_1111;
            return;
        }
        let end_byte = (row & !3) + (row >> 2) + (row & 3) + 1;

        self.0[end_byte] = ((self.0[end_byte - 2] & 0b0000_0011) << 6)
            | ((self.0[end_byte - 1]
                & (0b1111_1111_u8.unbounded_shl((8 - (2 * ((end_byte - 1) % 5))) as u32)))
            .unbounded_shr(2))
            | (self.0[end_byte]
                & (0b1111_1111_u8.unbounded_shr((8 - 2 * ((4 - (end_byte % 5)) % 4)) as u32)));
        for idx in (2..end_byte).rev() {
            self.0[idx] = ((self.0[idx - 2] & 0b0000_0011) << 6)
                | (self.0[idx - 1] & (0b1111_1111_u8.unbounded_shl(2 * (idx % 5 != 0) as u32)))
                    .unbounded_shr(2);
        }
        self.0[1] = self.0[0] >> 2;
        self.0[0] = 0;
    }
}

pub trait BitSetter {
    fn flip_bit(&mut self, col: usize, row: usize);
    fn set_bit(&mut self, col: usize, row: usize);
    fn unset_bit(&mut self, col: usize, row: usize);
    fn flip_random_bits(&mut self, num_bits: usize, seed: u64);
    fn get_bit(&self, col: usize, row: usize) -> bool;
}

/// Not perf critical, so we can use a naive implementation.
impl BitSetter for BoardRaw {
    fn flip_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.0[byte_idx] ^= 1 << bit_idx;
    }

    fn set_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.0[byte_idx] |= 1 << bit_idx;
    }

    fn unset_bit(&mut self, col: usize, row: usize) {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        self.0[byte_idx] &= !(1 << bit_idx);
    }

    fn flip_random_bits(&mut self, num_bits: usize, seed: u64) {
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        let mut rng = rand::rngs::StdRng::from_seed(seed_bytes);
        for _ in 0..num_bits {
            let col = rng.random_range(0..COLS);
            let row = rng.random_range(0..ROWS);
            self.flip_bit(col, row);
        }
    }

    fn get_bit(&self, col: usize, row: usize) -> bool {
        let raw_idx = row * COLS + col;
        let byte_idx = raw_idx / 8;
        let bit_idx = 7 - (raw_idx % 8) as u8;
        (self.0[byte_idx] & (1 << bit_idx)) != 0
    }
}

pub trait Mergeable {
    fn merge(&mut self, other: &Self);
}

impl Mergeable for BoardRaw {
    fn merge(&mut self, other: &Self) {
        // for i in 0..NUM_BYTES_FOR_BOARD {
        //     self.play_board[i] |= other.play_board[i];
        // }

        // self.0.copy_from_slice(&other.0);

        // Use unaligned reads/writes with decreasing sizes to exactly match 25 bytes
        unsafe {
            let src = other.0.as_ptr();
            let dst = self.0.as_mut_ptr();

            // First 16 bytes (128-bit)
            let chunk128 = (src as *const u128).read_unaligned();
            let existing128 = (dst as *const u128).read_unaligned();
            (dst as *mut u128).write_unaligned(chunk128 | existing128);

            // Next 8 bytes (64-bit)
            let chunk64 = (src.add(16) as *const u64).read_unaligned();
            let existing64 = (dst.add(16) as *const u64).read_unaligned();
            (dst.add(16) as *mut u64).write_unaligned(chunk64 | existing64);

            // Final byte (8-bit)
            self.0[24] |= other.0[24];
        }
    }
}

pub trait Clearer {
    /// Whenever a row is full, clear it and settle the rows above it.
    fn clear_rows(&mut self);

    /// Clear the entire board.
    fn clear_all(&mut self);
}

impl Clearer for BoardRaw {
    fn clear_rows(&mut self) {
        for chunk_idx in 0..(NUM_BYTES_FOR_BOARD / BYTES_PER_ROW_CHUNK) {
            let chunk_base_idx = chunk_idx * ROW_CHUNK;

            let chunk = u64::from_be_bytes(
                unsafe {
                    std::slice::from_raw_parts(
                        self.0[((chunk_base_idx & !3)
                            + (chunk_base_idx >> 2)
                            + (chunk_base_idx & 3))..]
                            .as_ptr(),
                        8,
                    )
                }
                .try_into()
                .unwrap(),
            );

            if (chunk & 0xFF_C0_00_00_00_00_00_00_u64) == 0xFF_C0_00_00_00_00_00_00_u64 {
                self.shift_down_from(chunk_base_idx);
            }
            if (chunk & (0x00_3F_F0_00_00_00_00_00_u64)) == (0x00_3F_F0_00_00_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 1);
            }
            if (chunk & (0x00_00_0F_FC_00_00_00_00_u64)) == (0x00_00_0F_FC_00_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 2);
            }
            if (chunk & (0x00_00_00_03_FF_00_00_00_u64)) == (0x00_00_00_03_FF_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 3);
            }
        }
    }

    fn clear_all(&mut self) {
        self.0 = [0_u8; NUM_BYTES_FOR_BOARD];
    }
}

pub trait Collides {
    fn collides(&self, other: &Self) -> bool;
}

impl Collides for BoardRaw {
    #[inline(always)]
    fn collides(&self, other: &Self) -> bool {
        unsafe {
            let src_a = self.0.as_ptr();
            let src_b = other.0.as_ptr();

            let chunk128_a = std::ptr::read_unaligned(src_a as *const u128);
            let chunk128_b = std::ptr::read_unaligned(src_b as *const u128);
            if chunk128_a & chunk128_b != 0 {
                return true;
            }

            let chunk128_c =
                std::ptr::read_unaligned(src_a.add(NUM_BYTES_FOR_BOARD - 16) as *const u128);
            let chunk128_d =
                std::ptr::read_unaligned(src_b.add(NUM_BYTES_FOR_BOARD - 16) as *const u128);
            if chunk128_c & chunk128_d != 0 {
                return true;
            }
            return false;
        }
    }
}

pub trait Countable {
    fn count(&self) -> usize;
}

impl Countable for BoardRaw {
    fn count(&self) -> usize {
        // unsafe {
        //     let ptr = self.0.as_ptr();
        //     ((std::ptr::read_unaligned(ptr as *const u128).count_ones())
        //         + ((std::ptr::read_unaligned(ptr.add(NUM_BYTES_FOR_BOARD - 16) as *const u128)
        //             & 0xFFFF_FFFF_FFFF_FFFF_FF00_0000_0000_0000_u128)
        //             .count_ones())) as usize
        // }
        self.0
            .iter()
            .fold(0, |acc, &byte| acc + byte.count_ones() as usize)
    }
}

pub trait FromBytes {
    fn from_bytes(bytes: BoardRaw) -> Self;
    fn from_bytes_mut(&mut self, bytes: BoardRaw);
}

impl FromBytes for BoardRaw {
    fn from_bytes(bytes: BoardRaw) -> Self {
        Self(bytes.0)
    }

    fn from_bytes_mut(&mut self, bytes: BoardRaw) {
        *self = Self(bytes.0);
    }
}

pub trait Losable {
    fn loss(&self) -> bool;
}

impl Losable for BoardRaw {
    /// A loss is defined when we have any cell in the
    /// top 4 rows is filled.
    ///
    /// To compute this quickly, we read the first 8 bytes of the board
    /// as a u64, mask out the last 3 bytes, and check if the result is non-zero.
    /// Any bits in the first 5 bytes make the value positive (a loss).
    #[inline]
    fn loss(&self) -> bool {
        unsafe {
            (std::ptr::read_unaligned(self.0.as_ptr() as *const u64) & 0x000000FFFFFFFFFF) != 0
        }
    }
}

impl BoardRaw {
    pub fn next(&self) -> Self {
        let mut board = self.clone();
        board.next_mut();
        board
    }

    #[inline(always)]
    pub fn next_mut(&mut self) {
        unsafe {
            let ptr = self.0.as_mut_ptr();

            // First try to increment the last byte (index 24)
            let last_byte = ptr.add(24);
            *last_byte = last_byte.read().wrapping_add(1);
            if *last_byte != 0 {
                return;
            }

            // If the last byte overflowed, increment the previous 8 bytes as a u64
            let chunk64_ptr = ptr.add(16) as *mut u64;
            let chunk64 = chunk64_ptr.read_unaligned();
            let new_chunk64 = chunk64.wrapping_add(1);
            chunk64_ptr.write_unaligned(new_chunk64);
            if new_chunk64 != 0 {
                return;
            }

            // If the u64 chunk overflowed, increment the first 16 bytes as a u128
            let chunk128_ptr = ptr as *mut u128;
            let chunk128 = chunk128_ptr.read_unaligned();
            let new_chunk128 = chunk128.wrapping_add(1);
            chunk128_ptr.write_unaligned(new_chunk128);
        }

        // for i in (0..NUM_BYTES_FOR_BOARD).rev() {
        //     self.0[i] = self.0[i].wrapping_add(1);
        //     if self.0[i] != 0 {
        //         break;
        //     }
        // }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TetrisBoard {
    pub play_board: BoardRaw,
    pub piece_board: BoardRaw,
}

impl PartialOrd for TetrisBoard {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.play_board.cmp(&other.play_board))
    }
}

impl Ord for TetrisBoard {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.play_board.cmp(&other.play_board)
    }
}

impl Hash for TetrisBoard {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.play_board.hash(state);
    }
}

impl TetrisBoard {
    pub fn from_raw(raw: BoardRaw) -> Self {
        Self {
            play_board: raw,
            piece_board: BoardRaw::default(),
        }
    }

    #[inline(always)]
    pub fn loss(&self) -> bool {
        self.play_board.loss()
    }

    #[inline(always)]
    pub fn happy_state(&self) -> bool {
        // if the row height is <= 4, then we are in a happy state
        unsafe {
            let ptr = self.play_board.0.as_ptr();

            // first 16 bytes (12.6 rows)
            // if any of the first 12.6 rows are filled, then
            // we are not in a happy state
            if (ptr as *const u128).read_unaligned() != 0 {
                return false;
            }

            // next 3 bytes (2.4 rows)
            // same thing here
            if ((ptr.add(16) as *const u64).read_unaligned() & 0xFF_FF_FF_FF_FF_FF_F0_00_u64) != 0 {
                return false;
            }

            return true;
        }
    }

    pub fn play_piece(&mut self, piece: TetrisPiece, rotation: Rotation, col: u8) {
        // add piece to top of piece board
        // this should always be possible
        self.add_piece_top(piece, rotation, col);
        for _ in 0..(ROWS_U8 - piece.height(rotation)) {
            self.piece_board.shift_down();
            if self.play_board.collides(&self.piece_board) {
                self.piece_board.shift_up();
                self.play_board.merge(&self.piece_board);
                self.piece_board.clear_all();
                self.play_board.clear_rows();
                return;
            }
        }
        self.play_board.merge(&self.piece_board);
        self.piece_board.clear_all();
        self.play_board.clear_rows();
    }

    fn add_piece_top(&mut self, piece: TetrisPiece, rotation: Rotation, col: u8) {
        match (piece.0, rotation.0, col) {
            (0b0000_0001, _, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1100000000
                1100000000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0110000000
                0110000000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011000000
                0011000000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001100000
                0001100000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000110000
                0000110000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011000
                0000011000
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001100
                0000001100
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000110
                0000000110
                0000000000
                0000000000
            }),
            (0b0000_0001, _, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000011
                0000000011
                0000000000
                0000000000
            }),

            // I piece
            (0b0000_0010, 0 | 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1111000000
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0111100000
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011110000
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001111000
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000111100
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011110
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 0 | 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001111
                0000000000
                0000000000
                0000000000
            }),
            (0b0000_0010, 1 | 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1000000000
                1000000000
                1000000000
                1000000000
            }),
            (0b0000_0010, 1 | 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0100000000
                0100000000
                0100000000
            }),
            (0b0000_0010, 1 | 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0010000000
                0010000000
                0010000000
            }),
            (0b0000_0010, 1 | 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001000000
                0001000000
                0001000000
            }),
            (0b0000_0010, 1 | 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000100000
                0000100000
                0000100000
            }),
            (0b0000_0010, 1 | 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000010000
                0000010000
                0000010000
            }),
            (0b0000_0010, 1 | 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001000
                0000001000
                0000001000
            }),
            (0b0000_0010, 1 | 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000100
                0000000100
                0000000100
            }),
            (0b0000_0010, 1 | 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000010
                0000000010
                0000000010
            }),
            (0b0000_0010, 1 | 3, 9) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000001
                0000000001
                0000000001
                0000000001
            }),

            // S piece
            (0b0000_0100, 0 | 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0110000000
                1100000000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011000000
                0110000000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001100000
                0011000000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000110000
                0001100000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011000
                0000110000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001100
                0000011000
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000110
                0000001100
                0000000000
                0000000000
            }),
            (0b0000_0100, 0 | 2, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000011
                0000000110
                0000000000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1000000000
                1100000000
                0100000000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0110000000
                0010000000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0011000000
                0001000000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001100000
                0000100000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000110000
                0000010000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000011000
                0000001000
                0000000000
            }),
            (0b0000_0100, 1 | 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001100
                0000000100
                0000000000
            }),
            (0b0000_0100, 1 | 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000110
                0000000010
                0000000000
            }),
            (0b0000_0100, 1 | 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000011
                0000000001
                0000000000
            }),
            // Z piece
            (0b0000_1000, 0 | 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1100000000
                0110000000
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0110000000
                0011000000
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011000000
                0001100000
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001100000
                0000110000
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000110000
                0000011000
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011000
                0000001100
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001100
                0000000110
                0000000000
                0000000000
            }),
            (0b0000_1000, 0 | 2, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000110
                0000000011
                0000000000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                1100000000
                1000000000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0110000000
                0100000000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0011000000
                0010000000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0001100000
                0001000000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000110000
                0000100000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000011000
                0000010000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000001100
                0000001000
                0000000000
            }),
            (0b0000_1000, 1 | 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000110
                0000000100
                0000000000
            }),
            (0b0000_1000, 1 | 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000001
                0000000011
                0000000010
                0000000000
            }),

            // T piece
            (0b0001_0000, 0, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1110000000
                0100000000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0111000000
                0010000000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011100000
                0001000000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001110000
                0000100000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000111000
                0000010000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011100
                0000001000
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001110
                0000000100
                0000000000
                0000000000
            }),
            (0b0001_0000, 0, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000111
                0000000010
                0000000000
                0000000000
            }),
            (0b0001_0000, 1, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                1100000000
                0100000000
                0000000000
            }),
            (0b0001_0000, 1, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0110000000
                0010000000
                0000000000
            }),
            (0b0001_0000, 1, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0011000000
                0001000000
                0000000000
            }),
            (0b0001_0000, 1, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0001100000
                0000100000
                0000000000
            }),
            (0b0001_0000, 1, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000110000
                0000010000
                0000000000
            }),
            (0b0001_0000, 1, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000011000
                0000001000
                0000000000
            }),
            (0b0001_0000, 1, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000001100
                0000000100
                0000000000
            }),
            (0b0001_0000, 1, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000110
                0000000010
                0000000000
            }),
            (0b0001_0000, 1, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000001
                0000000011
                0000000001
                0000000000
            }),
            (0b0001_0000, 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                1110000000
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0111000000
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0011100000
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0001110000
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000111000
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000011100
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000001110
                0000000000
                0000000000
            }),
            (0b0001_0000, 2, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000111
                0000000000
                0000000000
            }),
            (0b0001_0000, 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1000000000
                1100000000
                1000000000
                0000000000
            }),
            (0b0001_0000, 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0110000000
                0100000000
                0000000000
            }),
            (0b0001_0000, 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0011000000
                0010000000
                0000000000
            }),
            (0b0001_0000, 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001100000
                0001000000
                0000000000
            }),
            (0b0001_0000, 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000110000
                0000100000
                0000000000
            }),
            (0b0001_0000, 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000011000
                0000010000
                0000000000
            }),
            (0b0001_0000, 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001100
                0000001000
                0000000000
            }),
            (0b0001_0000, 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000110
                0000000100
                0000000000
            }),
            (0b0001_0000, 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000011
                0000000010
                0000000000
            }),

            // L piece
            (0b0010_0000, 0, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                1110000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0111000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0011100000
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0001110000
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000111000
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000011100
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000001110
                0000000000
                0000000000
            }),
            (0b0010_0000, 0, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000001
                0000000111
                0000000000
                0000000000
            }),
            (0b0010_0000, 1, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1000000000
                1000000000
                1100000000
                0000000000
            }),
            (0b0010_0000, 1, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0100000000
                0110000000
                0000000000
            }),
            (0b0010_0000, 1, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0010000000
                0011000000
                0000000000
            }),
            (0b0010_0000, 1, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001000000
                0001100000
                0000000000
            }),
            (0b0010_0000, 1, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000100000
                0000110000
                0000000000
            }),
            (0b0010_0000, 1, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000010000
                0000011000
                0000000000
            }),
            (0b0010_0000, 1, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001000
                0000001100
                0000000000
            }),
            (0b0010_0000, 1, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000100
                0000000110
                0000000000
            }),
            (0b0010_0000, 1, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000010
                0000000011
                0000000000
            }),
            (0b0010_0000, 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1110000000
                1000000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0111000000
                0100000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011100000
                0010000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001110000
                0001000000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000111000
                0000100000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011100
                0000010000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001110
                0000001000
                0000000000
                0000000000
            }),
            (0b0010_0000, 2, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000111
                0000000100
                0000000000
                0000000000
            }),
            (0b0010_0000, 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1100000000
                0100000000
                0100000000
                0000000000
            }),
            (0b0010_0000, 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0110000000
                0010000000
                0010000000
                0000000000
            }),
            (0b0010_0000, 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011000000
                0001000000
                0001000000
                0000000000
            }),
            (0b0010_0000, 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001100000
                0000100000
                0000100000
                0000000000
            }),
            (0b0010_0000, 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000110000
                0000010000
                0000010000
                0000000000
            }),
            (0b0010_0000, 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011000
                0000001000
                0000001000
                0000000000
            }),
            (0b0010_0000, 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001100
                0000000100
                0000000100
                0000000000
            }),
            (0b0010_0000, 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000110
                0000000010
                0000000010
                0000000000
            }),
            (0b0010_0000, 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000011
                0000000001
                0000000001
                0000000000
            }),

            // J piece
            (0b0100_0000, 0, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1000000000
                1110000000
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0111000000
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0011100000
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001110000
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000111000
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000011100
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001110
                0000000000
                0000000000
            }),
            (0b0100_0000, 0, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000111
                0000000000
                0000000000
            }),
            (0b0100_0000, 1, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1100000000
                1000000000
                1000000000
                0000000000
            }),
            (0b0100_0000, 1, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0110000000
                0100000000
                0100000000
                0000000000
            }),
            (0b0100_0000, 1, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011000000
                0010000000
                0010000000
                0000000000
            }),
            (0b0100_0000, 1, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001100000
                0001000000
                0001000000
                0000000000
            }),
            (0b0100_0000, 1, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000110000
                0000100000
                0000100000
                0000000000
            }),
            (0b0100_0000, 1, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011000
                0000010000
                0000010000
                0000000000
            }),
            (0b0100_0000, 1, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001100
                0000001000
                0000001000
                0000000000
            }),
            (0b0100_0000, 1, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000110
                0000000100
                0000000100
                0000000000
            }),
            (0b0100_0000, 1, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000011
                0000000010
                0000000010
                0000000000
            }),
            (0b0100_0000, 2, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                1110000000
                0010000000
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0111000000
                0001000000
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0011100000
                0000100000
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001110000
                0000010000
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000111000
                0000001000
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000011100
                0000000100
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001110
                0000000010
                0000000000
                0000000000
            }),
            (0b0100_0000, 2, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000111
                0000000001
                0000000000
                0000000000
            }),
            (0b0100_0000, 3, 0) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0100000000
                0100000000
                1100000000
                0000000000
            }),
            (0b0100_0000, 3, 1) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0010000000
                0010000000
                0110000000
                0000000000
            }),
            (0b0100_0000, 3, 2) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0001000000
                0001000000
                0011000000
                0000000000
            }),
            (0b0100_0000, 3, 3) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000100000
                0000100000
                0001100000
                0000000000
            }),
            (0b0100_0000, 3, 4) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000010000
                0000010000
                0000110000
                0000000000
            }),
            (0b0100_0000, 3, 5) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000001000
                0000001000
                0000011000
                0000000000
            }),
            (0b0100_0000, 3, 6) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000100
                0000000100
                0000001100
                0000000000
            }),
            (0b0100_0000, 3, 7) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000010
                0000000010
                0000000110
                0000000000
            }),
            (0b0100_0000, 3, 8) => self.piece_board.0[..5].copy_from_slice(&piece_bytes! {
                0000000001
                0000000001
                0000000011
                0000000000
            }),
            _ => unreachable!("Invalid: {} {:?} {:?}", piece, rotation, col),
        }
    }
}

impl Default for TetrisBoard {
    fn default() -> Self {
        Self {
            play_board: BoardRaw::default(),
            piece_board: BoardRaw::default(),
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
        write!(f, " {:2} {:2} {:2}| game       | piece\n", "r", "a", "b")?;
        for row in 0..ROWS {
            let first_byte = row + (row / 4);
            let second_byte = first_byte + 1;
            write!(f, "{:2} {:2} {:2} | ", row, first_byte, second_byte)?;
            for col in 0..COLS {
                write!(f, "{}", self.play_board.get_bit(col, row) as u8)?;
            }
            write!(f, " | ")?;
            for col in 0..COLS {
                write!(f, "{}", self.piece_board.get_bit(col, row) as u8)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::IteratorRandom;

    use super::*;

    #[test]
    fn test_board_count() {
        let mut board = BoardRaw::default();
        let mut count = 0;
        for row in 0..ROWS {
            for col in 0..COLS {
                board.set_bit(col, row);
                count += 1;

                let board_count = board.count();
                assert_eq!(
                    count, board_count,
                    "Count should be {}, not {}\n{}",
                    count, board_count, board
                );
            }
        }
    }

    #[test]
    fn test_board_next() {
        let mut board = BoardRaw::default();
        while board.count() < 8 {
            board.next_mut();
        }
        assert_eq!(board.count(), 8);
        assert_eq!(board.0[NUM_BYTES_FOR_BOARD - 1], 255);
    }

    #[test]
    fn test_board_raw_ord() {
        let mut board1 = BoardRaw::default();
        board1.set_bit(0, 0);
        let mut board2 = BoardRaw::default();
        board2.set_bit(0, 1);
        assert!(board2 < board1);

        // set more lower bits on the 'smaller' board
        board2.set_bit(0, ROWS - 1);
        assert!(board2 < board1);

        // two empty boards should be equal
        let board1 = BoardRaw::default();
        let board2 = BoardRaw::default();
        assert!(board1 == board2);
    }

    #[test]
    fn test_piece_rotations() {
        fn rotations_reference_slow(i: u8) -> u8 {
            // match i {
            //     0 => 1,
            //     1 | 2 | 3 => 2,
            //     4 | 5 | 6 => 4,
            //     _ => panic!("Invalid piece"),
            // }
            match i {
                0b0000_0001 => 1,
                0b0000_0010 | 0b0000_0100 | 0b0000_1000 => 2,
                0b0001_0000 | 0b0010_0000 | 0b0100_0000 => 4,
                _ => panic!("Invalid piece"),
            }
        }
        for i in 0..7 {
            let piece = TetrisPiece::new(i);
            assert_eq!(
                piece.num_rotations(),
                rotations_reference_slow(piece.0),
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
        let mut board = BoardRaw::default();
        board.set_bit(0, 0);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_down() {
        let mut board = BoardRaw::default();
        board.set_bit(0, 0); // Top-left bit
        assert!(board.get_bit(0, 0));

        board.shift_down();
        assert!(!board.get_bit(0, 0));
        assert!(board.get_bit(0, 1));

        let mut board = BoardRaw::default();
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
    fn test_shift_down_from() {
        fn is_row_filled(board: &BoardRaw, row: usize) -> bool {
            for col in 0..COLS {
                if !board.get_bit(col, row) {
                    return false;
                }
            }
            true
        }
        fn fill_row(board: &mut BoardRaw, row: usize) {
            for col in 0..COLS {
                board.set_bit(col, row);
            }
        }

        // start with empty board
        // and fill the first row
        let mut board = BoardRaw::default();
        fill_row(&mut board, 0);
        assert!(is_row_filled(&board, 0));
        assert!(board.count() == COLS);

        // test 'shift down from' operates the same as 'shift down'
        for row in 0..ROWS {
            assert!(is_row_filled(&board, row));
            board.shift_down_from(ROWS - 1);
            assert!(!is_row_filled(&board, row));
        }
        assert!(board.count() == 0);

        // start with an initial board with a diagonal of 1s
        // and shift down from the bottom
        // making sure the diagonal is still there
        let mut board = BoardRaw::default();
        for col in 0..COLS {
            // set the diagonal to 1
            board.set_bit(col, col);
        }
        for start_idx in 0..COLS {
            // check that the diagonal is still there
            for i in 0..COLS {
                assert!(
                    board.get_bit(i, start_idx + i),
                    "Bit at ({}, {}) is false",
                    i,
                    i
                );
            }

            // shift down from the bottom
            board.shift_down_from(ROWS - 1);
        }
        for i in 0..COLS {
            assert!(
                board.get_bit(i, COLS + i),
                "Bit at ({}, {}) is false",
                i,
                COLS + i
            );
        }

        // start with the shifted down board with the diagonal of ones
        // and shift from rows above the diagonal. No changes should be made
        let base_board = BoardRaw::from_bytes(BoardRaw(board.0.clone()));
        for i in 0..COLS {
            board.shift_down_from(i);
            assert_eq!(base_board, board, "Shift down from {} failed", i);
        }

        // Now start shifting from the bottom row again.
        // This time, the diagonal should be shifted down
        // off the board
        for _ in 0..COLS {
            board.shift_down_from(ROWS - 1);
        }
        assert!(board.count() == 0);

        // Now start with a full board, and shift down from random rows
        let mut board = BoardRaw::default();
        for i in 0..ROWS {
            for j in 0..COLS {
                board.set_bit(j, i);
            }
        }

        let start_row = 10;
        for i in start_row..ROWS {
            board.shift_down_from(i);
            assert!(!is_row_filled(&board, i - start_row), "Row {} is filled", i);
        }
        for i in start_row..ROWS {
            assert!(is_row_filled(&board, i), "Row {} is not filled", i);
        }

        // Now test that if we have a filled row,
        // and we shift down from that same row, it
        // get's 'cleared'
        let mut board = BoardRaw::default();
        for r in 0..ROWS {
            for i in 0..COLS {
                board.set_bit(i, r);
            }
            board.shift_down_from(r);
            assert!(board.count() == 0);
        }
    }

    #[test]
    fn test_shift_up() {
        let mut board = BoardRaw::default();
        board.set_bit(0, 1); // Second row, first column
        assert!(board.get_bit(0, 1));

        board.shift_up();
        assert!(!board.get_bit(0, 1));
        assert!(board.get_bit(0, 0));
    }

    #[test]
    fn test_merge() {
        let mut boarda = BoardRaw::default();
        let mut boardb = BoardRaw::default();
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
        assert!(boardb.count() == (ROWS * COLS));
        boarda.merge(&boardb);
        assert!(boarda.count() == (ROWS * COLS));
    }

    #[test]
    fn test_clear() {
        let mut board = BoardRaw::default();
        board.set_bit(0, 0);
        board.set_bit(9, 19);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(9, 19));

        board.clear_all();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));

        // small fuzz test
        board.flip_random_bits(100, 1);
        assert!(board.count() > 0);
        board.clear_all();
        assert!(board.count() == 0);
    }

    #[test]
    fn test_clear_filled_rows() {
        let mut board = BoardRaw::default();
        assert!(board.count() == 0);

        // set whole bottom row to 1, then clear
        for col in 0..COLS {
            board.set_bit(col, ROWS - 1);
        }
        assert!(board.count() == COLS);
        board.clear_rows();
        assert!(board.count() == 0);

        // multiple rows
        for row in [0, 3, 6, 9, 12, 15, 18] {
            for col in 0..COLS {
                board.set_bit(col, row);
            }
        }
        // println!("{}", board);
        // assert!(board.count() == COLS as u8 * 7);
        // board.set_bit(0, 18);
        board.clear_rows();
        assert!(board.count() == 0);
        board.clear_all();

        // set 1 row to 1 and a few other places
        for col in 0..COLS {
            board.set_bit(col, ROWS - 1);
        }
        board.set_bit(0, 1);
        board.set_bit(1, 1);
        board.set_bit(2, 1);
        assert!(board.count() == COLS + 3);
        board.clear_rows();
        assert!(board.count() == 3);
        board.clear_all();

        // set every other bit to 1
        for i in 0..(ROWS * COLS) {
            if i % 2 == 0 {
                board.set_bit(i % COLS, i / COLS);
            }
        }
        assert!(board.count() == (ROWS * COLS) / 2);
        board.clear_rows();
        assert!(board.count() == (ROWS * COLS) / 2);
        board.clear_all();
        assert!(board.count() == 0);
    }

    #[test]
    fn test_loss_condition() {
        let mut board = BoardRaw::default();
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
        let mut board1 = BoardRaw::default();
        let mut board2 = BoardRaw::default();

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
        board1.from_bytes_mut(BoardRaw([255; NUM_BYTES_FOR_BOARD]));
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
    fn test_traverse_up() {
        let mut board = BoardRaw::default();
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
    fn test_add_piece_top() {
        let mut board = TetrisBoard::default();

        let o_piece = TetrisPiece::new(0);
        let rotation = Rotation(0);
        let col = 0;
        board.add_piece_top(o_piece, rotation, col);
        assert!(board.piece_board.count() == 4);
        assert!(board.piece_board.get_bit(0, 0));
        assert!(board.piece_board.get_bit(1, 0));
        assert!(board.piece_board.get_bit(0, 1));
        assert!(board.piece_board.get_bit(1, 1));
    }

    #[test]
    fn test_drop_piece() {
        // simple case
        let mut board = TetrisBoard::default();
        board.play_piece(TetrisPiece::new(0), Rotation(0), 0);
        assert!(board.play_board.count() == 4);
        assert!(board.play_board.get_bit(0, ROWS - 1));
        assert!(board.play_board.get_bit(1, ROWS - 1));
        assert!(board.play_board.get_bit(0, ROWS - 2));
        assert!(board.play_board.get_bit(1, ROWS - 2));

        // fuzz test
        for _ in 0..10_000 {
            let piece = TetrisPiece::new(rand::random::<u8>() % 7);
            let rotation = Rotation(rand::random::<u8>() % 4);
            let col = (rand::random::<u8>()) % ((COLS as u8) - piece.width(rotation));
            TetrisBoard::default().play_piece(piece, rotation, col);
        }
    }
}
