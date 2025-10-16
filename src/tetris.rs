use proc_macros::{inline_conditioned, piece_u32_cols};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngCore};
use rayon::prelude::*;

use std::fmt::Display;
use std::hash::Hash;
use std::ops::Range;
use std::ops::{Index, IndexMut};

use crate::repeat_idx_generic;
use crate::utils::fmix64;
use crate::utils::{HeaplessVec, rshift_slice_from_mask_u32, trailing_zeros_all};

const ROWS: usize = 20;
const ACTUAL_ROWS: usize = u32::BITS as usize;
const COLS: usize = 10;
const BOARD_SIZE: usize = ROWS * COLS;
const NUM_TETRIS_PIECES: usize = 7;

pub const NUM_TETRIS_CELL_STATES: usize = 2;

/// A rotation is an orientation of a Tetris piece. For
/// simplicity, we represent a rotation as a u8. There are
/// 4 maximum possible rotations for each piece.
///
/// ```text
/// 0 = 0 degrees
/// 1 = 90 degrees
/// 2 = 180 degrees
/// 3 = 270 degrees
/// ```
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Rotation(u8);

impl Display for Rotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Rotation {
    const MAX: u8 = 4;

    /// Get the next rotation by rotating an additional 90 degrees.
    ///
    /// ```text
    /// 0 -> 1
    /// 1 -> 2
    /// 2 -> 3
    /// 3 -> 0
    /// ```
    pub fn next(&mut self) {
        *self = unsafe { std::mem::transmute((self.0.wrapping_add(1)) % Self::MAX) };
    }

    pub fn is_last(&self) -> bool {
        self.0 == 3
    }
}

/// A column index on the Tetris board.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Default)]
pub struct Column(u8);

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column({})", self.0)
    }
}

impl Column {
    pub const MAX: u8 = COLS as u8;
}

/// A Tetris piece is a tetromino. There are 7 possible tetrominoes,
/// and we use a single byte with a single bit set to represent any one of them.
/// The index of the bit set in the byte represents the tetromino.
///
/// ```text
/// 0000_0001 = 1   = O -> 1 rotation
/// 0000_0010 = 2   = I -> 2 rotations
/// 0000_0100 = 4   = S -> 2 rotations
/// 0000_1000 = 8   = Z -> 2 rotations
/// 0001_0000 = 16  = T -> 4 rotations
/// 0010_0000 = 32  = L -> 4 rotations
/// 0100_0000 = 64  = J -> 4 rotations
/// 1000_0000 = 128 = "Null piece"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TetrisPiece(u8);

impl Default for TetrisPiece {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Distribution<TetrisPiece> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> TetrisPiece {
        TetrisPiece::new(rng.random_range(0..NUM_TETRIS_PIECES) as u8)
    }
}

impl TetrisPiece {
    pub const NUM_PIECES: usize = NUM_TETRIS_PIECES;

    pub const O_PIECE: Self = Self(0b0000_0001);
    pub const I_PIECE: Self = Self(0b0000_0010);
    pub const S_PIECE: Self = Self(0b0000_0100);
    pub const Z_PIECE: Self = Self(0b0000_1000);
    pub const T_PIECE: Self = Self(0b0001_0000);
    pub const L_PIECE: Self = Self(0b0010_0000);
    pub const J_PIECE: Self = Self(0b0100_0000);
    pub const NULL_PIECE: Self = Self(0b1000_0000);

    const DEFAULT: Self = Self::O_PIECE;

    /// Get all pieces as a slice.
    #[inline_conditioned(always)]
    pub const fn all() -> [Self; NUM_TETRIS_PIECES] {
        [
            Self::O_PIECE,
            Self::I_PIECE,
            Self::S_PIECE,
            Self::Z_PIECE,
            Self::T_PIECE,
            Self::L_PIECE,
            Self::J_PIECE,
        ]
    }

    /// Create a new Tetris piece from a `u8` representing the piece.
    ///
    /// ```text
    /// 0 = O
    /// 1 = I
    /// 2 = S
    /// 3 = Z
    /// 4 = T
    /// 5 = L
    /// 6 = J
    /// ```
    #[inline_conditioned(always)]
    pub const fn new(piece: u8) -> Self {
        Self(0b0000_0001 << piece)
    }

    /// Get the index of the piece.
    #[inline_conditioned(always)]
    pub const fn index(&self) -> u8 {
        self.0.trailing_zeros() as u8
    }

    /// From index (alias for `new`)
    #[inline_conditioned(always)]
    pub const fn from_index(index: u8) -> Self {
        Self::new(index)
    }

    #[inline_conditioned(always)]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0b1000_0000
    }

    #[inline_conditioned(always)]
    pub const fn to_empty(&mut self) {
        *self = unsafe { std::mem::transmute(0b1000_0000u8) };
    }

    #[inline_conditioned(always)]
    pub const fn is_last(&self) -> bool {
        self.0 == 0b0100_0000
    }

    /// Calculate the number of rotations for a Tetris piece.
    ///
    /// The mapping for each piece is:
    ///
    /// ```text
    /// 1 -> 1
    /// 2 | 4 | 8 -> 2
    /// 16 | 32 | 64 -> 4
    /// ```
    ///
    /// NOTE: I tried a simple lookup `[1, 2, 2, 2, 4, 4, 4]`,
    /// but this was slower.
    ///
    /// A more fine-grained approach for this function is:
    ///
    /// ```text
    /// b = 1 + (piece >= 16 && piece < 128)
    /// c = piece == 1
    /// return 2 * b - c
    /// ```
    ///
    #[inline_conditioned(always)]
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

    /// Calculate the width of a Tetris piece.
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
    /// We could implement this as a lookup table, but we can get more performance
    /// by thinking of the lookup table as follows:
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
    /// 1. Compute `b` if the piece is the line piece
    /// 2. Branch if the rotation is odd or the square piece
    /// 3. Use `b` to compute the width
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
    /// widths can be grouped based on common properties. This results in
    /// significantly faster code.
    #[inline_conditioned(always)]
    pub const fn width(&self, rotation: Rotation) -> u8 {
        // let b = (self.0 == Self::I_PIECE.0) as u8;
        // let cond = ((rotation.0 & 1_u8) | (self.0 == Self::O_PIECE.0) as u8) == 1;
        // let sub = 2_u8.wrapping_sub(b);
        // let add = 3_u8.wrapping_add(b);
        // (sub & (0_u8.wrapping_sub(cond as u8))) | (add & (0_u8.wrapping_sub(!cond as u8)))

        let b = (self.0 == Self::I_PIECE.0) as u8;
        if ((rotation.0 & 1_u8) | (self.0 == Self::O_PIECE.0) as u8) == 1 {
            2_u8 - b
        } else {
            3_u8 + b
        }

        // // slow implementation
        // match (self.0, rotation.0) {
        //     (0, _) => 2, // O piece: always 2x2

        //     (1, 0 | 2) => 4, // I piece: 4x1 when flat
        //     (1, 1 | 3) => 1, // I piece: 1x4 when tall

        //     (2..=6, 0 | 2) => 3, // S,Z,T,L,J: 3x2 when flat
        //     (2..=6, 1 | 3) => 2, // S,Z,T,L,J: 2x3 when tall

        //     _ => panic!("Invalid piece or rotation"),
        // }
    }

    /// Calculate the height of a Tetris piece.
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
    #[inline_conditioned(always)]
    pub const fn height(&self, rotation: Rotation) -> u8 {
        // let cond = (((rotation.0 & 1) == 0) as u8 | (self.0 == Self::O_PIECE.0) as u8) == 1;
        // let b = (self.0 == Self::I_PIECE.0) as u8;
        // let sub = 2_u8.wrapping_sub(b);
        // let add = 3_u8.wrapping_add(b);
        // (sub & (0_u8.wrapping_sub(cond as u8))) | (add & (0_u8.wrapping_sub(!cond as u8)))

        let b = (self.0 == Self::I_PIECE.0) as u8;
        if (((rotation.0 & 1) == 0) as u8 | (self.0 == Self::O_PIECE.0) as u8) == 1 {
            2_u8 - b
        } else {
            3_u8 + b
        }
    }
}

impl Display for TetrisPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            val if val == Self::O_PIECE.0 => write!(f, "O"),
            val if val == Self::I_PIECE.0 => write!(f, "I"),
            val if val == Self::S_PIECE.0 => write!(f, "S"),
            val if val == Self::Z_PIECE.0 => write!(f, "Z"),
            val if val == Self::T_PIECE.0 => write!(f, "T"),
            val if val == Self::L_PIECE.0 => write!(f, "L"),
            val if val == Self::J_PIECE.0 => write!(f, "J"),
            val if val == Self::NULL_PIECE.0 => write!(f, "Empty"),
            _ => panic!("Invalid piece"),
        }
    }
}

/// A Tetris orientation is a (rotation, column) pair that defines how a piece is positioned on the board.
/// Each piece type only allows certain orientations.
/// The following shows which pieces can be placed in each cell, laid out in row-major order:
/// First row:  [OIS]
/// Second row: [ZTL]  
/// Third row:  [J  ]
///
/// The "X" cell is reserved for the "null" orientation.
///
/// Columns →
/// Rotations ↓
/// ```text
///       0   1   2   3   4   5   6   7   8   9
///     +---+---+---+---+---+---+---+---+---+---+
///  0  |OIS|OIS|OIS|OIS|OIS|OIS|OIS|O S|O  |XXX|
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |XXX|
///     |J  |J  |J  |J  |J  |J  |J  |J  |   |XXX|
///     +---+---+---+---+---+---+---+---+---+---+
///  1  | IS| IS| IS| IS| IS| IS| IS| IS| IS| I |
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |J  |   |
///     +---+---+---+---+---+---+---+---+---+---+
///  2  |  S|  S|  S|  S|  S|  S|  S|  S|   |   |
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |   |   |
///     +---+---+---+---+---+---+---+---+---+---+
///  3  |  S|  S|  S|  S|  S|  S|  S|  S|  S|   |
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |J  |   |
///     +---+---+---+---+---+---+---+---+---+---+
/// ```
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Default)]
pub struct TetrisPieceOrientation {
    pub rotation: Rotation,
    pub column: Column,
}

impl Display for TetrisPieceOrientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TetrisPieceOrientation(rotation: {}, column: {})",
            self.rotation, self.column
        )
    }
}

impl TetrisPieceOrientation {
    pub const NUM_ORIENTATIONS: usize = Rotation::MAX as usize * Column::MAX as usize;

    pub const DEFAULT: Self = Self {
        rotation: Rotation(0),
        column: Column(0),
    };

    pub const NULL_ORIENTATION: Self = Self {
        rotation: Rotation(0),
        column: Column(Column::MAX),
    };

    pub const ALL: [Self; Self::NUM_ORIENTATIONS] = {
        let mut orientations = [Self::DEFAULT; Self::NUM_ORIENTATIONS];
        let mut i = 0;
        while i < orientations.len() {
            orientations[i] = Self {
                rotation: Rotation(i as u8 / Column::MAX),
                column: Column(i as u8 % Column::MAX),
            };
            i += 1;
        }
        orientations
    };

    // const NUM_ORIENTATIONS_BY_PIECE: [usize; NUM_TETRIS_PIECES] = [
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::O_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::I_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::S_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::Z_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::T_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::L_PIECE),
    //     TetrisPieceOrientation::num_orientations_from_piece(TetrisPiece::J_PIECE),
    // ];

    // /// Get the number of unique orientations for a piece.
    // ///
    // /// O piece: 1 rotation, 9 columns -> 9 orientations
    // const fn num_orientations_from_piece(piece: TetrisPiece) -> usize {
    //     let num_rotations = piece.num_rotations();
    //     let mut num_orientations = 0;
    //     let mut i = 0;
    //     while i < num_rotations {
    //         let width = piece.width(Rotation(i));
    //         let num_placeable_columns = (COLS as u8) - width + 1;
    //         num_orientations += num_placeable_columns;
    //         i += 1;
    //     }
    //     num_orientations as usize
    // }

    pub const fn new_from_piece(piece: TetrisPiece, rotation: Rotation, column: Column) -> Self {
        let rotation = Rotation(rotation.0 % piece.num_rotations());
        Self { rotation, column }
    }

    /// Get the index of this orientation.
    ///
    /// There are a finite number of possible orientations for a piece.
    /// This function returns the numeric index of this orientation
    pub const fn index(self) -> u8 {
        self.rotation.0 * Column::MAX + self.column.0
    }

    /// Get an orientation from an index.
    ///
    /// This is the inverse of [`index`](TetrisPieceOrientation::index).
    #[inline_conditioned(always)]
    pub const fn from_index(index: u8) -> Self {
        Self {
            rotation: Rotation(index / Column::MAX),
            column: Column(index % Column::MAX),
        }
    }

    /// Get a binary mask of all orientations for a piece.
    ///
    /// This is useful for creating a mask for a piece in a batch.
    pub fn binary_mask_from_piece(piece: TetrisPiece) -> [u8; Self::NUM_ORIENTATIONS] {
        let mut mask = [0u8; Self::NUM_ORIENTATIONS];
        for r in 0..piece.num_rotations() {
            for c in 0..(COLS as u8) - piece.width(Rotation(r)) + 1 {
                let orientation = Self::new_from_piece(piece, Rotation(r), Column(c));
                mask[orientation.index() as usize] = 1;
            }
        }
        mask
    }
}

/// A piece placement contains the following:
///
/// - The Tetris piece
/// - The orientation of the piece (rotation and column)
///
/// We use this struct to represent the "how" we are placing
/// a piece on a tetris board
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Default)]
pub struct TetrisPiecePlacement {
    pub piece: TetrisPiece,
    pub orientation: TetrisPieceOrientation,
}

impl Display for TetrisPiecePlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TetrisPiecePlacement(piece: {}, orientation: {})",
            self.piece, self.orientation
        )
    }
}

impl Distribution<TetrisPiecePlacement> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> TetrisPiecePlacement {
        TetrisPiecePlacement::random(rng)
    }
}

impl TetrisPiecePlacement {
    const DEFAULT: Self = Self {
        piece: TetrisPiece::DEFAULT,
        orientation: TetrisPieceOrientation::DEFAULT,
    };

    // Precompute placement counts per piece
    // This is also the number of orientations for each piece
    const PIECE_PLACEMENT_COUNTS: [usize; NUM_TETRIS_PIECES] = {
        let mut counts = [0; NUM_TETRIS_PIECES];
        let mut i = 0;
        let all_placements = TetrisPiecePlacement::all_placements();
        while i < all_placements.len() {
            let piece = all_placements[i].piece;
            let piece_idx = piece.index() as usize;
            counts[piece_idx] += 1;
            i += 1;
        }
        counts
    };

    // Precompute start indices for each piece in the global array
    const PIECE_START_INDICES: [usize; NUM_TETRIS_PIECES] = {
        let mut indices = [0; NUM_TETRIS_PIECES];
        let mut i = 1;
        while i < NUM_TETRIS_PIECES {
            indices[i] = indices[i - 1] + TetrisPiecePlacement::PIECE_PLACEMENT_COUNTS[i - 1];
            i += 1;
        }
        indices
    };

    // Precompute the maximum number of placements for a piece
    // (this helps create a fixed size array for all placements)
    pub const MAX_PIECE_PLACEMENT_COUNT: usize = {
        let mut max_count = 0;

        let all_pieces = TetrisPiece::all();
        let mut piece_idx = 0;
        while piece_idx < all_pieces.len() {
            let count = TetrisPiecePlacement::PIECE_PLACEMENT_COUNTS[piece_idx];
            if count > max_count {
                max_count = count;
            }
            piece_idx += 1;
        }
        max_count
    };

    /// Get all possible placements for all pieces.
    pub const NUM_PLACEMENTS: usize = Self::calculate_num_placements();

    pub const ALL_PLACEMENTS: [Self; Self::NUM_PLACEMENTS] = Self::all_placements();

    /// Calculate the number of possible placements for all pieces.
    ///
    /// This is const so we can use it for pre-calculating all placement
    /// structs themselves.
    const fn calculate_num_placements() -> usize {
        let all_pieces = TetrisPiece::all();
        let mut total_placements = 0;

        let mut i = 0;
        while i < all_pieces.len() {
            let piece = all_pieces[i];
            let mut r = 0;
            while r < piece.num_rotations() {
                total_placements += (((COLS as u8) - piece.width(Rotation(r))) + 1) as usize;
                r += 1;
            }
            i += 1;
        }

        total_placements
    }

    const fn all_placements() -> [Self; Self::NUM_PLACEMENTS] {
        let mut placements: [Self; Self::NUM_PLACEMENTS] = [Self::DEFAULT; Self::NUM_PLACEMENTS];

        let all_pieces = TetrisPiece::all();

        let mut placement_id = 0;

        let mut i = 0;
        while i < all_pieces.len() {
            let piece = all_pieces[i];
            let mut r = 0;
            while r < piece.num_rotations() {
                let mut c = 0;
                while c <= (COLS as u8) - piece.width(Rotation(r)) {
                    placements[placement_id] = Self {
                        piece,
                        orientation: TetrisPieceOrientation {
                            rotation: Rotation(r),
                            column: Column(c),
                        },
                    };
                    placement_id += 1;
                    c += 1;
                }
                r += 1;
            }
            i += 1;
        }
        placements
    }

    /// Get the indices of all placements for a given piece.
    pub const fn indices_from_piece(piece: TetrisPiece) -> Range<usize> {
        let piece_idx = piece.index() as usize;
        let start = TetrisPiecePlacement::PIECE_START_INDICES[piece_idx];
        let count = TetrisPiecePlacement::PIECE_PLACEMENT_COUNTS[piece_idx];

        start..start + count
    }

    /// Get all possible placements for a given piece
    pub fn all_from_piece(piece: TetrisPiece) -> &'static [Self] {
        let indices = TetrisPiecePlacement::indices_from_piece(piece);
        &TetrisPiecePlacement::ALL_PLACEMENTS[indices]
    }

    /// Get the number of placements for a given piece.
    pub const fn num_placements(piece: TetrisPiece) -> usize {
        let piece_idx = piece.index() as usize;
        TetrisPiecePlacement::PIECE_PLACEMENT_COUNTS[piece_idx]
    }

    /// Get the index of this placement in the list of all possible placements.
    #[inline_conditioned(always)]
    pub const fn index(&self) -> u8 {
        let piece_idx = self.piece.index() as usize;

        // Most pieces follow pattern: 8 columns (flat), 9 columns (tall), repeat
        // Except O piece (always 9) and I piece (7 flat, 10 tall)
        let rotation_offset: u8 = match piece_idx {
            0 => 0,                                            // O piece: only 1 rotation
            1 => (self.orientation.rotation.0 != 0) as u8 * 7, // I piece
            _ => {
                // S, Z, T, L, J pieces all follow same pattern
                let r = self.orientation.rotation.0;
                r * 8 + (r / 2) // 0→0, 1→8, 2→17, 3→25
            }
        };

        TetrisPiecePlacement::PIECE_START_INDICES[piece_idx] as u8
            + rotation_offset
            + self.orientation.column.0
    }

    pub const fn from_index(index: u8) -> Self {
        Self::ALL_PLACEMENTS[index as usize]
    }

    /// Get a random placement from the list of all possible placements.
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::ALL_PLACEMENTS[rng.random_range(0..Self::NUM_PLACEMENTS)]
    }

    pub const fn piece_orientation(&self) -> (TetrisPiece, TetrisPieceOrientation) {
        (self.piece, self.orientation)
    }

    // /// Orientations allow us to specify "how" to apply a piece
    // /// to a board.

    // // Precompute the number of unique orientations for all pieces
    // const UPPER_BOUND_ORIENTATIONS: usize = Rotation::MAX_ROTATIONS as usize * COLS;
    // pub const NUM_UNIQUE_ORIENTATIONS: usize = {
    //     let mut num_orientations = 0;
    //     let mut orientation_set: [bool; TetrisPiecePlacement::UPPER_BOUND_ORIENTATIONS] =
    //         [false; TetrisPiecePlacement::UPPER_BOUND_ORIENTATIONS];
    //     let mut i = 0;
    //     let all_placements = TetrisPiecePlacement::all_placements();
    //     while i < all_placements.len() {
    //         let placement = all_placements[i];
    //         let orientation = placement.orientation;
    //         let orientation_idx = orientation.index() as usize;
    //         if !orientation_set[orientation_idx] {
    //             orientation_set[orientation_idx] = true;
    //             num_orientations += 1;
    //         }
    //         i += 1;
    //     }
    //     num_orientations
    // };

    // pub const ALL_UNIQUE_ORIENTATIONS: [TetrisPieceOrientation; Self::NUM_UNIQUE_ORIENTATIONS] =
    //     Self::all_unique_orientations();

    // const fn all_unique_orientations() -> [TetrisPieceOrientation; Self::NUM_UNIQUE_ORIENTATIONS] {
    //     let mut orientations: [TetrisPieceOrientation; Self::NUM_UNIQUE_ORIENTATIONS] =
    //         [TetrisPieceOrientation::DEFAULT; Self::NUM_UNIQUE_ORIENTATIONS];

    //     let mut orientation_set: [bool; TetrisPiecePlacement::UPPER_BOUND_ORIENTATIONS] =
    //         [false; TetrisPiecePlacement::UPPER_BOUND_ORIENTATIONS];

    //     let mut result_idx = 0;
    //     let mut i = 0;
    //     let all_placements = TetrisPiecePlacement::all_placements();

    //     while i < all_placements.len() {
    //         let placement = all_placements[i];
    //         let orientation = placement.orientation;
    //         let orientation_idx = orientation.index() as usize;

    //         if !orientation_set[orientation_idx] {
    //             orientation_set[orientation_idx] = true;
    //             orientations[result_idx] = orientation;
    //             result_idx += 1;
    //         }
    //         i += 1;
    //     }

    //     orientations
    // }
}

/// A Tetris piece bag is a random selection algorithm to prevent
/// sequences of pieces that guarantee losses (e.g., long runs of
/// S and Z pieces).
///
/// A bag starts with 7 pieces. Then as a tetris game is played, pieces
/// are randomly removed from the bag one by one. Once the bag is
/// 'empty', a new bag is created with all the pieces. This process
/// is continually repeated.
///
/// A single bag of pieces is represented by a single byte where
/// multiple bits are set to represent what's left in the bag.
///
/// ```text
/// +---+---+---+---+---+---+---+---+
/// | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
/// +---+---+---+---+---+---+---+---+
/// | - | J | L | T | Z | S | I | O |
/// +---+---+---+---+---+---+---+---+
/// ```
///
/// The '-' bit is unused.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TetrisPieceBag(u8);

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

/// Default to a full bag.
///
/// We do this because when playing we 'start' with a full
/// bag, hence it's the default case.
impl Default for TetrisPieceBag {
    fn default() -> Self {
        Self(Self::FULL_MASK)
    }
}

impl TetrisPieceBag {
    pub const SIZE: usize = NUM_TETRIS_PIECES;
    pub const FULL_MASK: u8 = 0b0111_1111;

    pub fn new() -> Self {
        Self::default()
    }

    #[inline_conditioned(always)]
    pub fn inc(&mut self) -> Self {
        self.0 += 1;
        *self
    }

    /// Count the number of pieces in the bag.
    #[inline_conditioned(always)]
    pub const fn count(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Check if the bag contains a given piece.
    #[inline_conditioned(always)]
    pub const fn contains(&self, piece: TetrisPiece) -> bool {
        (self.0 & piece.0) > 0
    }

    /// Check if the bag is empty.
    #[inline_conditioned(always)]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Check if the bag is full.
    #[inline_conditioned(always)]
    pub const fn is_full(&self) -> bool {
        self.0 == Self::FULL_MASK
    }

    /// Fill the bag with all pieces.
    #[inline_conditioned(always)]
    pub const fn fill(&mut self) {
        self.0 = Self::FULL_MASK;
    }

    /// Get a new bag filled if it's empty.
    #[inline_conditioned(always)]
    pub const fn new_fill_if_empty(&self) -> Self {
        Self(self.0 | (self.is_empty() as u8).wrapping_neg() & Self::FULL_MASK)
    }

    /// Mutably fill the bag if it's empty.
    #[inline_conditioned(always)]
    pub const fn mut_fill_if_empty(&mut self) {
        *self = self.new_fill_if_empty();
    }

    /// Remove a piece from the bag.
    ///
    /// If the piece is not in the bag, this function will return
    /// the bag unchanged.
    #[inline_conditioned(always)]
    pub const fn remove(&mut self, piece: TetrisPiece) {
        self.0 &= !(piece.0);
    }

    /// Merge two bags together. Duplicate pieces are not added.
    #[inline_conditioned(always)]
    pub const fn merge(&mut self, other: &Self) {
        self.0 |= other.0;
    }

    /// Get all possible next bags if we were to remove any one piece.
    ///
    /// Once a bag is empty, we need to 'restart' the bag with a new
    /// set of pieces.
    #[inline_conditioned(always)]
    pub const fn next_bags(&self) -> NextBagsIter {
        // Check if there are no pieces left in the bag
        // If so, return the full bag
        NextBagsIter::new(self.new_fill_if_empty())
    }

    /// Get a random piece from the bag and remove it.
    /// If the bag is empty, fill it with all pieces first.
    /// Uses the provided RNG to select a random piece from the remaining pieces.
    #[inline_conditioned(always)]
    pub const fn rand_next(&mut self, rng: &mut TetrisGameRng) -> TetrisPiece {
        self.mut_fill_if_empty();

        // select a random "nth" set bit
        // then eliminate all other set bits
        let count = self.count();
        let n = (rng.next_u64() % count as u64) as usize;
        let mut x = self.0;
        repeat_idx_generic!(7, I, {
            let mask = ((I < n) as i8).wrapping_neg() as u8;
            x &= !(mask) | (x - 1);
        });
        x &= x.wrapping_neg();

        // get, then remove the piece
        let piece = TetrisPiece::new(x.trailing_zeros() as u8);
        self.remove(piece);
        piece
    }
}

/// An iterator over all possible next bags if we were to remove any one piece.
///
/// When a piece is chosen to be played, we use this iterator to get the
/// resulting bag without said piece. This iterator makes it convenient
/// to calculate all possible next bags.
#[derive(Debug)]
pub struct NextBagsIter {
    current_bag: TetrisPieceBag,
    next_bags: u64,
}

impl NextBagsIter {
    const fn new(current_bag: TetrisPieceBag) -> Self {
        Self {
            current_bag,
            // This is a bit of magic hackery.
            // broadcast the current bag to a u64 (all possible next future bags)
            // then use a mask to remove the bit that represents each
            // piece from it's respective future bag. Byte 1 represents removing
            // the O piece, byte 2 represents the I piece, etc.
            //
            // This lets us 'pre-calculate' all possible next bags before iteration.
            next_bags: ((current_bag.0 as u64) * 0x0101_0101_0101_0100_u64)
                & !0x40_20_10_08_04_02_01_FF_u64,
        }
    }
}

impl Iterator for NextBagsIter {
    type Item = (TetrisPieceBag, TetrisPiece);

    /// Get the next bag and the respective piece that was removed.
    #[inline_conditioned(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_bag.is_empty() {
            return None;
        }

        // Get the position of the least significant '1' in the byte.
        // This is the same as getting the next piece we want to remove.
        let pos = self.current_bag.0.trailing_zeros();

        // Remove the piece from the bag.
        self.current_bag.0 &= !(1 << pos);

        // Get the next bag by shifting the pre-calculated bags
        // to the right by the appropriate amount.
        //
        // If the bag is empty, fill it up.
        let mut next_bag = (self.next_bags >> ((pos << 3) + 8)) as u8;
        next_bag |= (0u8.wrapping_sub((next_bag == 0) as u8)) & 0b0111_1111;

        Some((TetrisPieceBag(next_bag), TetrisPiece::new(pos as u8)))
    }

    /// The size of the iterator is the remaining number of 'ones' in the byte.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.current_bag.0.count_ones() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NextBagsIter {}

mod tetris_piece_data {
    use super::piece_u32_cols;

    pub const O_PIECE_ROT_0: [u32; 2] = piece_u32_cols! {
        11
        11
        00
        00
    };
    pub const I_PIECE_ROT_0: [u32; 4] = piece_u32_cols! {
        1111
        0000
        0000
        0000
    };
    pub const I_PIECE_ROT_1: [u32; 1] = piece_u32_cols! {
        1
        1
        1
        1
    };
    pub const S_PIECE_ROT_0: [u32; 3] = piece_u32_cols! {
        011
        110
        000
        000
    };
    pub const S_PIECE_ROT_1: [u32; 2] = piece_u32_cols! {
        10
        11
        01
        00
    };
    pub const Z_PIECE_ROT_0: [u32; 3] = piece_u32_cols! {
        110
        011
        000
        000
    };
    pub const Z_PIECE_ROT_1: [u32; 2] = piece_u32_cols! {
        01
        11
        10
        00
    };
    pub const T_PIECE_ROT_0: [u32; 3] = piece_u32_cols! {
        111
        010
        000
        000
    };
    pub const T_PIECE_ROT_1: [u32; 2] = piece_u32_cols! {
        01
        11
        01
        00
    };
    pub const T_PIECE_ROT_2: [u32; 3] = piece_u32_cols! {
        010
        111
        000
        000
    };
    pub const T_PIECE_ROT_3: [u32; 2] = piece_u32_cols! {
        10
        11
        10
        00
    };
    pub const L_PIECE_ROT_0: [u32; 3] = piece_u32_cols! {
        001
        111
        000
        000
    };
    pub const L_PIECE_ROT_1: [u32; 2] = piece_u32_cols! {
        10
        10
        11
        00
    };
    pub const L_PIECE_ROT_2: [u32; 3] = piece_u32_cols! {
        111
        100
        000
        000
    };
    pub const L_PIECE_ROT_3: [u32; 2] = piece_u32_cols! {
        11
        01
        01
        00
    };
    pub const J_PIECE_ROT_0: [u32; 3] = piece_u32_cols! {
        100
        111
        000
        000
    };
    pub const J_PIECE_ROT_1: [u32; 2] = piece_u32_cols! {
        11
        10
        10
        00
    };
    pub const J_PIECE_ROT_2: [u32; 3] = piece_u32_cols! {
        111
        001
        000
        000
    };
    pub const J_PIECE_ROT_3: [u32; 2] = piece_u32_cols! {
        01
        01
        11
        00
    };
}

/// A Tetris board represented as an array of columns, where each column is a u32.
/// Each bit in the u32 represents a cell in that column, making it efficient to
/// check for collisions and manipulate individual columns.
/// The memory layout has one u32 per column, with bits representing rows from bottom to top.
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisBoard([u32; COLS]);

pub type TetrisBoardBinarySlice = [u8; BOARD_SIZE];

impl TetrisBoard {
    pub const SIZE: usize = BOARD_SIZE;
    pub const WIDTH: usize = COLS;
    pub const HEIGHT: usize = ROWS;
    pub const NUM_TETRIS_CELL_STATES: usize = NUM_TETRIS_CELL_STATES;

    pub const EMPTY_BOARD: Self = Self([0_u32; COLS]);
    pub const FULL_BOARD: Self = Self([u32::MAX; COLS]);

    pub const fn as_limbs(&self) -> [u32; COLS] {
        self.0
    }
}

impl TetrisBoard {
    /// Gets the value of a single bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - The row index (0 to ROWS-1)
    ///
    /// # Returns
    ///
    /// * `bool` - The bit value (false or true) at the specified position
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let board = TetrisBoard::new();
    /// assert_eq!(board.get_bit(0, 0), false); // Empty cell at bottom-left
    /// ```
    #[inline_conditioned(always)]
    pub fn get_bit(&self, col: usize, row: usize) -> bool {
        ((self.0[col] >> row) & 1) != 0
    }

    /// Sets a bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - The row index (0 to ROWS-1)
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0); // Set bottom-left cell
    /// assert_eq!(board.get_bit(0, 0), true);
    /// ```
    #[inline_conditioned(always)]
    pub fn set_bit(&mut self, col: usize, row: usize) {
        self.0[col] |= 1 << row;
    }

    /// Sets a specified number of random bits in the board.
    ///
    /// # Arguments
    ///
    /// * `num_bits` - The number of random bits to set
    /// * `rng` - Random number generator implementing the `Rng` trait
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// use rand::thread_rng;
    /// let mut board = TetrisBoard::new();
    /// let mut rng = thread_rng();
    /// board.set_random_bits(10, &mut rng);
    /// ```
    pub fn set_random_bits<R: Rng + ?Sized>(&mut self, num_bits: usize, rng: &mut R) {
        for _ in 0..num_bits {
            let rand = rng.random::<u32>();
            let col = ((rand >> 16) as u16 % COLS as u16) as usize;
            let row = ((rand & 0xFFFF) % ROWS as u32) as usize;
            self.set_bit(col, row);
        }
    }
}

impl Display for TetrisBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, " {:2}| board", "r")?;
        for row in (0..ACTUAL_ROWS).rev() {
            let mut row_str = String::new();
            for col in 0..COLS {
                let bit = self.get_bit(col, row);
                row_str.push(if bit { '#' } else { '.' });
            }
            writeln!(f, "{:2} | {}", row, row_str)?;
        }
        Ok(())
    }
}

impl TetrisBoard {
    /// Converts the board to a binary slice representation.
    /// Each cell is represented as a single bit (0 or 1).
    ///
    /// # Returns
    ///
    /// A fixed-size array of u8 representing the board state
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// let binary = board.to_binary_slice();
    /// assert!(binary.iter().all(|&x| x == 0));
    /// ```
    #[inline_conditioned(always)]
    pub const fn to_binary_slice(&self) -> TetrisBoardBinarySlice {
        let mut result: TetrisBoardBinarySlice = [0u8; Self::SIZE];
        repeat_idx_generic!(Self::SIZE / 8, I, {
            const I1: usize = 8 * I;
            result[I1] = ((self.0[I1 / ROWS] >> (I1 % ROWS)) & 1) as u8;
            const I2: usize = 8 * I + 1;
            result[I2] = ((self.0[I2 / ROWS] >> (I2 % ROWS)) & 1) as u8;
            const I3: usize = 8 * I + 2;
            result[I3] = ((self.0[I3 / ROWS] >> (I3 % ROWS)) & 1) as u8;
            const I4: usize = 8 * I + 3;
            result[I4] = ((self.0[I4 / ROWS] >> (I4 % ROWS)) & 1) as u8;
            const I5: usize = 8 * I + 4;
            result[I5] = ((self.0[I5 / ROWS] >> (I5 % ROWS)) & 1) as u8;
            const I6: usize = 8 * I + 5;
            result[I6] = ((self.0[I6 / ROWS] >> (I6 % ROWS)) & 1) as u8;
            const I7: usize = 8 * I + 6;
            result[I7] = ((self.0[I7 / ROWS] >> (I7 % ROWS)) & 1) as u8;
            const I8: usize = 8 * I + 7;
            result[I8] = ((self.0[I8 / ROWS] >> (I8 % ROWS)) & 1) as u8;
        });
        result
    }

    /// Creates a board from a binary slice representation.
    ///
    /// # Arguments
    ///
    /// * `binary` - Fixed-size array of u8 representing the board state
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// let binary = board.to_binary_slice();
    /// let board2 = TetrisBoard::from_binary_slice(binary);
    /// assert_eq!(board2.get_bit(0, 0), true);
    /// ```
    #[inline_conditioned(always)]
    pub const fn from_binary_slice(binary: TetrisBoardBinarySlice) -> Self {
        let mut result = Self::EMPTY_BOARD;
        repeat_idx_generic!(Self::SIZE / 8, I, {
            const I1: usize = 8 * I;
            result.0[I1 / ROWS] |= (binary[I1] as u32) << (I1 % ROWS);
            const I2: usize = 8 * I + 1;
            result.0[I2 / ROWS] |= (binary[I2] as u32) << (I2 % ROWS);
            const I3: usize = 8 * I + 2;
            result.0[I3 / ROWS] |= (binary[I3] as u32) << (I3 % ROWS);
            const I4: usize = 8 * I + 3;
            result.0[I4 / ROWS] |= (binary[I4] as u32) << (I4 % ROWS);
            const I5: usize = 8 * I + 4;
            result.0[I5 / ROWS] |= (binary[I5] as u32) << (I5 % ROWS);
            const I6: usize = 8 * I + 5;
            result.0[I6 / ROWS] |= (binary[I6] as u32) << (I6 % ROWS);
            const I7: usize = 8 * I + 6;
            result.0[I7 / ROWS] |= (binary[I7] as u32) << (I7 % ROWS);
            const I8: usize = 8 * I + 7;
            result.0[I8 / ROWS] |= (binary[I8] as u32) << (I8 % ROWS);
        });
        result
    }
}

impl TetrisBoard {
    /// Creates a new empty Tetris board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let board = TetrisBoard::new();
    /// assert_eq!(board.count(), 0);
    /// ```
    pub const fn new() -> Self {
        Self::EMPTY_BOARD
    }

    /// Creates a new Tetris board with random filled cells.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator implementing the `Rng` trait
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// use rand::thread_rng;
    /// let mut rng = thread_rng();
    /// let board = TetrisBoard::new_rand(&mut rng);
    /// ```
    pub fn new_rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self([rng.random::<u32>(); COLS])
    }

    /// Returns the total number of filled cells in the board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// assert_eq!(board.count(), 1);
    /// ```
    #[inline_conditioned(always)]
    pub fn count(&self) -> u32 {
        let mut acc = 0;
        repeat_idx_generic!(COLS, I, {
            acc += self.0[I].count_ones();
        });
        acc
    }

    /// Clears all cells in the board, setting them to empty.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// board.clear();
    /// assert_eq!(board.count(), 0);
    /// ```
    #[inline_conditioned(always)]
    pub const fn clear(&mut self) {
        *self = Self::EMPTY_BOARD;
    }

    /// Fills all cells in the board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.fill();
    /// assert!(board.get_bit(0, 0));
    /// ```
    #[inline_conditioned(always)]
    pub const fn fill(&mut self) {
        *self = Self::FULL_BOARD;
    }

    /// Returns the height of the highest block in the board.
    ///
    /// The height is determined by finding the highest set bit across all columns.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 1); // Set bit at row 1
    /// assert_eq!(board.height(), 2); // Height should be 2
    /// ```
    #[inline_conditioned(always)]
    pub const fn height(&self) -> u32 {
        let mut acc = 0;
        repeat_idx_generic!(COLS, I, {
            acc |= self.0[I];
        });
        u32::BITS - acc.leading_zeros()
    }

    /// Returns whether the game is lost by checking if any column exceeds the maximum height.
    ///
    /// A Tetris game is lost when any piece extends beyond the top of the board.
    /// This is detected by checking if any column has blocks above row ROWS-1.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 19); // Set bit at max valid height
    /// assert!(!board.is_lost());
    /// board.set_bit(0, 20); // Set bit above max height
    /// assert!(board.is_lost());
    /// ```
    #[inline_conditioned(always)]
    pub const fn is_lost(&self) -> bool {
        self.height() > ROWS as u32
    }

    /// Clears all completely filled rows and shifts remaining rows down.
    ///
    /// In Tetris, when a row is completely filled with blocks, that row is cleared and all rows above
    /// are shifted downward to fill the gap. This method implements that game mechanic.
    ///
    /// # Returns
    ///
    /// Returns the number of rows that were cleared.
    ///
    /// # Implementation Details
    ///
    /// The implementation works in two steps:
    /// 1. Creates a mask of filled rows by performing a bitwise AND across all columns
    /// 2. Shifts remaining rows downward based on the mask of filled rows
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_atlas::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// // Fill an entire row
    /// for col in 0..10 {
    ///     board.set_bit(col, 0);
    /// }
    /// assert_eq!(board.clear_filled_rows(), 1);
    /// ```
    #[inline_conditioned(always)]
    pub const fn clear_filled_rows(&mut self) -> u32 {
        let mut filled_rows_mask = u32::MAX;
        repeat_idx_generic!(COLS, I, {
            filled_rows_mask &= self.0[I];
        });
        rshift_slice_from_mask_u32::<COLS, 4>(&mut self.0, filled_rows_mask);
        filled_rows_mask.count_ones()
    }

    #[inline_conditioned(always)]
    pub fn apply_piece_placement(&mut self, placement: TetrisPiecePlacement) -> PlacementResult {
        // Early exit: if the board is already in a lost state (all bits set),
        // don't process the placement to preserve the end state marker
        if self.is_lost() {
            return PlacementResult {
                is_lost: IsLost::LOST,
                lines_cleared: 0,
            };
        }

        /// Place a piece on the board by:
        /// 1. Extracting the board column slice where the piece will land
        /// 2. Computing the minimum shift for each column to avoid collision
        /// 3. Taking the minimum (the column that collides first determines the shift)
        /// 4. Shifting the piece down by that amount and OR-ing it into the board
        macro_rules! place_piece_with_consts {
            ($board_cols:expr, $piece_cols:expr) => {{
                const NUM_COLS: usize = $piece_cols.len();
                const TRAILING: [u32; NUM_COLS] = trailing_zeros_all($piece_cols);

                let shift = {
                    let mut min_diff = u32::MAX;
                    repeat_idx_generic!(NUM_COLS, I, {
                        let col_height = u32::BITS - $board_cols[I].leading_zeros();
                        let diff = col_height.abs_diff(TRAILING[I]);

                        // equivalient to min(min_diff, diff)
                        min_diff = min_diff
                            ^ ((min_diff ^ diff) & ((diff < min_diff) as u32).wrapping_neg());
                    });
                    min_diff
                };

                // Place the piece by OR-ing it with the board
                repeat_idx_generic!(NUM_COLS, I, {
                    $board_cols[I] |= $piece_cols[I] >> shift;
                });
            }};
        }

        let piece_width = placement.piece.width(placement.orientation.rotation);
        let start_idx = placement.orientation.column.0 as usize;
        let end_idx = start_idx + (piece_width as usize);
        let board_cols = &mut self.0[start_idx..end_idx];

        match (placement.piece, placement.orientation.rotation) {
            (TetrisPiece::O_PIECE, _) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::O_PIECE_ROT_0)
            }
            (TetrisPiece::I_PIECE, Rotation(0) | Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::I_PIECE_ROT_0)
            }
            (TetrisPiece::I_PIECE, Rotation(1) | Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::I_PIECE_ROT_1)
            }
            (TetrisPiece::S_PIECE, Rotation(0) | Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::S_PIECE_ROT_0)
            }
            (TetrisPiece::S_PIECE, Rotation(1) | Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::S_PIECE_ROT_1)
            }
            (TetrisPiece::Z_PIECE, Rotation(0) | Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::Z_PIECE_ROT_0)
            }
            (TetrisPiece::Z_PIECE, Rotation(1) | Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::Z_PIECE_ROT_1)
            }
            (TetrisPiece::T_PIECE, Rotation(0)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_0)
            }
            (TetrisPiece::T_PIECE, Rotation(1)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_1)
            }
            (TetrisPiece::T_PIECE, Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_2)
            }
            (TetrisPiece::T_PIECE, Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_3)
            }
            (TetrisPiece::L_PIECE, Rotation(0)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_0)
            }
            (TetrisPiece::L_PIECE, Rotation(1)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_1)
            }
            (TetrisPiece::L_PIECE, Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_2)
            }
            (TetrisPiece::L_PIECE, Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_3)
            }
            (TetrisPiece::J_PIECE, Rotation(0)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_0)
            }
            (TetrisPiece::J_PIECE, Rotation(1)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_1)
            }
            (TetrisPiece::J_PIECE, Rotation(2)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_2)
            }
            (TetrisPiece::J_PIECE, Rotation(3)) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_3)
            }
            _ => panic!("Invalid piece or rotation"),
        };

        let lines_cleared = self.clear_filled_rows();
        let is_lost = self.is_lost();
        if is_lost {
            self.fill();
        }

        PlacementResult {
            lines_cleared,
            is_lost: IsLost(is_lost),
        }
    }
}

/// Wrapper type to indicate if the game is lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct IsLost(bool);

impl IsLost {
    pub const LOST: Self = Self(true);
    pub const NOT_LOST: Self = Self(false);
}

impl From<IsLost> for bool {
    fn from(val: IsLost) -> Self {
        val.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TetrisGameRng(u64);

impl TetrisGameRng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline_conditioned(always)]
    pub const fn next_u64(&mut self) -> u64 {
        fmix64(&mut self.0);
        self.0
    }
}

/// When a placement is applied, we return a `PlacementResult`
/// This represents what that placement did.
pub struct PlacementResult {
    pub is_lost: IsLost,
    pub lines_cleared: u32,
}

/// A Tetris game is:
///
/// 1. A board
/// 2. A bag of pieces
/// 3. The current piece
///
/// The interface for playing Tetris is "get placements" and "apply placements".
/// This ensures the caller only plays possible moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TetrisGame {
    board: TetrisBoard,
    bag: TetrisPieceBag,
    piece_buf: TetrisPiece,

    // Include starting seed for reset reproducibility
    seed: u64,
    rng: TetrisGameRng,

    // stats
    pub lines_cleared: u32,
    pub piece_count: u32,
}

impl Display for TetrisGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TetrisGame:")?;
        writeln!(f, "  bag:")?;
        writeln!(f, "    {}", self.bag)?;
        writeln!(f, "  piece_buf:")?;
        writeln!(f, "    {}", self.piece_buf)?;
        writeln!(f, "  lines_cleared:")?;
        writeln!(f, "    {}", self.lines_cleared)?;
        writeln!(f, "  board:")?;
        writeln!(f, "    {}", self.board)?;
        Ok(())
    }
}

impl TetrisGame {
    /// Create a new Tetris game.
    ///
    /// The game is initialized with a new board, a new bag, and a random piece
    /// popped from the bag.
    pub fn new() -> Self {
        let seed = rand::rng().next_u64();
        let mut rng = TetrisGameRng::new(seed);
        let board = TetrisBoard::new();
        let mut bag = TetrisPieceBag::new();
        let piece_buf = bag.rand_next(&mut rng);
        Self {
            board,
            bag,
            piece_buf,
            seed,
            rng,
            lines_cleared: 0,
            piece_count: 0,
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        let mut rng = TetrisGameRng::new(seed);
        let board = TetrisBoard::new();
        let mut bag = TetrisPieceBag::new();
        let piece_buf = bag.rand_next(&mut rng);
        Self {
            board,
            bag,
            piece_buf,
            seed,
            rng,
            lines_cleared: 0,
            piece_count: 0,
        }
    }

    /// Get the current board.
    #[inline_conditioned(always)]
    pub const fn board(&self) -> TetrisBoard {
        self.board
    }

    /// Get the current piece.
    ///
    /// This is the piece that is currently in play.
    pub const fn current_piece(&self) -> TetrisPiece {
        self.piece_buf
    }

    /// Get the current placements that can be applied to the current piece.
    pub fn current_placements(&self) -> &[TetrisPiecePlacement] {
        TetrisPiecePlacement::all_from_piece(self.piece_buf)
    }

    /// Apply a placement to the board.
    ///
    /// Returns `true` if the game is lost; otherwise `false`.
    /// Lines cleared are tracked by the difference in height before and after placement.
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement(&mut self, placement: TetrisPiecePlacement) -> IsLost {
        // debug_assert!(
        //     self.current_placements().contains(&placement),
        //     "Placement {} is not valid for current piece {}",
        //     placement,
        //     self.piece_buf
        // );

        let PlacementResult {
            is_lost,
            lines_cleared,
        } = self.board.apply_piece_placement(placement);
        if is_lost.into() {
            return IsLost::LOST;
        }
        self.lines_cleared += lines_cleared as u32;
        self.piece_buf = self.bag.rand_next(&mut self.rng);
        self.piece_count += 1;
        IsLost::NOT_LOST
    }

    /// Reset the game to a new board, bag, and piece.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.board.clear();
        self.bag.fill();
        self.rng = TetrisGameRng::new(new_seed.unwrap_or(self.seed));
        self.piece_buf = self.bag.rand_next(&mut self.rng);
        self.lines_cleared = 0;
        self.piece_count = 0;
    }
}

const MAX_GAMES: usize = 1024;

/// A set of Tetris games.
#[derive(Clone, Copy, Debug)]
pub struct TetrisGameSet(HeaplessVec<TetrisGame, MAX_GAMES>);

impl TetrisGameSet {
    /// Create a new TetrisGameSet with N default games.
    pub fn new(num_games: usize) -> Self {
        debug_assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new()));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the provided seed.
    /// Each game gets a slightly different seed (seed + index).
    pub fn new_with_seed(seed: u64, num_games: usize) -> Self {
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|i| games.push(TetrisGame::new_with_seed(seed + i as u64)));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the same seed.
    pub fn new_with_same_seed(seed: u64, num_games: usize) -> Self {
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new_with_seed(seed)));
        Self(games)
    }

    /// Permute the gameset using the provided permutation vector.
    ///
    /// The permutation vector must be the same length as the gameset and contain
    /// valid indices (0..len). Each index should appear exactly once.
    pub fn permute(&mut self, permutation: &[usize]) {
        assert_eq!(permutation.len(), self.len(), "Permutation length mismatch");
        let mut new_games = HeaplessVec::new();
        for &idx in permutation {
            new_games.push(*self.0.get(idx).unwrap());
        }
        self.0 = new_games;
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_games(games: &[TetrisGame]) -> Self {
        assert!(games.len() <= MAX_GAMES, "Too many games");
        let mut input_games = HeaplessVec::new();
        input_games.fill_from_slice(games);
        Self(input_games)
    }

    pub fn boards(&self) -> HeaplessVec<TetrisBoard, MAX_GAMES> {
        self.0.map(|game| game.board())
    }

    pub fn current_pieces(&self) -> HeaplessVec<TetrisPiece, MAX_GAMES> {
        self.0.map(|game| game.current_piece())
    }

    pub fn piece_counts(&self) -> HeaplessVec<u32, MAX_GAMES> {
        self.0.map(|game| game.piece_count)
    }

    /// Get the current placements for all games.
    ///
    /// These are the placements that can be applied to the current piece.
    pub fn current_placements(&self) -> Vec<&[TetrisPiecePlacement]> {
        self.0
            .iter()
            .map(|game| game.current_placements())
            .collect()
    }

    /// Apply a placement to the board.
    ///
    /// This will return true if the game is lost, false otherwise.
    /// Lines cleared are tracked by measuring the difference in height before and after the placement.
    ///
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement(&mut self, placements: &[TetrisPiecePlacement]) -> Vec<IsLost> {
        self.0
            .iter_mut()
            .zip(placements)
            .par_bridge()
            .into_par_iter()
            .map(|(game, &placement)| game.apply_placement(placement))
            .collect()
    }

    /// Apply a placement from orientations to the board.
    ///
    /// This will return true if the game is lost, false otherwise.
    /// Lines cleared are tracked by measuring the difference in height before and after the placement.
    ///
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement_from_orientations(
        &mut self,
        orientations: &[TetrisPieceOrientation],
    ) -> Vec<IsLost> {
        self.0
            .iter_mut()
            .zip(orientations)
            .par_bridge()
            .into_par_iter()
            .map(|(game, &orientation)| {
                game.apply_placement(TetrisPiecePlacement {
                    piece: game.current_piece(),
                    orientation,
                })
            })
            .collect()
    }

    /// Reset the game to a new board, bag, and piece.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.0.apply_mut(|game| game.reset(new_seed));
    }
}

impl Index<usize> for TetrisGameSet {
    type Output = TetrisGame;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0.get(index).unwrap()
    }
}

impl IndexMut<usize> for TetrisGameSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proc_macros::piece_u32_cols;
    use rand::seq::IteratorRandom;

    use crate::{repeat_exact_idx, repeat_idx_generic};

    use super::*;

    #[test]
    fn test_board_count() {
        let mut board = TetrisBoard::new();
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
    fn test_set_and_get_bit() {
        let mut board = TetrisBoard::new();
        board.set_bit(0, 0);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
    }

    #[test]
    fn test_drop_piece() {
        // simple case
        let mut board = TetrisBoard::new();
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::new(0),
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(0),
            },
        });
        assert!(board.count() == 4);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(1, 0));
        assert!(board.get_bit(0, 1));
        assert!(board.get_bit(1, 1));

        // fuzz test
        for _ in 0..10_000 {
            let piece = TetrisPiece::new(rand::random::<u8>() % 7);
            let rotation = Rotation(rand::random::<u8>() % 4);
            let col = (rand::random::<u8>()) % ((COLS as u8) - piece.width(rotation));
            TetrisBoard::new().apply_piece_placement(TetrisPiecePlacement {
                piece,
                orientation: TetrisPieceOrientation {
                    rotation,
                    column: Column(col),
                },
            });
        }
    }

    #[test]
    fn test_piece_rotations() {
        fn rotations_reference_slow(i: u8) -> u8 {
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
    fn test_clear() {
        let mut board = TetrisBoard::new();
        board.set_bit(0, 0);
        board.set_bit(9, 19);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(9, 19));

        board.clear();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));

        // small fuzz test
        board.set_random_bits(100, &mut rand::rng());
        assert!(board.count() > 0);
        board.clear();
        assert!(board.count() == 0);
    }

    #[test]
    fn test_height() {
        let mut board = TetrisBoard::new();
        assert_eq!(board.height(), 0);

        // Test each row
        for row in 0..ROWS {
            // Test each column in this row
            for col in 0..COLS {
                board.clear();
                board.set_bit(col, row);
                assert_eq!(
                    board.height(),
                    row as u32 + 1,
                    "Height should be {} when bit set at col {}, row {}",
                    row + 1,
                    col,
                    row
                );
            }
        }
    }

    #[test]
    fn test_clear_filled_rows() {
        let mut board = TetrisBoard::new();
        assert!(board.count() == 0);

        // Test clearing each row
        for row in 0..ROWS {
            // Fill entire row
            for col in 0..COLS {
                board.set_bit(col, row);
            }
            assert!(board.count() == COLS as u32);
            board.clear_filled_rows();
            assert!(
                board.count() == 0,
                "count should be 0, not {}",
                board.count()
            );
        }
    }

    #[test]
    fn test_loss_condition() {
        let mut board = TetrisBoard::new();
        assert!(!Into::<bool>::into(board.is_lost()));

        // Set a bit in the top 4 rows
        board.set_bit(0, ROWS + 1);
        assert!(Into::<bool>::into(board.is_lost()));
        board.clear();
        assert!(!Into::<bool>::into(board.is_lost()));

        // Set a bit in the bottom row
        board.set_bit(0, 0);
        assert!(!Into::<bool>::into(board.is_lost()));
    }

    #[test]
    fn test_bag() {
        // take individually from the bag
        let mut bag = TetrisPieceBag::new();
        bag.remove(TetrisPiece::new(0));
        assert_eq!(bag.count(), 6);
        assert!(!bag.contains(TetrisPiece::new(0)));
        bag.remove(TetrisPiece::new(1));
        assert_eq!(bag.count(), 5);
        assert!(!bag.contains(TetrisPiece::new(1)));
        bag.remove(TetrisPiece::new(2));
        assert_eq!(bag.count(), 4);
        assert!(!bag.contains(TetrisPiece::new(2)));
        bag.remove(TetrisPiece::new(3));
        assert_eq!(bag.count(), 3);
        assert!(!bag.contains(TetrisPiece::new(3)));
        bag.remove(TetrisPiece::new(4));
        assert_eq!(bag.count(), 2);
        assert!(!bag.contains(TetrisPiece::new(4)));
        bag.remove(TetrisPiece::new(5));
        assert_eq!(bag.count(), 1);
        assert!(!bag.contains(TetrisPiece::new(5)));
        bag.remove(TetrisPiece::new(6));
        assert_eq!(bag.count(), 0);
        assert!(!bag.contains(TetrisPiece::new(6)));

        // take 6 pieces and check that the bag has reset
        let mut bag = TetrisPieceBag::new();
        assert_eq!(bag.count(), 7);
        let mut next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 7);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 6));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 6);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 6);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 5));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 5);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 5);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 4));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 4);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 4);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 3));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 3);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 3);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 2));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 2);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 2);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 1));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 1);
        assert!(!bag.contains(next_bags[0].1));
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 1);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 7));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 7);
        // assert_eq!(bag.contains(next_bags[0].1), false); // don't include
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 7);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 6));

        // fuzz test sampling a bunch from the bag
        let num_bags = 10_000;
        let mut all_collected_pieces = Vec::new();
        let mut bag = TetrisPieceBag::new();
        let mut rng = rand::rng();
        for _ in 0..(num_bags * NUM_TETRIS_PIECES) {
            let (next_bag, piece) = bag.next_bags().choose(&mut rng).unwrap();
            bag = next_bag;
            all_collected_pieces.push(piece);
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
    fn test_rand_next() {
        let mut tetris_rng = TetrisGameRng::new(42);

        let mut bag = TetrisPieceBag::new();
        let mut pieces = std::collections::HashSet::new();

        for _ in 0..7 {
            let piece = bag.rand_next(&mut tetris_rng);
            assert!(!bag.contains(piece), "Piece {} should not be in bag", piece);
            pieces.insert(piece);
        }
        assert_eq!(pieces.len(), 7, "Should get 7 unique pieces");

        // Helper function to perform KS test on a sequence of samples
        fn assert_ks_test(samples: &[f32], num_categories: usize, sample_size: usize) {
            // Expected uniform distribution
            let expected = vec![1.0 / num_categories as f32; num_categories];
            let mut observed = vec![0.0; num_categories];

            // Calculate observed frequencies
            for sample in samples {
                observed[*sample as usize] += 1.0 / sample_size as f32;
            }

            // Kolmogorov-Smirnov test
            let mut max_diff = 0.0f32;
            let mut cumsum_expected = 0.0f32;
            let mut cumsum_observed = 0.0f32;

            for i in 0..num_categories {
                cumsum_expected += expected[i];
                cumsum_observed += observed[i];
                let diff = (cumsum_expected - cumsum_observed).abs();
                max_diff = max_diff.max(diff);
            }

            // Critical value for alpha=0.05 is approximately 1.36/sqrt(n)
            let critical_value = 1.36 / (sample_size as f32).sqrt();
            assert!(
                max_diff < critical_value,
                "KS test failed {} > {}",
                max_diff,
                critical_value
            );
        }

        // Generate 1000 samples of 7 pieces each and run Kolmogorov-Smirnov test
        // to verify that each position in the sequence has a uniform distribution
        // across all piece types. This ensures the RNG is unbiased for each draw.
        let mut samples = vec![
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for _ in 0..1000 {
            let mut bag = TetrisPieceBag::new();
            (0..7).for_each(|nth_sample| {
                let ith_piece = bag.rand_next(&mut tetris_rng);
                samples[nth_sample].push(ith_piece.index() as f32);
            });
        }
        samples.iter().for_each(|sample| {
            assert!(
                sample.len() == 1000,
                "Sample length should be 1000, got {}",
                sample.len()
            );
            assert_ks_test(sample, 7, 1000);
        });
    }

    #[test]
    fn test_to_binary_slice() {
        let board = TetrisBoard::new();

        let binary_slice = board.to_binary_slice();
        assert_eq!(binary_slice.len(), BOARD_SIZE);

        let board2 = TetrisBoard::from_binary_slice(binary_slice);
        assert_eq!(board, board2);

        // fuzz test
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut board = TetrisBoard::new();
            board.set_random_bits(1024, &mut rng);
            let binary_slice = board.to_binary_slice();
            let board2 = TetrisBoard::from_binary_slice(binary_slice);
            assert_eq!(board, board2);
        }
    }

    #[test]
    fn test_piece_placement_index() {
        // check that mapping all placement to indices is injective
        let mut index_set = HashSet::new();
        for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
            let index = placement.index();
            assert!(!index_set.contains(&index));
            index_set.insert(index);
        }
        assert_eq!(index_set.len(), TetrisPiecePlacement::NUM_PLACEMENTS);

        // test to index and from index is bijective
        for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
            let index = placement.index();
            let placement2 = TetrisPiecePlacement::from_index(index);
            assert_eq!(placement, placement2);
        }
    }

    #[test]
    fn test_tetris_piece_orientation() {
        let mut set = HashSet::new();
        for orientation in TetrisPieceOrientation::ALL {
            let index = orientation.index();
            let orientation2 = TetrisPieceOrientation::from_index(index);
            assert_eq!(orientation, orientation2);

            assert!(!set.contains(&index));
            set.insert(index);
        }
        assert_eq!(set.len(), TetrisPieceOrientation::NUM_ORIENTATIONS);

        // test that all orientations are present
        for orientation in TetrisPieceOrientation::ALL {
            let index = orientation.index();
            assert!(set.contains(&index));
        }
    }

    #[test]
    fn test_tetris_board_raw2() {
        let mut board = TetrisBoard::new();
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::J_PIECE,
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(0),
            },
        });
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::new(0),
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(3),
            },
        });
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::new(0),
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(5),
            },
        });
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::new(0),
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(7),
            },
        });
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::L_PIECE,
            orientation: TetrisPieceOrientation {
                rotation: Rotation(3),
                column: Column(8),
            },
        });
        board.clear_filled_rows();
    }
}
