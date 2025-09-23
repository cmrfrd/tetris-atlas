use macros::piece_bytes;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngCore};
use ruint::Uint;

use std::fmt::Display;
use std::hash::Hash;
use std::ops::Range;

use crate::utils::HeaplessVec;
use crate::utils::{bits_to_byte, fmix64};

const ROWS: usize = 24;
const COLS: usize = 10;
const BOARD_SIZE: usize = ROWS * COLS;
const NUM_BYTES_FOR_BOARD: usize = BOARD_SIZE / 8;
const ROW_CHUNK: usize = 4;
const BYTES_PER_ROW_CHUNK: usize = ROW_CHUNK * COLS / 8;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Rotation(u8);

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
    #[inline(always)]
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
    pub const fn new(piece: u8) -> Self {
        Self(0b0000_0001 << piece)
    }

    /// Get the index of the piece.
    #[inline(always)]
    pub const fn index(&self) -> u8 {
        self.0.trailing_zeros() as u8
    }

    /// From index (alias for `new`)
    #[inline(always)]
    pub const fn from_index(index: u8) -> Self {
        Self::new(index)
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0b1000_0000
    }

    #[inline(always)]
    pub const fn to_empty(&mut self) {
        *self = unsafe { std::mem::transmute(0b1000_0000u8) };
    }

    #[inline(always)]
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
    #[inline(always)]
    pub const fn width(&self, rotation: Rotation) -> u8 {
        let b = (self.0 == Self::I_PIECE.0) as u8;
        if ((rotation.0 & 1_u8) | (self.0 == Self::O_PIECE.0) as u8) == 1 {
            2_u8.wrapping_sub(b)
        } else {
            3_u8.wrapping_add(b)
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
    #[inline(always)]
    pub const fn height(&self, rotation: Rotation) -> u8 {
        let b = (self.0 == Self::I_PIECE.0) as u8;
        if (((rotation.0 & 1) == 0) as u8 | (self.0 == Self::O_PIECE.0) as u8) == 1 {
            2_u8.wrapping_sub(b)
        } else {
            3_u8.wrapping_add(b)
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
/// First row:  [O][I][S]
/// Second row: [Z][T][L]  
/// Third row:  [J][ ][ ]
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
    #[inline(always)]
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
    #[inline(always)]
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
        Self(0b0111_1111)
    }
}

impl TetrisPieceBag {
    pub const SIZE: usize = NUM_TETRIS_PIECES;

    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn inc(&mut self) -> Self {
        self.0 += 1;
        *self
    }

    /// Count the number of pieces in the bag.
    #[inline(always)]
    pub const fn count(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Check if the bag contains a given piece.
    #[inline(always)]
    pub const fn contains(&self, piece: TetrisPiece) -> bool {
        (self.0 & piece.0) > 0
    }

    /// Check if the bag is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Check if the bag is full.
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.0 == 0b0111_1111
    }

    /// Fill the bag with all pieces.
    #[inline(always)]
    pub fn fill(&mut self) {
        self.0 = 0b0111_1111;
    }

    /// Remove a piece from the bag.
    ///
    /// If the piece is not in the bag, this function will return
    /// the bag unchanged.
    #[inline(always)]
    pub fn remove(&mut self, piece: TetrisPiece) -> TetrisPieceBag {
        self.0 &= !(piece.0);
        *self
    }

    /// Get an iterator over all pieces in the bag.
    #[inline(always)]
    pub fn pieces(&self) -> impl Iterator<Item = TetrisPiece> {
        (0..NUM_TETRIS_PIECES)
            .map(|i| TetrisPiece::new(i as u8))
            .filter(|&i| self.contains(i))
    }

    /// Merge two bags together. Duplicate pieces are not added.
    #[inline(always)]
    pub fn merge(&mut self, other: &Self) {
        self.0 |= other.0;
    }

    /// Get all possible next bags if we were to remove any one piece.
    ///
    /// Once a bag is empty, we need to 'restart' the bag with a new
    /// set of pieces.
    #[inline(always)]
    pub fn next_bags(&self) -> NextBagsIter {
        // Check if there are no pieces left in the bag
        // If so, return the full bag
        if self.is_empty() {
            return NextBagsIter::new(Self::default());
        }
        NextBagsIter::new(*self)
    }

    /// Get a random piece from the bag and remove it.
    /// If the bag is empty, fill it with all pieces first.
    /// Uses the provided RNG to select a random piece from the remaining pieces.
    pub(self) fn rand_next(&mut self, rng: &mut TetrisGameRng) -> TetrisPiece {
        if self.is_empty() {
            self.fill();
        }

        let count = self.count();
        let idx = (rng.next_u64() % count as u64) as usize;
        let piece = self.pieces().nth(idx).unwrap();
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
    fn new(current_bag: TetrisPieceBag) -> Self {
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
    #[inline(always)]
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

/// A Tetris board is a 20x10 grid of cells. Each cell is a bit.
/// However we use 24x10 to make the math easier and for detecting
/// losses easier.
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
/// 20  | 00000000 00 |   25   |   26
/// 21  | 000000 0000 |   26   |   27
/// 22  | 0000 000000 |   27   |   28
/// 23  | 00 00000000 |   28   |   29
/// ```
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisBoardRaw([u8; NUM_BYTES_FOR_BOARD]);

pub type TetrisUint = Uint<240, 4>;

/// This is a utility function for getting the byte and bit index
/// for a given coordinate. This is used when we want to read/write
/// a bit on the board.
#[inline(always)]
const fn get_byte_bit_idx(col: usize, row: usize) -> (usize, usize) {
    let raw_idx = row * COLS + col;
    let byte_idx = raw_idx / 8;
    let bit_idx = 7 - (raw_idx % 8);
    (byte_idx, bit_idx)
}

#[inline(always)]
const fn get_row_col_idx(byte_idx: usize, bit_idx: usize) -> (usize, usize) {
    let raw_idx = byte_idx * 8 + (7 - bit_idx);
    let row = raw_idx / COLS;
    let col = raw_idx % COLS;
    (row, col)
}

impl Default for TetrisBoardRaw {
    fn default() -> Self {
        Self::EMPTY_BOARD
    }
}

impl Display for TetrisBoardRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        write!(f, " {:2} {:2} {:2}| board\n", "r", "a", "b")?;
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

impl From<TetrisBoardRaw> for TetrisUint {
    fn from(board: TetrisBoardRaw) -> Self {
        TetrisUint::from_be_bytes(board.0)
    }
}

impl TetrisBoardRaw {
    pub const SIZE: usize = BOARD_SIZE;
    pub const WIDTH: usize = COLS;
    pub const HEIGHT: usize = ROWS;
    pub const NUM_TETRIS_CELL_STATES: usize = NUM_TETRIS_CELL_STATES;

    pub const EMPTY_BOARD: Self = Self([0_u8; NUM_BYTES_FOR_BOARD]);
    pub const FULL_BOARD: Self = Self([0xFF_u8; NUM_BYTES_FOR_BOARD]);
}

/// Implement all the shifting based methods.
///
/// These are used for core game logic when a line is cleared
/// and all rows above are shifted down.
impl TetrisBoardRaw {
    /// Shift all rows up by 1.
    /// Used after dropping a piece when a collision is detected:
    /// shift the piece up before merging into the board.
    #[inline(always)]
    pub(crate) fn shift_up(&mut self) {
        for i in 0..(NUM_BYTES_FOR_BOARD - 2) {
            self.0[i] = (self.0[i + 1] << 2) | ((self.0[i + 2] & 0b1100_0000) >> 6);
        }
        self.0[NUM_BYTES_FOR_BOARD - 2] = self.0[NUM_BYTES_FOR_BOARD - 1] << 2;
        self.0[NUM_BYTES_FOR_BOARD - 1] = 0;
    }

    /// Shift all rows down by 1.
    /// When placing a piece, shift it down until it hits the bottom
    /// of the board or collides with another cell.
    pub(crate) fn shift_down(&mut self) {
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

    /// Shift all rows down by 1 relative to a given row.
    /// This is used when we want to 'clear' a row.
    /// All the row bits above the given row are shifted down
    /// by 1 row.
    pub(crate) fn shift_down_from(&mut self, row: usize) {
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

/// Implement all the bit setting based methods.
///
/// We use these for fuzzing, benchmarking, and for data augmentation.
impl TetrisBoardRaw {
    #[inline(always)]
    pub const fn get_bit(&self, col: usize, row: usize) -> bool {
        let (byte_idx, bit_idx) = get_byte_bit_idx(col, row);
        (self.0[byte_idx] & (1 << bit_idx)) != 0
    }

    pub(crate) fn flip_bit(&mut self, col: usize, row: usize) {
        let (byte_idx, bit_idx) = get_byte_bit_idx(col, row);
        self.0[byte_idx] ^= 1 << bit_idx;
    }

    pub(crate) fn set_bit(&mut self, col: usize, row: usize) {
        let (byte_idx, bit_idx) = get_byte_bit_idx(col, row);
        self.0[byte_idx] |= 1 << bit_idx;
    }

    pub(crate) fn unset_bit(&mut self, col: usize, row: usize) {
        let (byte_idx, bit_idx) = get_byte_bit_idx(col, row);
        self.0[byte_idx] &= !(1 << bit_idx);
    }

    pub(crate) fn flip_random_bits<R: Rng + ?Sized>(
        &mut self,
        num_bits: usize,
        rng: &mut R,
    ) -> &mut Self {
        for _ in 0..num_bits {
            let col = rng.random_range(0..COLS);
            let row = rng.random_range(0..ROWS);
            self.flip_bit(col, row);
        }
        self
    }
}

impl TetrisBoardRaw {
    /// Clear the entire board by setting all the bits to 0.
    pub(crate) fn clear_all(&mut self) -> &mut Self {
        *self = Self::EMPTY_BOARD;
        self
    }

    /// Fill the entire board with 1s.
    pub(crate) fn fill_all(&mut self) -> &mut Self {
        *self = Self::FULL_BOARD;
        self
    }

    /// Check if the board is clear.
    #[inline(always)]
    pub fn is_clear(&self) -> bool {
        self == &Self::EMPTY_BOARD
    }

    /// Check if the board is full.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.0 == Self::FULL_BOARD.0
    }

    /// Merge another board into this one (bitwise OR).
    /// Used when placing a piece onto the play board after collision tests.
    pub(crate) fn merge(&mut self, other: &Self) -> &mut Self {
        unsafe {
            let src = other.0.as_ptr();
            let dst = self.0.as_mut_ptr();

            // Process 1 u128 chunk at offset 0 (bytes 0-15)
            {
                let chunk = (src as *const u128).read_unaligned();
                let existing = (dst as *const u128).read_unaligned();
                (dst as *mut u128).write_unaligned(chunk | existing);
            }

            // Process 1 u64 chunk at offset 16 (bytes 16-23)
            {
                let chunk = (src.add(16) as *const u64).read_unaligned();
                let existing = (dst.add(16) as *const u64).read_unaligned();
                (dst.add(16) as *mut u64).write_unaligned(chunk | existing);
            }

            // Process 1 u32 chunk at offset 24 (bytes 24-27)
            {
                let chunk = (src.add(24) as *const u32).read_unaligned();
                let existing = (dst.add(24) as *const u32).read_unaligned();
                (dst.add(24) as *mut u32).write_unaligned(chunk | existing);
            }

            // Process 1 u16 chunk at offset 28 (bytes 28-29)
            {
                let chunk = (src.add(28) as *const u16).read_unaligned();
                let existing = (dst.add(28) as *const u16).read_unaligned();
                (dst.add(28) as *mut u16).write_unaligned(chunk | existing);
            }
        }
        self
    }

    /// Whenever a row is full, clear it and settle the rows above it.
    /// Returns the number of lines cleared.
    pub(crate) fn clear_rows(&mut self) -> u32 {
        let mut lines_cleared = 0;

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
                lines_cleared += 1;
            }
            if (chunk & (0x00_3F_F0_00_00_00_00_00_u64)) == (0x00_3F_F0_00_00_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 1);
                lines_cleared += 1;
            }
            if (chunk & (0x00_00_0F_FC_00_00_00_00_u64)) == (0x00_00_0F_FC_00_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 2);
                lines_cleared += 1;
            }
            if (chunk & (0x00_00_00_03_FF_00_00_00_u64)) == (0x00_00_00_03_FF_00_00_00_u64) {
                self.shift_down_from(chunk_base_idx + 3);
                lines_cleared += 1;
            }
        }

        lines_cleared
    }

    /// Count the number of cells / bits set in the board.
    pub const fn count(&self) -> usize {
        unsafe {
            // Read bytes 0-15 as a u128
            let upper_chunk = self.0.as_ptr();
            let chunk128_a = std::ptr::read_unaligned(upper_chunk as *const u128);
            let count_a = chunk128_a.count_ones() as usize;

            // Read bytes 14-29 (16 bytes) with 2-byte overlap
            let lower_chunk = self.0.as_ptr().add(NUM_BYTES_FOR_BOARD - 16);
            let chunk128_b = std::ptr::read_unaligned(lower_chunk as *const u128);

            // Mask out the overlapping 2 bytes (bytes 14-15) from the second chunk
            // Keep only bytes 16-29 in little-endian format
            const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_0000_u128;
            let count_b = (chunk128_b & MASK).count_ones() as usize;

            count_a + count_b
        }
    }

    /// Compute the height of the board.
    ///
    /// The height is the row number (relative to the bottom of the board),
    /// of the first non-zero cell / bit.
    #[inline(always)]
    pub const fn height(&self) -> u8 {
        unsafe {
            // Upper 16 bytes represent the first 12.8 rows
            let upper_chunk = self.0.as_ptr();
            let chunk128_a = u128::from_be(std::ptr::read_unaligned(upper_chunk as *const u128));
            let first_nonzero_idx = chunk128_a.leading_zeros() as usize;
            if first_nonzero_idx < 128 {
                let byte_idx = first_nonzero_idx / 8;
                let bit_idx = 7 - (first_nonzero_idx % 8);
                let (row, _) = get_row_col_idx(byte_idx, bit_idx);
                return (ROWS - row) as u8;
            }

            // Lower 16 bytes represent the last 12.8 rows
            const LOWER_CHUNK_OFFSET: usize = NUM_BYTES_FOR_BOARD - 16;
            let lower_chunk = self.0.as_ptr().add(LOWER_CHUNK_OFFSET);
            let chunk128_b = u128::from_be(std::ptr::read_unaligned(lower_chunk as *const u128));
            let first_nonzero_idx = chunk128_b.leading_zeros() as usize;
            if first_nonzero_idx < 128 {
                let byte_idx = (first_nonzero_idx) / 8 + LOWER_CHUNK_OFFSET;
                let bit_idx = 7 - ((first_nonzero_idx) % 8);
                let (row, _) = get_row_col_idx(byte_idx, bit_idx);
                return (ROWS - row) as u8;
            }
            0
        }
    }

    /// Check if two boards collide.
    ///
    /// This is used to check if a piece can be placed on a board.
    /// We do this by checking if the bitwise AND of the two boards
    /// is non-zero.
    #[inline(always)]
    pub const fn collides(&self, other: &Self) -> bool {
        // For performance reasons, we read each board as 2 * u128 chunks
        // then check if any bits are aligned.
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

    /// Loss is defined as any cell being filled in the top 4 rows.
    ///
    /// To compute this quickly, we read the first 8 bytes of the board
    /// as a u64, mask out the last 3 bytes, and check if the result is non-zero.
    /// Any bits in the first 5 bytes make the value positive (a loss).
    #[inline]
    pub const fn loss(&self) -> IsLost {
        unsafe {
            IsLost(
                (std::ptr::read_unaligned(self.0.as_ptr() as *const u64) & 0x000000FFFFFFFFFF) != 0,
            )
        }
    }

    /// Convert the board’s byte representation to a binary slice of 0/1 values.
    ///
    /// This is used to "vectorize" the board for training purposes.
    #[inline(always)]
    pub const fn to_binary_slice(&self) -> [u8; BOARD_SIZE] {
        let mut result = [0u8; BOARD_SIZE];
        let mut i = 0;
        while i < NUM_BYTES_FOR_BOARD {
            let byte = self.0[i];
            let base_idx = i * 8;
            result[base_idx + 0] = (byte >> 7) & 1;
            result[base_idx + 1] = (byte >> 6) & 1;
            result[base_idx + 2] = (byte >> 5) & 1;
            result[base_idx + 3] = (byte >> 4) & 1;
            result[base_idx + 4] = (byte >> 3) & 1;
            result[base_idx + 5] = (byte >> 2) & 1;
            result[base_idx + 6] = (byte >> 1) & 1;
            result[base_idx + 7] = byte & 1;
            i += 1;
        }
        result
    }

    /// Convert a binary slice to a Tetris board. This is the inverse
    /// of `to_binary_slice`.
    pub const fn from_binary_slice(binary_slice: [u8; BOARD_SIZE]) -> Self {
        let mut board = Self::EMPTY_BOARD;
        let mut byte_idx = 0;
        while byte_idx < NUM_BYTES_FOR_BOARD {
            let base_idx = byte_idx * 8;
            let bits = [
                binary_slice[base_idx + 0],
                binary_slice[base_idx + 1],
                binary_slice[base_idx + 2],
                binary_slice[base_idx + 3],
                binary_slice[base_idx + 4],
                binary_slice[base_idx + 5],
                binary_slice[base_idx + 6],
                binary_slice[base_idx + 7],
            ];
            board.0[byte_idx] = bits_to_byte(&bits);
            byte_idx += 1;
        }
        board
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

impl Into<bool> for IsLost {
    fn into(self) -> bool {
        self.0
    }
}

/// When a placement is applied, we return a `PlacementResult`
/// This represents what that placement did.
pub struct PlacementResult {
    pub is_lost: IsLost,
    pub lines_cleared: u32,
}

/// A Tetris board.
///
/// This is a wrapper around a `BoardRaw` that contains the game board
/// and the piece board.
///
/// The game board is the main board that we play on.
/// The piece board is a 'buffer' we use to test piece placements.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TetrisBoard {
    play_board: TetrisBoardRaw,
    piece_board: TetrisBoardRaw,
}

impl TetrisBoard {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for TetrisBoard {
    fn default() -> Self {
        Self {
            play_board: TetrisBoardRaw::default(),
            piece_board: TetrisBoardRaw::default(),
        }
    }
}

impl Display for TetrisBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        write!(f, " {:2} {:2} {:2}| board\n", "r", "a", "b")?;
        for row in 0..ROWS {
            let first_byte = row + (row / 4);
            let second_byte = first_byte + 1;
            write!(f, "{:2} {:2} {:2} | ", row, first_byte, second_byte)?;
            for col in 0..COLS {
                write!(f, "{}", self.play_board.get_bit(col, row) as u8)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl From<TetrisBoardRaw> for TetrisBoard {
    fn from(raw: TetrisBoardRaw) -> Self {
        Self {
            play_board: raw,
            piece_board: TetrisBoardRaw::default(),
        }
    }
}

impl From<TetrisBoard> for TetrisUint {
    fn from(board: TetrisBoard) -> Self {
        board.play_board.into()
    }
}

/// Small helper functions for `TetrisBoard`.
impl TetrisBoard {
    #[inline(always)]
    pub fn height(&self) -> u8 {
        self.play_board.height()
    }

    #[inline(always)]
    pub fn loss(&self) -> IsLost {
        self.play_board.loss()
    }
}

/// Main functions for `TetrisBoard`.
impl TetrisBoard {
    /// Convert the Tetris board to a binary slice by bits.
    /// This is used to "vectorize" the board.
    pub const fn to_binary_slice(&self) -> [u8; BOARD_SIZE] {
        self.play_board.to_binary_slice()
    }

    /// Convert a binary slice to a Tetris board.
    /// This is used to "de-vectorize" the board.
    pub fn from_binary_slice(binary_slice: [u8; BOARD_SIZE]) -> Self {
        Self::from(TetrisBoardRaw::from_binary_slice(binary_slice))
    }

    /// Get the current play board.
    #[inline(always)]
    pub const fn board(&self) -> TetrisBoardRaw {
        self.play_board
    }

    /// Reset the board to the initial empty state.
    pub fn reset(&mut self) {
        self.play_board.clear_all();
        self.piece_board.clear_all();
    }

    /// Apply a piece placement to the board. This is the main play function.
    ///
    /// To play a piece, we first add it to the top of the piece board.
    /// Then we shift the piece down row by row until we find a collision or
    /// the piece is at the bottom of the board. If we find a collision, we
    /// shift the piece up, merge it with the main board, then exit.
    ///
    /// If a board is lost and we try to continue playing, we fill the board
    /// with 1s and return a lost state.
    pub fn apply_piece_placement(&mut self, placement: TetrisPiecePlacement) -> PlacementResult {
        if self.loss().into() {
            self.play_board.fill_all();
            return PlacementResult {
                is_lost: IsLost::LOST,
                lines_cleared: 0,
            };
        }

        self.add_piece_top(placement);
        for _ in 0..((ROWS as u8) - placement.piece.height(placement.orientation.rotation)) {
            self.piece_board.shift_down();
            if self.play_board.collides(&self.piece_board) {
                self.piece_board.shift_up();
                self.play_board.merge(&self.piece_board);
                self.piece_board.clear_all();
                let lines_cleared = self.play_board.clear_rows();
                return PlacementResult {
                    is_lost: self.loss(),
                    lines_cleared,
                };
            }
        }
        self.play_board.merge(&self.piece_board);
        self.piece_board.clear_all();
        let lines_cleared = self.play_board.clear_rows();
        PlacementResult {
            is_lost: self.loss(),
            lines_cleared,
        }
    }

    /// Add a piece to the top of the piece board.
    ///
    /// Every piece, rotation, and column combination is expanded
    /// explicitly.
    fn add_piece_top(&mut self, placement: TetrisPiecePlacement) {
        match (
            placement.piece.0,
            placement.orientation.rotation.0,
            placement.orientation.column.0,
        ) {
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
            _ => unreachable!("Invalid: {}", placement),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct TetrisGameRng(u64);

impl TetrisGameRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = fmix64(self.0);
        self.0
    }
}

/// A Tetris game is:
///
/// 1. A board
/// 2. A bag of pieces
/// 3. The current piece
///
/// The interface for playing Tetris is “get placements” and “apply placements”.
/// This ensures the caller only plays possible moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
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

    /// Get the current playing board.
    #[inline(always)]
    pub const fn board(&self) -> TetrisBoardRaw {
        self.board.board()
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
        debug_assert!(
            self.current_placements().contains(&placement),
            "Placement {} is not valid for current piece {}",
            placement,
            self.piece_buf
        );

        let PlacementResult {
            is_lost,
            lines_cleared,
        } = self.board.apply_piece_placement(placement);
        if is_lost.into() {
            return IsLost::LOST;
        }
        self.lines_cleared += lines_cleared;
        self.piece_buf = self.bag.rand_next(&mut self.rng);
        self.piece_count += 1;
        IsLost::NOT_LOST
    }

    /// Reset the game to a new board, bag, and piece.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.board.reset();
        self.bag.fill();
        self.rng = TetrisGameRng::new(new_seed.unwrap_or(self.seed));
        self.piece_buf = self.bag.rand_next(&mut self.rng);
        self.lines_cleared = 0;
        self.piece_count = 0;
    }

    /// Export the current board as a `u256`.
    pub fn export_board(&self) -> TetrisUint {
        self.board.into()
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

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_games(games: &[TetrisGame]) -> Self {
        assert!(games.len() <= MAX_GAMES, "Too many games");
        let mut input_games = HeaplessVec::new();
        input_games.fill_from_slice(games);
        Self(input_games)
    }

    #[inline(always)]
    pub fn boards(&self) -> HeaplessVec<TetrisBoardRaw, MAX_GAMES> {
        self.0.map(|game| game.board())
    }

    pub fn current_pieces(&self) -> HeaplessVec<TetrisPiece, MAX_GAMES> {
        self.0.map(|game| game.current_piece())
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
            .map(|(game, &placement)| game.apply_placement(placement))
            .collect()
    }

    /// Reset the game to a new board, bag, and piece.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.0.apply_mut(|game| game.reset(new_seed));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::seq::IteratorRandom;

    use super::*;

    #[test]
    fn test_board_count() {
        let mut board = TetrisBoardRaw::default();
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
        let mut board = TetrisBoardRaw::default();
        board.set_bit(0, 0);
        assert!(board.get_bit(0, 0));
        assert!(!board.get_bit(1, 0));
        assert!(!board.get_bit(0, 1));
    }

    #[test]
    fn test_shift_down() {
        let mut board = TetrisBoardRaw::default();
        board.set_bit(0, 0); // Top-left bit
        assert!(board.get_bit(0, 0));

        board.shift_down();
        assert!(!board.get_bit(0, 0));
        assert!(board.get_bit(0, 1));

        let mut board = TetrisBoardRaw::default();
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
        fn is_row_filled(board: &TetrisBoardRaw, row: usize) -> bool {
            for col in 0..COLS {
                if !board.get_bit(col, row) {
                    return false;
                }
            }
            true
        }
        fn fill_row(board: &mut TetrisBoardRaw, row: usize) {
            for col in 0..COLS {
                board.set_bit(col, row);
            }
        }

        // start with empty board
        // and fill the first row
        let mut board = TetrisBoardRaw::default();
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
        let mut board = TetrisBoardRaw::default();
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
        // shift down by 4 into the "alive" range
        for _ in 0..4 {
            board.shift_down_from(ROWS - 1);
        }

        // start with the shifted down board with the diagonal of ones
        // and shift from rows above the diagonal. No changes should be made
        let base_board = board.clone();
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
        let mut board = TetrisBoardRaw::default();
        for i in 0..ROWS {
            for j in 0..COLS {
                board.set_bit(j, i);
            }
        }

        let start_row = ROWS - 10;
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
        let mut board = TetrisBoardRaw::default();
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
        let mut board = TetrisBoardRaw::default();
        board.set_bit(0, 1); // Second row, first column
        assert!(board.get_bit(0, 1));

        board.shift_up();
        assert!(!board.get_bit(0, 1));
        assert!(board.get_bit(0, 0));
    }

    #[test]
    fn test_merge() {
        let mut boarda = TetrisBoardRaw::default();
        let mut boardb = TetrisBoardRaw::default();
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
        let mut board = TetrisBoardRaw::default();
        board.set_bit(0, 0);
        board.set_bit(9, 19);
        assert!(board.get_bit(0, 0));
        assert!(board.get_bit(9, 19));

        board.clear_all();
        assert!(!board.get_bit(0, 0));
        assert!(!board.get_bit(9, 19));

        // small fuzz test
        board.flip_random_bits(100, &mut rand::rng());
        assert!(board.count() > 0);
        board.clear_all();
        assert!(board.count() == 0);
    }

    #[test]
    fn test_clear_filled_rows() {
        let mut board = TetrisBoardRaw::default();
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
        let mut board = TetrisBoardRaw::default();
        assert!(!Into::<bool>::into(board.loss()));

        // Set a bit in the top 4 rows
        board.set_bit(0, 1);
        assert!(Into::<bool>::into(board.loss()));
        board.clear_all();
        assert!(!Into::<bool>::into(board.loss()));

        // Set a bit in the bottom row
        board.set_bit(0, ROWS - 1);
        assert!(!Into::<bool>::into(board.loss()));
    }

    #[test]
    fn test_collision() {
        let mut board1 = TetrisBoardRaw::default();
        let mut board2 = TetrisBoardRaw::default();

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
        // and ensure it doesn't collide with an empty board
        board1.fill_all();
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
        let mut board = TetrisBoardRaw::default();
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
        // take individually from the bag
        let mut bag = TetrisPieceBag::new();
        bag.remove(TetrisPiece::new(0));
        assert_eq!(bag.count(), 6);
        assert_eq!(bag.contains(TetrisPiece::new(0)), false);
        bag.remove(TetrisPiece::new(1));
        assert_eq!(bag.count(), 5);
        assert_eq!(bag.contains(TetrisPiece::new(1)), false);
        bag.remove(TetrisPiece::new(2));
        assert_eq!(bag.count(), 4);
        assert_eq!(bag.contains(TetrisPiece::new(2)), false);
        bag.remove(TetrisPiece::new(3));
        assert_eq!(bag.count(), 3);
        assert_eq!(bag.contains(TetrisPiece::new(3)), false);
        bag.remove(TetrisPiece::new(4));
        assert_eq!(bag.count(), 2);
        assert_eq!(bag.contains(TetrisPiece::new(4)), false);
        bag.remove(TetrisPiece::new(5));
        assert_eq!(bag.count(), 1);
        assert_eq!(bag.contains(TetrisPiece::new(5)), false);
        bag.remove(TetrisPiece::new(6));
        assert_eq!(bag.count(), 0);
        assert_eq!(bag.contains(TetrisPiece::new(6)), false);

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
        assert_eq!(bag.contains(next_bags[0].1), false);
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 6);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 5));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 5);
        assert_eq!(bag.contains(next_bags[0].1), false);
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 5);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 4));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 4);
        assert_eq!(bag.contains(next_bags[0].1), false);
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 4);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 3));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 3);
        assert_eq!(bag.contains(next_bags[0].1), false);
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 3);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 2));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 2);
        assert_eq!(bag.contains(next_bags[0].1), false);
        next_bags = bag.next_bags().collect::<Vec<_>>();
        assert_eq!(next_bags.len(), 2);
        next_bags
            .iter()
            .for_each(|(bag, _)| assert_eq!(bag.count(), 1));

        bag = next_bags[0].0;
        assert_eq!(bag.count(), 1);
        assert_eq!(bag.contains(next_bags[0].1), false);
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
    fn test_add_piece_top() {
        let mut board = TetrisBoard::default();

        let o_piece = TetrisPiece::new(0);
        let rotation = Rotation(0);
        let col = 0;
        board.add_piece_top(TetrisPiecePlacement {
            piece: o_piece,
            orientation: TetrisPieceOrientation {
                rotation,
                column: Column(col),
            },
        });
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
        board.apply_piece_placement(TetrisPiecePlacement {
            piece: TetrisPiece::new(0),
            orientation: TetrisPieceOrientation {
                rotation: Rotation(0),
                column: Column(0),
            },
        });
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
            TetrisBoard::default().apply_piece_placement(TetrisPiecePlacement {
                piece,
                orientation: TetrisPieceOrientation {
                    rotation,
                    column: Column(col),
                },
            });
        }
    }

    #[test]
    fn test_line_height() {
        let board = TetrisBoard::default();
        assert_eq!(board.height(), 0);

        let mut board = TetrisBoard::default();
        for i in 0..COLS {
            for j in 0..ROWS {
                board.play_board.set_bit(i, j);
            }
        }
        assert_eq!(board.height(), ROWS as u8);

        // Test that line_height decreases as we clear rows from the bottom
        let mut board = TetrisBoard::default();
        // Fill the entire board
        for i in 0..COLS {
            for j in 0..ROWS {
                board.play_board.set_bit(i, j);
            }
        }
        assert_eq!(board.height(), ROWS as u8);

        // Shift down and verify line height decreases by 1 each time
        for expected_height in (0..ROWS as u8).rev() {
            board.play_board.shift_down();
            assert_eq!(board.height(), expected_height);
        }
    }

    #[test]
    fn test_coord_iff_byte_bit() {
        for i in 0..COLS {
            for j in 0..ROWS {
                let (byte_idx, bit_idx) = get_byte_bit_idx(i, j);
                let (row, col) = get_row_col_idx(byte_idx, bit_idx);
                assert_eq!(row, j, "row: {}, col: {}", row, col);
                assert_eq!(col, i, "row: {}, col: {}", row, col);
            }
        }
    }

    #[test]
    fn test_to_binary_slice() {
        let board_raw = TetrisBoardRaw::default();

        let binary_slice = board_raw.to_binary_slice();
        assert_eq!(binary_slice.len(), BOARD_SIZE);

        let board_raw2 = TetrisBoardRaw::from_binary_slice(binary_slice);
        assert_eq!(board_raw, board_raw2);

        // fuzz test
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut board_raw = TetrisBoardRaw::default();
            board_raw.flip_random_bits(1_000, &mut rng);
            let binary_slice = board_raw.to_binary_slice();
            let board_raw2 = TetrisBoardRaw::from_binary_slice(binary_slice);
            assert_eq!(board_raw, board_raw2);
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
}
