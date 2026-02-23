use proc_macros::{inline_conditioned, piece_u32_cols};
use rand::distr::{Distribution, StandardUniform};
use rand::Rng;

use std::fmt::Display;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use crate::repeat_idx_unroll;
use crate::utils::{rshift_slice_from_mask_u32, BitMask, HeaplessVec};

/// Core constants for Tetris game dimensions and pieces.
pub mod constants {
    /// Visible Tetris board dimensions
    pub const ROWS: usize = 20;
    pub const COLS: usize = 10;
    pub const BOARD_SIZE: usize = ROWS * COLS;

    /// Tetris piece rotations and pieces
    pub const NUM_ROTATIONS: u8 = 4;
    pub const NUM_TETRIS_PIECES: usize = 7;

    /// Column representation for the board: each `u32` bit is a row (0 = bottom, 19 = top, 20-31 = overflow).
    pub type BackingColType = u32;
    pub const ACTUAL_ROWS: usize = BackingColType::BITS as usize;
    const _: () = assert!(
        super::constants::ACTUAL_ROWS > super::constants::ROWS,
        "ACTUAL_ROWS must be greater than ROWS"
    );

    /// Number of possible cell states (0 = empty, 1 = filled).
    pub const NUM_TETRIS_CELL_STATES: usize = 2;
}

/// Piece shapes encoded as column bitmasks for fast collision detection.
///
/// Each piece rotation is stored as an array of `u32` column bitmasks, where:
/// - Array length = piece width in that rotation
/// - Each BackingColType should encode the filled cells in that column (bit 0 = bottom)
///
/// The `piece_u32_cols!` macro generates these from visual representations,
/// making the piece shapes easy to read and verify.
///
/// # Naming Convention
///
/// `{PIECE}_PIECE_ROT_{N}` where:
/// - `PIECE` = O, I, S, Z, T, L, or J
/// - `N` = rotation index (0 = 0°, 1 = 90° CW, 2 = 180°, 3 = 270° CW)
///
/// Note: O piece has only 1 rotation; I, S, Z have 2 distinct rotations.
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

/// A rotation state for a Tetris piece.
///
/// Represents the clockwise rotation of a piece from its spawn orientation.
/// All standard Tetris pieces have at most 4 distinct rotations. 0°, 90°, 180°, 270°.
//
/// Note: Some pieces have fewer distinct rotations (O=1, I/S/Z=2, T/L/J=4).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct Rotation(u8);

impl Display for Rotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Rotation {
    /// Maximum rotation value (exclusive). Rotations are in range `0..MAX`.
    pub const MAX: u8 = constants::NUM_ROTATIONS;
}

/// A column index on the Tetris board (minimum is the leftmost column, maximum is the rightmost column).
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Default)]
#[repr(transparent)]
pub struct Column(u8);

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column({})", self.0)
    }
}

impl Column {
    pub const MAX: u8 = constants::COLS as u8;
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
        TetrisPiece::new(rng.random_range(0..constants::NUM_TETRIS_PIECES) as u8)
    }
}

impl const From<TetrisPiece> for u8 {
    fn from(piece: TetrisPiece) -> Self {
        piece.0
    }
}

impl const From<u8> for TetrisPiece {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl TetrisPiece {
    pub const NUM_PIECES: usize = constants::NUM_TETRIS_PIECES;

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
    #[must_use]
    pub const fn all() -> [Self; constants::NUM_TETRIS_PIECES] {
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
    #[must_use]
    pub const fn new(piece_idx: u8) -> Self {
        Self(0b0000_0001 << piece_idx)
    }

    /// Get the index of the piece.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn index(&self) -> u8 {
        self.0.trailing_zeros() as u8
    }

    /// From index (alias for `new`)
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn from_index(index: u8) -> Self {
        Self::new(index)
    }

    /// Returns the number of distinct rotations for this piece.
    ///
    /// | Piece | Rotations |
    /// |-------|-----------|
    /// | O     | 1         |
    /// | I, S, Z | 2       |
    /// | T, L, J | 4       |
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn num_rotations(&self) -> u8 {
        // Branchless computation: 2 * (1 + is_TLJ) - is_O
        // This avoids match/lookup overhead while remaining readable.
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

    /// Returns the width (number of columns) of this piece at the given rotation.
    ///
    /// | Piece | Rotation 0/2 | Rotation 1/3 |
    /// |-------|--------------|--------------|
    /// | O     | 2            | 2            |
    /// | I     | 4            | 1            |
    /// | S,Z,T,L,J | 3        | 2            |
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn width(&self, rotation: Rotation) -> u8 {
        // Branchless: groups pieces by common width patterns.
        // For odd rotation OR O-piece: returns 2 (or 1 for I)
        // For even rotation (non-O): returns 3 (or 4 for I)
        let is_i = (self.0 == Self::I_PIECE.0) as u8;
        if ((rotation.0 & 1_u8) | (self.0 == Self::O_PIECE.0) as u8) == 1 {
            2_u8.wrapping_sub(is_i)
        } else {
            3_u8.wrapping_add(is_i)
        }
    }

    /// Returns the height (number of rows) of this piece at the given rotation.
    ///
    /// This is the transpose of [`width`](Self::width):
    ///
    /// | Piece | Rotation 0/2 | Rotation 1/3 |
    /// |-------|--------------|--------------|
    /// | O     | 2            | 2            |
    /// | I     | 1            | 4            |
    /// | S,Z,T,L,J | 2        | 3            |
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn height(&self, rotation: Rotation) -> u8 {
        // Branchless: inverse of width logic.
        // For even rotation OR O-piece: returns 2 (or 1 for I)
        // For odd rotation (non-O): returns 3 (or 4 for I)
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
            val if val == Self::NULL_PIECE.0 => write!(f, "Null"),
            _ => write!(f, "Unknown(0x{:02x})", self.0),
        }
    }
}

/// A Tetris orientation is a (rotation, column) pair that defines how a piece is positioned on the board.
/// Each piece only allows certain valid orientations.
///
/// Bellow we show all possible (rotation, column) pairs and which pieces can have such an orientation.
///
/// Entries in this table are laid out in row-major order:
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
    pub const TOTAL_NUM_ORIENTATIONS: usize = (Rotation::MAX as usize) * (Column::MAX as usize);

    /// A sentinel / "null" orientation.
    ///
    /// This corresponds to the reserved "X" cell in the orientation table: `(rotation=0, column=9)`.
    /// It is intentionally invalid for all pieces (every piece has width >= 1, so column 9 is out of
    /// bounds for rotation 0).
    pub const NULL: Self = Self {
        rotation: Rotation(0),
        column: Column(Column::MAX - 1),
    };

    pub const DEFAULT: Self = Self::NULL;

    pub const ALL: [Self; Self::TOTAL_NUM_ORIENTATIONS] = {
        let mut orientations = [Self::DEFAULT; Self::TOTAL_NUM_ORIENTATIONS];
        let mut i = 0;
        while i < orientations.len() {
            orientations[i] = Self::from_index(i as u8);
            i += 1;
        }
        orientations
    };

    /// Returns true if this orientation is the sentinel "null" orientation.
    #[inline_conditioned(always)]
    pub const fn is_null(self) -> bool {
        self.rotation.0 == 0 && self.column.0 == (Column::MAX - 1)
    }

    // One mask per piece (O, I, S, Z, T, L, J) in piece-index order.
    //
    // Bit i corresponds to orientation index i (see `TetrisPieceOrientation::index`).
    pub const ORIENTATION_MASKS_BY_PIECE: [u64; constants::NUM_TETRIS_PIECES] = {
        let mut out = [0u64; constants::NUM_TETRIS_PIECES];
        let pieces = TetrisPiece::all();

        let mut i: usize = 0;
        while i < constants::NUM_TETRIS_PIECES {
            let piece = pieces[i];
            // out[i] = Self::mask_for_piece_const(piece);
            out[i] = {
                let mut mask = 0u64;
                let cols = Column::MAX as usize; // 10
                let mut r: u8 = 0;
                let r_max: u8 = piece.num_rotations();

                while r < r_max {
                    let width = piece.width(Rotation(r)) as usize;
                    let k = cols - width + 1; // number of valid columns for this rotation
                    let base = (r as usize) * cols;

                    let mut c: usize = 0;
                    while c < k {
                        let idx = (base + c) as u32;
                        mask |= 1u64 << idx;
                        c += 1;
                    }

                    r += 1;
                }

                mask
            };
            i += 1;
        }

        out
    };

    pub const NUM_ORIENTATIONS_BY_PIECE: [u32; constants::NUM_TETRIS_PIECES] = {
        let mut counts = [0u32; constants::NUM_TETRIS_PIECES];
        let mut i = 0;
        while i < constants::NUM_TETRIS_PIECES {
            counts[i] = Self::ORIENTATION_MASKS_BY_PIECE[i].count_ones();
            i += 1;
        }
        counts
    };

    /// Get the index of this orientation.
    ///
    /// There are a finite number of possible orientations for a piece.
    /// This function returns the numeric index of this orientation
    pub const fn index(self) -> u8 {
        debug_assert!(
            self.column.0 < Column::MAX,
            "TetrisPieceOrientation::index: column out of bounds"
        );
        self.rotation.0 * Column::MAX + self.column.0
    }

    /// Get an orientation from an index.
    ///
    /// This is the inverse of [`index`](TetrisPieceOrientation::index).
    #[inline_conditioned(always)]
    pub const fn from_index(index: u8) -> Self {
        debug_assert!(
            index < Self::TOTAL_NUM_ORIENTATIONS as u8,
            "TetrisPieceOrientation::from_index: index out of bounds"
        );
        Self {
            rotation: Rotation(index / Column::MAX),
            column: Column(index % Column::MAX),
        }
    }

    /// Get a binary mask of all orientations for a piece.
    ///
    /// This is useful for creating a mask for a piece in a batch.
    #[inline_conditioned(always)]
    pub const fn bitmask_from_piece(
        piece: TetrisPiece,
    ) -> BitMask<{ TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS }> {
        BitMask::new_from_u64(Self::ORIENTATION_MASKS_BY_PIECE[piece.index() as usize])
    }

    #[inline_conditioned(always)]
    pub fn rand_orientation_from_piece<R: Rng + ?Sized>(
        piece: TetrisPiece,
        rng: &mut R,
    ) -> TetrisPieceOrientation {
        let mask = Self::ORIENTATION_MASKS_BY_PIECE[piece.index() as usize];
        let rand_idx = crate::utils::choose_set_bit_u64(mask, rng)
            .expect("rand_orientation_from_piece: piece orientation mask is empty");
        TetrisPieceOrientation::from_index(rand_idx as u8)
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
        let piece: TetrisPiece = rng.sample(StandardUniform);
        let orientation: TetrisPieceOrientation =
            TetrisPieceOrientation::rand_orientation_from_piece(piece, rng);
        TetrisPiecePlacement { piece, orientation }
    }
}

impl TetrisPiecePlacement {
    const DEFAULT: Self = Self {
        piece: TetrisPiece::DEFAULT,
        orientation: TetrisPieceOrientation::DEFAULT,
    };

    // Precompute placement counts per piece.
    // This is also the number of orientations for each piece.
    const PIECE_PLACEMENT_COUNTS: [usize; constants::NUM_TETRIS_PIECES] = {
        let pieces = TetrisPiece::all();
        let mut counts = [0usize; constants::NUM_TETRIS_PIECES];

        let mut i = 0usize;
        while i < constants::NUM_TETRIS_PIECES {
            let piece = pieces[i];
            let mut r = 0u8;
            let mut count = 0usize;
            while r < piece.num_rotations() {
                count += (((constants::COLS as u8) - piece.width(Rotation(r))) + 1) as usize;
                r += 1;
            }
            counts[i] = count;
            i += 1;
        }
        counts
    };

    // Precompute start indices for each piece in the global array
    const PIECE_START_INDICES: [usize; constants::NUM_TETRIS_PIECES] = {
        let mut indices = [0; constants::NUM_TETRIS_PIECES];
        let mut i = 1;
        while i < constants::NUM_TETRIS_PIECES {
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
    pub const NUM_PLACEMENTS: usize = {
        let all_pieces = TetrisPiece::all();
        let mut total_placements = 0;

        let mut i = 0;
        while i < all_pieces.len() {
            let piece = all_pieces[i];
            let mut r = 0;
            while r < piece.num_rotations() {
                total_placements +=
                    (((constants::COLS as u8) - piece.width(Rotation(r))) + 1) as usize;
                r += 1;
            }
            i += 1;
        }

        total_placements
    };

    pub const ALL_PLACEMENTS: [Self; Self::NUM_PLACEMENTS] = {
        let mut placements: [Self; Self::NUM_PLACEMENTS] = [Self::DEFAULT; Self::NUM_PLACEMENTS];

        let all_pieces = TetrisPiece::all();

        let mut placement_id = 0;

        let mut i = 0;
        while i < all_pieces.len() {
            let piece = all_pieces[i];
            let mut r = 0;
            while r < piece.num_rotations() {
                let mut c = 0;
                while c <= (constants::COLS as u8) - piece.width(Rotation(r)) {
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
    };

    /// Get the global placement index (into `ALL_PLACEMENTS`) for this placement.
    ///
    /// The global ordering is piece-major, then rotation-major, then column-major,
    /// matching `all_placements()`.
    #[inline_conditioned(always)]
    pub const fn index(&self) -> u8 {
        let piece = self.piece;
        let piece_idx = piece.index() as usize;
        let start = Self::PIECE_START_INDICES[piece_idx];

        let mut offset = 0usize;
        let mut r: u8 = 0;
        while r < self.orientation.rotation.0 {
            let k = ((constants::COLS as u8) - piece.width(Rotation(r)) + 1) as usize;
            offset += k;
            r += 1;
        }

        offset += self.orientation.column.0 as usize;
        debug_assert!(
            start + offset < Self::NUM_PLACEMENTS,
            "TetrisPiecePlacement::index: out of bounds"
        );
        (start + offset) as u8
    }

    /// Get a placement from its global placement index (into `ALL_PLACEMENTS`).
    #[inline_conditioned(always)]
    pub const fn from_index(index: u8) -> Self {
        debug_assert!(
            (index as usize) < Self::NUM_PLACEMENTS,
            "TetrisPiecePlacement::from_index: index out of bounds"
        );
        Self::ALL_PLACEMENTS[index as usize]
    }

    /// Get the contiguous global index range in `ALL_PLACEMENTS` for a given piece.
    #[inline_conditioned(always)]
    pub const fn indices_from_piece(piece: TetrisPiece) -> std::ops::Range<usize> {
        let piece_idx = piece.index() as usize;
        let start = Self::PIECE_START_INDICES[piece_idx];
        let len = Self::PIECE_PLACEMENT_COUNTS[piece_idx];
        start..(start + len)
    }

    /// Get all placements for a given piece, in the same order as `ALL_PLACEMENTS`.
    #[inline_conditioned(always)]
    pub fn all_from_piece<'a>(piece: TetrisPiece) -> &'a [Self] {
        let r = Self::indices_from_piece(piece);
        &Self::ALL_PLACEMENTS[r]
    }
}

// Compile-time sanity check: placement indices must fit in a `u8` (many parts of the codebase store
// placement indices as `u8`).
const _: () = assert!(
    TetrisPiecePlacement::NUM_PLACEMENTS <= (u8::MAX as usize),
    "TetrisPiecePlacement::NUM_PLACEMENTS must fit in u8"
);

/// Swap two 3-bit "lanes" inside a packed `u64`.
///
/// Treats `x` as a sequence of 3-bit chunks (lane 0 is bits 0..=2, lane 1 is bits 3..=5, etc.)
/// and returns a copy of `x` with the chunks at indices `i` and `j` swapped.
///
/// # Example
/// ```rust
/// use tetris_game::tetris::swap_3bit_chunks;
/// // lanes: [1,2,3]
/// let x = (1u64) | (2u64 << 3) | (3u64 << 6);
/// // swap lane 0 and lane 2 -> [3,2,1]
/// let y = swap_3bit_chunks(x, 0, 2);
/// assert_eq!(y & 0b111, 3);
/// assert_eq!((y >> 3) & 0b111, 2);
/// assert_eq!((y >> 6) & 0b111, 1);
/// ```
#[inline_conditioned(always)]
pub const fn swap_3bit_chunks(x: u64, i: u32, j: u32) -> u64 {
    let bit_idx_i = i * 3;
    let bit_idx_j = j * 3;
    let d = ((x >> bit_idx_i) ^ (x >> bit_idx_j)) & 0b111;
    x ^ (d << bit_idx_i) ^ (d << bit_idx_j)
}

/// The default ordered 7-bag stream with pieces 0-6 packed in 3-bit chunks.
///
/// Layout: `[O=0, I=1, S=2, Z=3, T=4, L=5, J=6]` where piece 0 (O) is in bits 0..3.
/// This constant is used as the starting point for Fisher-Yates shuffling.
pub const ORDERED_7: u64 =
    (1u64 << 3) | (2u64 << 6) | (3u64 << 9) | (4u64 << 12) | (5u64 << 15) | (6u64 << 18);

/// Generates a shuffled 7-bag stream using Fisher-Yates algorithm.
///
/// Uses 6 bytes of random data from the RNG to produce a uniformly random
/// permutation of the 7 pieces packed into a `u64`.
#[inline_conditioned(always)]
pub const fn fisher_yates_7bag_stream_from_seed(rng: &mut TetrisGameRng) -> u64 {
    let mut s = ORDERED_7;
    let r = rng.next_u64();

    // Unrolled Fisher–Yates:
    // j = floor(byte * m / 256), where m is 7, 6, 5, 4, 3, 2 for i = 6..1.
    repeat_idx_unroll!(6, K, {
        const I: u32 = (6usize.wrapping_sub(K as usize)) as u32;
        const M: u16 = (7usize.wrapping_sub(K as usize)) as u16;
        let byte = (r >> ((K * 8) as u32)) as u8;
        let j = ((byte as u16 * M) >> 8) as u32;
        s = swap_3bit_chunks(s, I, j);
    });

    s
}

/// A Tetris piece bag (7-bag) encoded as a compact `u64` stream.
///
/// A bag starts with 7 pieces (one of each tetromino). Pieces are drawn without replacement;
/// once the bag is empty, it is refilled with a new random permutation.
///
/// This bag stores the current permutation as **7 packed 3-bit chunks** in a `u64`:
///
/// ```text
/// bit index:  ... 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
/// chunks:     ... [ 6 ] [ 5 ] [ 4 ] [ 3 ] [ 2 ] [ 1 ] [ 0 ]
///                each chunk is a 3-bit piece index in 0..=6 (TetrisPiece::index()).
/// ```
///
/// - `stream & 0b111` is the **next** piece to draw.
/// - `rand_next()` / `next_piece()` consumes by shifting `stream >>= 3` and decrementing `remaining`.
/// - When `remaining == 0`, the bag is refilled with a fresh Fisher–Yates-shuffled 7-bag stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TetrisPieceBag {
    stream: u64,
    pub remaining: u8,
}

impl Default for TetrisPieceBag {
    fn default() -> Self {
        Self::new_ordered()
    }
}

impl Display for TetrisPieceBag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pieces = Vec::new();
        for i in 0..(self.remaining as u32) {
            let idx = ((self.stream >> (i * 3)) & 0b111) as u8;
            pieces.push(TetrisPiece::new(idx).to_string());
        }
        write!(f, "{:?}", pieces)
    }
}

impl TetrisPieceBag {
    /// Creates a new bag with pieces in deterministic order (O, I, S, Z, T, L, J).
    ///
    /// Useful for testing when you need predictable piece sequences.
    #[must_use]
    pub const fn new_ordered() -> Self {
        Self {
            stream: ORDERED_7,
            remaining: 7,
        }
    }

    /// Creates a new bag with pieces in random order using the provided RNG.
    ///
    /// This is the standard way to initialize a bag for gameplay.
    #[must_use]
    pub const fn new_rand(rng: &mut TetrisGameRng) -> Self {
        Self {
            stream: fisher_yates_7bag_stream_from_seed(rng),
            remaining: 7,
        }
    }

    /// Refills the bag with a new random permutation of all 7 pieces.
    ///
    /// Called automatically by [`rand_next`](Self::rand_next) when the bag is empty.
    #[inline_conditioned(always)]
    pub const fn rand_fill(&mut self, rng: &mut TetrisGameRng) {
        self.stream = fisher_yates_7bag_stream_from_seed(rng);
        self.remaining = 7;
    }

    /// Creates a bag from a [`TetrisPieceBagState`] bitmask.
    ///
    /// The pieces are added in index order (O first, then I, S, Z, T, L, J).
    /// Use [`shuffle`](Self::shuffle) afterwards if you need random ordering.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn from_bag_state(bag_state: TetrisPieceBagState) -> Self {
        let m: u8 = (u8::from(bag_state)) & TetrisPieceBagState::FULL_MASK;

        let mut stream: u64 = 0;
        let mut out_k: u32 = 0;

        repeat_idx_unroll!(7, K, {
            let present: u32 = ((m >> K) & 1) as u32; // 0 or 1
            let mask: u64 = 0u64.wrapping_sub(present as u64); // 0x0 or 0xFFFF...

            // If present==0, mask clears it; if present==1, it writes into lane out_k.
            stream |= ((K as u64) << (out_k * 3)) & mask;

            // Only advances when present==1
            out_k += present;
        });

        Self {
            stream,
            remaining: out_k as u8,
        }
    }

    /// Shuffles the remaining pieces in the bag using a Fisher–Yates algorithm.
    ///
    /// This performs an in-place shuffle of the current bag (encoded in `self.stream`) so that
    /// all permutations are equally likely, using random bits drawn from the given RNG.
    ///
    /// Only the currently remaining pieces (up to 7) are shuffled; the rest of the stream is untouched.
    /// The algorithm is fully unrolled for performance and works even if there are fewer than 7 pieces remaining.
    #[inline_conditioned(always)]
    pub const fn shuffle(&mut self, rng: &mut TetrisGameRng) {
        let rem: u32 = self.remaining as u32;
        let r: u64 = rng.next_u64();

        let mut s = self.stream;

        // Fisher–Yates over lanes [0..rem):
        // for i in 0..rem-1: j = i + rand(0..rem-i); swap(i,j)
        //
        // Unrolled to 6 steps (enough for max rem=7), and masked to no-op when m<=1.
        repeat_idx_unroll!(6, K, {
            let m: u32 = rem.saturating_sub(K as u32);
            let active: u32 = (m > 1) as u32;
            let byte: u8 = (r >> ((K * 8) as u32)) as u8;
            let j_off: u32 = ((((byte as u16) * (m as u16)) >> 8) as u32) * active;
            s = swap_3bit_chunks(s, K as u32, (K as u32) + j_off);
        });

        self.stream = s;
    }

    /// Checks if the bag currently contains the specified piece.
    ///
    /// Uses SIMD-friendly parallel comparison across all 3-bit lanes.
    #[inline_conditioned(always)]
    #[must_use]
    pub fn contains(&self, piece: TetrisPiece) -> bool {
        // Broadcast the lowest 3 bits of the piece index across the u64, repeating
        // the 3-bit pattern in each 3-bit lane.
        //
        // Example (piece_idx = 0b101):
        // lanes: [101][101][101]...
        const LANE_LSB_MASK: u64 = 0x9249_2492_4924_9249; // bits set at positions 0, 3, 6, ...

        // Broadcast the piece index across the u64 by 3-bit chunks.
        // Then XOR the broadcasted piece index with the stream.
        let p_idx = piece.index() as u8 & 0b111;
        let y = self.stream ^ (p_idx as u64).wrapping_mul(LANE_LSB_MASK);

        // Compare each 3-bit lane against `piece_idx` in parallel:
        // - XOR yields 0 in a lane iff that lane equals piece_idx
        // - `z = y | (y>>1) | (y>>2)` collapses each lane to its LSB position
        // - lane is equal iff the collapsed bit at the lane's LSB is 0
        let z = y | (y >> 1) | (y >> 2);

        // Only consider the first `remaining` lanes (avoid counting unused lanes).
        let used = ((1u64 << (self.remaining as u32 * 3)) - 1) & LANE_LSB_MASK; // works for remaining=0 too
        ((!z) & used) != 0
    }

    /// Draws and removes the next piece from the bag.
    ///
    /// If the bag is empty, it is automatically refilled with a new random
    /// permutation before drawing.
    #[inline_conditioned(always)]
    pub fn rand_next(&mut self, rng: &mut TetrisGameRng) -> TetrisPiece {
        if self.remaining == 0 {
            self.rand_fill(rng);
        }

        let piece = TetrisPiece::new(self.stream as u8 & 0b111);
        self.stream >>= 3;
        self.remaining -= 1;
        piece
    }
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
/// | J | L | T | Z | S | I | O | - |
/// +---+---+---+---+---+---+---+---+
/// ```
///
/// The '-' bit is unused.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TetrisPieceBagState(u8);

impl std::fmt::Display for TetrisPieceBagState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BagState(")?;
        let mut first = true;
        for piece in TetrisPiece::all() {
            if self.contains(piece) {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                write!(f, "{}", piece)?;
            }
        }
        write!(f, ")")
    }
}

impl const From<u8> for TetrisPieceBagState {
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl const From<TetrisPieceBagState> for u8 {
    fn from(state: TetrisPieceBagState) -> Self {
        state.0
    }
}

impl Default for TetrisPieceBagState {
    fn default() -> Self {
        Self::new()
    }
}

impl TetrisPieceBagState {
    pub const SIZE: usize = constants::NUM_TETRIS_PIECES;
    pub const FULL_MASK: u8 = 0b0111_1111;

    /// Creates a new bag containing all 7 standard Tetris pieces.
    ///
    /// The bag is initialized with all pieces (O, I, S, Z, T, L, J) present,
    /// represented internally as a bitmask where each bit corresponds to a piece.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::TetrisPieceBagState;
    ///
    /// let bag = TetrisPieceBagState::new();
    /// assert_eq!(bag.count(), 7);
    /// ```
    pub const fn new() -> Self {
        Self(Self::FULL_MASK)
    }

    /// Returns the number of pieces remaining in the bag.
    ///
    /// This operation counts the set bits in the internal bitmask representation,
    /// where each set bit represents a piece that is still in the bag.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::new();
    /// assert_eq!(bag.count(), 7);
    ///
    /// bag.remove(TetrisPiece::O_PIECE);
    /// assert_eq!(bag.count(), 6);
    /// ```
    #[inline_conditioned(always)]
    pub const fn count(&self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Returns `true` if the bag is empty (contains no pieces).
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::TetrisPieceBagState;
    ///
    /// let mut bag = TetrisPieceBagState::new();
    /// assert!(!bag.is_empty());
    /// for _ in 0..7 {
    ///     bag.pop();
    /// }
    /// assert!(bag.is_empty());
    /// ```
    #[inline_conditioned(always)]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Checks if the bag contains a specific piece.
    ///
    /// Returns `true` if the piece is still in the bag, `false` otherwise.
    /// This operation uses a bitwise AND to check if the piece's bit is set
    /// in the internal bitmask.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::new();
    /// assert!(bag.contains(TetrisPiece::I_PIECE));
    ///
    /// bag.remove(TetrisPiece::I_PIECE);
    /// assert!(!bag.contains(TetrisPiece::I_PIECE));
    /// ```
    #[inline_conditioned(always)]
    pub const fn contains(&self, piece: TetrisPiece) -> bool {
        let mask: u8 = piece.into();
        (self.0 & mask) != 0
    }

    /// Refills the bag with all 7 standard Tetris pieces.
    ///
    /// This operation resets the bag to its initial state, containing all pieces
    /// (O, I, S, Z, T, L, J). Any previously removed pieces are restored.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::new();
    /// bag.remove(TetrisPiece::O_PIECE);
    /// bag.remove(TetrisPiece::I_PIECE);
    /// assert_eq!(bag.count(), 5);
    ///
    /// bag.fill();
    /// assert_eq!(bag.count(), 7);
    /// assert!(bag.contains(TetrisPiece::O_PIECE));
    /// assert!(bag.contains(TetrisPiece::I_PIECE));
    /// ```
    #[inline_conditioned(always)]
    pub const fn fill(&mut self) {
        self.0 = Self::FULL_MASK;
    }

    /// Removes a specific piece from the bag.
    ///
    /// This operation clears the bit corresponding to the specified piece using
    /// a bitwise AND with the inverted piece mask. If the piece is not in the bag,
    /// this operation has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::new();
    /// assert_eq!(bag.count(), 7);
    ///
    /// bag.remove(TetrisPiece::T_PIECE);
    /// assert_eq!(bag.count(), 6);
    /// assert!(!bag.contains(TetrisPiece::T_PIECE));
    ///
    /// // Removing again has no effect
    /// bag.remove(TetrisPiece::T_PIECE);
    /// assert_eq!(bag.count(), 6);
    /// ```
    pub const fn remove(&mut self, piece: TetrisPiece) {
        let mask: u8 = piece.into();
        self.0 &= !mask;
    }

    /// Removes and returns the lowest-index piece from the bag.
    ///
    /// This operation finds the lowest set bit in the internal bitmask using
    /// `trailing_zeros()`, removes that piece from the bag, and returns it.
    /// The piece order is: O(0), I(1), S(2), Z(3), T(4), L(5), J(6).
    ///
    /// When the bag is empty, returns [`TetrisPiece::NULL_PIECE`] without
    /// modifying the bag. This is implemented using branchless bit manipulation
    /// for optimal performance.
    ///
    /// # Returns
    ///
    /// * The lowest-index piece still in the bag, or
    /// * [`TetrisPiece::NULL_PIECE`] if the bag is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::new();
    ///
    /// // Pop pieces in order: O, I, S, Z, T, L, J
    /// assert_eq!(bag.pop(), TetrisPiece::O_PIECE);
    /// assert_eq!(bag.count(), 6);
    ///
    /// assert_eq!(bag.pop(), TetrisPiece::I_PIECE);
    /// assert_eq!(bag.count(), 5);
    ///
    /// // Remove all remaining pieces
    /// for _ in 0..5 {
    ///     bag.pop();
    /// }
    ///
    /// // Empty bag returns NULL_PIECE
    /// assert_eq!(bag.pop(), TetrisPiece::NULL_PIECE);
    /// assert_eq!(bag.count(), 0);
    /// ```
    pub const fn pop(&mut self) -> TetrisPiece {
        let tz = self.0.trailing_zeros() as u8;
        let tz = tz - (tz >> 3);
        let piece = TetrisPiece::new(tz);
        self.remove(piece);
        piece
    }

    /// Get all pieces in the bag as a stack-allocated HeaplessVec.
    ///
    /// This is much faster than heap allocation and works in const contexts.
    #[inline]
    #[must_use]
    pub fn as_heapless_vec(&self) -> HeaplessVec<TetrisPiece, 7> {
        let mut pieces = HeaplessVec::new();
        for piece in TetrisPiece::all() {
            if self.contains(piece) {
                let _ = pieces.try_push(piece);
            }
        }
        pieces
    }

    /// Iterate over pieces in the bag efficiently using bit manipulation.
    ///
    /// Much faster than checking each piece individually.
    #[inline]
    pub fn iter_pieces(&self) -> impl Iterator<Item = TetrisPiece> + '_ {
        TetrisPiece::all()
            .into_iter()
            .filter(move |&piece| self.contains(piece))
    }

    /// Iterate over all possible next bag states with the piece that was popped.
    ///
    /// Returns `(popped_piece, resulting_bag_state)` for each piece in the current bag.
    #[inline]
    pub fn iter_next_states(
        &self,
    ) -> impl Iterator<Item = (TetrisPiece, TetrisPieceBagState)> + '_ {
        self.iter_pieces().map(move |piece| {
            let mut next_bag = *self;
            next_bag.remove(piece);
            if next_bag.is_empty() {
                next_bag.fill();
            }
            (piece, next_bag)
        })
    }
}

/// A Tetris board represented as an array of columns, where each column is a u32.
///
/// # Memory Layout
///
/// The board uses a **column-major** representation optimized for piece placement:
/// - Each column is stored as a `u32`
/// - Bit 0 represents the bottom row (row 0)
/// - Higher bits represent higher rows (bit 1 = row 1, bit 2 = row 2, etc.)
///
/// # Why Column-Major?
///
/// This layout enables efficient piece placement:
/// 1. Pieces span multiple columns but have a fixed height per column
/// 2. Collision detection can be done column-by-column using bitwise operations
/// 3. Row clearing can use SIMD-friendly operations across columns
/// 4. Height calculation is a simple `leading_zeros()` operation per column
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisBoard([u32; constants::COLS]);

pub type TetrisBoardBinarySlice = [u8; constants::BOARD_SIZE];

impl Default for TetrisBoard {
    fn default() -> Self {
        Self::EMPTY_BOARD
    }
}

impl TetrisBoard {
    pub const SIZE: usize = constants::BOARD_SIZE;
    pub const WIDTH: usize = constants::COLS;
    pub const HEIGHT: usize = constants::ROWS;
    pub const NUM_TETRIS_CELL_STATES: usize = constants::NUM_TETRIS_CELL_STATES;

    pub const EMPTY_BOARD: Self = Self([0_u32; constants::COLS]);
    pub const FULL_BOARD: Self = {
        let mask = (1u32 << constants::ROWS) - 1; // Only fill playable rows (0..ROWS)
        Self([mask; constants::COLS])
    };

    #[must_use]
    pub const fn as_limbs(&self) -> [u32; constants::COLS] {
        self.0
    }
}

/// Converts TetrisBoard to a 40-byte array (column-major u32 native-endian limbs).
///
/// This is a zero-cost operation using `transmute`. The byte representation is in
/// the platform's native endianness.
///
/// # Examples
/// ```
/// use tetris_game::TetrisBoard;
/// let board = TetrisBoard::new();
/// let bytes: [u8; 40] = board.into();
/// // or
/// let bytes = <[u8; 40]>::from(board);
/// ```
impl const From<TetrisBoard> for [u8; 40] {
    #[inline(always)]
    fn from(board: TetrisBoard) -> Self {
        unsafe { std::mem::transmute(board.0) }
    }
}

/// Converts a 40-byte array to TetrisBoard (column-major u32 native-endian limbs).
///
/// This is a zero-cost operation using `transmute`. The byte representation must be
/// in the platform's native endianness.
///
/// # Examples
/// ```
/// use tetris_game::TetrisBoard;
/// let bytes = [0u8; 40];
/// let board: TetrisBoard = bytes.into();
/// // or
/// let board = TetrisBoard::from(bytes);
/// ```
impl const From<[u8; 40]> for TetrisBoard {
    #[inline(always)]
    fn from(bytes: [u8; 40]) -> Self {
        // SAFETY: [u8; 40] and [u32; 10] have the same size and alignment requirements
        unsafe {
            Self(std::mem::transmute::<[u8; 40], [u32; constants::COLS]>(
                bytes,
            ))
        }
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
    /// use tetris_game::tetris::TetrisBoard;
    /// let board = TetrisBoard::new();
    /// assert_eq!(board.get_bit(0, 0), false); // Empty cell at bottom-left
    /// ```
    #[inline_conditioned(always)]
    pub const fn get_bit(&self, col: usize, row: usize) -> bool {
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
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0); // Set bottom-left cell
    /// assert_eq!(board.get_bit(0, 0), true);
    /// ```
    #[inline_conditioned(always)]
    pub fn set_bit(&mut self, col: usize, row: usize) {
        self.0[col] |= 1 << row;
    }

    /// Clears a bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - The row index (0 to ROWS-1)
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// board.clear_bit(0, 0);
    /// assert_eq!(board.get_bit(0, 0), false);
    /// ```
    #[inline_conditioned(always)]
    pub fn clear_bit(&mut self, col: usize, row: usize) {
        self.0[col] &= !(1 << row);
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
    /// use tetris_game::tetris::TetrisBoard;
    /// use rand::thread_rng;
    /// let mut board = TetrisBoard::new();
    /// let mut rng = thread_rng();
    /// board.set_random_bits(10, &mut rng);
    /// ```
    pub fn set_random_bits<R: Rng + ?Sized>(&mut self, num_bits: usize, rng: &mut R) {
        for _ in 0..num_bits {
            let rand = rng.random::<u32>();
            let col = ((rand >> 16) as u16 % constants::COLS as u16) as usize;
            let row = ((rand & 0xFFFF) % constants::ROWS as u32) as usize;
            self.set_bit(col, row);
        }
    }
}

impl Display for TetrisBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, " {:2}| board", "r")?;
        for row in (0..constants::ACTUAL_ROWS).rev() {
            let mut row_str = String::new();
            for col in 0..constants::COLS {
                let bit = self.get_bit(col, row);
                row_str.push(if bit { '#' } else { '.' });
            }
            writeln!(f, "{:2} | {}", row, row_str)?;
        }
        Ok(())
    }
}

impl TetrisBoard {
    /// Converts the board to a binary slice representation in row-major order.
    /// Each cell is represented as a single bit (0 or 1).
    ///
    /// # Memory Layout
    ///
    /// The output is in row-major order, meaning the array is laid out as:
    /// - Indices 0-9: Row 0 (columns 0-9)
    /// - Indices 10-19: Row 1 (columns 0-9)
    /// - ...
    /// - Indices 190-199: Row 19 (columns 0-9)
    ///
    /// This layout is compatible with standard tensor reshape operations that
    /// expect row-major ordering (e.g., reshaping to [HEIGHT, WIDTH]).
    ///
    /// # Returns
    ///
    /// A fixed-size array of u8 representing the board state in row-major order
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// let binary = board.to_binary_slice();
    /// assert!(binary.iter().all(|&x| x == 0));
    /// ```
    #[inline_conditioned(always)]
    pub const fn to_binary_slice(&self) -> TetrisBoardBinarySlice {
        let mut result: TetrisBoardBinarySlice = [0u8; Self::SIZE];
        repeat_idx_unroll!(Self::SIZE / 8, I, {
            const I1: usize = 8 * I;
            result[I1] = ((self.0[I1 % constants::COLS] >> (I1 / constants::COLS)) & 1) as u8;
            const I2: usize = 8 * I + 1;
            result[I2] = ((self.0[I2 % constants::COLS] >> (I2 / constants::COLS)) & 1) as u8;
            const I3: usize = 8 * I + 2;
            result[I3] = ((self.0[I3 % constants::COLS] >> (I3 / constants::COLS)) & 1) as u8;
            const I4: usize = 8 * I + 3;
            result[I4] = ((self.0[I4 % constants::COLS] >> (I4 / constants::COLS)) & 1) as u8;
            const I5: usize = 8 * I + 4;
            result[I5] = ((self.0[I5 % constants::COLS] >> (I5 / constants::COLS)) & 1) as u8;
            const I6: usize = 8 * I + 5;
            result[I6] = ((self.0[I6 % constants::COLS] >> (I6 / constants::COLS)) & 1) as u8;
            const I7: usize = 8 * I + 6;
            result[I7] = ((self.0[I7 % constants::COLS] >> (I7 / constants::COLS)) & 1) as u8;
            const I8: usize = 8 * I + 7;
            result[I8] = ((self.0[I8 % constants::COLS] >> (I8 / constants::COLS)) & 1) as u8;
        });
        result
    }

    /// Creates a board from a binary slice representation in row-major order.
    ///
    /// # Arguments
    ///
    /// * `binary` - Fixed-size array of u8 representing the board state in row-major order
    ///              (see `to_binary_slice` for memory layout details)
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// let binary = board.to_binary_slice();
    /// let board2 = TetrisBoard::from_binary_slice(binary);
    /// assert_eq!(board2.get_bit(0, 0), true);
    /// ```
    #[inline_conditioned(always)]
    pub const fn from_binary_slice(binary: TetrisBoardBinarySlice) -> Self {
        let mut result = Self::EMPTY_BOARD;
        repeat_idx_unroll!(Self::SIZE / 8, I, {
            const I1: usize = 8 * I;
            result.0[I1 % constants::COLS] |= (binary[I1] as u32) << (I1 / constants::COLS);
            const I2: usize = 8 * I + 1;
            result.0[I2 % constants::COLS] |= (binary[I2] as u32) << (I2 / constants::COLS);
            const I3: usize = 8 * I + 2;
            result.0[I3 % constants::COLS] |= (binary[I3] as u32) << (I3 / constants::COLS);
            const I4: usize = 8 * I + 3;
            result.0[I4 % constants::COLS] |= (binary[I4] as u32) << (I4 / constants::COLS);
            const I5: usize = 8 * I + 4;
            result.0[I5 % constants::COLS] |= (binary[I5] as u32) << (I5 / constants::COLS);
            const I6: usize = 8 * I + 5;
            result.0[I6 % constants::COLS] |= (binary[I6] as u32) << (I6 / constants::COLS);
            const I7: usize = 8 * I + 6;
            result.0[I7 % constants::COLS] |= (binary[I7] as u32) << (I7 / constants::COLS);
            const I8: usize = 8 * I + 7;
            result.0[I8 % constants::COLS] |= (binary[I8] as u32) << (I8 / constants::COLS);
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
    /// use tetris_game::tetris::TetrisBoard;
    /// let board = TetrisBoard::new();
    /// assert_eq!(board.count(), 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self::EMPTY_BOARD
    }

    /// Returns the total number of filled cells in the board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// assert_eq!(board.count(), 1);
    /// ```
    #[inline_conditioned(always)]
    pub const fn count(&self) -> u32 {
        let mut acc = 0;
        repeat_idx_unroll!(constants::COLS, I, {
            acc += self.0[I].count_ones();
        });
        acc
    }

    /// Clears all cells in the board, setting them to empty.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
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
    /// use tetris_game::tetris::TetrisBoard;
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
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 1); // Set bit at row 1
    /// assert_eq!(board.height(), 2); // Height should be 2
    /// ```
    #[inline_conditioned(always)]
    pub const fn height(&self) -> u32 {
        let mut acc = 0;
        repeat_idx_unroll!(constants::COLS, I, {
            acc |= self.0[I];
        });
        u32::BITS - acc.leading_zeros()
    }

    /// Returns an array of the heights of each column (number of occupied cells from the bottom up).
    ///
    /// Height in Tetris is defined as 0 if the column is empty, otherwise 1 + the highest row index with a set bit.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 0);
    /// board.set_bit(1, 5);
    /// let heights = board.heights();
    /// assert_eq!(heights[0], 1);
    /// assert_eq!(heights[1], 6);
    /// ```
    #[inline_conditioned(always)]
    pub const fn heights(&self) -> [u32; constants::COLS] {
        let mut out = [0u32; constants::COLS];
        repeat_idx_unroll!(constants::COLS, I, {
            out[I] = u32::BITS - self.0[I].leading_zeros();
        });
        out
    }

    /// Returns the number of holes in each column of the board.
    ///
    /// A "hole" is defined as an empty cell (bit not set) that has at least one filled cell above it in the same column.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    ///
    /// // Create a hole: fill row 0, skip row 1, fill row 2 in column 0
    /// board.set_bit(0, 0);
    /// board.set_bit(0, 2);
    ///
    /// let holes = board.holes();
    /// assert_eq!(holes[0], 1); // Column 0 has 1 hole at row 1
    /// assert_eq!(holes[1], 0); // Column 1 has no holes
    /// ```
    #[inline_conditioned(always)]
    pub const fn holes(&self) -> [u32; constants::COLS] {
        let mut out = [0u32; constants::COLS];
        repeat_idx_unroll!(constants::COLS, I, {
            let height = u32::BITS - self.0[I].leading_zeros();
            let filled_cells = self.0[I].count_ones();
            out[I] = height - filled_cells;
        });
        out
    }

    /// Returns the total number of holes across all columns in the board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    ///
    /// // Create holes in two columns
    /// // Column 0: fill row 0, skip row 1, fill row 2 (1 hole)
    /// board.set_bit(0, 0);
    /// board.set_bit(0, 2);
    ///
    /// // Column 1: fill row 0, skip rows 1-2, fill row 3 (2 holes)
    /// board.set_bit(1, 0);
    /// board.set_bit(1, 3);
    ///
    /// assert_eq!(board.total_holes(), 3); // 1 + 2 = 3 total holes
    /// ```
    #[inline_conditioned(always)]
    pub const fn total_holes(&self) -> u32 {
        let mut sum = 0;
        repeat_idx_unroll!(constants::COLS, I, {
            let height = u32::BITS - self.0[I].leading_zeros();
            let filled_cells = self.0[I].count_ones();
            sum += height - filled_cells;
        });
        sum
    }

    /// Returns whether the game is lost by checking if any column exceeds the maximum height.
    ///
    /// A Tetris game is lost when any piece extends beyond the top of the board.
    /// This is detected by checking if any column has blocks above row ROWS-1.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// board.set_bit(0, 19); // Set bit at max valid height
    /// assert!(!board.is_lost());
    /// board.set_bit(0, 20); // Set bit above max height
    /// assert!(board.is_lost());
    /// ```
    #[inline_conditioned(always)]
    pub const fn is_lost(&self) -> bool {
        self.height() > constants::ROWS as u32
    }

    /// Clears all completely filled rows and shifts all cells above the cleared rows down.
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
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board = TetrisBoard::new();
    /// // Fill an entire row
    /// for col in 0..10 {
    ///     board.set_bit(col, 0);
    /// }
    /// assert_eq!(board.clear_filled_rows(), 1);
    /// assert_eq!(board.count(), 0);
    /// ```
    #[inline_conditioned(always)]
    pub const fn clear_filled_rows(&mut self) -> u32 {
        let mut filled_rows_mask = u32::MAX;
        repeat_idx_unroll!(constants::COLS, I, {
            filled_rows_mask &= self.0[I];
        });
        rshift_slice_from_mask_u32::<{ constants::COLS }, 4>(&mut self.0, filled_rows_mask);
        filled_rows_mask.count_ones()
    }

    /// Places a piece on the board; handles collision, drop, and line clearing.
    ///
    /// Steps:
    /// 1. If board is lost, return immediately.
    /// 2. Find where the piece will land by checking column heights.
    /// 3. Drop the piece as far as possible, then merge it into the board.
    /// 4. Clear filled rows and check for loss.
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
                const TRAILING: [u32; NUM_COLS] = {
                    let mut trailing_zeros = [0; NUM_COLS];
                    repeat_idx_unroll!(NUM_COLS, I, {
                        trailing_zeros[I] = $piece_cols[I].trailing_zeros();
                    });
                    trailing_zeros
                };

                let shift = {
                    let mut min_diff = u32::MAX;
                    repeat_idx_unroll!(NUM_COLS, I, {
                        let col_height = u32::BITS - $board_cols[I].leading_zeros();
                        let diff = col_height.abs_diff(TRAILING[I]);

                        // equivalient to min(min_diff, diff)
                        min_diff = min_diff
                            ^ ((min_diff ^ diff) & ((diff < min_diff) as u32).wrapping_neg());
                    });
                    min_diff
                };

                // Place the piece by OR-ing it with the board
                repeat_idx_unroll!(NUM_COLS, I, {
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
            _ => {
                debug_assert!(false, "Invalid piece or rotation");
                unsafe { std::hint::unreachable_unchecked() }
            }
        };

        let lines_cleared = self.clear_filled_rows();
        let is_lost = self.is_lost();
        if is_lost {
            // Mark board as lost by filling beyond playable area
            *self = Self([u32::MAX; constants::COLS]);
        }

        PlacementResult {
            lines_cleared,
            is_lost: IsLost(is_lost),
        }
    }
}

/// A newtype indicating whether a game has ended.
///
/// This type provides semantic clarity over a raw `bool` and enables
/// pattern matching with named constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct IsLost(bool);

impl Display for IsLost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IsLost({})", self.0)
    }
}

impl IsLost {
    pub const LOST: Self = Self(true);
    pub const NOT_LOST: Self = Self(false);
}

impl From<IsLost> for bool {
    fn from(val: IsLost) -> Self {
        val.0
    }
}

/// A deterministic random number generator for Tetris games.
///
/// # Purpose
///
/// This RNG provides reproducible randomness for Tetris games, enabling:
/// - Deterministic game replays from a seed
/// - Consistent testing and debugging
/// - Game state serialization/deserialization
///
/// # Usage
///
/// ```rust
/// use tetris_game::tetris::TetrisGameRng;
/// let mut rng = TetrisGameRng::new(42);
/// let value = rng.next_u64();  // Deterministic based on seed
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TetrisGameRng {
    pub seed: u64,
    pub state: u64,
    pub n: u64,
}

impl Default for TetrisGameRng {
    fn default() -> Self {
        Self {
            seed: 123,
            state: 123,
            n: 0,
        }
    }
}

impl TetrisGameRng {
    /// Applies the MurmurHash3 64-bit finalization mixer to a value.
    ///
    /// # Arguments
    ///
    /// * `x` - The 64-bit value to mix
    ///
    /// # Returns
    ///
    /// The mixed 64-bit value with improved bit distribution
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tetris_game::tetris::TetrisGameRng;
    ///
    /// let mut x = 12345u64;
    /// TetrisGameRng::fmix64(&mut x);
    /// assert_eq!(x, 1716623506685013753u64); // known pre-image
    /// ```
    #[inline_conditioned(always)]
    const fn fmix64(x: &mut u64) {
        *x ^= *x >> 33;
        *x = x.wrapping_mul(0xff51afd7ed558ccd);
        *x ^= *x >> 33;
        *x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        *x ^= *x >> 33;
    }

    /// Creates a new RNG with the given seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            state: seed,
            n: 0,
        }
    }

    /// Generates and returns the next random `u64` value.
    ///
    /// Advances the internal state using the MurmurHash3 finalizer.
    #[inline_conditioned(always)]
    pub const fn next_u64(&mut self) -> u64 {
        Self::fmix64(&mut self.state);
        self.n += 1;
        self.state
    }

    /// Peeks at the n-th future value without advancing state.
    ///
    /// `peek_n(1)` returns what `next_u64()` would return.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn peek_n(&self, n: usize) -> u64 {
        let mut state = self.state;
        let mut i = 0;
        while i < n {
            Self::fmix64(&mut state);
            i += 1;
        }
        state
    }

    /// Resets the RNG to its initial state (same seed).
    ///
    /// After calling `reset()`, the RNG will produce the same sequence
    /// of values as when it was first created.
    #[inline_conditioned(always)]
    pub const fn reset(&mut self) {
        self.state = self.seed;
        self.n = 0;
    }

    /// Resets the RNG with a new seed.
    ///
    /// This changes both the stored seed and the current state.
    #[inline_conditioned(always)]
    pub const fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.state = seed;
        self.n = 0;
    }
}

/// The result of applying a piece placement to a board.
///
/// Contains information about the outcome of the placement:
/// - Whether the game was lost (piece extended above the board)
/// - How many lines were cleared by the placement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementResult {
    /// Whether the game ended due to this placement.
    pub is_lost: IsLost,
    /// Number of complete rows cleared (0-4).
    pub lines_cleared: u32,
}

/// A complete Tetris game state.
///
/// This struct encapsulates all state needed to play and replay a Tetris game:
///
/// - **Board**: The current state of placed pieces
/// - **Current piece**: The piece the player is about to place
/// - **Bag**: The 7-bag randomizer ensuring fair piece distribution  
/// - **RNG**: Deterministic random number generator for reproducibility
/// - **Statistics**: Lines cleared and piece count
///
/// # Usage
///
/// The game follows a simple loop:
/// 1. Get valid placements via [`current_placements()`](Self::current_placements)
/// 2. Choose a placement and apply it via [`apply_placement()`](Self::apply_placement)
/// 3. Check if the game is lost via the returned [`PlacementResult`]
/// 4. Repeat until game over
///
/// # Determinism
///
/// Games created with the same seed produce identical piece sequences,
/// enabling replay and testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TetrisGame {
    /// The current board state.
    pub board: TetrisBoard,
    /// The piece that will be placed next.
    pub current_piece: TetrisPiece,
    /// The deterministic RNG (stores seed for reset).
    pub rng: TetrisGameRng,
    /// The 7-bag piece randomizer.
    pub bag: TetrisPieceBag,
    /// Total lines cleared since game start.
    pub lines_cleared: u32,
    /// Lines cleared by the most recent placement.
    pub recent_lines_cleared: u32,
    /// Total pieces placed since game start.
    pub piece_count: u32,
}

impl Display for TetrisGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TetrisGame:")?;
        writeln!(f, "  bag:")?;
        writeln!(f, "    {}", self.bag)?;
        writeln!(f, "  piece_buf:")?;
        writeln!(f, "    {}", self.current_piece)?;
        writeln!(f, "  lines_cleared:")?;
        writeln!(f, "    {}", self.lines_cleared)?;
        writeln!(f, "  board:")?;
        writeln!(f, "    {}", self.board)?;
        Ok(())
    }
}

impl Default for TetrisGame {
    fn default() -> Self {
        Self::new()
    }
}

impl TetrisGame {
    /// Create a new Tetris game.
    ///
    /// The game is initialized with a new board, a new bag, and a random piece
    /// popped from the bag.
    #[must_use]
    pub fn new() -> Self {
        let mut rng = TetrisGameRng::default();
        let mut bag = TetrisPieceBag::new_rand(&mut rng);
        let current_piece = bag.rand_next(&mut rng);
        Self {
            board: TetrisBoard::new(),
            current_piece,
            rng,
            bag,
            lines_cleared: 0,
            recent_lines_cleared: 0,
            piece_count: 0,
        }
    }

    /// Creates a new game with a specific seed for reproducible piece sequences.
    #[must_use]
    pub fn new_with_seed(seed: u64) -> Self {
        let mut rng = TetrisGameRng::new(seed);
        let mut bag = TetrisPieceBag::new_rand(&mut rng);
        let current_piece = bag.rand_next(&mut rng);
        Self {
            board: TetrisBoard::new(),
            current_piece,
            rng,
            bag,
            lines_cleared: 0,
            recent_lines_cleared: 0,
            piece_count: 0,
        }
    }

    /// Creates a game from a specific board state, bag state, piece, and seed.
    ///
    /// Useful for resuming a game or testing specific scenarios.
    #[must_use]
    pub fn new_with_board_bag_piece_seeded(
        board: TetrisBoard,
        bag_state: TetrisPieceBagState,
        current_piece: TetrisPiece,
        seed: u64,
    ) -> Self {
        let mut rng = TetrisGameRng::new(seed);
        let mut bag = TetrisPieceBag::from_bag_state(bag_state);
        bag.shuffle(&mut rng);

        Self {
            board,
            current_piece,
            rng,
            bag,
            lines_cleared: 0,
            recent_lines_cleared: 0,
            piece_count: 0,
        }
    }

    /// Replaces the bag, current piece, and reseeds the RNG.
    ///
    /// The board state and statistics are preserved.
    pub fn set_bag_piece_seeded(
        &mut self,
        bag_state: TetrisPieceBagState,
        piece: TetrisPiece,
        seed: u64,
    ) {
        self.rng.reseed(seed);
        self.bag = TetrisPieceBag::from_bag_state(bag_state);
        self.bag.shuffle(&mut self.rng);
        self.current_piece = piece;
    }

    /// Get the current placements that can be applied to the current piece.
    #[must_use]
    pub fn current_placements(&self) -> &[TetrisPiecePlacement] {
        TetrisPiecePlacement::all_from_piece(self.current_piece)
    }

    /// Apply a placement from orientation to the board.
    ///
    /// This will return true if the game is lost, false otherwise.
    /// Lines cleared are tracked by measuring the difference in height before and after the placement.
    ///
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_orientation(&mut self, orientation: TetrisPieceOrientation) -> PlacementResult {
        self.apply_placement(TetrisPiecePlacement {
            piece: self.current_piece,
            orientation,
        })
    }

    /// Apply a placement to the board.
    ///
    /// Returns `true` if the game is lost; otherwise `false`.
    /// Lines cleared are tracked by the difference in height before and after placement.
    /// If the game is not lost, the current piece is replaced with a new random piece.
    pub fn apply_placement(&mut self, placement: TetrisPiecePlacement) -> PlacementResult {
        debug_assert!(
            self.current_placements().contains(&placement),
            "Placement {} is not valid for current piece {}",
            placement,
            self.current_piece
        );

        let PlacementResult {
            is_lost,
            lines_cleared,
        } = self.board.apply_piece_placement(placement);
        if is_lost.into() {
            return PlacementResult {
                is_lost: IsLost::LOST,
                lines_cleared: 0,
            };
        }
        self.lines_cleared += lines_cleared as u32;
        self.recent_lines_cleared = lines_cleared as u32;
        self.current_piece = self.bag.rand_next(&mut self.rng);
        self.piece_count += 1;
        PlacementResult {
            is_lost: IsLost::NOT_LOST,
            lines_cleared: lines_cleared as u32,
        }
    }

    /// Plays a random valid placement.
    ///
    /// Selects uniformly at random from [`current_placements()`](Self::current_placements)
    /// and applies it. Useful for Monte Carlo simulations and random playouts.
    pub fn play_random(&mut self) -> PlacementResult {
        let placements = self.current_placements();

        let n = placements.len() as u64;
        let x = (self.rng.peek_n(1) >> 32) as u64;
        let idx = ((x * n) >> 32) as usize;
        let placement = placements[idx];

        self.apply_placement(placement)
    }

    /// Returns all possible resulting boards from placing the current piece.
    ///
    /// Each board represents the state after one valid placement (before the next piece).
    #[must_use]
    pub fn next_boards(&self) -> Vec<TetrisBoard> {
        let placements = TetrisPiecePlacement::all_from_piece(self.current_piece);
        placements
            .iter()
            .map(|placement| {
                let mut board_copy = self.board;
                board_copy.apply_piece_placement(*placement);
                board_copy
            })
            .collect()
    }

    /// Peek at the n-th next piece: n = 0 is the current piece, 1 is the next, etc.
    ///
    /// Returns the piece at offset n in the piece stream (without modifying state).
    #[must_use]
    pub fn peek_nth_next_piece(&self, n: usize) -> TetrisPiece {
        if n == 0 {
            self.current_piece
        } else {
            let mut bag_copy = self.bag;
            let mut rng_copy = self.rng;
            let mut next_piece = self.current_piece;
            for _ in 0..n {
                next_piece = bag_copy.rand_next(&mut rng_copy);
            }
            next_piece
        }
    }

    /// Resets the game to its initial state.
    ///
    /// Clears the board, resets statistics, and generates a new piece sequence.
    ///
    /// # Arguments
    ///
    /// * `new_seed` - If `Some(seed)`, uses a new seed. If `None`, replays
    ///   the original seed (same piece sequence as the initial game).
    pub fn reset(&mut self, new_seed: Option<u64>) {
        self.board.clear();
        new_seed
            .map(|s| self.rng.reseed(s))
            .unwrap_or_else(|| self.rng.reset());
        self.bag.rand_fill(&mut self.rng);
        self.current_piece = self.bag.rand_next(&mut self.rng);
        self.lines_cleared = 0;
        self.piece_count = 0;
    }
}

/// Maximum number of games in a [`TetrisGameSet`].
///
/// This limit exists because `TetrisGameSet` is stack-allocated for performance.
/// Adjust this constant if you need larger batch sizes.
pub const MAX_GAMES: usize = 1024;

/// Error type for `TetrisGameSet` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TetrisGameSetError {
    /// The requested number of games exceeds the maximum capacity.
    TooManyGames { requested: usize, max: usize },
}

impl std::fmt::Display for TetrisGameSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyGames { requested, max } => {
                write!(f, "requested {requested} games, but maximum is {max}")
            }
        }
    }
}

impl std::error::Error for TetrisGameSetError {}

/// A collection of Tetris games for parallel/batch operations.
///
/// This struct enables efficient batch processing of multiple games simultaneously,
/// useful for:
/// - Monte Carlo simulations
/// - Reinforcement learning with parallel environments
/// - Batch evaluation of AI agents
///
/// The set is stack-allocated with a maximum capacity of [`MAX_GAMES`].
#[derive(Clone, Copy, Debug)]
pub struct TetrisGameSet(pub HeaplessVec<TetrisGame, MAX_GAMES>);

impl TetrisGameSet {
    /// Create a new TetrisGameSet with N default games.
    pub fn new(num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new()));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the provided seed.
    ///
    /// Each game gets a slightly different seed (seed + index).
    ///
    pub fn new_with_seed(seed: u64, num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|i| games.push(TetrisGame::new_with_seed(seed + i as u64)));
        Self(games)
    }

    /// Create a new TetrisGameSet with N games using the same seed.
    pub fn new_with_same_seed(seed: u64, num_games: usize) -> Self {
        assert!(
            num_games <= MAX_GAMES,
            "Too many games. MAX_GAMES = {}",
            MAX_GAMES
        );
        let mut games = HeaplessVec::new();
        (0..num_games).for_each(|_| games.push(TetrisGame::new_with_seed(seed)));
        Self(games)
    }

    /// Returns the number of games in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the set contains no games.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Create a `TetrisGameSet` from a slice of games.
    pub fn from_games(games: &[TetrisGame]) -> Self {
        assert!(
            games.len() <= MAX_GAMES,
            "games.len() must be less than or equal to MAX_GAMES"
        );
        let mut input_games = HeaplessVec::new();
        input_games.fill_from_slice(games);
        Self(input_games)
    }

    /// Returns all boards as a vector.
    #[must_use]
    pub fn boards(&self) -> HeaplessVec<TetrisBoard, MAX_GAMES> {
        self.0.map(|game| game.board)
    }

    /// Returns all current pieces as a vector.
    #[must_use]
    pub fn current_pieces(&self) -> HeaplessVec<TetrisPiece, MAX_GAMES> {
        self.0.map(|game| game.current_piece)
    }

    /// Returns all piece counts as a vector.
    #[must_use]
    pub fn piece_counts(&self) -> HeaplessVec<u32, MAX_GAMES> {
        self.0.map(|game| game.piece_count)
    }

    /// Get the current placements for all games.
    ///
    /// These are the placements that can be applied to the current piece.
    #[must_use]
    pub fn current_placements(&self) -> Vec<&[TetrisPiecePlacement]> {
        self.0
            .into_iter()
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
            .into_iter_mut()
            .zip(placements)
            .map(|(game, &placement)| game.apply_placement(placement).is_lost)
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
            .into_iter_mut()
            .zip(orientations)
            .map(|(game, &orientation)| {
                game.apply_placement(TetrisPiecePlacement {
                    piece: game.current_piece,
                    orientation,
                })
                .is_lost
            })
            .collect()
    }

    /// Resets any games that are in a lost state.
    ///
    /// Each lost game is reset with a new seed derived from its RNG.
    /// Returns the number of games that were reset.
    pub fn reset_lost_games(&mut self) -> usize {
        self.0
            .into_iter_mut()
            .map(|game| {
                if game.board.is_lost() {
                    let next_seed = game.rng.next_u64();
                    game.reset(Some(next_seed));
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    /// Resets all games to their initial state (original seeds).
    pub fn reset_all(&mut self) {
        self.0.into_iter_mut().for_each(|game| game.reset(None));
    }

    /// Permute the gameset using the provided permutation vector.
    ///
    /// The permutation vector must be the same length as the gameset and contain
    /// valid indices (0..len). Each index should appear exactly once.
    pub fn permute(&mut self, permutation: &[usize]) {
        assert_eq!(permutation.len(), self.len(), "Permutation length mismatch");
        let mut new_games = HeaplessVec::new();
        for &idx in permutation {
            new_games.push(*self.0.get(idx).expect("permutation index out of bounds"));
        }
        self.0 = new_games;
    }

    /// Removes all games that are in a lost state from the set.
    pub fn drop_lost_games(&mut self) {
        self.0.retain(|game| !game.board.is_lost());
    }
}

impl Index<usize> for TetrisGameSet {
    type Output = TetrisGame;

    fn index(&self, index: usize) -> &Self::Output {
        self.0
            .get(index)
            .expect("TetrisGameSet index out of bounds")
    }
}

impl IndexMut<usize> for TetrisGameSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0
            .get_mut(index)
            .expect("TetrisGameSet index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_rotations() {
        fn rotations_reference_slow(piece: TetrisPiece) -> u8 {
            match piece {
                TetrisPiece::O_PIECE => 1,
                TetrisPiece::I_PIECE => 2,
                TetrisPiece::S_PIECE => 2,
                TetrisPiece::Z_PIECE => 2,
                TetrisPiece::T_PIECE => 4,
                TetrisPiece::L_PIECE => 4,
                TetrisPiece::J_PIECE => 4,
                _ => panic!("Invalid piece"),
            }
        }
        for i in 0..(TetrisPiece::NUM_PIECES as u8) {
            let piece = TetrisPiece::new(i);
            assert_eq!(
                piece.num_rotations(),
                rotations_reference_slow(piece),
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

        for i in 0..(TetrisPiece::NUM_PIECES as u8) {
            for r in 0..Rotation::MAX {
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
    fn test_orientation_index() {
        // Round-trip for the full orientation space.
        for index in 0..TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS {
            let orientation = TetrisPieceOrientation::from_index(index as u8);
            assert_eq!(
                orientation.index(),
                index as u8,
                "TetrisPieceOrientation index round-trip failed: from_index({index}) -> {orientation} -> {}",
                orientation.index()
            );
        }

        for piece in TetrisPiece::all() {
            for rotation in 0..Rotation::MAX {
                for column in 0..Column::MAX {
                    let orientation = TetrisPieceOrientation {
                        rotation: Rotation(rotation),
                        column: Column(column),
                    };
                    let idx = orientation.index();
                    let round_tripped = TetrisPieceOrientation::from_index(idx);
                    assert_eq!(
                        round_tripped, orientation,
                        "Orientation round-trip failed for piece={piece}, rotation={rotation}, column={column}: {orientation} -> idx={idx} -> {round_tripped}"
                    );
                }
            }
        }
    }

    /// Test board cell counting: empty, full, and each position
    #[test]
    fn test_board_count() {
        // Test empty board
        let mut board = TetrisBoard::new();
        assert_eq!(board.count(), 0);

        // Test full board (only the playable area)
        board.fill();
        assert_eq!(board.count(), (constants::ROWS * constants::COLS) as u32);

        // clear and check zero
        board.clear();
        assert_eq!(board.count(), 0);

        // Test each position individually counting along the way
        for row in 0..constants::ROWS {
            for col in 0..constants::COLS {
                let expected_count = (row * constants::COLS + col + 1) as u32;
                board.set_bit(col, row);
                assert_eq!(
                    board.count(),
                    expected_count,
                    "Count mismatch at col={}, row={}: expected {}, got {}",
                    col,
                    row,
                    expected_count,
                    board.count()
                );
            }
        }
    }

    #[test]
    fn test_board_set_and_get_bit() {
        let mut board = TetrisBoard::new();

        // Set every bit, assert it's set, and others are not
        for row in 0..constants::ROWS {
            for col in 0..constants::COLS {
                board.set_bit(col, row);
                assert!(
                    board.get_bit(col, row),
                    "Bit at (col {}, row {}) should be set",
                    col,
                    row
                );
            }
        }
        // Now unset them one by one and check they're unset
        for row in 0..constants::ROWS {
            for col in 0..constants::COLS {
                board.clear_bit(col, row);
                assert!(
                    !board.get_bit(col, row),
                    "Bit at (col {}, row {}) should be unset",
                    col,
                    row
                );
            }
        }
    }

    // For each cell in the board, set a single bit and assert the height
    #[test]
    fn test_board_height() {
        let mut board = TetrisBoard::new();
        assert_eq!(board.height(), 0);
        for row in 0..constants::ROWS {
            for col in 0..constants::COLS {
                board.clear();
                board.set_bit(col, row);
                let expected_height = (row + 1) as u32;
                assert_eq!(
                    board.height(),
                    expected_height,
                    "Failed at col={}, row={}: expected height {}, got {}",
                    col,
                    row,
                    expected_height,
                    board.height()
                );
            }
        }
    }

    #[test]
    fn test_drop_piece_simple() {
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
    }

    #[test]
    fn test_drop_piece_fuzz() {
        // fuzz test
        for _ in 0..10_000 {
            let piece = TetrisPiece::new(rand::random::<u8>() % (TetrisPiece::NUM_PIECES as u8));
            let rotation = Rotation(rand::random::<u8>() % Rotation::MAX);
            let column =
                Column((rand::random::<u8>()) % ((constants::COLS as u8) - piece.width(rotation)));
            TetrisBoard::new().apply_piece_placement(TetrisPiecePlacement {
                piece,
                orientation: TetrisPieceOrientation { rotation, column },
            });
        }
    }

    /// Test clearing 1, 2, 3, and 4 filled rows at all possible positions
    /// Summary: This test covers:
    /// - 1 row:  C(20,1) = 20 combinations
    /// - 2 rows: C(20,2) = 190 combinations
    /// - 3 rows: C(20,3) = 1140 combinations
    /// - 4 rows: C(20,4) = 4845 combinations
    /// Total: 6195 test cases
    #[test]
    fn test_clear_filled_rows() {
        let mut board = TetrisBoard::new();

        // Helper to generate all combinations of size k from 0..ROWS
        fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
            let mut result = Vec::new();
            let mut current = Vec::new();

            fn backtrack(
                start: usize,
                n: usize,
                k: usize,
                current: &mut Vec<usize>,
                result: &mut Vec<Vec<usize>>,
            ) {
                if current.len() == k {
                    result.push(current.clone());
                    return;
                }

                for i in start..n {
                    current.push(i);
                    backtrack(i + 1, n, k, current, result);
                    current.pop();
                }
            }

            backtrack(0, n, k, &mut current, &mut result);
            result
        }

        // Helper to fill specific rows
        let fill_rows = |board: &mut TetrisBoard, rows: &[usize]| {
            board.clear();
            for &row in rows {
                for col in 0..constants::COLS {
                    board.set_bit(col, row);
                }
            }
        };

        // Test all combinations for 1, 2, 3, and 4 filled rows
        for num_rows in 1..=4 {
            let all_combinations = combinations(constants::ROWS, num_rows);

            for (idx, rows) in all_combinations.iter().enumerate() {
                fill_rows(&mut board, rows);

                let cleared = board.clear_filled_rows();
                assert_eq!(
                    cleared,
                    num_rows as u32,
                    "Failed to clear {} rows at positions {:?} (combination {}/{})",
                    num_rows,
                    rows,
                    idx + 1,
                    all_combinations.len()
                );

                assert_eq!(
                    board.count(),
                    0,
                    "Board should be empty after clearing {} rows at {:?}, but has {} cells filled (combination {}/{})",
                    num_rows,
                    rows,
                    board.count(),
                    idx + 1,
                    all_combinations.len()
                );
            }
        }
    }

    #[test]
    fn test_loss_condition() {
        let mut board = TetrisBoard::new();
        assert!(!Into::<bool>::into(board.is_lost()));

        // Set a bit in the top 4 rows
        board.set_bit(0, constants::ROWS + 1);
        assert!(Into::<bool>::into(board.is_lost()));
        board.clear();
        assert!(!Into::<bool>::into(board.is_lost()));

        // Set a bit in the bottom row
        board.set_bit(0, 0);
        assert!(!Into::<bool>::into(board.is_lost()));
    }

    #[test]
    fn test_bag_state() {
        let mut bag = TetrisPieceBagState::new();
        assert_eq!(bag.count(), 7, "Bag should have 7 pieces");

        // New bag should have all pieces
        for piece in TetrisPiece::all() {
            assert!(bag.contains(piece), "Bag should contain piece {}", piece);
        }

        // Popping all pieces should empty the bag
        for _ in 0..7 {
            let piece = bag.pop();
            assert!(
                !bag.contains(piece),
                "Bag should not contain piece {}",
                piece
            );
        }
        assert_eq!(bag.count(), 0, "Bag should be empty");

        // popping an empty bag should result in a null piece
        let piece = bag.pop();
        assert_eq!(
            piece,
            TetrisPiece::NULL_PIECE,
            "Popping an empty bag should result in a null piece"
        );

        // check we can gofrom bag state to tetris piece bag
        let mut bag_state = TetrisPieceBagState::new();

        let bag = TetrisPieceBag::from_bag_state(bag_state);
        assert_eq!(bag.remaining, 7, "Bag should have 7 pieces");

        for i in 1..=7 {
            let piece = bag_state.pop();
            assert!(
                !bag_state.contains(piece),
                "Bag state should not contain piece {}",
                piece
            );
            assert_eq!(
                bag_state.count(),
                7 - i,
                "Bag state should have {} pieces",
                7 - i
            );

            let bag = TetrisPieceBag::from_bag_state(bag_state);
            assert_eq!(bag.remaining, 7 - i, "Bag should have {} pieces", 7 - i);
            assert!(!bag.contains(piece), "Bag should contain piece {}", piece);
        }
    }

    #[test]
    fn test_bag_shuffle_is_deterministic_for_seed() {
        let bag_state = TetrisPieceBagState::new(); // full bag
        let mut a = TetrisPieceBag::from_bag_state(bag_state);
        let mut b = TetrisPieceBag::from_bag_state(bag_state);

        let mut rng1 = TetrisGameRng::new(123);
        let mut rng2 = TetrisGameRng::new(123);

        a.shuffle(&mut rng1);
        b.shuffle(&mut rng2);

        assert_eq!(a, b);
    }

    #[test]
    fn test_bag_shuffle_preserves_multiset() {
        let bag_state = TetrisPieceBagState::new();
        let mut bag = TetrisPieceBag::from_bag_state(bag_state);

        let mut rng = TetrisGameRng::new(999);
        bag.shuffle(&mut rng);

        // Draw everything and ensure exactly one of each piece.
        let mut seen = [0u8; 7];
        let mut rng_draw = TetrisGameRng::new(0); // unused if bag.remaining>0
        for _ in 0..7 {
            let p = bag.rand_next(&mut rng_draw);
            let idx = p.index() as usize;
            assert!(idx < 7);
            seen[idx] += 1;
        }

        assert_eq!(seen, [1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_bag_shuffle_respects_remaining() {
        // Only {O,I,T} present (bits 0,1,4)
        let mut bag_state = TetrisPieceBagState::new();
        bag_state.remove(TetrisPiece::S_PIECE);
        bag_state.remove(TetrisPiece::Z_PIECE);
        bag_state.remove(TetrisPiece::L_PIECE);
        bag_state.remove(TetrisPiece::J_PIECE);

        let mut bag = TetrisPieceBag::from_bag_state(bag_state);
        assert_eq!(bag.remaining, 3);

        let mut rng = TetrisGameRng::new(42);
        bag.shuffle(&mut rng);

        let mut rng_draw = TetrisGameRng::new(0);
        let mut drawn = [0u8; 7];
        for _ in 0..3 {
            let p = bag.rand_next(&mut rng_draw);
            drawn[p.index() as usize] += 1;
        }

        assert_eq!(drawn[0], 1); // O
        assert_eq!(drawn[1], 1); // I
        assert_eq!(drawn[4], 1); // T
        assert_eq!(drawn.iter().sum::<u8>(), 3);
    }

    #[test]
    fn test_rand_next() {
        let mut tetris_rng = TetrisGameRng::new(42);
        let mut bag = TetrisPieceBag::new_rand(&mut tetris_rng);
        let mut pieces = std::collections::HashSet::new();

        for _ in 0..(constants::NUM_TETRIS_PIECES as usize) {
            let piece = bag.rand_next(&mut tetris_rng);
            assert!(!bag.contains(piece), "Piece {} should not be in bag", piece);
            pieces.insert(piece);
        }
        assert_eq!(
            pieces.len(),
            constants::NUM_TETRIS_PIECES as usize,
            "Should get 7 unique pieces"
        );

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
            let mut bag = TetrisPieceBag::new_rand(&mut tetris_rng);
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
    fn test_to_from_binary_slice() {
        let board = TetrisBoard::new();

        let binary_slice = board.to_binary_slice();
        assert_eq!(binary_slice.len(), constants::BOARD_SIZE);

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

    /// Test placement index bijection: index <-> placement
    #[test]
    fn test_to_from_piece_placement_index() {
        for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
            let index = placement.index() as u8;
            let placement2 = TetrisPiecePlacement::from_index(index);
            assert_eq!(
                placement, placement2,
                "Bijection failed for placement {:?}",
                placement
            );
        }
    }

    /// Test basic game flow: create, apply placement, piece changes
    #[test]
    fn test_tetris_game_basic() {
        let mut game = TetrisGame::new_with_seed(42);
        let initial_piece = game.current_piece;

        let placements = game.current_placements();
        assert!(!placements.is_empty(), "Game should have valid placements");

        let result = game.apply_placement(placements[0]);
        assert!(
            !Into::<bool>::into(result.is_lost),
            "Game should not be lost after first move"
        );
        assert_ne!(
            game.current_piece, initial_piece,
            "Piece should change after placement"
        );
        assert_eq!(game.piece_count, 1, "Piece count should be 1");

        // Play a few more moves to ensure game progresses
        for i in 1..10 {
            let placements = game.current_placements();
            assert!(!placements.is_empty());
            let result = game.apply_placement(placements[0]);
            if Into::<bool>::into(result.is_lost) {
                break;
            }
            assert_eq!(game.piece_count, i + 1);
        }
    }

    /// Test that placing pieces to exceed height triggers loss
    #[test]
    fn test_tetris_game_loss_from_placement() {
        let mut game = TetrisGame::new_with_seed(42);

        // Stack pieces until loss (always place in same column)
        for _ in 0..100 {
            let placements = game.current_placements();
            // Always place in column 0 to guarantee stacking
            let placement = placements
                .iter()
                .find(|p| p.orientation.column.0 == 0)
                .unwrap_or(&placements[0]);

            let result = game.apply_placement(*placement);
            if Into::<bool>::into(result.is_lost) {
                assert!(game.board.is_lost(), "Board should be in lost state");
                return;
            }
        }
        panic!("Game should have been lost after 100 pieces in same column");
    }

    #[test]
    fn test_tetris_game_correct_reset() {
        let mut game = TetrisGame::new_with_seed(42);
        let original_rng = game.rng;
        let original_bag = game.bag;
        let original_piece = game.current_piece;

        // make one placement
        let placements = game.current_placements();
        let choice_placement = placements[0];
        let _ = game.apply_placement(choice_placement);

        // After a single placement, the rng should stay the same
        let next_rng = game.rng;
        assert_eq!(
            original_rng, next_rng,
            "RNG should stay the same after single placement"
        );

        // Make enough successful placements to trigger a bag refill
        // We need to draw 6 more pieces (bag started with 6 remaining after init)
        // plus 1 more to trigger the refill
        let mut pieces_drawn = 1; // Already drew 1 above
        let target_pieces = 8; // Need to draw 8 total to trigger refill

        while pieces_drawn < target_pieces {
            let placements = game.current_placements();

            // Select a placement based on column distribution to avoid stacking in one place
            // Try to find a placement that places the piece in a relatively empty area
            let mut best_placement = placements[0];
            let mut min_max_height = u32::MAX;

            for placement in placements.iter() {
                // Simulate the placement and check the resulting height
                let mut test_board = game.board;
                let _ = test_board.apply_piece_placement(*placement);
                let height = test_board.height();

                if height < min_max_height {
                    min_max_height = height;
                    best_placement = *placement;
                }
            }

            let result = game.apply_placement(best_placement);
            if result.is_lost.into() {
                panic!(
                    "Game was lost after {} pieces, but test needs {} pieces to trigger bag refill",
                    pieces_drawn, target_pieces
                );
            }
            pieces_drawn += 1;
        }

        let next_rng = game.rng;
        assert_ne!(
            original_rng, next_rng,
            "RNG should have changed after drawing {} pieces (triggering bag refill)",
            pieces_drawn
        );

        game.reset(None);
        let new_rng = game.rng;
        assert_eq!(original_rng, new_rng, "RNG should be the same after reset");
        assert_eq!(
            game.current_piece, original_piece,
            "Current piece should be the same after reset"
        );
        assert_eq!(
            game.board,
            TetrisBoard::new(),
            "Board should be the same after reset"
        );
        assert_eq!(game.bag, original_bag, "Bag should be the same after reset");
        assert_eq!(
            game.lines_cleared, 0,
            "Lines cleared should be 0 after reset"
        );
        assert_eq!(game.piece_count, 0, "Piece count should be 0 after reset");
    }

    #[test]
    fn test_tetris_board_bytes_roundtrip() {
        const NUM_BOARDS: usize = 1000;
        let mut rng = rand::rng();

        for _ in 0..NUM_BOARDS {
            // Create a random tetris board by setting random u32 values for each column
            let limbs: [u32; constants::COLS] = std::array::from_fn(|_| rng.random());
            let board = TetrisBoard(limbs);

            // Convert to bytes using Into trait
            let bytes: [u8; 40] = board.into();

            // Convert back using From trait
            let reconstructed = TetrisBoard::from(bytes);

            // Verify roundtrip
            assert_eq!(
                board, reconstructed,
                "Board should roundtrip correctly through bytes"
            );
        }
    }
}
