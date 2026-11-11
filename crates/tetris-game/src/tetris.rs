use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};
use tetris_macros::{Getters, inline_conditioned};

use std::fmt::Display;
use std::hash::Hash;

use tetris_utils::{BitMask, const_assert, repeat_idx_unroll, swap_bit_chunks, transpose_bits};

pub use crate::config::constants;
use crate::config::{StandardTetris, TetrisGameConfig, constants::BackingColType};

/// Tetromino shapes and rotations shared by placement and display.
///
/// Shapes are encoded as column bitmasks, ordered from left to right. Occupied cells
/// are aligned to the top of the column storage, then shifted downward during placement
/// before merging into the board. This gives placement and display the same piece geometry.
///
/// Diagrams show the shapes from top to bottom, using `1` for occupied cells and `0`
/// for empty cells. Rotations advance clockwise in quarter turns; symmetric rotations
/// reuse the same geometry.
#[rustfmt::skip]
mod tetris_piece_data {
    use super::BackingColType;

    /// Aligns a four-row column pattern to the top of the backing column.
    const fn column(pattern: u8) -> BackingColType {
        const {
            assert!(BackingColType::BITS >= 4, "backing columns must hold four rows");
        }
        assert!(pattern < 0b1_0000, "piece columns must fit in four bits");

        (pattern as BackingColType) << (BackingColType::BITS - 4)
    }

    // 11
    // 11
    // 00
    // 00
    pub const O_PIECE_ROT_0: [BackingColType; 2] = [
        column(0b1100),
        column(0b1100),
    ];

    // 1111
    // 0000
    // 0000
    // 0000
    pub const I_PIECE_ROT_0: [BackingColType; 4] = [
        column(0b1000),
        column(0b1000),
        column(0b1000),
        column(0b1000),
    ];

    // 1
    // 1
    // 1
    // 1
    pub const I_PIECE_ROT_1: [BackingColType; 1] = [
        column(0b1111),
    ];

    // 011
    // 110
    // 000
    // 000
    pub const S_PIECE_ROT_0: [BackingColType; 3] = [
        column(0b0100),
        column(0b1100),
        column(0b1000),
    ];

    // 10
    // 11
    // 01
    // 00
    pub const S_PIECE_ROT_1: [BackingColType; 2] = [
        column(0b1100),
        column(0b0110),
    ];

    // 110
    // 011
    // 000
    // 000
    pub const Z_PIECE_ROT_0: [BackingColType; 3] = [
        column(0b1000),
        column(0b1100),
        column(0b0100),
    ];

    // 01
    // 11
    // 10
    // 00
    pub const Z_PIECE_ROT_1: [BackingColType; 2] = [
        column(0b0110),
        column(0b1100),
    ];

    // 111
    // 010
    // 000
    // 000
    pub const T_PIECE_ROT_0: [BackingColType; 3] = [
        column(0b1000),
        column(0b1100),
        column(0b1000),
    ];

    // 01
    // 11
    // 01
    // 00
    pub const T_PIECE_ROT_1: [BackingColType; 2] = [
        column(0b0100),
        column(0b1110),
    ];

    // 010
    // 111
    // 000
    // 000
    pub const T_PIECE_ROT_2: [BackingColType; 3] = [
        column(0b0100),
        column(0b1100),
        column(0b0100),
    ];

    // 10
    // 11
    // 10
    // 00
    pub const T_PIECE_ROT_3: [BackingColType; 2] = [
        column(0b1110),
        column(0b0100),
    ];

    // 001
    // 111
    // 000
    // 000
    pub const L_PIECE_ROT_0: [BackingColType; 3] = [
        column(0b0100),
        column(0b0100),
        column(0b1100),
    ];

    // 10
    // 10
    // 11
    // 00
    pub const L_PIECE_ROT_1: [BackingColType; 2] = [
        column(0b1110),
        column(0b0010),
    ];

    // 111
    // 100
    // 000
    // 000
    pub const L_PIECE_ROT_2: [BackingColType; 3] = [
        column(0b1100),
        column(0b1000),
        column(0b1000),
    ];

    // 11
    // 01
    // 01
    // 00
    pub const L_PIECE_ROT_3: [BackingColType; 2] = [
        column(0b1000),
        column(0b1110),
    ];

    // 100
    // 111
    // 000
    // 000
    pub const J_PIECE_ROT_0: [BackingColType; 3] = [
        column(0b1100),
        column(0b0100),
        column(0b0100),
    ];

    // 11
    // 10
    // 10
    // 00
    pub const J_PIECE_ROT_1: [BackingColType; 2] = [
        column(0b1110),
        column(0b1000),
    ];

    // 111
    // 001
    // 000
    // 000
    pub const J_PIECE_ROT_2: [BackingColType; 3] = [
        column(0b1000),
        column(0b1000),
        column(0b1100),
    ];

    // 01
    // 01
    // 11
    // 00
    pub const J_PIECE_ROT_3: [BackingColType; 2] = [
        column(0b0010),
        column(0b1110),
    ];
}

/// Clockwise rotation from a piece's spawn orientation, in quarter turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum Rotation {
    R0 = 0,
    R90 = 1,
    R180 = 2,
    R270 = 3,
}

const _: () = assert!(
    std::mem::variant_count::<Rotation>() == constants::NUM_ROTATIONS as usize,
    "Rotation variant count must match NUM_ROTATIONS"
);

impl Display for Rotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.index())
    }
}

impl Rotation {
    pub const ALL: [Self; constants::NUM_ROTATIONS as usize] =
        [Self::R0, Self::R90, Self::R180, Self::R270];

    pub const fn index(self) -> u8 {
        self as u8
    }
}

/// A byte-sized column index with an exclusive limit of `MAX`.
/// `new()` validates the width and index; `Default` stores zero without validating the width.
///
/// The stored index is read-only; use [`Self::inner`] to access it.
///
/// ```compile_fail,E0616
/// use tetris_game::Column;
/// let mut column = Column::<10>::new(0);
/// column.inner = 1;
/// ```
#[derive(Debug, Default, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd, Getters)]
#[repr(transparent)]
pub struct Column<const MAX: usize> {
    /// Returns the stored zero-based column index.
    #[getter(copy)]
    inner: u8,
}

impl<const MAX: usize> Display for Column<MAX> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Column({})", self.inner())
    }
}

impl<const MAX: usize> Column<MAX> {
    /// Creates a column within the configured board.
    /// Requires `1 <= MAX <= u8::MAX`, checked at compile time when instantiated.
    ///
    /// # Panics
    /// Panics if `index` is outside `0..MAX`.
    pub const fn new(index: u8) -> Self {
        let _ = const { const_assert!(MAX > 0, "columns must be greater than zero") };
        let _ = const { const_assert!(MAX <= u8::MAX as usize, "column limit must fit in u8") };
        assert!((index as usize) < MAX, "column out of bounds");
        Self { inner: index }
    }
}

/// The seven standard tetrominoes (O, I, S, Z, T, L, J).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum TetrisPiece {
    O = 0,
    I = 1,
    S = 2,
    Z = 3,
    T = 4,
    L = 5,
    J = 6,
}

const _: () = assert!(
    std::mem::variant_count::<TetrisPiece>() == constants::NUM_TETRIS_PIECES,
    "TetrisPiece variant count must match NUM_TETRIS_PIECES"
);

impl Distribution<TetrisPiece> for StandardUniform {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> TetrisPiece {
        TetrisPiece::from_packed_index(rng.random_range(0..constants::NUM_TETRIS_PIECES) as u8)
    }
}

impl TetrisPiece {
    // Piece indices occupy the low three bits of each packed bag slot.
    #[inline_conditioned(always)]
    const fn from_packed_index(index: u8) -> Self {
        let piece = match index {
            0 => Some(Self::O),
            1 => Some(Self::I),
            2 => Some(Self::S),
            3 => Some(Self::Z),
            4 => Some(Self::T),
            5 => Some(Self::L),
            6 => Some(Self::J),
            _ => None,
        };
        piece.expect("invalid packed piece index")
    }

    /// Returns the column bitmasks for this piece at the given rotation.
    ///
    /// Returns `(cols, width)` using the same geometry as placement and display.
    /// The first `width` columns contain the shape; the remaining columns are zero.
    /// Masks are top-aligned: the highest occupied cell uses the backing column's
    /// highest bit. Shift right by the backing bit width minus [`Self::height`]
    /// to obtain a floor-aligned footprint.
    ///
    /// ```
    /// use tetris_game::{Rotation, TetrisPiece, constants::BackingColType};
    ///
    /// const SHAPE: ([BackingColType; 4], u8) = TetrisPiece::I.shape_cols(Rotation::R0);
    /// assert_eq!(SHAPE.1, 4);
    /// assert_eq!(SHAPE.0, [1 << (BackingColType::BITS - 1); 4]);
    /// ```
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    pub const fn shape_cols(&self, rotation: Rotation) -> ([BackingColType; 4], u8) {
        let mut cols = [0; 4];
        let w = self.width(rotation) as usize;
        match (*self, rotation) {
            (Self::O, _) => cols[..w].copy_from_slice(&tetris_piece_data::O_PIECE_ROT_0),
            (Self::I, Rotation::R0 | Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::I_PIECE_ROT_0)
            }
            (Self::I, _) => cols[..w].copy_from_slice(&tetris_piece_data::I_PIECE_ROT_1),
            (Self::S, Rotation::R0 | Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::S_PIECE_ROT_0)
            }
            (Self::S, _) => cols[..w].copy_from_slice(&tetris_piece_data::S_PIECE_ROT_1),
            (Self::Z, Rotation::R0 | Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::Z_PIECE_ROT_0)
            }
            (Self::Z, _) => cols[..w].copy_from_slice(&tetris_piece_data::Z_PIECE_ROT_1),
            (Self::T, Rotation::R0) => cols[..w].copy_from_slice(&tetris_piece_data::T_PIECE_ROT_0),
            (Self::T, Rotation::R90) => {
                cols[..w].copy_from_slice(&tetris_piece_data::T_PIECE_ROT_1)
            }
            (Self::T, Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::T_PIECE_ROT_2)
            }
            (Self::T, Rotation::R270) => {
                cols[..w].copy_from_slice(&tetris_piece_data::T_PIECE_ROT_3)
            }
            (Self::L, Rotation::R0) => cols[..w].copy_from_slice(&tetris_piece_data::L_PIECE_ROT_0),
            (Self::L, Rotation::R90) => {
                cols[..w].copy_from_slice(&tetris_piece_data::L_PIECE_ROT_1)
            }
            (Self::L, Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::L_PIECE_ROT_2)
            }
            (Self::L, Rotation::R270) => {
                cols[..w].copy_from_slice(&tetris_piece_data::L_PIECE_ROT_3)
            }
            (Self::J, Rotation::R0) => cols[..w].copy_from_slice(&tetris_piece_data::J_PIECE_ROT_0),
            (Self::J, Rotation::R90) => {
                cols[..w].copy_from_slice(&tetris_piece_data::J_PIECE_ROT_1)
            }
            (Self::J, Rotation::R180) => {
                cols[..w].copy_from_slice(&tetris_piece_data::J_PIECE_ROT_2)
            }
            (Self::J, Rotation::R270) => {
                cols[..w].copy_from_slice(&tetris_piece_data::J_PIECE_ROT_3)
            }
        }
        (cols, w as u8)
    }

    /// All seven playable pieces, in index order.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn all() -> [Self; constants::NUM_TETRIS_PIECES] {
        [
            Self::O,
            Self::I,
            Self::S,
            Self::Z,
            Self::T,
            Self::L,
            Self::J,
        ]
    }

    /// Get the index of the piece.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn index(&self) -> u8 {
        *self as u8
    }

    /// Returns the piece's bit in a bag mask: `1 << self.index()`.
    ///
    /// Bits 0 through 6 represent O, I, S, Z, T, L, and J, respectively.
    ///
    /// ```
    /// use tetris_game::tetris::TetrisPiece;
    ///
    /// assert_eq!(TetrisPiece::O.mask(), 0b000_0001);
    /// assert_eq!(TetrisPiece::T.mask(), 0b001_0000);
    /// ```
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn mask(self) -> u8 {
        1 << self as u8
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
        // T/L/J have four rotations; O has one; I/S/Z have two.
        let four_rotations = matches!(self, Self::T | Self::L | Self::J) as u8;
        2 * (1 + four_rotations) - matches!(self, Self::O) as u8
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
        // Group pieces by their width pattern.
        // For odd rotation OR O-piece: returns 2 (or 1 for I)
        // For even rotation (non-O): returns 3 (or 4 for I)
        let is_i = matches!(self, Self::I) as u8;
        if ((rotation.index() & 1_u8) | matches!(self, Self::O) as u8) == 1 {
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
        // Inverse of the width pattern.
        // For even rotation OR O-piece: returns 2 (or 1 for I)
        // For odd rotation (non-O): returns 3 (or 4 for I)
        let b = matches!(self, Self::I) as u8;
        if (((rotation.index() & 1) == 0) as u8 | matches!(self, Self::O) as u8) == 1 {
            2_u8 - b
        } else {
            3_u8 + b
        }
    }
}

impl Display for TetrisPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::O => "O",
            Self::I => "I",
            Self::S => "S",
            Self::Z => "Z",
            Self::T => "T",
            Self::L => "L",
            Self::J => "J",
        })
    }
}

/// A rotation and column that position a piece on a configured board.
///
/// The table shows canonical orientations on the standard 10-column board.
/// Each cell lists the pieces that fit (repeated symmetric rotations omitted).
/// Within each cell, pieces are arranged as `OIS`, `ZTL`, then `J`.
/// [`index()`](Self::index) numbers cells across each rotation row, starting at zero.
/// An orientation does not contain a piece; piece fit is checked when indexing or applying a placement.
///
/// ```text
/// Columns →
/// Rotations ↓
///       0   1   2   3   4   5   6   7   8   9
///     +---+---+---+---+---+---+---+---+---+---+
///  0  |OIS|OIS|OIS|OIS|OIS|OIS|OIS|O S|O  |   |
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |   |   |
///     +---+---+---+---+---+---+---+---+---+---+
///  1  | IS| IS| IS| IS| IS| IS| IS| IS| IS| I |
///     |ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|ZTL|   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |J  |   |
///     +---+---+---+---+---+---+---+---+---+---+
///  2  |   |   |   |   |   |   |   |   |   |   |
///     | TL| TL| TL| TL| TL| TL| TL| TL|   |   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |   |   |
///     +---+---+---+---+---+---+---+---+---+---+
///  3  |   |   |   |   |   |   |   |   |   |   |
///     | TL| TL| TL| TL| TL| TL| TL| TL| TL|   |
///     |J  |J  |J  |J  |J  |J  |J  |J  |J  |   |
///     +---+---+---+---+---+---+---+---+---+---+
/// ```
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisPieceOrientation<const COLS: usize> {
    pub rotation: Rotation,
    pub column: Column<COLS>,
}

impl<const COLS: usize> Display for TetrisPieceOrientation<COLS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rot={} col={}", self.rotation, self.column.inner())
    }
}

impl<const COLS: usize> TetrisPieceOrientation<COLS> {
    /// Number of rotation/column pairs in the configured board.
    pub const TOTAL_NUM_ORIENTATIONS: usize = Rotation::ALL.len() * COLS;

    /// Returns the rotation-major orientation index in `0..Self::TOTAL_NUM_ORIENTATIONS`.
    /// All indices at this width must fit in a byte.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    pub const fn index(self) -> u8 {
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let _ = const {
            const_assert!(
                Self::TOTAL_NUM_ORIENTATIONS <= u8::MAX as usize + 1,
                "orientation indices must fit in u8"
            )
        };
        self.rotation.index() * COLS as u8 + self.column.inner()
    }

    /// Decodes a rotation-major orientation index. This does not check piece fit.
    /// All indices at this width must fit in a byte.
    ///
    /// # Panics
    /// Panics if `index` is outside `0..Self::TOTAL_NUM_ORIENTATIONS`, in every build profile.
    #[inline_conditioned(always)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    pub const fn from_index(index: u8) -> Self {
        let _ = const {
            const_assert!(
                Self::TOTAL_NUM_ORIENTATIONS <= u8::MAX as usize + 1,
                "orientation indices must fit in u8"
            )
        };
        assert!(
            (index as usize) < Self::TOTAL_NUM_ORIENTATIONS,
            "TetrisPieceOrientation::from_index: index out of bounds"
        );
        Self {
            rotation: match index / COLS as u8 {
                0 => Some(Rotation::R0),
                1 => Some(Rotation::R90),
                2 => Some(Rotation::R180),
                3 => Some(Rotation::R270),
                _ => None,
            }
            .expect("invalid rotation index"),
            column: Column::new(index % COLS as u8),
        }
    }

    // One mask per piece (O, I, S, Z, T, L, J) in piece-index order.
    //
    // Bit i corresponds to orientation index i (see `TetrisPieceOrientation::index`).
    const ORIENTATION_MASKS_BY_PIECE: [u64; constants::NUM_TETRIS_PIECES] = {
        let mut masks = [0u64; constants::NUM_TETRIS_PIECES];
        let pieces = TetrisPiece::all();
        let mut piece_index = 0;
        while piece_index < pieces.len() {
            let piece = pieces[piece_index];
            let mut rotation = 0;
            while rotation < piece.num_rotations() as usize {
                let width = piece.width(Rotation::ALL[rotation]) as usize;
                let valid_columns = COLS.saturating_sub(width - 1);
                let mut column = 0;
                while column < valid_columns {
                    masks[piece_index] |= 1u64 << (rotation * COLS + column);
                    column += 1;
                }
                rotation += 1;
            }
            piece_index += 1;
        }
        masks
    };

    /// Returns the mask of fitting canonical orientations for a piece.
    /// Requires between one and 64 orientation bits, as checked by `BitMask`.
    #[inline_conditioned(always)]
    pub const fn bitmask_from_piece(
        piece: TetrisPiece,
    ) -> BitMask<{ Self::TOTAL_NUM_ORIENTATIONS }> {
        BitMask::new_from_u64(Self::ORIENTATION_MASKS_BY_PIECE[piece.index() as usize])
    }
}

/// A piece and its rotation and column on a board of `COLS` columns.
/// Indexing or applying the placement checks that the piece fits the board width.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisPiecePlacement<const COLS: usize> {
    pub piece: TetrisPiece,
    pub orientation: TetrisPieceOrientation<COLS>,
}

impl<const COLS: usize> Display for TetrisPiecePlacement<COLS> {
    #[allow(
        clippy::needless_range_loop,
        reason = "Indices align board coordinates or parallel buffers; retain the existing traversal and independent reference calculation."
    )]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.piece, self.orientation)?;
        let (cols, w) = self.piece.shape_cols(self.orientation.rotation);
        let w = w as usize;
        let max_bit = cols[..w]
            .iter()
            .map(|c| BackingColType::BITS - c.leading_zeros())
            .max()
            .unwrap_or(0);
        let min_bit = cols[..w]
            .iter()
            .filter(|&&c| c != 0)
            .map(|c| c.trailing_zeros())
            .min()
            .unwrap_or(0);
        for row in (min_bit..max_bit).rev() {
            writeln!(f)?;
            for col in 0..w {
                write!(
                    f,
                    "{}",
                    if (cols[col] >> row) & 1 == 1 {
                        '#'
                    } else {
                        '.'
                    }
                )?;
            }
        }
        Ok(())
    }
}

/// Samples a piece uniformly, then one of its fitting canonical orientations.
/// Requires at least four columns so every rotation can fit.
/// Game-specific piece sets are enforced by `TetrisGame`.
impl<const COLS: usize> Distribution<TetrisPiecePlacement<COLS>> for StandardUniform {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    fn sample<R: RngExt + ?Sized>(&self, rng: &mut R) -> TetrisPiecePlacement<COLS> {
        let _ =
            const { const_assert!(COLS >= 4, "random placements require at least four columns") };
        let _ = const {
            const_assert!(
                TetrisPieceOrientation::<COLS>::TOTAL_NUM_ORIENTATIONS <= u64::BITS as usize,
                "orientation mask must fit in u64"
            )
        };
        let _ = TetrisPiecePlacement::<COLS>::NUM_PLACEMENTS;
        let piece: TetrisPiece = rng.sample(StandardUniform);
        let mask =
            TetrisPieceOrientation::<COLS>::ORIENTATION_MASKS_BY_PIECE[piece.index() as usize];
        let rand_idx =
            tetris_utils::choose_set_bit_u64(mask, rng).expect("piece has no fitting orientation");
        TetrisPiecePlacement {
            piece,
            orientation: TetrisPieceOrientation::from_index(rand_idx as u8),
        }
    }
}

impl<const COLS: usize> TetrisPiecePlacement<COLS> {
    // Piece starts followed by the end of the catalog; adjacent entries define each range.
    const PIECE_START_INDICES: [usize; constants::NUM_TETRIS_PIECES + 1] = {
        let pieces = TetrisPiece::all();
        let mut indices = [0; constants::NUM_TETRIS_PIECES + 1];

        let mut i = 0usize;
        while i < constants::NUM_TETRIS_PIECES {
            let piece = pieces[i];
            let mut r = 0u8;
            let mut end = indices[i];
            while r < piece.num_rotations() {
                end += COLS.saturating_sub(piece.width(Rotation::ALL[r as usize]) as usize - 1);
                r += 1;
            }
            indices[i + 1] = end;
            i += 1;
        }
        indices
    };

    /// Maximum number of canonical placements for any single piece.
    pub const MAX_PIECE_PLACEMENT_COUNT: usize = {
        let starts = Self::PIECE_START_INDICES;
        let mut max_count = 0;
        let mut i = 0;
        while i < constants::NUM_TETRIS_PIECES {
            let count = starts[i + 1] - starts[i];
            if count > max_count {
                max_count = count;
            }
            i += 1;
        }
        max_count
    };

    /// Number of canonical placements across all seven pieces on the configured board.
    pub const NUM_PLACEMENTS: usize = {
        assert!(COLS > 0, "board must have at least one column");
        let total = Self::PIECE_START_INDICES[constants::NUM_TETRIS_PIECES];
        assert!(total <= u8::MAX as usize, "placement count must fit in u8");
        total
    };

    /// Canonical placements in piece, rotation, then column order.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The complete placement catalog is compile-time bounded by u8; each column is below COLS."
    )]
    pub const ALL_PLACEMENTS: &'static [Self] = &{
        let _ = Self::NUM_PLACEMENTS;
        // Capacity matches the largest allowed catalog; the public slice excludes unused entries.
        let mut placements = [Self {
            piece: TetrisPiece::O,
            orientation: TetrisPieceOrientation {
                rotation: Rotation::R0,
                column: Column::new(0),
            },
        }; u8::MAX as usize];

        let all_pieces = TetrisPiece::all();

        let mut placement_id = 0;

        let mut i = 0;
        while i < all_pieces.len() {
            let piece = all_pieces[i];
            let mut r = 0;
            while r < piece.num_rotations() {
                let mut c = 0;
                while c < COLS.saturating_sub(piece.width(Rotation::ALL[r as usize]) as usize - 1) {
                    placements[placement_id] = Self {
                        piece,
                        orientation: TetrisPieceOrientation {
                            rotation: Rotation::ALL[r as usize],
                            column: Column::new(c as u8),
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
    }[..Self::NUM_PLACEMENTS];

    /// Get all placements for a given piece, in the same order as `ALL_PLACEMENTS`.
    #[inline_conditioned(always)]
    pub fn all_from_piece(piece: TetrisPiece) -> &'static [Self] {
        let piece_idx = piece.index() as usize;
        let start = Self::PIECE_START_INDICES[piece_idx];
        let end = Self::PIECE_START_INDICES[piece_idx + 1];
        &Self::ALL_PLACEMENTS[start..end]
    }

    /// Get the global placement index (into `ALL_PLACEMENTS`) for this placement.
    ///
    /// The global ordering is piece-major, then rotation-major, then column-major,
    /// matching [`Self::ALL_PLACEMENTS`]. Symmetric rotations share the same ID.
    ///
    /// # Panics
    /// Panics if the piece extends beyond the configured board width.
    #[inline_conditioned(always)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Piece widths are at most four; orientation and placement counts are compile-time bounded by their packed u8 representation, and sampled indices stay within those catalogs."
    )]
    pub const fn index(&self) -> u8 {
        // Validate the placement count at compile time so the index cannot overflow u8.
        let _ = Self::NUM_PLACEMENTS;
        let column = self.orientation.column.inner() as usize;
        assert!(
            column + self.piece.width(self.orientation.rotation) as usize <= COLS,
            "placement does not fit the configured board"
        );

        // Rotation counts are powers of two; mask off symmetric rotations.
        let rotation =
            (self.orientation.rotation.index() & (self.piece.num_rotations() - 1)) as usize;
        let even_columns = COLS.saturating_sub(self.piece.width(Rotation::R0) as usize - 1);
        let odd_columns = COLS.saturating_sub(self.piece.width(Rotation::R90) as usize - 1);

        // Each rotation pair contributes one run of each width.
        let rotation_offset =
            (rotation / 2) * (even_columns + odd_columns) + (rotation % 2) * even_columns;
        let piece_offset = Self::PIECE_START_INDICES[self.piece.index() as usize];
        (piece_offset + rotation_offset + column) as u8
    }

    /// Get a placement from its global placement index (into `ALL_PLACEMENTS`).
    ///
    /// # Panics
    /// Panics if `index` is outside `0..Self::NUM_PLACEMENTS`.
    #[inline_conditioned(always)]
    pub const fn from_index(index: u8) -> Self {
        Self::ALL_PLACEMENTS[index as usize]
    }
}

/// Invalid contents supplied to `TetrisPieceBag::from_slice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidBagPieces {
    TooMany(usize),
    Duplicate(TetrisPiece),
}

impl Display for InvalidBagPieces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooMany(count) => write!(f, "bag has {count} pieces; at most seven are allowed"),
            Self::Duplicate(piece) => write!(f, "bag contains duplicate piece {piece}"),
        }
    }
}

impl std::error::Error for InvalidBagPieces {}

/// A finite draw sequence of up to seven distinct Tetris pieces.
///
/// The sequence is packed into a single `u64`, using four bits per piece.
/// Each slot stores `0b1000 | piece.index()`: the low three bits identify the
/// piece, and the high bit marks the slot as occupied. This distinguishes O
/// (index zero, stored as `0b1000`) from an unused slot (`0b0000`).
///
/// The next piece is in the lowest four bits. Drawing reads that piece and shifts
/// the word right by four bits, moving the next slot into place. Remaining pieces
/// stay packed together at the low end; all unused bits are zero. A zero word is
/// an empty bag, and the highest occupied slot determines the remaining count.
/// `Default` contains all seven pieces in piece-index order; shuffle and refill are explicit.
///
/// ```text
/// Layout:
/// +--------+--------+--------+--------+--------+--------+--------+--------+
/// | 63..28 | 27..24 | 23..20 | 19..16 | 15..12 | 11..8  |  7..4  |  3..0  |
/// +--------+--------+--------+--------+--------+--------+--------+--------+
/// | unused | slot6  | slot5  | slot4  | slot3  | slot2  | slot1  | slot0  |
/// +--------+--------+--------+--------+--------+--------+--------+--------+
///                                                             next draw
///
/// Slot:
/// +----------+-------------+
/// |    3     |    2..0     |
/// +----------+-------------+
/// | occupied | piece index |
/// +----------+-------------+
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct TetrisPieceBag(u64);

impl Default for TetrisPieceBag {
    fn default() -> Self {
        Self::from_bag_state(TetrisPieceBagState::full())
    }
}

impl Display for TetrisPieceBag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        let mut bag = *self;
        let mut separator = "";
        while let Some(piece) = bag.draw() {
            f.write_str(separator)?;
            write!(f, "\"{piece}\"")?;
            separator = ", ";
        }
        f.write_str("]")
    }
}

impl TetrisPieceBag {
    const NUM_SHUFFLE_STEPS: usize = constants::NUM_TETRIS_PIECES - 1;
    const NUM_BAG_COUNTS: usize = constants::NUM_TETRIS_PIECES + 1;
    const NUM_BYTE_VALUES: usize = u8::MAX as usize + 1;

    /// Fisher–Yates targets indexed by remaining count, swap step, and random byte.
    /// This 12 KiB table is built at compile time using the original shuffle formula.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Steps and choice counts are below eight, so every Fisher-Yates target fits u8."
    )]
    const SHUFFLE_TARGETS: [[[u8; Self::NUM_BYTE_VALUES]; Self::NUM_SHUFFLE_STEPS];
        Self::NUM_BAG_COUNTS] = {
        let mut targets =
            [[[0; Self::NUM_BYTE_VALUES]; Self::NUM_SHUFFLE_STEPS]; Self::NUM_BAG_COUNTS];
        let mut count = 0usize;
        while count < targets.len() {
            let mut step = 0usize;
            while step < Self::NUM_SHUFFLE_STEPS {
                let choices = count.saturating_sub(step);
                let mut byte = 0usize;
                while byte < Self::NUM_BYTE_VALUES {
                    targets[count][step][byte] = (step + ((byte * choices) >> u8::BITS)) as u8;
                    byte += 1;
                }
                step += 1;
            }
            count += 1;
        }
        targets
    };

    /// Returns a mask identifying the occupied slot containing `piece`, or zero if absent.
    ///
    /// A match in slot `i` sets bit `4 * i`, the slot's lowest bit. `contains()`
    /// checks for a nonzero mask; `remove()` uses its trailing zeros as the bit offset.
    ///
    /// The piece index is repeated across all seven slots and XORed with the packed
    /// bag. Shifts combine each slot's three index bits into its lowest bit, which
    /// stays zero only when the indices match. Inverting this result and masking
    /// with the occupancy bits selects matches without treating unused slots as O.
    #[inline_conditioned(always)]
    fn matching_slots(&self, piece: TetrisPiece) -> u64 {
        const SLOT_LSB_MASK: u64 = 0x0000_0000_0111_1111; // Bits 0, 4, ..., 24.
        let broadcast = (piece.index() as u64).wrapping_mul(SLOT_LSB_MASK);
        let different = self.0 ^ broadcast;
        let nonzero = different | (different >> 1) | (different >> 2);
        let used = (self.0 >> 3) & SLOT_LSB_MASK;
        !nonzero & used
    }

    /// Returns whether the piece remains in the current bag.
    #[inline_conditioned(always)]
    #[must_use]
    pub fn contains(&self, piece: TetrisPiece) -> bool {
        self.matching_slots(piece) != 0
    }

    /// Removes a piece from the current bag, preserving the other pieces' draw order.
    ///
    /// Returns `true` if the piece was removed, or `false` if it was absent.
    /// Removing the last piece leaves the bag empty; subsequent draws return `None`.
    ///
    /// ```
    /// use tetris_game::{TetrisPiece, TetrisPieceBag};
    ///
    /// let mut bag = TetrisPieceBag::default();
    /// assert!(bag.remove(TetrisPiece::T));
    /// assert!(!bag.contains(TetrisPiece::T));
    /// assert_eq!(bag.count(), 6);
    /// assert!(!bag.remove(TetrisPiece::T));
    /// ```
    #[inline_conditioned(always)]
    pub fn remove(&mut self, piece: TetrisPiece) -> bool {
        let matches = self.matching_slots(piece);
        if matches == 0 {
            return false;
        }
        let shift = matches.trailing_zeros();
        let lower_mask = (1u64 << shift) - 1;
        // Keep lower slots and close the four-bit gap above the removed piece.
        self.0 = (self.0 & lower_mask) | ((self.0 >> 4) & !lower_mask);
        true
    }

    /// Replaces the current contents with the given set in piece-index order.
    /// Accepts an empty set and returns this bag for chaining.
    /// Call `shuffle()` to randomize the draw order.
    #[inline_conditioned(always)]
    pub const fn refill(&mut self, pieces: TetrisPieceBagState) -> &mut Self {
        *self = Self::from_bag_state(pieces);
        self
    }

    /// Shuffles the pieces in place and returns this bag for chaining.
    /// Consumes one RNG word, even when the bag contains zero or one piece.
    ///
    /// ```
    /// use tetris_game::{TetrisGameRng, TetrisPieceBag, TetrisPieceBagState};
    /// let mut rng = TetrisGameRng::new(42);
    /// let mut bag = TetrisPieceBag::default();
    /// let piece = bag.refill(TetrisPieceBagState::full()).shuffle(&mut rng).draw();
    /// assert!(piece.is_some());
    /// assert_eq!(bag.count(), 6);
    /// ```
    #[inline_conditioned(always)]
    #[allow(
        clippy::out_of_bounds_indexing,
        reason = "The constant-unrolled body is guarded by the actual array length; Clippy also visits unreachable expanded indices."
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Bag storage holds at most sixteen nibbles; shuffle steps are below seven, pop checks 0..=6, and draw intentionally extracts the low piece bits."
    )]
    pub const fn shuffle(&mut self, rng: &mut TetrisGameRng) -> &mut Self {
        // Valid bags contain at most seven pieces; the mask keeps indexing branchless.
        let rem = (self.count() & 0b111) as usize;
        let random = rng.next_u64().to_le_bytes();
        repeat_idx_unroll!(Self::NUM_SHUFFLE_STEPS, K, {
            let target = Self::SHUFFLE_TARGETS[rem][K][random[K] as usize];
            swap_bit_chunks::<4>(&mut self.0, K as u32, target as u32);
        });
        self
    }

    /// Returns the number of pieces remaining.
    ///
    /// ```
    /// use tetris_game::TetrisPieceBag;
    ///
    /// let mut bag = TetrisPieceBag::default();
    /// assert_eq!(bag.count(), 7);
    /// assert!(bag.draw().is_some());
    /// assert_eq!(bag.count(), 6);
    /// ```
    #[inline_conditioned(always)]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Bag storage holds at most sixteen nibbles; shuffle steps are below seven, pop checks 0..=6, and draw intentionally extracts the low piece bits."
    )]
    pub const fn count(&self) -> u8 {
        // Each occupied nibble has its high bit set; leading zeros count empty high slots.
        (u64::BITS / 4 - self.0.leading_zeros() / 4) as u8
    }

    /// Removes and returns the next piece, or `None` when the bag is empty.
    /// Does not shuffle or refill the bag.
    #[inline_conditioned(always)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Bag storage holds at most sixteen nibbles; shuffle steps are below seven, pop checks 0..=6, and draw intentionally extracts the low piece bits."
    )]
    pub fn draw(&mut self) -> Option<TetrisPiece> {
        if self.0 == 0 {
            return None;
        }
        let piece = TetrisPiece::from_packed_index(self.0 as u8 & 0b111);
        self.0 >>= 4;
        Some(piece)
    }

    /// Creates a bag in the supplied draw order. An empty slice creates an empty bag.
    ///
    /// # Errors
    /// Returns an error for duplicates or more than seven pieces.
    ///
    /// ```
    /// use tetris_game::{InvalidBagPieces, TetrisPiece, TetrisPieceBag};
    /// const BAG: Result<TetrisPieceBag, InvalidBagPieces> =
    ///     TetrisPieceBag::from_slice(&[TetrisPiece::T, TetrisPiece::I]);
    /// let mut bag = BAG?;
    /// assert_eq!(bag.draw(), Some(TetrisPiece::T));
    /// assert_eq!(bag.draw(), Some(TetrisPiece::I));
    /// assert_eq!(bag.draw(), None);
    /// # Ok::<(), tetris_game::InvalidBagPieces>(())
    /// ```
    #[allow(
        unused_assignments,
        reason = "The unrolled duplicate check updates the mask uniformly, including its final iteration."
    )]
    pub const fn from_slice(pieces: &[TetrisPiece]) -> Result<Self, InvalidBagPieces> {
        if pieces.len() > constants::NUM_TETRIS_PIECES {
            return Err(InvalidBagPieces::TooMany(pieces.len()));
        }
        let mut stream = 0;
        let mut seen = 0;
        repeat_idx_unroll!(constants::NUM_TETRIS_PIECES, K, {
            if K < pieces.len() {
                let piece = pieces[K];
                let mask = piece.mask();
                if seen & mask != 0 {
                    return Err(InvalidBagPieces::Duplicate(piece));
                }
                seen |= mask;
                stream |= (0b1000 | piece.index() as u64) << (K * 4);
            }
        });
        Ok(Self(stream))
    }

    /// Creates a bag in piece-index order from the given set, which may be empty.
    /// Call `shuffle()` to randomize the draw order.
    #[inline_conditioned(always)]
    #[must_use]
    pub const fn from_bag_state(pieces: TetrisPieceBagState) -> Self {
        const NUM_BAG_STATES: usize = 1 << constants::NUM_TETRIS_PIECES;

        // Packed streams in piece-index order, indexed by the remaining-piece mask.
        // Built at compile time; the complete table uses 1 KiB of read-only data.
        const BAG_STREAMS: [u64; NUM_BAG_STATES] = {
            let mut streams = [0u64; NUM_BAG_STATES];
            let mut mask = 0usize;
            while mask < streams.len() {
                let mut index = 0usize;
                let mut shift = 0u32;
                while index < constants::NUM_TETRIS_PIECES {
                    if mask & (1 << index) != 0 {
                        streams[mask] |= (0b1000 | index as u64) << shift;
                        shift += 4;
                    }
                    index += 1;
                }
                mask += 1;
            }
            streams
        };

        let mask = u8::from(pieces) & TetrisPieceBagState::FULL_MASK;
        Self(BAG_STREAMS[mask as usize])
    }
}

/// The pieces remaining in a bag, represented as an unordered membership mask.
///
/// A standard seven-piece bag limits long droughts of any piece.
/// It starts with 7 pieces. As a Tetris game is played, pieces are
/// randomly removed from the bag one by one. Once the bag is empty,
/// `TetrisGame` refills it with the configured piece set. This process continually repeats.
///
/// This type records the pieces remaining in a bag as a single byte,
/// with one bit set for each remaining piece. It stores no draw order or RNG state;
/// `pop()` removes the lowest-index piece, and `fill()` restores all seven pieces.
/// `Default` contains all seven pieces.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TetrisPieceBagState(u8);

impl std::fmt::Display for TetrisPieceBagState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BagState(")?;
        let mut separator = "";
        for piece in self.iter_pieces() {
            f.write_str(separator)?;
            write!(f, "{piece}")?;
            separator = ", ";
        }
        f.write_str(")")
    }
}

/// A raw bag byte has its reserved bit set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidBagMask;

impl Display for InvalidBagMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bag mask has reserved bit 7 set")
    }
}

impl std::error::Error for InvalidBagMask {}

impl const TryFrom<u8> for TetrisPieceBagState {
    type Error = InvalidBagMask;

    fn try_from(bits: u8) -> Result<Self, Self::Error> {
        if bits & !Self::FULL_MASK == 0 {
            Ok(Self(bits))
        } else {
            Err(InvalidBagMask)
        }
    }
}

impl const From<TetrisPieceBagState> for u8 {
    fn from(state: TetrisPieceBagState) -> Self {
        state.0
    }
}

impl Default for TetrisPieceBagState {
    fn default() -> Self {
        Self::full()
    }
}

impl TetrisPieceBagState {
    /// Mask of all seven standard pieces; bit 7 is unused.
    pub const FULL_MASK: u8 = 0b0111_1111;

    /// Returns a bag state containing all seven standard Tetris pieces.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::TetrisPieceBagState;
    ///
    /// let bag = TetrisPieceBagState::full();
    /// assert_eq!(bag.count(), 7);
    /// ```
    pub const fn full() -> Self {
        Self(Self::FULL_MASK)
    }

    /// Returns the number of pieces remaining in the bag.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::full();
    /// assert_eq!(bag.count(), 7);
    ///
    /// bag.remove(TetrisPiece::O);
    /// assert_eq!(bag.count(), 6);
    /// ```
    #[inline_conditioned(always)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Bag storage holds at most sixteen nibbles; shuffle steps are below seven, pop checks 0..=6, and draw intentionally extracts the low piece bits."
    )]
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
    /// let mut bag = TetrisPieceBagState::full();
    /// assert!(!bag.empty());
    /// for _ in 0..7 {
    ///     let _ = bag.pop();
    /// }
    /// assert!(bag.empty());
    /// ```
    #[inline_conditioned(always)]
    pub const fn empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns whether the bag contains the given piece.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::full();
    /// assert!(bag.contains(TetrisPiece::I));
    ///
    /// bag.remove(TetrisPiece::I);
    /// assert!(!bag.contains(TetrisPiece::I));
    /// ```
    #[inline_conditioned(always)]
    pub const fn contains(&self, piece: TetrisPiece) -> bool {
        self.0 & piece.mask() != 0
    }

    /// Restores all seven standard pieces, including any previously removed pieces.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::full();
    /// bag.remove(TetrisPiece::O);
    /// bag.remove(TetrisPiece::I);
    /// assert_eq!(bag.count(), 5);
    ///
    /// bag.fill();
    /// assert_eq!(bag.count(), 7);
    /// assert!(bag.contains(TetrisPiece::O));
    /// assert!(bag.contains(TetrisPiece::I));
    /// ```
    #[inline_conditioned(always)]
    pub const fn fill(&mut self) {
        self.0 = Self::FULL_MASK;
    }

    /// Removes the given piece. Does nothing if the piece is absent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::full();
    /// assert_eq!(bag.count(), 7);
    ///
    /// bag.remove(TetrisPiece::T);
    /// assert_eq!(bag.count(), 6);
    /// assert!(!bag.contains(TetrisPiece::T));
    ///
    /// // Removing again has no effect
    /// bag.remove(TetrisPiece::T);
    /// assert_eq!(bag.count(), 6);
    /// ```
    pub const fn remove(&mut self, piece: TetrisPiece) {
        self.0 &= !piece.mask();
    }

    /// Removes and returns the lowest-index piece from the bag.
    ///
    /// Pieces are removed in O, I, S, Z, T, L, J order.
    /// Returns `None` without changing an empty bag.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// let mut bag = TetrisPieceBagState::full();
    ///
    /// // Pop pieces in order: O, I, S, Z, T, L, J
    /// assert_eq!(bag.pop(), Some(TetrisPiece::O));
    /// assert_eq!(bag.count(), 6);
    ///
    /// assert_eq!(bag.pop(), Some(TetrisPiece::I));
    /// assert_eq!(bag.count(), 5);
    ///
    /// // Remove all remaining pieces
    /// for _ in 0..5 {
    ///     let _ = bag.pop();
    /// }
    ///
    /// // Empty bag returns None
    /// assert_eq!(bag.pop(), None);
    /// assert_eq!(bag.count(), 0);
    /// ```
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Bag storage holds at most sixteen nibbles; shuffle steps are below seven, pop checks 0..=6, and draw intentionally extracts the low piece bits."
    )]
    pub const fn pop(&mut self) -> Option<TetrisPiece> {
        // Setting the unused bit makes an empty mask resolve to index 7.
        let index = (self.0 | !Self::FULL_MASK).trailing_zeros();
        self.0 &= !(1u8 << index);
        match index {
            0..=6 => Some(TetrisPiece::from_packed_index(index as u8)),
            _ => None,
        }
    }

    /// Iterates over a snapshot of the present pieces in O, I, S, Z, T, L, J order.
    /// Later changes to the original state do not affect the iterator.
    ///
    /// ```
    /// use tetris_game::{TetrisPiece, TetrisPieceBagState};
    ///
    /// let mut state = TetrisPieceBagState::from_pieces([TetrisPiece::I, TetrisPiece::T]);
    /// let pieces = state.iter_pieces();
    /// state.remove(TetrisPiece::I);
    /// assert_eq!(pieces.collect::<Vec<_>>(), vec![TetrisPiece::I, TetrisPiece::T]);
    /// assert!(!state.contains(TetrisPiece::I));
    /// ```
    #[inline]
    pub fn iter_pieces(self) -> impl Iterator<Item = TetrisPiece> {
        let mut remaining = self;
        std::iter::from_fn(move || remaining.pop())
    }

    /// Creates a bag state from distinct pieces. Input order does not affect the state.
    /// Arrays with more than seven elements are rejected at compile time.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{TetrisPiece, TetrisPieceBagState, constants};
    ///
    /// let pieces = [TetrisPiece::O; constants::NUM_TETRIS_PIECES + 1];
    /// let _ = TetrisPieceBagState::from_pieces(pieces);
    /// ```
    ///
    /// # Panics
    /// Panics if any piece appears more than once.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisPieceBagState, TetrisPiece};
    ///
    /// const I_ONLY: TetrisPieceBagState = TetrisPieceBagState::from_pieces([TetrisPiece::I]);
    /// assert_eq!(I_ONLY.count(), 1);
    /// assert!(I_ONLY.contains(TetrisPiece::I));
    ///
    /// const TWO: TetrisPieceBagState = TetrisPieceBagState::from_pieces([TetrisPiece::S, TetrisPiece::Z]);
    /// assert_eq!(TWO.count(), 2);
    /// ```
    pub const fn from_pieces<const N: usize>(pieces: [TetrisPiece; N]) -> Self {
        let _ = const {
            const_assert!(
                N <= constants::NUM_TETRIS_PIECES,
                "bag state cannot contain more than seven pieces"
            )
        };
        let mut mask = 0u8;
        repeat_idx_unroll!(N, K, {
            assert!(
                mask & pieces[K].mask() == 0,
                "duplicate piece in from_pieces"
            );
            mask |= pieces[K].mask();
        });
        Self(mask)
    }
}

/// A Tetris board represented as an array of columns.
///
/// `ROWS` is the playable height; each column also stores overflow bits up to
/// `BackingColType::BITS`. Raw limb access does not validate playable geometry.
/// Placement and conversion methods enforce the bounds needed by their operations.
///
/// # Memory Layout
///
/// The board uses a **column-major** representation optimized for piece placement:
/// - Each column is stored as an unsigned integer.
/// - Bit 0 represents the bottom row (row 0)
/// - Higher bits represent higher rows (bit 1 = row 1, bit 2 = row 2, etc.)
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct TetrisBoard<const COLS: usize, const ROWS: usize>([BackingColType; COLS]);

impl<const COLS: usize, const ROWS: usize> const Default for TetrisBoard<COLS, ROWS> {
    fn default() -> Self {
        Self([0; COLS])
    }
}

impl<const COLS: usize, const ROWS: usize> TetrisBoard<COLS, ROWS> {
    /// Gets the value of a single bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - A backing row in `0..BackingColType::BITS`, including overflow rows
    ///
    /// # Returns
    ///
    /// * `bool` - The bit value (false or true) at the specified position
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// assert_eq!(board.get_bit(0, 0), false); // Empty cell at bottom-left
    /// ```
    #[inline_conditioned(always)]
    pub const fn get_bit(&self, col: usize, row: usize) -> bool {
        assert!(
            row < BackingColType::BITS as usize,
            "row is outside board storage"
        );
        ((self.0[col] >> row) & 1) != 0
    }

    /// Sets a bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - A backing row in `0..BackingColType::BITS`, including overflow rows
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0); // Set bottom-left cell
    /// assert_eq!(board.get_bit(0, 0), true);
    /// ```
    #[inline_conditioned(always)]
    pub fn set_bit(&mut self, col: usize, row: usize) {
        assert!(
            row < BackingColType::BITS as usize,
            "row is outside board storage"
        );
        self.0[col] |= 1 << row;
    }

    /// Clears a bit at the specified column and row position.
    ///
    /// # Arguments
    ///
    /// * `col` - The column index (0 to COLS-1)
    /// * `row` - A backing row in `0..BackingColType::BITS`, including overflow rows
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0);
    /// board.clear_bit(0, 0);
    /// assert_eq!(board.get_bit(0, 0), false);
    /// ```
    #[inline_conditioned(always)]
    pub fn clear_bit(&mut self, col: usize, row: usize) {
        assert!(
            row < BackingColType::BITS as usize,
            "row is outside board storage"
        );
        self.0[col] &= !(1 << row);
    }

    /// Sets randomly selected cells within the configured playable area.
    ///
    /// Cells may be selected more than once, so fewer than `num_bits` new cells may be set.
    ///
    /// # Arguments
    ///
    /// * `num_bits` - The number of cells to select
    /// * `rng` - Random number generator implementing the `RngExt` trait
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// use rand::rng;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// let mut rng = rng();
    /// board.set_random_bits(10, &mut rng);
    /// ```
    pub fn set_random_bits<R: RngExt + ?Sized>(&mut self, num_bits: usize, rng: &mut R) {
        let _ = const { const_assert!(ROWS > 0, "playable rows must be greater than zero") };
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let _ = const {
            const_assert!(
                ROWS <= constants::ACTUAL_ROWS,
                "rows must fit in board storage"
            )
        };
        for _ in 0..num_bits {
            let col = rng.random_range(0..COLS);
            let row = rng.random_range(0..ROWS);
            self.set_bit(col, row);
        }
    }
}

impl<const COLS: usize, const ROWS: usize> TetrisBoard<COLS, ROWS> {
    /// Number of playable cells on the board.
    pub const BOARD_SIZE: usize = ROWS * COLS;

    /// Borrows the board's column limbs, including overflow bits.
    #[must_use]
    pub const fn as_limbs(&self) -> &[BackingColType; COLS] {
        &self.0
    }

    /// Creates a board from raw column limbs.
    #[must_use]
    pub const fn from_limbs(limbs: [BackingColType; COLS]) -> Self {
        Self(limbs)
    }

    /// Converts the board to a flat cell array where each byte is 0 or 1.
    /// Includes only `ROWS` playable rows; overflow bits are omitted.
    ///
    /// # Arguments
    ///
    /// * `order` - Layout order for the output array:
    ///   - `Major::Row`: index `row * COLS + col`, with bottom row first.
    ///     Compatible with tensor reshape to `[ROWS, COLS]`.
    ///   - `Major::Col`: index `col * ROWS + row`, with column zero first.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::{tetris::{TetrisBoard, Major}, StandardTetris, TetrisGameConfig};
    /// let col = StandardTetris::COLS.min(4).saturating_sub(1);
    /// let row = StandardTetris::ROWS.min(6).saturating_sub(1);
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(col, row);
    /// let cells = board.to_cell_array(Major::Row);
    /// assert_eq!(cells[row * StandardTetris::COLS + col], 1);
    /// let cells = board.to_cell_array(Major::Col);
    /// assert_eq!(cells[col * StandardTetris::ROWS + row], 1);
    /// ```
    #[inline_conditioned(always)]
    pub const fn to_cell_array(&self, order: Major) -> [u8; Self::BOARD_SIZE] {
        let _ = const {
            const_assert!(
                ROWS <= constants::ACTUAL_ROWS,
                "rows must fit in board storage"
            )
        };
        let mut result = [0u8; Self::BOARD_SIZE];
        let mut i = 0;
        while i < Self::BOARD_SIZE {
            let (col, row) = match order {
                Major::Row => (i % COLS, i / COLS),
                Major::Col => (i / ROWS, i % ROWS),
            };
            result[i] = ((self.0[col] >> row) & 1) as u8;
            i += 1;
        }
        result
    }

    /// Creates a board from a flat cell array where each byte is 0 or 1.
    /// Overflow rows start empty.
    ///
    /// # Arguments
    ///
    /// * `cells` - Fixed-size array of u8 (0 or 1) representing cell states
    /// * `order` - Layout order of the input array (see [`Self::to_cell_array`])
    ///
    /// # Panics
    /// Panics if any cell is not `0` or `1`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisBoard, Major};
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0);
    /// let cells = board.to_cell_array(Major::Row);
    /// let board2: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::from_cell_array(cells, Major::Row);
    /// assert_eq!(board, board2);
    /// ```
    #[inline_conditioned(always)]
    pub const fn from_cell_array(cells: [u8; Self::BOARD_SIZE], order: Major) -> Self {
        let _ = const {
            const_assert!(
                ROWS <= constants::ACTUAL_ROWS,
                "rows must fit in board storage"
            )
        };
        let mut board = Self::EMPTY_BOARD;
        let mut cell_bits = 0u8;
        let mut i = 0;
        while i < Self::BOARD_SIZE {
            let (col, row) = match order {
                Major::Row => (i % COLS, i / COLS),
                Major::Col => (i / ROWS, i % ROWS),
            };
            cell_bits |= cells[i];
            board.0[col] |= (cells[i] as BackingColType) << row;
            i += 1;
        }
        assert!(cell_bits <= 1, "cell values must be 0 or 1");
        board
    }

    /// Serializes all column limbs, including overflow rows, as native-endian bytes.
    ///
    /// `Major::Col` stores columns consecutively; `Major::Row` transposes the
    /// packed bits into row order.
    /// This native-endian representation is not a portable cross-endian file format.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisBoard, Major};
    /// let board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// let bytes = board.to_bytes(Major::Col);
    /// assert_eq!(bytes.len(), std::mem::size_of::<TetrisBoard<10, 20>>());
    /// assert_eq!(<TetrisBoard<10, 20>>::from_bytes(bytes, Major::Col), board);
    /// ```
    #[inline(always)]
    pub const fn to_bytes(&self, order: Major) -> [u8; size_of::<Self>()] {
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let mut bytes: [u8; size_of::<Self>()] = unsafe { std::mem::transmute_copy(&self.0) };
        match order {
            Major::Col => bytes,
            Major::Row => {
                transpose_bits::<{ COLS }, { constants::ACTUAL_ROWS }, { size_of::<Self>() }>(
                    &mut bytes,
                );
                bytes
            }
        }
    }

    /// Deserializes a board from packed bytes (limbs as native-endian bytes).
    ///
    /// Inverse of [`Self::to_bytes`]. The `order` parameter must match what was used
    /// to produce the bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::{TetrisBoard, Major};
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(3, 5);
    /// let bytes = board.to_bytes(Major::Col);
    /// assert_eq!(<TetrisBoard<10, 20>>::from_bytes(bytes, Major::Col), board);
    /// ```
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; size_of::<Self>()], order: Major) -> Self {
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        match order {
            Major::Col => unsafe {
                Self(std::mem::transmute_copy::<
                    [u8; size_of::<Self>()],
                    [BackingColType; COLS],
                >(&bytes))
            },
            Major::Row => {
                let mut bytes = bytes;
                transpose_bits::<{ constants::ACTUAL_ROWS }, { COLS }, { size_of::<Self>() }>(
                    &mut bytes,
                );
                unsafe {
                    Self(std::mem::transmute_copy::<
                        [u8; size_of::<Self>()],
                        [BackingColType; COLS],
                    >(&bytes))
                }
            }
        }
    }
}

impl<const COLS: usize, const ROWS: usize> TetrisBoard<COLS, ROWS> {
    /// RFC 4648 base32 alphabet used for board ID encoding.
    const BASE32_ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    /// Maximum number of base32 characters in a board ID.
    const BOARD_ID_SIZE: usize = Self::BOARD_SIZE.div_ceil(5);

    /// Encodes the board as a compact base32 board ID string.
    ///
    /// The encoding packs the playable cells into a bit stream in row-major order
    /// (row 0 / bottom first, col 0 first),
    /// groups into 5-bit chunks (MSB first within each chunk), maps each
    /// chunk to an RFC 4648 base32 character, and truncates trailing 'A's.
    /// A partial final chunk is padded with zero bits.
    ///
    /// Overflow bits are omitted. A board with no occupied playable cells encodes to `""`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// // Round-trip a board whose cells are within the playable area.
    /// let board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// let id = board.to_board_id();
    /// assert_eq!(<TetrisBoard<10, 20>>::from_board_id(&id), Some(board));
    /// ```
    #[must_use]
    #[allow(
        clippy::needless_range_loop,
        reason = "Indices align board coordinates or parallel buffers; retain the existing traversal and independent reference calculation."
    )]
    pub fn to_board_id(&self) -> String
    where
        [(); Self::BOARD_ID_SIZE]:,
    {
        let _ = const {
            const_assert!(
                ROWS <= constants::ACTUAL_ROWS,
                "rows must fit in board storage"
            )
        };
        let mut chars = [b'A'; Self::BOARD_ID_SIZE];

        for chunk_idx in 0..Self::BOARD_ID_SIZE {
            let mut value: u8 = 0;
            for bit in 0..5 {
                let bit_idx = chunk_idx * 5 + bit;
                let row = bit_idx / COLS;
                let col = bit_idx % COLS;
                if bit_idx < Self::BOARD_SIZE && self.get_bit(col, row) {
                    value |= 1 << (4 - bit);
                }
            }
            chars[chunk_idx] = Self::BASE32_ALPHABET[value as usize];
        }

        let len = chars.iter().rposition(|&c| c != b'A').map_or(0, |i| i + 1);
        unsafe { String::from_utf8_unchecked(chars[..len].to_vec()) }
    }

    /// Decodes a board ID string back into a `TetrisBoard`.
    ///
    /// Returns `None` for invalid base32 characters, nonzero padding bits, or an ID
    /// longer than `ceil(BOARD_SIZE / 5)` characters (40 for the standard board).
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::from_board_id("PXDHAE").expect("documented example must succeed");
    /// assert_eq!(board.get_bit(1, 0), true);
    /// assert_eq!(board.get_bit(0, 0), false);
    /// assert_eq!(board.to_board_id(), "PXDHAE");
    /// ```
    #[must_use]
    pub fn from_board_id(id: &str) -> Option<Self> {
        let _ = const {
            const_assert!(
                ROWS <= constants::ACTUAL_ROWS,
                "rows must fit in board storage"
            )
        };
        if id.len() > Self::BOARD_ID_SIZE {
            return None;
        }

        let id_bytes = id.as_bytes();
        let mut board = Self::EMPTY_BOARD;

        for chunk_idx in 0..Self::BOARD_ID_SIZE {
            let ch = if chunk_idx < id_bytes.len() {
                id_bytes[chunk_idx]
            } else {
                b'A' // implicit trailing zeros
            };

            let value = match ch {
                b'A'..=b'Z' => ch - b'A',
                b'2'..=b'7' => ch - b'2' + 26,
                _ => return None,
            };

            for bit in 0..5 {
                if (value >> (4 - bit)) & 1 != 0 {
                    let bit_idx = chunk_idx * 5 + bit;
                    if bit_idx >= Self::BOARD_SIZE {
                        return None;
                    }
                    let row = bit_idx / COLS;
                    let col = bit_idx % COLS;
                    board.set_bit(col, row);
                }
            }
        }

        Some(board)
    }
}

impl<const COLS: usize, const ROWS: usize> TetrisBoard<COLS, ROWS> {
    const EMPTY_BOARD: Self = Self::default();
    const FULL_BOARD: Self = Self([BackingColType::MAX; COLS]);

    // A hard drop needs room for the whole piece before clearing rows and checking loss.
    const MAX_PLAYABLE_ROWS: usize = {
        let pieces = TetrisPiece::all();
        let mut tallest = 0;
        let mut p = 0;
        while p < pieces.len() {
            let mut r = 0;
            while r < Rotation::ALL.len() {
                let height = pieces[p].height(Rotation::ALL[r]) as usize;
                if height > tallest {
                    tallest = height;
                }
                r += 1;
            }
            p += 1;
        }
        BackingColType::BITS as usize - tallest
    };

    /// Creates a new empty Tetris board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// assert_eq!(board.count(), 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self::EMPTY_BOARD
    }

    /// Returns the total number of filled cells, including overflow rows.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0);
    /// assert_eq!(board.count(), 1);
    /// ```
    #[inline_conditioned(always)]
    pub const fn count(&self) -> u32 {
        let mut acc = 0;
        repeat_idx_unroll!(COLS, I, {
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
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0);
    /// board.clear();
    /// assert_eq!(board.count(), 0);
    /// ```
    #[inline_conditioned(always)]
    pub const fn clear(&mut self) {
        *self = Self::EMPTY_BOARD;
    }

    /// Fills every backing-storage cell, including overflow rows.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.fill();
    /// assert!(board.get_bit(0, 0));
    /// assert!(board.is_lost());
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
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 1); // Set bit at row 1
    /// assert_eq!(board.height(), 2); // Height should be 2
    /// ```
    #[inline_conditioned(always)]
    pub const fn height(&self) -> u32 {
        let mut bits: BackingColType = 0;
        repeat_idx_unroll!(COLS, I, {
            bits |= self.0[I];
        });
        BackingColType::BITS - bits.leading_zeros()
    }

    /// Returns each column's height, including overflow rows and gaps below its highest cell.
    ///
    /// Height in Tetris is defined as 0 if the column is empty, otherwise 1 + the highest row index with a set bit.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::tetris::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0);
    /// board.set_bit(1, 5);
    /// let heights = board.heights();
    /// assert_eq!(heights[0], 1);
    /// assert_eq!(heights[1], 6);
    /// ```
    #[inline_conditioned(always)]
    pub const fn heights(&self) -> [u32; COLS] {
        let mut out = [0u32; COLS];
        repeat_idx_unroll!(COLS, I, {
            out[I] = BackingColType::BITS - self.0[I].leading_zeros();
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
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
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
    pub const fn holes(&self) -> [u32; COLS] {
        let mut out = [0u32; COLS];
        repeat_idx_unroll!(COLS, I, {
            let height = BackingColType::BITS - self.0[I].leading_zeros();
            let filled_cells = self.0[I].count_ones();
            out[I] = height - filled_cells;
        });
        out
    }

    /// Returns the surface roughness of the board.
    ///
    /// Roughness is defined as the sum of absolute height differences
    /// between adjacent columns.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, 0); // col 0 height = 1
    /// board.set_bit(1, 2); // col 1 height = 3
    /// // |1 - 3| + |3 - 0| across columns 0..2, then zeros afterward.
    /// assert_eq!(board.roughness(), 5);
    /// ```
    #[inline_conditioned(always)]
    pub const fn roughness(&self) -> u32 {
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let mut sum = 0;
        repeat_idx_unroll!(COLS - 1, I, {
            let left = BackingColType::BITS - self.0[I].leading_zeros();
            let right = BackingColType::BITS - self.0[I + 1].leading_zeros();
            sum += left.abs_diff(right);
        });
        sum
    }

    /// Returns the total number of holes across all columns in the board.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::TetrisBoard;
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
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
        repeat_idx_unroll!(COLS, I, {
            let height = BackingColType::BITS - self.0[I].leading_zeros();
            let filled_cells = self.0[I].count_ones();
            sum += height - filled_cells;
        });
        sum
    }

    /// Returns whether any occupied cell is at or above row `ROWS`.
    /// Placement checks this after clearing complete rows.
    ///
    /// # Example
    ///
    /// ```
    /// use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig};
    ///
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// board.set_bit(0, StandardTetris::ROWS - 1); // Highest playable row
    /// assert!(!board.is_lost());
    /// board.set_bit(0, StandardTetris::ROWS); // First overflow row
    /// assert!(board.is_lost());
    /// ```
    #[inline_conditioned(always)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "The board layout requires ROWS below the 32-bit backing height at compile time."
    )]
    pub const fn is_lost(&self) -> bool {
        let _ = const {
            const_assert!(
                ROWS < constants::ACTUAL_ROWS,
                "playable rows must be less than actual rows"
            )
        };
        self.height() > ROWS as u32
    }

    /// Clears up to four complete rows, including overflow rows, and shifts cells above them down.
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
    /// # Panics
    /// Panics if more than four rows are complete, before changing the board.
    ///
    /// # Examples
    ///
    /// ```
    /// use tetris_game::{tetris::TetrisBoard, StandardTetris, TetrisGameConfig};
    /// let mut board: TetrisBoard<10, 20> = TetrisBoard::<10, 20>::new();
    /// // Fill an entire row at the current board width.
    /// for col in 0..StandardTetris::COLS {
    ///     board.set_bit(col, 0);
    /// }
    /// board.set_bit(0, 2); // This cell shifts down one row.
    /// assert_eq!(board.clear_and_collapse_rows(), 1);
    /// assert_eq!(board.count(), 1);
    /// assert!(board.get_bit(0, 1));
    /// ```
    #[inline_conditioned(always)]
    pub const fn clear_and_collapse_rows(&mut self) -> u32 {
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let mut filled_rows_mask = BackingColType::MAX;
        repeat_idx_unroll!(COLS, I, {
            filled_rows_mask &= self.0[I];
        });
        if filled_rows_mask == 0 {
            return 0;
        }
        let cleared = filled_rows_mask.count_ones();
        assert!(
            cleared <= 4,
            "a placement cannot complete more than four rows"
        );
        repeat_idx_unroll!(4, _ROW, {
            let pos = filled_rows_mask.trailing_zeros();
            let pivot = (1 as BackingColType) << pos;
            let keep = pivot - 1;
            let above = !pivot & !keep;
            repeat_idx_unroll!(COLS, COL, {
                self.0[COL] = (self.0[COL] & keep) | ((self.0[COL] & above) >> 1);
            });
            filled_rows_mask = (filled_rows_mask & !pivot) >> 1;
            if filled_rows_mask == 0 {
                return cleared;
            }
        });
        cleared
    }

    /// Drops and merges a piece, clears completed rows, and checks for loss.
    ///
    /// Accepts all four rotations, including symmetric aliases for O/I/S/Z.
    /// A terminal board ignores the placement and returns a lost result without changing it.
    /// Playable rows must leave storage room for the tallest piece, checked at compile time.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard, TetrisPiecePlacement};
    ///
    /// // Three overflow rows cannot hold a vertical I before the loss check.
    /// let mut board = TetrisBoard::<4, { constants::ACTUAL_ROWS - 3 }>::new();
    /// board.apply_piece_placement(TetrisPiecePlacement::<4>::from_index(0));
    /// ```
    ///
    /// Steps:
    /// 1. Return if the board is terminal; otherwise check placement bounds.
    /// 2. Hard-drop the piece and merge it into the board.
    /// 3. Clear completed rows and shift the remaining rows down.
    /// 4. Check for loss and normalize a lost board to the terminal marker.
    ///
    /// # Panics
    /// Panics if the board is not terminal and the piece extends beyond its boundary.
    /// The board is unchanged if bounds validation fails.
    #[inline_conditioned(always)]
    #[allow(
        unconditional_panic,
        reason = "The unroller emits constant-false branches containing out-of-range indices."
    )] // The unroller emits unreachable out-of-range array indices.
    #[allow(
        clippy::out_of_bounds_indexing,
        reason = "The constant-unrolled body is guarded by the actual array length; Clippy also visits unreachable expanded indices."
    )]
    pub fn apply_piece_placement(
        &mut self,
        placement: TetrisPiecePlacement<COLS>,
    ) -> PlacementResult {
        let _ = const { const_assert!(ROWS > 0, "playable rows must be greater than zero") };
        let _ = const { const_assert!(COLS > 0, "columns must be greater than zero") };
        let _ = const {
            const_assert!(
                ROWS <= Self::MAX_PLAYABLE_ROWS,
                "playable rows must leave room for the tallest piece"
            )
        };
        // Preserve terminal boards without processing another placement.
        if self.is_lost() {
            return PlacementResult {
                is_lost: true,
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
                        let col_height = BackingColType::BITS - $board_cols[I].leading_zeros();
                        let diff = col_height.abs_diff(TRAILING[I]);

                        // Equivalent to min(min_diff, diff).
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
        let start_idx = placement.orientation.column.inner() as usize;
        let end_idx = start_idx + piece_width as usize;
        assert!(end_idx <= COLS, "piece extends beyond the board boundary");
        let board_cols = &mut self.0[start_idx..end_idx];

        match (placement.piece, placement.orientation.rotation) {
            (TetrisPiece::O, _) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::O_PIECE_ROT_0)
            }
            (TetrisPiece::I, Rotation::R0 | Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::I_PIECE_ROT_0)
            }
            (TetrisPiece::I, Rotation::R90 | Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::I_PIECE_ROT_1)
            }
            (TetrisPiece::S, Rotation::R0 | Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::S_PIECE_ROT_0)
            }
            (TetrisPiece::S, Rotation::R90 | Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::S_PIECE_ROT_1)
            }
            (TetrisPiece::Z, Rotation::R0 | Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::Z_PIECE_ROT_0)
            }
            (TetrisPiece::Z, Rotation::R90 | Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::Z_PIECE_ROT_1)
            }
            (TetrisPiece::T, Rotation::R0) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_0)
            }
            (TetrisPiece::T, Rotation::R90) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_1)
            }
            (TetrisPiece::T, Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_2)
            }
            (TetrisPiece::T, Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::T_PIECE_ROT_3)
            }
            (TetrisPiece::L, Rotation::R0) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_0)
            }
            (TetrisPiece::L, Rotation::R90) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_1)
            }
            (TetrisPiece::L, Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_2)
            }
            (TetrisPiece::L, Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::L_PIECE_ROT_3)
            }
            (TetrisPiece::J, Rotation::R0) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_0)
            }
            (TetrisPiece::J, Rotation::R90) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_1)
            }
            (TetrisPiece::J, Rotation::R180) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_2)
            }
            (TetrisPiece::J, Rotation::R270) => {
                place_piece_with_consts!(board_cols, tetris_piece_data::J_PIECE_ROT_3)
            }
        };

        let lines_cleared = self.clear_and_collapse_rows();
        let is_lost = self.is_lost();
        if is_lost {
            self.fill();
        }

        PlacementResult {
            lines_cleared,
            is_lost,
        }
    }
}

impl<const COLS: usize, const ROWS: usize> Display for TetrisBoard<COLS, ROWS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, " {:2}| board", "r")?;
        for row in (0..constants::ACTUAL_ROWS).rev() {
            write!(f, "{row:2} | ")?;
            for &column in &self.0 {
                let bit = (column >> row) & 1 != 0;
                f.write_str(if bit { "#" } else { "." })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Selects the layout for cell arrays and packed bytes.
/// Cell arrays cover `ROWS`; packed bytes include every backing row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Major {
    /// Row-major: bottom row first, with columns left-to-right within each row.
    Row,
    /// Column-major: column zero first, with rows bottom-to-top within each column.
    Col,
}

/// A deterministic random number generator for Tetris games.
///
/// # Purpose
///
/// This RNG provides reproducible randomness for Tetris games, enabling:
/// - Deterministic game replays from a seed
/// - Consistent testing and debugging
/// - Copying RNG state for piece-stream previews
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
    seed: u64,
    state: u64,
    pub n: u64,
}

impl Default for TetrisGameRng {
    fn default() -> Self {
        Self::new(0x5EED)
    }
}

impl TetrisGameRng {
    /// Applies the MurmurHash3 64-bit finalization mixer to a value.
    ///
    /// # Arguments
    ///
    /// * `x` - The 64-bit value to mix
    ///
    /// # Effect
    ///
    /// Replaces `x` with the mixed value. Returns `()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tetris_game::tetris::TetrisGameRng;
    ///
    /// let mut x = 12345u64;
    /// TetrisGameRng::fmix64(&mut x);
    /// assert_eq!(x, 1716623506685013753u64);
    /// ```
    #[inline_conditioned(always)]
    const fn fmix64(x: &mut u64) {
        *x ^= *x >> 33;
        *x = x.wrapping_mul(0xff51afd7ed558ccd);
        *x ^= *x >> 33;
        *x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        *x ^= *x >> 33;
    }

    /// Creates a new RNG with the given nonzero seed.
    ///
    /// # Panics
    /// Panics if `seed` is zero, which would produce only zero outputs.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        assert!(seed != 0, "RNG seed must be nonzero");
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
    /// `peek_n(0)` returns the current state; `peek_n(1)` returns the next output.
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
    ///
    /// # Panics
    /// Panics if `seed` is zero, without changing the RNG.
    #[inline_conditioned(always)]
    pub const fn reseed(&mut self, seed: u64) {
        *self = Self::new(seed);
    }
}

/// The loss state and cleared-row count returned by a placement.
/// Board placements report actual clears even on loss; game placements report zero clears on loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementResult {
    /// Whether the board is lost after the call, including an already-terminal board.
    pub is_lost: bool,
    /// Number of complete rows cleared (0-4).
    pub lines_cleared: u32,
}

/// A complete Tetris game state.
///
/// This struct encapsulates all state needed to play and replay a Tetris game:
///
/// - **Board**: The current state of placed pieces
/// - **Current piece**: The piece the player is about to place
/// - **Bag**: The remaining draw order, refilled from the configured piece set
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
/// Fresh games with the same configuration and seed produce identical piece sequences,
/// enabling replay and testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Getters)]
pub struct TetrisGame<C: TetrisGameConfig = StandardTetris>
where
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    /// The current board state.
    pub board: TetrisBoard<{ C::COLS }, { C::ROWS }>,
    /// The piece that will be placed next.
    #[getter(copy)]
    current_piece: TetrisPiece,
    /// The deterministic RNG (stores seed for reset).
    pub rng: TetrisGameRng,
    /// Pieces remaining in the current bag.
    #[getter]
    bag: TetrisPieceBag,
    /// Total lines cleared by successful placements.
    pub lines_cleared: u32,
    /// Lines cleared by the most recent successful placement.
    pub recent_lines_cleared: u32,
    /// Number of successful placements.
    pub piece_count: u32,
}

impl<C: TetrisGameConfig> Display for TetrisGame<C>
where
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TetrisGame:")?;
        writeln!(f, "  bag:")?;
        writeln!(f, "    {}", self.bag)?;
        writeln!(f, "  current_piece:")?;
        writeln!(f, "    {}", self.current_piece)?;
        writeln!(f, "  lines_cleared:")?;
        writeln!(f, "    {}", self.lines_cleared)?;
        writeln!(f, "  board:")?;
        writeln!(f, "    {}", self.board)?;
        Ok(())
    }
}

impl<C: TetrisGameConfig> Default for TetrisGame<C>
where
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C: TetrisGameConfig> TetrisGame<C>
where
    [(); C::COLS]:,
    [(); C::ROWS]:,
{
    const ASSERT_CONFIG: () = {
        // Board geometry: leave room for a whole piece before clearing rows and checking loss.
        assert!(C::ROWS > 0, "playable rows must be greater than zero");
        assert!(
            C::ROWS <= TetrisBoard::<{ C::COLS }, { C::ROWS }>::MAX_PLAYABLE_ROWS,
            "playable rows must leave room for the tallest piece"
        );

        // Placement catalog: require positive width and byte-sized placement IDs.
        let _ = TetrisPiecePlacement::<{ C::COLS }>::NUM_PLACEMENTS;

        // Piece set: require at least one piece, with a fitting placement for each.
        assert!(!C::PIECE_SET.empty(), "game piece set must not be empty");
        let starts = TetrisPiecePlacement::<{ C::COLS }>::PIECE_START_INDICES;
        let mut i = 0;
        while i < constants::NUM_TETRIS_PIECES {
            assert!(
                !C::PIECE_SET.contains(TetrisPiece::all()[i]) || starts[i] < starts[i + 1],
                "configured piece cannot fit the game width"
            );
            i += 1;
        }
    };

    #[inline_conditioned(always)]
    fn draw_piece(bag: &mut TetrisPieceBag, rng: &mut TetrisGameRng) -> TetrisPiece {
        bag.draw()
            .or_else(|| bag.refill(C::PIECE_SET).shuffle(rng).draw())
            .expect("validated game piece set must not be empty")
    }

    /// Creates an empty game using the fixed default RNG seed.
    #[must_use]
    pub fn new() -> Self {
        Self::new_with_seed(TetrisGameRng::default().seed)
    }

    /// Creates a new game with a specific seed for reproducible piece sequences.
    ///
    /// Configuration dimensions and piece sets are checked at compile time.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{TetrisGame, TetrisGameConfig, TetrisPieceBagState};
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct NoRows;
    /// impl TetrisGameConfig for NoRows {
    ///     const COLS: usize = 4;
    ///     const ROWS: usize = 0;
    ///     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// }
    /// let _ = TetrisGame::<NoRows>::new_with_seed(42);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisGame, TetrisGameConfig, TetrisPieceBagState};
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct InsufficientHeadroom;
    /// impl TetrisGameConfig for InsufficientHeadroom {
    ///     const COLS: usize = 4;
    ///     const ROWS: usize = constants::ACTUAL_ROWS - 3;
    ///     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// }
    /// let _ = TetrisGame::<InsufficientHeadroom>::new_with_seed(42);
    /// ```
    ///
    /// # Panics
    /// Panics if `seed` is zero.
    #[must_use]
    pub fn new_with_seed(seed: u64) -> Self {
        let () = Self::ASSERT_CONFIG;
        let mut rng = TetrisGameRng::new(seed);
        let mut bag = TetrisPieceBag::from_bag_state(C::PIECE_SET);
        bag.shuffle(&mut rng);
        let current_piece = Self::draw_piece(&mut bag, &mut rng);
        Self {
            board: TetrisBoard::<{ C::COLS }, { C::ROWS }>::new(),
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
    /// Configuration dimensions and piece sets are checked at compile time.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece, TetrisPieceBagState};
    ///
    /// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// struct InsufficientHeadroom;
    /// impl TetrisGameConfig for InsufficientHeadroom {
    ///     const COLS: usize = 4;
    ///     const ROWS: usize = constants::ACTUAL_ROWS - 3;
    ///     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// }
    /// let _ = TetrisGame::<InsufficientHeadroom>::new_with_board_bag_piece_seeded(
    ///     TetrisBoard::<4, { constants::ACTUAL_ROWS - 3 }>::new(),
    ///     TetrisPieceBagState::full(), TetrisPiece::O, 42,
    /// );
    /// ```
    ///
    /// # Panics
    /// Panics if `seed` is zero or the current piece or remainder is outside `C::PIECE_SET`.
    #[must_use]
    pub fn new_with_board_bag_piece_seeded(
        board: TetrisBoard<{ C::COLS }, { C::ROWS }>,
        bag_state: TetrisPieceBagState,
        current_piece: TetrisPiece,
        seed: u64,
    ) -> Self {
        let () = Self::ASSERT_CONFIG;
        let piece_set = C::PIECE_SET;
        assert!(
            piece_set.contains(current_piece),
            "current piece is outside the configured piece set"
        );
        let mut rng = TetrisGameRng::new(seed);
        assert!(
            u8::from(bag_state) & !u8::from(piece_set) == 0,
            "remaining pieces must belong to the refill piece set"
        );
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

    /// Replaces the remaining bag and current piece, reseeds the RNG, and shuffles the bag.
    /// Preserves the board and statistics. The bag describes future draws;
    /// the supplied current piece is not drawn from it.
    ///
    /// # Panics
    /// Panics if `seed` is zero or the supplied piece or remaining bag is outside `C::PIECE_SET`.
    pub fn set_bag_piece_seeded(
        &mut self,
        bag_state: TetrisPieceBagState,
        piece: TetrisPiece,
        seed: u64,
    ) {
        assert!(
            C::PIECE_SET.contains(piece),
            "current piece is outside the configured piece set"
        );
        assert!(
            u8::from(bag_state) & !u8::from(C::PIECE_SET) == 0,
            "remaining pieces must belong to the refill piece set"
        );
        self.rng.reseed(seed);
        self.bag.refill(bag_state).shuffle(&mut self.rng);
        self.current_piece = piece;
    }

    /// Returns placements for the current piece that fit the configured board width.
    #[must_use]
    pub fn current_placements(&self) -> &[TetrisPiecePlacement<{ C::COLS }>] {
        TetrisPiecePlacement::<{ C::COLS }>::all_from_piece(self.current_piece)
    }

    /// Drops the current piece at this orientation.
    /// Returns the loss state and lines cleared; draws the next piece if play continues.
    ///
    /// # Panics
    /// Panics if a nonterminal board receives an out-of-bounds placement, without changing the game.
    pub fn apply_orientation(
        &mut self,
        orientation: TetrisPieceOrientation<{ C::COLS }>,
    ) -> PlacementResult {
        self.apply_placement(TetrisPiecePlacement {
            piece: self.current_piece,
            orientation,
        })
    }

    /// Drops a placement and returns its loss state and lines cleared.
    /// Draws the next piece if play continues.
    /// A losing placement leaves the piece stream and statistics unchanged.
    ///
    /// # Panics
    /// Panics if the piece differs from the current piece or the placement is out of bounds.
    /// A terminal board ignores bounds after the current-piece check.
    /// The game is unchanged on invalid input.
    pub fn apply_placement(
        &mut self,
        placement: TetrisPiecePlacement<{ C::COLS }>,
    ) -> PlacementResult {
        assert!(
            placement.piece == self.current_piece,
            "placement piece does not match the current piece"
        );

        let PlacementResult {
            is_lost,
            lines_cleared,
        } = self.board.apply_piece_placement(placement);
        if is_lost {
            return PlacementResult {
                is_lost: true,
                lines_cleared: 0,
            };
        }
        self.lines_cleared += lines_cleared;
        self.recent_lines_cleared = lines_cleared;
        self.current_piece = Self::draw_piece(&mut self.bag, &mut self.rng);
        self.piece_count += 1;
        PlacementResult {
            is_lost: false,
            lines_cleared,
        }
    }

    /// Iterates over the current piece and future draws without changing the game.
    ///
    /// The infinite sequence uses a copy of the bag and RNG, including automatic refills.
    /// Later changes to the game do not affect the iterator. Use `take(n)` to limit it.
    ///
    /// ```
    /// use tetris_game::{StandardTetris, TetrisGame};
    ///
    /// let game = TetrisGame::<StandardTetris>::new_with_seed(42);
    /// let preview: Vec<_> = game.iter_next_pieces().take(5).collect();
    /// assert_eq!(preview.len(), 5);
    /// assert_eq!(preview[0], game.current_piece());
    /// ```
    pub fn iter_next_pieces(&self) -> impl Iterator<Item = TetrisPiece> + use<C> {
        let mut bag = self.bag;
        let mut rng = self.rng;
        std::iter::once(self.current_piece).chain(std::iter::repeat_with(move || {
            Self::draw_piece(&mut bag, &mut rng)
        }))
    }

    /// Starts a fresh empty game and clears all statistics.
    ///
    /// # Arguments
    ///
    /// * `new_seed` - If `Some(seed)`, uses that seed. If `None`, replays the seed
    ///   currently stored in the RNG, including a seed set by a scenario update.
    ///
    /// # Panics
    /// Panics if `new_seed` is `Some(0)`, without changing the game.
    pub fn reset(&mut self, new_seed: Option<u64>) {
        *self = Self::new_with_seed(new_seed.unwrap_or(self.rng.seed));
    }
}

// These cases must fail at compile time, so they are separate from runtime unit tests.
#[cfg(doctest)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixture casts use explicitly bounded board dimensions and case counts, or deliberately extract test bits."
)]
mod configuration_tests {
    /// Both board dimensions must be supplied in type signatures.
    ///
    /// ```compile_fail,E0107
    /// use tetris_game::TetrisBoard;
    /// let _: TetrisBoard;
    /// ```
    ///
    /// ```compile_fail,E0107
    /// use tetris_game::TetrisBoard;
    /// let _: TetrisBoard<10>;
    /// ```
    fn board_dimensions_have_no_defaults() {}

    /// Orientation and placement types also require an explicit board width.
    ///
    /// ```compile_fail,E0107
    /// use tetris_game::TetrisPieceOrientation;
    /// let _: TetrisPieceOrientation;
    /// ```
    ///
    /// ```compile_fail,E0107
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement;
    /// ```
    fn piece_geometry_has_no_default_width() {}

    /// Column construction checks its exclusive limit at compile time.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::Column;
    /// let _ = Column::<0>::new(0);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::Column;
    /// let _ = Column::<{ u8::MAX as usize + 1 }>::new(0);
    /// ```
    fn column_construction_requires_a_representable_limit() {}

    /// Byte-sized orientation encoding rejects widths requiring more than 256 indices.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPieceOrientation;
    /// let _ = TetrisPieceOrientation::<65>::from_index(0);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{Column, Rotation, TetrisPieceOrientation};
    /// let _ = TetrisPieceOrientation::<65> {
    ///     rotation: Rotation::R0,
    ///     column: Column::new(0),
    /// }.index();
    /// ```
    fn orientation_indices_require_a_representable_width() {}

    /// Default columns cannot make zero-width orientation indexing valid.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{Column, Rotation, TetrisPieceOrientation};
    /// let _ = TetrisPieceOrientation::<0> {
    ///     rotation: Rotation::R0,
    ///     column: Column::default(),
    /// }.index();
    /// ```
    fn zero_width_orientations_have_no_index() {}

    /// Cell conversions reject rows outside the backing column storage.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, Major, TetrisBoard};
    /// let board = TetrisBoard::<1, { constants::ACTUAL_ROWS + 1 }>::new();
    /// let _ = board.to_cell_array(Major::Col);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, Major, TetrisBoard};
    /// const ROWS: usize = constants::ACTUAL_ROWS + 1;
    /// let _ = TetrisBoard::<1, ROWS>::from_cell_array([0; ROWS], Major::Col);
    /// ```
    fn cell_conversions_reject_rows_outside_storage() {}

    /// Board ID conversion also requires every encoded row to fit in storage.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard};
    /// let _ = TetrisBoard::<1, { constants::ACTUAL_ROWS + 1 }>::new().to_board_id();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard};
    /// let _ = TetrisBoard::<1, { constants::ACTUAL_ROWS + 1 }>::from_board_id("");
    /// ```
    fn board_ids_reject_rows_outside_storage() {}

    /// Random selection requires nonempty dimensions and addressable rows.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisBoard;
    /// TetrisBoard::<0, 20>::new().set_random_bits(1, &mut rand::rng());
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisBoard;
    /// TetrisBoard::<10, 0>::new().set_random_bits(1, &mut rand::rng());
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard};
    /// TetrisBoard::<1, { constants::ACTUAL_ROWS + 1 }>::new()
    ///     .set_random_bits(1, &mut rand::rng());
    /// ```
    fn random_cells_require_nonempty_dimensions_within_storage() {}

    /// Adjacent-column metrics and complete rows require at least one column.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisBoard;
    /// let _ = TetrisBoard::<0, 20>::new().roughness();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisBoard;
    /// let _ = TetrisBoard::<0, 20>::new().clear_and_collapse_rows();
    /// ```
    fn column_operations_require_nonempty_columns() {}

    /// Packed byte transposition requires at least one column.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{Major, TetrisBoard};
    /// let _ = TetrisBoard::<0, 20>::new().to_bytes(Major::Row);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{Major, TetrisBoard};
    /// let _ = TetrisBoard::<0, 20>::from_bytes([], Major::Row);
    /// ```
    fn packed_bytes_require_nonempty_columns() {}

    /// Loss detection requires an overflow row above the playable area.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{constants, TetrisBoard};
    /// let _ = TetrisBoard::<1, { constants::ACTUAL_ROWS }>::new().is_lost();
    /// ```
    fn loss_detection_requires_overflow_storage() {}

    /// Placement requires a nonempty playable area, even without a game wrapper.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::{TetrisBoard, TetrisPiece, TetrisPiecePlacement};
    /// let placement = TetrisPiecePlacement::<4>::all_from_piece(TetrisPiece::O)[0];
    /// TetrisBoard::<4, 0>::new().apply_piece_placement(placement);
    /// ```
    fn placement_requires_nonempty_playable_rows() {}

    /// Zero-column games are rejected at compile time.
    ///
    /// ```compile_fail,E0080
    /// # use tetris_game::{StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece, TetrisPieceBagState};
    /// # #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// # struct InvalidConfig;
    /// # impl TetrisGameConfig for InvalidConfig {
    /// #     const ROWS: usize = StandardTetris::ROWS;
    /// #     const COLS: usize = 0;
    /// #     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// # }
    /// let _ = TetrisGame::<InvalidConfig>::new();
    /// ```
    fn rejects_zero_columns() {}

    /// Placement IDs must fit in a byte.
    ///
    /// ```compile_fail,E0080
    /// # use tetris_game::{StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece, TetrisPieceBagState};
    /// # #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// # struct InvalidConfig;
    /// # impl TetrisGameConfig for InvalidConfig {
    /// #     const ROWS: usize = StandardTetris::ROWS;
    /// #     const COLS: usize = 15;
    /// #     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// # }
    /// let _ = TetrisGame::<InvalidConfig>::new();
    /// ```
    fn rejects_widths_outside_the_placement_encoding() {}

    /// Orientation masks must fit in a `u64`.
    ///
    /// ```compile_fail,E0080
    /// # use tetris_game::{StandardTetris, Rotation, TetrisGameConfig, TetrisPieceBagState, TetrisPieceOrientation};
    /// # #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// # struct TooWide;
    /// # impl TetrisGameConfig for TooWide {
    /// #     const ROWS: usize = StandardTetris::ROWS;
    /// #     const COLS: usize = u64::BITS as usize / Rotation::ALL.len() + 1;
    /// #     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// # }
    /// let _ = TetrisPieceOrientation::<{ TooWide::COLS }>::bitmask_from_piece(tetris_game::TetrisPiece::O);
    /// ```
    fn rejects_widths_outside_the_orientation_mask() {}

    /// Placements from another board width cannot be applied to a board.
    ///
    /// ```compile_fail,E0308
    /// use tetris_game::{TetrisBoard, TetrisPiecePlacement};
    /// let mut board = TetrisBoard::<4, 20>::new();
    /// let placement = <TetrisPiecePlacement<10>>::from_index(0);
    /// board.apply_piece_placement(placement);
    /// ```
    fn rejects_placements_from_a_different_width() {}

    /// The configured refill set must not be empty.
    ///
    /// ```compile_fail,E0080
    /// # use tetris_game::{StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece, TetrisPieceBagState};
    /// # #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// # struct InvalidConfig;
    /// # impl TetrisGameConfig for InvalidConfig {
    /// #     const ROWS: usize = StandardTetris::ROWS;
    /// #     const COLS: usize = StandardTetris::COLS;
    /// #     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::from_pieces([]);
    /// # }
    /// let _ = TetrisGame::<InvalidConfig>::new_with_seed(42);
    /// ```
    fn rejects_empty_piece_sets() {}

    /// Placement sampling requires four columns and byte-sized placement IDs.
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement<0> = rand::random();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement<1> = rand::random();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement<2> = rand::random();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement<3> = rand::random();
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use tetris_game::TetrisPiecePlacement;
    /// let _: TetrisPiecePlacement<15> = rand::random();
    /// ```
    fn sampling_rejects_unrepresentable_widths() {}

    /// Every configured piece must have a fitting rotation.
    ///
    /// ```compile_fail,E0080
    /// # use tetris_game::{StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece, TetrisPieceBagState};
    /// # #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    /// # struct InvalidConfig;
    /// # impl TetrisGameConfig for InvalidConfig {
    /// #     const ROWS: usize = StandardTetris::ROWS;
    /// #     const COLS: usize = 1;
    /// #     const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    /// # }
    /// let _ = TetrisGame::<InvalidConfig>::new_with_board_bag_piece_seeded(
    ///     TetrisBoard::<1, 20>::new(), TetrisPieceBagState::from_pieces([]), TetrisPiece::I, 42,
    /// );
    /// ```
    fn rejects_pieces_without_fitting_rotations() {}
}

// Measure production code coverage, not assertion failure paths in the tests.
#[cfg(test)]
#[coverage(off)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixture casts use explicitly bounded board dimensions and case counts, or deliberately extract test bits."
)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    // Standard-board cases use concrete aliases; configured cases name their config.
    type Column = super::Column<{ StandardTetris::COLS }>;
    type TetrisPieceOrientation = super::TetrisPieceOrientation<10>;
    type TetrisPiecePlacement = super::TetrisPiecePlacement<10>;

    type Board = TetrisBoard<10, 20>;
    type Game = TetrisGame<StandardTetris>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct TestWidth<const WIDTH: usize>;
    impl<const WIDTH: usize> TetrisGameConfig for TestWidth<WIDTH> {
        const ROWS: usize = StandardTetris::ROWS;
        const COLS: usize = WIDTH;
        const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct OnlyO;
    impl TetrisGameConfig for OnlyO {
        const ROWS: usize = StandardTetris::ROWS;
        const COLS: usize = 4;
        const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::from_pieces([TetrisPiece::O]);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct OnlyI;
    impl TetrisGameConfig for OnlyI {
        const ROWS: usize = StandardTetris::ROWS;
        const COLS: usize = 1;
        const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::from_pieces([TetrisPiece::I]);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct TestPieceSet<const MASK: u8>;
    impl<const MASK: u8> TetrisGameConfig for TestPieceSet<MASK> {
        const ROWS: usize = StandardTetris::ROWS;
        const COLS: usize = StandardTetris::COLS;
        const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState(MASK);
    }

    // Includes holes, an empty column, and bits above the playable ceiling.
    const METRIC_COLUMNS: [u32; StandardTetris::COLS] = [
        0,
        1,
        4,
        0b101,
        0b11010,
        0b111,
        0x8000_0000,
        0b1010_1010,
        1 << StandardTetris::ROWS,
        0b10001,
    ];

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct TestDimensions<const COLS: usize, const ROWS: usize>;
    impl<const COLS: usize, const ROWS: usize> TetrisGameConfig for TestDimensions<COLS, ROWS> {
        const COLS: usize = COLS;
        const ROWS: usize = ROWS;
        const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
    }

    fn assert_display_propagates_write_errors(value: &impl std::fmt::Display) {
        use std::fmt::Write;
        struct LimitedWriter {
            remaining: usize,
        }
        impl std::fmt::Write for LimitedWriter {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                if s.len() > self.remaining {
                    return Err(std::fmt::Error);
                }
                self.remaining -= s.len();
                Ok(())
            }
        }
        let rendered = value.to_string();
        for limit in 0..rendered.len() {
            assert!(write!(&mut LimitedWriter { remaining: limit }, "{value}").is_err());
        }
        assert!(
            write!(
                &mut LimitedWriter {
                    remaining: rendered.len()
                },
                "{value}"
            )
            .is_ok()
        );
    }

    // Independent canonical zero-orientation grids, rotated clockwise without using production shape tables.
    fn reference_shape(piece: TetrisPiece, rotation: Rotation) -> ([u32; 4], usize, usize) {
        let rows: &[&str] = match piece {
            TetrisPiece::O => &["##", "##"],
            TetrisPiece::I => &["####"],
            TetrisPiece::S => &[".##", "##."],
            TetrisPiece::Z => &["##.", ".##"],
            TetrisPiece::T => &["###", ".#."],
            TetrisPiece::L => &["..#", "###"],
            TetrisPiece::J => &["#..", "###"],
        };
        let mut grid = rows
            .iter()
            .map(|row| row.bytes().map(|cell| cell == b'#').collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for _ in 0..rotation.index() {
            let mut rotated = vec![vec![false; grid.len()]; grid[0].len()];
            for (row, cells) in grid.iter().enumerate() {
                for (col, &cell) in cells.iter().enumerate() {
                    rotated[col][grid.len() - row - 1] = cell;
                }
            }
            grid = rotated;
        }
        let mut columns = [0; 4];
        for (row, cells) in grid.iter().enumerate() {
            for (col, &cell) in cells.iter().enumerate() {
                columns[col] |= (cell as u32) << (BackingColType::BITS as usize - row - 1);
            }
        }
        (columns, grid[0].len(), grid.len())
    }

    mod pieces {
        use super::*;

        #[test]
        fn packed_decoder_accepts_only_valid_indices() {
            for index in 0..=u8::MAX {
                let decoded = std::panic::catch_unwind(|| TetrisPiece::from_packed_index(index));
                if index < constants::NUM_TETRIS_PIECES as u8 {
                    assert_eq!(
                        decoded.expect("piece index"),
                        TetrisPiece::all()[index as usize]
                    );
                } else {
                    assert!(decoded.is_err(), "index={index}");
                }
            }
        }

        #[test]
        fn display_uses_standard_letters() {
            let rendered: String = TetrisPiece::all().iter().map(ToString::to_string).collect();
            assert_eq!(rendered, "OISZTLJ");
        }

        #[test]
        fn display_propagates_write_errors() {
            for piece in TetrisPiece::all() {
                assert_display_propagates_write_errors(&piece);
            }
        }

        #[test]
        fn sampling_matches_uniform_index_sampling() {
            use rand::{
                SeedableRng,
                rngs::{SmallRng, StdRng},
            };

            fn compare<R: RngExt + SeedableRng>() {
                for seed in [0, 1, 0xDEAD_BEEF_CAFE, u64::MAX] {
                    let mut actual = R::seed_from_u64(seed);
                    let mut reference = R::seed_from_u64(seed);
                    let mut seen = 0u8;
                    for draw in 0..10_000 {
                        let expected = TetrisPiece::all()
                            [reference.random_range(0..constants::NUM_TETRIS_PIECES)];
                        let piece: TetrisPiece = actual.random();
                        assert_eq!(
                            piece,
                            expected,
                            "rng={}, seed={seed}, draw={draw}",
                            std::any::type_name::<R>()
                        );
                        seen |= piece.mask();
                    }
                    assert_eq!(
                        seen,
                        u8::from(TetrisPieceBagState::full()),
                        "rng={}, seed={seed}",
                        std::any::type_name::<R>()
                    );
                    assert_eq!(
                        actual.random::<u64>(),
                        reference.random::<u64>(),
                        "rng={}, seed={seed}",
                        std::any::type_name::<R>()
                    );
                }
            }

            compare::<SmallRng>();
            compare::<StdRng>();
        }

        #[test]
        fn indices_and_masks_preserve_standard_piece_order() {
            let pieces = [
                TetrisPiece::O,
                TetrisPiece::I,
                TetrisPiece::S,
                TetrisPiece::Z,
                TetrisPiece::T,
                TetrisPiece::L,
                TetrisPiece::J,
            ];
            assert_eq!(TetrisPiece::all(), pieces);
            for (index, piece) in pieces.into_iter().enumerate() {
                assert_eq!(piece.mask(), 1 << index);
                assert_eq!(piece.index(), index as u8);
            }
        }

        #[test]
        fn value_and_option_fit_in_one_byte() {
            assert_eq!(std::mem::size_of::<TetrisPiece>(), 1);
            assert_eq!(std::mem::size_of::<Option<TetrisPiece>>(), 1);
        }

        #[test]
        fn every_piece_rotation_matches_independent_grid_geometry() {
            const SHAPES: [[([BackingColType; 4], u8); 4]; constants::NUM_TETRIS_PIECES] = {
                let pieces = TetrisPiece::all();
                let mut shapes = [[([0; 4], 0); 4]; constants::NUM_TETRIS_PIECES];
                let mut p = 0;
                while p < pieces.len() {
                    let mut r = 0;
                    while r < Rotation::ALL.len() {
                        shapes[p][r] = pieces[p].shape_cols(Rotation::ALL[r]);
                        r += 1;
                    }
                    p += 1;
                }
                shapes
            };

            for piece in TetrisPiece::all() {
                let mut unique_shapes = Vec::new();
                for rotation in Rotation::ALL {
                    let (columns, width, height) = reference_shape(piece, rotation);
                    if !unique_shapes.contains(&(columns, width, height)) {
                        unique_shapes.push((columns, width, height));
                    }
                    assert_eq!(
                        columns.iter().map(|col| col.count_ones()).sum::<u32>(),
                        4,
                        "piece={piece}, rotation={rotation}"
                    );
                    assert_eq!(
                        piece.width(rotation) as usize,
                        width,
                        "piece={piece}, rotation={rotation}"
                    );
                    assert_eq!(
                        piece.height(rotation) as usize,
                        height,
                        "piece={piece}, rotation={rotation}"
                    );
                    assert_eq!(
                        piece.shape_cols(rotation),
                        (columns, width as u8),
                        "piece={piece}, rotation={rotation}"
                    );
                    assert_eq!(
                        SHAPES[piece.index() as usize][rotation.index() as usize],
                        (columns, width as u8),
                        "const piece={piece}, rotation={rotation}"
                    );
                }
                assert_eq!(
                    piece.num_rotations() as usize,
                    unique_shapes.len(),
                    "piece={piece}"
                );
            }
        }
    }

    mod orientations {
        use super::*;

        #[test]
        fn display_uses_rotation_and_column_indices() {
            for index in 0..TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS {
                let orientation = TetrisPieceOrientation::from_index(index as u8);
                assert_eq!(
                    orientation.to_string(),
                    format!(
                        "rot={} col={}",
                        index / StandardTetris::COLS,
                        index % StandardTetris::COLS
                    )
                );
                assert_display_propagates_write_errors(&orientation);
            }
        }

        #[test]
        fn column_display_includes_its_index() {
            assert_eq!(Column::new(9).to_string(), "Column(9)");
        }

        #[test]
        fn rotation_indices_preserve_quarter_turn_order() {
            let rotations = [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270];
            assert_eq!(Rotation::ALL, rotations);
            for (index, rotation) in rotations.into_iter().enumerate() {
                assert_eq!(rotation.index(), index as u8);
            }
        }

        #[test]
        fn rotation_fits_in_one_byte() {
            assert_eq!(std::mem::size_of::<Rotation>(), 1);
        }

        #[test]
        fn rotation_display_uses_its_numeric_index() {
            for (index, rotation) in Rotation::ALL.into_iter().enumerate() {
                assert_eq!(rotation.to_string(), index.to_string());
            }
        }
    }

    mod placements {
        use super::*;

        #[test]
        fn maximum_placement_count_matches_the_largest_piece_catalog() {
            let largest = TetrisPiece::all()
                .into_iter()
                .map(|piece| TetrisPiecePlacement::all_from_piece(piece).len())
                .max()
                .expect("seven pieces");
            assert_eq!(TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT, largest);
            assert_eq!(largest, 34);
        }

        #[test]
        fn symmetric_rotation_ids_decode_to_equivalent_placements() {
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    for col in 0..=StandardTetris::COLS - piece.width(rotation) as usize {
                        let input = TetrisPiecePlacement {
                            piece,
                            orientation: TetrisPieceOrientation {
                                rotation,
                                column: crate::Column::new(col as u8),
                            },
                        };
                        let expected = TetrisPiecePlacement::ALL_PLACEMENTS
                            .iter()
                            .copied()
                            .find(|p| {
                                p.piece == piece
                                    && usize::from(p.orientation.column.inner()) == col
                                    && p.orientation.rotation.index()
                                        == rotation.index() % piece.num_rotations()
                            })
                            .expect("canonical placement");
                        let decoded = TetrisPiecePlacement::from_index(input.index());
                        assert_eq!(
                            decoded, expected,
                            "piece={piece}, rotation={rotation}, col={col}"
                        );
                        let mut actual_board = Board::new();
                        let mut expected_board = Board::new();
                        assert_eq!(
                            actual_board.apply_piece_placement(input),
                            expected_board.apply_piece_placement(decoded)
                        );
                        assert_eq!(actual_board, expected_board);
                    }
                }
            }
        }

        #[test]
        fn random_placements_cover_every_canonical_move() {
            let mut expected = [0u64; constants::NUM_TETRIS_PIECES];
            for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
                expected[placement.piece.index() as usize] |= 1 << placement.orientation.index();
            }

            let mut rng = SmallRng::seed_from_u64(0x5eed);
            let mut seen = [0u64; constants::NUM_TETRIS_PIECES];
            for _ in 0..4096 * constants::NUM_TETRIS_PIECES {
                let placement: TetrisPiecePlacement = rng.random();
                seen[placement.piece.index() as usize] |= 1 << placement.orientation.index();
            }
            assert_eq!(seen, expected);
        }

        #[test]
        fn random_placements_are_canonical_and_cover_every_piece() {
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            let mut seen = 0u8;
            for _ in 0..4096 {
                let placement: TetrisPiecePlacement = rng.random();
                assert_eq!(
                    TetrisPiecePlacement::from_index(placement.index()),
                    placement
                );
                assert!(!Board::new().apply_piece_placement(placement).is_lost);
                seen |= placement.piece.mask();
            }
            assert_eq!(seen, TetrisPieceBagState::FULL_MASK);
        }

        #[test]
        fn display_matches_independent_piece_geometry() {
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    let placement = TetrisPiecePlacement {
                        piece,
                        orientation: TetrisPieceOrientation {
                            rotation,
                            column: crate::Column::new(0),
                        },
                    };
                    let (columns, width, height) = reference_shape(piece, rotation);
                    let mut expected = format!("{piece} rot={} col=0", rotation.index());
                    for row in (BackingColType::BITS as usize - height
                        ..BackingColType::BITS as usize)
                        .rev()
                    {
                        expected.push('\n');
                        for column in &columns[..width] {
                            expected.push(if column & (1 << row) != 0 { '#' } else { '.' });
                        }
                    }
                    assert_eq!(placement.to_string(), expected);
                }
            }
        }

        #[test]
        fn display_propagates_write_errors() {
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    assert_display_propagates_write_errors(&TetrisPiecePlacement {
                        piece,
                        orientation: TetrisPieceOrientation {
                            rotation,
                            column: crate::Column::new(0),
                        },
                    });
                }
            }
        }

        #[test]
        fn ids_preserve_legacy_piece_rotation_column_order() {
            // Independent legacy widths: zero means a symmetric rotation excluded
            // from the canonical IDs. A round-trip alone could hide an encoding change.
            let widths = [
                [2, 0, 0, 0], // O
                [4, 1, 0, 0], // I
                [3, 2, 0, 0], // S
                [3, 2, 0, 0], // Z
                [3, 2, 3, 2], // T
                [3, 2, 3, 2], // L
                [3, 2, 3, 2], // J
            ];
            let rotations = [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270];
            let mut id = 0;
            for (piece_index, widths) in widths.into_iter().enumerate() {
                let start = id;
                for (rotation, width) in rotations.into_iter().zip(widths) {
                    if width == 0 {
                        continue;
                    }
                    for column in 0..=10 - width {
                        let placement = TetrisPiecePlacement::from_index(id);
                        assert_eq!(placement.piece.mask(), 1 << piece_index);
                        assert_eq!(placement.orientation.rotation, rotation);
                        assert_eq!(usize::from(placement.orientation.column.inner()), column);
                        assert_eq!(placement.index(), id);
                        id += 1;
                    }
                }
                assert_eq!(
                    TetrisPiecePlacement::all_from_piece(TetrisPiece::all()[piece_index]),
                    &TetrisPiecePlacement::ALL_PLACEMENTS[start as usize..id as usize],
                );
            }
            assert_eq!(id as usize, TetrisPiecePlacement::NUM_PLACEMENTS);
        }

        #[test]
        fn display_renders_recognizable_standard_shapes() {
            for (piece, rotation, expected) in [
                (TetrisPiece::T, Rotation::R0, "T rot=0 col=0\n###\n.#."),
                (TetrisPiece::I, Rotation::R0, "I rot=0 col=0\n####"),
                (TetrisPiece::I, Rotation::R90, "I rot=1 col=0\n#\n#\n#\n#"),
                (TetrisPiece::O, Rotation::R0, "O rot=0 col=0\n##\n##"),
            ] {
                let placement = TetrisPiecePlacement {
                    piece,
                    orientation: TetrisPieceOrientation {
                        rotation,
                        column: crate::Column::new(0),
                    },
                };
                assert_eq!(placement.to_string(), expected);
            }
        }
    }

    mod bag_state {
        use super::*;

        #[test]
        fn decodes_every_valid_mask() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                assert_eq!(u8::from(state), mask, "mask={mask:#09b}");
            }
        }

        #[test]
        fn rejects_every_mask_with_reserved_bits() {
            for mask in TetrisPieceBagState::FULL_MASK + 1..=u8::MAX {
                assert_eq!(
                    TetrisPieceBagState::try_from(mask),
                    Err(InvalidBagMask),
                    "mask={mask:#010b}"
                );
            }
        }

        #[test]
        fn count_matches_set_bits() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                assert_eq!(state.count(), mask.count_ones() as u8, "mask={mask:#09b}");
            }
        }

        #[test]
        fn empty_matches_zero_mask() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                assert_eq!(state.empty(), mask == 0, "mask={mask:#09b}");
            }
        }

        #[test]
        fn contains_matches_each_piece_bit() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                for piece in TetrisPiece::all() {
                    let bit = 1u8 << piece.index();
                    assert_eq!(
                        state.contains(piece),
                        mask & bit != 0,
                        "mask={mask:#09b}, piece={piece}"
                    );
                }
            }
        }

        #[test]
        fn remove_clears_only_the_requested_piece() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                for piece in TetrisPiece::all() {
                    let mut state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                    state.remove(piece);
                    let expected = mask & !(1u8 << piece.index());
                    assert_eq!(u8::from(state), expected, "mask={mask:#09b}, piece={piece}");
                }
            }
        }

        #[test]
        fn removing_a_piece_twice_has_the_same_effect_as_once() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                for piece in TetrisPiece::all() {
                    let mut state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                    state.remove(piece);
                    let after_first_removal = state;
                    state.remove(piece);
                    assert_eq!(
                        state, after_first_removal,
                        "mask={mask:#09b}, piece={piece}"
                    );
                }
            }
        }

        #[test]
        fn fill_restores_every_subset_to_full() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let mut state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                state.fill();
                assert_eq!(state, TetrisPieceBagState::full(), "mask={mask:#09b}");
            }
        }

        #[test]
        fn invalid_bag_mask_display_explains_reserved_bit() {
            assert_eq!(
                InvalidBagMask.to_string(),
                "bag mask has reserved bit 7 set"
            );
        }

        #[test]
        fn display_lists_remaining_pieces() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                let pieces: Vec<_> = TetrisPiece::all()
                    .into_iter()
                    .filter(|piece| mask & (1u8 << piece.index()) != 0)
                    .map(|piece| piece.to_string())
                    .collect();
                assert_eq!(
                    state.to_string(),
                    format!("BagState({})", pieces.join(", "))
                );
            }
        }

        #[test]
        fn display_propagates_write_errors() {
            for state in [
                TetrisPieceBagState::from_pieces([]),
                TetrisPieceBagState::full(),
            ] {
                assert_display_propagates_write_errors(&state);
            }
        }

        #[test]
        fn iteration_matches_set_bits_in_piece_order() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let bag = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                let expected: Vec<_> = TetrisPiece::all()
                    .into_iter()
                    .filter(|piece| mask & piece.mask() != 0)
                    .collect();
                assert_eq!(
                    bag.iter_pieces().collect::<Vec<_>>(),
                    expected,
                    "mask={mask:#09b}"
                );
            }
        }

        #[test]
        fn pop_removes_the_lowest_piece_for_every_remainder() {
            for initial in 0..=TetrisPieceBagState::FULL_MASK {
                let mut remaining = initial;
                let mut bag = TetrisPieceBagState::try_from(initial).expect("valid bag mask");
                while remaining != 0 {
                    let lowest_bit = remaining & remaining.wrapping_neg();
                    let piece = bag.pop().expect("nonempty bag");
                    assert_eq!(
                        piece.mask(),
                        lowest_bit,
                        "initial={initial:#09b}, remaining={remaining:#09b}"
                    );
                    assert!(
                        !bag.contains(piece),
                        "initial={initial:#09b}, remaining={remaining:#09b}"
                    );
                    remaining &= !lowest_bit;
                    assert_eq!(
                        u8::from(bag),
                        remaining,
                        "initial={initial:#09b}, remaining={remaining:#09b}"
                    );
                    assert_eq!(
                        bag.count(),
                        remaining.count_ones() as u8,
                        "initial={initial:#09b}, remaining={remaining:#09b}"
                    );
                }
                assert_eq!(
                    bag.pop(),
                    None,
                    "initial={initial:#09b}, remaining={remaining:#09b}"
                );
                assert_eq!(
                    u8::from(bag),
                    0,
                    "initial={initial:#09b}, remaining={remaining:#09b}"
                );
            }
        }

        #[test]
        fn full_contains_every_standard_piece() {
            let state = TetrisPieceBagState::full();
            assert_eq!(state.count() as usize, constants::NUM_TETRIS_PIECES);
            for piece in TetrisPiece::all() {
                assert!(state.contains(piece), "piece={piece}");
            }
        }

        #[test]
        fn default_is_full() {
            assert_eq!(TetrisPieceBagState::default(), TetrisPieceBagState::full());
        }

        #[test]
        fn from_pieces_sets_only_the_requested_bits_in_const_context() {
            const SINGLE: TetrisPieceBagState = TetrisPieceBagState::from_pieces([TetrisPiece::I]);
            const MULTI: TetrisPieceBagState = TetrisPieceBagState::from_pieces([
                TetrisPiece::S,
                TetrisPiece::Z,
                TetrisPiece::T,
                TetrisPiece::L,
            ]);
            const ALL: TetrisPieceBagState = TetrisPieceBagState::from_pieces(TetrisPiece::all());
            for (state, expected) in [
                (SINGLE, 0b000_0010u8),
                (MULTI, 0b011_1100),
                (ALL, 0b111_1111),
            ] {
                assert_eq!(u8::from(state), expected);
                assert_eq!(state.count(), expected.count_ones() as u8);
                for piece in TetrisPiece::all() {
                    assert_eq!(state.contains(piece), expected & piece.mask() != 0);
                }
            }
        }

        #[test]
        fn mutation_and_queries_work_in_const_context() {
            const STATE: TetrisPieceBagState = {
                let mut state = TetrisPieceBagState::from_pieces([TetrisPiece::O, TetrisPiece::I]);
                state.remove(TetrisPiece::O);
                assert!(!state.contains(TetrisPiece::O));
                assert!(state.contains(TetrisPiece::I));
                assert!(state.count() == 1);
                assert!(matches!(state.pop(), Some(TetrisPiece::I)));
                assert!(state.empty());
                state.fill();
                state
            };
            assert_eq!(STATE, TetrisPieceBagState::full());
        }

        #[test]
        #[should_panic(expected = "duplicate piece")]
        fn from_pieces_rejects_duplicates() {
            let _ = TetrisPieceBagState::from_pieces([TetrisPiece::I, TetrisPiece::I]);
        }
    }

    mod bag {
        use super::*;

        #[test]
        fn display_lists_pieces_in_draw_order() {
            let mut bag =
                TetrisPieceBag::from_slice(&[TetrisPiece::J, TetrisPiece::O, TetrisPiece::I])
                    .expect("distinct pieces");
            assert_eq!(bag.to_string(), r#"["J", "O", "I"]"#);
            bag.draw();
            assert_eq!(bag.to_string(), r#"["O", "I"]"#);
            bag.draw();
            assert_eq!(bag.to_string(), r#"["I"]"#);
            bag.draw();
            assert_eq!(bag.to_string(), "[]");
        }

        #[test]
        fn shuffle_table_matches_fisher_yates_for_every_random_byte() {
            for count in 0..=constants::NUM_TETRIS_PIECES {
                for step in 0..constants::NUM_TETRIS_PIECES - 1 {
                    for byte in 0..=u8::MAX as usize {
                        let target = TetrisPieceBag::SHUFFLE_TARGETS[count][step][byte] as usize;
                        let expected = if step + 1 < count {
                            step + byte * (count - step) / 256
                        } else {
                            step
                        };
                        assert_eq!(target, expected, "count={count}, step={step}, byte={byte}");
                    }
                }
            }
        }

        #[test]
        fn from_slice_rejects_duplicates_at_every_pair_of_positions() {
            for duplicate in TetrisPiece::all() {
                let other: Vec<_> = TetrisPiece::all()
                    .into_iter()
                    .filter(|&p| p != duplicate)
                    .collect();
                for length in 2..=constants::NUM_TETRIS_PIECES {
                    for first in 0..length {
                        for second in first + 1..length {
                            let mut remaining = other.iter().copied();
                            let pieces: Vec<_> = (0..length)
                                .map(|index| {
                                    if index == first || index == second {
                                        duplicate
                                    } else {
                                        remaining.next().expect("enough distinct pieces")
                                    }
                                })
                                .collect();
                            assert_eq!(
                                TetrisPieceBag::from_slice(&pieces),
                                Err(InvalidBagPieces::Duplicate(duplicate)),
                                "pieces={pieces:?}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn membership_tracks_remaining_pieces_after_each_draw() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("fixture conversion must validate in membership_tracks_remaining_pieces_after_each_draw");
                let mut bag = TetrisPieceBag::from_bag_state(state);
                let mut expected = state;
                for (position, piece) in state.iter_pieces().enumerate() {
                    for p in TetrisPiece::all() {
                        assert_eq!(
                            bag.contains(p),
                            expected.contains(p),
                            "mask={mask:#09b}, position={position}, queried_piece={p}"
                        );
                    }
                    assert_eq!(
                        bag.draw(),
                        Some(piece),
                        "mask={mask:#09b}, position={position}"
                    );
                    expected.remove(piece);
                }
                for p in TetrisPiece::all() {
                    assert!(!bag.contains(p), "mask={mask:#09b}, queried_piece={p}");
                }
            }
        }

        #[test]
        fn display_propagates_write_errors() {
            for state in [
                TetrisPieceBagState::from_pieces([]),
                TetrisPieceBagState::full(),
            ] {
                assert_display_propagates_write_errors(&TetrisPieceBag::from_bag_state(state));
            }
        }

        #[test]
        fn default_contains_all_standard_pieces() {
            assert_eq!(
                TetrisPieceBag::default(),
                TetrisPieceBag::from_bag_state(TetrisPieceBagState::full())
            );
        }

        #[test]
        fn size_and_alignment_match_one_u64() {
            assert_eq!(
                std::mem::size_of::<TetrisPieceBag>(),
                std::mem::size_of::<u64>()
            );
            assert_eq!(
                std::mem::align_of::<TetrisPieceBag>(),
                std::mem::align_of::<u64>()
            );
        }

        #[test]
        fn shuffle_returns_a_reference_to_the_original_bag() {
            for pieces in [&[][..], &[TetrisPiece::O][..], &TetrisPiece::all()[..]] {
                let mut bag = TetrisPieceBag::from_slice(pieces).expect("unique pieces");
                let returned = bag.shuffle(&mut TetrisGameRng::new(42)) as *mut TetrisPieceBag;
                assert!(std::ptr::eq(returned, &bag));
            }
        }

        #[test]
        fn refill_shuffle_chain_works_in_const_context() {
            const SHUFFLED: TetrisPieceBag = {
                let mut bag = TetrisPieceBag::from_bag_state(TetrisPieceBagState::from_pieces([]));
                let mut rng = TetrisGameRng::new(42);
                bag.refill(TetrisPieceBagState::full()).shuffle(&mut rng);
                bag
            };
            assert_eq!(SHUFFLED.count(), 7);
        }

        #[test]
        fn refill_shuffle_draw_chain_updates_the_original_bag() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let pieces = TetrisPieceBagState::try_from(mask).expect("valid mask");
                let mut expected = TetrisPieceBag::from_bag_state(pieces);
                let mut expected_rng = TetrisGameRng::new(42);
                expected.shuffle(&mut expected_rng);
                let expected_piece = expected.draw();

                let mut bag = TetrisPieceBag::default();
                let mut rng = TetrisGameRng::new(42);
                assert_eq!(
                    bag.refill(pieces).shuffle(&mut rng).draw(),
                    expected_piece,
                    "mask={mask:#09b}, seed=42"
                );
                assert_eq!(bag, expected, "mask={mask:#09b}, seed=42");
                assert_eq!(rng, expected_rng, "mask={mask:#09b}, seed=42");
            }
        }

        #[test]
        fn from_bag_state_preserves_every_subset_in_piece_order() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                let pieces: Vec<_> = state.iter_pieces().collect();
                let expected = TetrisPieceBag::from_slice(&pieces).expect("unique pieces");
                let mut bag = TetrisPieceBag::from_bag_state(state);
                assert_eq!(bag, expected, "mask={mask:#09b}");
                assert_eq!(bag.count() as usize, pieces.len(), "mask={mask:#09b}");
                for (position, piece) in pieces.into_iter().enumerate() {
                    assert_eq!(
                        bag.draw(),
                        Some(piece),
                        "mask={mask:#09b}, position={position}"
                    );
                }
                assert_eq!(bag.draw(), None, "mask={mask:#09b}");
            }
        }

        #[test]
        fn from_bag_state_works_in_const_context_for_every_subset() {
            const FROM_STATES: [TetrisPieceBag; TetrisPieceBagState::FULL_MASK as usize + 1] = {
                let empty = TetrisPieceBag::from_bag_state(TetrisPieceBagState::from_pieces([]));
                let mut bags = [empty; TetrisPieceBagState::FULL_MASK as usize + 1];
                let mut mask = 0;
                while mask < bags.len() {
                    bags[mask] = TetrisPieceBag::from_bag_state(TetrisPieceBagState(mask as u8));
                    mask += 1;
                }
                bags
            };
            for (mask, bag) in FROM_STATES.into_iter().enumerate() {
                let pieces: Vec<_> = TetrisPiece::all()
                    .into_iter()
                    .filter(|piece| mask as u8 & piece.mask() != 0)
                    .collect();
                assert_eq!(
                    bag,
                    TetrisPieceBag::from_slice(&pieces).expect("unique pieces"),
                    "mask={mask:#09b}"
                );
            }
        }

        #[test]
        fn refill_replaces_existing_contents_with_the_requested_subset() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                let mut bag = TetrisPieceBag::from_slice(&[TetrisPiece::J, TetrisPiece::O])
                    .expect("unique pieces");
                let returned = bag.refill(state) as *mut TetrisPieceBag;
                assert!(std::ptr::eq(returned, &bag), "mask={mask:#09b}");
                assert_eq!(
                    bag,
                    TetrisPieceBag::from_bag_state(state),
                    "mask={mask:#09b}"
                );
            }
        }

        #[test]
        fn drawing_an_exhausted_bag_returns_none_without_refilling() {
            for mask in 0..=TetrisPieceBagState::FULL_MASK {
                let state = TetrisPieceBagState::try_from(mask).expect("valid bag mask");
                let mut bag = TetrisPieceBag::from_bag_state(state);
                for position in 0..state.count() {
                    assert!(
                        bag.draw().is_some(),
                        "mask={mask:#09b}, position={position}"
                    );
                }
                let exhausted = bag;
                assert_eq!(bag.draw(), None, "mask={mask:#09b}");
                assert_eq!(bag.draw(), None, "mask={mask:#09b}");
                assert_eq!(bag, exhausted, "mask={mask:#09b}");
            }
        }

        #[test]
        fn from_slice_preserves_draw_order_in_const_context() {
            const BAG: Result<TetrisPieceBag, InvalidBagPieces> =
                TetrisPieceBag::from_slice(&[TetrisPiece::J, TetrisPiece::O, TetrisPiece::T]);
            let mut bag = BAG.expect("unique pieces");
            for piece in [TetrisPiece::J, TetrisPiece::O, TetrisPiece::T] {
                assert_eq!(bag.draw(), Some(piece));
            }
            assert_eq!(bag.draw(), None);
        }

        #[test]
        fn from_slice_accepts_empty_input_in_const_context() {
            const EMPTY: Result<TetrisPieceBag, InvalidBagPieces> = TetrisPieceBag::from_slice(&[]);
            assert_eq!(EMPTY.expect("empty slice").count(), 0);
        }

        #[test]
        fn from_slice_rejects_duplicates_in_const_context() {
            const DUPLICATE: Result<TetrisPieceBag, InvalidBagPieces> =
                TetrisPieceBag::from_slice(&[
                    TetrisPiece::O,
                    TetrisPiece::I,
                    TetrisPiece::I,
                    TetrisPiece::O,
                ]);
            assert_eq!(DUPLICATE, Err(InvalidBagPieces::Duplicate(TetrisPiece::I)));
        }

        #[test]
        fn from_slice_rejects_excess_pieces_in_const_context() {
            const LENGTH: usize = constants::NUM_TETRIS_PIECES + 1;
            const TOO_MANY: Result<TetrisPieceBag, InvalidBagPieces> =
                TetrisPieceBag::from_slice(&[TetrisPiece::O; LENGTH]);
            assert_eq!(TOO_MANY, Err(InvalidBagPieces::TooMany(LENGTH)));
        }

        #[test]
        fn from_slice_reports_the_first_repeated_piece() {
            for (pieces, duplicate) in [
                (
                    [
                        TetrisPiece::O,
                        TetrisPiece::O,
                        TetrisPiece::I,
                        TetrisPiece::I,
                    ],
                    TetrisPiece::O,
                ),
                (
                    [
                        TetrisPiece::O,
                        TetrisPiece::I,
                        TetrisPiece::I,
                        TetrisPiece::O,
                    ],
                    TetrisPiece::I,
                ),
            ] {
                assert_eq!(
                    TetrisPieceBag::from_slice(&pieces),
                    Err(InvalidBagPieces::Duplicate(duplicate))
                );
            }
        }

        #[test]
        fn from_slice_rejects_more_than_seven_pieces_before_duplicates() {
            const LENGTH: usize = constants::NUM_TETRIS_PIECES + 1;
            assert_eq!(
                TetrisPieceBag::from_slice(&[TetrisPiece::O; LENGTH]),
                Err(InvalidBagPieces::TooMany(LENGTH))
            );
        }

        #[test]
        fn invalid_bag_pieces_display_explains_the_invalid_contents() {
            assert_eq!(
                InvalidBagPieces::Duplicate(TetrisPiece::T).to_string(),
                "bag contains duplicate piece T"
            );
            assert_eq!(
                InvalidBagPieces::TooMany(8).to_string(),
                "bag has 8 pieces; at most seven are allowed"
            );
        }

        #[test]
        fn remove_preserves_other_pieces_for_every_permutation_and_prefix() {
            fn check(pieces: &mut [TetrisPiece], position: usize) {
                if position < pieces.len() {
                    for next in position..pieces.len() {
                        pieces.swap(position, next);
                        check(pieces, position + 1);
                        pieces.swap(position, next);
                    }
                    return;
                }
                for count in 0..=pieces.len() {
                    let initial =
                        TetrisPieceBag::from_slice(&pieces[..count]).expect("unique permutation");
                    let mut ordered = initial;
                    for (position, &piece) in pieces[..count].iter().enumerate() {
                        assert_eq!(
                            ordered.draw(),
                            Some(piece),
                            "pieces={pieces:?}, count={count}, position={position}"
                        );
                    }
                    assert_eq!(ordered.draw(), None, "pieces={pieces:?}, count={count}");
                    for piece in TetrisPiece::all() {
                        let mut bag = initial;
                        let present = pieces[..count].contains(&piece);
                        assert_eq!(
                            bag.remove(piece),
                            present,
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        assert_eq!(
                            bag.count() as usize,
                            count - usize::from(present),
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        assert!(
                            !bag.contains(piece),
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        let after = bag;
                        assert!(
                            !bag.remove(piece),
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        assert_eq!(
                            bag, after,
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        if !present {
                            assert_eq!(
                                bag, initial,
                                "pieces={pieces:?}, count={count}, removed_piece={piece}"
                            );
                        }
                        for (position, &expected) in
                            pieces[..count].iter().filter(|&&p| p != piece).enumerate()
                        {
                            assert_eq!(
                                bag.draw(),
                                Some(expected),
                                "pieces={pieces:?}, count={count}, removed_piece={piece}, position={position}"
                            );
                        }
                        assert_eq!(
                            bag.count(),
                            0,
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                        assert_eq!(
                            bag.draw(),
                            None,
                            "pieces={pieces:?}, count={count}, removed_piece={piece}"
                        );
                    }
                }
            }
            check(&mut TetrisPiece::all(), 0);
        }

        #[test]
        fn shuffle_matches_unpacked_fisher_yates_for_all_remainders() {
            for seed in [1, 2, 42, 123, u64::MAX] {
                for mask in 0..=TetrisPieceBagState::FULL_MASK {
                    for reversed in [false, true] {
                        let mut pieces: Vec<_> = TetrisPiece::all()
                            .into_iter()
                            .filter(|piece| mask & piece.mask() != 0)
                            .collect();
                        if reversed {
                            pieces.reverse();
                        }
                        let mut bag = TetrisPieceBag::from_slice(&pieces).expect("unique pieces");
                        let mut rng = TetrisGameRng::new(seed);
                        let mut reference_rng = rng;
                        for shuffle in 0..3 {
                            let word = reference_rng.next_u64();
                            for step in 0..pieces.len().saturating_sub(1) {
                                let byte = ((word >> (8 * step)) & 255) as usize;
                                let target = step + ((byte * (pieces.len() - step)) >> 8);
                                pieces.swap(step, target);
                            }
                            bag.shuffle(&mut rng);
                            assert_eq!(
                                bag.count() as usize,
                                pieces.len(),
                                "seed={seed}, mask={mask:#09b}, reversed={reversed}, shuffle={shuffle}"
                            );
                            let mut drawn = bag;
                            for (position, &piece) in pieces.iter().enumerate() {
                                assert_eq!(
                                    drawn.draw(),
                                    Some(piece),
                                    "seed={seed}, mask={mask:#09b}, reversed={reversed}, shuffle={shuffle}, position={position}"
                                );
                                assert_eq!(
                                    drawn.count() as usize,
                                    pieces.len() - position - 1,
                                    "seed={seed}, mask={mask:#09b}, reversed={reversed}, shuffle={shuffle}, position={position}"
                                );
                            }
                            assert_eq!(
                                drawn.draw(),
                                None,
                                "seed={seed}, mask={mask:#09b}, reversed={reversed}, shuffle={shuffle}"
                            );
                            assert_eq!(
                                rng, reference_rng,
                                "seed={seed}, mask={mask:#09b}, reversed={reversed}, shuffle={shuffle}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        #[allow(
            clippy::cast_precision_loss,
            reason = "Floating-point values here are approximate scores, tensor math, or reporting statistics; exact state identifiers remain integers."
        )]
        fn shuffle_position_frequencies_stay_within_fixed_seed_tolerance() {
            const SAMPLES: usize = 1000;
            let mut counts = [[0usize; constants::NUM_TETRIS_PIECES]; constants::NUM_TETRIS_PIECES];
            let mut rng = TetrisGameRng::new(42);
            rng.next_u64(); // Preserve the original regression sample's starting RNG position.
            for _ in 0..SAMPLES {
                let mut bag = TetrisPieceBag::default();
                bag.shuffle(&mut rng);
                for position in &mut counts {
                    let piece = bag.draw().expect("nonempty bag");
                    position[piece.index() as usize] += 1;
                }
            }
            // A fixed-seed distribution smoke check, not a proof of unbiased sampling.
            let tolerance = 1.36 / (SAMPLES as f64).sqrt();
            for (position, counts) in counts.into_iter().enumerate() {
                let mut cumulative = 0;
                for (piece, count) in counts.into_iter().enumerate() {
                    cumulative += count;
                    let observed = cumulative as f64 / SAMPLES as f64;
                    let expected = (piece + 1) as f64 / constants::NUM_TETRIS_PIECES as f64;
                    assert!(
                        (observed - expected).abs() < tolerance,
                        "position={position}, piece={piece}"
                    );
                }
            }
        }
    }

    mod board {
        use super::*;

        #[test]
        fn empty_storage_metrics_do_not_require_playable_geometry() {
            const EMPTY: TetrisBoard<0, 0> = TetrisBoard::default();
            let mut board = TetrisBoard::<0, 0>::new();
            board.clear();
            assert_eq!(board, EMPTY);
            assert_eq!(board.as_limbs(), &[]);
            assert_eq!(board.count(), 0);
            assert_eq!(board.height(), 0);
            assert_eq!(board.heights(), []);
            assert_eq!(board.holes(), []);
            assert_eq!(board.total_holes(), 0);
        }

        #[test]
        fn raw_storage_round_trips_without_using_playable_rows() {
            type RawBoard = TetrisBoard<2, { constants::ACTUAL_ROWS + 1 }>;
            let mut board = RawBoard::from_limbs([BackingColType::MAX, 1]);
            for order in [Major::Row, Major::Col] {
                assert_eq!(RawBoard::from_bytes(board.to_bytes(order), order), board);
            }
            assert_eq!(board.height(), BackingColType::BITS);
            board.clear_bit(0, constants::ACTUAL_ROWS - 1);
            assert!(!board.get_bit(0, constants::ACTUAL_ROWS - 1));
            board.set_bit(0, constants::ACTUAL_ROWS - 1);
            assert_eq!(board.as_limbs(), &[BackingColType::MAX, 1]);
            board.clear();
            assert_eq!(board, RawBoard::default());
        }

        #[test]
        fn cell_and_id_conversions_support_every_storage_row() {
            type StorageBoard = TetrisBoard<2, { constants::ACTUAL_ROWS }>;
            let board = StorageBoard::from_limbs([BackingColType::MAX, 1]);
            for order in [Major::Row, Major::Col] {
                let cells = board.to_cell_array(order);
                assert_eq!(
                    cells.iter().map(|&cell| u32::from(cell)).sum::<u32>(),
                    BackingColType::BITS + 1
                );
                assert_eq!(StorageBoard::from_cell_array(cells, order), board);
            }
            assert_eq!(
                StorageBoard::from_board_id(&board.to_board_id()),
                Some(board)
            );
        }

        #[test]
        fn zero_sized_conversions_return_empty_representations() {
            let no_columns = TetrisBoard::<0, 20>::new();
            let no_rows = TetrisBoard::<2, 0>::new();
            for order in [Major::Row, Major::Col] {
                assert_eq!(no_columns.to_cell_array(order), []);
                assert_eq!(TetrisBoard::<0, 20>::from_cell_array([], order), no_columns);
                assert_eq!(no_rows.to_cell_array(order), []);
                assert_eq!(TetrisBoard::<2, 0>::from_cell_array([], order), no_rows);
            }
            assert_eq!(no_columns.to_board_id(), "");
            assert_eq!(no_rows.to_board_id(), "");
            assert_eq!(TetrisBoard::<0, 20>::from_board_id(""), Some(no_columns));
            assert_eq!(TetrisBoard::<2, 0>::from_board_id(""), Some(no_rows));
            assert_eq!(TetrisBoard::<0, 20>::from_board_id("A"), None);
        }

        #[test]
        fn random_cells_can_use_the_full_storage_height() {
            let mut board = TetrisBoard::<1, { constants::ACTUAL_ROWS }>::new();
            board.set_random_bits(4096, &mut SmallRng::seed_from_u64(42));
            assert_eq!(board.as_limbs(), &[BackingColType::MAX]);
        }

        #[test]
        fn hard_drops_match_a_wider_cell_model_at_supported_height_boundaries() {
            fn check<const COLS: usize, const ROWS: usize>() {
                for occupied in [0, 1, ROWS] {
                    let mut columns = [0u32; COLS];
                    for (col, value) in columns[..COLS - 1].iter_mut().enumerate() {
                        *value = ((1u32 << occupied) - 1)
                            & if col % 2 == 0 { u32::MAX } else { 0x5555_5555 };
                    }
                    // The empty last column excludes pre-existing complete rows.
                    for piece in TetrisPiece::all() {
                        for rotation in Rotation::ALL {
                            let (raw, width, height) = reference_shape(piece, rotation);
                            if width > COLS {
                                continue;
                            }
                            let shape = raw.map(|col| {
                                (col >> (BackingColType::BITS as usize - height)) as u64
                            });
                            for start in 0..=COLS - width {
                                let mut landing = BackingColType::BITS;
                                while landing > 0
                                    && (0..width).all(|col| {
                                        (shape[col] << (landing - 1)) & columns[start + col] as u64
                                            == 0
                                    })
                                {
                                    landing -= 1;
                                }
                                let mut merged = columns.map(|col| col as u64);
                                for col in 0..width {
                                    merged[start + col] |= shape[col] << landing;
                                }
                                let mut expected = [0u64; COLS];
                                let mut next_row = 0;
                                let mut cleared = 0;
                                for row in 0..u64::BITS {
                                    if merged.iter().all(|col| col & (1u64 << row) != 0) {
                                        cleared += 1;
                                    } else {
                                        for col in 0..COLS {
                                            expected[col] |= ((merged[col] >> row) & 1) << next_row;
                                        }
                                        next_row += 1;
                                    }
                                }
                                let lost = expected.iter().any(|col| col >> ROWS != 0);
                                let expected =
                                    expected.map(|col| if lost { u32::MAX } else { col as u32 });
                                let mut board = TetrisBoard::<COLS, ROWS>::from_limbs(columns);
                                let placement = crate::TetrisPiecePlacement::<COLS> {
                                    piece,
                                    orientation: crate::TetrisPieceOrientation::<COLS> {
                                        rotation,
                                        column: crate::Column::new(start as u8),
                                    },
                                };
                                let result = board.apply_piece_placement(placement);
                                assert_eq!(
                                    board.as_limbs(),
                                    &expected,
                                    "cols={COLS}, rows={ROWS}, occupied={occupied}, placement={placement:?}"
                                );
                                assert_eq!(
                                    result.is_lost, lost,
                                    "cols={COLS}, rows={ROWS}, occupied={occupied}, placement={placement:?}"
                                );
                                assert_eq!(
                                    result.lines_cleared, cleared,
                                    "cols={COLS}, rows={ROWS}, occupied={occupied}, placement={placement:?}"
                                );
                            }
                        }
                    }
                }
            }
            check::<4, 1>();
            check::<4, 4>();
            check::<4, 20>();
            check::<4, { Board::MAX_PLAYABLE_ROWS }>();
            check::<10, { Board::MAX_PLAYABLE_ROWS }>();
        }

        #[test]
        fn vertical_i_above_the_playable_ceiling_is_lost_at_the_headroom_limit() {
            const ROWS: usize = Board::MAX_PLAYABLE_ROWS;
            // A vertical I is the tallest tetromino and must fit before checking for loss.
            assert_eq!(ROWS + 4, BackingColType::BITS as usize);
            let mut board = TetrisBoard::<4, ROWS>::new();
            board.set_bit(0, ROWS - 1);
            let placement = crate::TetrisPiecePlacement::<4> {
                piece: TetrisPiece::I,
                orientation: crate::TetrisPieceOrientation {
                    rotation: Rotation::R90,
                    column: crate::Column::new(0),
                },
            };

            let result = board.apply_piece_placement(placement);

            assert!(result.is_lost);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(board.as_limbs(), &[BackingColType::MAX; 4]);
        }

        #[test]
        fn display_includes_playable_and_overflow_rows_in_order() {
            let mut board = Board::new();
            board.set_bit(0, 0);
            board.set_bit(1, constants::ACTUAL_ROWS - 1);
            let rendered = board.to_string();
            let rows: Vec<_> = rendered.lines().skip(2).collect();
            assert_eq!(rows.len(), constants::ACTUAL_ROWS);
            for (index, &row) in rows.iter().enumerate() {
                let height = constants::ACTUAL_ROWS - index - 1;
                let cells = match height {
                    31 => ".#........",
                    0 => "#.........",
                    _ => "..........",
                };
                assert_eq!(row, format!("{height:2} | {cells}"));
            }
        }

        #[test]
        fn random_bit_setup_preserves_existing_cells_and_overflow_rows() {
            let mut board = Board::new();
            board.set_bit(0, constants::ACTUAL_ROWS - 1);
            let mut rng = SmallRng::seed_from_u64(42);
            let unchanged_rng = rng.clone();
            let before = board;
            board.set_random_bits(0, &mut rng);
            assert_eq!(board, before);
            assert_eq!(rng.random::<u64>(), unchanged_rng.clone().random::<u64>());
            board.set_random_bits(1024, &mut rng);
            assert_eq!(
                board.as_limbs()[0] >> StandardTetris::ROWS,
                before.as_limbs()[0] >> StandardTetris::ROWS
            );
            for col in 1..StandardTetris::COLS {
                assert_eq!(board.as_limbs()[col] >> StandardTetris::ROWS, 0);
            }
            assert!(board.count() > before.count());
        }

        #[test]
        fn random_bit_setup_uses_configured_playable_dimensions() {
            let mut board = TetrisBoard::<4, 3>::new();
            let mut rng = SmallRng::seed_from_u64(42);
            board.set_random_bits(128, &mut rng);

            // Repeated selections fill the small playable area without touching overflow rows.
            assert_eq!(board.as_limbs(), &[0b111; 4]);
        }

        #[test]
        fn random_bit_setup_supports_widths_that_do_not_fit_in_u16() {
            fn check<const WIDTH: usize>() {
                let mut board = TetrisBoard::<WIDTH, 1>::new();
                let mut rng = SmallRng::seed_from_u64(42);
                board.set_random_bits(64, &mut rng);

                // Truncating WIDTH to u16 used to panic or restrict every draw to column zero.
                let filled_columns = board
                    .as_limbs()
                    .iter()
                    .filter(|&&column| column != 0)
                    .count();
                assert!(filled_columns > 1);
                assert!(board.as_limbs()[1..].iter().any(|&column| column != 0));
                assert!(board.as_limbs().iter().all(|&column| column <= 1));
            }

            check::<{ u16::MAX as usize + 1 }>();
            check::<{ u16::MAX as usize + 2 }>();
        }

        #[test]
        fn random_bit_setup_is_reproducible_with_the_same_seed() {
            let mut first = Board::new();
            let mut second = Board::new();
            first.set_random_bits(64, &mut SmallRng::seed_from_u64(42));
            second.set_random_bits(64, &mut SmallRng::seed_from_u64(42));
            assert_eq!(first, second);
        }

        #[test]
        fn clear_bit_rejects_columns_outside_storage() {
            let mut initial = Board::new();
            initial.set_bit(0, 1);
            for col in [StandardTetris::COLS, StandardTetris::COLS + 1, usize::MAX] {
                let mut board = initial;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    board.clear_bit(col, 0)
                }));
                assert!(rejected.is_err(), "col={col}");
                assert_eq!(board, initial, "col={col}");
            }
        }

        #[test]
        fn set_bit_rejects_columns_outside_storage() {
            let mut initial = Board::new();
            initial.set_bit(0, 1);
            for col in [StandardTetris::COLS, StandardTetris::COLS + 1, usize::MAX] {
                let mut board = initial;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    board.set_bit(col, 0)
                }));
                assert!(rejected.is_err(), "col={col}");
                assert_eq!(board, initial, "col={col}");
            }
        }

        #[test]
        fn get_bit_rejects_columns_outside_storage() {
            let mut initial = Board::new();
            initial.set_bit(0, 1);
            for col in [StandardTetris::COLS, StandardTetris::COLS + 1, usize::MAX] {
                let board = initial;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    board.get_bit(col, 0)
                }));
                assert!(rejected.is_err(), "col={col}");
                assert_eq!(board, initial, "col={col}");
            }
        }

        #[test]
        fn get_bit_rejects_rows_outside_storage() {
            for row in [
                BackingColType::BITS as usize,
                BackingColType::BITS as usize + 1,
                64,
                usize::MAX,
            ] {
                for col in [0, StandardTetris::COLS - 1] {
                    let mut board = Board::new();
                    board.set_bit(col, 0);
                    let before = board;
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        board.get_bit(col, row)
                    }));
                    assert!(rejected.is_err(), "col={col}, row={row}");
                    assert_eq!(board, before, "col={col}, row={row}");
                }
            }
        }

        #[test]
        fn set_bit_rejects_rows_outside_storage() {
            for row in [
                BackingColType::BITS as usize,
                BackingColType::BITS as usize + 1,
                64,
                usize::MAX,
            ] {
                for col in [0, StandardTetris::COLS - 1] {
                    let mut board = Board::new();
                    board.set_bit(col, 0);
                    let before = board;
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        board.set_bit(col, row)
                    }));
                    assert!(rejected.is_err(), "col={col}, row={row}");
                    assert_eq!(board, before, "col={col}, row={row}");
                }
            }
        }

        #[test]
        fn clear_bit_rejects_rows_outside_storage() {
            for row in [
                BackingColType::BITS as usize,
                BackingColType::BITS as usize + 1,
                64,
                usize::MAX,
            ] {
                for col in [0, StandardTetris::COLS - 1] {
                    let mut board = Board::new();
                    board.set_bit(col, 0);
                    let before = board;
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        board.clear_bit(col, row)
                    }));
                    assert!(rejected.is_err(), "col={col}, row={row}");
                    assert_eq!(board, before, "col={col}, row={row}");
                }
            }
        }

        #[test]
        fn bit_access_supports_every_storage_row() {
            let mut board = Board::new();
            for row in 0..BackingColType::BITS as usize {
                board.set_bit(0, row);
                assert!(board.get_bit(0, row), "row={row}");
                board.clear_bit(0, row);
                assert!(!board.get_bit(0, row), "row={row}");
            }
        }

        #[test]
        fn limbs_round_trip_without_changing_bits() {
            const BOARD: Board = Board::from_limbs(METRIC_COLUMNS);
            const LIMBS: &[u32; StandardTetris::COLS] = BOARD.as_limbs();
            assert_eq!(LIMBS, &METRIC_COLUMNS);
            assert_eq!(Board::from_limbs(*LIMBS), BOARD);
        }

        #[test]
        fn column_heights_match_highest_occupied_cells() {
            let expected = METRIC_COLUMNS.map(|column| {
                (0..BackingColType::BITS)
                    .rev()
                    .find(|&row| column & (1 << row) != 0)
                    .map_or(0, |row| row + 1)
            });
            assert_eq!(Board::from_limbs(METRIC_COLUMNS).heights(), expected);
        }

        #[test]
        fn height_includes_overflow_rows() {
            assert_eq!(
                Board::from_limbs(METRIC_COLUMNS).height(),
                BackingColType::BITS
            );
        }

        #[test]
        fn holes_match_empty_cells_below_each_column_top() {
            let expected: [u32; StandardTetris::COLS] = METRIC_COLUMNS.map(|column| {
                let height = (0..BackingColType::BITS)
                    .rev()
                    .find(|&row| column & (1 << row) != 0)
                    .map_or(0, |row| row + 1);
                (0..height).filter(|&row| column & (1 << row) == 0).count() as u32
            });
            let board = Board::from_limbs(METRIC_COLUMNS);
            assert_eq!(board.holes(), expected);
            assert_eq!(board.total_holes(), expected.iter().sum::<u32>());
        }

        #[test]
        fn roughness_sums_adjacent_height_differences() {
            let board = Board::from_limbs(METRIC_COLUMNS);
            let expected = [0u32, 1, 3, 3, 5, 3, 32, 8, 21, 5]
                .windows(2)
                .map(|pair| pair[0].abs_diff(pair[1]))
                .sum::<u32>();
            assert_eq!(board.roughness(), expected);
        }

        #[test]
        fn default_is_empty() {
            const EMPTY: Board = Board::default();
            assert_eq!(EMPTY.as_limbs(), &[0; StandardTetris::COLS]);
            assert_eq!(Board::default(), EMPTY);
            assert_eq!(Board::new(), EMPTY);
        }

        #[test]
        fn storage_supports_widths_outside_game_action_encoding() {
            // A board can be wider than the game's placement encoding allows.
            assert_eq!(TetrisBoard::<11, 20>::new().as_limbs(), &[0; 11]);
        }

        #[test]
        fn holes_exclude_empty_space_above_the_highest_cell() {
            let mut board = Board::new();
            board.set_bit(0, 4);
            board.set_bit(0, 1);
            assert_eq!(board.holes()[0], 3);
            assert_eq!(board.total_holes(), 3);
        }

        #[test]
        fn display_propagates_write_errors() {
            let mut board = Board::new();
            board.set_bit(0, 0);
            board.set_bit(1, 2);
            assert_display_propagates_write_errors(&board);
        }

        #[test]
        fn placement_zero_drops_an_o_at_the_left_edge() {
            let placement = TetrisPiecePlacement::from_index(0);
            assert_eq!(placement.piece, TetrisPiece::O);
            assert_eq!(placement.orientation.rotation, Rotation::R0);
            assert_eq!(placement.orientation.column.inner(), 0);
            assert_eq!(
                TetrisPiecePlacement::from_index(placement.index()),
                placement
            );

            let mut board = Board::new();
            let result = board.apply_piece_placement(placement);
            assert_eq!(result.lines_cleared, 0);
            assert!(!result.is_lost);
            assert_eq!(board.count(), 4);
        }

        #[test]
        fn out_of_bounds_placements_leave_occupied_boards_unchanged() {
            let mut initial = Board::new();
            initial.set_bit(2, 3);
            for piece in [TetrisPiece::O, TetrisPiece::I] {
                let placement = TetrisPiecePlacement {
                    piece,
                    orientation: TetrisPieceOrientation {
                        rotation: Rotation::R0,
                        column: crate::Column::new(9),
                    },
                };
                let mut board = initial;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    board.apply_piece_placement(placement)
                }));
                assert!(rejected.is_err());
                assert_eq!(board, initial);
            }
        }

        #[test]
        fn every_fitting_orientation_adds_four_cells_to_an_empty_board() {
            // Include symmetric rotations omitted from canonical placement IDs.
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    for column in 0..StandardTetris::COLS {
                        let orientation = TetrisPieceOrientation {
                            rotation,
                            column: Column::new(column as u8),
                        };

                        if usize::from(orientation.column.inner())
                            + piece.width(orientation.rotation) as usize
                            > StandardTetris::COLS
                        {
                            continue;
                        }
                        let mut board = Board::new();
                        let result = board
                            .apply_piece_placement(TetrisPiecePlacement { piece, orientation });
                        assert!(!result.is_lost, "piece={piece}, orientation={orientation}");
                        assert_eq!(result.lines_cleared, 0);
                        assert_eq!(board.count(), 4);
                        assert_eq!(board.height(), piece.height(orientation.rotation) as u32);
                    }
                }
            }
        }

        #[test]
        fn every_out_of_bounds_orientation_is_rejected_without_mutation() {
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    for column in 0..StandardTetris::COLS {
                        let orientation = TetrisPieceOrientation {
                            rotation,
                            column: Column::new(column as u8),
                        };

                        if usize::from(orientation.column.inner())
                            + piece.width(orientation.rotation) as usize
                            <= StandardTetris::COLS
                        {
                            continue;
                        }
                        let mut board = Board::new();
                        let rejected =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                board.apply_piece_placement(TetrisPiecePlacement {
                                    piece,
                                    orientation,
                                })
                            }));
                        assert!(
                            rejected.is_err(),
                            "piece={piece}, orientation={orientation}"
                        );
                        assert_eq!(board, Board::new());
                    }
                }
            }
        }

        #[test]
        fn terminal_boards_ignore_fitting_and_out_of_bounds_placements() {
            let mut terminal = Board::new();
            terminal.fill();
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    for column in 0..StandardTetris::COLS {
                        let orientation = TetrisPieceOrientation {
                            rotation,
                            column: Column::new(column as u8),
                        };

                        let mut board = terminal;
                        let result = board
                            .apply_piece_placement(TetrisPiecePlacement { piece, orientation });
                        assert!(result.is_lost);
                        assert_eq!(result.lines_cleared, 0);
                        assert_eq!(board, terminal);
                    }
                }
            }
        }

        #[test]
        fn empty_board_has_no_occupied_cells() {
            let board = Board::new();
            assert_eq!(board.count(), 0);
            for row in 0..StandardTetris::ROWS {
                for col in 0..StandardTetris::COLS {
                    assert!(!board.get_bit(col, row));
                }
            }
        }

        #[test]
        fn setting_board_cells_increments_occupied_count() {
            let mut board = Board::new();
            for row in 0..StandardTetris::ROWS {
                for col in 0..StandardTetris::COLS {
                    board.set_bit(col, row);
                    assert!(board.get_bit(col, row), "col={col}, row={row}");
                    assert_eq!(board.count(), (row * StandardTetris::COLS + col + 1) as u32);
                }
            }
        }

        #[test]
        fn filling_a_board_sets_every_storage_bit_and_marks_it_lost() {
            let mut board = Board::new();
            board.fill();
            assert_eq!(
                board.as_limbs(),
                &[BackingColType::MAX; StandardTetris::COLS]
            );
            assert_eq!(
                board.count(),
                BackingColType::BITS * StandardTetris::COLS as u32
            );
            assert_eq!(board.height(), BackingColType::BITS);
            assert!(board.is_lost());
        }

        #[test]
        fn clearing_board_cells_removes_each_occupied_bit() {
            let mut board = Board::new();
            board.fill();
            for row in 0..constants::ACTUAL_ROWS {
                for col in 0..StandardTetris::COLS {
                    board.clear_bit(col, row);
                    assert!(!board.get_bit(col, row), "col={col}, row={row}");
                }
            }
            assert_eq!(board.count(), 0);
        }

        #[test]
        fn clearing_a_board_removes_playable_and_overflow_cells() {
            let mut board = Board::from_limbs([u32::MAX; StandardTetris::COLS]);
            board.clear();
            assert_eq!(board, Board::new());
        }

        #[test]
        fn height_matches_a_single_occupied_cell() {
            let mut board = Board::new();
            assert_eq!(board.height(), 0);
            for row in 0..StandardTetris::ROWS {
                for col in 0..StandardTetris::COLS {
                    board.clear();
                    board.set_bit(col, row);
                    let expected_height = (row + 1) as u32;
                    assert_eq!(board.height(), expected_height, "col={col}, row={row}");
                }
            }
        }

        #[test]
        fn dropping_an_o_fills_the_bottom_left_square() {
            let mut board = Board::new();
            board.apply_piece_placement(TetrisPiecePlacement {
                piece: TetrisPiece::O,
                orientation: TetrisPieceOrientation {
                    rotation: Rotation::R0,
                    column: crate::Column::new(0),
                },
            });
            assert_eq!(board.count(), 4);
            assert!(board.get_bit(0, 0));
            assert!(board.get_bit(1, 0));
            assert!(board.get_bit(0, 1));
            assert!(board.get_bit(1, 1));
        }

        #[test]
        fn row_collapse_supports_const_evaluation() {
            const CLEARED: (Board, u32) = {
                let mut columns = [0b1010; StandardTetris::COLS];
                columns[0] |= 0b10101;
                let mut board = Board::from_limbs(columns);
                let cleared = board.clear_and_collapse_rows();
                (board, cleared)
            };
            let mut expected = [0; StandardTetris::COLS];
            expected[0] = 0b111;
            assert_eq!(CLEARED.0.as_limbs(), &expected);
            assert_eq!(CLEARED.1, 2);
        }

        #[test]
        fn clearing_rows_collapses_survivors_in_order() {
            let mut board = Board::new();
            for row in [1, 3] {
                for col in 0..StandardTetris::COLS {
                    board.set_bit(col, row);
                }
            }
            for (col, row) in [
                (0, 0),
                (1, 2),
                (2, 4),
                (3, 6),
                (StandardTetris::COLS - 1, StandardTetris::ROWS),
            ] {
                board.set_bit(col, row);
            }
            assert!(board.is_lost());
            assert_eq!(board.clear_and_collapse_rows(), 2);

            let mut expected = Board::new();
            for (col, row) in [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 4),
                (StandardTetris::COLS - 1, StandardTetris::ROWS - 2),
            ] {
                expected.set_bit(col, row);
            }
            assert_eq!(board, expected);
            assert!(!board.is_lost());
            assert_eq!(board.clear_and_collapse_rows(), 0);
            assert_eq!(board, expected);
        }

        #[test]
        fn clearing_up_to_four_complete_rows_empties_otherwise_empty_boards() {
            fn check_rows(rows: &mut Vec<usize>, start: usize) {
                let mut board = Board::new();
                for &row in rows.iter() {
                    for col in 0..StandardTetris::COLS {
                        board.set_bit(col, row);
                    }
                }
                assert_eq!(
                    board.clear_and_collapse_rows(),
                    rows.len() as u32,
                    "rows={rows:?}"
                );
                assert_eq!(board, Board::new(), "rows={rows:?}");

                // Visit every row combination without allocating a separate list of combinations.
                if rows.len() == 4 {
                    return;
                }
                for row in start..StandardTetris::ROWS {
                    rows.push(row);
                    check_rows(rows, row + 1);
                    rows.pop();
                }
            }
            check_rows(&mut Vec::new(), 0);
        }

        #[test]
        fn loss_starts_above_the_playable_ceiling() {
            assert!(!Board::new().is_lost());
            for col in 0..StandardTetris::COLS {
                for row in 0..BackingColType::BITS as usize {
                    let mut board = Board::new();
                    board.set_bit(col, row);
                    assert_eq!(board.is_lost(), row >= StandardTetris::ROWS);
                    board.clear();
                    assert!(!board.is_lost());
                }
            }
        }

        #[test]
        fn row_collapse_matches_cellwise_reference_for_up_to_four_full_rows() {
            let mut rng = TetrisGameRng::new(0xc011a95e);
            let masks = [0, 1, 0xf, 0x0101_0101, 0x8000_0007, 0xa000_0005];
            let random_masks = (0..128u32).map(|case| {
                let mut mask = rng.next_u64() as u32;
                while mask.count_ones() > case % 5 {
                    mask &= mask - 1;
                }
                mask
            });
            let rotated_masks = (0..=u8::MAX)
                .filter(|mask| mask.count_ones() <= 4)
                .flat_map(|mask| {
                    (0..BackingColType::BITS)
                        .map(move |offset| (mask as BackingColType).rotate_left(offset))
                });
            for mask in masks.into_iter().chain(random_masks).chain(rotated_masks) {
                let columns: [u32; StandardTetris::COLS] = std::array::from_fn(|col| {
                    // Each non-full row has some occupied cells and some gaps.
                    let survivors = if col % 2 == 0 {
                        0x5555_5555
                    } else {
                        0xaaaa_aaaa
                    };
                    mask | survivors
                });
                let mut expected = [0; StandardTetris::COLS];
                let mut destination = 0;
                for row in 0..BackingColType::BITS {
                    if columns.iter().all(|col| col & (1 << row) != 0) {
                        continue;
                    }
                    for col in 0..StandardTetris::COLS {
                        expected[col] |= ((columns[col] >> row) & 1) << destination;
                    }
                    destination += 1;
                }
                let mut board = Board::from_limbs(columns);
                assert_eq!(board.clear_and_collapse_rows(), mask.count_ones());
                assert_eq!(board.as_limbs(), &expected, "full row mask={mask:#010x}");
                assert_eq!(board.clear_and_collapse_rows(), 0);
                assert_eq!(board.as_limbs(), &expected);
            }
        }

        #[test]
        fn clearing_four_rows_rescues_placements_above_the_ceiling() {
            // Clearing four rows must rescue a piece that initially extends above the ceiling.
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    let (shape, width, height) = reference_shape(piece, rotation);
                    for column in 0..=StandardTetris::COLS - width {
                        let mut columns = [0xf; StandardTetris::COLS];
                        let mut expected = [0; StandardTetris::COLS];
                        for i in 0..width {
                            columns[column + i] |= 1 << (StandardTetris::ROWS - 1);
                            expected[column + i] = (1 << (StandardTetris::ROWS - 1 - 4))
                                | (shape[i]
                                    >> (BackingColType::BITS as usize - StandardTetris::ROWS + 4
                                        - height));
                        }
                        let placement = TetrisPiecePlacement {
                            piece,
                            orientation: TetrisPieceOrientation {
                                rotation,
                                column: crate::Column::new(column as u8),
                            },
                        };
                        let mut board = Board::from_limbs(columns);
                        let result = board.apply_piece_placement(placement);
                        assert!(!result.is_lost, "placement={placement:?}");
                        assert_eq!(result.lines_cleared, 4);
                        assert_eq!(board.as_limbs(), &expected);
                    }
                }
            }
        }

        #[test]
        fn row_collapse_rejects_more_than_four_rows_before_mutating() {
            let masks = (5..=BackingColType::BITS)
                .map(|count| u32::MAX >> (BackingColType::BITS - count))
                .chain([0x8001_0103, 0x5555_5555, 0xaaaa_aaaa]);
            for mask in masks {
                let mut board = Board::from_limbs([mask; StandardTetris::COLS]);
                let before = board;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    board.clear_and_collapse_rows()
                }));
                assert!(result.is_err(), "full row mask={mask:#010x}");
                assert_eq!(
                    board, before,
                    "failed invariant must not partially clear the board"
                );
            }
        }

        #[test]
        fn hard_drops_match_cellwise_reference() {
            let mut rng = TetrisGameRng::new(0x5eed);
            let placements = TetrisPiece::all()
                .into_iter()
                .flat_map(|piece| {
                    Rotation::ALL.into_iter().flat_map(move |rotation| {
                        let (_, width, _) = reference_shape(piece, rotation);
                        (0..=StandardTetris::COLS - width).map(move |column| TetrisPiecePlacement {
                            piece,
                            orientation: TetrisPieceOrientation {
                                rotation,
                                column: crate::Column::new(column as u8),
                            },
                        })
                    })
                })
                .collect::<Vec<_>>();
            for case in 0..64 {
                let height = case % (StandardTetris::ROWS + 1);
                let mask = (1u32 << height) - 1;
                let mut columns = [0u32; StandardTetris::COLS];
                for column in &mut columns[..StandardTetris::COLS - 1] {
                    *column = rng.next_u64() as u32 & mask;
                }
                // An empty column prevents pre-existing full rows; a placement can
                // then complete at most four rows, as in reachable gameplay.
                for &placement in &placements {
                    let mut expected = columns;
                    let (mut shape, width, _) =
                        reference_shape(placement.piece, placement.orientation.rotation);
                    let start = placement.orientation.column.inner() as usize;
                    while shape[..width].iter().all(|col| col & 1 == 0)
                        && (0..width).all(|i| (shape[i] >> 1) & columns[start + i] == 0)
                    {
                        for col in &mut shape[..width] {
                            *col >>= 1;
                        }
                    }
                    for i in 0..width {
                        expected[start + i] |= shape[i];
                    }
                    let placed = expected;
                    expected.fill(0);
                    let mut destination = 0;
                    let mut cleared = 0;
                    for row in 0..BackingColType::BITS {
                        if placed.iter().all(|col| col & (1 << row) != 0) {
                            cleared += 1;
                        } else {
                            for i in 0..StandardTetris::COLS {
                                expected[i] |= ((placed[i] >> row) & 1) << destination;
                            }
                            destination += 1;
                        }
                    }
                    let lost = expected.iter().any(|col| col >> StandardTetris::ROWS != 0);
                    if lost {
                        expected.fill(u32::MAX);
                    }
                    let mut actual = <TetrisBoard<10, 20>>::from_limbs(columns);
                    let result = actual.apply_piece_placement(placement);
                    assert_eq!(actual.0, expected, "case={case}, placement={placement:?}");
                    assert_eq!(
                        result.lines_cleared, cleared,
                        "case={case}, placement={placement:?}"
                    );
                    assert_eq!(result.is_lost, lost, "case={case}, placement={placement:?}");
                }
            }
        }
    }

    mod board_encoding {
        use super::*;

        #[test]
        fn cell_arrays_round_trip_through_generic_signatures_using_board_size() {
            fn round_trip<const COLS: usize, const ROWS: usize>(
                cells: [u8; TetrisBoard::<COLS, ROWS>::BOARD_SIZE],
            ) -> [u8; TetrisBoard::<COLS, ROWS>::BOARD_SIZE] {
                TetrisBoard::<COLS, ROWS>::from_cell_array(cells, Major::Row)
                    .to_cell_array(Major::Row)
            }

            let cells = [1, 0, 1, 0, 1, 0];
            assert_eq!(round_trip::<3, 2>(cells), cells);
        }

        #[test]
        fn packed_bytes_round_trip_through_generic_signatures_using_board_size_of() {
            fn round_trip<const COLS: usize, const ROWS: usize>(
                bytes: [u8; size_of::<TetrisBoard<COLS, ROWS>>()],
            ) -> [u8; size_of::<TetrisBoard<COLS, ROWS>>()] {
                TetrisBoard::<COLS, ROWS>::from_bytes(bytes, Major::Row).to_bytes(Major::Row)
            }

            let mut bytes = [0; size_of::<TetrisBoard<3, 2>>()];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
            }
            assert_eq!(round_trip::<3, 2>(bytes), bytes);
        }

        #[test]
        fn fill_sets_all_backing_bits_independently_of_playable_geometry() {
            fn check<const COLS: usize, const ROWS: usize>() {
                let mut board = TetrisBoard::<COLS, ROWS>::new();
                board.fill();
                for column in 0..COLS {
                    for row in 0..constants::ACTUAL_ROWS {
                        assert!(board.get_bit(column, row));
                    }
                }
                assert_eq!(board.as_limbs(), &[BackingColType::MAX; COLS]);
            }
            check::<0, 0>();
            check::<2, 0>();
            check::<2, 1>();
            check::<10, 20>();
            check::<2, { constants::ACTUAL_ROWS - 1 }>();
            check::<2, { constants::ACTUAL_ROWS }>();
            check::<2, { constants::ACTUAL_ROWS + 1 }>();
        }

        #[test]
        fn conversions_preserve_rows_at_the_playable_height_limit() {
            type FullHeightBoard = TetrisBoard<2, { constants::ACTUAL_ROWS - 1 }>;
            const BOARD: FullHeightBoard = FullHeightBoard::from_limbs([u32::MAX, 1 << 31]);
            const CELLS: [u8; 2 * (constants::ACTUAL_ROWS - 1)] = BOARD.to_cell_array(Major::Row);
            const FROM_CELLS: FullHeightBoard = FullHeightBoard::from_cell_array(CELLS, Major::Row);
            const BYTES: [u8; 2 * std::mem::size_of::<u32>()] = BOARD.to_bytes(Major::Row);
            const FROM_BYTES: FullHeightBoard = FullHeightBoard::from_bytes(BYTES, Major::Row);
            let playable = FullHeightBoard::from_limbs([u32::MAX >> 1, 0]);
            assert_eq!(FROM_CELLS, playable);
            assert_eq!(FROM_BYTES, BOARD);
            assert_eq!(
                FullHeightBoard::from_board_id(&BOARD.to_board_id()),
                Some(playable)
            );
        }

        #[test]
        fn packed_row_bytes_match_geometry_at_widths_one_through_sixteen() {
            fn check<const COLS: usize, const ROWS: usize>()
            where
                [(); size_of::<TetrisBoard<COLS, ROWS>>()]:,
            {
                let mut limbs = [0; COLS];
                for (i, limb) in limbs.iter_mut().enumerate() {
                    *limb = 0x8123_4567u32.rotate_left(i as u32);
                }
                let board = TetrisBoard::<COLS, ROWS>::from_limbs(limbs);
                let mut expected = [0u8; size_of::<TetrisBoard<COLS, ROWS>>()];
                for (col, &limb) in limbs.iter().enumerate() {
                    for row in 0..BackingColType::BITS as usize {
                        let offset = row * COLS + col;
                        expected[offset / 8] |= (((limb >> row) & 1) as u8) << (offset % 8);
                    }
                }
                assert_eq!(board.to_bytes(Major::Row), expected, "width={}", COLS);
                assert_eq!(
                    TetrisBoard::<COLS, ROWS>::from_bytes(expected, Major::Row),
                    board
                );
            }
            check::<1, 20>();
            check::<2, 20>();
            check::<3, 20>();
            check::<4, 20>();
            check::<5, 20>();
            check::<6, 20>();
            check::<7, 20>();
            check::<8, 20>();
            check::<9, 20>();
            check::<10, 20>();
            check::<11, 20>();
            check::<12, 20>();
            check::<13, 20>();
            check::<14, 20>();
            check::<15, 20>();
            check::<16, 20>();
        }
        #[test]
        fn seventeen_column_metrics_match_plain_iteration() {
            let columns =
                core::array::from_fn::<_, 17, _>(|i| 0x1234_5678u32.rotate_left(i as u32));
            let board = TetrisBoard::<17, 20>::from_limbs(columns);
            let heights = columns.map(|column| BackingColType::BITS - column.leading_zeros());
            let holes = core::array::from_fn::<_, 17, _>(|i| heights[i] - columns[i].count_ones());
            assert_eq!(board.count(), columns.iter().map(|c| c.count_ones()).sum());
            assert_eq!(board.height(), *heights.iter().max().expect("nonempty"));
            assert_eq!(board.heights(), heights);
            assert_eq!(board.holes(), holes);
            assert_eq!(board.total_holes(), holes.iter().sum());
            assert_eq!(
                board.roughness(),
                heights.windows(2).map(|w| w[0].abs_diff(w[1])).sum()
            );
        }

        #[test]
        fn cell_arrays_preserve_each_playable_cell_in_both_layouts() {
            for row in 0..StandardTetris::ROWS {
                for col in 0..StandardTetris::COLS {
                    let mut board = Board::new();
                    board.set_bit(col, row);
                    for layout in [Major::Row, Major::Col] {
                        let index = match layout {
                            Major::Row => row * StandardTetris::COLS + col,
                            Major::Col => col * StandardTetris::ROWS + row,
                        };
                        let mut expected = [0; Board::BOARD_SIZE];
                        expected[index] = 1;
                        assert_eq!(
                            board.to_cell_array(layout),
                            expected,
                            "col={col}, row={row}, layout={layout:?}"
                        );
                        assert_eq!(
                            Board::from_cell_array(expected, layout),
                            board,
                            "col={col}, row={row}, layout={layout:?}"
                        );
                    }
                }
            }
        }

        #[test]
        fn cell_arrays_reject_every_nonbinary_byte() {
            for layout in [Major::Row, Major::Col] {
                for value in 2..=u8::MAX {
                    let mut cells = [0; Board::BOARD_SIZE];
                    cells[0] = value;
                    let rejected =
                        std::panic::catch_unwind(|| Board::from_cell_array(cells, layout));
                    assert!(rejected.is_err(), "value={value}, layout={layout:?}");
                }
            }
        }

        #[test]
        fn cell_arrays_validate_every_position_in_both_layouts() {
            for layout in [Major::Row, Major::Col] {
                for index in 0..Board::BOARD_SIZE {
                    for value in [2, u8::MAX] {
                        let mut cells = [0; Board::BOARD_SIZE];
                        cells[index] = value;
                        let rejected =
                            std::panic::catch_unwind(|| Board::from_cell_array(cells, layout));
                        assert!(
                            rejected.is_err(),
                            "index={index}, value={value}, layout={layout:?}"
                        );
                    }
                }
            }
        }

        #[test]
        fn cell_array_construction_is_const_in_both_layouts() {
            type SmallBoard = TetrisBoard<2, 3>;
            const ROW_MAJOR: SmallBoard =
                SmallBoard::from_cell_array([1, 0, 0, 1, 1, 0], Major::Row);
            const COL_MAJOR: SmallBoard =
                SmallBoard::from_cell_array([1, 0, 0, 1, 1, 0], Major::Col);
            assert_eq!(ROW_MAJOR.as_limbs(), &[0b101, 0b010]);
            assert_eq!(COL_MAJOR.as_limbs(), &[0b001, 0b011]);
        }

        #[test]
        fn packed_board_bytes_preserve_every_backing_cell_in_both_layouts() {
            for row in 0..constants::ACTUAL_ROWS {
                for col in 0..StandardTetris::COLS {
                    let mut board = Board::new();
                    board.set_bit(col, row);
                    let mut row_bytes = [0u8; std::mem::size_of::<Board>()];
                    let offset = row * StandardTetris::COLS + col;
                    row_bytes[offset / 8] = 1 << (offset % 8);
                    assert_eq!(
                        board.to_bytes(Major::Row),
                        row_bytes,
                        "col={col}, row={row}"
                    );
                    assert_eq!(
                        Board::from_bytes(row_bytes, Major::Row),
                        board,
                        "col={col}, row={row}"
                    );
                    assert_eq!(
                        Board::from_bytes(board.to_bytes(Major::Col), Major::Col),
                        board,
                        "col={col}, row={row}"
                    );
                }
            }
        }

        #[test]
        fn board_id_rejects_nonzero_final_padding_bits() {
            type SingleCell = TetrisBoard<1, 1>;
            let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
            for (value, &letter) in alphabet.iter().enumerate() {
                let id = (letter as char).to_string();
                let decoded = SingleCell::from_board_id(&id);
                if value & 0b01111 == 0 {
                    assert_eq!(
                        decoded.map(|b| b.count()),
                        Some((value >> 4) as u32),
                        "id={id}"
                    );
                } else {
                    assert!(decoded.is_none(), "accepted nonzero padding in {id}");
                }
            }
            assert!(SingleCell::from_board_id("QQ").is_none());
        }

        #[test]
        fn board_ids_round_trip_every_small_board_with_partial_chunks() {
            fn check<const COLS: usize, const ROWS: usize>()
            where
                [(); TetrisBoard::<COLS, ROWS>::BOARD_ID_SIZE]:,
            {
                for mask in 0..1u64 << TetrisBoard::<COLS, ROWS>::BOARD_SIZE {
                    let mut board = TetrisBoard::<COLS, ROWS>::new();
                    for bit in 0..TetrisBoard::<COLS, ROWS>::BOARD_SIZE {
                        if mask & (1 << bit) != 0 {
                            board.set_bit(bit % COLS, bit / COLS);
                        }
                    }
                    let encoded = board.to_board_id();
                    assert_eq!(
                        TetrisBoard::<COLS, ROWS>::from_board_id(&encoded),
                        Some(board),
                        "cols={}, rows={}, mask={mask:#x}, id={encoded}",
                        COLS,
                        ROWS
                    );
                }
            }
            // Cover every possible number of padding bits, plus a multi-character board.
            check::<1, 1>();
            check::<1, 2>();
            check::<1, 3>();
            check::<1, 4>();
            check::<1, 5>();
            check::<2, 3>();
            check::<3, 3>();
        }

        #[test]
        fn board_id_preserves_cells_in_a_partial_final_chunk() {
            type SmallBoard = TetrisBoard<2, 3>;
            let mut board = SmallBoard::new();
            // The sixth cell belongs to a second, partially occupied base32 character.
            board.set_bit(1, 2);
            assert_eq!(board.to_board_id(), "AQ");
            assert_eq!(SmallBoard::from_board_id("AQ"), Some(board));
        }

        #[test]
        fn native_byte_conversions_preserve_column_storage() {
            let board = Board::from_limbs(METRIC_COLUMNS);
            let bytes = board.to_bytes(Major::Col);
            let expected: Vec<_> = METRIC_COLUMNS
                .iter()
                .flat_map(|col| col.to_ne_bytes())
                .collect();
            assert_eq!(bytes.as_slice(), expected);
            assert_eq!(Board::from_bytes(bytes, Major::Col), board);
        }

        #[test]
        fn cell_arrays_round_trip_in_both_layouts() {
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            for major in [Major::Row, Major::Col] {
                // Empty board
                let board = Board::new();
                let cells = board.to_cell_array(major);
                assert_eq!(cells.len(), Board::BOARD_SIZE);
                assert_eq!(board, Board::from_cell_array(cells, major));

                // Random boards
                for _ in 0..100 {
                    let mut board = Board::new();
                    board.set_random_bits(1024, &mut rng);
                    let cells = board.to_cell_array(major);
                    assert_eq!(board, Board::from_cell_array(cells, major));
                }
            }
        }

        #[test]
        fn board_id_preserves_the_known_standard_board_encoding() {
            let mut board = Board::new();
            for column in [1, 2, 3, 4, 5, 7, 8, 9] {
                board.set_bit(column, 0);
            }
            for column in [3, 4, 7, 8, 9] {
                board.set_bit(column, 1);
            }
            board.set_bit(7, 2);
            assert_eq!(board.to_board_id(), "PXDHAE");
        }

        #[test]
        fn board_ids_round_trip() {
            const NUM_BOARDS: usize = 1000;
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            for _ in 0..NUM_BOARDS {
                let limbs: [u32; StandardTetris::COLS] = std::array::from_fn(|_| {
                    rng.random::<u32>() & ((1 << StandardTetris::ROWS) - 1)
                });
                let board = Board::from_limbs(limbs);
                let id = board.to_board_id();
                let reconstructed =
                    Board::from_board_id(&id).expect("valid board ID should decode");
                assert_eq!(board, reconstructed, "board ID roundtrip failed for {id}");
            }
        }

        #[test]
        fn empty_board_id_round_trips_as_an_empty_string() {
            let board = Board::new();
            assert_eq!(board.to_board_id(), "");
            let decoded = Board::from_board_id("").expect("empty string is valid");
            assert_eq!(decoded, board);
        }

        #[test]
        fn board_id_rejects_invalid_characters_and_excess_length() {
            assert!(Board::from_board_id("!!!").is_none());
            assert!(Board::from_board_id("01").is_none());
            let long = "A".repeat(Board::BOARD_SIZE / 5 + 1);
            assert!(Board::from_board_id(&long).is_none());
        }

        #[test]
        fn column_major_bytes_round_trip() {
            const NUM_BOARDS: usize = 1000;
            let mut rng = SmallRng::seed_from_u64(0x5eed);

            for _ in 0..NUM_BOARDS {
                let limbs: [u32; StandardTetris::COLS] = std::array::from_fn(|_| rng.random());
                let board = Board::from_limbs(limbs);

                let bytes = board.to_bytes(Major::Col);
                let reconstructed = Board::from_bytes(bytes, Major::Col);

                assert_eq!(
                    board, reconstructed,
                    "Board should roundtrip correctly through bytes"
                );
            }
        }

        #[test]
        fn row_major_bytes_round_trip() {
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            for _ in 0..1000 {
                // Only set bits in the valid row range (0..ROWS) to avoid padding issues
                let limbs: [u32; StandardTetris::COLS] = std::array::from_fn(|_| {
                    rng.random::<u32>() & ((1 << StandardTetris::ROWS) - 1)
                });
                let board = Board::from_limbs(limbs);

                let bytes = board.to_bytes(Major::Row);
                let reconstructed = Board::from_bytes(bytes, Major::Row);
                assert_eq!(board, reconstructed, "Row-major bytes roundtrip failed");
            }
        }

        // Compare against cell arrays to catch layout bugs that a round-trip could hide.
        #[test]
        fn row_major_byte_encoding_matches_cell_array_bits() {
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            for _ in 0..1000 {
                let limbs: [u32; StandardTetris::COLS] = std::array::from_fn(|_| {
                    rng.random::<u32>() & ((1 << StandardTetris::ROWS) - 1)
                });
                let board = Board::from_limbs(limbs);

                let row_bytes = board.to_bytes(Major::Row);
                let row_cells = board.to_cell_array(Major::Row);

                // Cell (row, column) occupies bit row * COLS + column.
                for r in 0..StandardTetris::ROWS {
                    for c in 0..StandardTetris::COLS {
                        let bit_pos = r * StandardTetris::COLS + c;
                        let from_bytes = (row_bytes[bit_pos / 8] >> (bit_pos % 8)) & 1;
                        let from_cells = row_cells[bit_pos];
                        assert_eq!(
                            from_bytes, from_cells,
                            "mismatch at (r={r}, c={c}): bytes={from_bytes}, cells={from_cells}"
                        );
                    }
                }

                // Rows above the playable area are zero in these fixtures.
                for r in StandardTetris::ROWS..constants::ACTUAL_ROWS {
                    for c in 0..StandardTetris::COLS {
                        let bit_pos = r * StandardTetris::COLS + c;
                        let from_bytes = (row_bytes[bit_pos / 8] >> (bit_pos % 8)) & 1;
                        assert_eq!(from_bytes, 0, "padding bit set at (r={r}, c={c})");
                    }
                }
            }
        }

        // Pack cells independently of the production byte encoder.
        #[test]
        fn row_major_byte_decoding_matches_manually_packed_cells() {
            let mut rng = SmallRng::seed_from_u64(0x5eed);
            for _ in 0..1000 {
                let limbs: [u32; StandardTetris::COLS] = std::array::from_fn(|_| {
                    rng.random::<u32>() & ((1 << StandardTetris::ROWS) - 1)
                });
                let board = Board::from_limbs(limbs);

                // The unused rows remain zero because the output buffer starts empty.
                let row_cells = board.to_cell_array(Major::Row);
                let mut row_bytes = [0u8; std::mem::size_of::<Board>()];
                for r in 0..StandardTetris::ROWS {
                    for c in 0..StandardTetris::COLS {
                        let bit_pos = r * StandardTetris::COLS + c;
                        row_bytes[bit_pos / 8] |= row_cells[bit_pos] << (bit_pos % 8);
                    }
                }

                let decoded = Board::from_bytes(row_bytes, Major::Row);
                assert_eq!(
                    decoded, board,
                    "from_bytes(Row) of manually packed cells must reconstruct the board"
                );
            }
        }
    }

    mod rng {
        use super::*;

        #[test]
        fn default_matches_seed_5eed() {
            assert_eq!(TetrisGameRng::default(), TetrisGameRng::new(0x5EED));
        }

        #[test]
        #[should_panic(expected = "RNG seed must be nonzero")]
        fn rejects_zero_seed() {
            #[allow(
                clippy::let_underscore_must_use,
                reason = "The zero seed constructor must panic before returning a value."
            )]
            let _ = TetrisGameRng::new(0);
        }

        #[test]
        fn zero_reseed_is_rejected_without_mutation() {
            let mut rng = TetrisGameRng::new(42);
            rng.next_u64();
            let before = rng;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rng.reseed(0)));
            assert!(rejected.is_err());
            assert_eq!(rng, before);
        }

        #[test]
        fn next_values_and_draw_counts_match_known_vectors() {
            let cases = [
                (
                    1,
                    [
                        12994781566227106604,
                        9038243705893100514,
                        6536476596838423914,
                        16455181681707637869,
                    ],
                ),
                (
                    42,
                    [
                        9297814886316923340,
                        3534707431943866523,
                        10870387891257067482,
                        16542507315027075359,
                    ],
                ),
                (
                    u64::MAX,
                    [
                        7256831767414464289,
                        18041524855838566013,
                        1925008239895493462,
                        11358393048657778147,
                    ],
                ),
            ];
            for (seed, expected) in cases {
                let mut rng = TetrisGameRng::new(seed);
                assert_eq!(rng.n, 0);
                for (index, value) in expected.into_iter().enumerate() {
                    assert_eq!(rng.next_u64(), value, "seed={seed}, index={index}");
                    assert_eq!(rng.n, index as u64 + 1);
                }
            }
        }

        #[test]
        fn peek_matches_advancing_a_copy_without_changing_state() {
            for seed in [1, 42, u64::MAX] {
                let rng = TetrisGameRng::new(seed);
                for offset in 0..20 {
                    let mut expected = rng;
                    for _ in 0..offset {
                        expected.next_u64();
                    }
                    assert_eq!(rng.peek_n(offset), expected.state);
                    assert_eq!(rng, TetrisGameRng::new(seed));
                }
            }
        }

        #[test]
        fn reset_replays_the_original_seed() {
            for seed in [1, 42, u64::MAX] {
                let mut rng = TetrisGameRng::new(seed);
                rng.next_u64();
                rng.next_u64();
                rng.reset();
                assert_eq!(rng, TetrisGameRng::new(seed));
            }
        }

        #[test]
        fn reseed_matches_a_fresh_generator() {
            for seed in [1, 42, u64::MAX] {
                let mut rng = TetrisGameRng::new(seed);
                rng.next_u64();
                rng.reseed(17);
                assert_eq!(rng, TetrisGameRng::new(17));
            }
        }

        #[test]
        fn murmur_finalizer_matches_documented_value() {
            let mut x = 12345u64;
            TetrisGameRng::fmix64(&mut x);
            assert_eq!(x, 1716623506685013753u64);
        }
    }

    mod game {
        use super::*;

        #[test]
        fn both_constructors_accept_the_headroom_limit() {
            const ROWS: usize = Board::MAX_PLAYABLE_ROWS;
            type TallGame = TetrisGame<TestDimensions<4, ROWS>>;

            let seeded = TallGame::new_with_seed(42);
            assert_eq!(seeded.board, TetrisBoard::<4, ROWS>::new());
            let scenario = TallGame::new_with_board_bag_piece_seeded(
                TetrisBoard::<4, ROWS>::new(),
                TetrisPieceBagState::full(),
                TetrisPiece::I,
                42,
            );
            assert_eq!(scenario.current_piece(), TetrisPiece::I);
            assert_eq!(scenario.board, seeded.board);
        }

        #[test]
        fn loss_after_a_line_clear_preserves_the_last_successful_statistics() {
            let mut columns = [1 << (StandardTetris::ROWS - 1); StandardTetris::COLS];
            columns[StandardTetris::COLS - 1] = 1 << (StandardTetris::ROWS - 2);
            let mut game = Game::new_with_board_bag_piece_seeded(
                Board::from_limbs(columns),
                TetrisPieceBagState::full(),
                TetrisPiece::I,
                42,
            );
            game.lines_cleared = 7;
            game.recent_lines_cleared = 2;
            game.piece_count = 9;
            let before = game;
            let placement = *game
                .current_placements()
                .last()
                .expect("rightmost vertical I");
            let mut expected_board = game.board;
            let board_result = expected_board.apply_piece_placement(placement);
            assert!(board_result.is_lost);
            assert_eq!(board_result.lines_cleared, 1);
            // Game statistics count successful moves; losing moves return zero lines.
            let result = game.apply_placement(placement);
            assert!(result.is_lost);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(game.board, expected_board);
            game.board = before.board;
            assert_eq!(game, before);
        }

        #[test]
        fn vertical_i_above_a_28_row_stack_loses() {
            type TallGame = TetrisGame<TestDimensions<10, 28>>;
            let mut game = TallGame::new_with_board_bag_piece_seeded(
                TetrisBoard::<10, 28>::from_limbs([0x0fff_ffff, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                TetrisPieceBagState::full(),
                TetrisPiece::I,
                42,
            );
            let result = game.apply_orientation(crate::TetrisPieceOrientation {
                rotation: Rotation::R90,
                column: crate::Column::new(0),
            });
            assert!(result.is_lost);
            assert_eq!(game.board.as_limbs(), &[u32::MAX; 10]);
        }

        #[test]
        fn display_includes_its_current_piece_bag_board_and_line_count() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            game.lines_cleared = 7;
            game.board.set_bit(2, 3);
            let expected = format!(
                "TetrisGame:\n  bag:\n    []\n  current_piece:\n    O\n  lines_cleared:\n    7\n  board:\n    {}\n",
                game.board
            );
            assert_eq!(game.to_string(), expected);
        }

        #[test]
        fn out_of_bounds_game_placement_is_rejected_without_mutation() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            let before = game;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                game.apply_orientation(crate::TetrisPieceOrientation {
                    rotation: Rotation::R0,
                    column: crate::Column::new(OnlyO::COLS as u8 - 1),
                });
            }));
            assert!(rejected.is_err());
            assert_eq!(game, before);
        }

        #[test]
        fn terminal_game_ignores_further_placements_without_drawing() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            game.board = TetrisBoard::<4, 20>::from_limbs([u32::MAX; OnlyO::COLS]);
            let before = game;
            let result = game.apply_orientation(crate::TetrisPieceOrientation {
                rotation: Rotation::R0,
                column: crate::Column::new(OnlyO::COLS as u8 - 1),
            });
            assert!(result.is_lost);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(game, before);
        }

        #[test]
        fn a_losing_placement_does_not_draw_or_change_statistics() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            game.board =
                TetrisBoard::<4, 20>::from_limbs([((1 << StandardTetris::ROWS) - 1), 0, 0, 0]);
            game.lines_cleared = 7;
            game.recent_lines_cleared = 2;
            game.piece_count = 9;
            let before = game;
            let result = game.apply_placement(game.current_placements()[0]);
            assert!(result.is_lost);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(game.board.as_limbs(), &[u32::MAX; OnlyO::COLS]);
            // The board becomes terminal; all other game state stays at the last successful move.
            game.board = before.board;
            assert_eq!(game, before);
        }

        #[test]
        fn a_placement_without_lines_cleared_resets_recent_count() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            game.lines_cleared = 9;
            game.recent_lines_cleared = 2;
            let result = game.apply_placement(game.current_placements()[0]);
            assert!(!result.is_lost);
            assert_eq!(result.lines_cleared, 0);
            assert_eq!(game.lines_cleared, 9);
            assert_eq!(game.recent_lines_cleared, 0);
        }

        #[test]
        fn line_clears_update_total_recent_and_piece_counts() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            // Complete two rows by dropping O into their two-column gap.
            game.board = TetrisBoard::<4, 20>::from_limbs([0, 0, 0b11, 0b11]);
            game.lines_cleared = 7;
            game.recent_lines_cleared = 1;
            game.piece_count = 9;
            let result = game.apply_placement(game.current_placements()[0]);
            assert!(!result.is_lost);
            assert_eq!(result.lines_cleared, 2);
            assert_eq!(game.lines_cleared, 9);
            assert_eq!(game.recent_lines_cleared, 2);
            assert_eq!(game.piece_count, 10);
            assert_eq!(game.board, TetrisBoard::<4, 20>::new());
        }

        // Exercise piece draws without a placement strategy or board loss affecting the fixture.
        fn draw_from_game_bag<C: TetrisGameConfig>(game: &mut TetrisGame<C>) -> TetrisPiece
        where
            [(); C::COLS]:,
            [(); C::ROWS]:,
        {
            TetrisGame::<C>::draw_piece(&mut game.bag, &mut game.rng)
        }

        fn assert_game_draws_set<C: TetrisGameConfig>(game: &mut TetrisGame<C>, expected: u8)
        where
            [(); C::COLS]:,
            [(); C::ROWS]:,
        {
            let mut seen = 0;
            for drawn in 0..expected.count_ones() {
                let piece = draw_from_game_bag(game);
                assert_eq!(game.bag.count() as u32, expected.count_ones() - drawn - 1);
                assert_ne!(expected & piece.mask(), 0);
                assert_eq!(seen & piece.mask(), 0);
                seen |= piece.mask();
            }
            assert_eq!(seen, expected);
            assert_eq!(game.bag.count(), 0);
        }

        #[test]
        fn reset_matches_a_fresh_game_for_original_and_new_seeds() {
            fn check<C: TetrisGameConfig>()
            where
                [(); C::COLS]:,
                [(); C::ROWS]:,
            {
                // Leave a two-column gap so an O clears two lines and changes every statistic.
                let mut board = TetrisBoard::<{ C::COLS }, { C::ROWS }>::new();
                for col in 2..C::COLS {
                    board.set_bit(col, 0);
                    board.set_bit(col, 1);
                }
                for new_seed in [None, Some(99)] {
                    let mut game = TetrisGame::<C>::new_with_seed(42);
                    game.board = board;
                    game.set_bag_piece_seeded(C::PIECE_SET, TetrisPiece::O, 42);
                    assert_eq!(
                        game.apply_placement(
                            crate::TetrisPiecePlacement::<{ C::COLS }>::from_index(0)
                        )
                        .lines_cleared,
                        2
                    );
                    assert_eq!(game.recent_lines_cleared, 2);
                    game.reset(new_seed);
                    let mut expected = TetrisGame::<C>::new_with_seed(new_seed.unwrap_or(42));
                    assert_eq!(game, expected);
                    for _ in 0..21 {
                        assert_eq!(
                            draw_from_game_bag(&mut game),
                            draw_from_game_bag(&mut expected)
                        );
                    }
                }
            }
            check::<StandardTetris>();
            check::<TestPieceSet<3>>();
            check::<OnlyO>();
        }

        #[test]
        fn constructors_draw_only_configured_pieces() {
            for mut game in [
                TetrisGame::<OnlyO>::new(),
                TetrisGame::<OnlyO>::default(),
                TetrisGame::<OnlyO>::new_with_seed(42),
            ] {
                assert_eq!(game.current_piece, TetrisPiece::O);
                for _ in 0..28 {
                    assert_eq!(draw_from_game_bag(&mut game), TetrisPiece::O);
                }
            }
        }

        #[test]
        fn current_placements_fit_the_configured_width() {
            macro_rules! check {
                ($config:ty) => {{
                    for piece in <$config as TetrisGameConfig>::PIECE_SET.iter_pieces() {
                        let mut game = TetrisGame::<$config>::new_with_seed(42);
                        // Check every enabled piece without advancing the game.
                        game.current_piece = piece;
                        let placements = game.current_placements();
                        let mut expected_count = 0;
                        for rotation in Rotation::ALL
                            .into_iter()
                            .take(game.current_piece.num_rotations() as usize)
                        {
                            let (_, width, _) = reference_shape(game.current_piece, rotation);
                            expected_count +=
                                <$config as TetrisGameConfig>::COLS.saturating_sub(width - 1);
                        }
                        assert_eq!(placements.len(), expected_count);
                        for placement in placements {
                            let (_, width, _) =
                                reference_shape(placement.piece, placement.orientation.rotation);
                            assert_eq!(placement.piece, game.current_piece);
                            assert!(
                                usize::from(placement.orientation.column.inner()) + width
                                    <= <$config as TetrisGameConfig>::COLS
                            );
                        }
                    }
                }};
            }
            check!(TestWidth<2>);
            check!(TestWidth<4>);
            check!(TestWidth<10>);
            check!(OnlyI);
        }

        #[test]
        fn zero_seed_scenario_update_preserves_the_game() {
            let mut game = Game::new_with_seed(42);
            let before = game;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                game.set_bag_piece_seeded(TetrisPieceBagState::full(), TetrisPiece::I, 0);
            }));
            assert!(rejected.is_err());
            assert_eq!(game, before);
        }

        #[test]
        fn zero_seed_reset_preserves_the_game() {
            let mut game = Game::new_with_seed(42);
            let before = game;
            let rejected =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| game.reset(Some(0))));
            assert!(rejected.is_err());
            assert_eq!(game, before);
        }

        #[test]
        fn narrow_game_reset_preserves_its_piece_configuration() {
            let mut game = TetrisGame::<OnlyI>::new_with_seed(42);
            game.reset(Some(43));
            assert_eq!(game, TetrisGame::<OnlyI>::new_with_seed(43));
        }

        #[test]
        fn narrow_game_scenario_update_keeps_drawing_configured_pieces() {
            let mut game = TetrisGame::<OnlyI>::new_with_seed(42);
            game.set_bag_piece_seeded(TetrisPieceBagState::from_pieces([]), TetrisPiece::I, 44);
            for n in 0..21 {
                assert_eq!(game.iter_next_pieces().nth(n), Some(TetrisPiece::I));
                assert_eq!(draw_from_game_bag(&mut game), TetrisPiece::I);
            }
        }

        #[test]
        fn iter_next_pieces_preserves_snapshots_across_partial_bags_and_refills() {
            fn check<C: TetrisGameConfig>()
            where
                [(); C::COLS]:,
                [(); C::ROWS]:,
            {
                for mask in 0..=TetrisPieceBagState::FULL_MASK {
                    if mask & !u8::from(C::PIECE_SET) != 0 {
                        continue;
                    }
                    let game = TetrisGame::<C>::new_with_board_bag_piece_seeded(
                        TetrisBoard::<{ C::COLS }, { C::ROWS }>::new(),
                        TetrisPieceBagState::try_from(mask).expect("valid bag mask"),
                        C::PIECE_SET.iter_pieces().next().expect("nonempty set"),
                        42,
                    );
                    let before = game;
                    let mut preview = game.iter_next_pieces();
                    assert_eq!(
                        preview.next(),
                        Some(game.current_piece),
                        "config={}, remaining={mask:#09b}, seed=42, offset=0",
                        std::any::type_name::<C>()
                    );
                    assert_eq!(
                        game,
                        before,
                        "config={}, remaining={mask:#09b}, seed=42",
                        std::any::type_name::<C>()
                    );
                    let mut expected = game;
                    for offset in 1..=28 {
                        assert_eq!(
                            preview.next(),
                            Some(draw_from_game_bag(&mut expected)),
                            "config={}, remaining={mask:#09b}, seed=42, offset={offset}",
                            std::any::type_name::<C>()
                        );
                        assert_eq!(
                            game,
                            before,
                            "config={}, remaining={mask:#09b}, seed=42",
                            std::any::type_name::<C>()
                        );
                    }
                }
            }
            check::<StandardTetris>();
            check::<OnlyO>();
            check::<OnlyI>();
            check::<TestPieceSet<3>>();
        }

        #[test]
        fn iter_next_pieces_owns_a_snapshot_independent_of_later_game_changes() {
            let mut game = Game::new_with_seed(42);
            let mut expected = game;
            let mut preview = game.iter_next_pieces();
            // The iterator does not borrow the game, and resetting cannot alter its sequence.
            game.reset(Some(99));
            assert_eq!(preview.next(), Some(expected.current_piece));
            for _ in 0..56 {
                assert_eq!(preview.next(), Some(draw_from_game_bag(&mut expected)));
            }
        }

        #[test]
        fn iter_next_pieces_nth_matches_sequential_draws() {
            for seed in [1, 42, u64::MAX] {
                let game = Game::new_with_seed(seed);
                let before = game;
                let mut bag = game.bag;
                let mut rng = game.rng;
                assert_eq!(game.iter_next_pieces().next(), Some(game.current_piece));
                for offset in 1..=28 {
                    assert_eq!(
                        game.iter_next_pieces().nth(offset),
                        Some(Game::draw_piece(&mut bag, &mut rng))
                    );
                    assert_eq!(game, before);
                }
            }
        }

        #[test]
        fn default_matches_new() {
            assert_eq!(Game::default(), Game::new());
        }

        #[test]
        fn applying_an_orientation_matches_applying_its_placement() {
            const MOVES: usize = 16;
            let mut by_orientation = Game::new_with_seed(42);
            let mut by_placement = by_orientation;
            let initial_rng_draws = by_orientation.rng.n;
            for step in 0..MOVES {
                // Keep the piece stream advancing without a stacking policy causing an early loss.
                by_orientation.board.clear();
                by_placement.board.clear();
                let placements = by_orientation.current_placements();
                let action = placements[step % placements.len()];
                let result = by_orientation.apply_orientation(action.orientation);
                assert_eq!(
                    result,
                    by_placement.apply_placement(action),
                    "step={step}, placement={action:?}"
                );
                assert!(!result.is_lost, "step={step}, placement={action:?}");
                assert_eq!(by_orientation, by_placement, "step={step}");
            }
            assert_eq!(by_orientation.piece_count, MOVES as u32);
            assert_eq!(
                by_orientation.rng.n,
                initial_rng_draws + (MOVES / constants::NUM_TETRIS_PIECES) as u64
            );
        }

        #[test]
        fn configured_placement_catalog_matches_independent_geometry_at_every_width() {
            fn check<const COLS: usize>() {
                let placements = crate::TetrisPiecePlacement::<COLS>::ALL_PLACEMENTS;
                let starts = &crate::TetrisPiecePlacement::<COLS>::PIECE_START_INDICES;
                assert_eq!(starts[0], 0, "cols={COLS}");
                let mut expected = Vec::new();
                let mut largest_piece_count = 0;
                for piece in TetrisPiece::all() {
                    let start = expected.len();
                    assert_eq!(
                        starts[piece.index() as usize],
                        start,
                        "cols={COLS}, piece={piece}"
                    );
                    for rotation in Rotation::ALL
                        .into_iter()
                        .take(piece.num_rotations() as usize)
                    {
                        let (_, width, _) = reference_shape(piece, rotation);
                        if width > COLS {
                            continue;
                        }
                        for col in 0..=COLS - width {
                            expected.push(crate::TetrisPiecePlacement::<COLS> {
                                piece,
                                orientation: crate::TetrisPieceOrientation::<COLS> {
                                    rotation,
                                    column: crate::Column::new(col as u8),
                                },
                            });
                        }
                    }
                    let end = expected.len();
                    assert_eq!(
                        starts[piece.index() as usize + 1],
                        end,
                        "cols={COLS}, piece={piece}"
                    );
                    assert_eq!(
                        crate::TetrisPiecePlacement::<COLS>::all_from_piece(piece),
                        &expected[start..end],
                        "cols={COLS}, piece={piece}"
                    );
                    largest_piece_count = largest_piece_count.max(end - start);
                }
                assert_eq!(placements, expected.as_slice(), "cols={COLS}");
                assert_eq!(
                    crate::TetrisPiecePlacement::<COLS>::NUM_PLACEMENTS,
                    expected.len(),
                    "cols={COLS}"
                );
                assert_eq!(
                    crate::TetrisPiecePlacement::<COLS>::MAX_PIECE_PLACEMENT_COUNT,
                    largest_piece_count,
                    "cols={COLS}"
                );
            }
            check::<1>();
            check::<2>();
            check::<3>();
            check::<4>();
            check::<5>();
            check::<6>();
            check::<7>();
            check::<8>();
            check::<9>();
            check::<10>();
            check::<11>();
            check::<12>();
            check::<13>();
            check::<14>();
        }

        #[test]
        fn scenario_constructor_rejects_an_unconfigured_current_piece() {
            let rejected = std::panic::catch_unwind(|| {
                TetrisGame::<OnlyO>::new_with_board_bag_piece_seeded(
                    TetrisBoard::<4, 20>::new(),
                    OnlyO::PIECE_SET,
                    TetrisPiece::I,
                    42,
                )
            });
            assert!(rejected.is_err());
        }

        #[test]
        fn scenario_update_rejects_an_unconfigured_piece_without_mutation() {
            let mut game = TetrisGame::<OnlyO>::new_with_seed(42);
            let before = game;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                game.set_bag_piece_seeded(OnlyO::PIECE_SET, TetrisPiece::I, 123);
            }));
            assert!(rejected.is_err());
            assert_eq!(game, before);
        }

        #[test]
        fn scenario_constructor_rejects_an_unconfigured_remainder() {
            let rejected = std::panic::catch_unwind(|| {
                TetrisGame::<OnlyO>::new_with_board_bag_piece_seeded(
                    TetrisBoard::<4, 20>::new(),
                    TetrisPieceBagState::from_pieces([TetrisPiece::I]),
                    TetrisPiece::O,
                    42,
                )
            });
            assert!(rejected.is_err());
        }

        #[test]
        fn display_propagates_write_errors() {
            let mut game = Game::new_with_seed(42);
            game.board.set_bit(0, 0);
            game.board.set_bit(1, 2);
            assert_display_propagates_write_errors(&game);
        }

        #[test]
        fn wrong_piece_is_rejected_before_game_mutation() {
            let mut game = Game::new_with_seed(123);
            let before = game;
            let wrong_piece = TetrisPiece::all()
                .into_iter()
                .find(|piece| *piece != game.current_piece)
                .expect("another piece exists");
            let placement = TetrisPiecePlacement::all_from_piece(wrong_piece)[0];
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                game.apply_placement(placement)
            }));
            assert!(rejected.is_err());
            assert_eq!(game, before);
        }

        #[test]
        fn refills_every_custom_piece_set_after_every_valid_remainder() {
            fn check<const MASK: u8>()
            where
                [(); TestPieceSet::<MASK>::COLS]:,
                [(); TestPieceSet::<MASK>::ROWS]:,
            {
                let mask = MASK;
                let set = TestPieceSet::<MASK>::PIECE_SET;
                let piece = set.iter_pieces().next().expect("nonempty set");
                for remaining in 0..=TetrisPieceBagState::FULL_MASK {
                    if remaining & !mask != 0 {
                        continue;
                    }
                    let mut game = TetrisGame::<TestPieceSet<MASK>>::new_with_seed(42);
                    game.set_bag_piece_seeded(
                        TetrisPieceBagState::try_from(remaining).expect("valid mask"),
                        piece,
                        123,
                    );
                    // Independent unpacked reference for the pre-refactor shuffle and RNG use.
                    let mut reference_rng = TetrisGameRng::new(123);
                    for (refill, current_mask) in
                        [remaining, mask, mask, mask].into_iter().enumerate()
                    {
                        let mut pieces: Vec<_> = TetrisPiece::all()
                            .into_iter()
                            .filter(|piece| current_mask & piece.mask() != 0)
                            .collect();
                        let word = reference_rng.next_u64();
                        for index in 0..pieces.len().saturating_sub(1) {
                            let byte = ((word >> (index * 8)) & 255) as usize;
                            let next = index + ((byte * (pieces.len() - index)) >> 8);
                            pieces.swap(index, next);
                        }
                        for (index, piece) in pieces.iter().enumerate() {
                            assert_eq!(
                                draw_from_game_bag(&mut game),
                                *piece,
                                "seed=123, piece_set={mask:#09b}, remaining={remaining:#09b}, refill={refill}, position={index}"
                            );
                            assert_eq!(
                                game.bag.count() as usize,
                                pieces.len() - index - 1,
                                "seed=123, piece_set={mask:#09b}, remaining={remaining:#09b}, refill={refill}, position={index}"
                            );
                        }
                        assert_eq!(
                            game.bag.count(),
                            0,
                            "seed=123, piece_set={mask:#09b}, remaining={remaining:#09b}, refill={refill}"
                        );
                        assert_eq!(
                            game.rng, reference_rng,
                            "seed=123, piece_set={mask:#09b}, remaining={remaining:#09b}, refill={refill}"
                        );
                    }
                }
            }
            // Instantiate each nonempty configuration so refills are checked against its static set.
            macro_rules! check_sets {
                ($($mask:literal),* $(,)?) => { $(check::<$mask>();)* };
            }
            check_sets!(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                122, 123, 124, 125, 126, 127,
            );
        }

        #[test]
        fn scenario_constructor_refills_standard_bags_after_partial_or_empty_remainders() {
            for remaining in [0, 3] {
                let mut board = Board::new();
                board.set_bit(8, 0);
                let mut game = Game::new_with_board_bag_piece_seeded(
                    board,
                    TetrisPieceBagState::try_from(remaining).expect("valid bag mask"),
                    TetrisPiece::T,
                    42,
                );
                assert_eq!(game.board, board);
                assert_eq!(game.current_piece, TetrisPiece::T);
                assert_game_draws_set(&mut game, remaining);
                for _ in 0..3 {
                    assert_game_draws_set(&mut game, TetrisPieceBagState::FULL_MASK);
                }
            }
        }

        #[test]
        fn scenario_update_samples_the_same_continuation_as_the_constructor() {
            for remaining in [0, 3] {
                let state = TetrisPieceBagState::try_from(remaining).expect("valid bag mask");
                let mut expected =
                    Game::new_with_board_bag_piece_seeded(Board::new(), state, TetrisPiece::T, 42);
                let mut updated = Game::new_with_seed(999);
                updated.set_bag_piece_seeded(state, TetrisPiece::T, 42);
                assert_eq!(updated.bag, expected.bag);
                assert_eq!(updated.rng, expected.rng);
                assert_eq!(updated.current_piece, expected.current_piece);
                for _ in 0..28 {
                    assert_eq!(
                        draw_from_game_bag(&mut updated),
                        draw_from_game_bag(&mut expected)
                    );
                }
            }
        }

        #[test]
        fn scenario_update_preserves_the_board_and_statistics() {
            let mut game = Game::new_with_seed(999);
            game.board.set_bit(8, 0);
            game.lines_cleared = 12;
            game.recent_lines_cleared = 2;
            game.piece_count = 30;
            let before = game;
            for remaining in [0, 3] {
                game.set_bag_piece_seeded(
                    TetrisPieceBagState::try_from(remaining).expect("valid bag mask"),
                    TetrisPiece::T,
                    42,
                );
                assert_eq!(game.board, before.board);
                assert_eq!(game.lines_cleared, before.lines_cleared);
                assert_eq!(game.recent_lines_cleared, before.recent_lines_cleared);
                assert_eq!(game.piece_count, before.piece_count);
            }
        }

        #[test]
        fn scenario_update_rejects_an_unconfigured_remainder_without_mutation() {
            for remaining in [0, TetrisPiece::I.mask()] {
                let mut game = TetrisGame::<TestPieceSet<3>>::new_with_seed(123);
                game.set_bag_piece_seeded(
                    TetrisPieceBagState::try_from(remaining).expect("valid bag mask"),
                    TetrisPiece::O,
                    42,
                );
                let before = game;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    game.set_bag_piece_seeded(
                        TetrisPieceBagState::from_pieces([TetrisPiece::S]),
                        TetrisPiece::O,
                        999,
                    );
                }));
                assert!(rejected.is_err());
                assert_eq!(game, before);
            }
        }

        #[test]
        fn full_bag_seeded_replay_is_unchanged() {
            let mut game = Game::new_with_seed(42);
            let expected = [
                5, 3, 1, 0, 2, 4, 6, 4, 0, 6, 5, 2, 1, 3, 5, 0, 2, 4, 3, 1, 6,
            ];
            assert_eq!(game.current_piece.index(), expected[0]);
            for &index in &expected[1..] {
                assert_eq!(draw_from_game_bag(&mut game).index(), index);
            }
            assert_eq!(game.rng.state, 10_870_387_891_257_067_482);
            assert_eq!(game.rng.n, 3);
        }

        #[test]
        fn successful_placement_draws_the_next_piece() {
            let mut game = Game::new_with_seed(42);
            let expected = game
                .iter_next_pieces()
                .nth(1)
                .expect("infinite piece stream");
            let remaining = game.bag.count();
            assert!(!game.apply_placement(game.current_placements()[0]).is_lost);
            assert_eq!(game.current_piece, expected);
            assert_eq!(game.bag.count(), remaining - 1);
        }

        #[test]
        fn successful_placements_increment_piece_count() {
            let mut game = Game::new_with_seed(42);
            for count in 1..=10 {
                // Isolate counting from the placement policy's ability to avoid a loss.
                game.board = Board::new();
                assert!(!game.apply_placement(game.current_placements()[0]).is_lost);
                assert_eq!(game.piece_count, count);
            }
        }

        #[test]
        fn stacking_leftmost_placements_eventually_loses() {
            let mut game = Game::new_with_seed(42);

            // Stack pieces until loss (always place in same column)
            for _ in 0..100 {
                let placements = game.current_placements();
                // Always place in column 0 to guarantee stacking
                let placement = placements
                    .iter()
                    .find(|p| p.orientation.column.inner() == 0)
                    .expect("a leftmost placement exists for the current piece");

                let result = game.apply_placement(*placement);
                if result.is_lost {
                    assert!(game.board.is_lost(), "Board should be in lost state");
                    return;
                }
            }
            assert!(
                game.board.is_lost(),
                "Game should have been lost after 100 pieces in same column"
            );
        }

        #[test]
        fn placement_draws_from_a_nonempty_bag_without_advancing_rng() {
            let mut game = Game::new_with_seed(42);
            let rng = game.rng;
            assert!(!game.apply_placement(game.current_placements()[0]).is_lost);
            assert_eq!(game.rng, rng);
        }

        #[test]
        fn placement_refills_an_empty_bag_and_advances_rng_once() {
            let mut game = Game::new_with_board_bag_piece_seeded(
                Board::new(),
                TetrisPieceBagState::from_pieces([]),
                TetrisPiece::O,
                42,
            );
            let mut expected_rng = game.rng;
            expected_rng.next_u64();
            assert!(!game.apply_placement(game.current_placements()[0]).is_lost);
            assert_eq!(game.rng, expected_rng);
            assert_eq!(game.bag.count() as usize, constants::NUM_TETRIS_PIECES - 1);
        }

        #[test]
        fn reset_replays_the_original_seed_after_refilling() {
            let mut game = Game::new_with_seed(42);
            let initial = game;
            // Consume the six remaining pieces, then trigger one refill.
            for _ in 0..constants::NUM_TETRIS_PIECES {
                draw_from_game_bag(&mut game);
            }
            assert_ne!(game.rng, initial.rng);
            game.reset(None);
            assert_eq!(game, initial);
        }
    }
    mod configured_encodings {
        use super::*;

        #[test]
        fn columns_accept_exactly_the_configured_range() {
            fn check<const COLS: usize>() {
                assert_eq!(crate::Column::<COLS>::default().inner(), 0, "cols={COLS}");
                for index in 0..=u8::MAX {
                    let result = std::panic::catch_unwind(|| crate::Column::<COLS>::new(index));
                    assert_eq!(
                        result.is_ok(),
                        (index as usize) < COLS,
                        "cols={COLS}, index={index}"
                    );
                    if let Ok(column) = result {
                        assert_eq!(column.inner(), index, "cols={COLS}, index={index}");
                    }
                }
            }
            check::<1>();
            check::<4>();
            check::<10>();
            check::<14>();
            check::<{ u8::MAX as usize }>();
        }

        #[test]
        fn configured_types_keep_their_compact_layout() {
            assert_eq!(std::mem::size_of::<crate::Column<4>>(), 1);
            assert_eq!(std::mem::size_of::<crate::TetrisPieceOrientation<4>>(), 2);
            assert_eq!(std::mem::size_of::<crate::TetrisPiecePlacement<4>>(), 3);
            assert_eq!(std::mem::size_of::<crate::TetrisPiecePlacement<10>>(), 3);
        }

        #[test]
        fn orientation_count_does_not_impose_a_mask_limit() {
            assert_eq!(
                crate::TetrisPieceOrientation::<0>::TOTAL_NUM_ORIENTATIONS,
                0
            );
            assert_eq!(
                crate::TetrisPieceOrientation::<17>::TOTAL_NUM_ORIENTATIONS,
                68
            );
            assert_eq!(
                crate::TetrisPieceOrientation::<65>::TOTAL_NUM_ORIENTATIONS,
                260
            );
        }

        #[test]
        fn orientation_indices_support_widths_beyond_a_mask() {
            fn check<const COLS: usize>() {
                let count = COLS * Rotation::ALL.len();
                for (rotation_index, rotation) in Rotation::ALL.into_iter().enumerate() {
                    for column in 0..COLS {
                        let orientation = crate::TetrisPieceOrientation::<COLS> {
                            rotation,
                            column: crate::Column::new(column as u8),
                        };
                        let index = rotation_index * COLS + column;
                        assert_eq!(
                            usize::from(orientation.index()),
                            index,
                            "cols={COLS}, rotation={rotation}, column={column}"
                        );
                        assert_eq!(
                            crate::TetrisPieceOrientation::<COLS>::from_index(index as u8),
                            orientation,
                            "cols={COLS}, rotation={rotation}, column={column}, index={index}"
                        );
                    }
                }
                for index in count..=u8::MAX as usize {
                    assert!(
                        std::panic::catch_unwind(|| {
                            crate::TetrisPieceOrientation::<COLS>::from_index(index as u8)
                        })
                        .is_err(),
                        "cols={COLS}, index={index}"
                    );
                }
            }
            check::<17>();
            check::<64>();
        }

        #[test]
        fn orientations_round_trip_and_masks_match_independent_geometry() {
            fn check<const COLS: usize>()
            where
                [(); crate::TetrisPieceOrientation::<COLS>::TOTAL_NUM_ORIENTATIONS]:,
            {
                let count = COLS * Rotation::ALL.len();
                for (rotation_index, rotation) in Rotation::ALL.into_iter().enumerate() {
                    for column in 0..COLS {
                        let orientation = crate::TetrisPieceOrientation::<COLS> {
                            rotation,
                            column: crate::Column::new(column as u8),
                        };
                        let index = rotation_index * COLS + column;
                        assert_eq!(
                            usize::from(orientation.index()),
                            index,
                            "cols={COLS}, rotation={rotation}, column={column}"
                        );
                        assert_eq!(
                            crate::TetrisPieceOrientation::<COLS>::from_index(index as u8),
                            orientation,
                            "cols={COLS}, rotation={rotation}, column={column}, index={index}"
                        );
                    }
                }
                for piece in TetrisPiece::all() {
                    let mut expected = 0u64;
                    for (rotation_index, rotation) in Rotation::ALL
                        .into_iter()
                        .enumerate()
                        .take(piece.num_rotations() as usize)
                    {
                        let (_, width, _) = reference_shape(piece, rotation);
                        if width <= COLS {
                            for column in 0..=COLS - width {
                                expected |= 1 << (rotation_index * COLS + column);
                            }
                        }
                    }
                    assert_eq!(
                        crate::TetrisPieceOrientation::<COLS>::bitmask_from_piece(piece).inner(),
                        expected,
                        "cols={COLS}, piece={piece}"
                    );
                }
                for index in count..=u8::MAX as usize {
                    assert!(
                        std::panic::catch_unwind(|| {
                            crate::TetrisPieceOrientation::<COLS>::from_index(index as u8)
                        })
                        .is_err(),
                        "cols={COLS}, index={index}"
                    );
                }
            }
            check::<1>();
            check::<2>();
            check::<3>();
            check::<4>();
            check::<5>();
            check::<6>();
            check::<7>();
            check::<8>();
            check::<9>();
            check::<10>();
            check::<11>();
            check::<12>();
            check::<13>();
            check::<14>();
            check::<15>();
            check::<16>();
        }

        #[test]
        fn placement_ids_round_trip_and_preserve_symmetric_aliases_at_every_width() {
            fn check<const COLS: usize>() {
                let all = crate::TetrisPiecePlacement::<COLS>::ALL_PLACEMENTS;
                for (index, &placement) in all.iter().enumerate() {
                    assert_eq!(
                        placement.index() as usize,
                        index,
                        "cols={COLS}, placement={placement:?}"
                    );
                    assert_eq!(
                        crate::TetrisPiecePlacement::<COLS>::from_index(index as u8),
                        placement,
                        "cols={COLS}, index={index}"
                    );
                }
                for piece in TetrisPiece::all() {
                    for rotation in Rotation::ALL {
                        let (_, width, _) = reference_shape(piece, rotation);
                        for column in 0..COLS {
                            let placement = crate::TetrisPiecePlacement::<COLS> {
                                piece,
                                orientation: crate::TetrisPieceOrientation {
                                    rotation,
                                    column: crate::Column::new(column as u8),
                                },
                            };
                            if column + width > COLS {
                                assert!(
                                    std::panic::catch_unwind(|| placement.index()).is_err(),
                                    "cols={COLS}, piece={piece}, rotation={rotation}, column={column}"
                                );
                                continue;
                            }
                            let canonical =
                                Rotation::ALL[(rotation.index() % piece.num_rotations()) as usize];
                            let expected = all
                                .iter()
                                .position(|p| {
                                    p.piece == piece
                                        && p.orientation.rotation == canonical
                                        && usize::from(p.orientation.column.inner()) == column
                                })
                                .expect("fitting canonical placement");
                            assert_eq!(
                                placement.index() as usize,
                                expected,
                                "cols={COLS}, piece={piece}, rotation={rotation}, column={column}"
                            );
                        }
                    }
                }
                for index in all.len()..=u8::MAX as usize {
                    assert!(
                        std::panic::catch_unwind(|| {
                            crate::TetrisPiecePlacement::<COLS>::from_index(index as u8)
                        })
                        .is_err(),
                        "cols={COLS}, index={index}"
                    );
                }
            }
            check::<1>();
            check::<2>();
            check::<3>();
            check::<4>();
            check::<5>();
            check::<6>();
            check::<7>();
            check::<8>();
            check::<9>();
            check::<10>();
            check::<11>();
            check::<12>();
            check::<13>();
            check::<14>();
        }

        #[test]
        fn same_width_placements_work_across_board_heights() {
            let placement = crate::TetrisPiecePlacement::<4>::from_index(0);
            let mut short = TetrisBoard::<4, 8>::new();
            let mut tall = TetrisBoard::<4, 20>::new();
            assert_eq!(
                short.apply_piece_placement(placement),
                tall.apply_piece_placement(placement)
            );
            assert_eq!(short.as_limbs(), tall.as_limbs());
            assert_eq!(short.as_limbs(), &[0b11, 0b11, 0, 0]);
        }

        #[test]
        fn every_piece_rotation_fits_four_columns() {
            for piece in TetrisPiece::all() {
                for rotation in Rotation::ALL {
                    let (_, width, _) = reference_shape(piece, rotation);
                    assert!(width <= 4, "{piece} at {rotation} must fit four columns");
                }
            }
        }

        #[test]
        fn random_placements_only_sample_pieces_and_orientations_that_fit() {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut sampled_pieces = TetrisPieceBagState::from_pieces([]);
            for _ in 0..128 {
                let placement: crate::TetrisPiecePlacement<4> = rng.random();
                sampled_pieces.0 |= placement.piece.mask();
                let width = placement.piece.width(placement.orientation.rotation) as usize;
                assert!(usize::from(placement.orientation.column.inner()) + width <= 4);
                assert!(
                    (placement.orientation.rotation.index() as usize)
                        < placement.piece.num_rotations() as usize
                );
            }
            assert_eq!(sampled_pieces, TetrisPieceBagState::full());
        }

        #[test]
        fn configured_constructors_and_indices_work_in_constants() {
            const COLUMN: crate::Column<4> = crate::Column::new(2);
            const ORIENTATION: crate::TetrisPieceOrientation<4> =
                crate::TetrisPieceOrientation::from_index(6);
            const PLACEMENT: crate::TetrisPiecePlacement<4> =
                crate::TetrisPiecePlacement::from_index(2);
            const INDEX: u8 = PLACEMENT.index();
            const INNER: u8 = COLUMN.inner();
            assert_eq!(INNER, 2);
            assert_eq!(COLUMN.inner(), 2);
            assert_eq!(ORIENTATION.column, COLUMN);
            assert_eq!(ORIENTATION.rotation, Rotation::R90);
            assert_eq!(INDEX, 2);
            assert_eq!(PLACEMENT.orientation.column, COLUMN);
        }
    }
}
