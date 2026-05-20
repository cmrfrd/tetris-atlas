use crate::tetris::TetrisPieceBagState;

/// Universal constants for Tetris pieces and column representation.
///
/// Config-dependent values (`ROWS`, `COLS`, `BOARD_SIZE`, `BOARD_BYTES`)
/// live on [`TetrisGameConfig`] and its implementors (e.g. [`StandardTetris`]).
pub mod constants {
    /// Tetris piece rotations and pieces
    pub const NUM_ROTATIONS: u8 = 4;
    pub const NUM_TETRIS_PIECES: usize = 7;

    /// Column representation for the board: each `u32` bit is a row
    /// (0 = bottom, 19 = top, 20-31 = overflow).
    pub type BackingColType = u32;
    pub const ACTUAL_ROWS: usize = BackingColType::BITS as usize;

    /// Bytes per column limb — derived from the backing type rather than
    /// hard-coding `4`. Update [`BackingColType`] and this follows.
    pub const BYTES_PER_LIMB: usize = std::mem::size_of::<BackingColType>();

    const _: () = assert!(
        BYTES_PER_LIMB * 8 == ACTUAL_ROWS,
        "BYTES_PER_LIMB must match BackingColType bit-width"
    );

    /// Number of possible cell states (0 = empty, 1 = filled).
    pub const NUM_TETRIS_CELL_STATES: usize = 2;
}

/// Configuration trait for Tetris game variants.
///
/// Defines the board dimensions and allowed piece set at compile time.
/// Implement this trait to create custom game configurations that can
/// be used with generic game/board types.
///
/// # Example
///
/// ```
/// use tetris_game::config::{TetrisGameConfig, StandardTetris};
///
/// assert_eq!(StandardTetris::ROWS, 20);
/// assert_eq!(StandardTetris::COLS, 10);
/// assert_eq!(StandardTetris::BOARD_SIZE, 200);
/// ```
pub trait TetrisGameConfig:
    Copy + Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + PartialOrd + Ord
{
    /// Number of visible rows on the board.
    const ROWS: usize;
    /// Number of columns on the board.
    const COLS: usize;
    /// The set of pieces used in this game variant.
    const PIECE_SET: TetrisPieceBagState;

    /// Total number of visible cells (`ROWS * COLS`).
    const BOARD_SIZE: usize = Self::ROWS * Self::COLS;
    /// Total rows in the backing column type (includes overflow bits).
    const ACTUAL_ROWS: usize = constants::BackingColType::BITS as usize;
    /// Bytes per column limb.
    const BYTES_PER_LIMB: usize = std::mem::size_of::<constants::BackingColType>();
    /// Total byte size of the packed board representation.
    const BOARD_BYTES: usize = Self::COLS * Self::BYTES_PER_LIMB;
    /// Bitmask covering the visible rows in a column limb.
    const ROW_MASK: u32 = (1u32 << Self::ROWS) - 1;

    /// Compile-time check: ROWS must fit in the backing column type.
    const _ASSERT_ROWS_FIT: () = assert!(
        Self::ACTUAL_ROWS > Self::ROWS,
        "ACTUAL_ROWS must be greater than ROWS"
    );
}

/// The canonical Tetris configuration: 20×10 board with all 7 pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StandardTetris;

impl TetrisGameConfig for StandardTetris {
    const ROWS: usize = 20;
    const COLS: usize = 10;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::new();
}
