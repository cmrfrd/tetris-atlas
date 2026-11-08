use crate::tetris::TetrisPieceBagState;

/// Universal constants for Tetris pieces and column representation.
///
/// Game dimensions and piece sets live on [`TetrisGameConfig`].
/// Board geometry and storage values are derived by [`crate::tetris::TetrisBoard`].
pub mod constants {
    /// Number of quarter-turn orientations.
    pub const NUM_ROTATIONS: u8 = 4;

    /// Number of standard tetrominoes.
    pub const NUM_TETRIS_PIECES: usize = 7;

    /// Unsigned storage word for one board column; bit zero is the bottom row.
    pub type BackingColType = u32;

    pub const ACTUAL_ROWS: usize = BackingColType::BITS as usize;

    /// Number of possible cell states (0 = empty, 1 = filled).
    pub const NUM_TETRIS_CELL_STATES: usize = 2;
}

/// Configuration trait for Tetris game variants.
///
/// Defines game dimensions and the allowed piece set.
/// Boards, columns, orientations, and placements take their dimensions directly as const parameters.
/// Game constructors check geometry and piece-set invariants at compile time when instantiated.
///
/// # Example
///
/// ```
/// use tetris_game::{TetrisBoard, TetrisGameConfig, StandardTetris};
///
/// type Board = TetrisBoard<{ StandardTetris::COLS }, { StandardTetris::ROWS }>;
///
/// assert_eq!(StandardTetris::ROWS, 20);
/// assert_eq!(StandardTetris::COLS, 10);
/// const CELLS: usize = Board::BOARD_SIZE;
/// assert_eq!(CELLS, 200);
/// ```
pub trait TetrisGameConfig:
    'static + Copy + Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + PartialOrd + Ord
{
    /// Number of playable rows; loss is checked above this height after line clearing.
    const ROWS: usize;
    /// Number of columns on the board.
    const COLS: usize;
    /// The set of pieces used in this game variant.
    const PIECE_SET: TetrisPieceBagState;
}

/// The canonical Tetris configuration: 20×10 board with all 7 pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StandardTetris;

impl TetrisGameConfig for StandardTetris {
    const ROWS: usize = 20;
    const COLS: usize = 10;
    const PIECE_SET: TetrisPieceBagState = TetrisPieceBagState::full();
}
