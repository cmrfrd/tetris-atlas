/// Board dimensions — change these to experiment with smaller boards.
pub const BOARD_COLS: u32 = 10;
pub const BOARD_ROWS: u32 = 20;

/// Bitvector width for z3 encoding. Must be >= BOARD_ROWS + 4 to detect
/// pieces placed above the board (loss condition).
pub const BV_WIDTH: u32 = 32;

/// Maximum lines that can be cleared in a single placement (tetromino = 4 cells).
pub const MAX_LINE_CLEARS: u32 = 4;

/// Base invariant parameters — the starting candidate invariant P(board).
/// CEGIS will iteratively strengthen this by learning additional constraints.
///
/// A board satisfies the base invariant iff ALL of the following hold:
///   - Every column has height <= MAX_HEIGHT
///   - Total board holes <= MAX_TOTAL_HOLES (holes = height - popcount, summed)
///   - Max adjacent column height difference <= MAX_ROUGHNESS
pub const MAX_HEIGHT: u32 = 4;
pub const MAX_TOTAL_HOLES: u32 = 2;
pub const MAX_ROUGHNESS: u32 = 2;

/// Full-bag (reset point) invariant — tighter than the general base.
/// Boards at the start of a new bag cycle must be very flat.
pub const FULL_BAG_MAX_HEIGHT: u32 = 2;
pub const FULL_BAG_MAX_ROUGHNESS: u32 = 1;

/// Invariant parameters: structural bounds on board state.
/// Used for parameterized invariant encoding (bag-aware CEGIS).
#[derive(Debug, Clone, Copy)]
pub struct InvariantParams {
    pub max_height: u32,
    pub max_total_holes: u32,
    pub max_roughness: u32,
    /// Require at least this many adjacent columns of equal height (0 = no constraint).
    /// Ensures pieces like the I-piece horizontal have a valid placement.
    pub min_flat_block: u32,
}

impl InvariantParams {
    pub const fn from_config() -> Self {
        Self {
            max_height: MAX_HEIGHT,
            max_total_holes: MAX_TOTAL_HOLES,
            max_roughness: MAX_ROUGHNESS,
            min_flat_block: 0,
        }
    }
}
