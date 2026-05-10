/// Board dimensions.
pub const BOARD_COLS: u32 = 10;
pub const BOARD_ROWS: u32 = 20;

/// Bitvector width for z3 encoding. Must be >= BOARD_ROWS + 4 to detect loss.
pub const BV_WIDTH: u32 = 32;

/// Maximum lines that can be cleared in a single placement.
pub const MAX_LINE_CLEARS: u32 = 4;

/// Number of VM registers.
pub const NUM_REGISTERS: u8 = 8;

/// VM value width. 8 bits is sufficient for column bitmasks (max_height ≤ 8)
/// and placement indices (0–116). Keeping this small is critical for z3 synthesis
/// performance — 32-bit bitvectors make synthesis intractable.
pub const VM_VALUE_BITS: u32 = 8;

/// Number of column word inputs (one full `u32` bitmask per column).
pub const NUM_COL_INPUTS: u8 = 10;

/// Number of piece input slots (current piece index 0-6).
pub const NUM_PIECE_INPUT: u8 = 1;

/// Total operand slots: registers + column inputs + piece input.
pub const NUM_OPERAND_SLOTS: u8 = NUM_REGISTERS + NUM_COL_INPUTS + NUM_PIECE_INPUT; // 19

/// Number of opcodes (0..8 inclusive: ADD, SUB, AND, OR, CMP_LT, MUX, CONST_LO, MIN, NOP).
pub const NUM_OPCODES: u8 = 9;

/// Invariant parameters: structural bounds on board state.
#[derive(Debug, Clone, Copy)]
pub struct InvariantParams {
    pub max_height: u32,
    pub max_total_holes: u32,
    pub max_roughness: u32,
    /// Require at least this many adjacent columns of equal height (0 = no constraint).
    pub min_flat_block: u32,
}

impl InvariantParams {
    pub const fn new(max_height: u32, max_total_holes: u32, max_roughness: u32) -> Self {
        Self {
            max_height,
            max_total_holes,
            max_roughness,
            min_flat_block: 0,
        }
    }
}

/// Configuration for simulation-based verification.
#[derive(Debug, Clone, Copy)]
pub struct SimulationConfig {
    /// Number of random seeds to test per verification round.
    pub num_seeds: u32,
    /// Number of pieces each game must survive.
    pub survival_target: u32,
}
