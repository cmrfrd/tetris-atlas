use tetris_game::TetrisPiecePlacement;

pub type BoardId = u32;
pub type StateId = u32;
pub type PackedPlacement = u8;

/// A state in the fixed-cycle graph: a board at a specific position in the cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateKey {
    pub board_id: BoardId,
    /// Position within the 7-piece cycle (0..7).
    pub cycle_pos: u8,
}

impl StateKey {
    pub const fn new(board_id: BoardId, cycle_pos: u8) -> Self {
        Self {
            board_id,
            cycle_pos,
        }
    }
}

/// A contiguous range of edges in the flat edge array.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeRange {
    pub start: u32,
    pub len: u32,
}

/// A successor edge: the resulting state and the placement that produced it.
#[derive(Debug, Clone, Copy)]
pub struct FlatEdge {
    pub succ: StateId,
    pub placement: PackedPlacement,
}

/// A backward reference from a successor to its parent state.
#[derive(Debug, Clone, Copy)]
pub struct PredecessorRef {
    pub parent: StateId,
}

#[inline]
pub fn pack_placement(placement: TetrisPiecePlacement) -> PackedPlacement {
    placement.index()
}

#[inline]
pub fn unpack_placement(packed: PackedPlacement) -> TetrisPiecePlacement {
    TetrisPiecePlacement::from_index(packed)
}
