use tetris_game::TetrisPieceBagState;

use crate::state::PackedPlacement;

pub type BoardId = u32;
pub type StateId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateKey {
    pub board_id: BoardId,
    pub bag: TetrisPieceBagState,
}

impl StateKey {
    pub const fn new(board_id: BoardId, bag: TetrisPieceBagState) -> Self {
        Self { board_id, bag }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdgeRange {
    pub start: u32,
    pub len: u32,
}

impl EdgeRange {
    pub const EMPTY: Self = Self { start: 0, len: 0 };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateIndex {
    pub bag: TetrisPieceBagState,
    pub piece_ranges: [EdgeRange; 7],
}

impl Default for StateIndex {
    fn default() -> Self {
        Self {
            bag: TetrisPieceBagState::from_pieces([]),
            piece_ranges: [EdgeRange::EMPTY; 7],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatEdge {
    pub succ: StateId,
    pub placement: PackedPlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredecessorRef {
    pub parent: StateId,
    pub piece_idx: u8,
    pub placement: PackedPlacement,
}
