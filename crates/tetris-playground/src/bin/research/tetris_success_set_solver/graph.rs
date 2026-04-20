use tetris_game::TetrisPieceBagState;

use crate::state::{PackedPlacement, StateId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdgeRange {
    pub start: u32,
    pub len: u32,
}

impl EdgeRange {
    pub const EMPTY: Self = Self { start: 0, len: 0 };

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateIndex {
    pub bag: TetrisPieceBagState,
    pub piece_ranges: [EdgeRange; 7],
}

impl Default for StateIndex {
    fn default() -> Self {
        Self {
            bag: TetrisPieceBagState::from(0),
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
