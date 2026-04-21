use tetris_game::TetrisPieceBagState;

use crate::state::{PackedPlacement, StateId};

/// A contiguous range of edges in the flat edge array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EdgeRange {
    pub start: u32,
    pub len: u32,
}

impl EdgeRange {
    pub const EMPTY: Self = Self { start: 0, len: 0 };
}

/// Per-state metadata: the bag and per-piece edge ranges.
///
/// `piece_ranges[i]` contains the edge range for piece with `index() == i`.
/// Pieces not in the bag have `EdgeRange::EMPTY`.
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

/// A successor edge: the resulting state and the placement that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatEdge {
    pub succ: StateId,
    pub placement: PackedPlacement,
}

/// A backward reference carrying the parent state and which piece was played.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredecessorRef {
    pub parent: StateId,
    pub piece_idx: u8,
}
