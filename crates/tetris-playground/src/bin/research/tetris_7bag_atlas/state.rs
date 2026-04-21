use tetris_game::{TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

pub type BoardId = u32;
pub type StateId = u32;
pub type PackedPlacement = u8;

/// A state in the 7-bag graph: a board paired with the remaining bag contents.
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

/// One branch of the adversary's choice: a drawable piece and the resulting bag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceBranch {
    pub piece: TetrisPiece,
    pub next_bag: TetrisPieceBagState,
}

/// Compute the bag state after drawing `piece`. Returns `None` if piece is not
/// in the bag. Refills to a full bag when the last piece is drawn.
#[inline]
pub fn next_bag_state(bag: TetrisPieceBagState, piece: TetrisPiece) -> Option<TetrisPieceBagState> {
    if !bag.contains(piece) {
        return None;
    }
    let mut next = bag;
    next.remove(piece);
    if next.is_empty() {
        next.fill();
    }
    Some(next)
}

/// Iterate over all piece branches drawable from the given bag.
#[inline]
pub fn piece_branches(bag: TetrisPieceBagState) -> impl Iterator<Item = PieceBranch> {
    TetrisPiece::all().into_iter().filter_map(move |piece| {
        next_bag_state(bag, piece).map(|next_bag| PieceBranch { piece, next_bag })
    })
}

#[inline]
pub fn pack_placement(placement: TetrisPiecePlacement) -> PackedPlacement {
    placement.index()
}

#[inline]
pub fn unpack_placement(packed: PackedPlacement) -> TetrisPiecePlacement {
    TetrisPiecePlacement::from_index(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_branches_matches_bag_count() {
        let bag = TetrisPieceBagState::new();
        let branches: Vec<_> = piece_branches(bag).collect();
        assert_eq!(branches.len(), bag.count() as usize);
        for branch in branches {
            assert!(!branch.next_bag.contains(branch.piece));
        }
    }

    #[test]
    fn next_bag_state_removes_piece() {
        let bag = TetrisPieceBagState::new();
        let next = next_bag_state(bag, TetrisPiece::O_PIECE).expect("piece in bag");
        assert!(!next.contains(TetrisPiece::O_PIECE));
        assert_eq!(next.count(), 6);
    }

    #[test]
    fn next_bag_state_refills_on_last() {
        let bag = TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE));
        let next = next_bag_state(bag, TetrisPiece::O_PIECE).expect("piece in bag");
        assert_eq!(next, TetrisPieceBagState::new());
    }

    #[test]
    fn placement_round_trips() {
        for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
            assert_eq!(unpack_placement(pack_placement(placement)), placement);
        }
    }
}
