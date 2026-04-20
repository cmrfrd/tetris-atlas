use tetris_game::{TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

pub type BoardId = u32;
pub type StateId = u32;
pub type PackedPlacement = u8;

pub const FULL_BAG_MASK: TetrisPieceBagState = TetrisPieceBagState::new();

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionEdge {
    pub succ: StateId,
    pub placement: PackedPlacement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceBranch {
    pub piece: TetrisPiece,
    pub next_bag: TetrisPieceBagState,
}

#[inline]
pub fn pack_placement(placement: TetrisPiecePlacement) -> PackedPlacement {
    placement.index()
}

#[inline]
pub fn unpack_placement(placement: PackedPlacement) -> TetrisPiecePlacement {
    TetrisPiecePlacement::from_index(placement)
}

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

#[inline]
pub fn piece_branches(bag: TetrisPieceBagState) -> impl Iterator<Item = PieceBranch> {
    TetrisPiece::all().into_iter().filter_map(move |piece| {
        next_bag_state(bag, piece).map(|next_bag| PieceBranch { piece, next_bag })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_branches_matches_bag_count() {
        let bag = TetrisPieceBagState::new();
        let branches = piece_branches(bag).collect::<Vec<_>>();

        assert_eq!(branches.len(), bag.count() as usize);
        for branch in branches {
            assert!(!branch.next_bag.contains(branch.piece));
        }
    }

    #[test]
    fn next_bag_state_removes_piece_without_reset() {
        let bag = TetrisPieceBagState::new();
        let next = next_bag_state(bag, TetrisPiece::O_PIECE).expect("piece should exist");

        assert!(!next.contains(TetrisPiece::O_PIECE));
        assert_eq!(next.count(), 6);
    }

    #[test]
    fn next_bag_state_resets_when_last_piece_removed() {
        let mut bag = TetrisPieceBagState::from(u8::from(TetrisPiece::O_PIECE));
        let next = next_bag_state(bag, TetrisPiece::O_PIECE).expect("piece should exist");

        bag.remove(TetrisPiece::O_PIECE);
        assert!(bag.is_empty());
        assert_eq!(next, TetrisPieceBagState::new());
    }

    #[test]
    fn placement_pack_round_trips() {
        for placement in TetrisPiecePlacement::ALL_PLACEMENTS {
            let packed = pack_placement(placement);
            assert_eq!(unpack_placement(packed), placement);
        }
    }
}
