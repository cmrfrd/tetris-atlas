use tetris_game::{TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};

pub type PackedPlacement = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceBranch {
    pub piece: TetrisPiece,
    pub next_bag: TetrisPieceBagState,
}

#[inline]
pub fn pack_placement(placement: TetrisPiecePlacement<10>) -> PackedPlacement {
    placement.index()
}

#[inline]
pub fn unpack_placement(placement: PackedPlacement) -> TetrisPiecePlacement<10> {
    <TetrisPiecePlacement<10>>::from_index(placement)
}

#[inline]
pub fn next_bag_state(bag: TetrisPieceBagState, piece: TetrisPiece) -> Option<TetrisPieceBagState> {
    if !bag.contains(piece) {
        return None;
    }

    let mut next = bag;
    next.remove(piece);
    if next.empty() {
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
        let bag = TetrisPieceBagState::full();
        let branches = piece_branches(bag).collect::<Vec<_>>();

        assert_eq!(branches.len(), bag.count() as usize);
        for branch in branches {
            assert!(!branch.next_bag.contains(branch.piece));
        }
    }

    #[test]
    fn next_bag_state_removes_piece_without_reset() {
        let bag = TetrisPieceBagState::full();
        let next = next_bag_state(bag, TetrisPiece::O).expect("piece should exist");

        assert!(!next.contains(TetrisPiece::O));
        assert_eq!(next.count(), 6);
    }

    #[test]
    fn next_bag_state_resets_when_last_piece_removed() {
        let mut bag = TetrisPieceBagState::try_from(TetrisPiece::O.mask()).expect("valid bag mask");
        let next = next_bag_state(bag, TetrisPiece::O).expect("piece should exist");

        bag.remove(TetrisPiece::O);
        assert!(bag.empty());
        assert_eq!(next, TetrisPieceBagState::full());
    }

    #[test]
    fn placement_pack_round_trips() {
        for &placement in <TetrisPiecePlacement<10>>::ALL_PLACEMENTS {
            let packed = pack_placement(placement);
            assert_eq!(unpack_placement(packed), placement);
        }
    }
}
