//! Exact seed reachability for eight through twelve columns, a one-row history cap, and the full 7-bag.
//!
//! From empty, only a horizontal I can stay below the cap. It occupies four cells.
//! All other pieces have height >=2 and at most three cells in any single row, so
//! none can fill a row of at least eight cells on that four-cell board. No clear means they exceed
//! the cap. The bag contains no second I until every other piece has been played.
//! Hence the entire reachable board set is empty plus the horizontal I positions, giving COLS - 2 states.
//! Queries use shifts and trailing zeros; no graph or exclusion table is retained.
use tetris_macros::inline_conditioned;

/// Counts empty plus every horizontal-I position in the supported width range.
#[inline_conditioned]
pub(crate) const fn board_count(cols: usize) -> u64 {
    // Empty, plus every position of a four-cell horizontal I.
    (1 + (cols - 4 + 1)) as u64
}

/// Returns whether the full-bag, one-row reachability proof applies.
#[inline_conditioned]
pub(crate) const fn applies(cols: usize, rows: usize, full_bag: bool) -> bool {
    cols >= 8 && cols <= 12 && rows == 1 && full_bag
}

/// Ranks an empty row or a single horizontal I, rejecting every other pattern.
#[inline_conditioned]
pub(crate) fn rank(cols: usize, row: u16) -> Option<u64> {
    if row == 0 {
        return Some(0);
    }
    let start = row.trailing_zeros();
    if u64::from(start) < board_count(cols) - 1 && row == 15u16 << start {
        Some(u64::from(start) + 1)
    } else {
        None
    }
}

/// Restores the row mask for a valid seed-reachable board ID.
#[inline_conditioned]
pub(crate) fn select(cols: usize, id: u64) -> Option<u16> {
    if id == 0 {
        Some(0)
    } else if id < board_count(cols) {
        Some(15u16 << (id - 1))
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{BoardCodec, BoardId, CompiledConfig, Error, GameDimensions};
    use std::sync::LazyLock;
    use tetris_game::{
        StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPiece,
        TetrisPieceBagState, TetrisPiecePlacement,
    };

    /// Independent exhaustive engine closure, preserving the actual remaining bag.
    fn collect_single_row<const COLS: usize>() -> std::collections::HashSet<u16> {
        use std::collections::HashSet;

        let full = u8::from(TetrisPieceBagState::full());
        let empty = TetrisBoard::<COLS, 20>::new();
        let mut seen = HashSet::from([(*empty.as_limbs(), full)]);
        let mut pending = vec![(empty, full)];
        let mut boards = HashSet::from([0]);
        let mut head = 0;
        while head < pending.len() {
            let (parent, bag) = pending[head];
            head += 1;
            for &placement in TetrisPiecePlacement::<COLS>::ALL_PLACEMENTS {
                let bit = placement.piece.mask();
                if bag & bit == 0 {
                    continue;
                }
                let mut board = parent;
                board.apply_piece_placement(placement);
                if board.is_lost() || board.height() > 1 {
                    continue;
                }
                let next_bag = if bag == bit { full } else { bag & !bit };
                let row = board
                    .as_limbs()
                    .iter()
                    .enumerate()
                    .fold(0u16, |row, (x, &col)| row | (u16::from(col != 0) << x));
                boards.insert(row);
                if seen.insert((*board.as_limbs(), next_bag)) {
                    pending.push((board, next_bag));
                }
            }
        }
        boards
    }

    /// Caches the complete engine closure for each supported one-row width.
    pub(crate) fn single_row_reachable(cols: usize) -> &'static std::collections::HashSet<u16> {
        static SETS: LazyLock<[std::collections::HashSet<u16>; 5]> = LazyLock::new(|| {
            [
                collect_single_row::<8>(),
                collect_single_row::<9>(),
                collect_single_row::<10>(),
                collect_single_row::<11>(),
                collect_single_row::<12>(),
            ]
        });
        &SETS[cols - 8]
    }

    #[test]
    fn every_row_matches_exhaustive_canonical_seven_bag_reachability() {
        type Codec = BoardCodec<StandardTetris, 1, 0>;
        let codec = Codec::new();
        let reachable = single_row_reachable(10);
        assert_eq!(reachable.len(), 8);
        assert_eq!(Codec::BOARD_COUNT, 8);
        assert_eq!(
            <GameDimensions<StandardTetris, 1, 0> as CompiledConfig>::BOARD_COUNT,
            8
        );
        assert_eq!(Codec::ID_BITS, 3);
        assert_eq!(Codec::ID_BYTES, 1);
        assert_eq!(codec.table_payload_bytes(), 0);
        for row in 0u16..=u16::MAX {
            let expected = reachable.contains(&row);
            assert_eq!(rank(10, row).is_some(), expected, "{row:016b}");
            assert_eq!(codec.id_rows_u64(&[row]).is_ok(), expected, "{row:016b}");
            if row < 1024 {
                let board = TetrisBoard::<10, 20>::from_limbs(std::array::from_fn(|x| {
                    u32::from((row >> x) & 1)
                }));
                assert_eq!(codec.id_u64(&board).is_ok(), expected);
            }
        }
        for id in 0..board_count(10) {
            let row = select(10, id).expect("valid ID decodes");
            assert!(reachable.contains(&row));
            assert_eq!(rank(10, row), Some(id));
            let board = codec.board_from_u64(id).expect("valid ID decodes");
            let native: BoardId<StandardTetris, 1, 0> =
                codec.id(&board).expect("reachable board encodes");
            assert_eq!(u64::from(native), id);
            assert_eq!(codec.board_from_id(native), Ok(board));
            assert_eq!(codec.rows_from_id(native), Ok([row]));
        }
        assert_eq!(select(10, 8), None);
        assert_eq!(select(10, u64::MAX), None);
        assert_eq!(codec.rows_from_u64(8), Err(Error::InvalidId));
        assert_eq!(codec.board_from_u64(u64::MAX), Err(Error::InvalidId));
        let redundant_budget = BoardCodec::<StandardTetris, 1, 4>::new();
        assert_eq!(redundant_budget.len(), 8);
        assert_eq!(redundant_budget.table_payload_bytes(), 0);
    }

    #[test]
    fn every_admitted_nonempty_board_has_a_seeded_game_witness() {
        let seed = (1..=1024)
            .find(|&seed| {
                TetrisGame::<StandardTetris>::new_with_seed(seed).current_piece() == TetrisPiece::I
            })
            .expect("fixed seed search contains a first-I draw");
        let codec = BoardCodec::<StandardTetris, 1, 0>::new();
        for column in 0u8..7 {
            let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
            let placement = TetrisPiecePlacement::<10>::ALL_PLACEMENTS
                .iter()
                .copied()
                .find(|p| {
                    p.piece == TetrisPiece::I
                        && p.piece.width(p.orientation.rotation) == 4
                        && p.orientation.column.inner() == column
                })
                .expect("horizontal I witness exists");
            assert!(!game.apply_placement(placement).is_lost);
            assert_eq!(game.board.height(), 1);
            let id = codec.id_u64(&game.board).expect("seeded witness encodes");
            assert_eq!(id, 1 + u64::from(column));
            assert_eq!(codec.board_from_u64(id), Ok(game.board));
        }
        eprintln!("SINGLE_ROW_SEED_WITNESSES seed={seed} boards=8");
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct IOnly;

    impl TetrisGameConfig for IOnly {
        const COLS: usize = 10;
        const ROWS: usize = 20;
        const PIECE_SET: TetrisPieceBagState = {
            let mut bag = TetrisPieceBagState::full();
            bag.remove(TetrisPiece::O);
            bag.remove(TetrisPiece::S);
            bag.remove(TetrisPiece::Z);
            bag.remove(TetrisPiece::T);
            bag.remove(TetrisPiece::L);
            bag.remove(TetrisPiece::J);
            bag
        };
    }

    #[test]
    fn restricted_piece_games_do_not_share_the_full_bag_exclusion() {
        let mut game = TetrisGame::<IOnly>::new_with_seed(1);
        for column in [0, 4] {
            assert_eq!(game.current_piece(), TetrisPiece::I);
            let action = TetrisPiecePlacement::<10>::ALL_PLACEMENTS
                .iter()
                .copied()
                .find(|p| {
                    p.piece == TetrisPiece::I
                        && p.piece.width(p.orientation.rotation) == 4
                        && usize::from(p.orientation.column.inner()) == column
                })
                .expect("horizontal I placement exists");
            assert!(!game.apply_placement(action).is_lost);
        }
        assert_eq!(game.board.height(), 1);
        assert_eq!(game.board.count(), 8);
        let restricted = BoardCodec::<IOnly, 1, 0>::new();
        assert_eq!(restricted.len(), 134);
        assert!(restricted.id(&game.board).is_ok());
        assert_eq!(
            BoardCodec::<StandardTetris, 1, 0>::new().id(&game.board),
            Err(Error::ImpossibleBoard)
        );
    }

    #[test]
    fn a_low_target_after_a_higher_history_is_not_excluded_from_larger_caps() {
        let wanted = TetrisPiece::I.mask() | TetrisPiece::T.mask() | TetrisPiece::L.mask();
        let mut witness = None;
        for seed in 1..=1024 {
            let mut game = TetrisGame::<StandardTetris>::new_with_seed(seed);
            let mut played = 0;
            for _ in 0..3 {
                let piece = game.current_piece();
                let column = match piece {
                    TetrisPiece::I => 0,
                    TetrisPiece::T => 4,
                    TetrisPiece::L => 7,
                    _ => break,
                };
                let placement = TetrisPiecePlacement::<10>::ALL_PLACEMENTS
                    .iter()
                    .copied()
                    .find(|p| {
                        if p.piece != piece || p.orientation.column.inner() != column {
                            return false;
                        }
                        let width = p.piece.width(p.orientation.rotation);
                        if usize::from(width) != if piece == TetrisPiece::I { 4 } else { 3 } {
                            return false;
                        }
                        let mut footprint = TetrisBoard::<10, 20>::new();
                        footprint.apply_piece_placement(*p);
                        let start = usize::from(p.orientation.column.inner());
                        footprint.as_limbs()[start..start + usize::from(width)]
                            .iter()
                            .all(|col| col & 1 == 1)
                    })
                    .expect("piece can tile its disjoint floor interval");
                assert!(!game.apply_placement(placement).is_lost);
                assert!(game.board.height() <= 2);
                played |= piece.mask();
            }
            if played == wanted {
                witness = Some((seed, game.board));
                break;
            }
        }
        let (seed, board) = witness.expect("fixed seeds include an I/T/L first-bag prefix");
        assert_eq!(board.height(), 1);
        assert_eq!(board.count(), 2);
        assert_eq!(
            BoardCodec::<StandardTetris, 1, 0>::new().id_u64(&board),
            Err(Error::ImpossibleBoard)
        );
        let higher = BoardCodec::<StandardTetris, 2, 0>::new();
        let id = higher
            .id_u64(&board)
            .expect("legal capped-history witness encodes");
        assert_eq!(higher.board_from_u64(id), Ok(board));
        assert!(
            BoardCodec::<StandardTetris, 4, 1>::new()
                .id_u64(&board)
                .is_ok()
        );
        assert!(
            BoardCodec::<StandardTetris, 5, 1>::new()
                .id_u64(&board)
                .is_ok()
        );
        assert!(
            BoardCodec::<StandardTetris, 6, 1>::new()
                .id_u64(&board)
                .is_ok()
        );
        eprintln!("HIGHER_HISTORY_LOW_TARGET seed={seed} max_height=2 target_height=1");
    }

    #[test]
    fn every_supported_width_matches_an_independent_complete_engine_closure() {
        macro_rules! check {
            ($cols:literal) => {{
                type C = crate::fixtures::Game<$cols, 20>;
                let codec = BoardCodec::<C, 1, 0>::new();
                let reachable = single_row_reachable($cols);
                assert_eq!(codec.len(), board_count($cols));
                assert_eq!(codec.table_payload_bytes(), 0);
                assert_eq!(
                    usize::try_from(codec.len()).expect("tiny count fits usize"),
                    reachable.len()
                );
                for row in 0u16..(1 << $cols) {
                    assert_eq!(codec.id_rows_u64(&[row]).is_ok(), reachable.contains(&row));
                }
                for id in 0..codec.len() {
                    let row = codec.rows_from_u64(id).expect("dense ID decodes");
                    assert!(reachable.contains(&row[0]));
                    assert_eq!(codec.id_rows_u64(&row), Ok(id));
                }
            }};
        }
        check!(8);
        check!(9);
        check!(10);
        check!(11);
        check!(12);
    }
}
