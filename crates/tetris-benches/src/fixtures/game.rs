//! Shared deterministic benchmark inputs.

use super::SEED;
use tetris_game::{
    StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisGameRng, TetrisPiece,
    TetrisPieceBag, TetrisPieceBagState, TetrisPiecePlacement,
};

pub const GAME_SEEDS: std::ops::RangeInclusive<u64> = 1..=16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementFixture {
    pub board: TetrisBoard<10, 20>,
    pub placement: TetrisPiecePlacement<10>,
}

/// All candidate moves at 64 successive reachable positions for each of 16 seeds.
/// The fixture policy is deliberately fixed and uses integer features only.
pub fn placements() -> Vec<PlacementFixture> {
    let mut fixtures = Vec::new();
    for seed in GAME_SEEDS {
        let mut game: TetrisGame = <TetrisGame>::new_with_seed(seed);
        for _ in 0..64 {
            let mut best = None;
            for &placement in game.current_placements() {
                fixtures.push(PlacementFixture {
                    board: game.board,
                    placement,
                });

                let mut next = game.board;
                let result = next.apply_piece_placement(placement);
                let key = (
                    result.is_lost,
                    next.total_holes() * 16 + next.height() * 4 + next.roughness(),
                    placement.index(),
                );
                if best.is_none_or(|(old, _)| key < old) {
                    best = Some((key, placement));
                }
            }

            if let Some((_, placement)) = best
                && game.apply_placement(placement).is_lost
            {
                game.reset(Some(seed));
            }
        }
    }

    fixtures
}

/// Independent columns, with exactly 0..4 full playable rows per board.
#[allow(
    clippy::cast_possible_truncation,
    reason = "Fixture generation intentionally uses the low random word, masked to the playable rows."
)]
pub fn rows() -> Vec<TetrisBoard<10, 20>> {
    let mut rng = TetrisGameRng::new(SEED);
    (0..1_280)
        .map(|i| {
            let mut mask = 0u32;
            while mask.count_ones() < i % 5 {
                mask |= 1 << (rng.next_u64() % StandardTetris::ROWS as u64);
            }

            let mut cols = [0; 10];
            for col in &mut cols {
                *col = (rng.next_u64() as u32 & ((1 << StandardTetris::ROWS) - 1)) | mask;
            }

            let full = cols.iter().copied().fold(u32::MAX, |a, b| a & b);
            cols[0] &= !(full & !mask);
            TetrisBoard::<10, 20>::from_limbs(cols)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BagFixture {
    pub bag: TetrisPieceBag,
    pub rng: TetrisGameRng,
}

impl BagFixture {
    // Explicit refill policy for the bag draw/refill fixtures.
    pub fn draw(&mut self, pieces: TetrisPieceBagState) -> TetrisPiece {
        if self.bag.count() == 0 {
            self.bag.refill(pieces);
            self.bag.shuffle(&mut self.rng);
        }

        self.bag.draw().expect("nonempty fixture piece set")
    }
}

/// Independent snapshots keep the draw and refill branches separate on every iteration.
pub fn bags(piece_set: u8, empty: bool) -> Vec<BagFixture> {
    (1..=128)
        .map(|seed| {
            let mut rng = TetrisGameRng::new(seed);
            let mut bag = TetrisPieceBag::from_bag_state(
                TetrisPieceBagState::try_from(piece_set).expect("valid bag mask"),
            );
            bag.shuffle(&mut rng);
            if empty {
                while bag.count() != 0 {
                    let _ = bag.draw();
                }
            }
            BagFixture { bag, rng }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayFixture {
    pub seed: u64,
    pub actions: Vec<TetrisPiecePlacement<10>>,
    pub expected: TetrisGame,
}

/// A fixed legal policy trace: setup chooses actions, timing only replays them.
pub fn replays() -> Vec<ReplayFixture> {
    GAME_SEEDS
        .map(|seed| {
            let mut game: TetrisGame = <TetrisGame>::new_with_seed(seed);
            let mut actions = Vec::with_capacity(64);
            for _ in 0..64 {
                let placement = *game
                    .current_placements()
                    .iter()
                    .min_by_key(|&&placement| {
                        let mut next = game.board;
                        let result = next.apply_piece_placement(placement);
                        (
                            result.is_lost,
                            next.total_holes() * 16 + next.height() * 4 + next.roughness(),
                            placement.index(),
                        )
                    })
                    .expect("current piece has placements");
                let result = game.apply_placement(placement);
                assert!(!result.is_lost, "fixed replay must stay live");
                actions.push(placement);
            }
            ReplayFixture {
                seed,
                actions,
                expected: game,
            }
        })
        .collect()
}

/// Reachable positions plus explicit survival/loss boundaries.
pub fn board_cases() -> Vec<(&'static str, Vec<TetrisBoard<10, 20>>)> {
    let mut reachable = Vec::new();
    for replay in replays() {
        let mut game: TetrisGame = <TetrisGame>::new_with_seed(replay.seed);
        for action in replay.actions {
            reachable.push(game.board);
            game.apply_placement(action);
        }
    }

    let mut near_top = Vec::new();
    let mut lost = Vec::new();
    for column in 0..StandardTetris::COLS {
        for height in [19, 20, 21] {
            let mut columns = [0; 10];
            columns[column] = (1u32 << height) - 1;
            let board = TetrisBoard::<10, 20>::from_limbs(columns);
            if height <= StandardTetris::ROWS {
                near_top.push(board);
            } else {
                lost.push(board);
            }
        }
    }
    lost.push(TetrisBoard::<10, 20>::from_limbs([u32::MAX; 10]));
    vec![
        ("reachable", reachable),
        ("near_top", near_top),
        ("lost", lost),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementBoundary {
    pub name: &'static str,
    pub input: PlacementFixture,
    pub lines_cleared: u32,
    pub is_lost: bool,
}

/// Explicit clear counts and the rule that line clearing precedes loss detection.
pub fn placement_boundaries() -> Vec<PlacementBoundary> {
    let vertical_i = *<TetrisPiecePlacement<10>>::all_from_piece(TetrisPiece::I)
        .iter()
        .find(|p| {
            p.orientation.rotation == tetris_game::Rotation::R90
                && usize::from(p.orientation.column.inner()) == 9
        })
        .expect("rightmost vertical I placement");
    let right_o = *<TetrisPiecePlacement<10>>::all_from_piece(TetrisPiece::O)
        .last()
        .expect("rightmost O placement");
    let mut cases = vec![PlacementBoundary {
        name: "no_clear",
        input: PlacementFixture {
            board: TetrisBoard::<10, 20>::new(),
            placement: right_o,
        },
        lines_cleared: 0,
        is_lost: false,
    }];
    for (name, rows) in [
        ("clear_1", 1),
        ("clear_2", 2),
        ("clear_3", 3),
        ("clear_4", 4),
    ] {
        let mut columns = [(1u32 << rows) - 1; 10];
        columns[9] = 0;
        cases.push(PlacementBoundary {
            name,
            input: PlacementFixture {
                board: TetrisBoard::<10, 20>::from_limbs(columns),
                placement: vertical_i,
            },
            lines_cleared: rows,
            is_lost: false,
        });
    }

    let mut columns = [0; 10];
    columns[9] = (1 << StandardTetris::ROWS) - 1;
    let high_board = TetrisBoard::<10, 20>::from_limbs(columns);
    cases.push(PlacementBoundary {
        name: "near_top",
        input: PlacementFixture {
            board: high_board,
            placement: <TetrisPiecePlacement<10>>::ALL_PLACEMENTS[0],
        },
        lines_cleared: 0,
        is_lost: false,
    });
    cases.push(PlacementBoundary {
        name: "topout",
        input: PlacementFixture {
            board: high_board,
            placement: right_o,
        },
        lines_cleared: 0,
        is_lost: true,
    });

    let mut columns = [((1 << StandardTetris::ROWS) - 1); 10];
    columns[8] >>= 1;
    columns[9] = 0;

    // O reaches row 20, then clearing row 19 brings it back inside the board.
    cases.push(PlacementBoundary {
        name: "clear_before_loss",
        input: PlacementFixture {
            board: TetrisBoard::<10, 20>::from_limbs(columns),
            placement: right_o,
        },
        lines_cleared: 1,
        is_lost: false,
    });
    columns[8] = (1 << StandardTetris::ROWS) - 1;
    cases.push(PlacementBoundary {
        name: "near_top_clear_4",
        input: PlacementFixture {
            board: TetrisBoard::<10, 20>::from_limbs(columns),
            placement: vertical_i,
        },
        lines_cleared: 4,
        is_lost: false,
    });

    cases
}

pub type StateKey = (TetrisBoard<10, 20>, TetrisPiece);

pub fn state_keys() -> Vec<StateKey> {
    let mut keys = Vec::new();
    for replay in replays() {
        let mut game: TetrisGame = <TetrisGame>::new_with_seed(replay.seed);
        for action in replay.actions {
            keys.push((game.board, game.current_piece()));
            game.apply_placement(action);
        }
    }

    keys
}

pub fn bag_states() -> Vec<(TetrisPieceBagState, TetrisPiece)> {
    (0..=TetrisPieceBagState::FULL_MASK)
        .flat_map(|mask| {
            TetrisPiece::all().map(|piece| {
                (
                    TetrisPieceBagState::try_from(mask).expect("valid bag mask"),
                    piece,
                )
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasFixture {
    pub board: TetrisBoard<10, 20>,
    pub bag: TetrisPieceBagState,
}

/// Reachable parent states spanning every remaining-bag size, including refill boundaries.
pub fn atlas_parents() -> Vec<AtlasFixture> {
    let mut parents = Vec::new();
    for replay in replays() {
        let mut game: TetrisGame = <TetrisGame>::new_with_seed(replay.seed);
        for (step, action) in replay.actions.into_iter().enumerate() {
            if step % 8 == 0 {
                // Atlas states include the not-yet-placed current piece in the remainder.
                let mask = TetrisPiece::all()
                    .into_iter()
                    .filter(|&p| game.bag().contains(p))
                    .fold(game.current_piece().mask(), |mask, p| mask | p.mask());
                parents.push(AtlasFixture {
                    board: game.board,
                    bag: TetrisPieceBagState::try_from(mask).expect("valid bag mask"),
                });
            }
            game.apply_placement(action);
        }
    }

    parents
}

/// Fixed seeds for bounded production-beam gameplay.
pub(crate) const GAMEPLAY_SEEDS: [u64; 4] = [1, 2, 3, 4];

/// Builds a seeded, shuffled bag for the consumed-stream benchmark.
pub(crate) fn bag_stream() -> BagFixture {
    let mut rng = TetrisGameRng::new(42);
    let mut bag = TetrisPieceBag::from_bag_state(TetrisPieceBagState::full());
    bag.shuffle(&mut rng);
    BagFixture { bag, rng }
}

/// Enumerates every unordered piece set.
pub(crate) fn piece_sets() -> Vec<TetrisPieceBagState> {
    (0..=TetrisPieceBagState::FULL_MASK)
        .map(|mask| TetrisPieceBagState::try_from(mask).expect("valid mask"))
        .collect()
}

/// Covers every remaining count and requested piece with fixed shuffled bags.
pub(crate) fn bag_removals() -> Vec<(TetrisPieceBag, TetrisPiece)> {
    let mut input = Vec::new();
    for seed in 1..=128 {
        let mut rng = TetrisGameRng::new(seed);
        let mut bag = TetrisPieceBag::from_bag_state(TetrisPieceBagState::full());
        bag.shuffle(&mut rng);
        loop {
            for piece in TetrisPiece::all() {
                input.push((bag, piece));
            }

            if bag.count() == 0 {
                break;
            }

            let _ = bag.draw();
        }
    }

    input
}

/// Builds fixed board and draw-sequence inputs for bounded planning.
pub(crate) fn search_inputs<const DEPTH: usize>() -> Vec<(TetrisBoard<10, 20>, [TetrisPiece; DEPTH])>
{
    atlas_parents()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 8 == 3)
        .map(|(i, parent)| {
            let mut rng = tetris_game::TetrisGameRng::new((i / 8 + 1) as u64);
            let mut bag = tetris_game::TetrisPieceBag::from_bag_state(parent.bag);
            bag.shuffle(&mut rng);
            let mut state = BagFixture { bag, rng };
            (
                parent.board,
                std::array::from_fn(|_| state.draw(tetris_game::TetrisPieceBagState::full())),
            )
        })
        .collect()
}

/// Selects fixed reachable games for production-beam searches.
pub(crate) fn beam_inputs() -> Vec<TetrisGame> {
    replays()
        .into_iter()
        .take(4)
        .map(|fixture| fixture.expected)
        .collect()
}

/// A state lookup table using a deterministic hasher.
pub(crate) type StateMap = std::collections::HashMap<
    StateKey,
    usize,
    std::hash::BuildHasherDefault<std::hash::DefaultHasher>,
>;

/// Builds known-state queries together with explicit missing-state queries.
pub(crate) fn state_lookup(keys: &[StateKey]) -> (StateMap, Vec<StateKey>) {
    // Fixed hasher keys keep table layout comparable between implementations.
    let map: StateMap = keys
        .iter()
        .copied()
        .enumerate()
        .map(|(i, key)| (key, i))
        .collect();
    let mut queries = keys.to_vec();
    for &(board, _) in keys {
        if let Some(piece) = TetrisPiece::all()
            .into_iter()
            .find(|&piece| !map.contains_key(&(board, piece)))
        {
            queries.push((board, piece));
        }
    }

    assert!(
        queries.len() > keys.len(),
        "lookup fixtures must include misses"
    );
    (map, queries)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Fixture casts use explicitly bounded board dimensions and case counts, or deliberately extract test bits."
)]
mod tests {
    use super::*;

    #[test]
    fn placement_boundaries_cover_clears_and_post_clear_loss() {
        let cases = placement_boundaries();
        assert_eq!(cases, placement_boundaries());
        assert_eq!(cases.len(), 9);

        for rows in 0..=4 {
            assert!(
                cases
                    .iter()
                    .any(|case| case.lines_cleared == rows && !case.is_lost)
            );
        }

        for case in cases {
            assert!(!case.input.board.is_lost());
            assert_eq!(
                case.input
                    .board
                    .as_limbs()
                    .iter()
                    .copied()
                    .fold(u32::MAX, |rows, column| rows & column),
                0,
                "placement fixtures must not contain pre-existing full rows"
            );

            let mut board = case.input.board;
            let result = board.apply_piece_placement(case.input.placement);
            assert_eq!(result.lines_cleared, case.lines_cleared, "{}", case.name);
            assert_eq!(result.is_lost, case.is_lost, "{}", case.name);

            if case.is_lost {
                assert_eq!(board, TetrisBoard::<10, 20>::from_limbs([u32::MAX; 10]));
            }

            match case.name {
                "clear_4" => assert_eq!(board, TetrisBoard::<10, 20>::new()),
                "clear_before_loss" => assert_eq!(board.height(), 20),
                "near_top_clear_4" => assert_eq!(board.height(), 16),
                _ => {}
            }
        }
    }

    #[test]
    fn board_cases_cover_reachable_near_top_and_lost_states() {
        let cases = board_cases();
        assert_eq!(cases, board_cases());

        let reachable = &cases[0].1;
        assert_eq!(reachable.len(), 16 * 64);
        assert!(reachable.iter().all(|board| !board.is_lost()));
        assert!(reachable.iter().any(|board| board.count() == 0));
        assert!(reachable.iter().any(|board| board.count() > 0));

        for (name, boards) in cases {
            assert!(!boards.is_empty());

            for board in boards {
                assert_eq!(board.is_lost(), name == "lost");

                if name == "near_top" {
                    assert!([19, 20].contains(&board.height()));
                }

                if name != "lost" {
                    let cells = board.to_cell_array(tetris_game::Major::Row);
                    assert_eq!(
                        TetrisBoard::<10, 20>::from_cell_array(cells, tetris_game::Major::Row),
                        board
                    );
                }
            }
        }
    }

    #[test]
    fn bag_fixtures_exercise_exactly_the_requested_phase() {
        for piece_set in [0x7f, 0x13, 0x02] {
            for empty in [false, true] {
                let input = bags(piece_set, empty);
                assert_eq!(input, bags(piece_set, empty));
                assert_eq!(input.len(), 128);

                for template in input {
                    assert_eq!(template.bag.count() == 0, empty);

                    let mut state = template;
                    let piece =
                        state.draw(TetrisPieceBagState::try_from(piece_set).expect("valid mask"));
                    assert_ne!(piece.mask() & piece_set, 0);
                    assert!(!state.bag.contains(piece));
                    assert_eq!(state.bag.count(), piece_set.count_ones() as u8 - 1);
                    assert_eq!(state.rng == template.rng, !empty);
                }
            }
        }
    }

    #[test]
    fn fixed_replays_are_live_and_match_their_recorded_outcomes() {
        let input = replays();
        assert_eq!(input, replays());
        assert_eq!(input.len(), 16);

        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                digest = (digest ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
            }
        };

        for fixture in input {
            assert_eq!(fixture.actions.len(), 64);

            let mut game: TetrisGame = <TetrisGame>::new_with_seed(fixture.seed);
            feed(&fixture.seed.to_le_bytes());
            for action in fixture.actions {
                assert_eq!(action.piece, game.current_piece());
                feed(&[
                    action.piece.mask(),
                    action.orientation.index(),
                    action.index(),
                ]);
                assert!(!game.apply_placement(action).is_lost);
            }

            assert_eq!(game, fixture.expected);
            assert_eq!(game.piece_count, 64);
            assert!(game.lines_cleared > 0);

            for column in game.board.as_limbs() {
                feed(&column.to_le_bytes());
            }

            feed(&game.lines_cleared.to_le_bytes());
            feed(&[game.current_piece().mask(), game.bag().count()]);
            let mut draws = BagFixture {
                bag: *game.bag(),
                rng: game.rng,
            };

            for _ in 0..14 {
                feed(&[draws.draw(TetrisPieceBagState::full()).mask()]);
            }
        }

        for &placement in <TetrisPiecePlacement<10>>::ALL_PLACEMENTS {
            feed(&[
                placement.piece.mask(),
                placement.orientation.index(),
                placement.index(),
            ]);
        }
        use std::hash::{DefaultHasher, Hash, Hasher};
        for key in state_keys() {
            let mut hasher = DefaultHasher::new();
            key.hash(&mut hasher);
            feed(&hasher.finish().to_le_bytes());
        }

        for mask in [0x7f, 0x13, 0x02] {
            for empty in [false, true] {
                for mut fixture in bags(mask, empty) {
                    feed(&[fixture.bag.count(), mask]);
                    for _ in 0..14 {
                        feed(&[fixture
                            .draw(TetrisPieceBagState::try_from(mask).expect("valid mask"))
                            .mask()]);
                    }
                }
            }
        }

        // Compared between the baseline and candidate before accepting timings.
        println!("engine_fixture_digest={digest:016x}");
    }

    #[test]
    fn fixtures_repeat_and_cover_real_moves_and_clear_counts() {
        let placements = placements();
        assert_eq!(placements, super::placements());
        assert!(placements.len() > 20_000);
        assert!(placements.iter().all(|f| !f.board.is_lost()));
        assert!(placements.iter().any(|f| f.board.count() > 0));

        for (index, mut board) in rows().into_iter().enumerate() {
            assert_eq!(board.clear_and_collapse_rows(), index as u32 % 5);
        }
    }
}
