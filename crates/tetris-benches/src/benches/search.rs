//! Scoring and bounded sequence-search benchmarks.

use crate::fixtures::game::{self as fixtures, GAMEPLAY_SEEDS};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};

use tetris_game::{TetrisGame, TetrisPiecePlacement};
use tetris_search::{
    BeamTetrisState, TetrisBeamSearch, TetrisBoardScoreState, height_mse_beam_tetris_score,
    height_mse_board_score, height_mse_distance_from_empty,
};

// Complete bounded planner calls include scoring, candidate selection and frontier work.
const SEARCH_WIDTH: usize = 4;
const SEARCH_DEPTH: usize = 6;
const SEARCH_CAPACITY: usize = tetris_search::best_first_capacity(SEARCH_WIDTH, SEARCH_DEPTH);
type Planner = tetris_search::TetrisSequencePlanner<SEARCH_WIDTH, SEARCH_DEPTH, SEARCH_CAPACITY>;
type PlanOutcome = tetris_search::TetrisSequencePlanOutcome<SEARCH_DEPTH>;

fn search_digest(outcomes: &[PlanOutcome]) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut append = |word: u64| {
        for byte in word.to_le_bytes() {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        }
    };
    for outcome in outcomes {
        match outcome {
            PlanOutcome::Success(witness) => {
                append(1);
                for &col in witness.board.as_limbs() {
                    append(u64::from(col));
                }
                for placement in witness.placements {
                    append(u64::from(placement.index()));
                }
                append(witness.nodes_expanded as u64);
                append(witness.terminal_candidates_examined as u64);
            }
            PlanOutcome::Exhausted {
                nodes_expanded,
                terminal_candidates_examined,
            } => {
                append(0);
                append(*nodes_expanded as u64);
                append(*terminal_candidates_examined as u64);
            }
        }
    }
    digest
}

// Match tetris_demo_single's production beam width/depth and action capacity.
type ProductionBeam =
    TetrisBeamSearch<256, 8, { <tetris_game::TetrisPieceOrientation<10>>::TOTAL_NUM_ORIENTATIONS }>;

fn validate_beam_result(game: TetrisGame, result: &tetris_search::ScoredState<BeamTetrisState>) {
    let action = result.root_action.expect("beam must select a first move");
    assert!(<TetrisPiecePlacement<10>>::all_from_piece(game.current_piece()).contains(&action));
    let mut board = game.board;
    assert!(!board.apply_piece_placement(action).is_lost);
    assert!(!result.state.0.board.is_lost());
    assert_eq!(result.depth, 8);
    assert!(result.score.is_finite());
}

fn validate_sequence_results(outcomes: &[PlanOutcome]) {
    assert_eq!(search_digest(outcomes), 0xfb8b_574c_2ed0_26f5);
}

tetris_bench! {
    fn bench_search_fixed(c: &mut Criterion) {
        let input = fixtures::search_inputs::<SEARCH_DEPTH>();
        let mut planner = Planner::new_boxed();
        let mut output = input
            .iter()
            .map(|(board, pieces)| {
                planner.search(*board, pieces, &tetris_search::height_mse_plan_score)
            })
            .collect::<Vec<_>>();
        // Scalar reference includes witness boards, moves, and exact search work.
        validate_sequence_results(&output);
        let mut group = c.benchmark_group("search/sequence_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for ((board, pieces), out) in black_box(input).iter().zip(&mut output) {
                        *out =
                            planner.search(*board, pieces, &tetris_search::height_mse_plan_score);
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

tetris_bench! {
    #[allow(clippy::cast_possible_truncation, reason = "Fixture line counts are modulo five and GAMEPLAY_MOVES is a small fixed benchmark constant.")]
    fn bench_scores(c: &mut Criterion) {
        for (name, boards) in fixtures::board_cases()
            .into_iter()
            .filter(|(name, _)| *name != "lost")
        {
            let mut output = vec![0.0; boards.len()];
            let mut distance = c.benchmark_group("search/distance_fixed");
            distance.throughput(Throughput::Elements(boards.len() as u64));
            distance.bench_with_input(name, &boards, |b, boards| {
                b.iter(|| {
                    for (&board, out) in black_box(boards).iter().zip(&mut output) {
                        *out = height_mse_distance_from_empty(board);
                    }
                    black_box(&output);
                });
            });
            distance.finish();
            let states = boards
                .into_iter()
                .enumerate()
                .map(|(i, board)| TetrisBoardScoreState {
                    board,
                    recent_lines_cleared: (i % 5) as u32,
                })
                .collect::<Vec<_>>();
            let mut score = c.benchmark_group("search/score_fixed");
            score.throughput(Throughput::Elements(states.len() as u64));
            score.bench_with_input(name, &states, |b, states| {
                b.iter(|| {
                    for (state, out) in black_box(states).iter().zip(&mut output) {
                        *out = height_mse_board_score(state);
                    }
                    black_box(&output);
                });
            });
            score.finish();
        }
    }
}

tetris_bench! {
    fn bench_production_beam(c: &mut Criterion) {
        let input = fixtures::beam_inputs();
        let mut search = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        let mut output = vec![None; input.len()];
        for (&game, out) in input.iter().zip(&mut output) {
            *out = search.search_top_with_state_best_effort(BeamTetrisState::new(game), 8);
            validate_beam_result(game, out.as_ref().expect("live beam fixture"));
        }
        let mut group = c.benchmark_group("search/beam_fixed");
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("w256_d8", input.len()),
            &input,
            |b, input| {
                b.iter(|| {
                    for (&game, out) in black_box(input).iter().zip(&mut output) {
                        *out =
                            search.search_top_with_state_best_effort(BeamTetrisState::new(game), 8);
                    }
                    black_box(&output);
                });
            },
        );
        group.finish();
    }
}

const GAMEPLAY_MOVES: usize = 16;

fn play_game(search: &mut ProductionBeam, mut game: TetrisGame) -> Option<TetrisGame> {
    for _ in 0..GAMEPLAY_MOVES {
        let action = search
            .search_top_with_state_best_effort(BeamTetrisState::new(game), 8)?
            .root_action?;
        if game.apply_placement(action).is_lost {
            return None;
        }
    }
    Some(game)
}

tetris_bench! {
    #[allow(clippy::cast_possible_truncation, reason = "Fixture line counts are modulo five and GAMEPLAY_MOVES is a small fixed benchmark constant.")]
    fn bench_gameplay(c: &mut Criterion) {
        let input = GAMEPLAY_SEEDS.map(<TetrisGame>::new_with_seed);
        let mut search = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        let mut output = [None; GAMEPLAY_SEEDS.len()];
        for (&game, out) in input.iter().zip(&mut output) {
            *out = play_game(&mut search, game);
            assert_eq!(
                out.as_ref().expect("gameplay fixture must survive").piece_count,
                GAMEPLAY_MOVES as u32,
            );
        }
        let mut group = c.benchmark_group("search/gameplay_fixed");
        group.throughput(Throughput::Elements((GAMEPLAY_SEEDS.len() * GAMEPLAY_MOVES) as u64));
        group.bench_with_input("w256_d8_seeds1_4_moves16", &input, |b, input| {
            b.iter(|| {
                for (&game, out) in black_box(input).iter().zip(&mut output) {
                    *out = Some(
                        play_game(&mut search, game).expect("gameplay must complete every move"),
                    );
                }
                black_box(&output);
            });
        });
        group.finish();
    }
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
    #[ignore = "production-sized gameplay; run in release mode"]
    fn gameplay_fixture_survives_and_repeats_the_seeded_draw_stream() {
        let mut first = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        let mut second = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        for seed in GAMEPLAY_SEEDS {
            let initial = <TetrisGame>::new_with_seed(seed);
            let a = play_game(&mut first, initial).expect("game must survive every move");
            let b = play_game(&mut second, initial).expect("repeated game must survive");
            assert_eq!(a, b);
            assert_eq!(a.piece_count, GAMEPLAY_MOVES as u32);
            assert!(!a.board.is_lost());
            let mut draws = fixtures::BagFixture {
                bag: *initial.bag(),
                rng: initial.rng,
            };
            let mut piece = initial.current_piece();
            for _ in 0..GAMEPLAY_MOVES {
                piece = draws.draw(tetris_game::TetrisPieceBagState::full());
            }
            assert_eq!(a.current_piece(), piece);
            assert_eq!(*a.bag(), draws.bag);
            assert_eq!(a.rng, draws.rng);
        }
    }

    #[test]
    #[ignore = "production-sized beam replay; run in release mode"]
    fn production_beam_fixture_is_legal_and_deterministic() {
        let input = fixtures::beam_inputs();
        let mut first = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        let mut second = Box::new(ProductionBeam::new(height_mse_beam_tetris_score));
        for game in input {
            let a = first
                .search_top_with_state_best_effort(BeamTetrisState::new(game), 8)
                .expect("live fixture beam search must return a result in production_beam_fixture_is_legal_and_deterministic");
            let b = second
                .search_top_with_state_best_effort(BeamTetrisState::new(game), 8)
                .expect("live fixture beam search must return a result in production_beam_fixture_is_legal_and_deterministic");
            validate_beam_result(game, &a);
            assert_eq!(a.state.0, b.state.0);
            assert_eq!(a.root_action, b.root_action);
            assert_eq!(a.score.to_bits(), b.score.to_bits());
            assert_eq!(a.depth, b.depth);
        }
    }

    #[test]
    fn bounded_search_fixture_is_reproducible() {
        let input = fixtures::search_inputs::<SEARCH_DEPTH>();
        assert_eq!(input, fixtures::search_inputs::<SEARCH_DEPTH>());
        assert_eq!(input.len(), 16);
        let mut planner = Planner::new_boxed();
        let output = input
            .iter()
            .map(|(board, pieces)| {
                planner.search(*board, pieces, &tetris_search::height_mse_plan_score)
            })
            .collect::<Vec<_>>();
        assert!(
            output
                .iter()
                .all(|out| matches!(out, PlanOutcome::Success(_)))
        );
        assert_eq!(search_digest(&output), 0xfb8b_574c_2ed0_26f5);
    }
}
