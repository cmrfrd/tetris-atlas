// NOTE: This module contains the benchmark bodies. It is enabled only
// when the `bench` feature is active. The files under `benches/` are
// minimal wrappers that register these functions with Criterion.

use crate::tetris::{
    TetrisBoard, TetrisGame, TetrisGameRng, TetrisPiece, TetrisPieceBag, TetrisPiecePlacement,
};
use crate::utils::rshift_slice_from_mask_u32;
use crate::{beam_search::BeamSearch, tetris::TetrisPieceOrientation};
use criterion::{BenchmarkId, Criterion, black_box};
use rand::Rng;
use rand::seq::IndexedRandom;

const NUM_ELEMS: usize = 10_000;

/// Registered benchmark entrypoint.
///
/// Bench functions register themselves via `#[tetris_bench]`, which submits a function pointer
/// into this registry. This avoids maintaining a manual list of targets.
pub struct BenchSpec {
    pub f: fn(&mut Criterion),
}

inventory::collect!(BenchSpec);

/// Declarative helper that defines a benchmark entrypoint function and registers it into the
/// `inventory` bench registry. This avoids maintaining a central list while also avoiding a
/// proc-macro attribute.
macro_rules! tetris_bench {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident ( $c:ident : &mut Criterion ) $body:block
    ) => {
        $(#[$meta])*
        $vis fn $name($c: &mut Criterion) $body

        ::inventory::submit! {
            BenchSpec { f: $name }
        }
    };
}

pub fn all_benches(c: &mut Criterion) {
    for spec in inventory::iter::<BenchSpec> {
        (spec.f)(c);
    }
}

/// Backwards-compatible alias for the previous "all benches" entrypoint.
pub fn tetris_game(c: &mut Criterion) {
    all_benches(c);
}

tetris_bench! {
    pub fn bench_rshift_slice_from_mask_u32(c: &mut Criterion) {
        let mut rng = rand::rng();

        const N: usize = 10;
        const ITERS: usize = 4;
        let data: Vec<([u32; N], u32)> = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let num = [rng.random::<u32>(); N];
                    let mask = {
                        let mut m = 0u32;
                        let num_bits = rng.random::<u32>() % 5; // 0-4 bits
                        for _ in 0..num_bits {
                            m |= 1 << (rng.random::<u32>() % 32);
                        }
                        m
                    };
                    (num, mask)
                })
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("rshift_slice_from_mask_u32", format!("{}_{N}", NUM_ELEMS)),
            &(NUM_ELEMS, N),
            |b, _| {
                b.iter(|| {
                    data.iter()
                        .map(|(x, m)| {
                            let mut x = *x;
                            rshift_slice_from_mask_u32::<N, ITERS>(&mut x, *m);
                            x
                        })
                        .collect::<Vec<_>>()
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_num_rotations(c: &mut Criterion) {
        let rng = rand::rng();
        let pieces = black_box(
            rng.random_iter::<TetrisPiece>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("num_rotations", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| b.iter(|| pieces.iter().map(|p| p.num_rotations()).collect::<Vec<_>>()),
        );
    }
}

tetris_bench! {
    pub fn bench_width(c: &mut Criterion) {
        let rng = rand::rng();
        let placements = black_box(
            rng.random_iter::<TetrisPiecePlacement>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(BenchmarkId::new("width", NUM_ELEMS), &NUM_ELEMS, |b, _| {
            b.iter(|| {
                placements
                    .iter()
                    .map(|pl| pl.piece.width(pl.orientation.rotation))
                    .collect::<Vec<_>>()
            })
        });
    }
}

tetris_bench! {
    pub fn bench_height(c: &mut Criterion) {
        let rng = rand::rng();
        let placements = black_box(
            rng.random_iter::<TetrisPiecePlacement>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(BenchmarkId::new("height", NUM_ELEMS), &NUM_ELEMS, |b, _| {
            b.iter(|| {
                placements
                    .iter()
                    .map(|pl| pl.piece.height(pl.orientation.rotation))
                    .collect::<Vec<_>>()
            })
        });
    }
}

tetris_bench! {
    pub fn bench_bag_iter(c: &mut Criterion) {
        let mut rng = TetrisGameRng::new(42);
        let mut bag = TetrisPieceBag::new_rand(&mut rng);
        c.bench_with_input(
            BenchmarkId::new("bag_iter", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| {
                b.iter(|| {
                    for _ in 0..NUM_ELEMS {
                        black_box(bag.rand_next(&mut rng));
                    }
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_bag_rand_next(c: &mut Criterion) {
        let mut rng = TetrisGameRng::new(42);
        let mut bag = TetrisPieceBag::new_rand(&mut rng);
        c.bench_with_input(BenchmarkId::new("bag_rand_next", 1), &1, |b, _| {
            b.iter(|| {
                bag.rand_next(&mut rng);
            })
        });
    }
}

tetris_bench! {
    pub fn bench_is_lost(c: &mut Criterion) {
        let mut rng = rand::rng();
        let single_bit_boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let mut b = TetrisBoard::new();
                    b.set_random_bits(1, &mut rng);
                    b
                })
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("is_lost", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| {
                b.iter(|| {
                    single_bit_boards
                        .iter()
                        .map(|p| p.is_lost())
                        .collect::<Vec<_>>()
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_count(c: &mut Criterion) {
        let mut rng = rand::rng();
        let boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let mut b = TetrisBoard::new();
                    b.set_random_bits(1024, &mut rng);
                    b
                })
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(BenchmarkId::new("count", NUM_ELEMS), &NUM_ELEMS, |b, _| {
            b.iter(|| boards.iter().map(|p| p.count()).collect::<Vec<_>>())
        });
    }
}

tetris_bench! {
    pub fn bench_clear_filled_rows(c: &mut Criterion) {
        let mut rng = rand::rng();
        let mut boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let mut b = TetrisBoard::new();
                    b.set_random_bits(1024, &mut rng);
                    b
                })
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("clear_filled_rows", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| {
                b.iter(|| {
                    boards
                        .iter_mut()
                        .map(|p| p.clear_filled_rows())
                        .collect::<Vec<_>>()
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_clear(c: &mut Criterion) {
        let mut rng = rand::rng();
        let mut boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let mut b = TetrisBoard::new();
                    b.set_random_bits(1024, &mut rng);
                    b
                })
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(BenchmarkId::new("clear", NUM_ELEMS), &NUM_ELEMS, |b, _| {
            b.iter(|| {
                for board in boards.iter_mut() {
                    black_box(board.clear());
                }
            })
        });
    }
}

tetris_bench! {
    pub fn bench_apply_piece_placement(c: &mut Criterion) {
        let rng = rand::rng();
        let mut empty_boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| TetrisBoard::new())
                .collect::<Vec<_>>(),
        );
        let placements = black_box(
            rng.random_iter::<TetrisPiecePlacement>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("apply_piece_placement", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| {
                b.iter(|| {
                    empty_boards
                        .iter_mut()
                        .zip(placements.iter())
                        .map(|(b, &p)| b.apply_piece_placement(p))
                        .collect::<Vec<_>>()
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_from_binary_slice(c: &mut Criterion) {
        let mut rng = rand::rng();
        let boards = black_box(
            (0..NUM_ELEMS)
                .map(|_| {
                    let mut b = TetrisBoard::new();
                    b.set_random_bits(1024, &mut rng);
                    b
                })
                .collect::<Vec<_>>(),
        );
        let binary_slices = black_box(
            boards
                .iter()
                .map(|b| b.to_binary_slice())
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("from_binary_slice", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| {
                b.iter(|| {
                    binary_slices
                        .iter()
                        .map(|b| TetrisBoard::from_binary_slice(*b))
                        .collect::<Vec<_>>()
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_placement_index(c: &mut Criterion) {
        let rng = rand::rng();
        let placements = black_box(
            rng.random_iter::<TetrisPiecePlacement>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("placement_index", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| b.iter(|| placements.iter().map(|p| p.piece.index()).collect::<Vec<_>>()),
        );
    }
}

// tetris_bench! {
//     pub fn bench_placements_from_piece(c: &mut Criterion) {
//         let rng = rand::rng();
//         let pieces = black_box(
//             rng.random_iter::<TetrisPiece>()
//                 .take(NUM_ELEMS)
//                 .collect::<Vec<_>>(),
//         );
//         c.bench_function("placements_from_piece", |b| {
//             b.iter(|| {
//                 pieces
//                     .iter()
//                     .map(|p| TetrisPiecePlacement::all_from_piece(*p))
//                     .collect::<Vec<_>>()
//             })
//         });
//     }
// }

tetris_bench! {
    pub fn bench_play_placements(c: &mut Criterion) {
        let mut rng = rand::rng();
        let num_games: usize = 1_000;
        let seed = 123;
        c.bench_with_input(
            BenchmarkId::new("play_placements", num_games),
            &num_games,
            |b, &num_games| {
                let mut counter = 0usize;
                let mut game = TetrisGame::new_with_seed(seed);
                b.iter(|| {
                    counter = 0;
                    game.reset(Some(seed));

                    while counter < num_games {
                        let placement = *game.current_placements().choose(&mut rng).unwrap();
                        let placement_result = game.apply_placement(placement);
                        if placement_result.is_lost.into() {
                            counter += 1;
                            game.reset(Some(seed));
                        }
                    }
                })
            },
        );
    }
}

tetris_bench! {
    pub fn bench_beam_search_tetris(c: &mut Criterion) {
        const BEAM_WIDTH: usize = 128;
        const MAX_DEPTH: usize = 8;
        const LOOKAHEAD: usize = 8;
        const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
        const STEPS_PER_ITER: usize = 64;
        const SEED: u64 = 123;
        use crate::beam_search::BeamTetrisState;

        let mut search = BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();
        let mut game = TetrisGame::new_with_seed(SEED);

        // c.bench_with_input(
        //     BenchmarkId::new("beam_search_tetris", format!("bw{BEAM_WIDTH}_d{LOOKAHEAD}_steps{STEPS_PER_ITER}")),
        //     &(BEAM_WIDTH, LOOKAHEAD, STEPS_PER_ITER),
        //     |b, _| {
        //         b.iter(|| {
        //             // Keep the benchmark steady by resetting on loss.
        //             for _ in 0..STEPS_PER_ITER {
        //                 if game.board.is_lost() {
        //                     game.reset(Some(SEED));
        //                 }
        //                 let state = BeamTetrisState(game);
        //                 let mv = search
        //                     .search_first_action_with_state(black_box(state), LOOKAHEAD)
        //                     .unwrap();
        //                 let res = game.apply_placement(black_box(mv));
        //                 black_box(res);
        //             }
        //         })
        //     },
        // );
    }
}

tetris_bench! {
    pub fn bench_bits_to_byte(c: &mut Criterion) {
        let rng = rand::rng();
        let bits = black_box(
            rng.random_iter::<[u8; 8]>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("bits_to_byte", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| b.iter(|| bits.iter().map(|bits| crate::utils::bits_to_byte(bits)).collect::<Vec<_>>()),
        );
    }
}

tetris_bench! {
    pub fn bench_byte_to_bits(c: &mut Criterion) {
        let rng = rand::rng();
        let bytes = black_box(
            rng.random_iter::<u8>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new("byte_to_bits", NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| b.iter(|| bytes.iter().map(|b| crate::utils::byte_to_bits(*b)).collect::<Vec<_>>()),
        );
    }
}

tetris_bench! {
    pub fn bitmask_as_slice(c: &mut Criterion) {
        const N: usize = 40;
        let rng = rand::rng();
        let bits = black_box(
            rng.random_iter::<u64>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        );
        c.bench_with_input(
            BenchmarkId::new(format!("bitmask_as_slice_{N}"), NUM_ELEMS),
            &NUM_ELEMS,
            |b, _| b.iter(|| bits.iter().map(|bits| crate::utils::BitMask::<N>::new_from_u64(*bits).as_slice()).collect::<Vec<_>>()),
        );
    }
}
