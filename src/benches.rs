// NOTE: This module contains the benchmark bodies. It is enabled only
// when the `bench` feature is active. The files under `benches/` are
// minimal wrappers that register these functions with Criterion.

use crate::tetris::{
    IsLost, TetrisBoard, TetrisGame, TetrisGameRng, TetrisPiece, TetrisPieceBag,
    TetrisPiecePlacement,
};
use crate::utils::rshift_slice_from_mask_u32;
use criterion::{BenchmarkId, Criterion, black_box};
use rand::Rng;
use rand::seq::{IndexedRandom, IteratorRandom};

pub fn tetris_game(c: &mut Criterion) {
    const NUM_ELEMS: usize = 10_000;
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

    let rng = rand::rng();
    let placements = black_box(
        rng.random_iter::<TetrisPiecePlacement>()
            .take(NUM_ELEMS)
            .map(|p| p.piece_orientation())
            .collect::<Vec<_>>(),
    );
    c.bench_with_input(BenchmarkId::new("width", NUM_ELEMS), &NUM_ELEMS, |b, _| {
        b.iter(|| {
            placements
                .iter()
                .map(|(p, o)| p.width(o.rotation))
                .collect::<Vec<_>>()
        })
    });
    c.bench_with_input(BenchmarkId::new("height", NUM_ELEMS), &NUM_ELEMS, |b, _| {
        b.iter(|| {
            placements
                .iter()
                .map(|(p, o)| p.height(o.rotation))
                .collect::<Vec<_>>()
        })
    });

    let mut bag = TetrisPieceBag::new();
    c.bench_with_input(
        BenchmarkId::new("bag_iter", NUM_ELEMS),
        &NUM_ELEMS,
        |b, _| {
            b.iter(|| {
                for _ in 0..NUM_ELEMS {
                    bag = bag.next_bags().next().unwrap().0;
                }
            })
        },
    );

    c.bench_with_input(BenchmarkId::new("bag_rand_next", 1), &1, |b, _| {
        let mut bag = TetrisPieceBag::new();
        let mut tetris_rng = TetrisGameRng::new(42);
        b.iter(|| {
            bag.rand_next(&mut tetris_rng);
        })
    });

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

    let mut boards = black_box(
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

    c.bench_with_input(BenchmarkId::new("clear", NUM_ELEMS), &NUM_ELEMS, |b, _| {
        b.iter(|| {
            for board in boards.iter_mut() {
                black_box(board.clear());
            }
        })
    });

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

    c.bench_with_input(
        BenchmarkId::new("placement_index", NUM_ELEMS),
        &NUM_ELEMS,
        |b, _| b.iter(|| placements.iter().map(|p| p.index()).collect::<Vec<_>>()),
    );

    let mut rng = rand::rng();
    let pieces = {
        black_box(
            rng.random_iter::<TetrisPiece>()
                .take(NUM_ELEMS)
                .collect::<Vec<_>>(),
        )
    };
    c.bench_function("placements_from_piece", |b| {
        b.iter(|| {
            pieces
                .iter()
                .map(|p| TetrisPiecePlacement::all_from_piece(*p))
                .collect::<Vec<_>>()
        })
    });

    let mut rng = rand::rng();
    c.bench_function("play_placements", |b| {
        let mut counter = 0;
        let num_games = 1_000;
        let mut game = TetrisGame::new_with_seed(123);
        b.iter(|| {
            counter = 0;
            game.reset(None);

            while counter < num_games {
                let placement = *game.current_placements().choose(&mut rng).unwrap();
                let is_lost = game.apply_placement(placement);
                if is_lost.into() {
                    counter += 1;
                    game.reset(None);
                }
            }
        })
    });
}
