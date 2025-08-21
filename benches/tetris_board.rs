use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{
    Rng,
    seq::{IndexedRandom, IteratorRandom},
};
use tetris_atlas::tetris::{
    BitSetter, Clearer, Mergeable, Shiftable, TetrisBoard, TetrisBoardRaw, TetrisGame, TetrisPiece,
    TetrisPieceBag, TetrisPiecePlacement,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut bag = TetrisPieceBag::new();
    c.bench_function("bag_iter", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                bag = bag.next_bags().next().unwrap().0;
            }
        })
    });

    let mut rng = rand::rng();
    let mut bag = TetrisPieceBag::new();
    c.bench_function("bag_iter_random", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                bag = bag.next_bags().choose(&mut rng).unwrap().0;
            }
        })
    });

    let mut board = TetrisBoardRaw::default();
    c.bench_function("next_mut", |b| {
        b.iter(|| {
            for _ in 0..262_143 {
                board.next_mut();
            }
        })
    });

    let single_bit_boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = TetrisBoard::default();
                b.flip_random_bits(1, &mut rng);
                b
            })
            .collect::<Vec<_>>(),
    );
    c.bench_function("line_height", |b| {
        b.iter(|| {
            single_bit_boards
                .iter()
                .map(|p| p.height())
                .collect::<Vec<_>>()
        })
    });

    let mut boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = TetrisBoardRaw::default();
                b.flip_random_bits(1024, &mut rng);
                b
            })
            .collect::<Vec<_>>(),
    );

    c.bench_function("count", |b| {
        b.iter(|| boards.iter().map(|p| p.count()).collect::<Vec<_>>())
    });

    c.bench_function("clear_filled_rows", |b| {
        b.iter(|| {
            boards
                .iter_mut()
                .map(|p| p.clear_rows())
                .collect::<Vec<_>>()
        })
    });

    c.bench_function("loss", |b| {
        b.iter(|| boards.iter().map(|p| p.loss()).collect::<Vec<_>>())
    });

    c.bench_function("clear_all", |b| {
        b.iter(|| boards.iter_mut().map(|p| p.clear_all()).collect::<Vec<_>>())
    });

    c.bench_function("collides", |b| {
        b.iter(|| {
            boards
                .iter()
                .map(|p| p.collides(&boards[0]))
                .collect::<Vec<_>>()
        })
    });

    let mut shift_board = {
        let mut b = TetrisBoardRaw::default();
        b.flip_random_bits(16, &mut rng);
        b
    };
    c.bench_function("shift_down", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                shift_board.shift_down();
            }
        })
    });

    let mut shift_board = {
        let mut b = TetrisBoardRaw::default();
        b.flip_random_bits(16, &mut rng);
        b
    };
    c.bench_function("shift_down_from", |b| {
        b.iter(|| {
            for i in 0..10_000 {
                shift_board.shift_down_from(i % 20);
            }
        })
    });

    let mut shift_board = {
        let mut b = TetrisBoardRaw::default();
        b.flip_random_bits(16, &mut rng);
        b
    };
    c.bench_function("shift_up", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                shift_board.shift_up();
            }
        })
    });

    let base = {
        let mut b = TetrisBoardRaw::default();
        b.flip_random_bits(16, &mut rng);
        b
    };
    c.bench_function("merge", |b| {
        b.iter(|| {
            boards
                .iter_mut()
                .map(|p| p.merge(&base))
                .collect::<Vec<_>>()
        })
    });

    let mut empty_boards = black_box(
        (0..10_000)
            .map(|_| TetrisBoard::default())
            .collect::<Vec<_>>(),
    );
    let placements = black_box(
        rng.random_iter::<TetrisPiecePlacement>()
            .take(10_000)
            .collect::<Vec<_>>(),
    );
    c.bench_function("drop", |b| {
        b.iter(|| {
            empty_boards
                .iter_mut()
                .zip(placements.iter())
                .map(|(b, &p)| b.apply_piece_placement(p))
                .collect::<Vec<_>>()
        })
    });

    let mut rng = rand::rng();
    let boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = TetrisBoardRaw::default();
                b.flip_random_bits(1024, &mut rng);
                b
            })
            .collect::<Vec<_>>(),
    );
    c.bench_function("to_binary_slice", |b| {
        b.iter(|| {
            boards
                .iter()
                .map(|b| b.to_binary_slice())
                .collect::<Vec<_>>()
        })
    });

    let binary_slices = black_box(
        boards
            .iter()
            .map(|b| b.to_binary_slice())
            .collect::<Vec<_>>(),
    );
    c.bench_function("from_binary_slice", |b| {
        b.iter(|| {
            binary_slices
                .iter()
                .map(|b| TetrisBoard::from_binary_slice(*b))
                .collect::<Vec<_>>()
        })
    });

    c.bench_function("placement_index", |b| {
        b.iter(|| placements.iter().map(|p| p.index()).collect::<Vec<_>>())
    });

    let rng = rand::rng();
    let pieces = black_box(
        rng.random_iter::<TetrisPiece>()
            .take(10_000)
            .collect::<Vec<_>>(),
    );
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
        let num_games = 100;
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
