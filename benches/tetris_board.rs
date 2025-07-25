use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, seq::IteratorRandom};
use tetris_atlas::tetris_board::{
    BitSetter, Clearer, Mergeable, PiecePlacement, Shiftable, TetrisBoard, TetrisBoardRaw,
    TetrisPieceBag,
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
        rng.random_iter::<PiecePlacement>()
            .take(10_000)
            .collect::<Vec<_>>(),
    );
    c.bench_function("drop", |b| {
        b.iter(|| {
            empty_boards
                .iter_mut()
                .zip(placements.iter())
                .map(|(b, &p)| b.play_piece(p))
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
