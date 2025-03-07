use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::seq::{IndexedRandom, IteratorRandom};
use tetris_atlas::tetris_board::{
    BitSetter, BoardRaw, COLS, Clearer, Collides, Countable, Losable, Mergeable, Rotation,
    Shiftable, TetrisBoard, TetrisPiece, TetrisPieceBag,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = BoardRaw::default();
                b.flip_random_bits(1024, 42);
                b
            })
            .collect::<Vec<_>>(),
    );

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
        let mut b = BoardRaw::default();
        b.flip_random_bits(16, 42);
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
        let mut b = BoardRaw::default();
        b.flip_random_bits(16, 42);
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
        let mut b = BoardRaw::default();
        b.flip_random_bits(16, 42);
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
        let mut b = BoardRaw::default();
        b.flip_random_bits(16, 42);
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

    c.bench_function("count", |b| {
        b.iter(|| boards.iter().map(|p| p.count()).collect::<Vec<_>>())
    });

    let mut empty_boards = black_box(
        (0..10_000)
            .map(|_| TetrisBoard::default())
            .collect::<Vec<_>>(),
    );
    let pieces_rot_col = black_box(
        (0..10_000)
            .map(|_| {
                let piece = TetrisPiece::new(rand::random::<u8>() % 7);
                let rotation = Rotation(rand::random::<u8>() % 4);
                let col = rand::random::<u8>() % (COLS as u8 - piece.width(rotation));
                (piece, rotation, col)
            })
            .collect::<Vec<_>>(),
    );
    c.bench_function("drop", |b| {
        b.iter(|| {
            empty_boards
                .iter_mut()
                .zip(pieces_rot_col.iter())
                .map(|(b, (piece, rotation, col))| b.play_piece(*piece, *rotation, *col))
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
