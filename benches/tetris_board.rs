use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::seq::{IndexedRandom, IteratorRandom};
use tetris_atlas::tetris_board::{
    BitSetter, Clearer, Merge, Shiftable, TetrisBoard, TetrisPieceBag,
};

fn criterion_benchmark(c: &mut Criterion) {
    let mut boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = TetrisBoard::default();
                b.flip_random_bits(16);
                b
            })
            .collect::<Vec<_>>(),
    );
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

    c.bench_function("shift_down", |b| {
        b.iter(|| {
            boards
                .iter_mut()
                .map(|p| p.shift_down())
                .collect::<Vec<_>>()
        })
    });

    c.bench_function("shift_up", |b| {
        b.iter(|| boards.iter_mut().map(|p| p.shift_up()).collect::<Vec<_>>())
    });

    c.bench_function("loss", |b| {
        b.iter(|| boards.iter().map(|p| p.loss()).collect::<Vec<_>>())
    });

    c.bench_function("merge", |b| {
        let base = {
            let mut b = TetrisBoard::default();
            b.flip_random_bits(16);
            b
        };
        b.iter(|| {
            boards
                .iter_mut()
                .map(|p| p.merge(&base))
                .collect::<Vec<_>>()
        })
    });

    c.bench_function("bag_iter", |b| {
        b.iter(|| {
            let mut bag = TetrisPieceBag::new();
            for _ in 0..10_000 {
                bag = bag.next_bags().next().unwrap().0;
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
