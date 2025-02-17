use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tetris_atlas::{BitSetter, Clearer, Shiftable, TetrisBoard};

fn criterion_benchmark(c: &mut Criterion) {
    let mut boards = black_box(
        (0..10_000)
            .map(|_| {
                let mut b = TetrisBoard::default();
                b.set_random_bits(16);
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
