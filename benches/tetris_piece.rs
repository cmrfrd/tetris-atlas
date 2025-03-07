use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tetris_atlas::tetris_board::{Rotation, TetrisPiece};

fn criterion_benchmark(c: &mut Criterion) {
    let pieces = black_box(
        (0..10_000)
            .map(|_| TetrisPiece::new(rand::random::<u8>() % 7))
            .collect::<Vec<_>>(),
    );
    c.bench_function("num_rotations", |b| {
        b.iter(|| pieces.iter().map(|p| p.num_rotations()).collect::<Vec<_>>())
    });

    let rotations = black_box(
        (0..10_000)
            .map(|_| Rotation(rand::random::<u8>() % 4))
            .collect::<Vec<_>>(),
    );
    c.bench_function("width", |b| {
        b.iter(|| {
            pieces
                .iter()
                .zip(rotations.iter())
                .map(|(p, r)| p.width(*r))
                .collect::<Vec<_>>()
        })
    });
    c.bench_function("height", |b| {
        b.iter(|| {
            pieces
                .iter()
                .zip(rotations.iter())
                .map(|(p, r)| p.height(*r))
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
