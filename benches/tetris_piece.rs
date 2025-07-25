use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use tetris_atlas::tetris_board::{PiecePlacement, TetrisPiece};

fn criterion_benchmark(c: &mut Criterion) {
    let rng = rand::rng();
    let pieces = black_box(
        rng.random_iter::<TetrisPiece>()
            .take(10_000)
            .collect::<Vec<_>>(),
    );
    c.bench_function("num_rotations", |b| {
        b.iter(|| pieces.iter().map(|p| p.num_rotations()).collect::<Vec<_>>())
    });

    let rng = rand::rng();
    let placements = black_box(
        rng.random_iter::<PiecePlacement>()
            .take(10_000)
            .map(|p| p.piece_rotation())
            .collect::<Vec<_>>(),
    );
    c.bench_function("width", |b| {
        b.iter(|| {
            placements
                .iter()
                .map(|(p, r)| p.width(*r))
                .collect::<Vec<_>>()
        })
    });
    c.bench_function("height", |b| {
        b.iter(|| {
            placements
                .iter()
                .map(|(p, r)| p.height(*r))
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
