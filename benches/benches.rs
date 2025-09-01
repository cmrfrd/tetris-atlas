// Wrapper that registers the internal benchmark functions (in src/benches.rs)
// with Criterion. The actual benchmark bodies live inside the crate and are
// available only when the `bench` feature is enabled.
use criterion::{criterion_group, criterion_main};
use tetris_atlas::benches::{tetris_board, tetris_piece};

criterion_group!(benches, tetris_board, tetris_piece);
criterion_main!(benches);
