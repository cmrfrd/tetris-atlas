// Wrapper that registers the internal benchmark functions (in src/benches.rs)
// with Criterion. The actual benchmark bodies live inside the crate and are
// available only when the `bench` feature is enabled.

use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};
use tetris_atlas::benches::tetris_game;

criterion_group! {
    name = benches_prof;
    config = Criterion::default().with_profiler(PProfProfiler::new(1_000, Output::Flamegraph(None)));
    targets = tetris_game
}

criterion_main!(benches_prof);
