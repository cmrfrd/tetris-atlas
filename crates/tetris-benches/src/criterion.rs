//! Criterion configuration and profiling entry point.

use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn run(c: &mut Criterion) {
    for benchmark in tetris_benches::registry::registered() {
        (benchmark.run)(c);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(1_000, Output::Flamegraph(None)));
    targets = run
}

criterion_main!(benches);
