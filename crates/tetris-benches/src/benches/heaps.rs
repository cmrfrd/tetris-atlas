//! Fixed-capacity heap benchmarks.

use crate::fixtures::utils::heap_candidates;
use criterion::{Criterion, Throughput, black_box};

tetris_bench! {
    /// Benchmark push_if_better - the key beam search operation
    /// Tests inserting 1000 candidates into a heap of size 100
    fn bench_heap_push_if_better(c: &mut Criterion) {
        use tetris_utils::FixedBinMinHeap;

        const HEAP_SIZE: usize = 100;
        const NUM_CANDIDATES: usize = 1000;
        let candidates = heap_candidates([NUM_CANDIDATES]).remove(0);

        let mut group = c.benchmark_group("fixed_bin_min_heap");
        group.throughput(Throughput::Elements(NUM_CANDIDATES as u64));

        group.bench_function("push_if_better", |b| {
            let mut heap = FixedBinMinHeap::<i32, HEAP_SIZE>::new();
            b.iter(|| {
                heap.clear();
                for &v in candidates.iter() {
                    heap.push_if_better_min_heap(black_box(v));
                }

                black_box(heap.peek_min())
            })
        });

        group.finish();
    }
}

tetris_bench! {
    /// Benchmark mixed operations (realistic beam search workload)
    /// Simulates 100 iterations of adding 50 candidates each
    fn bench_heap_mixed_workload(c: &mut Criterion) {
        use tetris_utils::FixedBinMinHeap;

        const HEAP_SIZE: usize = 64;
        const NUM_ITERATIONS: usize = 100;
        const CANDIDATES_PER_ITER: usize = 50;

        let all_candidates = heap_candidates([CANDIDATES_PER_ITER; NUM_ITERATIONS]);

        let mut group = c.benchmark_group("fixed_bin_min_heap");
        group.throughput(Throughput::Elements(
            (NUM_ITERATIONS * CANDIDATES_PER_ITER) as u64,
        ));

        group.bench_function("mixed_workload", |b| {
            let mut heap = FixedBinMinHeap::<i32, HEAP_SIZE>::new();
            b.iter(|| {
                heap.clear();

                for candidates in &all_candidates {
                    for &v in candidates {
                        heap.push_if_better_min_heap(black_box(v));
                    }

                    black_box(heap.peek_min());
                }

                heap.peek_min()
            })
        });

        group.finish();
    }
}

tetris_bench! {
    /// Benchmark scaling with different heap sizes
    fn bench_heap_size_scaling(c: &mut Criterion) {
        use tetris_utils::FixedBinMinHeap;

        for (size, candidates) in [16, 64, 128]
            .into_iter()
            .zip(heap_candidates([160, 640, 1280]))
        {
            let mut group = c.benchmark_group("fixed_bin_min_heap");
            group.throughput(Throughput::Elements(candidates.len() as u64));

            match size {
                16 => {
                    group.bench_function("size_16", |b| {
                        let mut heap = FixedBinMinHeap::<i32, 16>::new();
                        b.iter(|| {
                            heap.clear();
                            for &v in &candidates {
                                heap.push_if_better_min_heap(black_box(v));
                            }
                            heap.peek_min()
                        })
                    });
                }
                64 => {
                    group.bench_function("size_64", |b| {
                        let mut heap = FixedBinMinHeap::<i32, 64>::new();
                        b.iter(|| {
                            heap.clear();
                            for &v in &candidates {
                                heap.push_if_better_min_heap(black_box(v));
                            }
                            heap.peek_min()
                        })
                    });
                }
                128 => {
                    group.bench_function("size_128", |b| {
                        let mut heap = FixedBinMinHeap::<i32, 128>::new();
                        b.iter(|| {
                            heap.clear();
                            for &v in &candidates {
                                heap.push_if_better_min_heap(black_box(v));
                            }
                            heap.peek_min()
                        })
                    });
                }
                _ => {
                    assert!(
                        matches!(size, 8 | 16 | 32 | 64 | 128),
                        "unsupported heap size: {size}"
                    );
                }
            }

            group.finish();
        }
    }
}
