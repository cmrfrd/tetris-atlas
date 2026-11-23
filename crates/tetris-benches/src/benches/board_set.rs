//! Exact BoardSet operations compared with standard and fast hash sets.

use criterion::{BatchSize, Criterion, Throughput};
use std::collections::HashSet;
use std::hint::black_box;
use std::time::Duration;
use tetris_board_set::BoardSet;

use super::board_set_fixtures as fixtures;

type FastSet = HashSet<[u32; 10], foldhash::fast::RandomState>;

fn bench(c: &mut Criterion) {
    for kind in ["engine", "random"] {
        let boards = fixtures::boards(kind, 16384);
        let oracle: HashSet<_> = boards.iter().copied().collect();
        let mut hits = boards.clone();
        let mut seed = 0xFEED_D00D_DEAD_BEEF;
        for i in (1..hits.len()).rev() {
            hits.swap(i, fixtures::random(&mut seed) as usize % (i + 1));
        }
        let misses: Vec<_> = hits
            .iter()
            .map(|key| {
                let mut query = key.map(|value| !value);
                while oracle.contains(&query) {
                    query[0] = query[0].wrapping_add(1);
                }
                query
            })
            .collect();
        let mut group = c.benchmark_group(format!("board_set_operations/{kind}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
        group.throughput(Throughput::Elements(boards.len() as u64));

        // Monomorphize each implementation: no enum/dynamic dispatch in kernels.
        // All cases share exact inputs, black-box boundaries and timing contracts.
        macro_rules! cases {
            ($name:literal, $make:expr) => {{
                let make = $make;
                let filled = || {
                    let mut set = make(boards.len());
                    for &key in &boards {
                        set.insert(key);
                    }
                    set
                };
                let mut set = filled();
                assert_eq!(set.len(), oracle.len());
                assert_eq!(
                    set.iter()
                        .map(|key| *std::borrow::Borrow::<[u32; 10]>::borrow(&key))
                        .collect::<HashSet<_>>(),
                    oracle
                );
                for query in &misses {
                    assert!(!set.contains(query));
                }
                for query in &hits {
                    assert!(set.contains(query));
                }
                for (policy, capacity) in [("reserved", boards.len()), ("growing", 0)] {
                    group.bench_function(format!("{}/insert_{policy}", $name), |b| {
                        // Exclude construction and destruction; growth triggered
                        // by insertion remains timed. Never reuse a populated set.
                        b.iter_batched_ref(
                            || make(capacity),
                            |set| {
                                for &key in &boards {
                                    black_box(set.insert(black_box(key)));
                                }
                            },
                            BatchSize::SmallInput,
                        );
                    });
                }
                // Compare this common two-search idiom with insert_reserved:
                // insert already checks membership and returns whether it added
                // the board. Keep the same fresh state and allocation policy.
                group.bench_function(concat!($name, "/contains_then_insert_reserved"), |b| {
                    b.iter_batched_ref(
                        || make(boards.len()),
                        |set| {
                            for &key in &boards {
                                let key = black_box(key);
                                if !set.contains(&key) {
                                    black_box(set.insert(key));
                                }
                            }
                        },
                        BatchSize::SmallInput,
                    );
                });
                group.bench_function(concat!($name, "/hits"), |b| {
                    b.iter(|| {
                        for query in &hits {
                            black_box(set.contains(black_box(query)));
                        }
                    })
                });
                group.bench_function(concat!($name, "/misses"), |b| {
                    b.iter(|| {
                        for query in &misses {
                            black_box(set.contains(black_box(query)));
                        }
                    })
                });
                group.bench_function(concat!($name, "/duplicates"), |b| {
                    b.iter(|| {
                        for &key in &hits {
                            black_box(set.insert(black_box(key)));
                        }
                    })
                });
                group.bench_function(concat!($name, "/clear_reuse"), |b| {
                    b.iter_batched_ref(
                        filled,
                        |set| {
                            set.clear();
                            for &key in &boards {
                                black_box(set.insert(black_box(key)));
                            }
                        },
                        BatchSize::SmallInput,
                    );
                });
                // Iteration visits unique keys; use its actual denominator.
                group.throughput(Throughput::Elements(oracle.len() as u64));
                group.bench_function(concat!($name, "/iteration"), |b| {
                    b.iter(|| {
                        for key in set.iter() {
                            black_box(key);
                        }
                    })
                });
                group.throughput(Throughput::Elements(boards.len() as u64));
            }};
        }
        cases!("trie", BoardSet::<10>::with_capacity);
        cases!("std_hash", HashSet::<[u32; 10]>::with_capacity);
        cases!("fast_hash", |capacity| FastSet::with_capacity_and_hasher(
            capacity,
            Default::default()
        ));
        group.finish();
    }
}

fn bench_empty_clear(c: &mut Criterion) {
    for capacity in [0, 16384, 1_000_000] {
        let mut group = c.benchmark_group(format!("board_set_empty_clear/{capacity}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
        group.throughput(Throughput::Elements(1));
        // All sets are empty. BoardSet stores a lazy capacity hint; HashSet
        // allocates its requested routing storage eagerly. Results are ns per
        // clear call, not per key or a clear-and-refill batch.
        let mut trie = BoardSet::<10>::with_capacity(capacity);
        group.bench_function("trie", |b| b.iter(|| black_box(&mut trie).clear()));
        assert!(trie.is_empty());
        let mut standard = HashSet::<[u32; 10]>::with_capacity(capacity);
        group.bench_function("std_hash", |b| b.iter(|| black_box(&mut standard).clear()));
        assert!(standard.is_empty());
        let mut fast = FastSet::with_capacity_and_hasher(capacity, Default::default());
        group.bench_function("fast_hash", |b| b.iter(|| black_box(&mut fast).clear()));
        assert!(fast.is_empty());
        group.finish();
    }
}

tetris_bench! {
    fn bench_board_set_operations(c: &mut Criterion) {
        bench(c);
        bench_empty_clear(c);
    }
}
