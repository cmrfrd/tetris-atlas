//! Fixed-seed BoardSet membership across supported generic widths.

use criterion::{Criterion, Throughput};
use std::{collections::HashSet, hint::black_box, time::Duration};
use tetris_board_set::BoardSet;

#[allow(
    clippy::cast_possible_truncation,
    reason = "The fixed-seed generator intentionally returns the low 32 bits of its xorshift state."
)]
fn random(seed: &mut u64) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed as u32
}

fn width<const N: usize>(c: &mut Criterion) {
    const COUNT: usize = 4096;
    let mut seed = 0xDEAD_BEEF_CAFE_u64;
    let mut keys: Vec<[u32; N]> = (0..COUNT)
        .map(|i| {
            std::array::from_fn(|column| {
                if column == 0 {
                    u32::try_from(i).expect("fixture index fits a column")
                } else {
                    let value = random(&mut seed);
                    match (column + i) % 5 {
                        0 => 0,
                        1 => value & 255,
                        2 => value & 65535,
                        3 => value & 0xFF_FFFF,
                        _ => value,
                    }
                }
            })
        })
        .collect();
    let oracle: HashSet<_> = keys.iter().copied().collect();
    let mut set = BoardSet::<N>::with_capacity(COUNT);
    for &key in &keys {
        assert!(set.insert(key));
    }
    assert_eq!(set.len(), oracle.len());
    assert_eq!(set.iter().collect::<HashSet<_>>(), oracle);
    for i in (1..keys.len()).rev() {
        let other = random(&mut seed) as usize % (i + 1);
        keys.swap(i, other);
    }
    for key in &keys {
        assert!(set.contains(key));
        let mut absent = *key;
        absent[0] += u32::try_from(COUNT).expect("4096 fits a column");
        assert!(!set.contains(&absent));
    }
    let mut group = c.benchmark_group("board_set_widths");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_function(format!("{N}/hits"), |b| {
        b.iter(|| {
            for key in &keys {
                black_box(set.contains(black_box(key)));
            }
        });
    });
    group.finish();
}

fn bench(c: &mut Criterion) {
    width::<4>(c);
    width::<10>(c);
    width::<11>(c);
    width::<21>(c);
    width::<22>(c);
    width::<33>(c);
    width::<64>(c);
    width::<200>(c);
}

tetris_bench! {
    fn bench_board_set_widths(c: &mut Criterion) {
        bench(c);
    }
}
