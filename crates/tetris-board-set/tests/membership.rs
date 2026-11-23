#![feature(generic_const_exprs)]
#![allow(
    incomplete_features,
    reason = "canonical engine fixtures require generic_const_exprs"
)]

use std::collections::HashSet;
use tetris_board_set::BoardSet;
use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiecePlacement};

#[allow(
    clippy::cast_possible_truncation,
    reason = "The fixed-seed generator intentionally returns the low 32 bits of its xorshift state."
)]
fn random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as u32
}

fn differential<const N: usize>() {
    let mut seed = 0xDEAD_BEEF_CAFE;
    for capacity in [0, 1, 4096] {
        let mut set = BoardSet::<N>::with_capacity(capacity);
        let mut oracle = HashSet::new();
        let mut order = Vec::new();
        for step in 0..8000 {
            if step == 4000 {
                set.clear();
                oracle.clear();
                order.clear();
            }
            let key = std::array::from_fn(|_| {
                random(&mut seed) & if step % 2 == 0 { 15 } else { u32::MAX }
            });
            assert_eq!(set.contains(&key), oracle.contains(&key));
            let fresh = oracle.insert(key);
            assert_eq!(set.insert(key), fresh);
            assert!(!set.insert(key));
            if fresh {
                order.push(key);
            }
            let query = std::array::from_fn(|_| random(&mut seed));
            assert_eq!(set.contains(&query), oracle.contains(&query));
            assert_eq!(set.len(), oracle.len());
            assert_eq!(set.is_empty(), oracle.is_empty());
            if step % 257 == 0 {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert!(set.iter().eq(sorted));
                for old in &order {
                    assert!(set.contains(old));
                }
            }
        }
        order.sort_unstable();
        assert!(set.iter().eq(order));
        assert_eq!(set.iter().collect::<HashSet<_>>(), oracle);
    }
}

#[test]
fn fixed_seed_differential_growth_and_reuse() {
    differential::<0>();
    differential::<1>();
    differential::<4>();
    differential::<10>();
}

#[test]
fn every_backing_bit_and_terminal_values_are_distinct() {
    let mut set = BoardSet::<10>::new();
    let mut order = Vec::new();
    for col in 0..10 {
        for bit in 0..u32::BITS {
            let mut key = [0; 10];
            key[col] = 1 << bit;
            assert!(set.insert(key));
            order.push(key);
        }
    }
    for key in [[0; 10], [u32::MAX; 10]] {
        assert!(set.insert(key));
        order.push(key);
    }
    for key in &order {
        assert!(set.contains(key));
        assert!(!set.insert(*key));
    }
    order.sort_unstable();
    assert!(set.iter().eq(order));
}

#[test]
fn column_correlations_and_small_universe_are_exact() {
    let mut set = BoardSet::<2>::new();
    for a in 0..16 {
        for b in 0..16 {
            if (a + b) % 3 == 0 {
                assert!(set.insert([a, b]));
            }
        }
    }
    for a in 0..16 {
        for b in 0..16 {
            assert_eq!(set.contains(&[a, b]), (a + b) % 3 == 0);
        }
    }
}

#[test]
fn canonical_placements_include_clears_and_normalized_losses() {
    let mut clearing = [3; 10];
    clearing[8] = 0;
    clearing[9] = 0;
    let mut tall = [0; 10];
    tall[0] = (1 << StandardTetris::ROWS) - 1;
    let mut set = BoardSet::new();
    let mut oracle = HashSet::new();
    let mut cleared = 0;
    let mut lost = 0;
    for parent in [[0; 10], clearing, tall] {
        for index in 0..TetrisPiecePlacement::<{ StandardTetris::COLS }>::NUM_PLACEMENTS {
            let mut board =
                TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::from_limbs(
                    parent,
                );
            let result = board.apply_piece_placement(TetrisPiecePlacement::from_index(
                u8::try_from(index).expect("standard placement indices fit u8"),
            ));
            cleared += result.lines_cleared;
            lost += usize::from(board.is_lost());
            let key = *board.as_limbs();
            assert_eq!(set.insert(key), oracle.insert(key));
            assert!(set.contains(&key));
        }
    }
    assert!(cleared > 0);
    assert!(lost > 0);
    assert_eq!(set.iter().collect::<HashSet<_>>(), oracle);
}

/// Longer differential coverage stays opt-in so normal workspace checks stay fast.
#[test]
#[ignore = "Seeded stress sweep; run explicitly with --release --ignored"]
fn seeded_mixed_distributions_and_repeated_clear_sweep() {
    for seed in 0..32_u64 {
        let mut state = 0xA5D3_1F70_CE29_486B ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let capacity = [0, 1, 256, 20_000]
            [usize::try_from(seed % 4).expect("capacity selector is below four")];
        let mut set = BoardSet::<10>::with_capacity(capacity);
        let mut oracle = HashSet::new();
        let mut history = Vec::new();
        for step in 0..20_000 {
            if step == 8192 || step == 16384 {
                set.clear();
                set.clear();
                oracle.clear();
                history.clear();
                assert!(!set.contains(&[u32::MAX; 10]));
            }
            let selector = random(&mut state);
            let key = if selector.is_multiple_of(7) && !history.is_empty() {
                history[random(&mut state) as usize % history.len()]
            } else {
                let mut key = std::array::from_fn(|_| random(&mut state));
                match selector % 4 {
                    0 => key = key.map(|value| value & 31),
                    1 => key[..8].fill(0x55),
                    2 => {}
                    _ => key = [u32::MAX; 10],
                }
                key
            };
            let fresh = oracle.insert(key);
            assert_eq!(set.insert(key), fresh, "seed={seed}, step={step}");
            if fresh {
                history.push(key);
            }
            assert!(!set.insert(key), "duplicate: seed={seed}, step={step}");
            assert!(set.contains(&key), "hit: seed={seed}, step={step}");
            let mut query = key;
            query[step % 10] ^= 1 << (step % 32);
            assert_eq!(
                set.contains(&query),
                oracle.contains(&query),
                "query: seed={seed}, step={step}"
            );
            assert_eq!(set.len(), oracle.len());
            if step % 2048 == 0 {
                let mut expected: Vec<_> = oracle.iter().copied().collect();
                expected.sort_unstable();
                assert!(
                    set.iter().eq(expected),
                    "iteration: seed={seed}, step={step}"
                );
            }
        }
        assert_eq!(set.iter().collect::<HashSet<_>>(), oracle, "seed={seed}");
    }
}
