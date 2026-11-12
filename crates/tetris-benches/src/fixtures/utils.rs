//! Deterministic bit, matrix and heap inputs.
use super::{NUM_ELEMS, SEED};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

type TransposeInputs = (Vec<[u8; 25]>, Vec<[u8; 40]>, Vec<[u8; 40]>);

/// Builds fixed eight-bit vectors.
pub(crate) fn bit_inputs() -> Vec<[u8; 8]> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let input: Vec<[u8; 8]> = (0..NUM_ELEMS)
        .map(|_| std::array::from_fn(|_| u8::from(rng.random::<bool>())))
        .collect();
    input
}

/// Builds fixed byte inputs.
pub(crate) fn byte_inputs() -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(SEED);

    (0..NUM_ELEMS)
        .map(|_| rng.random::<u8>())
        .collect::<Vec<_>>()
}

/// Builds fixed 64-bit masks.
pub(crate) fn mask_inputs() -> Vec<u64> {
    let mut rng = SmallRng::seed_from_u64(SEED);

    (0..NUM_ELEMS)
        .map(|_| rng.random::<u64>())
        .collect::<Vec<_>>()
}

/// Builds fixed matrices for each transpose shape.
pub(crate) fn transpose_inputs() -> TransposeInputs {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let small: Vec<[u8; 25]> = (0..NUM_ELEMS)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();
    let wide: Vec<[u8; 40]> = (0..NUM_ELEMS)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();
    let tall: Vec<[u8; 40]> = (0..NUM_ELEMS)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();
    (small, wide, tall)
}

/// Builds fixed heap candidates while preserving the requested batch lengths.
pub(crate) fn heap_candidates(lengths: impl IntoIterator<Item = usize>) -> Vec<Vec<i32>> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    lengths
        .into_iter()
        .map(|count| (0..count).map(|_| rng.random::<i32>()).collect())
        .collect()
}
