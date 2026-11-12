//! Deterministic input data and generators shared by benchmark collections.

pub(crate) mod codec;
pub(crate) mod game;
pub(crate) mod utils;

/// Number of inputs in each utility kernel batch.
pub(crate) const NUM_ELEMS: usize = 10_000;

/// Shared seed for deterministic utility and board inputs.
pub(crate) const SEED: u64 = 0xDEAD_BEEF_CAFE;
