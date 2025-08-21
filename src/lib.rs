#![feature(array_windows)]
#![feature(unbounded_shifts)]
#![feature(hash_set_entry)]
#![feature(new_range_api)]
#![feature(const_trait_impl)]
#![feature(iter_array_chunks)]

// pub mod atlas;
pub mod dataset;
pub mod hf;
pub mod infer;
pub mod model;
pub mod tetris;
pub mod tetris_explorer;
pub mod train;
pub mod utils;

pub use hf::*;

pub const ASSERT_LEVEL: u32 = 1;
pub static ARTIFACT_DIR: &str = "/tmp/tetris-game-transformer";

/// MurmurHash3 64-bit hash function
#[inline(always)]
pub fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}
