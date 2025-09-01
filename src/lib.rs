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

// With feature `bench`, expose the benchmark module from its own file.
#[cfg(feature = "bench")]
pub mod benches;

pub const ASSERT_LEVEL: u32 = 1;
