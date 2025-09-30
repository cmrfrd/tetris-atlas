#![feature(array_windows)]
#![feature(hash_set_entry)]
#![feature(new_range_api)]
#![feature(const_trait_impl)]
#![feature(iter_array_chunks)]

use rayon::ThreadPoolBuilder;

pub mod checkpointer;
pub mod data;
pub mod grad_accum;
pub mod model;
pub mod ops;
pub mod optim;
pub mod tensors;
pub mod tetris;
pub mod tetris_explorer;
pub mod train;
pub mod utils;
pub mod wrapped_tensor;

// With feature `bench`, expose the benchmark module from its own file.
#[cfg(feature = "bench")]
pub mod benches;

pub const ASSERT_LEVEL: u32 = 1;

pub fn set_global_threadpool() {
    const ENV_VAR_NAME: &str = "RAYON_NUM_THREADS";
    const DEFAULT_NUM_THREADS: usize = 4;
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var(ENV_VAR_NAME)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_NUM_THREADS),
        ) // fallback to 4
        .build_global()
        .unwrap();
}
