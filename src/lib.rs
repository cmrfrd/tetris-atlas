#![feature(array_windows)]
#![feature(hash_set_entry)]
#![feature(new_range_api)]
#![feature(const_trait_impl)]
#![feature(iter_array_chunks)]
#![feature(const_index)]

use candle_core::Device;
use rayon::ThreadPoolBuilder;

pub mod benches;
pub mod checkpointer;
pub mod data;
pub mod grad_accum;
pub mod modules;
pub mod ops;
pub mod optim;
pub mod tensors;
pub mod tetris;
pub mod tetris_evolution_player_model;
pub mod tetris_exceed_the_mean;
pub mod tetris_explorer;
pub mod tetris_simple_player_model;
pub mod tetris_transition_model;
pub mod tetris_transition_transformer_model;
pub mod tetris_world_model;
pub mod utils;
pub mod wrapped_tensor;

pub const ASSERT_LEVEL: u32 = 1;

pub fn set_global_threadpool() {
    const ENV_VAR_NAME: &str = "RAYON_NUM_THREADS";
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var(ENV_VAR_NAME)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(num_cpus::get()),
        )
        .build_global()
        .unwrap();
}

pub fn device() -> Device {
    if cfg!(feature = "candle-cuda") {
        Device::new_cuda(0).unwrap()
    } else {
        Device::Cpu
    }
}

pub fn dtype() -> candle_core::DType {
    candle_core::DType::F32
}
