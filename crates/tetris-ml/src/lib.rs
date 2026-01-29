#![feature(generic_const_exprs)]
#![feature(const_convert)]

pub mod beam_search;
pub mod checkpointer;
pub mod data;
pub mod grad_accum;
pub mod modules;
pub mod optim;
pub mod tetris_tui;

pub use beam_search::*;
pub use checkpointer::*;
pub use data::*;
pub use grad_accum::*;
pub use modules::*;
pub use optim::*;

use candle_core::Device;
use rayon::ThreadPoolBuilder;

pub fn device() -> Device {
    if cfg!(feature = "candle-cuda") {
        Device::new_cuda(0).unwrap()
    } else if cfg!(feature = "candle-metal") {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    }
}

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
