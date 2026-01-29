#![feature(generic_const_exprs)]
#![feature(const_convert)]

pub mod checkpointer;
pub mod data;
pub mod grad_accum;
pub mod modules;
pub mod optim;
pub mod tensor;
pub mod tetris_tui;

pub use checkpointer::*;
pub use data::*;
pub use grad_accum::*;
pub use modules::*;
pub use optim::*;
pub use tensor::*;

// Re-export beam search from tetris-search for backward compatibility
pub use tetris_search::beam_search;
pub use tetris_search::{
    BeamSearch, BeamSearchState, BeamTetrisState, MultiBeamSearch, ScoredState,
    set_global_threadpool,
};

use candle_core::Device;

pub fn device() -> Device {
    if cfg!(feature = "candle-cuda") {
        Device::new_cuda(0).unwrap()
    } else if cfg!(feature = "candle-metal") {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    }
}
