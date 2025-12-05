#![feature(array_windows)]
#![feature(hash_set_entry)]
#![feature(new_range_api)]
#![feature(const_trait_impl)]
#![feature(iter_array_chunks)]
#![feature(const_index)]
#![feature(once_cell_try)]

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
pub mod tetris_simple_imitation;
pub mod tetris_q_learning;
pub mod tetris_simple_player_model;
pub mod tetris_transition_model;
pub mod tetris_transition_transformer_model;
pub mod tetris_world_model;
pub mod tetris_tui;
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
    } else if cfg!(feature = "candle-metal") {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    }
}

pub type FDtype = candle_core::DType;

/// Returns the configured floating-point dtype.
/// This is determined at compile-time via feature flags:
/// - dtype-f16: Half precision (16-bit)
/// - dtype-bf16: Brain Float 16
/// - dtype-f32: Standard 32-bit float (default)
/// - dtype-f64: Double precision (64-bit)
#[inline]
pub const fn fdtype() -> FDtype {
    // Compile-time check: ensure at most one dtype feature is enabled
    const DTYPE_COUNT: usize = cfg!(feature = "dtype-f16") as usize
        + cfg!(feature = "dtype-bf16") as usize
        + cfg!(feature = "dtype-f32") as usize
        + cfg!(feature = "dtype-f64") as usize;

    const _: () = assert!(
        DTYPE_COUNT <= 1,
        "Multiple dtype features enabled. Enable only one: dtype-f16, dtype-bf16, dtype-f32, or dtype-f64"
    );

    #[cfg(feature = "dtype-f16")]
    return FDtype::F16;

    #[cfg(feature = "dtype-bf16")]
    return FDtype::BF16;

    #[cfg(feature = "dtype-f64")]
    return FDtype::F64;

    // Default to F32 if no dtype feature is specified
    #[cfg(not(any(feature = "dtype-f16", feature = "dtype-bf16", feature = "dtype-f64")))]
    return FDtype::F32;
}
