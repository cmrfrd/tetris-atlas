#![feature(generic_const_exprs)]

pub mod ops;
pub mod tensors;
pub mod wrapped_tensor;

pub use ops::*;
pub use tensors::*;
pub use wrapped_tensor::*;

use candle_core::DType;

/// Type alias for floating-point dtype
pub type FDtype = DType;

/// Get the configured floating-point dtype based on compile-time features
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
    FDtype::F32
}
