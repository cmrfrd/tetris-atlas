#![feature(unbounded_shifts)]

pub mod tetris_board;

pub use tetris_board::*;

pub const ASSERT_LEVEL: u32 = 1;

/// This custom assert is so we can can disable it at compile
/// time, to remove runtime impact.
#[macro_export]
macro_rules! assert_level {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            if $crate::ASSERT_LEVEL >= 1 {
                assert!($($arg)*);
            }
        }
    };
}
