#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_index)]
#![allow(incomplete_features)]
#![allow(clippy::expect_used)]
#![allow(clippy::needless_return)]
#![allow(clippy::out_of_bounds_indexing)]

pub mod tetris;

pub use tetris::*;
pub use tetris_utils::*;

// Re-export macros from tetris-utils so existing `use tetris_game::repeat_idx_unroll` works
pub use tetris_utils::rep1_at;
pub use tetris_utils::rep2_at;
pub use tetris_utils::rep4_at;
pub use tetris_utils::rep8_at;
pub use tetris_utils::rep16_at;
pub use tetris_utils::rep32_at;
pub use tetris_utils::repeat_exact_idx;
pub use tetris_utils::repeat_idx_unroll;
