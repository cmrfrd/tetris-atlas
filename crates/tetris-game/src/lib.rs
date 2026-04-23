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
