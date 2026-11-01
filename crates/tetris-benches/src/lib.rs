//! Benchmark suites shared by Cargo tests and Criterion.
#![allow(long_running_const_eval)]
#![feature(generic_const_exprs)]
#![deny(dead_code)]
#[macro_use]
pub mod registry;
mod benches;
mod board_codec;
