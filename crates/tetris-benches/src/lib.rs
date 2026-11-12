//! Benchmark suites shared by Cargo tests and Criterion.
#![allow(
    long_running_const_eval,
    reason = "Bounded codec graph construction may exceed the const evaluator progress threshold."
)]
#![feature(generic_const_exprs)]
#![deny(dead_code)]
#[macro_use]
pub mod registry;
mod benches;
mod fixtures;
