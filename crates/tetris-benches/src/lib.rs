//! Benchmark suites shared by Cargo tests and Criterion.

#![feature(generic_const_exprs)]
#![deny(dead_code)]

#[macro_use]
pub mod registry;

mod benches;
