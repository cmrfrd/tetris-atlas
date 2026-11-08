#![allow(
    clippy::panic,
    reason = "Panicking error handling is accepted in the TensorBoard logging crate."
)]

pub mod tensorboard;
#[allow(
    clippy::all,
    reason = "Generated protobuf bindings retain generator formatting."
)]
pub mod tensorboard_generated {
    include!(concat!(env!("OUT_DIR"), "/tensorboard_generated/mod.rs"));
}

pub use tensorboard::*;
