pub mod tensorboard;
#[allow(clippy::all)]
pub mod tensorboard_generated {
    include!(concat!(env!("OUT_DIR"), "/tensorboard_generated/mod.rs"));
}

pub use tensorboard::*;
