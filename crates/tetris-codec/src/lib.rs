#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(long_running_const_eval)]
#![doc(test(attr(
    feature(generic_const_exprs),
    allow(incomplete_features, long_running_const_eval)
)))]
//! Reversible, dense integer IDs for bounded families of canonical engine boards.
//!
//! [`BoardCodec<C, ROWS, MAX_HOLES>`] uses `C::COLS` and accepts native
//! [`tetris_game::TetrisBoard<C>`] values. Counts and immutable lookup tables are
//! evaluated on demand by the compiler; construction and lookup allocate nothing.
//! Counts describe admissible patterns, not proven reachable or winning states.
//!
//! ```
//! use tetris_codec::BoardCodec;
//! use tetris_game::{StandardTetris, TetrisBoard};
//! type Codec = BoardCodec<StandardTetris, 1, 0>;
//! const CODEC: Codec = Codec::new();
//! const COUNT: u64 = Codec::BOARD_COUNT;
//! const BYTES: u128 = Codec::required_bytes(64);
//! assert!(COUNT > 0);
//! assert_eq!(BYTES, u128::from(COUNT) * 8);
//! let board = TetrisBoard::<StandardTetris>::from_limbs([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
//! let id: u16 = CODEC.id(&board)?;
//! assert_eq!(CODEC.board_from_id(id)?, board);
//! # Ok::<(), tetris_codec::Error>(())
//! ```

pub mod board;
mod decision;
mod id;
mod rules;
pub use board::{BoardCodec, CompiledConfig, Dimensions, Error, GameDimensions, Tables};
pub use id::{BoardId, IdConfig};

mod private {
    pub trait Sealed {}
}

#[cfg(test)]
mod test_support;
