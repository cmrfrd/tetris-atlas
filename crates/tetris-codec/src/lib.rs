#![feature(generic_const_exprs)]
#![allow(
    long_running_const_eval,
    reason = "Bounded codec graph construction may exceed the const evaluator progress threshold."
)]
#![doc(test(attr(
    feature(generic_const_exprs),
    allow(incomplete_features, long_running_const_eval)
)))]
//! Reversible, dense integer IDs for bounded families of canonical engine boards.
//!
//! [`BoardCodec<C, ROWS, MAX_HOLES>`] uses `C::COLS` and accepts native
//! [`tetris_game::TetrisBoard`] values. Counts are evaluated by the compiler. Ten-column, one-hole codecs at heights
//! 4–6 rank algebraically; other configurations use compiled decision graphs.
//! Construction and lookup allocate nothing.
//! Height caps must be at least two. Counts describe admissible patterns, not
//! proven reachable or winning states.
//!
//! ```
//! use tetris_codec::BoardCodec;
//! use tetris_game::{StandardTetris, TetrisBoard, TetrisGameConfig};
//! type Codec = BoardCodec<StandardTetris, 2, 0>;
//! const CODEC: Codec = Codec::new();
//! const COUNT: u64 = Codec::BOARD_COUNT;
//! const BYTES: u128 = Codec::required_bytes(64);
//! assert!(COUNT > 0);
//! assert_eq!(BYTES, u128::from(COUNT) * 8);
//! let board = TetrisBoard::<{ StandardTetris::COLS }, { StandardTetris::ROWS }>::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
//! let id: u16 = CODEC.id(&board)?;
//! assert_eq!(CODEC.board_from_id(id)?, board);
//! # Ok::<(), tetris_codec::Error>(())
//! ```

mod arithmetic;
pub mod board;
mod decision;
mod id;
pub use board::{BoardCodec, CompiledConfig, Dimensions, Error, GameDimensions, Tables};
pub use id::{BoardId, IdConfig};

mod private {
    /// Restricts compiled carriers to the implementations supplied by this crate.
    pub trait Sealed {}
}
