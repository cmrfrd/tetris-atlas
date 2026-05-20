#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_index)]
#![allow(incomplete_features)]
#![allow(clippy::expect_used)]
#![allow(clippy::needless_return)]
#![allow(clippy::out_of_bounds_indexing)]
#![doc(test(attr(feature(generic_const_exprs), allow(incomplete_features))))]

pub mod config;
pub mod game_set;
pub mod tetris;

pub use config::{StandardTetris, TetrisGameConfig};
pub use game_set::{MAX_GAMES, TetrisGameSet, TetrisGameSetError};
pub use tetris::{
    Column, IsLost, Major, ORDERED_7, PlacementResult, Rotation, TetrisBoard, TetrisGame,
    TetrisGameRng, TetrisPiece, TetrisPieceBag, TetrisPieceBagState, TetrisPieceOrientation,
    TetrisPiecePlacement, constants, fisher_yates_7bag_stream_from_seed, swap_3bit_chunks,
};
