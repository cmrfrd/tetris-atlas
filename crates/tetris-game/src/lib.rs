#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_index)]
#![allow(incomplete_features)]
#![allow(clippy::expect_used)]
#![allow(clippy::needless_return)]
#![allow(clippy::out_of_bounds_indexing)]

pub mod tetris;

pub use tetris::{
    Column, IsLost, MAX_GAMES, ORDERED_7, PlacementResult, Rotation, TetrisBoard,
    TetrisBoardBinarySlice, TetrisGame, TetrisGameRng, TetrisGameSet, TetrisGameSetError,
    TetrisPiece, TetrisPieceBag, TetrisPieceBagState, TetrisPieceOrientation, TetrisPiecePlacement,
    constants, fisher_yates_7bag_stream_from_seed, swap_3bit_chunks,
};
