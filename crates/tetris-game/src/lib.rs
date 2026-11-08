#![feature(generic_const_exprs)]
#![feature(coverage_attribute)]
#![feature(const_trait_impl)]
#![feature(variant_count)]
#![feature(const_default)]
#![feature(const_convert)]
#![feature(const_index)]
#![doc(test(attr(feature(generic_const_exprs, const_trait_impl))))]

pub mod config;
pub mod game_set;
pub mod tetris;

pub use config::{StandardTetris, TetrisGameConfig};
pub use game_set::{MAX_GAMES, TetrisGameSet, TetrisGameSetError};
pub use tetris::{
    Column, InvalidBagMask, InvalidBagPieces, Major, PlacementResult, Rotation, TetrisBoard,
    TetrisGame, TetrisGameRng, TetrisPiece, TetrisPieceBag, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement, constants,
};
