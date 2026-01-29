//! Tetris Playground - Demos, experiments, and reference implementations.
//!
//! This crate contains all runnable binaries for:
//!
//! ## Demos
//! - `tetris_demo_single` - Single beam search gameplay
//! - `tetris_demo_multi` - Multi-beam search with voting
//! - `tetris_demo_multi_tiered` - Tiered beam search by board height
//!
//! ## Atlas Builders
//! - `tetris_atlas` - FeoxDB-based atlas builder
//! - `tetris_atlas_rocksdb` - RocksDB-based atlas builder
//! - `tetris_atlas_lmdb` - LMDB/heed-based atlas builder
//! - `tetris_atlas_inmemory` - In-memory atlas builder
//!
//! ## ML Training
//! - `tetris_train_*` - Various ML training experiments
//! - `optimal_board_sharding` - Neural network for board sharding
//!
//! ## Experiments
//! - `tetris_experiment_voting_*` - Voting distribution experiments
//! - `tetris_search_quality` - Search quality analysis
//! - `tetris_bench_performance` - Performance benchmarking

// Re-export common types for bin convenience
pub use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement, repeat_idx_unroll,
};
pub use tetris_ml::{
    BeamSearch, BeamSearchState, BeamTetrisState, MultiBeamSearch, ScoredState,
    device, set_global_threadpool,
};
