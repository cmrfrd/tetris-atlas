//! Tetris Playground - runnable demos, atlas builders, training jobs, and research tools.
//!
//! This crate contains all runnable binaries for:
//!
//! ## Demos
//! - `tetris_demo_single` - Single beam search gameplay
//! - `tetris_demo_multi` - Multi-beam search with voting
//! - `tetris_demo_adaptive_multi` - Adaptive beam search by board occupancy
//!
//! - `tetris_atlas_inmemory` - In-memory atlas builder
//! - `tetris_atlas_rocksdb` - RocksDB-backed atlas builder
//!
//! ## Tools
//! - `tetris_cli` - CLI/TUI entry point
//!
//! ## ML Training
//! - `tetris_train_beam_supervised` - Beam-supervised policy training
//! - `tetris_train_genetic` - Genetic heuristic tuning
//!
//! ## Research
//! - `tetris_invariant_synthesis` - Controlled-invariant synthesis prototype
//! - `tetris_safe_set` - Safe-set certifier prototype
//! - `tetris_success_set_solver` - Closed winning-set solver prototype

// Re-export common types for bin convenience
pub use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement, repeat_idx_unroll,
};
pub use tetris_ml::{
    BeamSearch, BeamSearchState, BeamTetrisState, MultiBeamSearch, ScoredState, device,
    set_global_threadpool,
};
