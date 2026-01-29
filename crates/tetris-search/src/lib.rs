//! Search algorithms for Tetris game state exploration.
//!
//! This crate provides:
//! - `beam_search`: Beam search algorithm with Tetris-specific heuristics
//! - `explorer`: BFS-based parallel state space exploration
//!
//! # Example
//!
//! ```ignore
//! use tetris_search::{BeamSearch, BeamTetrisState};
//! use tetris_game::TetrisGame;
//!
//! let game = TetrisGame::new();
//! let mut search = BeamSearch::<BeamTetrisState, 32, 8, 40>::new();
//! let result = search.search_top_with_state(BeamTetrisState::new(game), 8);
//! ```

#![feature(generic_const_exprs)]

pub mod beam_search;
pub mod explorer;

pub use beam_search::*;
pub use explorer::*;

use rayon::ThreadPoolBuilder;

/// Configure the global rayon threadpool based on environment or CPU count.
///
/// Reads `RAYON_NUM_THREADS` environment variable, falling back to `num_cpus::get()`.
pub fn set_global_threadpool() {
    const ENV_VAR_NAME: &str = "RAYON_NUM_THREADS";
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var(ENV_VAR_NAME)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(num_cpus::get()),
        )
        .build_global()
        .unwrap();
}
