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

/// Default stack size for rayon threads (64 MB).
pub const DEFAULT_RAYON_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Configure the global rayon threadpool based on environment or CPU count.
///
/// Reads environment variables:
/// - `RAYON_NUM_THREADS`: Number of threads (defaults to CPU count)
/// - `RAYON_STACK_SIZE_MB`: Stack size in megabytes (defaults to 64 MB)
pub fn set_global_threadpool() {
    set_global_threadpool_with_stack_size(None);
}

/// Configure the global rayon threadpool with a custom stack size.
///
/// # Arguments
/// - `stack_size`: Optional stack size in bytes. If `None`, reads from
///   `RAYON_STACK_SIZE_MB` env var (in MB) or uses `DEFAULT_RAYON_STACK_SIZE` (64 MB).
pub fn set_global_threadpool_with_stack_size(stack_size: Option<usize>) {
    const NUM_THREADS_ENV: &str = "RAYON_NUM_THREADS";
    const STACK_SIZE_MB_ENV: &str = "RAYON_STACK_SIZE_MB";

    let num_threads = std::env::var(NUM_THREADS_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_cpus::get());

    let stack_size = stack_size.unwrap_or_else(|| {
        std::env::var(STACK_SIZE_MB_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(DEFAULT_RAYON_STACK_SIZE)
    });

    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(stack_size)
        .build_global()
        .unwrap();
}
