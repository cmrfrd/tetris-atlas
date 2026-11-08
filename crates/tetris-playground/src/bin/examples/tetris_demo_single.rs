#![feature(generic_const_exprs)]
//! # Tetris Demo - Single Beam Search
//!
//! Runs a single beam-search agent and records its finite gameplay results.
//!
//! ## Purpose
//!
//! Evaluates a beam-search heuristic under seeded 7-bag draws. Long runs provide empirical
//! evidence; this demo does not verify a policy-closed set or prove infinite survival.
//!
//! ## Features
//!
//! - Plays Tetris using a single BeamSearch instance
//! - Optional "kink" moves (random placements) for exploration
//! - Tracks unique board states encountered
//! - Logs statistics: pieces placed, unique boards, height distribution
//!
//! ## Output
//!
//! - **CSV File**: `beam_search_output_evolved.csv` - Statistics over time
//! - **Console**: Periodic updates on progress and performance

use dashmap::DashMap;
use rand::RngExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tetris_game::{
    PlacementResult, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceOrientation,
    TetrisPiecePlacement,
};
use tetris_search::set_global_threadpool;
use tetris_search::{BeamTetrisState, ScoredState, TetrisBeamSearch, height_mse_beam_tetris_score};

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

const fn filter_fn(_state: &BeamTetrisState) -> bool {
    // state.0.board.height() <= 4
    true
}

pub struct TetrisGameIter<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> {
    pub game: TetrisGame,
    search: TetrisBeamSearch<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Default
    for TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        Self {
            game: <TetrisGame>::new(),
            search: TetrisBeamSearch::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(
                height_mse_beam_tetris_score,
            )
            .with_filter(filter_fn),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            game: <TetrisGame>::new_with_seed(seed),
            search: TetrisBeamSearch::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(
                height_mse_beam_tetris_score,
            )
            .with_filter(filter_fn),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.game.reset(seed);
    }

    pub fn kink<R: RngExt + ?Sized>(&mut self, rng: &mut R) -> PlacementResult {
        let placements = self.game.current_placements();
        let placement = placements[rng.random_range(0..placements.len())];
        self.game.apply_placement(placement)
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard<10, 20>, TetrisPiecePlacement<10>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;
        let ScoredState {
            root_action: first_action,
            ..
        } = self
            .search
            .search_top_with_state_best_effort(BeamTetrisState::new(self.game), MAX_DEPTH)?;
        (!self
            .game
            .apply_placement(
                first_action.expect("successful search must provide a first action in next"),
            )
            .is_lost)
            .then_some((
                board_before,
                first_action.expect("successful search must provide a first action in next"),
            ))
    }
}

/// Entry point for the `tetris_beam` binary.
fn main() {
    run_tetris_beam();
}

/// Shared state between worker and logger threads
#[derive(Clone)]
struct SharedState {
    /// Concurrent hashmap for (board, piece) -> orientation lookup
    lookup: Arc<DashMap<(TetrisBoard<10, 20>, TetrisPiece), TetrisPieceOrientation<10>>>,
    /// Atomic counter for steps/pieces placed
    steps: Arc<AtomicUsize>,
    /// Concurrent hashmap for height distribution
    height_counts: Arc<DashMap<usize, usize>>,
    /// Thread-safe lines cleared counter
    lines_cleared: Arc<AtomicUsize>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Start time
    start: Instant,
}

impl SharedState {
    fn new() -> Self {
        Self {
            lookup: Arc::new(DashMap::new()),
            steps: Arc::new(AtomicUsize::new(0)),
            height_counts: Arc::new(DashMap::new()),
            lines_cleared: Arc::new(AtomicUsize::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            start: Instant::now(),
        }
    }
}

/// Worker thread: runs beam search iteration and populates lookup table
#[allow(
    clippy::cast_precision_loss,
    reason = "Floating-point values here are approximate scores, tensor math, or reporting statistics; exact state identifiers remain integers."
)]
fn worker_thread<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>(
    state: SharedState,
) {
    let mut iter = TetrisGameIter::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();

    loop {
        // Check shutdown signal
        if state.shutdown.load(Ordering::Relaxed) {
            println!("Worker thread: shutdown signal received, stopping...");
            break;
        }

        // Get next move from beam search
        let Some((board_before, mv)) = iter.next() else {
            let steps = state.steps.load(Ordering::Relaxed);
            let secs = state.start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "Worker thread: No plan found at step={steps} (beam empty). avg_pieces_per_sec={rate:.2}"
            );
            state.shutdown.store(true, Ordering::Relaxed);
            break;
        };

        // Update lookup table (concurrent write)
        state
            .lookup
            .insert((board_before, mv.piece), mv.orientation);

        // Update statistics
        state.steps.fetch_add(1, Ordering::Relaxed);
        state
            .lines_cleared
            .store(iter.game.lines_cleared as usize, Ordering::Relaxed);

        let height = iter.game.board.height() as usize;
        state
            .height_counts
            .entry(height)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }

    println!("Worker thread: exiting cleanly");
}

/// Logger thread: periodically logs statistics
fn logger_thread(state: SharedState, log_every: usize) {
    let log_interval = Duration::from_millis(1000); // Check every second
    let mut last_logged_step = 0usize;
    let mut initial_log_done = false;

    loop {
        thread::sleep(log_interval);

        // Check shutdown signal
        if state.shutdown.load(Ordering::Relaxed) {
            println!("Logger thread: shutdown signal received, printing final stats...");
            print_stats(&state);
            break;
        }

        let steps = state.steps.load(Ordering::Relaxed);

        // Log early to show progress is happening
        if !initial_log_done && steps >= 10 {
            println!("Worker thread is running... (logged at {} steps)", steps);
            print_stats(&state);
            initial_log_done = true;
        }

        // Log whenever we've crossed a threshold
        if steps > 0 && steps >= last_logged_step + log_every {
            print_stats(&state);
            last_logged_step = (steps / log_every) * log_every; // Round down to nearest multiple
        }
    }

    println!("Logger thread: exiting cleanly");
}

/// Print current statistics
#[allow(
    clippy::cast_precision_loss,
    reason = "Floating-point values here are approximate scores, tensor math, or reporting statistics; exact state identifiers remain integers."
)]
fn print_stats(state: &SharedState) {
    let steps = state.steps.load(Ordering::Relaxed);
    let lines_cleared = state.lines_cleared.load(Ordering::Relaxed);
    let secs = state.start.elapsed().as_secs_f64().max(1e-9);
    let rate = steps as f64 / secs;

    println!("step={steps} total_lines={lines_cleared} avg_pieces_per_sec={rate:.2}");

    // Read height distribution (concurrent read)
    let mut sorted_heights: Vec<_> = state
        .height_counts
        .iter()
        .map(|entry| (*entry.key(), *entry.value()))
        .collect();
    sorted_heights.sort_by_key(|(height, _)| *height);
    for (height, count) in sorted_heights {
        println!("height={height} count={count}");
    }

    // Read lookup table size (concurrent read)
    println!("lookup len={}", state.lookup.len());
}

/// Run a beam-search-driven Tetris game with multithreading and graceful shutdown.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam() {
    set_global_threadpool();

    // --- Tunables ---
    const BEAM_WIDTH: usize = 256;
    const MAX_DEPTH: usize = 8;
    const MAX_MOVES: usize = <tetris_game::TetrisPieceOrientation<10>>::TOTAL_NUM_ORIENTATIONS;
    const LOG_EVERY: usize = 100;
    // --------------

    println!("Starting beam search with BEAM_WIDTH={BEAM_WIDTH}, MAX_DEPTH={MAX_DEPTH}");
    println!("Press Ctrl+C to stop gracefully...");

    // Create shared state
    let state = SharedState::new();

    // Setup Ctrl+C handler
    let shutdown_state = state.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received! Initiating graceful shutdown...");
        shutdown_state.shutdown.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    // Spawn worker thread
    let worker_state = state.clone();
    let worker_handle = thread::spawn(move || {
        worker_thread::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>(worker_state);
    });

    // Spawn logger thread
    let logger_state = state.clone();
    let logger_handle = thread::spawn(move || {
        logger_thread(logger_state, LOG_EVERY);
    });

    // Wait for threads to complete
    println!("Waiting for worker thread to finish...");
    worker_handle.join().expect("Worker thread panicked");

    println!("Waiting for logger thread to finish...");
    logger_handle.join().expect("Logger thread panicked");

    println!(
        "All threads exited cleanly. Final lookup table size: {}",
        state.lookup.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn random_moves_replay_without_changing_piece_stream() {
        let mut first = TetrisGameIter::<
            8,
            2,
            { <tetris_game::TetrisPieceOrientation<10>>::TOTAL_NUM_ORIENTATIONS },
        >::new_with_seed(42);
        let mut second = TetrisGameIter::<
            8,
            2,
            { <tetris_game::TetrisPieceOrientation<10>>::TOTAL_NUM_ORIENTATIONS },
        >::new_with_seed(42);
        let mut control = first.game;
        let mut first_rng = SmallRng::seed_from_u64(123);
        let mut second_rng = SmallRng::seed_from_u64(123);

        // Keep moves live while crossing multiple bag refills.
        for _ in 0..16 {
            first.game.board.clear();
            second.game.board.clear();
            control.board.clear();
            let before = first.game;
            let result = first.kink(&mut first_rng);
            assert!(!result.is_lost);
            assert_eq!(result, second.kink(&mut second_rng));
            assert_eq!(first.game, second.game);
            assert!(before.current_placements().iter().any(|&placement| {
                let mut board = before.board;
                board.apply_piece_placement(placement);
                board == first.game.board
            }));

            // A different move policy must see the same future piece stream.
            let placement = control.current_placements()[0];
            assert!(!control.apply_placement(placement).is_lost);
            assert_eq!(first.game.current_piece(), control.current_piece());
            assert_eq!(first.game.bag(), control.bag());
            assert_eq!(first.game.rng, control.rng);
        }
    }
}
