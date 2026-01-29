//! # Tetris Demo - Single Beam Search
//!
//! Demonstrates that a single beam search agent can play Tetris indefinitely (or up to N pieces)
//! without losing, proving the viability of beam search for Tetris gameplay.
//!
//! ## Purpose
//!
//! This is a proof-of-concept demo showing that beam search provides sufficient lookahead
//! to make consistently good decisions and avoid game-over states.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceOrientation,
    TetrisPiecePlacement,
};
use tetris_search::{BeamSearch, BeamTetrisState, ScoredState};
use tetris_search::set_global_threadpool;

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

pub struct TetrisGameIter<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> {
    pub game: TetrisGame,
    search: BeamSearch<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>
    TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        Self {
            game: TetrisGame::new(),
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            game: TetrisGame::new_with_seed(seed),
            search: BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.game.reset(seed);
    }

    pub fn kink(&mut self) -> PlacementResult {
        self.game.play_random()
    }
}

impl<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize> Iterator
    for TetrisGameIter<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;
        let ScoredState { root_action: first_action, .. } = self
            .search
            .search_top_with_state(BeamTetrisState::new(self.game), MAX_DEPTH)?;
        (self.game.apply_placement(first_action.unwrap()).is_lost != IsLost::LOST)
            .then_some((board_before, first_action.unwrap()))
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
    lookup: Arc<DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>>,
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
    const BEAM_WIDTH: usize = 32;
    const MAX_DEPTH: usize = 8;
    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const LOG_EVERY: usize = 5_000;
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
