//! # Tetris Demo - Multi Beam Search
//!
//! Demonstrates that a multi-beam search agent (voting across N parallel searches with different
//! seeds) can play Tetris indefinitely, proving the concept of ensemble decision-making.
//!
//! ## Purpose
//!
//! This proves that combining multiple beam searches with different random seeds provides
//! robust decision-making through voting, potentially more stable than a single search.
//!
//! ## Features
//!
//! - Uses MultiBeamSearch with N parallel searches
//! - Each search uses a different random seed for diversity
//! - Votes across all searches to pick the best move
//! - Tracks unique board states and height distributions
//!
//! ## Output
//!
//! - **CSV File**: `beam_search_multisearch_output.csv` - Statistics over time
//! - **Console**: Periodic progress updates with performance metrics

use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_ml::beam_search::{BeamTetrisState, MultiBeamSearch};
use tetris_ml::set_global_threadpool;

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

/// Output file path for multi-beam search results
const OUTPUT_FILE: &str = "beam_search_multisearch_output.csv";

pub struct TetrisGameMultiSearchIter<
    const N: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> {
    pub game: TetrisGame,
    multi_search:
        MultiBeamSearch<BeamTetrisState, N, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>,
    step_counter: u64, // Used for seed generation
}

impl<
    const N: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> TetrisGameMultiSearchIter<N, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    pub fn new() -> Self {
        Self {
            game: TetrisGame::new(),
            multi_search: MultiBeamSearch::new(),
            step_counter: 0,
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            game: TetrisGame::new_with_seed(seed),
            multi_search: MultiBeamSearch::new(),
            step_counter: seed,
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.game.reset(seed);
        if let Some(s) = seed {
            self.step_counter = s;
        }
    }

    pub fn kink(&mut self) -> PlacementResult {
        self.game.play_random()
    }
}

impl<
    const N: usize,
    const TOP_N_PER_BEAM: usize,
    const BEAM_WIDTH: usize,
    const MAX_DEPTH: usize,
    const MAX_MOVES: usize,
> Iterator for TetrisGameMultiSearchIter<N, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>
{
    type Item = (TetrisBoard, TetrisPiecePlacement);

    fn next(&mut self) -> Option<Self::Item> {
        if self.game.board.is_lost() {
            return None;
        }
        let board_before = self.game.board;

        // Use MultiBeamSearch with different seeds for each search
        let first_action = self.multi_search.search_with_seeds(
            BeamTetrisState::new(self.game),
            self.step_counter, // Base seed derived from step counter
            MAX_DEPTH,
        )?;

        self.step_counter += 1;

        (self.game.apply_placement(first_action).is_lost != IsLost::LOST)
            .then_some((board_before, first_action))
    }
}

/// Entry point for the `tetris_beam_multisearch` binary.
fn main() {
    run_tetris_beam_multisearch();
}

/// Run a multi-beam-search-driven Tetris game with voting across multiple seeds.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam_multisearch() {
    set_global_threadpool();

    // --- Tunables ---
    const N: usize = 64;
    const TOP_N_PER_BEAM: usize = 64;
    const BEAM_WIDTH: usize = 64;
    const MAX_DEPTH: usize = 7;
    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const LOG_EVERY: usize = 256;
    const KINK_PROB: f64 = 0.00;
    // --------------

    println!("Starting multi-beam search with params:");
    println!("  N               = {N} (parallel searches)");
    println!("  TOP_N_PER_BEAM = {TOP_N_PER_BEAM}");
    println!("  BEAM_WIDTH      = {BEAM_WIDTH}");
    println!("  MAX_DEPTH       = {MAX_DEPTH}");
    println!("  MAX_MOVES       = {MAX_MOVES}");
    println!("  LOG_EVERY       = {LOG_EVERY}");
    println!("  KINK_PROB       = {KINK_PROB}");

    // Open output file with buffering for maximum write speed
    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut steps = 0usize;
    let mut height_counts: HashMap<u32, usize> = HashMap::new();
    let mut holes_counts: HashMap<u32, usize> = HashMap::new();
    let mut iter =
        TetrisGameMultiSearchIter::<N, TOP_N_PER_BEAM, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();
    let start = Instant::now();
    let mut rng = rand::rng();
    println!("Starting Tetris game with multi-beam search...");

    loop {
        if rng.random_bool(KINK_PROB) {
            iter.kink();
        }

        let Some((_board_before, _mv)) = iter.next() else {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "No plan found at step={steps} (all searches failed). avg_pieces_per_sec={rate:.2}"
            );
            writer.flush().expect("Failed to flush output file");
            break;
        };
        steps += 1;

        let height = iter.game.board.height();
        *height_counts.entry(height).or_insert(0) += 1;

        let holes = iter.game.board.total_holes();
        *holes_counts.entry(holes).or_insert(0) += 1;

        if steps % LOG_EVERY == 0 {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "step={steps} total_lines={} avg_pieces_per_sec={rate:.2}",
                iter.game.lines_cleared
            );

            let mut sorted_heights: Vec<_> = height_counts.iter().collect();
            sorted_heights.sort_by_key(|(height, _)| *height);
            for (height, count) in sorted_heights {
                println!("height={height} count={count}");
            }

            let mut sorted_holes: Vec<_> = holes_counts.iter().collect();
            sorted_holes.sort_by_key(|(holes, _)| *holes);
            for (holes, count) in sorted_holes {
                println!("holes={holes} count={count}");
            }

            // let mut normalized_heights = height_counts
            //     .iter()
            //     .map(|(height, count)| (height, *count as f64 / steps as f64))
            //     .collect::<Vec<_>>();
            // normalized_heights.sort_by_key(|(height, _)| *height);
            // let height_str = normalized_heights
            //     .iter()
            //     .map(|(height, count)| format!("{height},{count:.4}"))
            //     .collect::<Vec<_>>()
            //     .join(",");
            // writeln!(
            //     writer,
            //     "{steps},{unique_num_boards},{compactness:.2},{height_str}"
            // )
            // .expect("Failed to write to output file");

            // write_count += 1;
            // if write_count % 100 == 0 {
            //     writer.flush().expect("Failed to flush output file");
            // }
        }
    }
}
