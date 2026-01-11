use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_atlas::beam_search::{BeamSearch, BeamTetrisState, ScoredState};
use tetris_atlas::set_global_threadpool;
use tetris_atlas::tetris::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPieceOrientation, TetrisPiecePlacement,
};

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

/// Output file path for beam search results
const OUTPUT_FILE: &str = "beam_search_output_evolved.csv";

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
        let ScoredState { first_action, .. } = self
            .search
            .search_first_action_with_state(BeamTetrisState::new(self.game), MAX_DEPTH)?;
        (self.game.apply_placement(first_action.unwrap()).is_lost != IsLost::LOST)
            .then_some((board_before, first_action.unwrap()))
    }
}

/// Entry point for the `tetris_beam` binary.
fn main() {
    run_tetris_beam();
}

/// Run a beam-search-driven Tetris game, continually planning and applying placements.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam() {
    set_global_threadpool();

    // --- Tunables ---
    const BEAM_WIDTH: usize = 16;
    const MAX_DEPTH: usize = 16;
    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS; // >= max placements per piece (usually ~40)
    const LOG_EVERY: usize = 1024;
    const KINK_PROB: f64 = 0.05;
    // --------------

    // Open output file with buffering for maximum write speed
    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);
    let mut write_count = 0usize;

    let mut state_set = HashSet::<TetrisBoard>::new();
    let mut steps = 0usize;
    let mut height_counts = HashMap::new();
    let mut iter = TetrisGameIter::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();
    let start = Instant::now();
    let mut rng = rand::rng();

    loop {
        if rng.random_bool(KINK_PROB) {
            iter.kink();
        }
        let Some((_board_before, _mv)) = iter.next() else {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!("No plan found at step={steps} (beam empty). avg_pieces_per_sec={rate:.2}");
            writer.flush().expect("Failed to flush output file");
            break;
        };
        steps += 1;

        state_set.insert(iter.game.board);

        let height = iter.game.board.height();
        *height_counts.entry(height).or_insert(0) += 1;

        if steps % LOG_EVERY == 0 {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "step={steps} total_lines={} avg_pieces_per_sec={rate:.2}",
                iter.game.lines_cleared
            );

            // let mut sorted_heights: Vec<_> = height_counts.iter().collect();
            // sorted_heights.sort_by_key(|(height, _)| *height);
            // for (height, count) in sorted_heights {
            //     println!("height={height} count={count}");
            // }

            // let mut sorted_states: Vec<_> = state_map.iter().collect();
            // sorted_states.sort_by_key(|(_board, count)| std::cmp::Reverse(*count));
            // for (i, (board, count)) in sorted_states.iter().enumerate().take(10) {
            //     println!("top {}: count={count}", i + 1);
            // }
            // println!("state_map len={}", state_map.len());

            let unique_num_boards = state_set.len();
            let compactness = unique_num_boards as f64 / steps as f64;
            let mut normalized_heights = height_counts
                .iter()
                .map(|(height, count)| (height, *count as f64 / steps as f64))
                .collect::<Vec<_>>();
            normalized_heights.sort_by_key(|(height, _)| *height);
            let height_str = normalized_heights
                .iter()
                .map(|(height, count)| format!("{height},{count:.4}"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                writer,
                "{steps},{unique_num_boards},{compactness:.2},{height_str}"
            )
            .expect("Failed to write to output file");

            write_count += 1;
            if write_count % 100 == 0 {
                writer.flush().expect("Failed to flush output file");
            }
        }

        // failsafe
        if state_set.len() > 1_000_000_000 {
            println!("state_map too large, breaking");
            writer.flush().expect("Failed to flush output file");
            break;
        }
    }
}
