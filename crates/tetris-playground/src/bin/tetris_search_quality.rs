//! # Tetris Search - Quality Parameter Search
//!
//! Performs a grid search over MultiBeamSearch parameters (N, beam_width, max_depth) to find
//! the configuration that produces the HIGHEST QUALITY gameplay (most pieces placed, not speed).
//!
//! ## Purpose
//!
//! Identifies which parameter combination leads to the longest survival and best gameplay,
//! helping optimize for decision quality rather than computational efficiency.
//!
//! ## Methodology
//!
//! - Tests all combinations of:
//!   - N (parallel searches): {1, 2, 4, 8, 16, 32}
//!   - beam_width: {4, 8, 16, 32}
//!   - max_depth: {1, 2, 4, 8, 16, 32}
//! - For each configuration: runs 10 games up to 1000 pieces, measures:
//!   - Average pieces placed
//!   - Average score
//!   - Average board height
//!
//! ## Output
//!
//! - **CSV File**: `param_search_results.csv` - Full results with all metrics
//! - **Console**: Top/bottom configurations sorted by pieces placed
//! - Real-time progress updates showing completion percentage

use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tetris_game::{TetrisGame, TetrisPieceOrientation};
use tetris_search::{BeamTetrisState, MultiBeamSearch};

const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

/// Macro to generate all parameter combinations and run them
/// The trick is to use incremental macro expansion for nested loops
macro_rules! cartesian_product_3 {
    // Base case: generate function calls for fixed beam_width and depth, varying n
    (@inner $callback:ident, $beam:expr, $depth:expr, [$($n:expr),*]) => {
        vec![
            $(
                Box::new(|| $callback::<$n, $beam, $depth>()) as Box<dyn Fn() -> ParamSearchResult + Send + Sync>,
            )*
        ]
    };

    // Recurse over depth values
    (@depth $callback:ident, $beam:expr, [$depth:expr $(, $rest_depth:expr)*], $n_list:tt) => {{
        let mut v = cartesian_product_3!(@inner $callback, $beam, $depth, $n_list);
        $(
            v.extend(cartesian_product_3!(@inner $callback, $beam, $rest_depth, $n_list));
        )*
        v
    }};

    // Entry point: recurse over beam_width values
    ($callback:ident, beam_width: [$beam:expr $(, $rest_beam:expr)*], depth: $depth_list:tt, n: $n_list:tt) => {{
        let mut tasks = cartesian_product_3!(@depth $callback, $beam, $depth_list, $n_list);
        $(
            tasks.extend(cartesian_product_3!(@depth $callback, $rest_beam, $depth_list, $n_list));
        )*
        tasks
    }};
}

#[derive(Debug, Clone)]
pub struct ParamSearchResult {
    pub n: usize,
    pub beam_width: usize,
    pub max_depth: usize,
    pub avg_score: f32,
    pub avg_pieces: f32,
    pub avg_height: f32,
    pub games_completed: usize,
}

// The function that runs for each parameter combination
fn run_single_param_search<const N: usize, const BEAM_WIDTH: usize, const MAX_DEPTH: usize>()
-> ParamSearchResult {
    let mut multi_search =
        MultiBeamSearch::<BeamTetrisState, N, 1, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();

    // Run your benchmark here
    let num_games = 10;
    let mut total_score = 0.0;
    let mut total_pieces = 0.0;
    let mut total_height = 0.0;
    let mut height_samples = 0;
    let mut completed = 0;

    for seed in 0..num_games {
        let mut game = TetrisGame::new_with_seed(seed);
        let mut pieces_placed = 0;

        while !game.board.is_lost() && pieces_placed < 1000 {
            let state = BeamTetrisState::new(game);

            // Track board height (max of all column heights)
            let heights = game.board.heights();
            let max_height = heights.iter().max().copied().unwrap_or(0);
            total_height += max_height as f32;
            height_samples += 1;

            if let Some(action) =
                multi_search.search_with_seeds(state, seed * 1000, MAX_DEPTH)
            {
                game.apply_placement(action);
                pieces_placed += 1;
            } else {
                break;
            }
        }

        total_pieces += pieces_placed as f32;
        if pieces_placed > 0 {
            completed += 1;
        }
    }

    ParamSearchResult {
        n: N,
        beam_width: BEAM_WIDTH,
        max_depth: MAX_DEPTH,
        avg_score: total_score / num_games as f32,
        avg_pieces: total_pieces / num_games as f32,
        avg_height: if height_samples > 0 {
            total_height / height_samples as f32
        } else {
            0.0
        },
        games_completed: completed,
    }
}

pub fn run_all_param_searches_parallel() -> Vec<ParamSearchResult> {
    let tasks = cartesian_product_3!(
        run_single_param_search,
        beam_width: [4, 8, 16, 32],
        depth: [1, 2, 4, 8, 16, 32],
        n: [1, 2, 4, 8, 16, 32]
    );

    let total = tasks.len();
    let completed = Arc::new(AtomicUsize::new(0));

    println!("Running {} parameter combinations in parallel...", total);

    let results: Vec<ParamSearchResult> = tasks
        .par_iter()
        .map(|task| {
            let result = task();
            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;

            println!(
                "Completed {}/{} ({:.1}%): N={}, BEAM={}, DEPTH={} -> {:.2} pieces",
                done,
                total,
                (done as f32 / total as f32) * 100.0,
                result.n,
                result.beam_width,
                result.max_depth,
                result.avg_pieces
            );

            result
        })
        .collect();

    println!("All {} combinations completed!", total);
    results
}

fn main() {
    // Optional: Set rayon thread pool size
    let num_threads = num_cpus::get();
    println!("Using {} threads for parallel execution", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let start = Instant::now();
    let results = run_all_param_searches_parallel();
    let duration = start.elapsed();

    println!("\n=== Parameter Search Results ===");
    println!("Total combinations tested: {}", results.len());
    println!("Total time: {:.2}s", duration.as_secs_f32());
    println!(
        "Average time per combination: {:.2}s",
        duration.as_secs_f32() / results.len() as f32
    );

    // Sort by average pieces (descending)
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.avg_pieces.partial_cmp(&a.avg_pieces).unwrap());

    println!("\nTop 10 configurations by pieces placed:");
    println!(
        "{:>4} {:>10} {:>6} | {:>12} {:>12} {:>11} {:>8}",
        "N", "BEAM", "DEPTH", "AVG_PIECES", "AVG_SCORE", "AVG_HEIGHT", "GAMES"
    );
    println!("{}", "-".repeat(80));

    for result in sorted.iter() {
        println!(
            "{:>4} {:>10} {:>6} | {:>12.2} {:>12.2} {:>11.2} {:>8}",
            result.n,
            result.beam_width,
            result.max_depth,
            result.avg_pieces,
            result.avg_score,
            result.avg_height,
            result.games_completed
        );
    }

    // Find best by each metric
    let best_pieces = sorted.first().unwrap();
    let best_score = results
        .iter()
        .max_by(|a, b| a.avg_score.partial_cmp(&b.avg_score).unwrap())
        .unwrap();

    println!("\n=== Best Configurations ===");
    println!(
        "Best by pieces placed: N={}, BEAM={}, DEPTH={} -> {:.2} pieces",
        best_pieces.n, best_pieces.beam_width, best_pieces.max_depth, best_pieces.avg_pieces
    );
    println!(
        "Best by score: N={}, BEAM={}, DEPTH={} -> {:.2} score",
        best_score.n, best_score.beam_width, best_score.max_depth, best_score.avg_score
    );

    // Save to CSV
    println!("\nSaving results to param_search_results.csv...");
    let mut csv =
        String::from("n,beam_width,max_depth,avg_pieces,avg_score,avg_height,games_completed\n");
    for result in &results {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            result.n,
            result.beam_width,
            result.max_depth,
            result.avg_pieces,
            result.avg_score,
            result.avg_height,
            result.games_completed
        ));
    }
    std::fs::write("param_search_results.csv", csv).expect("Failed to write CSV");

    println!("\nDone! Results saved to param_search_results.csv");

    // Print worst performers too for comparison
    println!("\nBottom 5 configurations by pieces placed:");
    println!(
        "{:>4} {:>10} {:>6} | {:>12} {:>12} {:>11} {:>8}",
        "N", "BEAM", "DEPTH", "AVG_PIECES", "AVG_SCORE", "AVG_HEIGHT", "GAMES"
    );
    println!("{}", "-".repeat(80));

    for result in sorted.iter().rev().take(5) {
        println!(
            "{:>4} {:>10} {:>6} | {:>12.2} {:>12.2} {:>11.2} {:>8}",
            result.n,
            result.beam_width,
            result.max_depth,
            result.avg_pieces,
            result.avg_score,
            result.avg_height,
            result.games_completed
        );
    }
}
