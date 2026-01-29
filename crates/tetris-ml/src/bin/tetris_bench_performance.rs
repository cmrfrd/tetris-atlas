//! # Tetris Benchmark - Performance Testing
//!
//! Benchmarks the SPEED/PERFORMANCE of different beam search parameter combinations
//! (beam width × depth) by measuring how fast each configuration can play 100,000 pieces.
//!
//! ## Purpose
//!
//! Identifies the performance characteristics of different beam search configurations,
//! helping find the sweet spot between search quality and computational cost.
//!
//! ## Methodology
//!
//! - Tests a 7×7 grid of (width, depth) combinations: {1,2,4,8,16,32,64} × {1,2,4,8,16,32,64}
//! - For each configuration, plays 100,000 pieces and measures time
//! - Calculates pieces/second throughput
//!
//! ## Output
//!
//! - **CSV File**: `beam_benchmark_results.csv` - Columns: beam_width, beam_depth, time_seconds
//! - **Console**: Real-time progress with throughput for each configuration
//! - Note: Configurations that lose before 100k pieces are marked as "FAILED"

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_ml::beam_search::{BeamSearch, BeamTetrisState, ScoredState};
use tetris_ml::set_global_threadpool;
use tetris_game::{IsLost, TetrisGame, TetrisPieceOrientation};

/// Output file path for benchmark results
const OUTPUT_FILE: &str = "beam_benchmark_results.csv";

/// Target number of pieces to play for each benchmark
const TARGET_PIECES: usize = 100_000;

/// Run a single benchmark configuration and return the elapsed time
fn run_benchmark<const BEAM_WIDTH: usize, const MAX_DEPTH: usize, const MAX_MOVES: usize>()
-> Option<f64> {
    let mut game = TetrisGame::new();
    let mut search = BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();
    let start = Instant::now();
    let mut pieces_played = 0;

    while pieces_played < TARGET_PIECES {
        if game.board.is_lost() {
            // Game ended before reaching target
            return None;
        }

        let ScoredState { root_action: first_action, .. } =
            search.search_top_with_state(BeamTetrisState::new(game), MAX_DEPTH)?;

        if game.apply_placement(first_action.unwrap()).is_lost == IsLost::LOST {
            // Lost the game
            return None;
        }

        pieces_played += 1;
    }

    Some(start.elapsed().as_secs_f64())
}

/// Macro to generate benchmark runs for different parameter combinations
macro_rules! run_benchmarks {
    ($writer:expr, $($width:expr, $depth:expr),* $(,)?) => {
        $(
            {
                const BEAM_WIDTH: usize = $width;
                const MAX_DEPTH: usize = $depth;
                const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

                print!("Running benchmark: width={}, depth={}...", BEAM_WIDTH, MAX_DEPTH);
                std::io::stdout().flush().unwrap();

                let elapsed = run_benchmark::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>();

                match elapsed {
                    Some(time) => {
                        println!(" completed in {:.2}s ({:.2} pieces/sec)",
                                 time, TARGET_PIECES as f64 / time);
                        writeln!($writer, "{},{},{:.6}", BEAM_WIDTH, MAX_DEPTH, time)
                            .expect("Failed to write to output file");
                    }
                    None => {
                        println!(" FAILED (game ended before reaching {} pieces)", TARGET_PIECES);
                        writeln!($writer, "{},{},FAILED", BEAM_WIDTH, MAX_DEPTH)
                            .expect("Failed to write to output file");
                    }
                }

                $writer.flush().expect("Failed to flush output file");
            }
        )*
    };
}

fn main() {
    set_global_threadpool();

    // Open output file
    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    // Write CSV header
    writeln!(writer, "beam_width,beam_depth,time_seconds").expect("Failed to write header");

    println!("Starting beam search parameter sweep benchmark");
    println!("Target: {} pieces per configuration", TARGET_PIECES);
    println!("Results will be saved to: {}", OUTPUT_FILE);
    println!();

    // Generate all combinations of widths and depths
    // Using powers of 2 and some intermediate values for a good coverage
    // This creates a 7x7 grid = 49 combinations
    run_benchmarks!(
        writer, // Width 1
        1, 1, 1, 2, 1, 4, 1, 8, 1, 16, 1, 32, 1, 64, // Width 2
        2, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2, 32, 2, 64, // Width 4
        4, 1, 4, 2, 4, 4, 4, 8, 4, 16, 4, 32, 4, 64, // Width 8
        8, 1, 8, 2, 8, 4, 8, 8, 8, 16, 8, 32, 8, 64, // Width 16
        16, 1, 16, 2, 16, 4, 16, 8, 16, 16, 16, 32, 16, 64, // Width 32
        32, 1, 32, 2, 32, 4, 32, 8, 32, 16, 32, 32, 32, 64, // Width 64
        64, 1, 64, 2, 64, 4, 64, 8, 64, 16, 64, 32, 64, 64,
    );

    println!();
    println!("Benchmark complete! Results saved to {}", OUTPUT_FILE);
}
