//! # Tetris Demo - Adaptive Multi Beam Search
//!
//! Similar to `tetris_demo_multi.rs`, but adapts search depth and beam width based on
//! board occupancy.
//!
//! This keeps cheap searches for easy boards while spending more compute on
//! difficult positions.

use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_game::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_search::set_global_threadpool;
use tetris_search::{BeamTetrisState, MultiBeamSearch};

/// Output file path for adaptive multi-beam search results
const OUTPUT_FILE: &str = "beam_search_multisearch_adaptive_output.csv";

/// Entry point for the `tetris_demo_adaptive_multi` binary.
fn main() {
    run_tetris_beam_multisearch_adaptive();
}

/// Run an adaptive multi-beam-search-driven Tetris game.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam_multisearch_adaptive() {
    set_global_threadpool();

    const N_BEAMS: usize = 32;
    const TOP_N_PER_BEAM: usize = 8;
    const BEAM_WIDTH: usize = 2048;
    const MAX_DEPTH: usize = 16;

    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const LOG_EVERY: usize = 64;
    const KINK_PROB: f64 = 0.00;
    // --------------

    println!("Starting adaptive multi-beam search with params:");
    println!(
        "  N_BEAMS       = {N_BEAMS} TOP_N_PER_BEAM={TOP_N_PER_BEAM} BEAM_WIDTH={BEAM_WIDTH} MAX_DEPTH={MAX_DEPTH}"
    );
    println!("  MAX_MOVES       = {MAX_MOVES}");
    println!("  LOG_EVERY       = {LOG_EVERY}");
    println!("  KINK_PROB       = {KINK_PROB}");

    // Open output file with buffering for maximum write speed
    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::new(file);

    let mut steps = 0usize;
    let mut height_counts: HashMap<u32, usize> = HashMap::new();
    let mut holes_counts: HashMap<u32, usize> = HashMap::new();
    let mut count_counts: HashMap<u32, usize> = HashMap::new();

    let mut game = TetrisGame::new();
    let mut step_counter = 0u64;

    let mut multi_beam_search: MultiBeamSearch<
        BeamTetrisState,
        N_BEAMS,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    > = MultiBeamSearch::new();

    let start = Instant::now();
    let mut rng = rand::rng();
    println!("Starting Tetris game with adaptive multi-beam search...");

    loop {
        if rng.random_bool(KINK_PROB) {
            let _ = game.play_random();
        }

        if game.board.is_lost() {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!("Game lost at step={steps}. avg_pieces_per_sec={rate:.2}");
            writer.flush().expect("Failed to flush output file");
            break;
        }

        let board_before = game.board;
        let height = board_before.height();
        let holes = board_before.total_holes();
        let count = board_before.count();

        let (depth, beam_width) = match count {
            0..=8 => (2, 64),
            10..=12 => (4, 256),
            14..=16 => (6, 512),
            18..=22 => (8, 1024),
            _ => (10, 2048),
        };

        let state = BeamTetrisState::new(game);
        let action = multi_beam_search.search_with_seeds(state, step_counter, depth, beam_width);

        let Some(first_action) = action else {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "No plan found at step={steps} (all searches failed). avg_pieces_per_sec={rate:.2}"
            );
            writer.flush().expect("Failed to flush output file");
            break;
        };

        step_counter += 1;

        if (game.apply_placement(first_action).is_lost == IsLost::LOST) {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!("Move lost at step={steps}. avg_pieces_per_sec={rate:.2}");
            writer.flush().expect("Failed to flush output file");
            break;
        }

        steps += 1;
        *height_counts.entry(height as u32).or_insert(0) += 1;
        *holes_counts.entry(holes as u32).or_insert(0) += 1;
        *count_counts.entry(count as u32).or_insert(0) += 1;

        if steps % LOG_EVERY == 0 {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = steps as f64 / secs;
            println!(
                "step={steps} total_lines={} avg_pieces_per_sec={rate:.2}",
                game.lines_cleared
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

            let mut sorted_counts: Vec<_> = count_counts.iter().collect();
            sorted_counts.sort_by_key(|(count, _)| *count);
            for (count, cell_count) in sorted_counts {
                println!("count={count} cell_count={cell_count}");
            }
        }
    }
}
