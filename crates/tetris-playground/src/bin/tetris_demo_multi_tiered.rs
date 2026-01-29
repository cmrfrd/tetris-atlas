//! # Tetris Demo - Tiered Multi Beam Search
//!
//! Similar to `tetris_demo_multi.rs`, but selects between low/med/high beam search
//! configurations based on board difficulty (height + holes).
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
use tetris_search::{BeamTetrisState, MultiBeamSearch};
use tetris_search::set_global_threadpool;

/// Output file path for tiered multi-beam search results
const OUTPUT_FILE: &str = "beam_search_multisearch_tiered_output.csv";

#[derive(Clone, Copy, Debug)]
enum BeamTier {
    Low,
    Med,
    High,
}

#[inline(always)]
fn select_tier(height: u32) -> BeamTier {
    // Height-first partition with holes as a soft gate
    match height {
        0..=1 => BeamTier::Low,
        2..=3 => BeamTier::Med,
        _ => BeamTier::High,
    }
}

/// Entry point for the `tetris_demo_multi_tiered` binary.
fn main() {
    run_tetris_beam_multisearch_tiered();
}

/// Run a tiered multi-beam-search-driven Tetris game.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam_multisearch_tiered() {
    set_global_threadpool();

    // --- Tunables ---
    const LOW_N: usize = 8;
    const LOW_TOP_N_PER_BEAM: usize = 16;
    const LOW_BEAM_WIDTH: usize = 16;
    const LOW_MAX_DEPTH: usize = 4;

    const MED_N: usize = 64;
    const MED_TOP_N_PER_BEAM: usize = 32;
    const MED_BEAM_WIDTH: usize = 64;
    const MED_MAX_DEPTH: usize = 6;

    const HIGH_N: usize = 96;
    const HIGH_TOP_N_PER_BEAM: usize = 48;
    const HIGH_BEAM_WIDTH: usize = 96;
    const HIGH_MAX_DEPTH: usize = 8;

    const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
    const LOG_EVERY: usize = 256;
    const KINK_PROB: f64 = 0.00;
    // --------------

    println!("Starting tiered multi-beam search with params:");
    println!(
        "  LOW  = N={LOW_N} TOP={LOW_TOP_N_PER_BEAM} WIDTH={LOW_BEAM_WIDTH} DEPTH={LOW_MAX_DEPTH}"
    );
    println!(
        "  MED  = N={MED_N} TOP={MED_TOP_N_PER_BEAM} WIDTH={MED_BEAM_WIDTH} DEPTH={MED_MAX_DEPTH}"
    );
    println!(
        "  HIGH = N={HIGH_N} TOP={HIGH_TOP_N_PER_BEAM} WIDTH={HIGH_BEAM_WIDTH} DEPTH={HIGH_MAX_DEPTH}"
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
    let mut tier_counts: HashMap<&'static str, usize> = HashMap::new();

    let mut game = TetrisGame::new();
    let mut step_counter = 0u64;

    let mut beam_low: MultiBeamSearch<
        BeamTetrisState,
        LOW_N,
        LOW_TOP_N_PER_BEAM,
        LOW_BEAM_WIDTH,
        LOW_MAX_DEPTH,
        MAX_MOVES,
    > = MultiBeamSearch::new();
    let mut beam_med: MultiBeamSearch<
        BeamTetrisState,
        MED_N,
        MED_TOP_N_PER_BEAM,
        MED_BEAM_WIDTH,
        MED_MAX_DEPTH,
        MAX_MOVES,
    > = MultiBeamSearch::new();
    let mut beam_high: MultiBeamSearch<
        BeamTetrisState,
        HIGH_N,
        HIGH_TOP_N_PER_BEAM,
        HIGH_BEAM_WIDTH,
        HIGH_MAX_DEPTH,
        MAX_MOVES,
    > = MultiBeamSearch::new();

    let start = Instant::now();
    let mut rng = rand::rng();
    println!("Starting Tetris game with tiered multi-beam search...");

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
        let tier = select_tier(height);

        let state = BeamTetrisState::new(game);
        let action = match tier {
            BeamTier::Low => beam_low.search_with_seeds(state, step_counter, LOW_MAX_DEPTH),
            BeamTier::Med => beam_med.search_with_seeds(state, step_counter, MED_MAX_DEPTH),
            BeamTier::High => beam_high.search_with_seeds(state, step_counter, HIGH_MAX_DEPTH),
        };

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
        *height_counts.entry(height).or_insert(0) += 1;
        *holes_counts.entry(holes).or_insert(0) += 1;
        let tier_key = match tier {
            BeamTier::Low => "low",
            BeamTier::Med => "med",
            BeamTier::High => "high",
        };
        *tier_counts.entry(tier_key).or_insert(0) += 1;

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

            let low = *tier_counts.get("low").unwrap_or(&0);
            let med = *tier_counts.get("med").unwrap_or(&0);
            let high = *tier_counts.get("high").unwrap_or(&0);
            println!("tiers: low={low} med={med} high={high}");
        }
    }
}
