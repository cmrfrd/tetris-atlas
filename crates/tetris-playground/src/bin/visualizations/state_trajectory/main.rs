#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # State Trajectory Recorder
//!
//! Plays Tetris using single beam search and records the trajectory of
//! `(board, piece, placement)` states to a compact binary file for visualization.
//!
//! ## Binary format
//!
//! The output is a flat binary file of fixed-size records (no header).
//! Each record is 218 bytes:
//!
//! | Field               | Type        | Bytes |
//! |---------------------|-------------|-------|
//! | board               | [u8; 200]   | 200   |
//! | piece               | u8          | 1     |
//! | placement           | u8          | 1     |
//! | height              | u32 LE      | 4     |
//! | lines_cleared       | u32 LE      | 4     |
//! | cumulative_lines    | u32 LE      | 4     |
//! | cell_count          | u32 LE      | 4     |
//!
//! The board is serialized as a flat 200-element array (10 cols x 20 rows,
//! column-major order) of 0/1 values.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release -p tetris-playground --bin tetris_state_trajectory
//! ```
//!
//! Then run the companion Python script:
//!
//! ```sh
//! uv run crates/tetris-playground/src/bin/visualizations/state_trajectory/plot_trajectory.py
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGame, TetrisGameConfig, TetrisPieceOrientation,
    constants,
};
use tetris_search::{BeamTetrisState, ScoredState, TetrisBeamSearch, height_mse_beam_tetris_score};

const BEAM_WIDTH: usize = 1024;
const MAX_DEPTH: usize = 8;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
const NUM_STEPS: usize = 500_000;
const LOG_EVERY: usize = 10_000;
const FLUSH_EVERY: usize = 1_000;
const OUTPUT_FILE: &str = "artifacts/data/state_trajectory.bin";

/// Per-record size in bytes: 200 (board) + 1 (piece) + 1 (placement) + 4*4 (u32 fields) = 218
const RECORD_SIZE: usize = 200 + 1 + 1 + 4 * 4;

const fn filter_fn(state: &BeamTetrisState) -> bool {
    let _ = state;
    true
}

fn board_to_flat_array(board: TetrisBoard) -> [u8; 200] {
    let limbs = board.as_limbs();
    let mut flat = [0u8; 200];
    let mut i = 0;
    for col in 0..StandardTetris::COLS {
        for row in 0..StandardTetris::ROWS {
            flat[i] = ((limbs[col] >> row) & 1) as u8;
            i += 1;
        }
    }
    flat
}

/// Write one record as raw bytes into the buffer.
fn write_record(
    writer: &mut BufWriter<File>,
    board: &[u8; 200],
    piece: u8,
    placement: u8,
    height: u32,
    lines_cleared: u32,
    cumulative_lines: u32,
    cell_count: u32,
) -> std::io::Result<()> {
    writer.write_all(board)?;
    writer.write_all(&[piece])?;
    writer.write_all(&[placement])?;
    writer.write_all(&height.to_le_bytes())?;
    writer.write_all(&lines_cleared.to_le_bytes())?;
    writer.write_all(&cumulative_lines.to_le_bytes())?;
    writer.write_all(&cell_count.to_le_bytes())?;
    Ok(())
}

fn main() {
    tetris_search::set_global_threadpool();

    let mut game = TetrisGame::new();
    let mut search =
        TetrisBeamSearch::<BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new(height_mse_beam_tetris_score)
            .with_filter(filter_fn);

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(OUTPUT_FILE).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    let file = File::create(OUTPUT_FILE).expect("Failed to create output file");
    let mut writer = BufWriter::with_capacity(RECORD_SIZE * FLUSH_EVERY, file);

    let start = Instant::now();
    let mut total_steps = 0u64;

    println!("Recording {NUM_STEPS} steps with BEAM_WIDTH={BEAM_WIDTH}, MAX_DEPTH={MAX_DEPTH}");
    println!("Record size: {RECORD_SIZE} bytes, flush every {FLUSH_EVERY} steps");

    for step in 0..NUM_STEPS {
        if game.board.is_lost() {
            println!("Game over at step {step}");
            break;
        }

        let state = BeamTetrisState::new(game);
        let Some(ScoredState {
            root_action: Some(placement),
            ..
        }) = search.search_top_with_state_best_effort(state, MAX_DEPTH)
        else {
            println!("No plan found at step {step}");
            break;
        };

        let board_flat = board_to_flat_array(game.board);
        let cell_count = game.board.count();

        let result = game.apply_placement(placement);
        if result.is_lost == IsLost::LOST {
            println!("Lost at step {step}");
            break;
        }

        write_record(
            &mut writer,
            &board_flat,
            placement.piece.into(),
            placement.orientation.index(),
            game.board.height(),
            result.lines_cleared,
            game.lines_cleared,
            cell_count,
        )
        .expect("Failed to write record");

        total_steps += 1;

        // Flush periodically so data is persisted incrementally
        if (step + 1) % FLUSH_EVERY == 0 {
            writer.flush().expect("Failed to flush");
        }

        if (step + 1) % LOG_EVERY == 0 {
            let secs = start.elapsed().as_secs_f64();
            let rate = (step + 1) as f64 / secs;
            println!(
                "step={} lines={} height={} rate={:.0}/s",
                step + 1,
                game.lines_cleared,
                game.board.height(),
                rate
            );
        }
    }

    writer.flush().expect("Failed to flush");
    let elapsed = start.elapsed().as_secs_f64();
    let file_size_mb = (total_steps as usize * RECORD_SIZE) as f64 / (1024.0 * 1024.0);
    println!("Recorded {total_steps} steps in {elapsed:.1}s ({file_size_mb:.1} MB)");
    println!("Wrote trajectory to {OUTPUT_FILE}");
}
