use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_search::scoring::height_mse_distance_from_empty;

use crate::common::*;

#[derive(Args)]
pub struct LfpArgs {
    /// Maximum board height for safe-set boards.
    #[arg(long, default_value_t = 3)]
    max_height: u32,

    /// Maximum holes for safe-set boards.
    #[arg(long, default_value_t = 0)]
    max_holes: u32,

    /// Maximum board height during intermediate placements within a bag.
    #[arg(long, default_value_t = 5)]
    inter_max_height: u32,

    /// Maximum holes during intermediate placements within a bag.
    #[arg(long, default_value_t = 2)]
    inter_max_holes: u32,

    /// Maximum safe-set size before aborting.
    #[arg(long, default_value_t = 10000)]
    max_boards: usize,

    /// Beam width for finding new boards when can_reach_any fails.
    #[arg(long, default_value_t = 256)]
    beam_width: usize,
}

/// Beam search over 7 pieces: returns up to `width` terminal boards meeting safe-set constraints.
/// Sorted by height_mse_distance (best first).
fn beam_find_admissible(
    start: TetrisBoard,
    perm: &[TetrisPiece; 7],
    width: usize,
    inter_max_height: u32,
    inter_max_holes: u32,
    safe_max_height: u32,
    safe_max_holes: u32,
) -> Vec<(u64, TetrisBoard, f32)> {
    // Beam: (board, score)
    let mut beam: Vec<(TetrisBoard, f32)> = vec![(start, -height_mse_distance_from_empty(start))];

    for step in 0..7 {
        let piece = perm[step];
        let mut candidates: Vec<(TetrisBoard, f32)> = Vec::with_capacity(beam.len() * 34);

        for &(parent_board, _) in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = parent_board;
                let result = next.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }
                if next.height() > inter_max_height {
                    continue;
                }
                if inter_max_holes != u32::MAX && has_too_many_holes(&next, inter_max_holes) {
                    continue;
                }
                let score = -height_mse_distance_from_empty(next);
                candidates.push((next, score));
            }
        }

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut seen = FxHashSet::default();
        beam.clear();
        for (board, score) in candidates {
            if seen.insert(board) {
                beam.push((board, score));
                if beam.len() >= width {
                    break;
                }
            }
        }

        if beam.is_empty() {
            break;
        }
    }

    // Filter to safe-set admissible boards
    let mut results: Vec<(u64, TetrisBoard, f32)> = beam
        .into_iter()
        .filter(|(board, _)| {
            board.height() <= safe_max_height
                && (safe_max_holes == u32::MAX || !has_too_many_holes(board, safe_max_holes))
        })
        .map(|(board, _)| {
            let dist = height_mse_distance_from_empty(board);
            (board_hash(&board), board, dist)
        })
        .collect();

    results.sort_by(|a, b| a.2.total_cmp(&b.2));
    results
}

pub fn run(args: LfpArgs) -> Result<()> {
    println!("tetris_bag_checkpoint lfp (constructive least-fixed-point expansion)");
    println!(
        "safe_set         = height<={} holes<={}",
        args.max_height, args.max_holes
    );
    println!(
        "intermediate     = height<={} holes<={}",
        args.inter_max_height, args.inter_max_holes
    );
    println!("max_boards       = {}", args.max_boards);
    println!("beam_width       = {}", args.beam_width);
    println!();

    let perms = generate_permutations();
    let num_perms = perms.len();

    // Start with the empty board
    let mut boards: Vec<TetrisBoard> = vec![TetrisBoard::EMPTY_BOARD];
    let mut board_hashes: FxHashSet<u64> = FxHashSet::default();
    board_hashes.insert(board_hash(&TetrisBoard::EMPTY_BOARD));

    let mut verified_up_to = 0usize;
    let overall_start = Instant::now();
    let mut iteration = 0u32;

    println!("=== LFP Expansion ===");
    println!("Starting with S = {{empty_board}}");
    println!();

    loop {
        iteration += 1;
        let iter_start = Instant::now();
        let check_from = verified_up_to;
        let check_to = boards.len();

        if check_from >= check_to {
            println!(
                "LFP CONVERGED at iteration {} — safe set size = {}",
                iteration,
                boards.len()
            );
            break;
        }

        println!(
            "--- iteration {} | checking boards [{}, {}) | set_size={} ---",
            iteration,
            check_from,
            check_to,
            boards.len()
        );

        let mut boards_added = 0usize;
        let mut boards_ok = 0usize;
        let mut perms_failed = 0usize;
        let mut dead_perms = 0usize; // perms where no admissible board found even via beam

        for board_idx in check_from..check_to {
            let board = boards[board_idx];
            let mut board_fails = 0u32;

            for perm in &perms {
                let mut nodes = 0u64;
                let reached = can_reach_any(
                    board,
                    perm,
                    0,
                    args.inter_max_height,
                    args.inter_max_holes,
                    &board_hashes,
                    &mut nodes,
                );

                if reached {
                    continue;
                }

                perms_failed += 1;
                board_fails += 1;

                if boards.len() >= args.max_boards {
                    continue;
                }

                // Use beam search to find admissible boards (fast)
                let candidates = beam_find_admissible(
                    board,
                    perm,
                    args.beam_width,
                    args.inter_max_height,
                    args.inter_max_holes,
                    args.max_height,
                    args.max_holes,
                );

                // Add the best candidate not already in the set
                let mut added = false;
                for (hash, new_board, _) in &candidates {
                    if !board_hashes.contains(hash) {
                        board_hashes.insert(*hash);
                        boards.push(*new_board);
                        boards_added += 1;
                        added = true;
                        break;
                    }
                }

                if !added && candidates.is_empty() {
                    dead_perms += 1;
                }
            }

            if board_fails == 0 {
                boards_ok += 1;
            }

            let checked = board_idx - check_from + 1;
            if checked % 10 == 0 || board_idx + 1 == check_to {
                eprint!(
                    "\r  [{}/{}] ok={} added={} failed_perms={} dead_perms={} set_size={} time={:.1}s",
                    checked,
                    check_to - check_from,
                    boards_ok,
                    boards_added,
                    perms_failed,
                    dead_perms,
                    boards.len(),
                    iter_start.elapsed().as_secs_f64(),
                );
                let _ = std::io::stderr().flush();
            }
        }
        eprintln!();

        verified_up_to = check_to;

        let iter_time = iter_start.elapsed().as_secs_f64();
        println!(
            "  result: ok={} added={} failed={} dead={} set={} time={:.1}s",
            boards_ok,
            boards_added,
            perms_failed,
            dead_perms,
            boards.len(),
            iter_time,
        );

        if boards.len() >= args.max_boards {
            println!("  Hit max_boards cap ({}).", args.max_boards);
            break;
        }

        if boards_added == 0 {
            if perms_failed > 0 {
                println!(
                    "  No new boards found but {} perms fail — dead end.",
                    perms_failed
                );
            }
            break;
        }
    }

    let expansion_time = overall_start.elapsed().as_secs_f64();

    println!();
    println!("=== LFP Expansion Result ===");
    println!("total_boards     = {}", boards.len());
    println!("expansion_time   = {:.1}s", expansion_time);

    // -----------------------------------------------------------------------
    // GFP pruning pass: remove boards that can't reach the set (parallel)
    // -----------------------------------------------------------------------
    println!();
    println!("=== GFP Pruning (parallel) ===");

    let board_hash_vec: Vec<u64> = boards.iter().map(|b| board_hash(b)).collect();
    let mut live_hashes: FxHashSet<u64> = board_hashes.clone();
    let mut gfp_iter = 0u32;

    loop {
        gfp_iter += 1;
        let iter_start = Instant::now();
        let prev_count = live_hashes.len();
        let current_safe: FxHashSet<u64> = live_hashes.clone();

        let checked = AtomicUsize::new(0);
        let ok_count = AtomicUsize::new(0);
        let fail_count = AtomicUsize::new(0);

        let results: Vec<(u64, bool)> = boards
            .par_iter()
            .zip(board_hash_vec.par_iter())
            .filter_map(|(board, hash)| {
                if !current_safe.contains(hash) {
                    return None;
                }

                let mut all_ok = true;
                for perm in &perms {
                    let mut nodes = 0u64;
                    if !can_reach_any(
                        *board,
                        perm,
                        0,
                        args.inter_max_height,
                        args.inter_max_holes,
                        &current_safe,
                        &mut nodes,
                    ) {
                        all_ok = false;
                        break;
                    }
                }

                if all_ok {
                    ok_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    fail_count.fetch_add(1, Ordering::Relaxed);
                }

                let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == prev_count {
                    let ok = ok_count.load(Ordering::Relaxed);
                    let fail = fail_count.load(Ordering::Relaxed);
                    eprint!(
                        "\r  [GFP iter {}] {}/{} checked  ok={}  fail={}  time={:.1}s",
                        gfp_iter,
                        done,
                        prev_count,
                        ok,
                        fail,
                        iter_start.elapsed().as_secs_f64(),
                    );
                    let _ = std::io::stderr().flush();
                }

                Some((*hash, all_ok))
            })
            .collect();

        eprintln!();

        let mut removed = 0usize;
        for (hash, passed) in &results {
            if !passed {
                live_hashes.remove(hash);
                removed += 1;
            }
        }

        let iter_time = iter_start.elapsed().as_secs_f64();
        println!(
            "  GFP iter {}: {} - {} = {} ({:.1}s)",
            gfp_iter,
            prev_count,
            removed,
            live_hashes.len(),
            iter_time
        );

        if removed == 0 {
            println!("  GFP converged!");
            break;
        }
        if live_hashes.is_empty() {
            println!("  GFP converged to EMPTY.");
            break;
        }
    }

    // -----------------------------------------------------------------------
    // Final result
    // -----------------------------------------------------------------------
    let total_time = overall_start.elapsed().as_secs_f64();

    if !live_hashes.is_empty() {
        let empty_hash = board_hash(&TetrisBoard::EMPTY_BOARD);
        let live_boards: Vec<&TetrisBoard> = boards
            .iter()
            .filter(|b| live_hashes.contains(&board_hash(b)))
            .collect();

        println!();
        println!("=============================================");
        println!("  SAFE SET IS NON-EMPTY: {} boards!", live_hashes.len());
        println!(
            "  EMPTY BOARD IN SET: {}",
            live_hashes.contains(&empty_hash)
        );
        println!("  INFINITE PLAY PROVEN.");
        println!("=============================================");

        println!();
        println!("--- surviving boards ---");
        let mut by_cells: FxHashMap<u32, Vec<&&TetrisBoard>> = FxHashMap::default();
        for board in &live_boards {
            by_cells.entry(board.count()).or_default().push(board);
        }
        let mut cell_counts: Vec<u32> = by_cells.keys().copied().collect();
        cell_counts.sort();
        for cells in &cell_counts {
            let group = &by_cells[cells];
            println!("  cells={}: {} boards", cells, group.len());
            for board in group.iter().take(3) {
                print_board(board);
                println!();
            }
        }
    } else {
        println!();
        println!("Safe set converged to empty after LFP + GFP.");
        println!("No cycle exists under these constraints.");

        // Print cell count distribution of boards discovered
        let mut by_cells: FxHashMap<u32, usize> = FxHashMap::default();
        for board in &boards {
            *by_cells.entry(board.count()).or_insert(0) += 1;
        }
        let mut cell_counts: Vec<(u32, usize)> = by_cells.into_iter().collect();
        cell_counts.sort();
        println!();
        println!("--- boards discovered by cell count ---");
        for (cells, count) in &cell_counts {
            println!("  cells={}: {} boards", cells, count);
        }
    }

    println!();
    println!("total_time       = {:.1}s", total_time);

    Ok(())
}
