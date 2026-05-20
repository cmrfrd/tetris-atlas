use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Args;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tetris_game::{IsLost, TetrisBoard, TetrisPiecePlacement};

use crate::common::*;

#[derive(Args)]
pub struct GfpArgs {
    /// Maximum board height for safe set membership.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes for safe set membership.
    #[arg(long, default_value_t = 0)]
    max_holes: u32,

    /// Maximum board height during intermediate placements within a bag.
    #[arg(long, default_value_t = 6)]
    inter_max_height: u32,

    /// Maximum holes during intermediate placements within a bag.
    #[arg(long, default_value_t = 4)]
    inter_max_holes: u32,

    /// Safety cap on board discovery.
    #[arg(long, default_value_t = 500_000)]
    max_boards: usize,

    /// Path to SQLite database for results.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,
}

/// Discover all boards reachable from the empty board via BFS,
/// where each step places ONE piece (any of 7 types × all placements).
/// Only boards meeting the safe-set admissibility are kept.
fn discover_boards(max_height: u32, max_holes: u32, max_boards: usize) -> Vec<TetrisBoard> {
    let start = Instant::now();

    let mut boards: Vec<TetrisBoard> = Vec::new();
    let mut board_set: FxHashSet<TetrisBoard> = FxHashSet::default();

    let empty = TetrisBoard::EMPTY_BOARD;
    boards.push(empty);
    board_set.insert(empty);

    let mut frontier_idx: usize = 0;
    let mut hit_cap = false;
    let mut last_report = Instant::now();

    while frontier_idx < boards.len() {
        if boards.len() >= max_boards {
            hit_cap = true;
            break;
        }

        let board = boards[frontier_idx];
        frontier_idx += 1;

        for piece in tetris_game::TetrisPiece::all() {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = board;
                let result = next.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }
                if next.height() > max_height {
                    continue;
                }
                if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
                    continue;
                }
                if board_set.contains(&next) {
                    continue;
                }
                if boards.len() >= max_boards {
                    hit_cap = true;
                    break;
                }
                board_set.insert(next);
                boards.push(next);
            }
            if hit_cap {
                break;
            }
        }

        if last_report.elapsed().as_secs() >= 2 {
            eprintln!(
                "[discover] {:.1}s | boards={} frontier={}",
                start.elapsed().as_secs_f64(),
                boards.len(),
                boards.len() - frontier_idx,
            );
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    if hit_cap {
        eprintln!(
            "[discover] hit cap ({}) with frontier={} remaining",
            max_boards,
            boards.len().saturating_sub(frontier_idx),
        );
    }
    eprintln!(
        "[discover] done in {:.2}s | boards={}",
        elapsed,
        boards.len()
    );

    boards
}

pub fn run(args: GfpArgs) -> Result<()> {
    println!("tetris_bag_checkpoint gfp (non-adversarial)");
    println!(
        "safe_set         = height<={} holes<={}",
        args.max_height, args.max_holes
    );
    println!(
        "intermediate     = height<={} holes<={}",
        args.inter_max_height, args.inter_max_holes
    );
    println!("max_boards       = {}", args.max_boards);
    println!();

    // Phase 1: Board Discovery
    let discover_start = Instant::now();
    let boards = discover_boards(args.max_height, args.max_holes, args.max_boards);
    let discover_time = discover_start.elapsed().as_secs_f64();

    if boards.is_empty() {
        bail!("No boards discovered");
    }

    println!("--- phase 1: discovery ---");
    println!("boards           = {}", boards.len());
    println!("discover_time    = {:.3}s", discover_time);
    println!();

    // Build hash set for each board
    let board_hashes: Vec<u64> = boards.iter().map(|b| board_hash(b)).collect();

    // Initial safe set = all discovered boards
    let mut safe_hashes: FxHashSet<u64> = board_hashes.iter().copied().collect();

    let perms = generate_permutations();
    let num_perms = perms.len();

    // Phase 2: GFP Iteration
    println!("=== Phase 2: GFP iteration ===");
    println!(
        "Checking {} boards × {} perms per iteration",
        boards.len(),
        num_perms
    );
    println!();

    let overall_gfp_start = Instant::now();
    let mut iteration: u32 = 0;
    let mut total_removed: usize = 0;

    loop {
        iteration += 1;
        let iter_start = Instant::now();

        let current_safe: FxHashSet<u64> = safe_hashes.clone();
        let current_safe_count = current_safe.len();

        let checked = AtomicUsize::new(0);
        let ok_count = AtomicUsize::new(0);
        let fail_count = AtomicUsize::new(0);
        let total_nodes = AtomicUsize::new(0);

        // Check each board in the current safe set
        let results: Vec<(u64, bool)> = boards
            .par_iter()
            .zip(board_hashes.par_iter())
            .filter_map(|(board, hash)| {
                if !current_safe.contains(hash) {
                    return None;
                }

                let mut all_ok = true;
                let mut board_nodes = 0u64;

                for perm in &perms {
                    let mut nodes = 0u64;
                    let reached = can_reach_any(
                        *board,
                        perm,
                        0,
                        args.inter_max_height,
                        args.inter_max_holes,
                        &current_safe,
                        &mut nodes,
                    );
                    board_nodes += nodes;

                    if !reached {
                        all_ok = false;
                        break;
                    }
                }

                total_nodes.fetch_add(board_nodes as usize, Ordering::Relaxed);

                if all_ok {
                    ok_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    fail_count.fetch_add(1, Ordering::Relaxed);
                }

                let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 500 == 0 || done == current_safe_count {
                    let ok = ok_count.load(Ordering::Relaxed);
                    let fail = fail_count.load(Ordering::Relaxed);
                    let nodes = total_nodes.load(Ordering::Relaxed);
                    eprint!(
                        "\r[gfp iter {}] {}/{} checked  ok={}  fail={}  nodes={:.1}M  time={:.1}s",
                        iteration,
                        done,
                        current_safe_count,
                        ok,
                        fail,
                        nodes as f64 / 1e6,
                        iter_start.elapsed().as_secs_f64(),
                    );
                    let _ = std::io::stderr().flush();
                }

                Some((*hash, all_ok))
            })
            .collect();

        eprintln!(); // newline after progress

        // Remove failing boards
        let mut removed = 0usize;
        for (hash, passed) in &results {
            if !passed {
                safe_hashes.remove(hash);
                removed += 1;
            }
        }
        total_removed += removed;

        let iter_time = iter_start.elapsed().as_secs_f64();
        println!(
            "iteration {}  | checked={} ok={} removed={} remaining={} time={:.1}s",
            iteration,
            results.len(),
            results.len() - removed,
            removed,
            safe_hashes.len(),
            iter_time,
        );

        if removed == 0 {
            println!("GFP converged — no boards removed.");
            break;
        }

        if safe_hashes.is_empty() {
            println!("Safe set is EMPTY — no surviving boards.");
            break;
        }
    }

    let gfp_time = overall_gfp_start.elapsed().as_secs_f64();

    // Check if empty board survived
    let empty_hash = board_hash(&TetrisBoard::EMPTY_BOARD);
    let empty_in_safe = safe_hashes.contains(&empty_hash);

    println!();
    println!("--- GFP result ---");
    println!("iterations       = {}", iteration);
    println!("safe_set         = {}", safe_hashes.len());
    println!("total_removed    = {}", total_removed);
    println!("empty_in_safe    = {}", empty_in_safe);
    println!("gfp_time         = {:.3}s", gfp_time);

    if !safe_hashes.is_empty() {
        println!();
        println!("=============================================");
        println!("  SAFE SET IS NON-EMPTY ({} boards)", safe_hashes.len());
        if empty_in_safe {
            println!("  The empty board is in the safe set.");
            println!("  INFINITE PLAY IS PROVEN:");
            println!("  For every bag permutation, the player can");
            println!("  always reach another safe board.");
        } else {
            println!("  The empty board is NOT in the safe set.");
            println!("  Infinite play proven for boards in the set,");
            println!("  but reachability from empty not established.");
        }
        println!("=============================================");

        // Print surviving boards
        println!();
        println!("--- surviving boards ---");
        let mut survivors: Vec<_> = boards
            .iter()
            .zip(board_hashes.iter())
            .filter(|(_, h)| safe_hashes.contains(h))
            .collect();
        survivors.sort_by_key(|(b, _)| (b.count(), b.height()));

        for (i, (board, hash)) in survivors.iter().enumerate().take(50) {
            println!(
                "[{:>3}] hash={:<20} h={} c={} holes={} rough={}",
                i,
                **hash as i64,
                board.height(),
                board.count(),
                board.total_holes(),
                board.roughness(),
            );
            print_board(board);
            println!();
        }
        if survivors.len() > 50 {
            println!("  ... ({} more boards)", survivors.len() - 50);
        }
    } else {
        println!();
        println!("No surviving boards. The non-adversarial GFP");
        println!("converged to empty under these constraints.");
    }

    Ok(())
}
