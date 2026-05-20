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
pub struct Gfp2Args {
    /// Maximum board height for inner safe set (tier 1).
    #[arg(long, default_value_t = 2)]
    inner_max_height: u32,

    /// Maximum holes for inner safe set (tier 1).
    #[arg(long, default_value_t = 0)]
    inner_max_holes: u32,

    /// Maximum board height for outer buffer set (tier 2).
    #[arg(long, default_value_t = 4)]
    outer_max_height: u32,

    /// Maximum holes for outer buffer set (tier 2).
    #[arg(long, default_value_t = 2)]
    outer_max_holes: u32,

    /// Maximum board height during intermediate placements within a bag.
    #[arg(long, default_value_t = 6)]
    inter_max_height: u32,

    /// Maximum holes during intermediate placements within a bag.
    #[arg(long, default_value_t = 4)]
    inter_max_holes: u32,

    /// Safety cap on board discovery per tier.
    #[arg(long, default_value_t = 500_000)]
    max_boards: usize,

    /// Path to SQLite database for results.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,
}

/// Discover all boards reachable from the empty board via BFS,
/// where each step places ONE piece (any of 7 types × all placements).
/// Only boards meeting the admissibility are kept.
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

/// Check if a board meets inner tier admissibility.
fn is_inner(board: &TetrisBoard, inner_h: u32, inner_holes: u32) -> bool {
    if board.height() > inner_h {
        return false;
    }
    if inner_holes != u32::MAX && has_too_many_holes(board, inner_holes) {
        return false;
    }
    true
}

pub fn run(args: Gfp2Args) -> Result<()> {
    println!("tetris_bag_checkpoint gfp2 (two-tier non-adversarial)");
    println!(
        "inner (tier 1)   = height<={} holes<={}",
        args.inner_max_height, args.inner_max_holes
    );
    println!(
        "outer (tier 2)   = height<={} holes<={}",
        args.outer_max_height, args.outer_max_holes
    );
    println!(
        "intermediate     = height<={} holes<={}",
        args.inter_max_height, args.inter_max_holes
    );
    println!("max_boards       = {}", args.max_boards);
    println!();

    // Phase 1: Discover boards — two separate BFS passes to avoid bias
    let discover_start = Instant::now();

    // Pass 1: discover inner boards (tight constraints, full exploration)
    eprintln!(
        "[phase 1a] discovering inner boards (h≤{}, holes≤{})...",
        args.inner_max_height, args.inner_max_holes
    );
    let inner_boards =
        discover_boards(args.inner_max_height, args.inner_max_holes, args.max_boards);

    // Pass 2: discover outer boards (looser constraints)
    eprintln!(
        "[phase 1b] discovering outer boards (h≤{}, holes≤{})...",
        args.outer_max_height, args.outer_max_holes
    );
    let outer_candidates =
        discover_boards(args.outer_max_height, args.outer_max_holes, args.max_boards);

    // Merge: inner boards + outer-only boards (not already in inner)
    let inner_set: FxHashSet<TetrisBoard> = inner_boards.iter().copied().collect();
    let mut all_boards: Vec<TetrisBoard> = inner_boards.clone();
    for board in &outer_candidates {
        if !inner_set.contains(board) {
            all_boards.push(*board);
        }
    }

    let discover_time = discover_start.elapsed().as_secs_f64();

    if all_boards.is_empty() {
        bail!("No boards discovered");
    }

    let all_hashes: Vec<u64> = all_boards.iter().map(|b| board_hash(b)).collect();

    let mut inner_hashes: FxHashSet<u64> = FxHashSet::default();
    let mut outer_hashes: FxHashSet<u64> = FxHashSet::default();

    for (board, hash) in all_boards.iter().zip(all_hashes.iter()) {
        if is_inner(board, args.inner_max_height, args.inner_max_holes) {
            inner_hashes.insert(*hash);
        } else {
            outer_hashes.insert(*hash);
        }
    }

    println!("--- phase 1: discovery ---");
    println!("total boards     = {}", all_boards.len());
    println!("inner (tier 1)   = {}", inner_hashes.len());
    println!("outer (tier 2)   = {}", outer_hashes.len());
    println!("discover_time    = {:.3}s", discover_time);
    println!();

    let perms = generate_permutations();
    let num_perms = perms.len();

    // Phase 2: Two-tier GFP iteration
    //
    // Invariant maintained by the two-tier system:
    //   - Inner boards (S) can transition to S ∪ X (any tier) in one bag
    //   - Outer boards (X) MUST transition back to S (inner) in one bag
    //
    // This gives a 2-bag guarantee: after 2 bags, you're always back in S.
    // Boards can temporarily "rise" to the outer tier but must recover.
    println!("=== Phase 2: Two-tier GFP iteration ===");
    println!(
        "inner={} outer={} perms={}",
        inner_hashes.len(),
        outer_hashes.len(),
        num_perms
    );
    println!();

    let overall_gfp_start = Instant::now();
    let mut iteration: u32 = 0;
    let mut total_inner_removed: usize = 0;
    let mut total_outer_removed: usize = 0;

    loop {
        iteration += 1;
        let iter_start = Instant::now();

        let current_inner = inner_hashes.clone();
        let current_outer = outer_hashes.clone();
        let current_inner_count = current_inner.len();
        let current_outer_count = current_outer.len();

        // Combined target for inner boards: S ∪ X
        let mut combined_target: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(
            current_inner.len() + current_outer.len(),
            Default::default(),
        );
        for h in &current_inner {
            combined_target.insert(*h);
        }
        for h in &current_outer {
            combined_target.insert(*h);
        }

        // --- Check inner boards: must reach S ∪ X ---
        let checked_inner = AtomicUsize::new(0);
        let ok_inner = AtomicUsize::new(0);
        let fail_inner = AtomicUsize::new(0);

        let inner_results: Vec<(u64, bool)> = all_boards
            .par_iter()
            .zip(all_hashes.par_iter())
            .filter_map(|(board, hash)| {
                if !current_inner.contains(hash) {
                    return None;
                }

                let mut all_ok = true;
                for perm in &perms {
                    let mut nodes = 0u64;
                    let reached = can_reach_any(
                        *board,
                        perm,
                        0,
                        args.inter_max_height,
                        args.inter_max_holes,
                        &combined_target,
                        &mut nodes,
                    );
                    if !reached {
                        all_ok = false;
                        break;
                    }
                }

                if all_ok {
                    ok_inner.fetch_add(1, Ordering::Relaxed);
                } else {
                    fail_inner.fetch_add(1, Ordering::Relaxed);
                }

                let done = checked_inner.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 500 == 0 || done == current_inner_count {
                    let ok = ok_inner.load(Ordering::Relaxed);
                    let fail = fail_inner.load(Ordering::Relaxed);
                    eprint!(
                        "\r[iter {} inner] {}/{} ok={} fail={} time={:.1}s",
                        iteration,
                        done,
                        current_inner_count,
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

        // --- Check outer boards: must reach S (inner only) ---
        let checked_outer = AtomicUsize::new(0);
        let ok_outer = AtomicUsize::new(0);
        let fail_outer = AtomicUsize::new(0);
        let outer_start = Instant::now();

        let outer_results: Vec<(u64, bool)> = all_boards
            .par_iter()
            .zip(all_hashes.par_iter())
            .filter_map(|(board, hash)| {
                if !current_outer.contains(hash) {
                    return None;
                }

                let mut all_ok = true;
                for perm in &perms {
                    let mut nodes = 0u64;
                    let reached = can_reach_any(
                        *board,
                        perm,
                        0,
                        args.inter_max_height,
                        args.inter_max_holes,
                        &current_inner, // outer must reach INNER
                        &mut nodes,
                    );
                    if !reached {
                        all_ok = false;
                        break;
                    }
                }

                if all_ok {
                    ok_outer.fetch_add(1, Ordering::Relaxed);
                } else {
                    fail_outer.fetch_add(1, Ordering::Relaxed);
                }

                let done = checked_outer.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 500 == 0 || done == current_outer_count {
                    let ok = ok_outer.load(Ordering::Relaxed);
                    let fail = fail_outer.load(Ordering::Relaxed);
                    eprint!(
                        "\r[iter {} outer] {}/{} ok={} fail={} time={:.1}s",
                        iteration,
                        done,
                        current_outer_count,
                        ok,
                        fail,
                        outer_start.elapsed().as_secs_f64(),
                    );
                    let _ = std::io::stderr().flush();
                }

                Some((*hash, all_ok))
            })
            .collect();

        eprintln!();

        // Apply removals
        let mut inner_removed = 0usize;
        for (hash, passed) in &inner_results {
            if !passed {
                inner_hashes.remove(hash);
                inner_removed += 1;
            }
        }

        let mut outer_removed = 0usize;
        for (hash, passed) in &outer_results {
            if !passed {
                outer_hashes.remove(hash);
                outer_removed += 1;
            }
        }

        total_inner_removed += inner_removed;
        total_outer_removed += outer_removed;

        let iter_time = iter_start.elapsed().as_secs_f64();
        println!(
            "iteration {}  | inner: {}-{}={} | outer: {}-{}={} | time={:.1}s",
            iteration,
            inner_results.len(),
            inner_removed,
            inner_hashes.len(),
            outer_results.len(),
            outer_removed,
            outer_hashes.len(),
            iter_time,
        );

        if inner_removed == 0 && outer_removed == 0 {
            println!("Two-tier GFP converged — no boards removed.");
            break;
        }

        if inner_hashes.is_empty() {
            println!("Inner safe set is EMPTY.");
            break;
        }
    }

    let gfp_time = overall_gfp_start.elapsed().as_secs_f64();

    let empty_hash = board_hash(&TetrisBoard::EMPTY_BOARD);
    let empty_in_inner = inner_hashes.contains(&empty_hash);

    println!();
    println!("--- Two-tier GFP result ---");
    println!("iterations       = {}", iteration);
    println!("inner_safe_set   = {}", inner_hashes.len());
    println!("outer_buffer     = {}", outer_hashes.len());
    println!("inner_removed    = {}", total_inner_removed);
    println!("outer_removed    = {}", total_outer_removed);
    println!("empty_in_inner   = {}", empty_in_inner);
    println!("gfp_time         = {:.3}s", gfp_time);

    if !inner_hashes.is_empty() {
        println!();
        println!("=============================================");
        println!(
            "  INNER SAFE SET IS NON-EMPTY ({} boards)",
            inner_hashes.len()
        );
        println!("  OUTER BUFFER: {} boards", outer_hashes.len());
        if empty_in_inner {
            println!("  The empty board is in the inner safe set.");
        }
        println!("  TWO-BAG INVARIANT PROVEN:");
        println!("  Every inner board can reach inner∪outer in 1 bag.");
        println!("  Every outer board can reach inner in 1 bag.");
        println!("  → After at most 2 bags, always back to inner.");
        println!("  → INFINITE PLAY IS PROVEN.");
        println!("=============================================");

        // Print surviving inner boards
        println!();
        println!("--- surviving inner boards ---");
        let mut survivors: Vec<_> = all_boards
            .iter()
            .zip(all_hashes.iter())
            .filter(|(_, h)| inner_hashes.contains(h))
            .collect();
        survivors.sort_by_key(|(b, _)| (b.count(), b.height()));

        for (i, (board, hash)) in survivors.iter().enumerate().take(30) {
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
        if survivors.len() > 30 {
            println!("  ... ({} more inner boards)", survivors.len() - 30);
        }
    } else {
        println!();
        println!("No surviving boards. The two-tier GFP");
        println!("converged to empty under these constraints.");
    }

    Ok(())
}
