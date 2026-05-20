use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use clap::Args;
use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{IsLost, TetrisBoard, TetrisPiecePlacement};
use tetris_search::scoring::height_mse_distance_from_empty;

use crate::common::*;

#[derive(Args)]
pub struct HubsArgs {
    /// Maximum board height for safe-set boards.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes for safe-set boards.
    #[arg(long, default_value_t = 0)]
    max_holes: u32,

    /// Maximum board height during intermediate placements within a bag.
    #[arg(long, default_value_t = 6)]
    inter_max_height: u32,

    /// Maximum holes during intermediate placements within a bag.
    #[arg(long, default_value_t = 2)]
    inter_max_holes: u32,

    /// Beam width for forward play (per permutation).
    #[arg(long, default_value_t = 64)]
    beam_width: usize,

    /// Number of bag rounds to play forward from the seed boards.
    #[arg(long, default_value_t = 10)]
    rounds: usize,

    /// Maximum number of hub boards to verify.
    #[arg(long, default_value_t = 200)]
    max_hubs: usize,

    /// Minimum permutation hit count to qualify as a hub candidate.
    #[arg(long, default_value_t = 100)]
    min_hits: u32,
}

/// A scored board for beam search.
#[derive(Clone, Copy)]
struct ScoredBoard {
    board: TetrisBoard,
    score: f32,
}

/// Mini beam search: given a starting board and a 7-piece permutation,
/// return the top `width` boards reachable after placing all 7 pieces.
fn beam_play(
    start: TetrisBoard,
    perm: &[tetris_game::TetrisPiece; 7],
    width: usize,
    inter_max_height: u32,
    inter_max_holes: u32,
) -> Vec<TetrisBoard> {
    let mut beam: Vec<ScoredBoard> = vec![ScoredBoard {
        board: start,
        score: -height_mse_distance_from_empty(start),
    }];

    for step in 0..7 {
        let piece = perm[step];
        let mut candidates: Vec<ScoredBoard> = Vec::with_capacity(beam.len() * 34);

        for parent in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = parent.board;
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
                candidates.push(ScoredBoard { board: next, score });
            }
        }

        // Sort by score descending, deduplicate, keep top `width`
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut seen = FxHashSet::default();
        beam.clear();
        for c in candidates {
            if seen.insert(c.board) {
                beam.push(c);
                if beam.len() >= width {
                    break;
                }
            }
        }

        if beam.is_empty() {
            break;
        }
    }

    beam.iter().map(|s| s.board).collect()
}

pub fn run(args: HubsArgs) -> Result<()> {
    println!("tetris_bag_checkpoint hubs (constructive hub discovery + verification)");
    println!(
        "safe_set         = height<={} holes<={}",
        args.max_height, args.max_holes
    );
    println!(
        "intermediate     = height<={} holes<={}",
        args.inter_max_height, args.inter_max_holes
    );
    println!("beam_width       = {}", args.beam_width);
    println!("rounds           = {}", args.rounds);
    println!("max_hubs         = {}", args.max_hubs);
    println!("min_hits         = {}", args.min_hits);
    println!();

    let perms = generate_permutations();
    let num_perms = perms.len();

    // -----------------------------------------------------------------------
    // Phase 1: Forward beam play — discover hub boards
    // -----------------------------------------------------------------------
    println!("=== Phase 1: Forward beam play ===");
    let phase1_start = Instant::now();

    // board → visit count across all (round, perm) combinations
    let mut visit_counts: FxHashMap<TetrisBoard, u32> = FxHashMap::default();

    // Start from the empty board
    let mut frontier: Vec<TetrisBoard> = vec![TetrisBoard::EMPTY_BOARD];
    *visit_counts.entry(TetrisBoard::EMPTY_BOARD).or_insert(0) += 1;

    for round in 0..args.rounds {
        let round_start = Instant::now();
        let mut next_frontier_counts: FxHashMap<TetrisBoard, u32> = FxHashMap::default();

        for start_board in &frontier {
            for perm in &perms {
                let results = beam_play(
                    *start_board,
                    perm,
                    args.beam_width,
                    args.inter_max_height,
                    args.inter_max_holes,
                );

                // Record terminals that meet safe-set admissibility
                for board in &results {
                    if board.height() <= args.max_height
                        && (args.max_holes == u32::MAX
                            || !has_too_many_holes(board, args.max_holes))
                    {
                        *next_frontier_counts.entry(*board).or_insert(0) += 1;
                        *visit_counts.entry(*board).or_insert(0) += 1;
                    }
                }
            }
        }

        // Next frontier: top boards by visit count in this round
        let mut round_boards: Vec<(TetrisBoard, u32)> = next_frontier_counts.into_iter().collect();
        round_boards.sort_by(|a, b| b.1.cmp(&a.1));

        // Keep boards visited by many permutations as frontier for next round
        frontier = round_boards
            .iter()
            .take(args.max_hubs)
            .map(|(b, _)| *b)
            .collect();

        let round_time = round_start.elapsed().as_secs_f64();
        println!(
            "  round {}/{}: frontier={} unique_this_round={} total_unique={} time={:.1}s",
            round + 1,
            args.rounds,
            frontier.len(),
            round_boards.len(),
            visit_counts.len(),
            round_time,
        );

        if frontier.is_empty() {
            println!("  No admissible boards found — aborting.");
            break;
        }
    }

    let phase1_time = phase1_start.elapsed().as_secs_f64();
    println!();
    println!("--- phase 1 done ---");
    println!("total_unique_boards = {}", visit_counts.len());
    println!("phase1_time         = {:.1}s", phase1_time);
    println!();

    // -----------------------------------------------------------------------
    // Phase 2: Select hub candidates
    // -----------------------------------------------------------------------
    println!("=== Phase 2: Select hub candidates ===");

    let mut sorted_boards: Vec<(TetrisBoard, u32)> = visit_counts.into_iter().collect();
    sorted_boards.sort_by(|a, b| b.1.cmp(&a.1));

    println!("--- top 30 boards by visit count ---");
    for (i, (board, count)) in sorted_boards.iter().take(30).enumerate() {
        println!(
            "  [{:>3}] visits={:<8} h={} c={} holes={} rough={}",
            i,
            count,
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness(),
        );
        print_board(board);
        println!();
    }

    // Filter by min_hits and take up to max_hubs
    let hub_boards: Vec<TetrisBoard> = sorted_boards
        .iter()
        .filter(|(_, count)| *count >= args.min_hits)
        .take(args.max_hubs)
        .map(|(board, _)| *board)
        .collect();

    let hub_hashes: FxHashSet<u64> = hub_boards.iter().map(|b| board_hash(b)).collect();

    println!(
        "hub candidates    = {} (min_hits={})",
        hub_boards.len(),
        args.min_hits
    );
    if hub_boards.is_empty() {
        println!("No hub candidates meet the minimum hit threshold.");
        return Ok(());
    }

    // -----------------------------------------------------------------------
    // Phase 3: Targeted verification
    // -----------------------------------------------------------------------
    println!();
    println!("=== Phase 3: Targeted verification ===");
    println!(
        "Checking {} hubs × {} perms = {} checks",
        hub_boards.len(),
        num_perms,
        hub_boards.len() * num_perms,
    );
    let phase3_start = Instant::now();

    let mut total_pass = 0usize;
    let mut total_fail = 0usize;
    let mut surviving_boards: Vec<(TetrisBoard, u32, u32)> = Vec::new(); // (board, pass_count, fail_count)

    for (hub_idx, hub_board) in hub_boards.iter().enumerate() {
        let board_start = Instant::now();
        let mut pass_count = 0u32;
        let mut fail_count = 0u32;

        for perm in &perms {
            let mut nodes = 0u64;
            let reached = can_reach_any(
                *hub_board,
                perm,
                0,
                args.inter_max_height,
                args.inter_max_holes,
                &hub_hashes,
                &mut nodes,
            );
            if reached {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
        }

        let board_time = board_start.elapsed().as_secs_f64();

        if fail_count == 0 {
            total_pass += 1;
            surviving_boards.push((*hub_board, pass_count, fail_count));
        } else {
            total_fail += 1;
        }

        let pct = pass_count as f64 / num_perms as f64 * 100.0;
        if hub_idx < 20 || fail_count == 0 || (hub_idx + 1) % 50 == 0 {
            eprint!(
                "\r  [{:>4}/{}] pass={}/{} ({:.1}%) time={:.2}s | total: pass={} fail={}",
                hub_idx + 1,
                hub_boards.len(),
                pass_count,
                num_perms,
                pct,
                board_time,
                total_pass,
                total_fail,
            );
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();

    let phase3_time = phase3_start.elapsed().as_secs_f64();

    println!();
    println!("--- phase 3 done ---");
    println!("hubs_checked      = {}", hub_boards.len());
    println!("hubs_pass_all     = {}", total_pass);
    println!("hubs_fail_any     = {}", total_fail);
    println!("phase3_time       = {:.1}s", phase3_time);

    if surviving_boards.is_empty() {
        println!();
        println!("No hub boards survive all permutations.");
        println!("This confirms the GFP result: no safe set exists under these constraints.");

        // Show boards with highest pass rate
        println!();
        println!("--- boards with highest pass rate (for diagnostic) ---");
        let mut all_results: Vec<(TetrisBoard, u32)> = Vec::new();
        for hub_board in hub_boards.iter().take(50) {
            let mut pass_count = 0u32;
            for perm in &perms {
                let mut nodes = 0u64;
                if can_reach_any(
                    *hub_board,
                    perm,
                    0,
                    args.inter_max_height,
                    args.inter_max_holes,
                    &hub_hashes,
                    &mut nodes,
                ) {
                    pass_count += 1;
                }
            }
            all_results.push((*hub_board, pass_count));
        }
        all_results.sort_by(|a, b| b.1.cmp(&a.1));
        for (i, (board, pass)) in all_results.iter().take(10).enumerate() {
            let pct = *pass as f64 / num_perms as f64 * 100.0;
            println!(
                "  [{i}] pass={pass}/{num_perms} ({pct:.1}%) h={} c={} holes={}",
                board.height(),
                board.count(),
                board.total_holes(),
            );
            print_board(board);
            println!();
        }
    } else {
        println!();
        println!("=============================================");
        println!(
            "  {} HUB BOARDS SURVIVE ALL {} PERMUTATIONS!",
            surviving_boards.len(),
            num_perms
        );
        println!("=============================================");

        // Now do GFP pruning on just the surviving boards
        let survivor_hashes: FxHashSet<u64> = surviving_boards
            .iter()
            .map(|(b, _, _)| board_hash(b))
            .collect();

        println!();
        println!("--- GFP verification on surviving hubs ---");
        let mut stable_set = survivor_hashes.clone();
        let mut iteration = 0u32;

        loop {
            iteration += 1;
            let prev_count = stable_set.len();
            let mut to_remove: Vec<u64> = Vec::new();

            for (board, _, _) in &surviving_boards {
                let bh = board_hash(board);
                if !stable_set.contains(&bh) {
                    continue;
                }
                for perm in &perms {
                    let mut nodes = 0u64;
                    if !can_reach_any(
                        *board,
                        perm,
                        0,
                        args.inter_max_height,
                        args.inter_max_holes,
                        &stable_set,
                        &mut nodes,
                    ) {
                        to_remove.push(bh);
                        break;
                    }
                }
            }

            for h in &to_remove {
                stable_set.remove(h);
            }

            println!(
                "  GFP iter {}: {} - {} = {}",
                iteration,
                prev_count,
                to_remove.len(),
                stable_set.len()
            );

            if to_remove.is_empty() {
                println!("  GFP converged!");
                break;
            }
            if stable_set.is_empty() {
                println!("  GFP converged to EMPTY.");
                break;
            }
        }

        if !stable_set.is_empty() {
            let empty_hash = board_hash(&TetrisBoard::EMPTY_BOARD);
            println!();
            println!("=============================================");
            println!("  SAFE SET IS NON-EMPTY: {} boards!", stable_set.len());
            println!("  EMPTY BOARD IN SET: {}", stable_set.contains(&empty_hash));
            println!("  INFINITE PLAY PROVEN.");
            println!("=============================================");
        }
    }

    Ok(())
}
