use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Args;
use dashmap::DashMap;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use tetris_game::TetrisBoard;

use crate::common::*;

// ---------------------------------------------------------------------------
// Bitset for tracking 5040 permutations
// ---------------------------------------------------------------------------

const BITSET_WORDS: usize = (5040 + 63) / 64;
type PermBitset = [u64; BITSET_WORDS];

fn bitset_set(bits: &mut PermBitset, idx: usize) {
    bits[idx / 64] |= 1u64 << (idx % 64);
}

fn bitset_count(bits: &PermBitset) -> u32 {
    bits.iter().map(|w| w.count_ones()).sum()
}

fn bitset_and_not_count(a: &PermBitset, mask: &PermBitset) -> u32 {
    a.iter()
        .zip(mask.iter())
        .map(|(a, m)| (a & !m).count_ones())
        .sum()
}

fn bitset_or_assign(dst: &mut PermBitset, src: &PermBitset) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= s;
    }
}

fn bitset_full(num: usize) -> PermBitset {
    let mut bits = [0u64; BITSET_WORDS];
    for i in 0..num {
        bitset_set(&mut bits, i);
    }
    bits
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args)]
pub struct AnalyzeArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_holes: u32,

    /// Path to SQLite database (for loading a starting board).
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Board hash to start from (loaded from DB). If omitted, starts from empty board.
    #[arg(long)]
    start_hash: Option<i64>,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: AnalyzeArgs) -> Result<()> {
    let num_perms = 5040; // 7!
    let overall_start = Instant::now();

    // Load starting board
    let start_board = if let Some(hash) = args.start_hash {
        let conn = Connection::open(&args.db)?;
        let blob: Vec<u8> = conn
            .query_row(
                "SELECT board_blob FROM boards WHERE board_hash = ?1",
                params![hash],
                |row| row.get(0),
            )
            .or_else(|_| {
                conn.query_row(
                    "SELECT board_blob FROM chain_checkpoints WHERE board_hash = ?1",
                    params![hash],
                    |row| row.get(0),
                )
            })
            .or_else(|_| {
                conn.query_row(
                    "SELECT board_blob FROM analyze_covers WHERE board_hash = ?1",
                    params![hash],
                    |row| row.get(0),
                )
            })?;
        let arr: [u8; 200] = blob.as_slice().try_into().unwrap_or([0u8; 200]);
        TetrisBoard::from_binary_slice(arr)
    } else {
        TetrisBoard::new()
    };

    println!("tetris_bag_checkpoint analyze");
    println!("max_height = {}", args.max_height);
    println!("max_holes  = {}", args.max_holes);
    println!("perms      = {}", num_perms);
    println!(
        "start      = h={} c={} holes={}",
        start_board.height(),
        start_board.count(),
        start_board.total_holes()
    );
    if start_board.count() > 0 {
        print_board(&start_board);
    }
    println!();

    // -----------------------------------------------------------------------
    // Phase 1: Parallel BFS — each permutation uses level-by-level BFS with
    // deduplication at each step, preventing exponential DFS blowup.
    // -----------------------------------------------------------------------
    println!("=== Phase 1: counting boards by cell count ===");

    let perms = generate_permutations();

    // board_hash -> (cell_count, perm_count)
    let board_info: DashMap<u64, (u32, u32)> = DashMap::new();
    let perms_done = AtomicUsize::new(0);
    let zero_reach = AtomicUsize::new(0);

    perms.par_iter().for_each(|perm| {
        let mut boards: Vec<(u64, TetrisBoard)> = Vec::new();
        let mut nodes = 0u64;
        collect_reachable_boards(
            start_board,
            perm,
            0,
            args.max_height,
            args.max_holes,
            &mut boards,
            &mut nodes,
        );

        if boards.is_empty() {
            zero_reach.fetch_add(1, Ordering::Relaxed);
        }

        let mut seen = FxHashSet::default();
        for (hash, board) in &boards {
            if seen.insert(*hash) {
                board_info
                    .entry(*hash)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((board.count(), 1));
            }
        }

        let done = perms_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 100 == 0 || done == num_perms {
            println!(
                "[{:>5}/{}] unique_boards={} time={:.1}s",
                done,
                num_perms,
                board_info.len(),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_p1 = overall_start.elapsed().as_secs_f64();
    let zr = zero_reach.load(Ordering::Relaxed);
    println!();
    println!("--- phase 1 done ---");
    println!("time         = {elapsed_p1:.3}s");
    println!("total_boards = {}", board_info.len());
    println!("zero_reach   = {zr}");
    if zr > 0 {
        println!("WARNING: {zr} perms have NO reachable boards within constraints!");
    }

    // Group by cell count: cell_count -> list of (hash, perm_count)
    let mut by_cells: std::collections::BTreeMap<u32, Vec<(u64, u32)>> =
        std::collections::BTreeMap::new();
    for entry in board_info.iter() {
        let (hash, (cells, count)) = (*entry.key(), *entry.value());
        by_cells.entry(cells).or_default().push((hash, count));
    }
    drop(board_info);

    // Sort each group by perm_count descending
    for (_, boards) in by_cells.iter_mut() {
        boards.sort_by(|a, b| b.1.cmp(&a.1));
    }

    println!();
    println!("=== Cell count distribution ===");
    println!(
        "{:<8} {:<10} {:<12} {:<12}",
        "cells", "boards", "max_perms", "sum_perms"
    );
    for (cells, boards) in &by_cells {
        let max_perms = boards.first().map(|b| b.1).unwrap_or(0);
        let sum_perms: u64 = boards.iter().map(|b| b.1 as u64).sum();
        println!(
            "{:<8} {:<10} {:<12} {:<12}",
            cells,
            boards.len(),
            max_perms,
            sum_perms
        );
    }

    // -----------------------------------------------------------------------
    // Phase 2: For each cell count, build PermBitsets and check coverage
    // -----------------------------------------------------------------------
    println!();
    println!("=== Phase 2: per-cell-count coverage analysis ===");

    let all_mask = bitset_full(num_perms);
    let mut cover_boards_to_save: Vec<(u64, TetrisBoard, u32, u32)> = Vec::new();

    for (cells, board_list) in &by_cells {
        let target_hashes: FxHashSet<u64> = board_list.iter().map(|(h, _)| *h).collect();
        let num_targets = target_hashes.len();

        println!();
        println!("--- cells={} ({} boards) ---", cells, num_targets);

        let board_bitsets: DashMap<u64, PermBitset> = DashMap::new();
        let board_examples: DashMap<u64, TetrisBoard> = DashMap::new();
        let p2_done = AtomicUsize::new(0);

        perms.par_iter().enumerate().for_each(|(perm_idx, perm)| {
            let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
            let mut nodes = 0u64;
            collect_reachable_filtered(
                start_board,
                perm,
                0,
                args.max_height,
                args.max_holes,
                &target_hashes,
                &mut hits,
                &mut nodes,
            );

            let mut seen = FxHashSet::default();
            for (hash, board) in hits {
                if seen.insert(hash) {
                    board_bitsets
                        .entry(hash)
                        .and_modify(|bits| bitset_set(bits, perm_idx))
                        .or_insert_with(|| {
                            let mut bits = [0u64; BITSET_WORDS];
                            bitset_set(&mut bits, perm_idx);
                            bits
                        });
                    board_examples.entry(hash).or_insert(board);
                }
            }

            let done = p2_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 500 == 0 || done == num_perms {
                println!(
                    "  [{:>5}/{}] verified={}/{} time={:.1}s",
                    done,
                    num_perms,
                    board_bitsets.len(),
                    num_targets,
                    overall_start.elapsed().as_secs_f64(),
                );
                let _ = std::io::stdout().flush();
            }
        });

        // Union of all PermBitsets
        let mut union = [0u64; BITSET_WORDS];
        let entries: Vec<(u64, PermBitset)> = board_bitsets
            .iter()
            .map(|e| (*e.key(), *e.value()))
            .collect();
        for (_, bits) in &entries {
            bitset_or_assign(&mut union, bits);
        }
        let union_count: u32 = union
            .iter()
            .zip(all_mask.iter())
            .map(|(u, m)| (u & m).count_ones())
            .sum();

        let is_complete = union_count == num_perms as u32;
        println!(
            "  union_coverage = {}/{} {}",
            union_count,
            num_perms,
            if is_complete {
                "COMPLETE"
            } else {
                "INCOMPLETE"
            }
        );

        if !is_complete {
            println!(
                "  {} perms CANNOT reach any {}-cell board within constraints",
                num_perms as u32 - union_count,
                cells
            );
            continue;
        }

        // Greedy set cover
        let mut covered = [0u64; BITSET_WORDS];
        let mut cover: Vec<(u64, u32, u32)> = Vec::new(); // (hash, new, total_perms)

        loop {
            let all_done = covered
                .iter()
                .zip(all_mask.iter())
                .all(|(c, m)| (c & m) == *m);
            if all_done {
                break;
            }

            let mut best_new = 0u32;
            let mut best_i = 0;
            for (i, (_, bits)) in entries.iter().enumerate() {
                let nc = bitset_and_not_count(bits, &covered);
                if nc > best_new {
                    best_new = nc;
                    best_i = i;
                }
            }

            if best_new == 0 {
                break;
            }

            let (hash, bits) = &entries[best_i];
            bitset_or_assign(&mut covered, bits);
            let total_perms = bitset_count(bits);
            cover.push((*hash, best_new, total_perms));
        }

        println!("  cover_size = {}", cover.len());

        // Print and store cover boards
        for (i, (hash, new, total_perms)) in cover.iter().enumerate() {
            if let Some(board) = board_examples.get(hash) {
                println!(
                    "    cover[{:>2}]: +{:<5} perms={}/{} h={} holes={} rough={} hash={}",
                    i,
                    new,
                    total_perms,
                    num_perms,
                    board.height(),
                    board.total_holes(),
                    board.roughness(),
                    *hash as i64,
                );
                print_board(&board);
                cover_boards_to_save.push((*hash, *board, *cells, *total_perms));
            }
        }
    }

    // Save cover boards to DB for subsequent chaining
    if !cover_boards_to_save.is_empty() {
        let conn = Connection::open(&args.db)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS analyze_covers (
                 board_hash INTEGER PRIMARY KEY,
                 board_blob BLOB NOT NULL,
                 cell_count INTEGER NOT NULL,
                 perm_count INTEGER NOT NULL
             );",
        )?;
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO analyze_covers (board_hash, board_blob, cell_count, perm_count)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (hash, board, cells, perms) in &cover_boards_to_save {
                stmt.execute(params![
                    *hash as i64,
                    &board.to_binary_slice() as &[u8],
                    *cells as i64,
                    *perms as i64,
                ])?;
            }
        }
        tx.commit()?;
        println!("saved {} cover boards to DB", cover_boards_to_save.len());
    }

    let total_elapsed = overall_start.elapsed().as_secs_f64();
    println!();
    println!("total_time = {total_elapsed:.3}s");

    Ok(())
}
