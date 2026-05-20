use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use dashmap::DashMap;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use tetris_game::{Major, TetrisBoard};

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
pub struct DiscoverArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_holes: u32,

    /// Number of top candidate boards to verify in phase 2.
    #[arg(long, default_value_t = 5000)]
    top_k: usize,

    /// Path to SQLite database (created if missing).
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: DiscoverArgs) -> Result<()> {
    let perms = generate_permutations();
    let num_perms = perms.len();

    println!("tetris_bag_checkpoint discover");
    println!("permutations     = {num_perms}");
    println!("max_height       = {}", args.max_height);
    println!("max_holes        = {}", args.max_holes);
    println!("top_k            = {}", args.top_k);
    println!("db               = {}", args.db.display());
    println!();

    let overall_start = Instant::now();

    // -----------------------------------------------------------------------
    // Phase 1: DFS all perms, accumulate board_hash → count
    // -----------------------------------------------------------------------
    println!("=== Phase 1: counting reachable boards per perm ===");
    let board_counts: DashMap<u64, u32> = DashMap::new();
    let perms_done = AtomicUsize::new(0);

    perms.par_iter().for_each(|perm| {
        let mut hashes = FxHashSet::default();
        let mut nodes = 0u64;
        collect_reachable_hashes(
            TetrisBoard::EMPTY_BOARD,
            perm,
            0,
            args.max_height,
            args.max_holes,
            &mut hashes,
            &mut nodes,
        );

        for h in &hashes {
            board_counts.entry(*h).and_modify(|c| *c += 1).or_insert(1);
        }

        let done = perms_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done == 1 || done % 100 == 0 || done == num_perms {
            println!(
                "[{:>5}/{}] reachable={:<8} nodes={:<12} unique_boards={:<10} time={:.1}s",
                done,
                num_perms,
                hashes.len(),
                nodes,
                board_counts.len(),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_phase1 = overall_start.elapsed().as_secs_f64();
    let unique_boards = board_counts.len();

    println!();
    println!("--- phase 1 done ---");
    println!("phase1_time      = {elapsed_phase1:.3}s");
    println!("unique_boards    = {unique_boards}");

    // Sort by count descending, take top K
    let mut sorted: Vec<(u64, u32)> = board_counts
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let universal_count = sorted
        .iter()
        .filter(|(_, c)| *c == num_perms as u32)
        .count();
    println!("universal_boards = {universal_count} (reachable by all {num_perms} perms)");

    println!();
    println!("--- top 20 boards by perm count ---");
    for (i, (hash, count)) in sorted.iter().take(20).enumerate() {
        let pct = *count as f64 / num_perms as f64 * 100.0;
        println!("  [{i:>2}] hash={hash:<20} count={count}/{num_perms} ({pct:.1}%)");
    }

    // Count distribution
    let mut count_dist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for (_, count) in &sorted {
        let bucket = match *count {
            1 => 1,
            2..=10 => 10,
            11..=100 => 100,
            101..=500 => 500,
            501..=1000 => 1000,
            1001..=2000 => 2000,
            2001..=3000 => 3000,
            3001..=4000 => 4000,
            _ => 5040,
        };
        *count_dist.entry(bucket).or_insert(0) += 1;
    }
    println!();
    println!("--- count distribution ---");
    for (bucket, n) in &count_dist {
        println!("  count<={bucket:<5} boards={n}");
    }

    // -----------------------------------------------------------------------
    // Phase 2: Re-run DFS for top-K boards to get exact PermBitsets + boards
    // -----------------------------------------------------------------------
    let top_k = args.top_k.min(sorted.len());
    let target_hashes: FxHashSet<u64> = sorted.iter().take(top_k).map(|(h, _)| *h).collect();

    println!();
    println!("=== Phase 2: verifying top {} boards ===", top_k);

    let board_bitsets: DashMap<u64, PermBitset> = DashMap::new();
    let board_examples: DashMap<u64, TetrisBoard> = DashMap::new();
    let perms_done2 = AtomicUsize::new(0);

    perms.par_iter().enumerate().for_each(|(perm_idx, perm)| {
        let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
        let mut nodes = 0u64;
        collect_reachable_filtered(
            TetrisBoard::EMPTY_BOARD,
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

        let done = perms_done2.fetch_add(1, Ordering::Relaxed) + 1;
        if done == 1 || done % 100 == 0 || done == num_perms {
            println!(
                "[{:>5}/{}] verified boards={:<8} time={:.1}s",
                done,
                num_perms,
                board_bitsets.len(),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_phase2 = overall_start.elapsed().as_secs_f64();
    println!();
    println!("--- phase 2 done ---");
    println!("phase2_time      = {:.3}s", elapsed_phase2 - elapsed_phase1);
    println!("verified_boards  = {}", board_bitsets.len());

    // -----------------------------------------------------------------------
    // Greedy set cover
    // -----------------------------------------------------------------------
    println!();
    println!("=== Greedy set cover ===");

    let entries: Vec<(u64, PermBitset)> = board_bitsets
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();

    let all_mask = bitset_full(num_perms);
    let mut covered = [0u64; BITSET_WORDS];
    let mut cover: Vec<(u64, PermBitset, u32)> = Vec::new();

    loop {
        if covered == all_mask {
            break;
        }

        let mut best_idx = 0;
        let mut best_new = 0u32;
        for (i, (_, bits)) in entries.iter().enumerate() {
            let new_covered = bitset_and_not_count(bits, &covered);
            if new_covered > best_new {
                best_new = new_covered;
                best_idx = i;
            }
        }

        if best_new == 0 {
            break;
        }

        let (hash, bits) = &entries[best_idx];
        bitset_or_assign(&mut covered, bits);

        let total_covered: u32 = covered
            .iter()
            .zip(all_mask.iter())
            .map(|(c, m)| (c & m).count_ones())
            .sum();

        cover.push((*hash, *bits, bitset_count(bits)));

        let board = board_examples.get(hash).unwrap();
        println!(
            "  cover[{:>3}]: +{:<5} total={}/{}  h={} c={} holes={}",
            cover.len() - 1,
            best_new,
            total_covered,
            num_perms,
            board.height(),
            board.count(),
            board.total_holes(),
        );
        let _ = std::io::stdout().flush();
    }

    let covered_count: u32 = covered
        .iter()
        .zip(all_mask.iter())
        .map(|(c, m)| (c & m).count_ones())
        .sum();

    println!();
    println!("cover_size       = {}", cover.len());
    println!("covered_perms    = {}/{}", covered_count, num_perms);
    if covered_count == num_perms as u32 {
        println!("status           = COMPLETE");
    } else {
        println!(
            "status           = INCOMPLETE ({} uncovered)",
            num_perms as u32 - covered_count
        );
    }

    // Print cover boards
    println!();
    println!("--- cover boards ---");
    for (i, (hash, _, total_hits)) in cover.iter().enumerate() {
        if let Some(board) = board_examples.get(hash) {
            println!(
                "cover[{:>3}]: hits={}/{} height={} cells={} holes={} roughness={}",
                i,
                total_hits,
                num_perms,
                board.height(),
                board.count(),
                board.total_holes(),
                board.roughness(),
            );
            print_board(&board);
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Persist to SQLite
    // -----------------------------------------------------------------------
    println!("Writing to SQLite...");
    if let Some(parent) = args.db.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating db directory {}", parent.display()))?;
    }
    if args.db.exists() {
        std::fs::remove_file(&args.db)?;
    }
    let conn = Connection::open(&args.db)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;

         CREATE TABLE boards (
             board_hash INTEGER PRIMARY KEY,
             board_blob BLOB,
             height INTEGER,
             cells INTEGER,
             holes INTEGER,
             roughness INTEGER,
             perm_count INTEGER NOT NULL
         );

         CREATE TABLE cover_set (
             step INTEGER PRIMARY KEY,
             board_hash INTEGER NOT NULL,
             perm_count INTEGER NOT NULL
         );

         CREATE TABLE run_meta (
             key TEXT PRIMARY KEY,
             value TEXT NOT NULL
         );",
    )?;

    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES ('max_height', ?1)",
        params![args.max_height.to_string()],
    )?;
    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES ('max_holes', ?1)",
        params![args.max_holes.to_string()],
    )?;
    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES ('cover_size', ?1)",
        params![cover.len().to_string()],
    )?;
    conn.execute(
        "INSERT INTO run_meta (key, value) VALUES ('covered_perms', ?1)",
        params![covered_count.to_string()],
    )?;

    // Write top-K boards with their verified perm counts
    {
        let tx = conn.unchecked_transaction()?;
        let mut stmt = tx.prepare(
            "INSERT OR IGNORE INTO boards (board_hash, board_blob, height, cells, holes, roughness, perm_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (hash, bits) in &entries {
            let count = bitset_count(bits) as i64;
            if let Some(board) = board_examples.get(hash) {
                let blob = board.to_cell_array(Major::Row);
                stmt.execute(params![
                    *hash as i64,
                    blob.as_slice(),
                    board.height(),
                    board.count(),
                    board.total_holes(),
                    board.roughness(),
                    count,
                ])?;
            }
        }
        drop(stmt);
        tx.commit()?;
    }

    // Write cover set
    {
        let tx = conn.unchecked_transaction()?;
        let mut stmt =
            tx.prepare("INSERT INTO cover_set (step, board_hash, perm_count) VALUES (?1, ?2, ?3)")?;
        for (i, (hash, _, total_hits)) in cover.iter().enumerate() {
            stmt.execute(params![i as i64, *hash as i64, *total_hits as i64])?;
        }
        drop(stmt);
        tx.commit()?;
    }

    let elapsed_total = overall_start.elapsed().as_secs_f64();
    println!("total_time       = {elapsed_total:.3}s");

    Ok(())
}
