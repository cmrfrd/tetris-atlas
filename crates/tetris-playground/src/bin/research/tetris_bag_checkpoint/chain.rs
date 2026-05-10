use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, bail};
use clap::Args;
use dashmap::DashMap;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use tetris_game::TetrisBoard;

use crate::common::*;

// ---------------------------------------------------------------------------
// Bitset for tracking 5040 permutations (per source board)
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
pub struct ChainArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_holes: u32,

    /// Number of top candidate boards to verify in phase 2.
    #[arg(long, default_value_t = 5000)]
    top_k: usize,

    /// Path to SQLite database from `discover` command.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: ChainArgs) -> Result<()> {
    if !args.db.exists() {
        bail!(
            "Database not found at {}. Run `discover` first.",
            args.db.display()
        );
    }

    let conn = Connection::open(&args.db)?;

    // Ensure chain_checkpoints table exists
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chain_checkpoints (
             chain_step INTEGER NOT NULL,
             board_hash INTEGER NOT NULL,
             board_blob BLOB NOT NULL,
             height INTEGER,
             cells INTEGER,
             holes INTEGER,
             roughness INTEGER,
             PRIMARY KEY (chain_step, board_hash)
         );",
    )?;

    // Determine the current chain progress
    let max_step: Option<i64> = conn
        .query_row("SELECT MAX(chain_step) FROM chain_checkpoints", [], |row| {
            row.get(0)
        })
        .unwrap_or(None);

    let (current_step, source_boards) = if let Some(step) = max_step {
        // Load checkpoints from the latest step
        let mut stmt = conn.prepare(
            "SELECT board_hash, board_blob FROM chain_checkpoints WHERE chain_step = ?1",
        )?;
        let boards: Vec<(u64, TetrisBoard)> = stmt
            .query_map(params![step], |row| {
                let hash: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((hash as u64, blob))
            })?
            .filter_map(|r| r.ok())
            .map(|(hash, blob)| {
                let arr: [u8; 200] = blob.as_slice().try_into().unwrap_or([0u8; 200]);
                (hash, TetrisBoard::from_binary_slice(arr))
            })
            .collect();
        ((step + 1) as usize, boards)
    } else {
        // Import universal checkpoints from discover's boards table as step 0
        let mut stmt =
            conn.prepare("SELECT board_hash, board_blob FROM boards WHERE perm_count = 5040")?;
        let boards: Vec<(u64, TetrisBoard)> = stmt
            .query_map([], |row| {
                let hash: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((hash as u64, blob))
            })?
            .filter_map(|r| r.ok())
            .map(|(hash, blob)| {
                let arr: [u8; 200] = blob.as_slice().try_into().unwrap_or([0u8; 200]);
                (hash, TetrisBoard::from_binary_slice(arr))
            })
            .collect();

        if boards.is_empty() {
            bail!("No universal checkpoints in DB. Run `discover` first.");
        }

        // Save as step 0
        let tx = conn.unchecked_transaction()?;
        {
            let mut ins = tx.prepare(
                "INSERT OR IGNORE INTO chain_checkpoints
                 (chain_step, board_hash, board_blob, height, cells, holes, roughness)
                 VALUES (0, ?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (hash, board) in &boards {
                let blob = board.to_binary_slice();
                ins.execute(params![
                    *hash as i64,
                    blob.as_slice(),
                    board.height(),
                    board.count(),
                    board.total_holes(),
                    board.roughness(),
                ])?;
            }
        }
        tx.commit()?;

        (1usize, boards)
    };
    drop(conn);

    // Residue classes: each bag shifts cells by +8 mod 10
    // step 0 boards have cells ≡ 8 mod 10, step 1 ≡ 6, step 2 ≡ 4, step 3 ≡ 2, step 4 ≡ 0
    let residue_classes = [8, 6, 4, 2, 0];
    let source_residue = residue_classes[(current_step - 1) % 5];
    let target_residue = residue_classes[current_step % 5];

    if current_step > 5 {
        println!("Chain already has 5 steps (full residue cycle). Nothing to do.");
        println!("Run `cycle` to verify the full cycle closes.");
        return Ok(());
    }

    println!("tetris_bag_checkpoint chain (step {})", current_step);
    println!(
        "source_step      = {} (cells mod 10 = {})",
        current_step - 1,
        source_residue
    );
    println!(
        "target_step      = {} (cells mod 10 = {})",
        current_step, target_residue
    );
    println!("source_boards    = {}", source_boards.len());
    println!("max_height       = {}", args.max_height);
    println!("max_holes        = {}", args.max_holes);
    println!("top_k            = {}", args.top_k);
    println!();

    // Print source board summary
    println!("--- source boards ---");
    for (i, (hash, board)) in source_boards.iter().enumerate().take(10) {
        println!(
            "  src[{:>2}] hash={:<20} h={} c={} holes={} rough={}",
            i,
            hash,
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness(),
        );
    }
    if source_boards.len() > 10 {
        println!("  ... and {} more", source_boards.len() - 10);
    }
    println!();

    let perms = generate_permutations();
    let num_perms = perms.len();
    let num_sources = source_boards.len();
    let total_pairs = num_sources * num_perms;

    let overall_start = Instant::now();

    // -----------------------------------------------------------------------
    // Phase 1: For each (source, perm) pair, DFS and count target board hits
    // -----------------------------------------------------------------------
    println!("=== Phase 1: counting reachable boards ===");
    println!(
        "total_pairs = {} ({} sources x {} perms)",
        total_pairs, num_sources, num_perms
    );

    let board_counts: DashMap<u64, u32> = DashMap::new();
    let pairs_done = AtomicUsize::new(0);
    let zero_reach_pairs = AtomicUsize::new(0);

    // Build flat list of (source_idx, perm_idx) pairs for parallel iteration
    let source_perm_pairs: Vec<(usize, usize)> = (0..num_sources)
        .flat_map(|si| (0..num_perms).map(move |pi| (si, pi)))
        .collect();

    source_perm_pairs.par_iter().for_each(|&(si, pi)| {
        let (_, source_board) = &source_boards[si];
        let perm = &perms[pi];

        let mut hashes = FxHashSet::default();
        let mut nodes = 0u64;
        collect_reachable_hashes(
            *source_board,
            perm,
            0,
            args.max_height,
            args.max_holes,
            &mut hashes,
            &mut nodes,
        );

        if hashes.is_empty() {
            zero_reach_pairs.fetch_add(1, Ordering::Relaxed);
        }

        for h in &hashes {
            board_counts.entry(*h).and_modify(|c| *c += 1).or_insert(1);
        }

        let done = pairs_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 10000 == 0 || done == total_pairs {
            println!(
                "[{:>8}/{}] unique_boards={:<10} zero_reach={:<6} time={:.1}s",
                done,
                total_pairs,
                board_counts.len(),
                zero_reach_pairs.load(Ordering::Relaxed),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_phase1 = overall_start.elapsed().as_secs_f64();
    let unique_boards = board_counts.len();

    let zero_reach = zero_reach_pairs.load(Ordering::Relaxed);

    println!();
    println!("--- phase 1 done ---");
    println!("phase1_time      = {elapsed_phase1:.3}s");
    println!("unique_boards    = {unique_boards}");
    println!("zero_reach_pairs = {zero_reach}");
    if zero_reach > 0 {
        println!(
            "WARNING: {} pairs have NO reachable boards within constraints!",
            zero_reach
        );
    }

    // Sort by count descending
    let mut sorted: Vec<(u64, u32)> = board_counts
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let universal_count = sorted
        .iter()
        .filter(|(_, c)| *c == total_pairs as u32)
        .count();
    println!("universal_boards = {universal_count} (reachable by all {total_pairs} pairs)");

    println!();
    println!("--- top 20 boards by pair count ---");
    for (i, (hash, count)) in sorted.iter().take(20).enumerate() {
        let pct = *count as f64 / total_pairs as f64 * 100.0;
        println!("  [{i:>2}] hash={hash:<20} count={count}/{total_pairs} ({pct:.1}%)");
    }

    // -----------------------------------------------------------------------
    // Phase 2: Re-run DFS for top-K boards, track per-source PermBitsets
    // -----------------------------------------------------------------------
    let top_k = args.top_k.min(sorted.len());
    let target_hashes: FxHashSet<u64> = sorted.iter().take(top_k).map(|(h, _)| *h).collect();

    println!();
    println!("=== Phase 2: verifying top {} boards ===", top_k);

    // For each target board, store one PermBitset per source board
    let board_source_bitsets: DashMap<u64, Vec<PermBitset>> = DashMap::new();
    let board_examples: DashMap<u64, TetrisBoard> = DashMap::new();
    let pairs_done2 = AtomicUsize::new(0);

    source_perm_pairs.par_iter().for_each(|&(si, pi)| {
        let (_, source_board) = &source_boards[si];
        let perm = &perms[pi];

        let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
        let mut nodes = 0u64;
        collect_reachable_filtered(
            *source_board,
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
                board_source_bitsets
                    .entry(hash)
                    .and_modify(|bitsets| bitset_set(&mut bitsets[si], pi))
                    .or_insert_with(|| {
                        let mut bitsets = vec![[0u64; BITSET_WORDS]; num_sources];
                        bitset_set(&mut bitsets[si], pi);
                        bitsets
                    });
                board_examples.entry(hash).or_insert(board);
            }
        }

        let done = pairs_done2.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 10000 == 0 || done == total_pairs {
            println!(
                "[{:>8}/{}] verified_boards={:<8} time={:.1}s",
                done,
                total_pairs,
                board_source_bitsets.len(),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_phase2 = overall_start.elapsed().as_secs_f64();
    println!();
    println!("--- phase 2 done ---");
    println!("phase2_time      = {:.3}s", elapsed_phase2 - elapsed_phase1);
    println!("verified_boards  = {}", board_source_bitsets.len());

    // -----------------------------------------------------------------------
    // Greedy set cover over (source, perm) pairs
    // -----------------------------------------------------------------------
    println!();
    println!("=== Greedy set cover ===");
    println!(
        "Universe: {} sources x {} perms = {} pairs",
        num_sources, num_perms, total_pairs
    );

    let entries: Vec<(u64, Vec<PermBitset>)> = board_source_bitsets
        .iter()
        .map(|e| (*e.key(), e.value().clone()))
        .collect();

    // Track coverage per source
    let all_perm_mask = bitset_full(num_perms);
    let mut source_covered: Vec<PermBitset> = vec![[0u64; BITSET_WORDS]; num_sources];
    let mut cover: Vec<(u64, u32)> = Vec::new();

    loop {
        // Check if all sources are fully covered
        let all_done = source_covered.iter().all(|bits| {
            bits.iter()
                .zip(all_perm_mask.iter())
                .all(|(b, m)| (b & m) == *m)
        });
        if all_done {
            break;
        }

        // Find target that covers the most new (source, perm) pairs
        let mut best_idx = 0;
        let mut best_new = 0u32;
        for (i, (_, bitsets)) in entries.iter().enumerate() {
            let new_covered: u32 = bitsets
                .iter()
                .zip(source_covered.iter())
                .map(|(target_bits, covered_bits)| bitset_and_not_count(target_bits, covered_bits))
                .sum();
            if new_covered > best_new {
                best_new = new_covered;
                best_idx = i;
            }
        }

        if best_new == 0 {
            break;
        }

        let (hash, bitsets) = &entries[best_idx];
        for (si, target_bits) in bitsets.iter().enumerate() {
            bitset_or_assign(&mut source_covered[si], target_bits);
        }

        // Count total coverage
        let total_covered: u32 = source_covered
            .iter()
            .map(|bits| {
                bits.iter()
                    .zip(all_perm_mask.iter())
                    .map(|(b, m)| (b & m).count_ones())
                    .sum::<u32>()
            })
            .sum();

        cover.push((*hash, best_new));

        if let Some(board) = board_examples.get(hash) {
            println!(
                "  cover[{:>3}]: +{:<6} total={}/{} h={} c={} holes={}",
                cover.len() - 1,
                best_new,
                total_covered,
                total_pairs,
                board.height(),
                board.count(),
                board.total_holes(),
            );
            let _ = std::io::stdout().flush();
        }
    }

    // Report per-source coverage and collect uncovered pairs
    println!();
    let mut uncovered_pairs_list: Vec<(usize, usize)> = Vec::new();
    for (si, bits) in source_covered.iter().enumerate() {
        let covered: u32 = bits
            .iter()
            .zip(all_perm_mask.iter())
            .map(|(b, m)| (b & m).count_ones())
            .sum();
        if covered < num_perms as u32 {
            let (hash, board) = &source_boards[si];
            println!(
                "  UNCOVERED src[{}] hash={} covered={}/{} h={} c={}",
                si,
                hash,
                covered,
                num_perms,
                board.height(),
                board.count(),
            );
            // Collect the specific uncovered perm indices
            for pi in 0..num_perms {
                let word = pi / 64;
                let bit = pi % 64;
                let in_mask = (all_perm_mask[word] >> bit) & 1 == 1;
                let is_covered = (bits[word] >> bit) & 1 == 1;
                if in_mask && !is_covered {
                    uncovered_pairs_list.push((si, pi));
                }
            }
        }
    }

    let uncovered_sources = source_covered
        .iter()
        .filter(|bits| {
            !bits
                .iter()
                .zip(all_perm_mask.iter())
                .all(|(b, m)| (b & m) == *m)
        })
        .count();
    let total_uncovered_pairs = uncovered_pairs_list.len();

    println!();
    println!("--- after phase 2 set cover ---");
    println!("cover_size       = {}", cover.len());
    println!("total_pairs      = {}", total_pairs);
    println!("uncovered_sources= {}", uncovered_sources);
    println!("uncovered_pairs  = {}", total_uncovered_pairs);

    // -----------------------------------------------------------------------
    // Phase 3: Targeted recovery for uncovered pairs
    // -----------------------------------------------------------------------
    if !uncovered_pairs_list.is_empty() {
        println!();
        println!(
            "=== Phase 3: targeted recovery for {} uncovered pairs ===",
            uncovered_pairs_list.len()
        );

        // Run DFS for each uncovered pair, collect ALL reachable boards
        let recovery_boards: DashMap<u64, Vec<PermBitset>> = DashMap::new();
        let recovery_examples: DashMap<u64, TetrisBoard> = DashMap::new();
        let recovery_done = AtomicUsize::new(0);
        let recovery_zero = AtomicUsize::new(0);
        let recovery_total = uncovered_pairs_list.len();

        uncovered_pairs_list.par_iter().for_each(|&(si, pi)| {
            let (_, source_board) = &source_boards[si];
            let perm = &perms[pi];

            let mut hashes = FxHashSet::default();
            let mut nodes = 0u64;
            collect_reachable_hashes(
                *source_board,
                perm,
                0,
                args.max_height,
                args.max_holes,
                &mut hashes,
                &mut nodes,
            );

            if hashes.is_empty() {
                recovery_zero.fetch_add(1, Ordering::Relaxed);
            }

            // For each reachable board, record it
            for h in &hashes {
                recovery_boards
                    .entry(*h)
                    .and_modify(|bitsets| bitset_set(&mut bitsets[si], pi))
                    .or_insert_with(|| {
                        let mut bitsets = vec![[0u64; BITSET_WORDS]; num_sources];
                        bitset_set(&mut bitsets[si], pi);
                        bitsets
                    });
            }

            // Collect board examples from first hash (for display)
            if !hashes.is_empty() {
                // Re-run to get an actual board (we only have hashes from collect_reachable_hashes)
                let target_set: FxHashSet<u64> = hashes;
                let mut hits = Vec::new();
                let mut n2 = 0u64;
                collect_reachable_filtered(
                    *source_board,
                    perm,
                    0,
                    args.max_height,
                    args.max_holes,
                    &target_set,
                    &mut hits,
                    &mut n2,
                );
                for (hash, board) in hits {
                    recovery_examples.entry(hash).or_insert(board);
                }
            }

            let done = recovery_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 500 == 0 || done == recovery_total {
                println!(
                    "  [{}/{}] recovery_boards={} zero_reach={}",
                    done,
                    recovery_total,
                    recovery_boards.len(),
                    recovery_zero.load(Ordering::Relaxed),
                );
                let _ = std::io::stdout().flush();
            }
        });

        let recovery_zero_count = recovery_zero.load(Ordering::Relaxed);
        println!(
            "  recovery_boards = {}, zero_reach = {}",
            recovery_boards.len(),
            recovery_zero_count
        );

        if recovery_zero_count > 0 {
            println!(
                "  WARNING: {} pairs have NO reachable boards — constraint too tight!",
                recovery_zero_count
            );
        }

        // Greedy set cover over uncovered pairs using recovery boards
        if !recovery_boards.is_empty() {
            let recovery_entries: Vec<(u64, Vec<PermBitset>)> = recovery_boards
                .iter()
                .map(|e| (*e.key(), e.value().clone()))
                .collect();

            println!("  Running greedy set cover on recovery boards...");
            loop {
                let all_done = source_covered.iter().all(|bits| {
                    bits.iter()
                        .zip(all_perm_mask.iter())
                        .all(|(b, m)| (b & m) == *m)
                });
                if all_done {
                    break;
                }

                let mut best_idx = 0;
                let mut best_new = 0u32;
                for (i, (_, bitsets)) in recovery_entries.iter().enumerate() {
                    let new_covered: u32 = bitsets
                        .iter()
                        .zip(source_covered.iter())
                        .map(|(tb, cb)| bitset_and_not_count(tb, cb))
                        .sum();
                    if new_covered > best_new {
                        best_new = new_covered;
                        best_idx = i;
                    }
                }

                if best_new == 0 {
                    break;
                }

                let (hash, bitsets) = &recovery_entries[best_idx];
                for (si, target_bits) in bitsets.iter().enumerate() {
                    bitset_or_assign(&mut source_covered[si], target_bits);
                }

                cover.push((*hash, best_new));
                // Copy board example if we have it
                if let Some(board) = recovery_examples.get(hash) {
                    board_examples.entry(*hash).or_insert(*board);
                }

                let total_covered: u32 = source_covered
                    .iter()
                    .map(|bits| {
                        bits.iter()
                            .zip(all_perm_mask.iter())
                            .map(|(b, m)| (b & m).count_ones())
                            .sum::<u32>()
                    })
                    .sum();

                println!(
                    "  recovery cover[{:>3}]: +{:<5} total={}/{}",
                    cover.len() - 1,
                    best_new,
                    total_covered,
                    total_pairs,
                );
                let _ = std::io::stdout().flush();
            }
        }
    }

    // Final coverage report
    let final_uncovered_sources = source_covered
        .iter()
        .filter(|bits| {
            !bits
                .iter()
                .zip(all_perm_mask.iter())
                .all(|(b, m)| (b & m) == *m)
        })
        .count();
    let final_uncovered_pairs: u32 = source_covered
        .iter()
        .map(|bits| {
            let covered: u32 = bits
                .iter()
                .zip(all_perm_mask.iter())
                .map(|(b, m)| (b & m).count_ones())
                .sum();
            num_perms as u32 - covered
        })
        .sum();

    println!();
    println!("--- chain step {} final results ---", current_step);
    println!("cover_size       = {}", cover.len());
    println!("total_pairs      = {}", total_pairs);
    println!("uncovered_sources= {}", final_uncovered_sources);
    println!("uncovered_pairs  = {}", final_uncovered_pairs);

    if final_uncovered_pairs == 0 {
        println!("status           = COMPLETE");
    } else {
        println!(
            "status           = INCOMPLETE ({} pairs uncovered across {} sources)",
            final_uncovered_pairs, final_uncovered_sources
        );
    }

    // Print summary of cover boards (skip full board display for brevity)
    println!();
    println!(
        "--- cover summary (step {} targets): {} boards ---",
        current_step,
        cover.len()
    );
    // Show cell count distribution
    let mut cell_dist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for (hash, _) in &cover {
        if let Some(board) = board_examples.get(hash) {
            *cell_dist.entry(board.count()).or_insert(0) += 1;
        }
    }
    for (cells, n) in &cell_dist {
        println!("  cells={:<3} boards={}", cells, n);
    }

    // -----------------------------------------------------------------------
    // Save to DB
    // -----------------------------------------------------------------------
    let conn = Connection::open(&args.db)?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO chain_checkpoints
             (chain_step, board_hash, board_blob, height, cells, holes, roughness)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (hash, _) in &cover {
            if let Some(board) = board_examples.get(hash) {
                let blob = board.to_binary_slice();
                stmt.execute(params![
                    current_step as i64,
                    *hash as i64,
                    blob.as_slice(),
                    board.height(),
                    board.count(),
                    board.total_holes(),
                    board.roughness(),
                ])?;
            }
        }
    }
    tx.commit()?;

    let elapsed_total = overall_start.elapsed().as_secs_f64();
    println!("total_time       = {elapsed_total:.3}s");

    if current_step < 4 {
        println!();
        println!(
            "Chain step {} complete. Run `chain` again for step {}.",
            current_step,
            current_step + 1
        );
    } else {
        println!();
        println!("=== All 5 chain steps done (residue cycle 8->6->4->2->0) ===");
        println!("Next: verify that step 4 (mod 0) can reach step 0 (mod 8) to close the cycle.");
    }

    Ok(())
}
