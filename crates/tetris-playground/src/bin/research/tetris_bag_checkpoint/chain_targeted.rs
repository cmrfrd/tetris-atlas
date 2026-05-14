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
use tetris_game::{Major, TetrisBoard};

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
// Multi-target DFS: prune by min(targets), accept any target at terminal
// ---------------------------------------------------------------------------

fn collect_reachable_multi_target(
    board: TetrisBoard,
    pieces: &[tetris_game::TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    min_target: u32,
    target_set: &[u32],
    out: &mut Vec<(u64, TetrisBoard)>,
    nodes: &mut u64,
) {
    use tetris_game::{IsLost, TetrisPiecePlacement};

    if step == 7 {
        let c = board.count();
        if target_set.contains(&c) {
            out.push((board_hash(&board), board));
        }
        return;
    }

    let remaining = (7 - step) as u32;
    let current_cells = board.count();
    if current_cells + 4 * remaining < min_target {
        return;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
        let result = next.apply_piece_placement(placement);
        *nodes += 1;

        if result.is_lost == IsLost::LOST {
            continue;
        }
        if next.height() > max_height {
            continue;
        }
        if max_holes != u32::MAX && has_too_many_holes(&next, max_holes) {
            continue;
        }

        collect_reachable_multi_target(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            min_target,
            target_set,
            out,
            nodes,
        );
    }
}

// ---------------------------------------------------------------------------
// Short-circuit DFS: can this board reach ANY board at one of target_cells?
// Returns true on first hit. Used for forward-navigability testing.
// ---------------------------------------------------------------------------

fn can_reach_cell_count(
    board: TetrisBoard,
    pieces: &[tetris_game::TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    min_target: u32,
    target_set: &[u32],
) -> bool {
    use tetris_game::{IsLost, TetrisPiecePlacement};

    if step == 7 {
        let c = board.count();
        return target_set.contains(&c);
    }

    let remaining = (7 - step) as u32;
    let current_cells = board.count();
    if current_cells + 4 * remaining < min_target {
        return false;
    }

    let piece = pieces[step];
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

        if can_reach_cell_count(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            min_target,
            target_set,
        ) {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args)]
pub struct ChainTargetedArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 8)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = 6)]
    max_holes: u32,

    /// Cell count of source boards to load (ignored if --source-hashes is set).
    #[arg(long, default_value_t = 0)]
    source_cells: u32,

    /// Target cell count(s) for the next chain step (comma-separated, e.g. 32,42).
    #[arg(long, value_delimiter = ',')]
    target_cells: Vec<u32>,

    /// Number of top candidate boards to verify in phase 2.
    #[arg(long, default_value_t = 5000)]
    top_k: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Table to load source boards from.
    #[arg(long, default_value = "analyze_covers")]
    source_table: String,

    /// Comma-separated list of board hashes to use as sources.
    #[arg(long, value_delimiter = ',')]
    source_hashes: Option<Vec<i64>>,

    /// Maximum holes allowed in TARGET boards (for navigability filtering).
    /// If set, the greedy set cover and phase 2 only consider boards with holes <= this value.
    #[arg(long)]
    max_target_holes: Option<u32>,

    /// Maximum height allowed in TARGET boards.
    #[arg(long)]
    max_target_height: Option<u32>,

    /// Maximum roughness allowed in TARGET boards (for navigability filtering).
    #[arg(long)]
    max_target_rough: Option<u32>,

    /// Forward-navigability check: target cell counts for the NEXT chain step.
    /// If set, after Phase 1, each top-K candidate is tested as a SOURCE: it must
    /// reach at least one board at these cell counts for ALL 5040 permutations.
    /// Candidates that fail are excluded before the greedy set cover.
    #[arg(long, value_delimiter = ',')]
    verify_forward_cells: Option<Vec<u32>>,

    /// Max height for forward-navigability DFS (defaults to --max-height).
    #[arg(long)]
    verify_forward_max_height: Option<u32>,

    /// Max holes for forward-navigability DFS (defaults to --max-holes).
    #[arg(long)]
    verify_forward_max_holes: Option<u32>,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: ChainTargetedArgs) -> Result<()> {
    let num_perms = 5040usize;
    let overall_start = Instant::now();

    if args.target_cells.is_empty() {
        bail!("--target-cells requires at least one value");
    }
    let min_target = *args.target_cells.iter().min().unwrap();

    if !args.db.exists() {
        bail!("Database not found at {}", args.db.display());
    }

    // Load source boards
    let conn = Connection::open(&args.db)?;
    let source_boards: Vec<(u64, TetrisBoard)> = if let Some(ref hashes) = args.source_hashes {
        // Load by hash directly
        let mut boards = Vec::new();
        for &hash in hashes {
            for table in &[&args.source_table as &str, "boards", "chain_checkpoints"] {
                let sql = format!(
                    "SELECT board_hash, board_blob FROM {} WHERE board_hash = ?1",
                    table
                );
                if let Ok(row) = conn.query_row(&sql, params![hash], |row| {
                    let h: i64 = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((h as u64, blob))
                }) {
                    let arr: [u8; 200] = row.1.as_slice().try_into().unwrap_or([0u8; 200]);
                    boards.push((row.0, TetrisBoard::from_cell_array(arr, Major::Row)));
                    break;
                }
            }
        }
        boards
    } else {
        let query = format!(
            "SELECT board_hash, board_blob FROM {} WHERE cell_count = ?1",
            args.source_table
        );
        let mut stmt = conn.prepare(&query)?;
        stmt.query_map(params![args.source_cells as i64], |row| {
            let hash: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((hash as u64, blob))
        })?
        .filter_map(|r| r.ok())
        .map(|(hash, blob)| {
            let arr: [u8; 200] = blob.as_slice().try_into().unwrap_or([0u8; 200]);
            (hash, TetrisBoard::from_cell_array(arr, Major::Row))
        })
        .collect()
    };
    drop(conn);

    let num_sources = source_boards.len();
    if num_sources == 0 {
        bail!("No source boards found");
    }

    let total_pairs = num_sources * num_perms;

    println!("tetris_bag_checkpoint chain-targeted");
    println!("target_cells  = {:?}", args.target_cells);
    println!("max_height    = {}", args.max_height);
    println!("max_holes     = {}", args.max_holes);
    println!("source_boards = {}", num_sources);
    println!(
        "total_pairs   = {} ({} sources x {} perms)",
        total_pairs, num_sources, num_perms
    );
    println!("top_k         = {}", args.top_k);
    println!();

    for (i, (hash, board)) in source_boards.iter().enumerate().take(10) {
        println!(
            "  src[{:>2}] hash={:<21} h={} c={} holes={} rough={}",
            i,
            *hash as i64,
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness(),
        );
    }
    if num_sources > 10 {
        println!("  ... and {} more", num_sources - 10);
    }
    println!();

    let perms = generate_permutations();

    let source_perm_pairs: Vec<(usize, usize)> = (0..num_sources)
        .flat_map(|si| (0..num_perms).map(move |pi| (si, pi)))
        .collect();

    // -----------------------------------------------------------------------
    // Phase 1: For each (source, perm), count target board hits (lightweight)
    // -----------------------------------------------------------------------
    println!("=== Phase 1: counting target boards ===");

    let board_counts: DashMap<u64, u32> = DashMap::new();
    let board_examples: DashMap<u64, TetrisBoard> = DashMap::new();
    let pairs_done = AtomicUsize::new(0);
    let total_nodes = AtomicUsize::new(0);
    let zero_reach = AtomicUsize::new(0);

    source_perm_pairs.par_iter().for_each(|&(si, pi)| {
        let (_, source_board) = &source_boards[si];
        let perm = &perms[pi];

        let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
        let mut nodes = 0u64;
        collect_reachable_multi_target(
            *source_board,
            perm,
            0,
            args.max_height,
            args.max_holes,
            min_target,
            &args.target_cells,
            &mut hits,
            &mut nodes,
        );

        total_nodes.fetch_add(nodes as usize, Ordering::Relaxed);
        if hits.is_empty() {
            zero_reach.fetch_add(1, Ordering::Relaxed);
        }

        let mut seen = FxHashSet::default();
        for (hash, board) in &hits {
            if seen.insert(*hash) {
                // Apply target filtering if specified
                if let Some(max_h) = args.max_target_holes {
                    if has_too_many_holes(board, max_h) {
                        continue;
                    }
                }
                if let Some(max_th) = args.max_target_height {
                    if board.height() > max_th {
                        continue;
                    }
                }
                if let Some(max_r) = args.max_target_rough {
                    if board.roughness() > max_r {
                        continue;
                    }
                }
                board_counts
                    .entry(*hash)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
                board_examples.entry(*hash).or_insert(*board);
            }
        }

        let done = pairs_done.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 5000 == 0 || done == total_pairs {
            println!(
                "[{:>8}/{}] unique_targets={:<10} zero_reach={:<6} time={:.1}s nodes={:.1}M",
                done,
                total_pairs,
                board_counts.len(),
                zero_reach.load(Ordering::Relaxed),
                overall_start.elapsed().as_secs_f64(),
                total_nodes.load(Ordering::Relaxed) as f64 / 1e6,
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_p1 = overall_start.elapsed().as_secs_f64();
    let zr = zero_reach.load(Ordering::Relaxed);
    let unique_targets = board_counts.len();
    println!();
    println!("--- phase 1 done ---");
    println!("time           = {elapsed_p1:.3}s");
    println!("unique_targets = {unique_targets}");
    println!("zero_reach     = {zr}");

    if zr > 0 {
        println!(
            "WARNING: {} pairs have NO reachable {:?}-cell boards!",
            zr, args.target_cells
        );
    }

    // Sort by count descending, take top-K
    let mut sorted: Vec<(u64, u32)> = board_counts
        .iter()
        .map(|e| (*e.key(), *e.value()))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    drop(board_counts);

    println!();
    println!("--- top 10 targets by pair count ---");
    for (i, (hash, count)) in sorted.iter().take(10).enumerate() {
        let pct = *count as f64 / total_pairs as f64 * 100.0;
        println!("  [{i:>2}] hash={hash:<21} count={count}/{total_pairs} ({pct:.1}%)");
    }

    // -----------------------------------------------------------------------
    // Phase 1.5 (optional): Forward-navigability pre-filter
    // -----------------------------------------------------------------------
    let top_k = args.top_k.min(sorted.len());

    let sorted = if let Some(ref fwd_cells) = args.verify_forward_cells {
        let fwd_max_h = args.verify_forward_max_height.unwrap_or(args.max_height);
        let fwd_max_holes = args.verify_forward_max_holes.unwrap_or(args.max_holes);
        let fwd_min_target = *fwd_cells.iter().min().unwrap();

        println!();
        println!("=== Phase 1.5: forward-navigability pre-filter ===");
        println!("forward_cells     = {:?}", fwd_cells);
        println!("forward_max_h     = {}", fwd_max_h);
        println!("forward_max_holes = {}", fwd_max_holes);

        // Collect top-K candidates from Phase 1's board_examples
        let candidates: Vec<(u64, TetrisBoard)> = sorted
            .iter()
            .take(top_k)
            .filter_map(|(hash, _)| board_examples.get(hash).map(|b| (*hash, *b)))
            .collect();

        println!("candidates        = {}", candidates.len());
        println!();

        // Test each candidate as a SOURCE for forward navigability
        let tested = AtomicUsize::new(0);
        let passed = AtomicUsize::new(0);
        let failed = AtomicUsize::new(0);

        let navigable: DashMap<u64, bool> = DashMap::new();

        candidates.par_iter().for_each(|(hash, board)| {
            let mut all_ok = true;
            for perm in &perms {
                if !can_reach_cell_count(
                    *board,
                    perm,
                    0,
                    fwd_max_h,
                    fwd_max_holes,
                    fwd_min_target,
                    fwd_cells,
                ) {
                    all_ok = false;
                    break;
                }
            }

            navigable.insert(*hash, all_ok);
            let done = tested.fetch_add(1, Ordering::Relaxed) + 1;
            if all_ok {
                passed.fetch_add(1, Ordering::Relaxed);
            } else {
                failed.fetch_add(1, Ordering::Relaxed);
            }

            if done % 50 == 0 || done == candidates.len() {
                eprint!(
                    "\r  [{}/{}] passed={} failed={} time={:.1}s",
                    done,
                    candidates.len(),
                    passed.load(Ordering::Relaxed),
                    failed.load(Ordering::Relaxed),
                    overall_start.elapsed().as_secs_f64(),
                );
                let _ = std::io::stderr().flush();
            }
        });
        eprintln!();

        let pass_count = passed.load(Ordering::Relaxed);
        let fail_count = failed.load(Ordering::Relaxed);
        println!(
            "  forward-navigable: {}/{} passed ({} failed)",
            pass_count,
            pass_count + fail_count,
            fail_count
        );

        // Filter sorted to only navigable boards
        let filtered: Vec<(u64, u32)> = sorted
            .into_iter()
            .filter(|(h, _)| navigable.get(h).map(|v| *v).unwrap_or(false))
            .collect();

        println!("  candidates after filter: {}", filtered.len());
        filtered
    } else {
        sorted
    };

    // -----------------------------------------------------------------------
    // Phase 2: Re-run DFS for top-K, track per-source PermBitsets
    // -----------------------------------------------------------------------
    let top_k = top_k.min(sorted.len());
    let target_hashes: FxHashSet<u64> = sorted.iter().take(top_k).map(|(h, _)| *h).collect();

    println!();
    println!("=== Phase 2: verifying top {} targets ===", top_k);

    let board_source_bitsets: DashMap<u64, Vec<PermBitset>> = DashMap::new();
    let pairs_done2 = AtomicUsize::new(0);

    source_perm_pairs.par_iter().for_each(|&(si, pi)| {
        let (_, source_board) = &source_boards[si];
        let perm = &perms[pi];

        let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
        let mut nodes = 0u64;
        collect_reachable_multi_target(
            *source_board,
            perm,
            0,
            args.max_height,
            args.max_holes,
            min_target,
            &args.target_cells,
            &mut hits,
            &mut nodes,
        );

        let mut seen = FxHashSet::default();
        for (hash, board) in hits {
            if target_hashes.contains(&hash) && seen.insert(hash) {
                // Apply target filtering
                if let Some(max_h) = args.max_target_holes {
                    if has_too_many_holes(&board, max_h) {
                        continue;
                    }
                }
                if let Some(max_th) = args.max_target_height {
                    if board.height() > max_th {
                        continue;
                    }
                }
                if let Some(max_r) = args.max_target_rough {
                    if board.roughness() > max_r {
                        continue;
                    }
                }
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
        if done % 5000 == 0 || done == total_pairs {
            println!(
                "[{:>8}/{}] verified_targets={:<8} time={:.1}s",
                done,
                total_pairs,
                board_source_bitsets.len(),
                overall_start.elapsed().as_secs_f64(),
            );
            let _ = std::io::stdout().flush();
        }
    });

    let elapsed_p2 = overall_start.elapsed().as_secs_f64();
    println!();
    println!("--- phase 2 done ---");
    println!("time           = {:.3}s", elapsed_p2 - elapsed_p1);
    println!("verified       = {}", board_source_bitsets.len());

    // -----------------------------------------------------------------------
    // Greedy set cover over (source, perm) pairs
    // -----------------------------------------------------------------------
    println!();
    println!("=== Greedy set cover ===");

    let all_perm_mask = bitset_full(num_perms);
    let entries: Vec<(u64, Vec<PermBitset>)> = board_source_bitsets
        .iter()
        .map(|e| (*e.key(), e.value().clone()))
        .collect();
    drop(board_source_bitsets);
    let mut cover_perm_counts: std::collections::HashMap<u64, u32> = entries
        .iter()
        .map(|(hash, bitsets)| (*hash, bitsets.iter().map(bitset_count).sum()))
        .collect();

    let mut source_covered: Vec<PermBitset> = vec![[0u64; BITSET_WORDS]; num_sources];
    let mut cover: Vec<(u64, u32)> = Vec::new();

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
        for (i, (_, bitsets)) in entries.iter().enumerate() {
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

        let (hash, bitsets) = &entries[best_idx];
        for (si, target_bits) in bitsets.iter().enumerate() {
            bitset_or_assign(&mut source_covered[si], target_bits);
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

        cover.push((*hash, best_new));

        if let Some(board) = board_examples.get(hash) {
            println!(
                "  cover[{:>3}]: +{:<6} total={}/{} h={} c={} holes={} rough={}",
                cover.len() - 1,
                best_new,
                total_covered,
                total_pairs,
                board.height(),
                board.count(),
                board.total_holes(),
                board.roughness(),
            );
            let _ = std::io::stdout().flush();
        }
    }

    // Per-source coverage report
    let mut uncovered_sources = 0;
    let mut total_uncovered_pairs = 0u32;
    let mut uncovered_pairs: Vec<(usize, usize)> = Vec::new();
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
                *hash as i64,
                covered,
                num_perms,
                board.height(),
                board.count(),
            );
            uncovered_sources += 1;
            total_uncovered_pairs += num_perms as u32 - covered;
            for pi in 0..num_perms {
                let word = pi / 64;
                let bit = pi % 64;
                let is_permutation = (all_perm_mask[word] >> bit) & 1 == 1;
                let is_covered = (bits[word] >> bit) & 1 == 1;
                if is_permutation && !is_covered {
                    uncovered_pairs.push((si, pi));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: targeted recovery for pairs missed by the top-K candidate pass.
    // This keeps broad runs small while still giving rare source/permutation
    // pairs a chance to contribute specialized checkpoint boards.
    // -----------------------------------------------------------------------
    if !uncovered_pairs.is_empty() {
        println!();
        println!(
            "=== Phase 3: targeted recovery for {} uncovered pairs ===",
            uncovered_pairs.len()
        );
        if args.verify_forward_cells.is_some() {
            println!(
                "recovery note   = applying target shape filters; forward cell-count filter was already applied to top-K candidates"
            );
        }

        let recovery_boards: DashMap<u64, Vec<PermBitset>> = DashMap::new();
        let recovery_examples: DashMap<u64, TetrisBoard> = DashMap::new();
        let recovery_done = AtomicUsize::new(0);
        let recovery_zero = AtomicUsize::new(0);
        let recovery_total = uncovered_pairs.len();

        uncovered_pairs.par_iter().for_each(|&(si, pi)| {
            let (_, source_board) = &source_boards[si];
            let perm = &perms[pi];

            let mut hits: Vec<(u64, TetrisBoard)> = Vec::new();
            let mut nodes = 0u64;
            collect_reachable_multi_target(
                *source_board,
                perm,
                0,
                args.max_height,
                args.max_holes,
                min_target,
                &args.target_cells,
                &mut hits,
                &mut nodes,
            );

            if hits.is_empty() {
                recovery_zero.fetch_add(1, Ordering::Relaxed);
            }

            let mut seen = FxHashSet::default();
            for (hash, board) in hits {
                if !seen.insert(hash) {
                    continue;
                }
                if let Some(max_h) = args.max_target_holes {
                    if has_too_many_holes(&board, max_h) {
                        continue;
                    }
                }
                if let Some(max_th) = args.max_target_height {
                    if board.height() > max_th {
                        continue;
                    }
                }
                if let Some(max_r) = args.max_target_rough {
                    if board.roughness() > max_r {
                        continue;
                    }
                }

                recovery_boards
                    .entry(hash)
                    .and_modify(|bitsets| bitset_set(&mut bitsets[si], pi))
                    .or_insert_with(|| {
                        let mut bitsets = vec![[0u64; BITSET_WORDS]; num_sources];
                        bitset_set(&mut bitsets[si], pi);
                        bitsets
                    });
                recovery_examples.entry(hash).or_insert(board);
            }

            let done = recovery_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 || done == recovery_total {
                println!(
                    "  [{}/{}] recovery_targets={} zero_reach={}",
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
            "  recovery_targets = {}, zero_reach = {}",
            recovery_boards.len(),
            recovery_zero_count
        );

        let recovery_entries: Vec<(u64, Vec<PermBitset>)> = recovery_boards
            .iter()
            .map(|e| (*e.key(), e.value().clone()))
            .collect();
        for (hash, bitsets) in &recovery_entries {
            cover_perm_counts
                .entry(*hash)
                .or_insert_with(|| bitsets.iter().map(bitset_count).sum());
        }

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
            if let Some(board) = recovery_examples.get(hash) {
                board_examples.entry(*hash).or_insert(*board);
                println!(
                    "  recovery cover[{:>3}]: +{:<6} h={} c={} holes={} rough={}",
                    cover.len() - 1,
                    best_new,
                    board.height(),
                    board.count(),
                    board.total_holes(),
                    board.roughness(),
                );
            } else {
                println!("  recovery cover[{:>3}]: +{:<6}", cover.len() - 1, best_new);
            }
        }

        uncovered_sources = 0;
        total_uncovered_pairs = 0;
        for (si, bits) in source_covered.iter().enumerate() {
            let covered: u32 = bits
                .iter()
                .zip(all_perm_mask.iter())
                .map(|(b, m)| (b & m).count_ones())
                .sum();
            if covered < num_perms as u32 {
                let (hash, board) = &source_boards[si];
                println!(
                    "  STILL UNCOVERED src[{}] hash={} covered={}/{} h={} c={}",
                    si,
                    *hash as i64,
                    covered,
                    num_perms,
                    board.height(),
                    board.count(),
                );
                uncovered_sources += 1;
                total_uncovered_pairs += num_perms as u32 - covered;
            }
        }
    }

    println!();
    println!("--- results ---");
    println!("cover_size     = {}", cover.len());
    println!("total_pairs    = {}", total_pairs);
    println!("uncovered_src  = {}", uncovered_sources);
    println!("uncovered_pairs= {}", total_uncovered_pairs);
    println!(
        "status         = {}",
        if total_uncovered_pairs == 0 {
            "COMPLETE"
        } else {
            "INCOMPLETE"
        }
    );

    // Print and save cover boards
    if total_uncovered_pairs == 0 {
        println!();
        println!("--- cover boards ---");
        for (i, (hash, new)) in cover.iter().enumerate() {
            if let Some(board) = board_examples.get(hash) {
                println!(
                    "  cover[{:>2}]: +{:<5} h={} c={} holes={} rough={} hash={}",
                    i,
                    new,
                    board.height(),
                    board.count(),
                    board.total_holes(),
                    board.roughness(),
                    *hash as i64,
                );
                print_board(&board);
            }
        }

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
            for (hash, _) in &cover {
                if let Some(board) = board_examples.get(hash) {
                    let total_perms = { *cover_perm_counts.get(hash).unwrap_or(&0) };
                    stmt.execute(params![
                        *hash as i64,
                        &board.to_cell_array(Major::Row) as &[u8],
                        board.count() as i64,
                        total_perms as i64,
                    ])?;
                }
            }
        }
        tx.commit()?;
        println!();
        println!("saved {} cover boards to DB", cover.len());
    }

    let total_elapsed = overall_start.elapsed().as_secs_f64();
    println!("total_time     = {total_elapsed:.3}s");

    Ok(())
}
