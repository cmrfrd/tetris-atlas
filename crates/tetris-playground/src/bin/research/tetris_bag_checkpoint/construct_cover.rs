use std::collections::BinaryHeap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args;
use dashmap::DashMap;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_search::scoring::height_mse_distance_from_empty;

use crate::common::*;

#[derive(Args)]
pub struct ConstructCoverArgs {
    /// Path to SQLite database.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Load source boards from chain_checkpoints at this step.
    #[arg(long)]
    source_step: i64,

    /// Save selected target boards to chain_checkpoints at this step.
    #[arg(long)]
    target_step: i64,

    /// Target cell count(s), comma-separated.
    #[arg(long, value_delimiter = ',')]
    target_cells: Vec<u32>,

    /// Maximum board height allowed during the bag.
    #[arg(long, default_value_t = 8)]
    max_height: u32,

    /// Maximum holes allowed during the bag.
    #[arg(long, default_value_t = 6)]
    max_holes: u32,

    /// Maximum holes allowed in saved target boards.
    #[arg(long)]
    max_target_holes: Option<u32>,

    /// Maximum height allowed in saved target boards.
    #[arg(long)]
    max_target_height: Option<u32>,

    /// Maximum roughness allowed in saved target boards.
    #[arg(long)]
    max_target_rough: Option<u32>,

    /// Beam width for terminal candidate generation.
    #[arg(long, default_value_t = 128)]
    beam_width: usize,

    /// Maximum terminal target candidates retained per source/permutation pair.
    #[arg(long, default_value_t = 16)]
    terminals_per_pair: usize,

    /// Maximum DFS fallback candidates retained when the beam finds no target.
    /// Set to 0 to disable fallback and diagnose beam coverage.
    #[arg(long, default_value_t = 1)]
    fallback_terminals_per_pair: usize,
}

#[derive(Clone, Copy)]
struct ScoredBoard {
    board: TetrisBoard,
    score: f32,
}

struct Candidate {
    key: BoardKey,
    board: TetrisBoard,
    pairs: Vec<u32>,
}

struct GreedyCover {
    selected: Vec<(usize, usize)>,
    uncovered: usize,
    pruned: usize,
}

fn board_score(board: TetrisBoard) -> f32 {
    -height_mse_distance_from_empty(board)
        - (board.total_holes() as f32 * 0.25)
        - (board.roughness() as f32 * 0.05)
}

fn target_ok(board: &TetrisBoard, args: &ConstructCoverArgs) -> bool {
    if !args.target_cells.contains(&board.count()) {
        return false;
    }
    if let Some(max_height) = args.max_target_height {
        if board.height() > max_height {
            return false;
        }
    }
    if let Some(max_holes) = args.max_target_holes {
        if has_too_many_holes(board, max_holes) {
            return false;
        }
    }
    if let Some(max_rough) = args.max_target_rough {
        if board.roughness() > max_rough {
            return false;
        }
    }
    true
}

fn beam_terminal_candidates(
    start: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    min_target: u32,
    args: &ConstructCoverArgs,
    nodes: &mut u64,
) -> Vec<TetrisBoard> {
    if args.beam_width == 0 || args.terminals_per_pair == 0 {
        return Vec::new();
    }

    let mut beam = vec![ScoredBoard {
        board: start,
        score: board_score(start),
    }];

    for (step, &piece) in pieces.iter().enumerate() {
        let mut candidates = Vec::with_capacity(beam.len() * 34);
        for parent in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut next = parent.board;
                let result = next.apply_piece_placement(placement);
                *nodes += 1;

                if result.is_lost == IsLost::LOST {
                    continue;
                }
                if next.height() > args.max_height {
                    continue;
                }
                if args.max_holes != u32::MAX && has_too_many_holes(&next, args.max_holes) {
                    continue;
                }
                let remaining = (7 - (step + 1)) as u32;
                if next.count() + 4 * remaining < min_target {
                    continue;
                }

                candidates.push(ScoredBoard {
                    board: next,
                    score: board_score(next),
                });
            }
        }

        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut seen = FxHashSet::default();
        beam.clear();
        for candidate in candidates {
            if seen.insert(candidate.board) {
                beam.push(candidate);
                if beam.len() >= args.beam_width {
                    break;
                }
            }
        }

        if beam.is_empty() {
            return Vec::new();
        }
    }

    let mut terminals: Vec<ScoredBoard> = beam
        .into_iter()
        .filter(|candidate| target_ok(&candidate.board, args))
        .collect();
    terminals.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut seen = FxHashSet::default();
    terminals
        .into_iter()
        .filter_map(|candidate| seen.insert(candidate.board).then_some(candidate.board))
        .take(args.terminals_per_pair)
        .collect()
}

struct DfsCollectCtx<'a> {
    pieces: &'a [TetrisPiece; 7],
    min_target: u32,
    args: &'a ConstructCoverArgs,
    memo: FxHashSet<(BoardKey, u8)>,
    out: Vec<TetrisBoard>,
    limit: usize,
    nodes: &'a mut u64,
}

impl DfsCollectCtx<'_> {
    fn collect(&mut self, board: TetrisBoard, step: usize) -> bool {
        if self.out.len() >= self.limit {
            return true;
        }

        if step == 7 {
            if target_ok(&board, self.args) {
                self.out.push(board);
            }
            return self.out.len() >= self.limit;
        }

        let remaining = (7 - step) as u32;
        if board.count() + 4 * remaining < self.min_target {
            return false;
        }

        let memo_key = (board_key(&board), step as u8);
        if self.memo.contains(&memo_key) {
            return false;
        }

        let piece = self.pieces[step];
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next = board;
            let result = next.apply_piece_placement(placement);
            *self.nodes += 1;

            if result.is_lost == IsLost::LOST {
                continue;
            }
            if next.height() > self.args.max_height {
                continue;
            }
            if self.args.max_holes != u32::MAX && has_too_many_holes(&next, self.args.max_holes) {
                continue;
            }

            if self.collect(next, step + 1) {
                return true;
            }
        }

        self.memo.insert(memo_key);
        false
    }
}

fn dfs_terminal_candidates(
    start: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    min_target: u32,
    limit: usize,
    args: &ConstructCoverArgs,
    nodes: &mut u64,
) -> Vec<TetrisBoard> {
    if limit == 0 {
        return Vec::new();
    }
    let mut ctx = DfsCollectCtx {
        pieces,
        min_target,
        args,
        memo: FxHashSet::default(),
        out: Vec::new(),
        limit,
        nodes,
    };
    ctx.collect(start, 0);
    ctx.out
}

fn bit_is_set(bits: &[u64], idx: usize) -> bool {
    (bits[idx / 64] >> (idx % 64)) & 1 == 1
}

fn bit_set(bits: &mut [u64], idx: usize) {
    bits[idx / 64] |= 1u64 << (idx % 64);
}

fn count_new_pairs(pair_set: &[u32], covered: &[u64]) -> usize {
    pair_set
        .iter()
        .filter(|&&pair_id| !bit_is_set(covered, pair_id as usize))
        .count()
}

fn greedy_cover(pair_sets: &[Vec<u32>], universe_size: usize) -> GreedyCover {
    let mut covered = vec![0u64; universe_size.div_ceil(64)];
    let mut covered_count = 0usize;
    let mut heap = BinaryHeap::new();

    for (idx, pairs) in pair_sets.iter().enumerate() {
        if !pairs.is_empty() {
            heap.push((pairs.len(), idx));
        }
    }

    let mut selected = Vec::new();

    while covered_count < universe_size {
        let Some((score, idx)) = heap.pop() else {
            break;
        };
        let actual = count_new_pairs(&pair_sets[idx], &covered);
        if actual == 0 {
            continue;
        }
        if actual < score {
            heap.push((actual, idx));
            continue;
        }

        for &pair_id in &pair_sets[idx] {
            let pair_idx = pair_id as usize;
            if !bit_is_set(&covered, pair_idx) {
                bit_set(&mut covered, pair_idx);
                covered_count += 1;
            }
        }
        selected.push((idx, actual));
    }

    let before_prune = selected.len();
    reverse_prune_cover(pair_sets, &mut selected, universe_size);
    let pruned = before_prune - selected.len();

    GreedyCover {
        selected,
        uncovered: universe_size - covered_count,
        pruned,
    }
}

fn reverse_prune_cover(
    pair_sets: &[Vec<u32>],
    selected: &mut Vec<(usize, usize)>,
    universe_size: usize,
) {
    let mut coverage_counts = vec![0u16; universe_size];
    for (idx, _) in selected.iter() {
        for &pair_id in &pair_sets[*idx] {
            coverage_counts[pair_id as usize] += 1;
        }
    }

    let mut prune_order: Vec<usize> = (0..selected.len()).collect();
    prune_order.sort_by_key(|&selected_idx| {
        let candidate_idx = selected[selected_idx].0;
        (
            pair_sets[candidate_idx].len(),
            std::cmp::Reverse(selected_idx),
        )
    });

    let mut keep = vec![true; selected.len()];
    for selected_idx in prune_order {
        let candidate_idx = selected[selected_idx].0;
        if pair_sets[candidate_idx]
            .iter()
            .all(|&pair_id| coverage_counts[pair_id as usize] > 1)
        {
            keep[selected_idx] = false;
            for &pair_id in &pair_sets[candidate_idx] {
                coverage_counts[pair_id as usize] -= 1;
            }
        }
    }

    let mut keep_iter = keep.into_iter();
    selected.retain(|_| keep_iter.next().unwrap_or(false));
}

fn load_sources(db: &PathBuf, source_step: i64) -> Result<Vec<TetrisBoard>> {
    if !db.exists() {
        bail!("Database not found at {}", db.display());
    }

    let conn = Connection::open(db)?;
    let mut stmt = conn.prepare(
        "SELECT board_blob FROM chain_checkpoints WHERE chain_step = ?1 ORDER BY board_hash",
    )?;
    let rows = stmt.query_map(params![source_step], |row| {
        let blob: Vec<u8> = row.get(0)?;
        Ok(blob)
    })?;

    let mut boards = Vec::new();
    for row in rows {
        let blob = row?;
        boards.push(
            BoardKey::from_blob(&blob)
                .with_context(|| format!("decoding source step {source_step}"))?
                .to_board(),
        );
    }
    if boards.is_empty() {
        bail!("no source boards loaded for step {source_step}");
    }
    Ok(boards)
}

fn save_cover(
    args: &ConstructCoverArgs,
    candidates: &[Candidate],
    cover: &GreedyCover,
) -> Result<()> {
    let mut conn = Connection::open(&args.db)?;
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
         );

         CREATE TABLE IF NOT EXISTS construct_cover_meta (
             run_id TEXT NOT NULL,
             key TEXT NOT NULL,
             value TEXT NOT NULL,
             PRIMARY KEY (run_id, key)
         );",
    )?;

    let tx = conn.unchecked_transaction()?;
    {
        tx.execute(
            "DELETE FROM chain_checkpoints WHERE chain_step = ?1",
            params![args.target_step],
        )?;
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO chain_checkpoints
             (chain_step, board_hash, board_blob, height, cells, holes, roughness)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for (idx, _) in &cover.selected {
            let board = candidates[*idx].board;
            let blob = board.to_binary_slice();
            stmt.execute(params![
                args.target_step,
                board_hash(&board) as i64,
                blob.as_slice(),
                board.height(),
                board.count(),
                board.total_holes(),
                board.roughness(),
            ])?;
        }
    }
    tx.commit()?;

    Ok(())
}

pub fn run(args: ConstructCoverArgs) -> Result<()> {
    if args.target_cells.is_empty() {
        bail!("--target-cells requires at least one value");
    }
    if args.terminals_per_pair == 0 {
        bail!("--terminals-per-pair must be greater than zero");
    }

    let min_target = *args.target_cells.iter().min().unwrap();
    let sources = load_sources(&args.db, args.source_step)?;
    let perms = generate_permutations();
    let total_pairs = sources.len() * perms.len();

    println!("tetris_bag_checkpoint construct-cover");
    println!("db                 = {}", args.db.display());
    println!("source_step        = {}", args.source_step);
    println!("target_step        = {}", args.target_step);
    println!("source_boards      = {}", sources.len());
    println!("target_cells       = {:?}", args.target_cells);
    println!("max_height         = {}", args.max_height);
    println!("max_holes          = {}", args.max_holes);
    println!("max_target_height  = {:?}", args.max_target_height);
    println!("max_target_holes   = {:?}", args.max_target_holes);
    println!("max_target_rough   = {:?}", args.max_target_rough);
    println!("beam_width         = {}", args.beam_width);
    println!("terminals_per_pair = {}", args.terminals_per_pair);
    println!("fallback_terminals = {}", args.fallback_terminals_per_pair);
    println!("total_pairs        = {}", total_pairs);
    println!();

    let start = Instant::now();
    let checked = AtomicUsize::new(0);
    let ok_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);
    let beam_miss_count = AtomicUsize::new(0);
    let total_nodes = AtomicUsize::new(0);
    let candidate_pairs: DashMap<BoardKey, Vec<u32>> = DashMap::new();
    let candidate_examples: DashMap<BoardKey, TetrisBoard> = DashMap::new();

    let source_perm_pairs: Vec<(usize, usize)> = (0..sources.len())
        .flat_map(|source_idx| (0..perms.len()).map(move |perm_idx| (source_idx, perm_idx)))
        .collect();

    source_perm_pairs
        .par_iter()
        .for_each(|&(source_idx, perm_idx)| {
            let pair_id = (source_idx * perms.len() + perm_idx) as u32;
            let mut nodes = 0u64;
            let mut terminals = beam_terminal_candidates(
                sources[source_idx],
                &perms[perm_idx],
                min_target,
                &args,
                &mut nodes,
            );

            if terminals.is_empty() {
                beam_miss_count.fetch_add(1, Ordering::Relaxed);
                terminals = dfs_terminal_candidates(
                    sources[source_idx],
                    &perms[perm_idx],
                    min_target,
                    args.fallback_terminals_per_pair,
                    &args,
                    &mut nodes,
                );
            }

            total_nodes.fetch_add(nodes as usize, Ordering::Relaxed);
            if terminals.is_empty() {
                fail_count.fetch_add(1, Ordering::Relaxed);
            } else {
                ok_count.fetch_add(1, Ordering::Relaxed);
            }

            let mut seen = FxHashSet::default();
            for board in terminals {
                let key = board_key(&board);
                if !seen.insert(key) {
                    continue;
                }
                candidate_pairs
                    .entry(key)
                    .and_modify(|pairs| pairs.push(pair_id))
                    .or_insert_with(|| vec![pair_id]);
                candidate_examples.entry(key).or_insert(board);
            }

            let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 5000 == 0 || done == total_pairs {
                eprint!(
                    "\r  [{}/{}] ok={} fail={} beam_miss={} candidates={} nodes={:.1}M time={:.1}s",
                    done,
                    total_pairs,
                    ok_count.load(Ordering::Relaxed),
                    fail_count.load(Ordering::Relaxed),
                    beam_miss_count.load(Ordering::Relaxed),
                    candidate_pairs.len(),
                    total_nodes.load(Ordering::Relaxed) as f64 / 1e6,
                    start.elapsed().as_secs_f64(),
                );
                let _ = std::io::stderr().flush();
            }
        });
    eprintln!();

    let failed = fail_count.load(Ordering::Relaxed);
    if failed > 0 {
        bail!("construct-cover failed: {failed} source/permutation pairs had no target");
    }

    println!();
    println!("--- candidate generation done ---");
    println!("pairs_ok       = {}", ok_count.load(Ordering::Relaxed));
    println!(
        "beam_misses    = {}",
        beam_miss_count.load(Ordering::Relaxed)
    );
    println!("candidates     = {}", candidate_pairs.len());
    println!(
        "nodes          = {:.1}M",
        total_nodes.load(Ordering::Relaxed) as f64 / 1e6
    );
    println!("gen_time       = {:.3}s", start.elapsed().as_secs_f64());

    let mut candidates: Vec<Candidate> = candidate_pairs
        .iter()
        .map(|entry| {
            let key = *entry.key();
            let mut pairs = entry.value().clone();
            pairs.sort_unstable();
            pairs.dedup();
            let board = candidate_examples
                .get(&key)
                .map(|entry| *entry)
                .unwrap_or_else(|| key.to_board());
            Candidate { key, board, pairs }
        })
        .collect();
    candidates.sort_by(|a, b| a.key.cmp(&b.key));

    let pair_sets: Vec<Vec<u32>> = candidates
        .iter()
        .map(|candidate| candidate.pairs.clone())
        .collect();

    println!();
    println!("=== Greedy cover ===");
    let cover_start = Instant::now();
    let cover = greedy_cover(&pair_sets, total_pairs);
    let cover_time = cover_start.elapsed().as_secs_f64();

    println!("selected       = {}", cover.selected.len());
    println!("reverse_pruned = {}", cover.pruned);
    println!("uncovered      = {}", cover.uncovered);
    println!("cover_time     = {cover_time:.3}s");

    println!();
    println!("--- top selected boards ---");
    for (rank, (idx, new_count)) in cover.selected.iter().take(30).enumerate() {
        let board = candidates[*idx].board;
        println!(
            "  [{rank:>2}] +{new_count:<6} total_hits={:<6} h={} c={} holes={} rough={}",
            candidates[*idx].pairs.len(),
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness(),
        );
    }

    if cover.uncovered == 0 {
        save_cover(&args, &candidates, &cover)?;
        println!();
        println!(
            "saved {} selected targets to chain_step {}",
            cover.selected.len(),
            args.target_step
        );
    } else {
        println!();
        println!("not saving incomplete cover");
    }

    println!();
    println!("total_time     = {:.3}s", start.elapsed().as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_cover_finds_complete_synthetic_cover() {
        let pair_sets = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3]];
        let cover = greedy_cover(&pair_sets, 4);

        assert_eq!(cover.uncovered, 0);
        assert!(cover.selected.len() <= 2);
    }

    #[test]
    fn greedy_cover_reports_uncovered_pairs() {
        let pair_sets = vec![vec![0], vec![2]];
        let cover = greedy_cover(&pair_sets, 4);

        assert_eq!(cover.uncovered, 2);
    }

    #[test]
    fn greedy_cover_prunes_redundant_late_selection() {
        let pair_sets = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2]];
        let mut selected = vec![(0, 2), (1, 1), (2, 0)];
        reverse_prune_cover(&pair_sets, &mut selected, 3);

        assert_eq!(selected, vec![(2, 0)]);
    }
}
