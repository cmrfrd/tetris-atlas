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
use tetris_game::{IsLost, Major, TetrisBoard, TetrisPiece, TetrisPiecePlacement};
use tetris_search::scoring::height_mse_distance_from_empty;

use crate::common::*;

#[derive(Args)]
pub struct ConstructArgs {
    /// Path to SQLite database.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Load source boards from chain_checkpoints at this step.
    #[arg(long)]
    source_step: i64,

    /// Save constructed target boards to chain_checkpoints at this step.
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

    /// If non-zero, try a low-profile beam search of this width before
    /// falling back to exhaustive DFS.
    #[arg(long, default_value_t = 0)]
    beam_width: usize,
}

#[derive(Clone, Copy)]
struct ScoredBoard {
    board: TetrisBoard,
    score: f32,
}

fn target_ok(board: &TetrisBoard, args: &ConstructArgs) -> bool {
    if !args.target_cells.contains(&board.count()) {
        return false;
    }
    if let Some(max_h) = args.max_target_height {
        if board.height() > max_h {
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

fn find_target_board_memo(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    min_target: u32,
    args: &ConstructArgs,
    memo: &mut FxHashSet<(BoardKey, u8)>,
    nodes: &mut u64,
) -> Option<TetrisBoard> {
    if step == 7 {
        return target_ok(&board, args).then_some(board);
    }

    let remaining = (7 - step) as u32;
    if board.count() + 4 * remaining < min_target {
        return None;
    }

    let memo_key = (board_key(&board), step as u8);
    if memo.contains(&memo_key) {
        return None;
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = board;
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

        if let Some(target) =
            find_target_board_memo(next, pieces, step + 1, min_target, args, memo, nodes)
        {
            return Some(target);
        }
    }

    memo.insert(memo_key);
    None
}

fn find_target_board(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    min_target: u32,
    args: &ConstructArgs,
    nodes: &mut u64,
) -> Option<TetrisBoard> {
    let mut memo = FxHashSet::default();
    find_target_board_memo(board, pieces, 0, min_target, args, &mut memo, nodes)
}

fn beam_target_board(
    start: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    args: &ConstructArgs,
    nodes: &mut u64,
) -> Option<TetrisBoard> {
    let mut beam = vec![ScoredBoard {
        board: start,
        score: -height_mse_distance_from_empty(start),
    }];

    for &piece in pieces {
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

                candidates.push(ScoredBoard {
                    board: next,
                    score: -height_mse_distance_from_empty(next),
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
            return None;
        }
    }

    beam.into_iter()
        .filter(|candidate| target_ok(&candidate.board, args))
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .map(|candidate| candidate.board)
}

fn load_sources(args: &ConstructArgs) -> Result<Vec<TetrisBoard>> {
    if !args.db.exists() {
        bail!("Database not found at {}", args.db.display());
    }

    let conn = Connection::open(&args.db)?;
    let mut stmt = conn.prepare(
        "SELECT board_blob FROM chain_checkpoints WHERE chain_step = ?1 ORDER BY board_hash",
    )?;
    let rows = stmt.query_map(params![args.source_step], |row| {
        let blob: Vec<u8> = row.get(0)?;
        Ok(blob)
    })?;

    let mut boards = Vec::new();
    for row in rows {
        let blob = row?;
        boards.push(
            BoardKey::from_blob(&blob)
                .with_context(|| format!("decoding source step {}", args.source_step))?
                .to_board(),
        );
    }
    if boards.is_empty() {
        bail!("no source boards loaded for step {}", args.source_step);
    }
    Ok(boards)
}

pub fn run(args: ConstructArgs) -> Result<()> {
    if args.target_cells.is_empty() {
        bail!("--target-cells requires at least one value");
    }

    let min_target = *args.target_cells.iter().min().unwrap();
    let sources = load_sources(&args)?;
    let perms = generate_permutations();
    let total_pairs = sources.len() * perms.len();

    println!("tetris_bag_checkpoint construct");
    println!("db                = {}", args.db.display());
    println!("source_step       = {}", args.source_step);
    println!("target_step       = {}", args.target_step);
    println!("source_boards     = {}", sources.len());
    println!("target_cells      = {:?}", args.target_cells);
    println!("max_height        = {}", args.max_height);
    println!("max_holes         = {}", args.max_holes);
    println!("max_target_height = {:?}", args.max_target_height);
    println!("max_target_holes  = {:?}", args.max_target_holes);
    println!("max_target_rough  = {:?}", args.max_target_rough);
    println!("beam_width        = {}", args.beam_width);
    println!("total_pairs       = {}", total_pairs);
    println!();

    let start = Instant::now();
    let checked = AtomicUsize::new(0);
    let ok_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);
    let total_nodes = AtomicUsize::new(0);
    let target_counts: DashMap<TetrisBoard, u32> = DashMap::new();

    let pairs: Vec<(usize, usize)> = (0..sources.len())
        .flat_map(|si| (0..perms.len()).map(move |pi| (si, pi)))
        .collect();

    pairs.par_iter().for_each(|&(si, pi)| {
        let mut nodes = 0u64;
        let target = if args.beam_width == 0 {
            None
        } else {
            beam_target_board(sources[si], &perms[pi], &args, &mut nodes)
        }
        .or_else(|| find_target_board(sources[si], &perms[pi], min_target, &args, &mut nodes));
        total_nodes.fetch_add(nodes as usize, Ordering::Relaxed);
        if let Some(board) = target {
            ok_count.fetch_add(1, Ordering::Relaxed);
            target_counts
                .entry(board)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        } else {
            fail_count.fetch_add(1, Ordering::Relaxed);
        }

        let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 5000 == 0 || done == total_pairs {
            eprint!(
                "\r  [{}/{}] ok={} fail={} targets={} nodes={:.1}M time={:.1}s",
                done,
                total_pairs,
                ok_count.load(Ordering::Relaxed),
                fail_count.load(Ordering::Relaxed),
                target_counts.len(),
                total_nodes.load(Ordering::Relaxed) as f64 / 1e6,
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
    });
    eprintln!();

    if fail_count.load(Ordering::Relaxed) > 0 {
        bail!(
            "construct failed: {} source/permutation pairs had no target",
            fail_count.load(Ordering::Relaxed)
        );
    }

    let mut targets: Vec<(TetrisBoard, u32)> = target_counts
        .iter()
        .map(|entry| (*entry.key(), *entry.value()))
        .collect();
    targets.sort_by(|a, b| b.1.cmp(&a.1));

    println!();
    println!("--- construct result ---");
    println!("ok_pairs     = {}", ok_count.load(Ordering::Relaxed));
    println!("targets      = {}", targets.len());
    println!(
        "nodes        = {:.1}M",
        total_nodes.load(Ordering::Relaxed) as f64 / 1e6
    );
    println!("time         = {:.3}s", start.elapsed().as_secs_f64());
    println!();
    println!("--- top constructed targets ---");
    for (idx, (board, count)) in targets.iter().take(20).enumerate() {
        println!(
            "  [{idx:>2}] hits={count:<6} h={} c={} holes={} rough={}",
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness()
        );
    }

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
        for (board, _) in &targets {
            let blob = board.to_cell_array(Major::Row);
            stmt.execute(params![
                args.target_step,
                board_hash(board) as i64,
                blob.as_slice(),
                board.height(),
                board.count(),
                board.total_holes(),
                board.roughness(),
            ])?;
        }
    }
    tx.commit()?;

    println!();
    println!(
        "saved {} constructed targets to chain_step {}",
        targets.len(),
        args.target_step
    );

    Ok(())
}
