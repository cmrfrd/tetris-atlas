use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use tetris_game::{IsLost, TetrisBoard, TetrisPiece, TetrisPiecePlacement};

use crate::common::*;

#[derive(Args)]
pub struct CheckCellsArgs {
    /// Path to SQLite database.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Load source boards from chain_checkpoints at this step.
    #[arg(long)]
    source_step: Option<i64>,

    /// Table to load source boards from when --source-step is not set.
    #[arg(long, default_value = "analyze_covers")]
    source_table: String,

    /// Cell count of source boards when --source-step is not set.
    #[arg(long, default_value_t = 0)]
    source_cells: u32,

    /// Target cell count(s), comma-separated.
    #[arg(long, value_delimiter = ',')]
    target_cells: Vec<u32>,

    /// Maximum board height allowed during the bag.
    #[arg(long, default_value_t = 8)]
    max_height: u32,

    /// Maximum holes allowed during the bag.
    #[arg(long, default_value_t = 6)]
    max_holes: u32,
}

fn can_reach_cell_count_memo(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    min_target: u32,
    target_cells: &[u32],
    memo: &mut FxHashSet<(BoardKey, u8)>,
    nodes: &mut u64,
) -> bool {
    if step == 7 {
        return target_cells.contains(&board.count());
    }

    let remaining = (7 - step) as u32;
    if board.count() + 4 * remaining < min_target {
        return false;
    }

    let memo_key = (board_key(&board), step as u8);
    if memo.contains(&memo_key) {
        return false;
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

        if can_reach_cell_count_memo(
            next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            min_target,
            target_cells,
            memo,
            nodes,
        ) {
            return true;
        }
    }

    memo.insert(memo_key);
    false
}

fn can_reach_cell_count(
    board: TetrisBoard,
    pieces: &[TetrisPiece; 7],
    max_height: u32,
    max_holes: u32,
    min_target: u32,
    target_cells: &[u32],
    nodes: &mut u64,
) -> bool {
    let mut memo = FxHashSet::default();
    can_reach_cell_count_memo(
        board,
        pieces,
        0,
        max_height,
        max_holes,
        min_target,
        target_cells,
        &mut memo,
        nodes,
    )
}

fn load_sources(args: &CheckCellsArgs) -> Result<Vec<TetrisBoard>> {
    if !args.db.exists() {
        bail!("Database not found at {}", args.db.display());
    }

    let conn = Connection::open(&args.db)?;
    let query = if args.source_step.is_some() {
        "SELECT board_blob FROM chain_checkpoints WHERE chain_step = ?1".to_string()
    } else {
        format!(
            "SELECT board_blob FROM {} WHERE cell_count = ?1",
            args.source_table
        )
    };
    let param = args.source_step.unwrap_or(args.source_cells as i64);
    let mut stmt = conn
        .prepare(&query)
        .with_context(|| format!("preparing source query: {query}"))?;
    let rows = stmt.query_map(params![param], |row| {
        let blob: Vec<u8> = row.get(0)?;
        Ok(blob)
    })?;

    let mut boards = Vec::new();
    for row in rows {
        let blob = row?;
        boards.push(BoardKey::from_blob(&blob)?.to_board());
    }

    if boards.is_empty() {
        bail!("no source boards loaded");
    }
    Ok(boards)
}

pub fn run(args: CheckCellsArgs) -> Result<()> {
    if args.target_cells.is_empty() {
        bail!("--target-cells requires at least one value");
    }
    let min_target = *args.target_cells.iter().min().unwrap();
    let sources = load_sources(&args)?;
    let perms = generate_permutations();
    let total_pairs = sources.len() * perms.len();

    println!("tetris_bag_checkpoint check-cells");
    println!("db             = {}", args.db.display());
    println!("source_step    = {:?}", args.source_step);
    println!("source_boards  = {}", sources.len());
    println!("target_cells   = {:?}", args.target_cells);
    println!("max_height     = {}", args.max_height);
    println!("max_holes      = {}", args.max_holes);
    println!("total_pairs    = {}", total_pairs);
    println!();

    let start = Instant::now();
    let checked = AtomicUsize::new(0);
    let ok_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);
    let total_nodes = AtomicUsize::new(0);

    let pairs: Vec<(usize, usize)> = (0..sources.len())
        .flat_map(|si| (0..perms.len()).map(move |pi| (si, pi)))
        .collect();

    let failing_sources: FxHashSet<usize> = pairs
        .par_iter()
        .filter_map(|&(si, pi)| {
            let mut nodes = 0u64;
            let reached = can_reach_cell_count(
                sources[si],
                &perms[pi],
                args.max_height,
                args.max_holes,
                min_target,
                &args.target_cells,
                &mut nodes,
            );
            total_nodes.fetch_add(nodes as usize, Ordering::Relaxed);
            if reached {
                ok_count.fetch_add(1, Ordering::Relaxed);
            } else {
                fail_count.fetch_add(1, Ordering::Relaxed);
            }

            let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 5000 == 0 || done == total_pairs {
                eprint!(
                    "\r  [{}/{}] ok={} fail={} nodes={:.1}M time={:.1}s",
                    done,
                    total_pairs,
                    ok_count.load(Ordering::Relaxed),
                    fail_count.load(Ordering::Relaxed),
                    total_nodes.load(Ordering::Relaxed) as f64 / 1e6,
                    start.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }

            (!reached).then_some(si)
        })
        .collect();
    eprintln!();

    println!();
    println!("--- check-cells result ---");
    println!("ok_pairs        = {}", ok_count.load(Ordering::Relaxed));
    println!("fail_pairs      = {}", fail_count.load(Ordering::Relaxed));
    println!("failing_sources = {}", failing_sources.len());
    println!(
        "nodes           = {:.1}M",
        total_nodes.load(Ordering::Relaxed) as f64 / 1e6
    );
    println!("time            = {:.3}s", start.elapsed().as_secs_f64());

    Ok(())
}
