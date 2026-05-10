use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args;
use rayon::prelude::*;
use rusqlite::{Connection, params};
use rustc_hash::FxHashSet;
use tetris_game::TetrisBoard;

use crate::common::*;

#[derive(Args)]
pub struct CycleArgs {
    /// Maximum board height allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum holes allowed during intermediate placements.
    #[arg(long, default_value_t = 4)]
    max_holes: u32,

    /// Path to SQLite database from `discover` command.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Table to load checkpoint boards from. Default: "boards" (universals only).
    #[arg(long, default_value = "boards")]
    source_table: String,

    /// If set, load ALL boards from source_table (ignore perm_count filter).
    #[arg(long, default_value_t = false)]
    all_boards: bool,
}

pub fn run(args: CycleArgs) -> Result<()> {
    if !args.db.exists() {
        bail!(
            "Database not found at {}. Run `discover` first.",
            args.db.display()
        );
    }

    // -----------------------------------------------------------------------
    // Load universal checkpoints from DB
    // -----------------------------------------------------------------------
    let conn = Connection::open(&args.db)?;
    let num_perms = 5040usize;

    let checkpoints: Vec<(u64, TetrisBoard)> = {
        let query = if args.all_boards {
            format!("SELECT board_hash, board_blob FROM {}", args.source_table)
        } else {
            format!(
                "SELECT board_hash, board_blob FROM {} WHERE perm_count = {}",
                args.source_table, num_perms
            )
        };
        let mut stmt = conn.prepare(&query)?;
        let rows: Vec<(u64, TetrisBoard)> = stmt
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
        rows
    };
    drop(conn);

    if checkpoints.is_empty() {
        bail!(
            "No checkpoint boards found in table '{}'. Run `discover` first.",
            args.source_table
        );
    }

    let checkpoint_hashes: FxHashSet<u64> = checkpoints.iter().map(|(h, _)| *h).collect();

    println!("tetris_bag_checkpoint cycle");
    println!("checkpoints      = {}", checkpoints.len());
    println!("max_height       = {}", args.max_height);
    println!("max_holes        = {}", args.max_holes);
    println!("db               = {}", args.db.display());
    println!();

    println!("--- checkpoint boards ---");
    for (i, (hash, board)) in checkpoints.iter().enumerate() {
        println!(
            "[{:>2}] hash={:<20} h={} c={} holes={} rough={}",
            i,
            hash,
            board.height(),
            board.count(),
            board.total_holes(),
            board.roughness(),
        );
        print_board(board);
        println!();
    }

    let perms = generate_permutations();
    let overall_start = Instant::now();

    // -----------------------------------------------------------------------
    // For each checkpoint, check if every perm can reach some checkpoint
    // -----------------------------------------------------------------------
    println!("=== Cycle verification ===");
    println!(
        "For each of {} checkpoints × {} perms, checking if any checkpoint is reachable...",
        checkpoints.len(),
        num_perms
    );
    println!();

    let total_pairs = checkpoints.len() * num_perms;
    let pairs_done = AtomicUsize::new(0);
    let pairs_ok = AtomicUsize::new(0);
    let pairs_fail = AtomicUsize::new(0);

    // Result: for each checkpoint, how many perms can reach a checkpoint?
    let results: Vec<(u64, TetrisBoard, usize, usize, u64)> = checkpoints
        .par_iter()
        .map(|(cp_hash, cp_board)| {
            let mut ok_count = 0usize;
            let mut fail_count = 0usize;
            let mut total_nodes = 0u64;

            for perm in &perms {
                let mut nodes = 0u64;
                let reached = can_reach_any(
                    *cp_board,
                    perm,
                    0,
                    args.max_height,
                    args.max_holes,
                    &checkpoint_hashes,
                    &mut nodes,
                );
                total_nodes += nodes;

                if reached {
                    ok_count += 1;
                    pairs_ok.fetch_add(1, Ordering::Relaxed);
                } else {
                    fail_count += 1;
                    pairs_fail.fetch_add(1, Ordering::Relaxed);
                }

                let done = pairs_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 5000 == 0 || done == total_pairs {
                    let ok = pairs_ok.load(Ordering::Relaxed);
                    let fail = pairs_fail.load(Ordering::Relaxed);
                    println!(
                        "[{:>8}/{}] ok={:<8} fail={:<8} time={:.1}s",
                        done,
                        total_pairs,
                        ok,
                        fail,
                        overall_start.elapsed().as_secs_f64(),
                    );
                    let _ = std::io::stdout().flush();
                }
            }

            (*cp_hash, *cp_board, ok_count, fail_count, total_nodes)
        })
        .collect();

    let elapsed = overall_start.elapsed().as_secs_f64();

    // -----------------------------------------------------------------------
    // Report results
    // -----------------------------------------------------------------------
    println!();
    println!("--- cycle results ---");

    let mut all_cycle = true;
    let mut cycle_safe_count = 0;

    for (hash, board, ok, fail, nodes) in &results {
        let status = if *fail == 0 { "CYCLE-SAFE" } else { "BROKEN" };
        if *fail == 0 {
            cycle_safe_count += 1;
        } else {
            all_cycle = false;
        }
        println!(
            "  checkpoint hash={:<20} ok={}/{} fail={} nodes={:<12} → {}",
            hash, ok, num_perms, fail, nodes, status,
        );
        if *fail > 0 {
            print_board(board);
            println!();
        }
    }

    println!();
    println!(
        "cycle_safe       = {}/{}",
        cycle_safe_count,
        checkpoints.len()
    );
    println!("time             = {elapsed:.3}s");

    if all_cycle {
        println!();
        println!("=============================================================");
        println!("  ALL {} checkpoints are CYCLE-SAFE.", checkpoints.len());
        println!("  Every bag permutation from every checkpoint reaches");
        println!("  at least one checkpoint.");
        println!();
        println!("  INFINITE PLAY IS PROVEN by bag-cycle induction:");
        println!("  1. From empty board, any bag reaches a checkpoint.");
        println!("  2. From any checkpoint, any bag reaches a checkpoint.");
        println!("  3. Therefore the game never tops out.");
        println!("=============================================================");
    } else {
        println!();
        println!(
            "Cycle is NOT closed. {} checkpoints have failing perms.",
            checkpoints.len() - cycle_safe_count
        );
    }

    // Update DB with cycle results
    let conn = Connection::open(&args.db)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS cycle_results (
             board_hash INTEGER PRIMARY KEY,
             ok_count INTEGER NOT NULL,
             fail_count INTEGER NOT NULL,
             total_nodes INTEGER NOT NULL,
             is_cycle_safe INTEGER NOT NULL
         );",
    )?;
    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO cycle_results (board_hash, ok_count, fail_count, total_nodes, is_cycle_safe)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for (hash, _, ok, fail, nodes) in &results {
            stmt.execute(params![
                *hash as i64,
                *ok as i64,
                *fail as i64,
                *nodes as i64,
                if *fail == 0 { 1i64 } else { 0i64 },
            ])?;
        }
    }
    tx.commit()?;

    conn.execute(
        "INSERT OR REPLACE INTO run_meta (key, value) VALUES ('cycle_safe_count', ?1)",
        params![cycle_safe_count.to_string()],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO run_meta (key, value) VALUES ('cycle_verified', ?1)",
        params![if all_cycle { "true" } else { "false" }],
    )?;

    Ok(())
}
