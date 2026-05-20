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
pub struct ProveChainArgs {
    /// Path to SQLite database containing checkpoint sets.
    #[arg(long, default_value = "artifacts/databases/bag_checkpoint.db")]
    db: PathBuf,

    /// Table containing checkpoint sets. Must have chain_step and board_blob columns.
    #[arg(long, default_value = "chain_checkpoints")]
    checkpoint_table: String,

    /// Number of checkpoint sets in the cycle.
    #[arg(long, default_value_t = 5)]
    steps: usize,

    /// Maximum board height allowed during each one-bag witness.
    #[arg(long, default_value_t = 8)]
    max_height: u32,

    /// Maximum holes allowed during each one-bag witness.
    #[arg(long, default_value_t = 6)]
    max_holes: u32,

    /// Also verify the empty board can reach S0.
    #[arg(long, default_value_t = true)]
    verify_empty_entry: bool,

    /// Skip the empty-board entry transition. Useful when checking an already
    /// discovered checkpoint prefix without redoing the expensive entry search.
    #[arg(long, default_value_t = false)]
    skip_empty_entry: bool,

    /// Write one exact replay witness per (transition, source board, bag permutation).
    #[arg(long, default_value_t = false)]
    write_witnesses: bool,

    /// Stop checking after the first observed failure. By default the command
    /// reports all failures it finds within each transition.
    #[arg(long, default_value_t = false)]
    stop_after_first_failure: bool,

    /// Verify only the open prefix S0->S1->...->S(n-1), skipping the final
    /// S(n-1)->S0 cycle-closure transition.
    #[arg(long, default_value_t = false)]
    skip_cycle_closure: bool,
}

#[derive(Clone)]
struct CheckpointSet {
    step: usize,
    boards: Vec<BoardKey>,
    targets: FxHashSet<BoardKey>,
    target_cells: Vec<u32>,
}

struct TransitionSpec<'a> {
    transition_step: i64,
    label: String,
    sources: &'a [BoardKey],
    targets: &'a FxHashSet<BoardKey>,
    target_cells: &'a [u32],
}

struct WitnessRow {
    transition_step: i64,
    source: BoardKey,
    perm_idx: usize,
    target: BoardKey,
    placements: [u8; 7],
    nodes: u64,
}

struct TransitionResult {
    ok: bool,
    ok_count: usize,
    fail_count: usize,
    nodes: u64,
    witnesses: Vec<WitnessRow>,
}

fn transition_steps(steps: usize, skip_cycle_closure: bool) -> impl Iterator<Item = usize> {
    let transition_count = if skip_cycle_closure {
        steps.saturating_sub(1)
    } else {
        steps
    };
    0..transition_count
}

fn load_checkpoint_sets(
    conn: &Connection,
    table: &str,
    steps: usize,
) -> Result<Vec<CheckpointSet>> {
    let mut sets = Vec::with_capacity(steps);

    for step in 0..steps {
        let query =
            format!("SELECT board_blob FROM {table} WHERE chain_step = ?1 ORDER BY board_blob");
        let mut stmt = conn
            .prepare(&query)
            .with_context(|| format!("preparing checkpoint load query for table {table}"))?;

        let mut rows = stmt.query(params![step as i64])?;
        let mut boards = Vec::new();
        let mut targets = FxHashSet::default();
        while let Some(row) = rows.next()? {
            let blob: Vec<u8> = row.get(0)?;
            let key = BoardKey::from_blob(&blob)
                .with_context(|| format!("decoding board_blob at checkpoint step {step}"))?;
            if targets.insert(key) {
                boards.push(key);
            }
        }

        if boards.is_empty() {
            bail!("checkpoint step {step} has no boards in table {table}");
        }

        let target_cells = target_cell_counts(targets.iter().copied());

        sets.push(CheckpointSet {
            step,
            boards,
            targets,
            target_cells,
        });
    }

    Ok(sets)
}

fn verify_transition(
    spec: &TransitionSpec<'_>,
    perms: &[[tetris_game::TetrisPiece; 7]],
    max_height: u32,
    max_holes: u32,
    write_witnesses: bool,
    stop_after_first_failure: bool,
) -> TransitionResult {
    let total_pairs = spec.sources.len() * perms.len();
    let start = Instant::now();

    let checked = AtomicUsize::new(0);
    let ok_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);
    let total_nodes = AtomicUsize::new(0);

    let source_perm_pairs: Vec<(usize, usize)> = (0..spec.sources.len())
        .flat_map(|source_idx| (0..perms.len()).map(move |perm_idx| (source_idx, perm_idx)))
        .collect();

    let rows: Vec<Option<WitnessRow>> = source_perm_pairs
        .par_iter()
        .map(|&(source_idx, perm_idx)| {
            if stop_after_first_failure && fail_count.load(Ordering::Relaxed) > 0 {
                return None;
            }

            let source = spec.sources[source_idx];
            let mut nodes = 0u64;
            let witness = find_bag_witness_with_target_cells(
                source.to_board(),
                &perms[perm_idx],
                max_height,
                max_holes,
                spec.targets,
                spec.target_cells,
                &mut nodes,
            );

            total_nodes.fetch_add(nodes as usize, Ordering::Relaxed);
            let row = if let Some((target, placements)) = witness {
                ok_count.fetch_add(1, Ordering::Relaxed);
                write_witnesses.then_some(WitnessRow {
                    transition_step: spec.transition_step,
                    source,
                    perm_idx,
                    target,
                    placements,
                    nodes,
                })
            } else {
                fail_count.fetch_add(1, Ordering::Relaxed);
                None
            };

            let done = checked.fetch_add(1, Ordering::Relaxed) + 1;
            if done == 1 || done % 5000 == 0 || done == total_pairs {
                let ok = ok_count.load(Ordering::Relaxed);
                let fail = fail_count.load(Ordering::Relaxed);
                let nodes_seen = total_nodes.load(Ordering::Relaxed);
                eprint!(
                    "\r  {} [{}/{}] ok={} fail={} nodes={:.1}M time={:.1}s",
                    spec.label,
                    done,
                    total_pairs,
                    ok,
                    fail,
                    nodes_seen as f64 / 1e6,
                    start.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
            }

            row
        })
        .collect();
    eprintln!();

    let ok = ok_count.load(Ordering::Relaxed);
    let fail = fail_count.load(Ordering::Relaxed);
    let nodes = total_nodes.load(Ordering::Relaxed) as u64;
    let witnesses = if write_witnesses {
        rows.into_iter().flatten().collect()
    } else {
        Vec::new()
    };

    TransitionResult {
        ok: fail == 0 && ok == total_pairs,
        ok_count: ok,
        fail_count: fail,
        nodes,
        witnesses,
    }
}

fn write_witness_rows(conn: &mut Connection, rows: &[WitnessRow]) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS bag_checkpoint_witnesses (
             transition_step INTEGER NOT NULL,
             source_board BLOB NOT NULL,
             perm_index INTEGER NOT NULL,
             target_board BLOB NOT NULL,
             placements BLOB NOT NULL,
             nodes INTEGER NOT NULL,
             PRIMARY KEY (transition_step, source_board, perm_index)
         );",
    )?;

    let tx = conn.unchecked_transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO bag_checkpoint_witnesses
             (transition_step, source_board, perm_index, target_board, placements, nodes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )?;
        for row in rows {
            stmt.execute(params![
                row.transition_step,
                row.source.as_bytes().as_slice(),
                row.perm_idx as i64,
                row.target.as_bytes().as_slice(),
                row.placements.as_slice(),
                row.nodes as i64,
            ])?;
        }
    }
    tx.commit()?;

    Ok(())
}

pub fn run(args: ProveChainArgs) -> Result<()> {
    if args.steps == 0 {
        bail!("--steps must be greater than zero");
    }
    if !args.db.exists() {
        bail!("Database not found at {}", args.db.display());
    }

    let mut conn = Connection::open(&args.db)?;
    let checkpoint_sets = load_checkpoint_sets(&conn, &args.checkpoint_table, args.steps)?;
    let perms = generate_permutations();

    println!("tetris_bag_checkpoint prove-chain");
    println!("db               = {}", args.db.display());
    println!("checkpoint_table = {}", args.checkpoint_table);
    println!("steps            = {}", args.steps);
    println!("permutations     = {}", perms.len());
    println!("max_height       = {}", args.max_height);
    println!("max_holes        = {}", args.max_holes);
    let verify_empty_entry = args.verify_empty_entry && !args.skip_empty_entry;

    println!("write_witnesses  = {}", args.write_witnesses);
    println!("empty_entry      = {verify_empty_entry}");
    println!("cycle_closure    = {}", !args.skip_cycle_closure);
    println!();

    println!("--- checkpoint set sizes ---");
    for set in &checkpoint_sets {
        println!("S{}: {} boards", set.step, set.boards.len());
    }
    println!();

    let mut all_ok = true;
    let mut total_ok = 0usize;
    let mut total_fail = 0usize;
    let mut total_nodes = 0u64;
    let mut witness_rows = Vec::new();

    if verify_empty_entry {
        let empty_sources = [BoardKey::from_board(&TetrisBoard::EMPTY_BOARD)];
        let spec = TransitionSpec {
            transition_step: -1,
            label: "empty->S0".to_string(),
            sources: &empty_sources,
            targets: &checkpoint_sets[0].targets,
            target_cells: &checkpoint_sets[0].target_cells,
        };
        let result = verify_transition(
            &spec,
            &perms,
            args.max_height,
            args.max_holes,
            args.write_witnesses,
            args.stop_after_first_failure,
        );
        println!(
            "empty->S0        ok={} fail={} nodes={:.1}M status={}",
            result.ok_count,
            result.fail_count,
            result.nodes as f64 / 1e6,
            if result.ok { "COMPLETE" } else { "BROKEN" }
        );
        all_ok &= result.ok;
        total_ok += result.ok_count;
        total_fail += result.fail_count;
        total_nodes += result.nodes;
        witness_rows.extend(result.witnesses);
    }

    for step in transition_steps(args.steps, args.skip_cycle_closure) {
        let next_step = (step + 1) % args.steps;
        let label = format!("S{step}->S{next_step}");
        let spec = TransitionSpec {
            transition_step: step as i64,
            label: label.clone(),
            sources: &checkpoint_sets[step].boards,
            targets: &checkpoint_sets[next_step].targets,
            target_cells: &checkpoint_sets[next_step].target_cells,
        };

        let result = verify_transition(
            &spec,
            &perms,
            args.max_height,
            args.max_holes,
            args.write_witnesses,
            args.stop_after_first_failure,
        );
        println!(
            "{:<16} ok={} fail={} nodes={:.1}M status={}",
            label,
            result.ok_count,
            result.fail_count,
            result.nodes as f64 / 1e6,
            if result.ok { "COMPLETE" } else { "BROKEN" }
        );
        all_ok &= result.ok;
        total_ok += result.ok_count;
        total_fail += result.fail_count;
        total_nodes += result.nodes;
        witness_rows.extend(result.witnesses);

        if args.stop_after_first_failure && total_fail > 0 {
            break;
        }
    }

    if args.write_witnesses {
        println!();
        println!("writing {} witness rows...", witness_rows.len());
        write_witness_rows(&mut conn, &witness_rows)?;
    }

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS bag_checkpoint_proof_meta (
             key TEXT PRIMARY KEY,
             value TEXT NOT NULL
         );",
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO bag_checkpoint_proof_meta (key, value) VALUES ('verified', ?1)",
        params![if all_ok { "true" } else { "false" }],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO bag_checkpoint_proof_meta (key, value) VALUES ('total_ok', ?1)",
        params![total_ok.to_string()],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO bag_checkpoint_proof_meta (key, value) VALUES ('total_fail', ?1)",
        params![total_fail.to_string()],
    )?;
    conn.execute(
        "INSERT OR REPLACE INTO bag_checkpoint_proof_meta (key, value) VALUES ('total_nodes', ?1)",
        params![total_nodes.to_string()],
    )?;

    println!();
    println!("--- proof result ---");
    println!("total_ok         = {}", total_ok);
    println!("total_fail       = {}", total_fail);
    println!("total_nodes      = {:.1}M", total_nodes as f64 / 1e6);
    println!("verified         = {}", all_ok);

    if all_ok {
        println!();
        println!("INFINITE PLAY PROVEN for this checkpoint chain.");
        println!(
            "Every listed source board and every bag permutation has an exact replay witness to the next checkpoint set."
        );
    } else {
        println!();
        println!("Checkpoint chain is not closed under these constraints.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_steps_include_cycle_closure_by_default() {
        assert_eq!(
            transition_steps(4, false).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn transition_steps_can_skip_cycle_closure_for_prefix_checks() {
        assert_eq!(transition_steps(4, true).collect::<Vec<_>>(), vec![0, 1, 2]);
    }
}
