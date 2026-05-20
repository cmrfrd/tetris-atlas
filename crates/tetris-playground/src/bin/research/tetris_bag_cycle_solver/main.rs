#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Bag-Cycle Reset Solver
//!
//! Depth-bounded AND-OR search proving the player can force a configured
//! terminal target within N complete bag cycles, regardless of adversarial piece
//! ordering. For the strict empty-board target, infinite play follows by
//! chaining resets.
//!
//! ## Key constraint
//!
//! N must be a multiple of 5 for the strict empty-board target. Each piece adds
//! 4 cells; returning to empty requires clearing `28N/10 = 14N/5` complete
//! lines, which must be an integer.
//!
//! ## Algorithm
//!
//! Memoized AND-OR recursive search:
//! - **AND nodes**: adversary picks piece from bag (ALL must be survivable)
//! - **OR nodes**: player picks placement (ONE must work)
//! - **Target**: board satisfies the configured terminal target at the end of N bag cycles
//! - **Memo key**: `(board_id, bag, cycles_remaining)`
//!
//! ## Usage
//!
//! ```sh
//! # Small (fast, validate correctness)
//! cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
//!   --bag-cycles 5 --max-height 2 --max-holes 0
//!
//! # Medium
//! cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
//!   --bag-cycles 5 --max-height 3 --max-holes 0
//!
//! # Full
//! cargo run --release -p tetris-playground --bin tetris_bag_cycle_solver -- \
//!   --bag-cycles 5 --max-height 4 --max-holes 0
//! ```

mod config;
mod retrograde;
mod solver;
mod state;
mod transition;
mod verifier;

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use time::OffsetDateTime;

use config::{BoardAdmissibility, SearchAlgorithm, SolverConfig, TargetMode};
use retrograde::solve_with_retrograde;
use solver::Solver;
use tetris_game::TetrisPieceBagState;
use verifier::{PolicyVerification, verify_witness_policy};

/// Depth-bounded AND-OR search for bag-cycle reset in adversarial Tetris.
#[derive(Parser)]
#[command(name = "tetris_bag_cycle_solver")]
#[command(about = "Search for adversarial bag-cycle reset or relaxed terminal strategies")]
struct Cli {
    /// Number of complete bag cycles (must be a multiple of 5 for --target-mode empty).
    #[arg(long, default_value_t = 5)]
    bag_cycles: u32,

    /// Maximum board height (board.height()).
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Maximum total holes (board.total_holes()).
    #[arg(long, default_value_t = 0)]
    max_holes: u32,

    /// Maximum surface roughness (board.roughness()).
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,

    /// Maximum filled cells (board.count()).
    #[arg(long, default_value_t = u32::MAX)]
    max_count: u32,

    /// Search implementation to use.
    #[arg(long, value_enum, default_value_t = SearchAlgorithm::Dfs)]
    algorithm: SearchAlgorithm,

    /// Terminal target predicate.
    #[arg(long, value_enum, default_value_t = TargetMode::Empty)]
    target_mode: TargetMode,

    /// Closed-set target maximum board height.
    #[arg(long, default_value_t = 4)]
    closed_set_max_height: u32,

    /// Closed-set target maximum total holes.
    #[arg(long, default_value_t = 0)]
    closed_set_max_holes: u32,

    /// Closed-set target maximum surface roughness.
    #[arg(long, default_value_t = u32::MAX)]
    closed_set_max_roughness: u32,

    /// Closed-set target maximum filled cells.
    #[arg(long, default_value_t = u32::MAX)]
    closed_set_max_count: u32,

    /// Skip root piece-coverage diagnostics for DFS.
    #[arg(long)]
    no_diagnostics: bool,

    /// Write a JSON run summary to this path.
    #[arg(long)]
    summary_json: Option<PathBuf>,

    /// Do not write a JSON run summary.
    #[arg(long)]
    no_summary: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.target_mode == TargetMode::Empty && cli.bag_cycles % 5 != 0 {
        eprintln!(
            "error: --bag-cycles must be a multiple of 5 for --target-mode empty (got {}). \
             Each piece adds 4 cells; clearing requires 28N/10 = 14N/5 complete lines.",
            cli.bag_cycles
        );
        std::process::exit(2);
    }
    if cli.bag_cycles == 0 {
        eprintln!("error: --bag-cycles must be > 0");
        std::process::exit(2);
    }

    let config = SolverConfig {
        bag_cycles: cli.bag_cycles,
        admissibility: BoardAdmissibility {
            max_height: cli.max_height,
            max_holes: cli.max_holes,
            max_roughness: cli.max_roughness,
            max_count: cli.max_count,
        },
        target_mode: cli.target_mode,
        closed_set_admissibility: BoardAdmissibility {
            max_height: cli.closed_set_max_height,
            max_holes: cli.closed_set_max_holes,
            max_roughness: cli.closed_set_max_roughness,
            max_count: cli.closed_set_max_count,
        },
    };

    println!("tetris_bag_cycle_solver");
    println!("algorithm       = {:?}", cli.algorithm);
    println!("target_mode     = {:?}", config.target_mode);
    println!("bag_cycles      = {}", config.bag_cycles);
    println!("total_pieces    = {}", config.total_pieces());
    if config.uses_empty_reset() {
        println!("lines_required  = {}", config.bag_cycles * 14 / 5);
    }
    println!("max_height      = {}", config.admissibility.max_height);
    println!("max_holes       = {}", config.admissibility.max_holes);
    println!("max_roughness   = {}", config.admissibility.max_roughness);
    println!("max_count       = {}", config.admissibility.max_count);
    if config.target_mode == TargetMode::LowClean {
        println!(
            "closed_set      = height<={} holes<={} roughness<={} count<={}",
            config.closed_set_admissibility.max_height,
            config.closed_set_admissibility.max_holes,
            config.closed_set_admissibility.max_roughness,
            config.closed_set_admissibility.max_count
        );
    }
    println!();

    let start = Instant::now();
    let result = match cli.algorithm {
        SearchAlgorithm::Dfs => Solver::new(config, !cli.no_diagnostics).solve(),
        SearchAlgorithm::Retrograde => solve_with_retrograde(config),
    };
    let elapsed = start.elapsed().as_secs_f64();
    let verification = result.winning.then(|| {
        verify_witness_policy(
            config,
            &result.boards,
            &result.policy,
            0,
            TetrisPieceBagState::new(),
            config.bag_cycles,
        )
    });

    println!("--- result ---");
    println!(
        "winning         = {}",
        if result.winning { "YES" } else { "NO" }
    );
    println!("boards_interned = {}", result.boards_interned());
    println!("memo_entries    = {}", result.memo_entries);
    println!("nodes_explored  = {}", result.stats.nodes_explored);
    println!("memo_hits       = {}", result.stats.memo_hits);
    println!("memo_true       = {}", result.stats.memo_true);
    println!("memo_false      = {}", result.stats.memo_false);
    println!("cell_prunes     = {}", result.stats.cell_prunes);
    println!("target_hits     = {}", result.stats.target_hits);
    println!(
        "depth_reached   = {}/{}",
        result.stats.deepest_ply_reached, result.stats.total_plies
    );
    println!("lost_prunes     = {}", result.stats.transition.lost_prunes);
    println!(
        "adm_prunes      = {}",
        result.stats.transition.admissibility_prunes
    );
    println!(
        "transition_cache= {}",
        result.stats.transition_cache_entries
    );
    println!("policy_entries  = {}", result.policy.len());
    println!("root_status     = {}", result.root_progress.compact());
    if result.stats.retrograde_states > 0 {
        println!("retro_states    = {}", result.stats.retrograde_states);
        println!("retro_edges     = {}", result.stats.retrograde_edges);
        println!(
            "retro_winning   = {}",
            result.stats.retrograde_winning_states
        );
    }
    if let Some(root) = &result.root_diagnostics {
        println!(
            "root_coverage   = {}/{} ({:.2}%)",
            root.proven_pieces,
            root.total_pieces,
            root.piece_coverage * 100.0
        );
    }
    if let Some(verification) = verification {
        println!(
            "verification    = {} (states={} branches={} terminal={} terminal_target_hits={} witness_failures={} replay_failures={} target_failures={} child_failures={})",
            if verification.is_clean() {
                "clean"
            } else {
                "FAILED"
            },
            verification.states_checked,
            verification.piece_branches_checked,
            verification.terminal_branches_checked,
            verification.terminal_target_hits,
            verification.witness_failures,
            verification.replay_failures,
            verification.target_failures,
            verification.missing_child_failures,
        );
        if config.target_mode == TargetMode::Empty {
            println!(
                "verified_empty_paths = {}",
                verification.terminal_target_hits
            );
        } else {
            println!(
                "verified_target_paths= {}",
                verification.terminal_target_hits
            );
        }
    }
    println!("time            = {:.3}s", elapsed);

    if !cli.no_summary {
        let summary_path = write_summary(
            cli.summary_json.as_ref(),
            cli.algorithm,
            config,
            elapsed,
            &result,
            verification,
        )?;
        println!("summary_json    = {}", summary_path.display());
    }

    if let Some(verification) = verification {
        if !verification.is_clean() {
            eprintln!("error: witness verification failed; refusing to claim a proof");
            std::process::exit(1);
        }
    }

    if result.winning {
        println!();
        match config.target_mode {
            TargetMode::Empty => {
                println!(
                    "The player can ALWAYS return to the empty board within {} bag cycles.",
                    config.bag_cycles
                );
                println!("This proves infinite play by chaining resets.");
            }
            TargetMode::LowClean => {
                println!(
                    "The player can force a terminal board in the configured low-clean target after {} bag cycles.",
                    config.bag_cycles
                );
                println!(
                    "This is a relaxed target result; a full infinite-play proof still requires closed-set fixed-point validation."
                );
            }
        }
    } else {
        println!();
        println!(
            "No winning strategy found within {} bag cycles with these constraints.",
            config.bag_cycles
        );
        std::process::exit(1);
    }

    Ok(())
}

#[derive(Serialize)]
struct RunSummary<'a> {
    solver: &'static str,
    command: Vec<String>,
    git_commit: Option<String>,
    rustc_vv: Option<String>,
    os: &'static str,
    arch: &'static str,
    cpus: usize,
    algorithm: SearchAlgorithm,
    config: SolverConfig,
    elapsed_seconds: f64,
    nodes_per_second: f64,
    result: SummaryResult,
    stats: solver::SolveStats,
    root_diagnostics: Option<&'a solver::RootDiagnostics>,
    root_progress: solver::RootProgress,
    verification: Option<PolicyVerification>,
}

#[derive(Serialize)]
struct SummaryResult {
    winning: bool,
    boards_interned: usize,
    memo_entries: usize,
    policy_entries: usize,
}

fn write_summary(
    summary_json: Option<&PathBuf>,
    algorithm: SearchAlgorithm,
    config: SolverConfig,
    elapsed_seconds: f64,
    result: &solver::SolveResult,
    verification: Option<PolicyVerification>,
) -> Result<PathBuf> {
    let path = summary_json.cloned().unwrap_or_else(default_summary_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create summary directory {}", parent.display()))?;
    }

    let nodes_per_second = if elapsed_seconds > 0.0 {
        result.stats.nodes_explored as f64 / elapsed_seconds
    } else {
        0.0
    };

    let payload = RunSummary {
        solver: "tetris_bag_cycle_solver",
        command: std::env::args().collect(),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        rustc_vv: command_output("rustc", &["-Vv"]),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpus: num_cpus::get(),
        algorithm,
        config,
        elapsed_seconds,
        nodes_per_second,
        result: SummaryResult {
            winning: result.winning,
            boards_interned: result.boards_interned(),
            memo_entries: result.memo_entries,
            policy_entries: result.policy.len(),
        },
        stats: result.stats,
        root_diagnostics: result.root_diagnostics.as_ref(),
        root_progress: result.root_progress,
        verification,
    };

    let bytes = serde_json::to_vec_pretty(&payload)?;
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write summary JSON {}", path.display()))?;
    Ok(path)
}

fn default_summary_path() -> PathBuf {
    let timestamp = OffsetDateTime::now_utc().unix_timestamp();
    PathBuf::from(format!(
        "artifacts/output/tetris_bag_cycle_solver_{timestamp}_summary.json"
    ))
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
