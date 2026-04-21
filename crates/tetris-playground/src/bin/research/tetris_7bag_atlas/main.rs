//! # Full 7-Bag Adversarial Atlas Solver
//!
//! Solves the full 7-bag Tetris survival problem where the bag ordering is
//! adversarial: at each step, any piece remaining in the current bag could be
//! drawn. The player must survive regardless of which piece the adversary picks.
//!
//! A state `(board, bag_remaining)` is **winning** if for EVERY piece in the bag,
//! the player has at least one placement leading to another winning state.
//! A state is **losing** if there EXISTS a piece where ALL placements lead to
//! dead/losing states.
//!
//! This directly models full 7-bag Tetris with no simplification. If the root
//! state `(empty_board, full_bag)` is winning, we have a proof that Tetris is
//! survivable forever under adversarial piece selection.
//!
//! ## Algorithm
//!
//! 1. **Forward BFS**: Expand all reachable `(board, bag_remaining)` states from
//!    the empty board with a full bag, branching over all pieces in each bag.
//! 2. **AND-OR retrograde elimination** (greatest-fixed-point): Start all states
//!    alive. A state dies when any piece branch has zero alive successors (the
//!    adversary can force that piece). Propagate death backward until the winning
//!    set stabilizes.
//! 3. **Verification**: For every winning state, confirm all piece branches have
//!    valid actions to winning successors. For every losing state, confirm at
//!    least one piece branch has all successors dead.
//!
//! ## Usage
//!
//! ```sh
//! # Tight constraints (fast, small state space)
//! cargo run --release -p tetris-playground --bin tetris_7bag_atlas -- \
//!   --max-height 4 --max-holes 0 --max-roughness 8
//!
//! # Allow some holes (larger state space)
//! cargo run --release -p tetris-playground --bin tetris_7bag_atlas -- \
//!   --max-height 4 --max-holes 1 --max-roughness 8
//!
//! # With adversarial replay (first 50 steps)
//! cargo run --release -p tetris-playground --bin tetris_7bag_atlas -- \
//!   --max-height 4 --max-holes 0 --max-roughness 8 --replay-steps 50
//!
//! # Export results to JSON
//! cargo run --release -p tetris-playground --bin tetris_7bag_atlas -- \
//!   --max-height 4 --max-holes 0 --max-roughness 8 --export-path results.json
//! ```
//!
//! ## Admissibility Constraints
//!
//! | Flag | Board method | Recommended |
//! |------|-------------|-------------|
//! | `--max-height` | `board.height()` | 4 |
//! | `--max-holes` | `board.total_holes()` | 0-1 |
//! | `--max-roughness` | `board.roughness()` | 4-8 |
//! | `--max-count` | `board.count()` | - |
//! | `--max-states` | safety cap | - |
//!
//! ## Interpreting Results
//!
//! - `root_winning = true`: The empty board survives forever against any bag
//!   sequence within the admissibility constraints.
//! - `winning_frac`: Fraction of reachable states that are winning.
//! - `verified = true`: The atlas is self-consistent.

mod config;
mod demand;
mod graph;
mod retrograde;
mod state;
mod universe;
mod verifier;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use config::{BoardAdmissibility, SolverConfig};
use universe::Universe;

/// Full 7-bag adversarial winning-set solver for Tetris.
#[derive(Parser)]
#[command(name = "tetris_7bag_atlas")]
#[command(about = "Compute the winning set for full 7-bag adversarial Tetris")]
struct Cli {
    /// Maximum board height (board.height()).
    #[arg(long, default_value_t = u32::MAX)]
    max_height: u32,

    /// Maximum total holes (board.total_holes()).
    #[arg(long, default_value_t = u32::MAX)]
    max_holes: u32,

    /// Maximum surface roughness (board.roughness()).
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,

    /// Maximum filled cells (board.count()).
    #[arg(long, default_value_t = u32::MAX)]
    max_count: u32,

    /// Maximum number of states to expand before stopping (safety cap).
    #[arg(long, default_value_t = usize::MAX)]
    max_states: usize,

    /// Optional path to export JSON summary.
    #[arg(long)]
    export_path: Option<PathBuf>,

    /// After solving, replay adversarially from root for this many steps (0 = skip).
    #[arg(long, default_value_t = 0)]
    replay_steps: usize,

    /// After initial capped solve, iteratively verify the winning strategy
    /// by expanding only the states it depends on.
    #[arg(long)]
    demand_verify: bool,
}

fn main() {
    let cli = Cli::parse();
    let total_start = Instant::now();

    let config = SolverConfig {
        admissibility: BoardAdmissibility {
            max_height: cli.max_height,
            max_holes: cli.max_holes,
            max_roughness: cli.max_roughness,
            max_count: cli.max_count,
        },
        max_states: cli.max_states,
        ..SolverConfig::default()
    };

    println!("tetris_7bag_atlas");
    println!("max_height      = {}", config.admissibility.max_height);
    println!("max_holes       = {}", config.admissibility.max_holes);
    println!("max_roughness   = {}", config.admissibility.max_roughness);
    println!("max_count       = {}", config.admissibility.max_count);
    println!("max_states      = {}", config.max_states);
    println!();

    // Phase 1: Build state graph
    let build_start = Instant::now();
    let mut universe = Universe::build(&config);
    let build_time = build_start.elapsed().as_secs_f64();

    // Phase 2: AND-OR retrograde elimination
    let retro_start = Instant::now();
    let mut result = retrograde::solve(&universe);
    let retro_time = retro_start.elapsed().as_secs_f64();

    // Phase 3: Verification
    let verify_start = Instant::now();
    let verified = verifier::verify(&universe, &result);
    let verify_time = verify_start.elapsed().as_secs_f64();

    let total_time = total_start.elapsed().as_secs_f64();

    // Summary
    let root_winning = result.alive[universe.root_state_id as usize];
    let winning = result.winning_count();
    let losing = result.losing_count();
    let total_states = universe.state_count();

    println!();
    println!("--- initial solve ---");
    println!("boards          = {}", universe.boards.len());
    println!("states          = {}", total_states);
    println!("edges           = {}", universe.edge_count());
    println!("winning         = {}", winning);
    println!("losing          = {}", losing);
    println!(
        "winning_frac    = {:.4}",
        winning as f64 / total_states.max(1) as f64
    );
    println!("root_winning    = {}", root_winning);
    println!("verified        = {}", verified);
    println!("build_time      = {:.3}s", build_time);
    println!("retrograde_time = {:.3}s", retro_time);
    println!("verify_time     = {:.3}s", verify_time);
    println!("total_time      = {:.3}s", total_time);

    // Phase 4: Demand-driven verification (optional)
    if cli.demand_verify && root_winning {
        println!();
        println!("--- demand verification ---");
        let demand_start = Instant::now();
        let demand_result = demand::demand_verify(&mut universe, &mut result);
        let demand_time = demand_start.elapsed().as_secs_f64();

        match demand_result {
            demand::DemandResult::Verified { rounds, verified_states } => {
                println!("status          = VERIFIED");
                println!("rounds          = {}", rounds);
                println!("strategy_states = {}", verified_states);
                println!("universe_states = {}", universe.state_count());
                println!("demand_time     = {:.3}s", demand_time);
            }
            demand::DemandResult::Disproved { rounds } => {
                println!("status          = DISPROVED");
                println!("rounds          = {}", rounds);
                println!("universe_states = {}", universe.state_count());
                println!("demand_time     = {:.3}s", demand_time);
            }
            demand::DemandResult::RootNotWinning => {
                println!("status          = ROOT_NOT_WINNING");
            }
        }
    }

    // Recompute root_winning after potential demand verification
    let root_winning = result.alive[universe.root_state_id as usize];

    // Optional replay
    if cli.replay_steps > 0 && root_winning {
        println!();
        println!("--- adversarial replay ({} steps) ---", cli.replay_steps);
        verifier::replay_from_root(&universe, &result, cli.replay_steps);
    }

    // Optional JSON export
    if let Some(path) = &cli.export_path {
        export_json(
            path,
            &universe,
            &result,
            build_time,
            retro_time,
            verify_time,
        );
    }

    if !root_winning {
        std::process::exit(1);
    }
}

fn export_json(
    path: &PathBuf,
    universe: &Universe,
    result: &retrograde::SolveResult,
    build_time: f64,
    retro_time: f64,
    verify_time: f64,
) {
    use std::io::Write;

    let root_winning = result.alive[universe.root_state_id as usize];
    let json = format!(
        r#"{{"boards":{},"states":{},"edges":{},"winning":{},"losing":{},"root_winning":{},"build_time_s":{:.3},"retrograde_time_s":{:.3},"verify_time_s":{:.3}}}"#,
        universe.boards.len(),
        universe.state_count(),
        universe.edge_count(),
        result.winning_count(),
        result.losing_count(),
        root_winning,
        build_time,
        retro_time,
        verify_time,
    );

    match std::fs::File::create(path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(json.as_bytes()) {
                eprintln!("error writing export: {}", e);
            } else {
                eprintln!("[export] wrote {}", path.display());
            }
        }
        Err(e) => eprintln!("error creating export file: {}", e),
    }
}
