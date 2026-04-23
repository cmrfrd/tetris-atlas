//! # Fixed-Bag-Order Atlas Solver
//!
//! Solves a simplified variant of the Tetris survival problem where the bag
//! ordering is fixed: the same 7-piece permutation repeats every cycle. This
//! makes the piece sequence fully deterministic, reducing state to
//! `(board, position_in_cycle)` where position cycles through 0..6.
//!
//! ## Problem Variant
//!
//! Standard Tetris uses a 7-piece bag that reshuffles each cycle (5040 possible
//! orderings). This variant fixes one ordering — e.g. always O,I,S,Z,T,L,J —
//! so the game becomes fully deterministic once the player chooses placements.
//!
//! A state is **winning** if the player can survive forever from it. The solver
//! computes the exact winning set and produces an atlas: a mapping from every
//! winning state to its best placement.
//!
//! ## Algorithm
//!
//! 1. **Forward BFS**: Expand all reachable `(board, cycle_pos)` states from
//!    the empty board, pruning boards that violate admissibility constraints.
//! 2. **Retrograde elimination** (greatest-fixed-point): Start with all states
//!    alive. A state dies when every placement leads to a dead successor.
//!    Propagate death backward until the winning set stabilizes.
//! 3. **Verification**: For every winning state, confirm the best action leads
//!    to another winning state. For every losing state, confirm all actions
//!    lead to losing/dead states.
//!
//! ## Usage
//!
//! ```sh
//! # Basic run with admissibility constraints
//! cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- \
//!   --cycle O,I,S,Z,T,L,J --max-height 4 --max-holes 2 --max-roughness 8
//!
//! # Different cycle ordering
//! cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- \
//!   --cycle T,I,L,J,S,Z,O --max-height 4 --max-holes 2 --max-roughness 8
//!
//! # With replay of the winning strategy (first 50 steps)
//! cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- \
//!   --cycle O,I,S,Z,T,L,J --max-height 4 --max-holes 2 --max-roughness 8 \
//!   --replay-steps 50
//!
//! # Export results to JSON
//! cargo run --release -p tetris-playground --bin tetris_fixed_bag_atlas -- \
//!   --cycle O,I,S,Z,T,L,J --max-height 4 --max-holes 2 --max-roughness 8 \
//!   --export-path results.json
//! ```
//!
//! ## Admissibility Constraints
//!
//! The state space is infinite without bounds. Use these flags to limit it:
//!
//! | Flag | Description | Board method | Recommended |
//! |------|-------------|--------------|-------------|
//! | `--max-height` | Max column height | `board.height()` | 4-6 |
//! | `--max-holes` | Max total holes across all columns | `board.total_holes()` | 0-4 |
//! | `--max-roughness` | Max sum of adjacent column height diffs | `board.roughness()` | 4-10 |
//! | `--max-count` | Max total filled cells | `board.count()` | - |
//! | `--max-states` | Hard cap on states expanded (safety valve) | - | - |
//!
//! Tighter constraints = smaller state space = faster solve, but may exclude
//! valid survival strategies that temporarily use taller boards.
//!
//! ## Interpreting Results
//!
//! - `root_winning = true`: The empty board can survive forever under this cycle.
//! - `winning_frac`: Fraction of reachable states that are winning (higher = easier cycle).
//! - `verified = true`: The atlas is self-consistent (every winning state's action
//!   leads to another winning state).
//!
//! ## Preliminary Findings
//!
//! With `--max-height 4 --max-holes 2 --max-roughness 8`:
//! - Most cycle orderings yield ~25-27% winning fraction
//! - The empty board is typically winning (solvable forever)
//! - State spaces are ~25M states, solving in ~30s on a single core

mod config;
mod retrograde;
mod state;
mod universe;
mod verifier;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tetris_game::TetrisPiece;

use config::{BoardAdmissibility, SolverConfig};
use universe::Universe;

/// Fixed-cycle winning-set solver for Tetris with a repeating 7-piece permutation.
#[derive(Parser)]
#[command(name = "tetris_fixed_bag_atlas")]
#[command(about = "Compute the winning set for a fixed bag ordering in Tetris")]
struct Cli {
    /// The fixed cycle as 7 comma-separated piece names (O, I, S, Z, T, L, J).
    /// Must be a permutation of all 7 pieces.
    #[arg(long, value_delimiter = ',')]
    cycle: Vec<String>,

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

    /// After solving, replay from root for this many steps (0 = skip).
    #[arg(long, default_value_t = 0)]
    replay_steps: usize,
}

fn main() {
    let cli = Cli::parse();
    let total_start = Instant::now();

    // Parse and validate cycle
    let cycle = parse_cycle(&cli.cycle);
    let cycle_str = format_cycle(&cycle);

    let config = SolverConfig {
        admissibility: BoardAdmissibility {
            max_height: cli.max_height,
            max_holes: cli.max_holes,
            max_roughness: cli.max_roughness,
            max_count: cli.max_count,
        },
        cycle,
        max_states: cli.max_states,
    };

    println!("tetris_fixed_bag_atlas");
    println!("cycle           = {}", cycle_str);
    println!("max_height      = {}", config.admissibility.max_height);
    println!("max_holes       = {}", config.admissibility.max_holes);
    println!("max_roughness   = {}", config.admissibility.max_roughness);
    println!("max_count       = {}", config.admissibility.max_count);
    println!("max_states      = {}", config.max_states);
    println!();

    // Phase 1: Build state graph
    let build_start = Instant::now();
    let universe = Universe::build(&config);
    let build_time = build_start.elapsed().as_secs_f64();

    // Phase 2: Retrograde elimination
    let retro_start = Instant::now();
    let result = retrograde::solve(&universe);
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
    println!("--- results ---");
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

    // Optional replay
    if cli.replay_steps > 0 && root_winning {
        println!();
        println!("--- replay ({} steps) ---", cli.replay_steps);
        verifier::replay_from_root(&universe, &result, cli.replay_steps);
    }

    // Optional JSON export
    if let Some(path) = &cli.export_path {
        export_json(
            path,
            &universe,
            &result,
            &cycle_str,
            build_time,
            retro_time,
            verify_time,
        );
    }

    if !root_winning {
        std::process::exit(1);
    }
}

fn parse_cycle(pieces: &[String]) -> [TetrisPiece; 7] {
    if pieces.len() != 7 {
        eprintln!(
            "error: --cycle must specify exactly 7 pieces, got {}",
            pieces.len()
        );
        std::process::exit(2);
    }

    let mut cycle = [TetrisPiece::O_PIECE; 7];
    let mut seen = [false; 7];

    for (i, name) in pieces.iter().enumerate() {
        let piece = match name.to_uppercase().as_str() {
            "O" => TetrisPiece::O_PIECE,
            "I" => TetrisPiece::I_PIECE,
            "S" => TetrisPiece::S_PIECE,
            "Z" => TetrisPiece::Z_PIECE,
            "T" => TetrisPiece::T_PIECE,
            "L" => TetrisPiece::L_PIECE,
            "J" => TetrisPiece::J_PIECE,
            _ => {
                eprintln!(
                    "error: unknown piece name '{}' (expected O,I,S,Z,T,L,J)",
                    name
                );
                std::process::exit(2);
            }
        };

        let idx = piece.index() as usize;
        if seen[idx] {
            eprintln!("error: duplicate piece '{}' in cycle", name);
            std::process::exit(2);
        }
        seen[idx] = true;
        cycle[i] = piece;
    }

    cycle
}

fn format_cycle(cycle: &[TetrisPiece; 7]) -> String {
    cycle
        .iter()
        .map(|p| match *p {
            p if p == TetrisPiece::O_PIECE => "O",
            p if p == TetrisPiece::I_PIECE => "I",
            p if p == TetrisPiece::S_PIECE => "S",
            p if p == TetrisPiece::Z_PIECE => "Z",
            p if p == TetrisPiece::T_PIECE => "T",
            p if p == TetrisPiece::L_PIECE => "L",
            p if p == TetrisPiece::J_PIECE => "J",
            _ => "?",
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn export_json(
    path: &PathBuf,
    universe: &Universe,
    result: &retrograde::SolveResult,
    cycle_str: &str,
    build_time: f64,
    retro_time: f64,
    verify_time: f64,
) {
    use std::io::Write;

    let root_winning = result.alive[universe.root_state_id as usize];
    let json = format!(
        r#"{{"cycle":"{}","boards":{},"states":{},"edges":{},"winning":{},"losing":{},"root_winning":{},"build_time_s":{:.3},"retrograde_time_s":{:.3},"verify_time_s":{:.3}}}"#,
        cycle_str,
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
