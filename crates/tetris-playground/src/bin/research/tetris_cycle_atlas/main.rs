#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Cycle-Boundary Adversarial Atlas
//!
//! Reduces adversarial 7-bag Tetris state to **boards only** by working
//! exclusively at bag-refill boundaries. A board is "safe" if for ALL possible
//! adversarial piece sequences in one complete bag cycle, the player can reach
//! another safe board. This is a greatest-fixed-point (GFP) over boards, with
//! an AND-OR inner check per board.
//!
//! ## Key advantages over full (board x bag_state) approach
//!
//! - 127x fewer states in the GFP (boards only, not board x bag_state)
//! - More permissive: intermediate boards during a bag don't need to be in the safe set
//! - Allows more generous admissibility for intermediates
//!
//! ## Algorithm
//!
//! 1. **Board Discovery**: BFS from empty board, interning all boards passing
//!    safe_set_admissibility.
//! 2. **GFP Iteration**: Initialize safe_set = all boards. For each board, run
//!    inner AND-OR check. Remove boards that fail. Repeat until convergence.
//! 3. **Verification**: Re-check every board with fresh memos.
//! 4. **Atlas**: Save safe set for online play.
//!
//! ## Usage
//!
//! ```sh
//! # Build atlas with small constraints
//! cargo run --release -p tetris-playground --bin tetris_cycle_atlas -- build \
//!   --max-height 4 --max-holes 0 --max-roughness 4 \
//!   --inter-max-height 6 --inter-max-holes 2 --inter-max-roughness 8
//!
//! # Play from saved atlas
//! cargo run --release -p tetris-playground --bin tetris_cycle_atlas -- play \
//!   --atlas-path cycle_atlas.bin
//! ```

mod atlas;
mod config;
mod gfp;
mod transition;
mod universe;
mod verifier;

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rustc_hash::FxHashMap;
use serde::Serialize;
use time::OffsetDateTime;

use config::{BoardAdmissibility, SolverConfig};
use tetris_game::{TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement};
use universe::BoardUniverse;

#[derive(Parser)]
#[command(name = "tetris_cycle_atlas")]
#[command(about = "Cycle-boundary adversarial atlas: GFP over boards at bag-refill boundaries")]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    /// Build the atlas: discover boards, run GFP, verify, and save.
    Build {
        /// Number of complete bag cycles the player gets to return to a safe board.
        /// Higher values give the player more room but increase search cost exponentially.
        #[arg(long, default_value_t = 1)]
        bag_cycles: u32,

        /// Maximum board height for safe set.
        #[arg(long, default_value_t = u32::MAX)]
        max_height: u32,

        /// Maximum total holes for safe set.
        #[arg(long, default_value_t = u32::MAX)]
        max_holes: u32,

        /// Maximum surface roughness for safe set.
        #[arg(long, default_value_t = u32::MAX)]
        max_roughness: u32,

        /// Maximum filled cells for safe set.
        #[arg(long, default_value_t = u32::MAX)]
        max_count: u32,

        /// Maximum board height for intermediate boards during AND-OR search.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_height: u32,

        /// Maximum total holes for intermediate boards.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_holes: u32,

        /// Maximum surface roughness for intermediate boards.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_roughness: u32,

        /// Maximum filled cells for intermediate boards.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_count: u32,

        /// Safety cap on board discovery.
        #[arg(long, default_value_t = usize::MAX)]
        max_boards: usize,

        /// Safety cap on universe plus intermediate boards during GFP inner search.
        #[arg(long, default_value_t = usize::MAX)]
        max_registry_boards: usize,

        /// Path to save the atlas artifact.
        #[arg(long, default_value = "cycle_atlas.bin")]
        atlas_path: PathBuf,

        /// Skip verification step.
        #[arg(long)]
        no_verify: bool,

        /// Write a JSON run summary to this path.
        #[arg(long)]
        summary_json: Option<PathBuf>,

        /// Do not write a JSON run summary.
        #[arg(long)]
        no_summary: bool,
    },

    /// Play from a saved atlas using online AND-OR search each bag cycle.
    Play {
        /// Path to the atlas artifact.
        #[arg(long, default_value = "cycle_atlas.bin")]
        atlas_path: PathBuf,

        /// Seconds between moves.
        #[arg(long, default_value_t = 1.0)]
        delay: f64,

        /// Maximum board height for intermediate boards during play.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_height: u32,

        /// Maximum total holes for intermediate boards during play.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_holes: u32,

        /// Maximum surface roughness for intermediate boards during play.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_roughness: u32,

        /// Maximum filled cells for intermediate boards during play.
        #[arg(long, default_value_t = u32::MAX)]
        inter_max_count: u32,

        /// Safety cap on universe plus intermediate boards during online play.
        #[arg(long, default_value_t = usize::MAX)]
        max_registry_boards: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        CliCommand::Build {
            bag_cycles,
            max_height,
            max_holes,
            max_roughness,
            max_count,
            inter_max_height,
            inter_max_holes,
            inter_max_roughness,
            inter_max_count,
            max_boards,
            max_registry_boards,
            atlas_path,
            no_verify,
            summary_json,
            no_summary,
        } => run_build(
            bag_cycles,
            max_height,
            max_holes,
            max_roughness,
            max_count,
            inter_max_height,
            inter_max_holes,
            inter_max_roughness,
            inter_max_count,
            max_boards,
            max_registry_boards,
            &atlas_path,
            no_verify,
            summary_json.as_ref(),
            no_summary,
        ),
        CliCommand::Play {
            atlas_path,
            delay,
            inter_max_height,
            inter_max_holes,
            inter_max_roughness,
            inter_max_count,
            max_registry_boards,
        } => run_play(
            &atlas_path,
            delay,
            inter_max_height,
            inter_max_holes,
            inter_max_roughness,
            inter_max_count,
            max_registry_boards,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_build(
    bag_cycles: u32,
    max_height: u32,
    max_holes: u32,
    max_roughness: u32,
    max_count: u32,
    inter_max_height: u32,
    inter_max_holes: u32,
    inter_max_roughness: u32,
    inter_max_count: u32,
    max_boards: usize,
    max_registry_boards: usize,
    atlas_path: &PathBuf,
    no_verify: bool,
    summary_json: Option<&PathBuf>,
    no_summary: bool,
) -> Result<()> {
    let total_start = Instant::now();

    let config = SolverConfig {
        safe_set_admissibility: BoardAdmissibility {
            max_height,
            max_holes,
            max_roughness,
            max_count,
        },
        admissibility: BoardAdmissibility {
            max_height: inter_max_height,
            max_holes: inter_max_holes,
            max_roughness: inter_max_roughness,
            max_count: inter_max_count,
        },
        bag_cycles,
        max_boards,
        max_registry_boards,
    };

    println!("tetris_cycle_atlas build");
    println!("bag_cycles      = {}", bag_cycles);
    println!(
        "safe_set        = height<={} holes<={} roughness<={} count<={}",
        max_height, max_holes, max_roughness, max_count
    );
    println!(
        "intermediate    = height<={} holes<={} roughness<={} count<={}",
        inter_max_height, inter_max_holes, inter_max_roughness, inter_max_count
    );
    println!("max_boards      = {}", max_boards);
    println!("max_registry    = {}", max_registry_boards);
    println!();

    // Phase 1: Board Discovery
    let discover_start = Instant::now();
    let universe = BoardUniverse::discover(&config);
    let discover_time = discover_start.elapsed().as_secs_f64();

    println!("--- phase 1: discovery ---");
    println!("boards          = {}", universe.board_count());
    println!("discover_time   = {:.3}s", discover_time);
    println!();

    // Phase 2: GFP Iteration
    let gfp_start = Instant::now();
    let gfp_result = gfp::run_gfp(&universe, &config);
    let gfp_time = gfp_start.elapsed().as_secs_f64();

    let empty_in_safe = gfp_result.safe_set.contains(&universe.empty_board_id);

    println!("--- phase 2: GFP ---");
    println!("iterations      = {}", gfp_result.iterations);
    println!("safe_set        = {}", gfp_result.safe_set.len());
    println!("total_removed   = {}", gfp_result.total_removed);
    println!("empty_in_safe   = {}", empty_in_safe);
    println!(
        "safe_fraction   = {:.4}",
        gfp_result.safe_set.len() as f64 / universe.board_count().max(1) as f64
    );
    println!("transition_cache= {}", gfp_result.transition_cache_entries);
    println!("registry_cap_hits={}", gfp_result.registry_cap_hits);
    println!("gfp_time        = {:.3}s", gfp_time);
    println!();

    // Phase 3: Verification
    let mut verify_time = 0.0;
    let mut verified = true;
    if !no_verify && !gfp_result.safe_set.is_empty() {
        let verify_start = Instant::now();
        let vr = verifier::verify_safe_set(&universe, &gfp_result.safe_set, &config);
        verify_time = verify_start.elapsed().as_secs_f64();
        verified = vr.is_clean();

        println!("--- phase 3: verification ---");
        println!("boards_checked  = {}", vr.boards_checked);
        println!("boards_passed   = {}", vr.boards_passed);
        println!("boards_failed   = {}", vr.boards_failed);
        println!("verified        = {}", verified);
        println!("verify_time     = {:.3}s", verify_time);
        println!();
    }

    let total_time = total_start.elapsed().as_secs_f64();

    // Phase 4: Save atlas
    if !gfp_result.safe_set.is_empty() && verified {
        let cycle_atlas = atlas::extract(&universe.boards, &gfp_result.safe_set, config.bag_cycles);
        match atlas::save(&cycle_atlas, &universe.boards, atlas_path) {
            Ok(()) => {
                let file_size = std::fs::metadata(atlas_path).map(|m| m.len()).unwrap_or(0);
                println!(
                    "atlas saved to {} ({} safe boards, {:.1} KB)",
                    atlas_path.display(),
                    gfp_result.safe_set.len(),
                    file_size as f64 / 1024.0,
                );
            }
            Err(e) => eprintln!("error saving atlas: {e}"),
        }
    } else if gfp_result.safe_set.is_empty() {
        eprintln!("safe set is empty -- no atlas to save");
    } else {
        eprintln!("verification failed -- atlas not saved");
    }

    println!();
    println!("--- summary ---");
    println!("total_time      = {:.3}s", total_time);

    if empty_in_safe && verified {
        println!();
        println!("The empty board is in the safe set.");
        println!("This proves infinite play: from the empty board, for ANY adversarial");
        println!("7-bag piece ordering, the player can always reach another safe board.");
    } else if !gfp_result.safe_set.is_empty() && verified {
        println!();
        println!("Safe set is non-empty but the empty board is NOT in it.");
        println!("Infinite play is proven for states in the safe set, but reachability");
        println!("from the empty board is not established.");
    } else {
        println!();
        println!("GFP converged to empty safe set. No winning strategy found.");
        println!(
            "The adversary can force death within {} bag cycle(s) from every board",
            bag_cycles
        );
        println!("under these constraints.");
    }

    // JSON summary
    if !no_summary {
        let path = write_summary(
            summary_json,
            &config,
            &universe,
            &gfp_result,
            verified,
            discover_time,
            gfp_time,
            verify_time,
            total_time,
        )?;
        println!("summary_json    = {}", path.display());
    }

    if !verified {
        std::process::exit(1);
    }
    if gfp_result.safe_set.is_empty() {
        std::process::exit(1);
    }

    Ok(())
}

fn run_play(
    atlas_path: &PathBuf,
    delay: f64,
    inter_max_height: u32,
    inter_max_holes: u32,
    inter_max_roughness: u32,
    inter_max_count: u32,
    max_registry_boards: usize,
) -> Result<()> {
    let cycle_atlas = atlas::load(atlas_path)
        .with_context(|| format!("loading atlas from {}", atlas_path.display()))?;

    let inter_admissibility = BoardAdmissibility {
        max_height: inter_max_height,
        max_holes: inter_max_holes,
        max_roughness: inter_max_roughness,
        max_count: inter_max_count,
    };

    println!(
        "Loaded atlas: {} boards, {} safe boards",
        cycle_atlas.boards.len(),
        cycle_atlas.safe_board_ids.len(),
    );
    println!("atlas_bag_cycles={}", cycle_atlas.bag_cycles);
    println!("Playing with delay={delay:.1}s");
    println!();

    let mut registry = PlayRegistry::new(&cycle_atlas.boards, max_registry_boards);

    let sleep_dur = std::time::Duration::from_secs_f64(delay);
    let mut board = tetris_game::TetrisBoard::EMPTY_BOARD;
    let mut step: u64 = 0;
    let mut total_lines: u64 = 0;
    let mut transitions = transition::TransitionCache::new(inter_admissibility);

    loop {
        // Clear screen and display
        print!("\x1b[2J\x1b[H");
        println!("=== Cycle-Boundary Atlas Player ===");
        println!("step={:<8} lines={}", step, total_lines);
        println!("{board}");

        // Check if current board is in safe set
        let Some(board_id) = registry.get_id(&board) else {
            println!("Current board not in universe -- game over!");
            println!("Survived {step} steps, cleared {total_lines} lines.");
            break;
        };
        if !cycle_atlas.safe_board_ids.contains(&board_id) {
            println!("Current board not in safe set -- game over!");
            println!("Survived {step} steps, cleared {total_lines} lines.");
            break;
        }

        // Run online AND-OR for one bag cycle to find placements
        let plan = solve_canonical_plan(
            board_id,
            cycle_atlas.bag_cycles,
            &mut registry,
            &cycle_atlas.safe_board_ids,
            &mut transitions,
        );

        let Some(placements) = plan else {
            println!("AND-OR search failed to find a plan -- game over!");
            println!("Survived {step} steps, cleared {total_lines} lines.");
            break;
        };

        // Execute the 7 placements
        for placement in &placements {
            let piece = placement.piece;
            println!("  placing {piece} -> {placement}");
            let outcome = board.apply_piece_placement(*placement);
            total_lines += outcome.lines_cleared as u64;
            step += 1;
            std::thread::sleep(sleep_dur);
        }
    }

    Ok(())
}

struct PlayRegistry {
    boards: Vec<tetris_game::TetrisBoard>,
    board_to_id: FxHashMap<tetris_game::TetrisBoard, u32>,
    universe_count: u32,
    max_boards: usize,
}

impl PlayRegistry {
    fn new(boards: &[tetris_game::TetrisBoard], max_boards: usize) -> Self {
        Self {
            boards: boards.to_vec(),
            board_to_id: boards
                .iter()
                .enumerate()
                .map(|(i, &board)| (board, i as u32))
                .collect(),
            universe_count: boards.len() as u32,
            max_boards,
        }
    }

    fn get_id(&self, board: &tetris_game::TetrisBoard) -> Option<u32> {
        self.board_to_id.get(board).copied()
    }

    fn get_or_insert(&mut self, board: tetris_game::TetrisBoard) -> Option<u32> {
        if let Some(id) = self.get_id(&board) {
            return Some(id);
        }
        if self.boards.len() >= self.max_boards {
            return None;
        }
        let id = self.boards.len() as u32;
        self.boards.push(board);
        self.board_to_id.insert(board, id);
        Some(id)
    }

    fn terminal_is_safe(&self, board_id: u32, safe_set: &rustc_hash::FxHashSet<u32>) -> bool {
        board_id < self.universe_count && safe_set.contains(&board_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlayMemoKey {
    board_id: u32,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
}

/// Solve the atlas horizon online for the canonical bag order. The GFP proof is
/// adversarial; this player only needs one concrete order for display, but it
/// still must backtrack because greedy intermediate choices can dead-end.
fn solve_canonical_plan(
    start_board_id: u32,
    bag_cycles: u32,
    registry: &mut PlayRegistry,
    safe_set: &rustc_hash::FxHashSet<u32>,
    transitions: &mut transition::TransitionCache,
) -> Option<Vec<TetrisPiecePlacement>> {
    let mut memo = FxHashMap::default();
    solve_canonical_plan_inner(
        start_board_id,
        TetrisPieceBagState::new(),
        bag_cycles,
        registry,
        safe_set,
        transitions,
        &mut memo,
    )
}

fn solve_canonical_plan_inner(
    board_id: u32,
    bag: TetrisPieceBagState,
    cycles_remaining: u32,
    registry: &mut PlayRegistry,
    safe_set: &rustc_hash::FxHashSet<u32>,
    transitions: &mut transition::TransitionCache,
    memo: &mut FxHashMap<PlayMemoKey, bool>,
) -> Option<Vec<TetrisPiecePlacement>> {
    if cycles_remaining == 0 {
        return registry.terminal_is_safe(board_id, safe_set).then(Vec::new);
    }

    let key = PlayMemoKey {
        board_id,
        bag,
        cycles_remaining,
    };
    if memo.get(&key) == Some(&false) {
        return None;
    }

    let mut next_bag = bag;
    let piece = next_bag.pop();
    if piece == TetrisPiece::NULL_PIECE {
        memo.insert(key, false);
        return None;
    }
    if next_bag.is_empty() {
        next_bag.fill();
    }
    let next_cycles = if next_bag == TetrisPieceBagState::new() {
        cycles_remaining - 1
    } else {
        cycles_remaining
    };

    let board = registry.boards[board_id as usize];
    let mut successors = transitions.successors(board, piece).to_vec();
    successors.sort_unstable_by(|a, b| b.lines_cleared.cmp(&a.lines_cleared));

    for transition in successors {
        let Some(next_board_id) = registry.get_or_insert(transition.board) else {
            continue;
        };
        if let Some(mut suffix) = solve_canonical_plan_inner(
            next_board_id,
            next_bag,
            next_cycles,
            registry,
            safe_set,
            transitions,
            memo,
        ) {
            suffix.insert(0, transition.placement);
            return Some(suffix);
        }
    }

    memo.insert(key, false);
    None
}

// --- JSON Summary ---

#[derive(Serialize)]
struct RunSummary {
    solver: &'static str,
    command: Vec<String>,
    git_commit: Option<String>,
    rustc_vv: Option<String>,
    os: &'static str,
    arch: &'static str,
    cpus: usize,
    config: SolverConfig,
    discover_time_s: f64,
    gfp_time_s: f64,
    verify_time_s: f64,
    total_time_s: f64,
    boards_discovered: usize,
    safe_set_size: usize,
    gfp_iterations: u32,
    total_removed: usize,
    registry_cap_hits: u64,
    empty_board_safe: bool,
    verified: bool,
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    summary_json: Option<&PathBuf>,
    config: &SolverConfig,
    universe: &BoardUniverse,
    gfp_result: &gfp::GfpResult,
    verified: bool,
    discover_time: f64,
    gfp_time: f64,
    verify_time: f64,
    total_time: f64,
) -> Result<PathBuf> {
    let path = summary_json.cloned().unwrap_or_else(default_summary_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create summary directory {}", parent.display()))?;
    }

    let payload = RunSummary {
        solver: "tetris_cycle_atlas",
        command: std::env::args().collect(),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        rustc_vv: command_output("rustc", &["-Vv"]),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpus: num_cpus::get(),
        config: *config,
        discover_time_s: discover_time,
        gfp_time_s: gfp_time,
        verify_time_s: verify_time,
        total_time_s: total_time,
        boards_discovered: universe.board_count(),
        safe_set_size: gfp_result.safe_set.len(),
        gfp_iterations: gfp_result.iterations,
        total_removed: gfp_result.total_removed,
        registry_cap_hits: gfp_result.registry_cap_hits,
        empty_board_safe: gfp_result.safe_set.contains(&universe.empty_board_id),
        verified,
    };

    let bytes = serde_json::to_vec_pretty(&payload)?;
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write summary JSON {}", path.display()))?;
    Ok(path)
}

fn default_summary_path() -> PathBuf {
    let timestamp = OffsetDateTime::now_utc().unix_timestamp();
    PathBuf::from(format!(
        "artifacts/output/tetris_cycle_atlas_{timestamp}_summary.json"
    ))
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
