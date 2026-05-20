#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Bag Checkpoint Explorer
//!
//! Two subcommands:
//!
//! - `discover`: From the empty board, find boards reachable by ALL 5040 bag
//!   permutations (universal checkpoints).
//!
//! - `cycle`: Given a set of universal checkpoints, verify that from each one,
//!   every bag permutation can reach at least one checkpoint. If true for all
//!   checkpoints, infinite play is proven by induction.
//!
//! ## Usage
//!
//! ```sh
//! # Step 1: Discover universal checkpoints from empty board
//! cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
//!   discover --max-height 4 --max-holes 4
//!
//! # Step 2: Check if checkpoints form a closed cycle
//! cargo run --release -p tetris-playground --bin tetris_bag_checkpoint -- \
//!   cycle --max-height 4 --max-holes 4
//! ```

mod analyze;
mod analyze_targeted;
mod chain;
mod chain_targeted;
mod check_cells;
mod common;
mod construct;
mod construct_cover;
mod cycle;
mod discover;
mod five_bag_cycle;
mod five_bag_mcts;
mod gfp;
mod gfp2;
mod hubs;
mod lfp;
mod proof;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tetris_bag_checkpoint")]
#[command(about = "Bag checkpoint discovery and cycle verification for infinite play proof")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Discover universal checkpoint boards reachable by all bag permutations.
    Discover(discover::DiscoverArgs),
    /// Analyze per-cell-count coverage: which cell counts can form complete checkpoint covers?
    Analyze(analyze::AnalyzeArgs),
    /// Targeted analysis: skip Phase 1 and analyze a specific cell count directly.
    AnalyzeTargeted(analyze_targeted::AnalyzeTargetedArgs),
    /// Build the residue-class chain: find cover checkpoints at each cell-count mod 10.
    Chain(chain::ChainArgs),
    /// Targeted chain step: multi-source analysis with cell-count-pruned DFS.
    ChainTargeted(chain_targeted::ChainTargetedArgs),
    /// Fast feasibility probe: check whether every source/permutation can reach
    /// one of the requested terminal cell counts.
    CheckCells(check_cells::CheckCellsArgs),
    /// Construct a checkpoint set by selecting one reachable target board for
    /// each source/permutation pair, then deduplicating exact boards.
    Construct(construct::ConstructArgs),
    /// Construct a checkpoint set by generating multiple candidates per
    /// source/permutation pair, then running exact greedy set cover.
    ConstructCover(construct_cover::ConstructCoverArgs),
    /// Verify that checkpoints form a closed cycle (every perm returns to a checkpoint).
    Cycle(cycle::CycleArgs),
    /// Non-adversarial GFP: discover boards via BFS, then iteratively prune boards
    /// where any bag permutation fails to reach the safe set.
    Gfp(gfp::GfpArgs),
    /// Two-tier GFP: inner safe set + outer buffer. Inner boards can transit to
    /// inner∪outer; outer boards must return to inner. Proves 2-bag invariant.
    Gfp2(gfp2::Gfp2Args),
    /// Constructive hub discovery: beam-play forward from empty board to find
    /// frequently-visited boards, then verify cycle closure via can_reach_any.
    Hubs(hubs::HubsArgs),
    /// Least fixed point expansion: start from empty board, grow safe set by
    /// adding boards needed for failing permutations. Opposite of GFP.
    Lfp(lfp::LfpArgs),
    /// Exact proof verifier: checks empty->S0 and S0->...->S0 checkpoint
    /// closure using board bytes plus optional replay witnesses.
    ProveChain(proof::ProveChainArgs),
    /// Five-bag cycle feasibility: can we return to the empty board after 5 bags?
    FiveBagCycle(five_bag_cycle::FiveBagCycleArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Discover(args) => discover::run(args),
        Command::Analyze(args) => analyze::run(args),
        Command::AnalyzeTargeted(args) => analyze_targeted::run(args),
        Command::Chain(args) => chain::run(args),
        Command::ChainTargeted(args) => chain_targeted::run(args),
        Command::CheckCells(args) => check_cells::run(args),
        Command::Construct(args) => construct::run(args),
        Command::ConstructCover(args) => construct_cover::run(args),
        Command::Cycle(args) => cycle::run(args),
        Command::Gfp(args) => gfp::run(args),
        Command::Gfp2(args) => gfp2::run(args),
        Command::Hubs(args) => hubs::run(args),
        Command::Lfp(args) => lfp::run(args),
        Command::ProveChain(args) => proof::run(args),
        Command::FiveBagCycle(args) => five_bag_cycle::run(args),
    }
}
