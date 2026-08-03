#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod compiled;
mod config;
mod distill;
mod encode;
mod evolve;
mod forest;
mod policy;
mod puct;
mod train;
mod tree;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::config::{EvolveConfig, TrainConfig};
use crate::forest::DecisionForest;
use crate::policy::play_game;
use crate::train::{TrainArtifacts, fitness, train};

#[derive(Parser)]
#[command(name = "tetris_decision_forest")]
#[command(
    about = "Evolvable decision forest (boolean-op trees, placement votes) that plays Tetris"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evolve a population of forests up the piece-cap curriculum (fitness = survival).
    Train(TrainArgs),
    /// Distill a PUCT teacher (root search with the forest as prior) into a forest.
    Distill(DistillArgs),
    /// Play one game with a saved forest on a given seed.
    Play(PlayArgs),
    /// Evaluate a saved (or random) forest across N seeds.
    Eval(EvalArgs),
    /// Stress-test a saved forest at the 1,000,000-piece cap.
    Bench(BenchArgs),
    /// Print structural stats of a saved forest.
    Show(ShowArgs),
}

#[derive(clap::Args)]
struct TrainArgs {
    #[arg(long, default_value = "df-run")]
    run_name: String,
    #[arg(long, default_value_t = 64)]
    population: usize,
    #[arg(long, default_value_t = 1000)]
    generations: usize,
    #[arg(long, default_value_t = 8)]
    elite: usize,
    #[arg(long, default_value_t = 8)]
    seeds: usize,
    /// Freeze the eval-seed set for this many generations before rotating
    /// (stationary objective within a block; no overfitting across blocks).
    #[arg(long, default_value_t = 20)]
    seed_block: usize,
    #[arg(long, default_value_t = 5)]
    tournament_k: usize,
    #[arg(long, default_value_t = 64)]
    init_trees: usize,
    /// Depth for randomly GENERATED trees (small; trees grow via evolution).
    #[arg(long, default_value_t = 6)]
    init_tree_depth: u32,
    /// Structural cap on decision-tree depth (deep trees).
    #[arg(long, default_value_t = 24)]
    max_tree_depth: u32,
    /// Max boolean-condition depth per branch (wide conditions).
    #[arg(long, default_value_t = 6)]
    max_cond_depth: u32,
    /// Max trees per forest (many trees).
    #[arg(long, default_value_t = 256)]
    max_trees: usize,
    /// Regrowth floor: offspring are topped back up with fresh piece-gated
    /// trees to at least this many after the autokill (0 disables).
    #[arg(long, default_value_t = 32)]
    min_trees: usize,
    /// Autokill: cull trees whose valid-output rate is below this (1.0 = cull any invalid; 0 disables).
    #[arg(long, default_value_t = 1.0)]
    min_valid_rate: f64,
    /// Print a progress line every N generations (1 = every generation).
    #[arg(long, default_value_t = 1)]
    log_every: usize,
    /// Save a champion snapshot every N generations (omit to disable — the
    /// synchronous write pauses training; the champion is saved on every
    /// improvement regardless).
    #[arg(long)]
    snapshot_every: Option<usize>,
    #[arg(long, default_value_t = 200)]
    start_cap: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "./artifacts/decision_forest")]
    out_dir: PathBuf,
    /// Resume by seeding the population from this champion JSON.
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(clap::Args)]
struct DistillArgs {
    #[arg(long, default_value = "df-run")]
    run_name: String,
    /// Prior forest (usually the survival champion). The teacher searches
    /// with this forest's votes as the PUCT prior AND its policy for
    /// rollouts; omit to start from a random prior.
    #[arg(long)]
    prior: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    population: usize,
    #[arg(long, default_value_t = 500)]
    generations: usize,
    #[arg(long, default_value_t = 8)]
    elite: usize,
    #[arg(long, default_value_t = 5)]
    tournament_k: usize,
    /// Teacher-labeled states in the distillation dataset.
    #[arg(long, default_value_t = 512)]
    states: usize,
    /// Game seeds used to sample states with the prior forest.
    #[arg(long, default_value_t = 8)]
    game_seeds: usize,
    /// PUCT simulations per labeled state (must be >= 1; with zero sims the
    /// "label" would be an evidence-free lowest-index pick).
    #[arg(long, default_value_t = 192, value_parser = clap::value_parser!(u32).range(1..))]
    sims: u32,
    /// Rollout horizon (pieces) for the PUCT value estimate.
    #[arg(long, default_value_t = 24)]
    rollout_depth: u32,
    /// PUCT exploration constant.
    #[arg(long, default_value_t = 1.5)]
    c_puct: f32,
    /// Keep a disagreeing teacher label only if its Q beats the prior move's
    /// Q by this much (filters arbitrary labels from flat-Q states; 0 = off).
    #[arg(long, default_value_t = 0.0)]
    min_q_gap: f32,
    #[arg(long, default_value_t = 1)]
    log_every: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "./artifacts/decision_forest")]
    out_dir: PathBuf,
}

#[derive(clap::Args)]
struct PlayArgs {
    #[arg(long)]
    forest: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1_000_000)]
    max_pieces: u64,
}

#[derive(clap::Args)]
struct EvalArgs {
    /// Forest JSON; if omitted, a fixed-seed random forest is used.
    #[arg(long)]
    forest: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    seeds: usize,
    #[arg(long, default_value_t = 100_000)]
    max_pieces: u64,
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long)]
    forest: PathBuf,
    #[arg(long, default_value_t = 16)]
    seeds: usize,
}

#[derive(clap::Args)]
struct ShowArgs {
    #[arg(long)]
    forest: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => run_train(args),
        Command::Distill(args) => run_distill(args),
        Command::Play(args) => run_play(args),
        Command::Eval(args) => run_eval(args),
        Command::Bench(args) => run_bench(args),
        Command::Show(args) => run_show(args),
    }
}

fn run_train(args: TrainArgs) -> Result<()> {
    let evolve = EvolveConfig {
        init_tree_depth: args.init_tree_depth,
        max_tree_depth: args.max_tree_depth,
        max_cond_depth: args.max_cond_depth,
        max_trees: args.max_trees,
        min_trees: args.min_trees,
        min_valid_rate: args.min_valid_rate,
        ..EvolveConfig::default()
    };
    let cfg = TrainConfig {
        population: args.population,
        generations: args.generations,
        elite: args.elite,
        tournament_k: args.tournament_k,
        seeds: args.seeds,
        seed_block: args.seed_block,
        start_cap: args.start_cap,
        init_trees: args.init_trees,
        log_every: args.log_every,
        snapshot_every: args.snapshot_every,
        evolve,
    };
    let artifacts = TrainArtifacts {
        out_dir: args.out_dir,
        run_name: args.run_name,
    };
    let resume = match args.resume {
        Some(path) => Some(DecisionForest::load_json(&path)?),
        None => None,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        eprintln!("\n[ctrl-c] finishing current generation, then stopping…");
        stop_handler.store(true, Ordering::Relaxed);
    })
    .ok();

    let champion = train(&cfg, &artifacts, args.seed, resume, &stop)?;
    println!(
        "training complete: champion has {} trees, {} nodes",
        champion.trees.len(),
        champion.total_nodes()
    );
    Ok(())
}

fn run_distill(args: DistillArgs) -> Result<()> {
    use crate::distill::{DistillConfig, distill};
    use crate::puct::PuctConfig;
    let cfg = DistillConfig {
        population: args.population,
        generations: args.generations,
        elite: args.elite,
        tournament_k: args.tournament_k,
        states: args.states,
        game_seeds: args.game_seeds,
        min_q_gap: args.min_q_gap,
        log_every: args.log_every,
        evolve: EvolveConfig::default(),
        puct: PuctConfig {
            sims: args.sims,
            rollout_depth: args.rollout_depth,
            c_puct: args.c_puct,
        },
    };
    let artifacts = TrainArtifacts {
        out_dir: args.out_dir.clone(),
        run_name: args.run_name.clone(),
    };
    let prior = match args.prior {
        Some(path) => Some(DecisionForest::load_json(&path)?),
        None => None,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        eprintln!("\n[ctrl-c] finishing current generation, then stopping…");
        stop_handler.store(true, Ordering::Relaxed);
    })
    .ok();

    let champion = distill(&cfg, &artifacts, args.seed, prior, &stop)?;
    let distilled = args
        .out_dir
        .join(format!("{}_distilled.json", args.run_name));
    println!(
        "distilled champion: {} trees, {} nodes\ncontinue survival training with:\n  train --resume {}",
        champion.trees.len(),
        champion.total_nodes(),
        distilled.display()
    );
    Ok(())
}

fn run_play(args: PlayArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    let outcome = play_game(&forest, args.seed, args.max_pieces);
    println!(
        "seed {} | pieces {} | lines {} | max_height {} | {}",
        args.seed,
        outcome.pieces_placed,
        outcome.lines_cleared,
        outcome.max_height,
        if outcome.alive {
            "ALIVE (cap reached)"
        } else {
            "TOPPED OUT"
        }
    );
    Ok(())
}

fn run_eval(args: EvalArgs) -> Result<()> {
    let forest = match args.forest {
        Some(path) => DecisionForest::load_json(&path)?,
        None => {
            use rand::SeedableRng;
            use rand::rngs::SmallRng;
            let mut rng = SmallRng::seed_from_u64(0);
            DecisionForest::random(&mut rng, 64, 6, 6)
        }
    };
    let seeds: Vec<u64> = (0..args.seeds as u64).map(|i| i + 1).collect();
    let t0 = std::time::Instant::now();
    let r = fitness(&forest, &seeds, args.max_pieces);
    let pps = r.mean_pieces * args.seeds as f32 / t0.elapsed().as_secs_f32().max(1e-9);
    println!(
        "eval over {} seeds @ cap {} | survival mean {:.1} min {} max {} | mean_lines {:.1} | {:.0} pieces/sec",
        args.seeds, args.max_pieces, r.mean_pieces, r.min_pieces, r.max_pieces, r.mean_lines, pps
    );
    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    let seeds: Vec<u64> = (0..args.seeds as u64).map(|i| i + 1).collect();
    let t0 = std::time::Instant::now();
    let r = fitness(&forest, &seeds, 1_000_000);
    let pps = r.mean_pieces * args.seeds as f32 / t0.elapsed().as_secs_f32().max(1e-9);
    let cleared_all = r.min_pieces >= 1_000_000;
    println!(
        "BENCH @ 1,000,000 cap over {} seeds | survival mean {:.0} min {} max {} | mean_lines {:.0} | {:.0} pieces/sec | {}",
        args.seeds,
        r.mean_pieces,
        r.min_pieces,
        r.max_pieces,
        r.mean_lines,
        pps,
        if cleared_all {
            "*** GOAL REACHED: survived 1M on every seed ***"
        } else {
            "not yet 1M on all seeds"
        }
    );
    Ok(())
}

fn run_show(args: ShowArgs) -> Result<()> {
    let forest = DecisionForest::load_json(&args.forest)?;
    println!(
        "forest: {} trees, {} total nodes",
        forest.trees.len(),
        forest.total_nodes()
    );
    for (i, t) in forest.trees.iter().enumerate() {
        println!("  tree {i}: {} nodes, depth {}", t.count_nodes(), t.depth());
    }
    Ok(())
}
