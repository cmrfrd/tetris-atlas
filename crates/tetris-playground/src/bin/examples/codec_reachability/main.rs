#![feature(generic_const_exprs)]
#![allow(
    long_running_const_eval,
    reason = "Bounded codec construction exceeds the default const evaluation threshold."
)]
//! Continuous codec reachability research. See README.md beside this file.
mod proof;
mod sampling;
mod state;
mod storage;

use anyhow::{Result, ensure};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use state::{Codec, Workload};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Parser)]
#[command(
    about = "Sample codec states and legal histories; save independently checkable impossibility findings"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Run(Run),
    Verify {
        #[arg(long)]
        output: PathBuf,
        /// Replay facts against the current engine after reviewing a source change.
        #[arg(long)]
        recheck_engine: bool,
    },
}
#[derive(Args)]
struct Run {
    #[arg(long, default_value = "artifacts/data/codec-reachability")]
    output: PathBuf,
    /// Continue only with matching source fingerprints and workload options.
    #[arg(long)]
    resume: bool,
    /// Additional batches in this invocation; zero runs until Ctrl-C.
    #[arg(long, default_value_t = 0)]
    batches: u64,
    #[arg(long, default_value_t = 0)]
    max_seconds: u64,
    #[arg(long, default_value_t = 10_000)]
    parents: usize,
    /// Episodes per mode (strict and relaxed holes) per batch.
    #[arg(long, default_value_t = 1000)]
    episodes: usize,
    #[arg(long, default_value_t = 10_000)]
    candidates: usize,
    #[arg(long, default_value_t = 128)]
    horizon: usize,
    /// Maximum recorded paths per mode per batch; observations are always counted.
    #[arg(long, default_value_t = 4)]
    witnesses: usize,
    #[arg(long, default_value = "0xb04dc0de", value_parser = seed)]
    seed: u64,
    /// Zero selects up to four available workers; explicit values allow scaling tests.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Zero disables periodic checkpoints; a final checkpoint is always written.
    #[arg(long, default_value_t = 60)]
    checkpoint_seconds: u64,
    /// Upper bound for all four dense observation bitmaps.
    #[arg(long, default_value_t = 512)]
    max_bitmap_mib: u64,
    /// Include the 216 previously proved examples on the first batch.
    #[arg(long)]
    controls: bool,
    /// Fixed watched targets, exported with their evolving status at checkpoints.
    #[arg(long, default_value_t = 10000)]
    watch: usize,
    #[arg(long)]
    quiet: bool,
}
fn seed(value: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    let value = value.replace('_', "");
    if let Some(v) = value.strip_prefix("0x") {
        u64::from_str_radix(v, 16)
    } else {
        value.parse()
    }
}
#[derive(Serialize)]
struct Summary<'a> {
    totals: &'a state::Totals,
    invocation_seconds: f64,
    threads: usize,
    codec_board_count: u64,
    bitmap_bytes: u128,
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    let prover = proof::Prover::new()?;
    match cli.command {
        Command::Verify {
            output,
            recheck_engine,
        } => {
            let (proofs, witnesses) = storage::verify(&output, &prover, recheck_engine)?;
            println!("verified {proofs} impossibility findings and {witnesses} witnesses");
        }
        Command::Run(args) => run(args, &prover)?,
    }
    Ok(())
}
fn run(args: Run, prover: &proof::Prover) -> Result<()> {
    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(4))
    } else {
        args.threads
    };
    ensure!(
        (1..=64).contains(&threads),
        "threads must be in 1..=64, or zero for automatic"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(2 * 1024 * 1024)
        .build()?;
    let work = Workload {
        seed: args.seed,
        parents: args.parents,
        episodes: args.episodes,
        candidates: args.candidates,
        horizon: args.horizon,
        witnesses: args.witnesses,
        controls: args.controls,
        watch: args.watch,
    };
    work.validate()?;
    let bytes = u128::from(Codec::BOARD_COUNT.div_ceil(64)) * 8 * 4;
    ensure!(
        bytes <= u128::from(args.max_bitmap_mib) * 1024 * 1024,
        "codec needs {bytes} bitmap bytes, above --max-bitmap-mib"
    );
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
    let (mut store, mut state) = storage::Store::open(&args.output, work, args.resume)?;
    let start = Instant::now();
    let mut last_checkpoint = Instant::now();
    let mut completed = 0;
    while !stop.load(Ordering::Relaxed)
        && (args.batches == 0 || completed < args.batches)
        && (args.max_seconds == 0 || start.elapsed().as_secs() < args.max_seconds)
    {
        let findings = sampling::batch(&mut state, prover, &pool)?;
        store.append(&findings)?;
        completed += 1;
        if !args.quiet {
            println!(
                "{}",
                serde_json::to_string(&Summary {
                    totals: &state.header.totals,
                    invocation_seconds: start.elapsed().as_secs_f64(),
                    threads,
                    codec_board_count: Codec::BOARD_COUNT,
                    bitmap_bytes: bytes
                })?
            );
        }
        if args.checkpoint_seconds != 0
            && last_checkpoint.elapsed().as_secs() >= args.checkpoint_seconds
        {
            store.checkpoint(&state)?;
            last_checkpoint = Instant::now();
        }
    }
    state.validate()?;
    store.checkpoint(&state)?;
    let summary = Summary {
        totals: &state.header.totals,
        invocation_seconds: start.elapsed().as_secs_f64(),
        threads,
        codec_board_count: Codec::BOARD_COUNT,
        bitmap_bytes: bytes,
    };
    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(args.output.join("summary.json"), &json)?;
    println!("{json}");
    Ok(())
}
