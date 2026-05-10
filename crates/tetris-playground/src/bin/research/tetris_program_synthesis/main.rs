mod cegis;
mod config;
mod encoding;
mod synthesis;
mod verification;
mod vm;
mod vm_symbolic;

use std::ffi::CString;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::Parser;
use tracing::info;

use crate::cegis::{CegisResult, run_cegis};
use crate::config::{InvariantParams, SimulationConfig};
use crate::synthesis::SynthesisOptions;

#[derive(Parser, Debug)]
#[command(name = "tetris_program_synthesis")]
#[command(about = "Strategy synthesis via bitvector VM + CEGIS")]
struct Args {
    /// Board max column height for the invariant.
    #[arg(long, default_value_t = 2)]
    max_height: u32,

    /// Max total holes allowed in the invariant.
    #[arg(long, default_value_t = 0)]
    max_holes: u32,

    /// Max adjacent column height difference.
    #[arg(long, default_value_t = 2)]
    max_roughness: u32,

    /// Number of instructions per synthesized program.
    #[arg(long, default_value_t = 12)]
    program_length: usize,

    /// Maximum CEGIS rounds.
    #[arg(long, default_value_t = 200)]
    max_rounds: usize,

    /// Number of random seeds to test per verification round.
    #[arg(long, default_value_t = 64)]
    num_seeds: u32,

    /// Number of pieces each game must survive during verification.
    #[arg(long, default_value_t = 10_000)]
    survival_target: u32,

    /// Z3 global verbosity level. 0 leaves Z3 quiet.
    #[arg(long, default_value_t = 0)]
    z3_verbose: u32,

    /// Log Z3 solver statistics after each synthesis check.
    #[arg(long)]
    log_z3_stats: bool,

    /// Dump each synthesis query as SMT-LIB2 files under this directory.
    #[arg(long)]
    dump_smt2_dir: Option<PathBuf>,

    /// Per-synthesis-query Z3 timeout in milliseconds.
    #[arg(long)]
    z3_timeout_ms: Option<u32>,

    /// Write Z3's native API/formula interaction log to this file.
    #[arg(long)]
    z3_log_file: Option<PathBuf>,

    /// Enable a Z3 internal trace tag. Requires a debug-built Z3; repeatable.
    #[arg(long)]
    z3_trace: Vec<String>,
}

struct Z3LogGuard;

impl Drop for Z3LogGuard {
    fn drop(&mut self) {
        unsafe {
            z3_sys::Z3_close_log();
        }
    }
}

fn c_string(input: &str) -> anyhow::Result<CString> {
    CString::new(input).with_context(|| format!("string contains interior NUL byte: {input:?}"))
}

fn open_z3_log(path: &Path) -> anyhow::Result<Z3LogGuard> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating Z3 log directory {}", parent.display()))?;
    }

    let path_string = path.to_string_lossy();
    let path_c = c_string(&path_string)?;
    let opened = unsafe { z3_sys::Z3_open_log(path_c.as_ptr()) };
    if !opened {
        bail!("Z3 failed to open interaction log {}", path.display());
    }

    Ok(Z3LogGuard)
}

fn enable_z3_trace_tags(tags: &[String]) -> anyhow::Result<()> {
    for tag in tags {
        let tag_c = c_string(tag)?;
        unsafe {
            z3_sys::Z3_enable_trace(tag_c.as_ptr());
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let _z3_log_guard = match &args.z3_log_file {
        Some(path) => Some(open_z3_log(path)?),
        None => None,
    };
    enable_z3_trace_tags(&args.z3_trace)?;

    let params = InvariantParams {
        max_height: args.max_height,
        max_total_holes: args.max_holes,
        max_roughness: args.max_roughness,
        min_flat_block: 0,
    };
    let simulation = SimulationConfig {
        num_seeds: args.num_seeds,
        survival_target: args.survival_target,
    };
    let synthesis_options = SynthesisOptions {
        log_z3_stats: args.log_z3_stats,
        dump_smt2_dir: args.dump_smt2_dir.clone(),
        z3_timeout_ms: args.z3_timeout_ms,
        append_z3_log_comments: args.z3_log_file.is_some(),
    };
    if args.z3_verbose > 0 {
        z3::set_global_param("verbose", &args.z3_verbose.to_string());
    }

    info!("Tetris Program Synthesis via empirical CEGIS");
    info!("  Board: 10x20");
    info!(
        "  Invariant: max_height={}, max_holes={}, max_roughness={}",
        params.max_height, params.max_total_holes, params.max_roughness
    );
    info!("  Program length: {} instructions", args.program_length);
    info!("  Max CEGIS rounds: {}", args.max_rounds);
    info!(
        "  Verification: seeds=0..{}, survival_target={} pieces",
        args.num_seeds, args.survival_target
    );
    info!(
        "  Z3: verbose={}, stats={}, timeout_ms={:?}, smt2_dir={:?}, log_file={:?}, trace_tags={:?}",
        args.z3_verbose,
        args.log_z3_stats,
        args.z3_timeout_ms,
        args.dump_smt2_dir,
        args.z3_log_file,
        args.z3_trace
    );

    match run_cegis(
        &params,
        &simulation,
        &synthesis_options,
        args.max_rounds,
        args.program_length,
    ) {
        CegisResult::Survived {
            program,
            rounds,
            examples,
        } => {
            info!(
                "SUCCESS: program survived configured empirical verification in {rounds} rounds using {examples} examples"
            );
            info!("--- Program ---");
            info!("\n{}", program);
            Ok(())
        }
        CegisResult::SynthesisFailed { round, examples } => {
            info!("FAILED: synthesis became UNSAT at round {round} with {examples} examples");
            Ok(())
        }
        CegisResult::SynthesisUnknown {
            round,
            examples,
            reason,
        } => {
            info!(
                "FAILED: synthesis returned Unknown at round {round} with {examples} examples: {:?}",
                reason
            );
            Ok(())
        }
        CegisResult::NoCounterexamples {
            round, examples, ..
        } => {
            info!(
                "FAILED: verification failed at round {round}, but yielded no invariant-safe counterexamples ({examples} examples)"
            );
            Ok(())
        }
        CegisResult::NoNewCounterexamples {
            round, examples, ..
        } => {
            info!(
                "FAILED: verification failed at round {round}, but all counterexamples were duplicates ({examples} examples)"
            );
            Ok(())
        }
        CegisResult::MaxRoundsExceeded { examples, .. } => {
            info!("FAILED: max rounds exceeded with {examples} examples");
            info!(
                "Try increasing --program-length, --max-rounds, or relaxing invariant parameters."
            );
            Ok(())
        }
    }
}
