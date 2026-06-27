#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # tetris_phase_atlas — 5-bag, phase-layered perfect-clear atlas
//!
//! This binary fuses the *forward search* idea from
//! `examples/tetris_five_bag_empty.rs` with the *frontier-atlas* idea from
//! `atlas/tetris_atlas_inmemory.rs`, organized around the structure that makes
//! 5 bags special.
//!
//! ## Why five bags
//!
//! Every tetromino adds 4 cells. A perfect clear (board returns to empty)
//! requires the total cell count to be a multiple of 10, i.e. the number of
//! pieces placed must be `≡ 0 (mod 5)`. Bag boundaries land on multiples of 7.
//! The two align only at multiples of `lcm(5,7) = 35 = 5 bags`. So after exactly
//! 5 bags you simultaneously have (a) a *fresh* bag state again and (b) the
//! geometric possibility of an empty board (exactly 14 lines cleared). The
//! `empty → 5 bags → empty` loop is therefore the minimal cycle that, if it can
//! be sustained against every adversarial bag order, proves infinite play.
//!
//! ## The phase structure
//!
//! Because the bag refills to *full* at every boundary, a state observed at a
//! bag boundary is fully described by the **board alone** (the bag is always
//! fresh there). We therefore track six phase sets of boards:
//!
//! ```text
//! P0 = { empty }                       (start)
//! P1 = boards after 1 bag on a good trajectory
//! ...
//! P4 = boards after 4 bags on a good trajectory
//! P5 = { empty }                       (== P0, closes the cycle)
//! ```
//!
//! ## v1: cooperative discovery (this file)
//!
//! For a range of fixed 5-bag piece sequences (the adversary's possible bag
//! orders, enumerated as base-5040 tuples), run a depth-35 beam search toward
//! the empty board. A tuple is a **perfect clear** if the search reaches the
//! empty board at step 35, choosing all 35 placements with full knowledge of the
//! sequence (the non-adversarial, full-lookahead model — same as
//! `tetris_nbag_solver`). The headline metric is the **cooperative PC rate**:
//! the fraction of 5-bag sequences that admit *some* perfect clear. This is a
//! beam lower bound on the true rate.
//!
//! ### Soundness note on "merging"
//!
//! It is tempting to short-circuit: if a new tuple's trajectory lands, after `k`
//! bags, on a board already in `Pk`, declare it solved. That is **only sound if
//! `Pk` is adversarially closed** — i.e. the board returns to empty for *every*
//! upcoming bag order, not just the one bag order that originally deposited it.
//! Cooperative discovery does not establish closure, so v1 does **not** count
//! cross-tuple merges as solves; it only collects the boundary boards of genuine
//! perfect clears as a *candidate carrier*. Certifying adversarial closure of
//! that carrier (where merging becomes sound) is v2's job.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPiecePlacement,
};
use tetris_search::height_mse_distance_from_empty;

// --- Cycle geometry --------------------------------------------------------

const PIECES_PER_BAG: usize = 7;
const BAG_COUNT: usize = 5;
const TOTAL_PIECES: usize = BAG_COUNT * PIECES_PER_BAG; // 35
const BAG_PERM_COUNT: usize = 5_040; // 7!
/// Lines that must clear over a 5-bag cycle to return to empty: 35*4/10 = 14.
const LINES_FOR_PC: u32 = (TOTAL_PIECES as u32 * 4) / 10;
const TOTAL_FIVE_BAG_SEQUENCES: u128 = pow_u128(BAG_PERM_COUNT as u128, BAG_COUNT as u32);

const DEFAULT_PLACEMENT: TetrisPiecePlacement = TetrisPiecePlacement::from_index(0);

type ForcedBag = [TetrisPiece; PIECES_PER_BAG];
static ALL_BAG_PERMUTATIONS: [ForcedBag; BAG_PERM_COUNT] = generate_forced_bag_permutations();

// --- CLI -------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "tetris_phase_atlas")]
#[command(about = "Phase-layered N-bag perfect-clear atlas")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand, Debug)]
enum Mode {
    /// Cooperative discovery: beam-search 5-bag tuples for perfect clears and
    /// build a candidate phase carrier. Measures the (full-lookahead) PC rate.
    Discover(DiscoverArgs),
    /// Adversarial certification: exact per-piece AND-OR check of whether the
    /// empty board can be forced back to empty after `bag_cycles` bags against
    /// every adversarial bag order. A YES is a constructive infinite-play
    /// certificate (within the height cap as the proven-safe region).
    Certify(CertifyArgs),
    /// Closed safe-set growth: grow a board-only safe set `R` at bag boundaries
    /// from {empty} until it is closed under one adversarial bag (survival, not
    /// return-to-empty). A closed R with empty in it is an infinite-play atlas.
    Closure(ClosureArgs),
}

#[derive(Args, Debug)]
struct DiscoverArgs {
    /// First 5-bag ordinal (base-5040 tuple index) to scan.
    #[arg(long, default_value_t = 0)]
    start: u128,

    /// Number of 5-bag tuples to scan.
    #[arg(long, default_value_t = 1000)]
    limit: u64,

    /// Beam width (states retained per ply).
    #[arg(long, default_value_t = 2048)]
    beam_width: usize,

    /// Reject any intermediate board taller than this.
    #[arg(long, default_value_t = StandardTetris::ROWS as u32)]
    max_height: u32,

    /// Tuples processed per parallel chunk (also the progress/CSV cadence).
    #[arg(long, default_value_t = 200)]
    chunk: u64,

    /// Directory for CSV metrics and the JSON summary.
    #[arg(long, default_value = "artifacts/output/tetris_phase_atlas")]
    out_dir: String,
}

#[derive(Args, Debug)]
struct CertifyArgs {
    /// Number of complete bags in the reset cycle. Must be a multiple of 5 so
    /// that returning to empty is cell-count feasible (28*N divisible by 10).
    #[arg(long, default_value_t = 5)]
    bag_cycles: usize,

    /// Maximum board height permitted at any intermediate placement. This is the
    /// proof's admissible region: a YES proves a reset that never exceeds it.
    #[arg(long, default_value_t = 6)]
    max_height: u32,

    /// Abort and report INCONCLUSIVE after exploring this many AND-OR nodes.
    #[arg(long, default_value_t = 200_000_000)]
    node_budget: u64,

    /// Directory for the JSON summary.
    #[arg(long, default_value = "artifacts/output/tetris_phase_atlas")]
    out_dir: String,
}

#[derive(Args, Debug)]
struct ClosureArgs {
    /// Maximum board height permitted at any placement (the admissible band).
    #[arg(long, default_value_t = 4)]
    max_height: u32,

    /// Stop and report FLOOR once the safe set exceeds this many boards.
    #[arg(long, default_value_t = 2_000_000)]
    max_boards: usize,

    /// Maximum growth rounds before giving up.
    #[arg(long, default_value_t = 1000)]
    max_rounds: u32,

    /// Use the weaker per-bag adversary with full within-bag lookahead (the
    /// player sees the whole 7-piece bag before placing, ~6-piece preview) and
    /// answers each of the 5040 bag orders. Default is the per-piece online
    /// adversary (no lookahead).
    #[arg(long, default_value_t = false)]
    bag_lookahead: bool,

    /// Beam width for the per-bag within-bag reachability search.
    #[arg(long, default_value_t = 512)]
    beam_width: usize,

    /// Directory for the JSON summary.
    #[arg(long, default_value = "artifacts/output/tetris_phase_atlas")]
    out_dir: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Discover(args) => run_discover(&args),
        Mode::Certify(args) => run_certify(&args),
        Mode::Closure(args) => run_closure(&args),
    }
}

fn run_discover(args: &DiscoverArgs) -> Result<()> {
    validate_discover(args)?;
    fs::create_dir_all(&args.out_dir)?;

    println!("=== tetris_phase_atlas: discover (cooperative) ===");
    println!("total_pieces         = {TOTAL_PIECES}");
    println!("lines_for_pc         = {LINES_FOR_PC}");
    println!("total_5bag_sequences = {TOTAL_FIVE_BAG_SEQUENCES}");
    println!("start                = {}", args.start);
    println!("limit                = {}", args.limit);
    println!("beam_width           = {}", args.beam_width);
    println!("max_height           = {}", args.max_height);
    println!("chunk                = {}", args.chunk);
    println!("out_dir              = {}", args.out_dir);
    println!("threads              = {}", rayon::current_num_threads());
    println!();

    run_scan(args)
}

fn validate_discover(args: &DiscoverArgs) -> Result<()> {
    if args.start >= TOTAL_FIVE_BAG_SEQUENCES {
        bail!("--start must be < {TOTAL_FIVE_BAG_SEQUENCES}");
    }
    if args.beam_width == 0 {
        bail!("--beam-width must be > 0");
    }
    if args.chunk == 0 {
        bail!("--chunk must be > 0");
    }
    if args.max_height == 0 || args.max_height > StandardTetris::ROWS as u32 {
        bail!("--max-height must be in 1..={}", StandardTetris::ROWS);
    }
    Ok(())
}

// --- Candidate phase carrier ----------------------------------------------

/// The candidate carrier: boards observed at bag boundaries on cooperative
/// perfect-clear trajectories. `phases[0]` and `phases[5]` are the implicit
/// empty anchor and are never stored. These are *candidates* for v2's
/// adversarial-closure certification, not a proof of anything by themselves.
struct PhaseCarrier {
    phases: [FxHashSet<TetrisBoard>; BAG_COUNT + 1],
}

impl PhaseCarrier {
    fn new() -> Self {
        Self {
            phases: std::array::from_fn(|_| FxHashSet::default()),
        }
    }

    /// Deposit a boundary board. Phases 0/5 (the empty anchor) are implicit and
    /// never stored. Returns true if newly inserted.
    #[inline]
    fn deposit(&mut self, phase: usize, board: TetrisBoard) -> bool {
        if phase == 0 || phase == BAG_COUNT {
            return false;
        }
        self.phases[phase].insert(board)
    }

    /// Deposit all interior boundary boards of a perfect-clear trajectory.
    /// Returns the number of newly inserted boards.
    fn deposit_pc(&mut self, boundaries: &[TetrisBoard; BAG_COUNT]) -> usize {
        let mut inserted = 0;
        for bag in 0..BAG_COUNT {
            if self.deposit(bag + 1, boundaries[bag]) {
                inserted += 1;
            }
        }
        inserted
    }

    fn sizes(&self) -> [usize; BAG_COUNT + 1] {
        std::array::from_fn(|p| self.phases[p].len())
    }
}

// --- Beam search over one 5-bag sequence -----------------------------------

/// A partial trajectory through the 35-piece sequence.
#[derive(Clone, Copy)]
struct BeamNode {
    board: TetrisBoard,
    /// Placement chosen for each piece index placed so far.
    placements: [TetrisPiecePlacement; TOTAL_PIECES],
    /// Board observed at each completed bag boundary (`boundaries[k]` is the
    /// board after bag `k+1`); `boundaries[4]` is the final empty board.
    boundaries: [TetrisBoard; BAG_COUNT],
}

impl BeamNode {
    fn root() -> Self {
        Self {
            board: TetrisBoard::EMPTY_BOARD,
            placements: [DEFAULT_PLACEMENT; TOTAL_PIECES],
            boundaries: [TetrisBoard::EMPTY_BOARD; BAG_COUNT],
        }
    }
}

/// Beam search a single fixed 5-bag sequence for *some* perfect clear (empty
/// board after all 35 pieces). Returns the winning trajectory or `None`.
///
/// Ranking is by `height_mse_distance_from_empty` (smaller = closer to empty).
/// A perfect-clear feasibility filter `(cells + 4*remaining) % 10 == 0` prunes
/// boards that can never reach empty, since each line clear removes 10 cells.
fn search_pc(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    beam_width: usize,
    max_height: u32,
) -> Option<BeamNode> {
    let mut beam: Vec<(f32, BeamNode)> = vec![(f32::INFINITY, BeamNode::root())];
    let mut next: Vec<(f32, BeamNode)> = Vec::new();
    let mut seen: FxHashSet<TetrisBoard> = FxHashSet::default();

    for step in 0..TOTAL_PIECES {
        let piece = pieces[step];
        let placements = TetrisPiecePlacement::all_from_piece(piece);
        let is_boundary = (step + 1) % PIECES_PER_BAG == 0;
        let phase = (step + 1) / PIECES_PER_BAG; // 1..=5 when is_boundary
        let remaining = (TOTAL_PIECES - (step + 1)) as u32;

        next.clear();
        seen.clear();

        for (_, node) in &beam {
            for &placement in placements {
                let mut board = node.board;
                let result = board.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST || board.height() > max_height {
                    continue;
                }
                // Perfect-clear feasibility: a future empty board requires the
                // running cell count plus all future cells to be a multiple of
                // 10 (each line clear removes exactly 10 cells).
                if (board.count() + 4 * remaining) % 10 != 0 {
                    continue;
                }

                let mut child = *node;
                child.board = board;
                child.placements[step] = placement;
                if is_boundary {
                    child.boundaries[phase - 1] = board;
                    if phase == BAG_COUNT {
                        // Final boundary: only an empty board is a perfect clear.
                        if board == TetrisBoard::EMPTY_BOARD {
                            return Some(child);
                        }
                        continue;
                    }
                }

                if seen.insert(board) {
                    let distance = height_mse_distance_from_empty(board);
                    next.push((distance, child));
                }
            }
        }

        if next.is_empty() {
            return None;
        }
        if next.len() > beam_width {
            next.select_nth_unstable_by(beam_width, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            next.truncate(beam_width);
        }
        std::mem::swap(&mut beam, &mut next);
    }

    None
}

/// Replay a placement list from empty and confirm it is a genuine perfect clear:
/// each placement matches the dealt piece, nothing tops out, the final board is
/// empty, and exactly `LINES_FOR_PC` lines cleared.
fn verify_perfect_clear(
    pieces: &[TetrisPiece; TOTAL_PIECES],
    placements: &[TetrisPiecePlacement; TOTAL_PIECES],
    max_height: u32,
) -> bool {
    let mut board: TetrisBoard = TetrisBoard::EMPTY_BOARD;
    let mut lines = 0u32;
    for step in 0..TOTAL_PIECES {
        if placements[step].piece != pieces[step] {
            return false;
        }
        let result = board.apply_piece_placement(placements[step]);
        if result.is_lost == IsLost::LOST || board.height() > max_height {
            return false;
        }
        lines += result.lines_cleared;
    }
    board == TetrisBoard::EMPTY_BOARD && lines == LINES_FOR_PC
}

// --- Scan driver -----------------------------------------------------------

/// Per-tuple outcome from the parallel search.
enum TupleResult {
    /// Verified perfect clear; carries the interior boundary boards to deposit.
    PerfectClear([TetrisBoard; BAG_COUNT]),
    /// No perfect clear found within the beam budget.
    Failed,
    /// The beam reported a perfect clear that failed replay (should never
    /// happen; tracked defensively).
    Rejected,
}

fn solve_tuple(ordinal: u128, beam_width: usize, max_height: u32) -> TupleResult {
    let pieces = fill_sequence(tuple_from_ordinal(ordinal));
    match search_pc(&pieces, beam_width, max_height) {
        Some(node) if verify_perfect_clear(&pieces, &node.placements, max_height) => {
            TupleResult::PerfectClear(node.boundaries)
        }
        Some(_) => TupleResult::Rejected,
        None => TupleResult::Failed,
    }
}

#[derive(Default, Clone, Copy)]
struct ScanStats {
    scanned: u64,
    perfect_clears: u64,
    failures: u64,
    rejected: u64,
    boards_deposited: u64,
}

impl ScanStats {
    fn pc_rate(&self) -> f64 {
        if self.scanned == 0 {
            0.0
        } else {
            self.perfect_clears as f64 / self.scanned as f64
        }
    }
}

fn run_scan(cli: &DiscoverArgs) -> Result<()> {
    let started = Instant::now();
    let mut carrier = PhaseCarrier::new();
    let mut stats = ScanStats::default();

    let csv_path = format!("{}/phase_atlas.csv", cli.out_dir);
    let mut csv = CsvLogger::new(&csv_path)?;
    println!("csv -> {csv_path}");

    let end = cli
        .start
        .saturating_add(u128::from(cli.limit))
        .min(TOTAL_FIVE_BAG_SEQUENCES);

    let mut ordinal = cli.start;
    while ordinal < end {
        let chunk_end = ordinal.saturating_add(u128::from(cli.chunk)).min(end);
        let mut ordinals = Vec::with_capacity((chunk_end - ordinal) as usize);
        let mut o = ordinal;
        while o < chunk_end {
            ordinals.push(o);
            o += 1;
        }

        let results: Vec<TupleResult> = ordinals
            .par_iter()
            .map(|&o| solve_tuple(o, cli.beam_width, cli.max_height))
            .collect();

        for result in results {
            stats.scanned += 1;
            match result {
                TupleResult::PerfectClear(boundaries) => {
                    stats.perfect_clears += 1;
                    stats.boards_deposited += carrier.deposit_pc(&boundaries) as u64;
                }
                TupleResult::Failed => stats.failures += 1,
                TupleResult::Rejected => {
                    stats.rejected += 1;
                    eprintln!("WARN: a tuple reported a PC that failed replay");
                }
            }
        }

        let sizes = carrier.sizes();
        log_progress(&stats, &sizes, started);
        csv.write_row(&stats, &sizes, started)?;
        ordinal = chunk_end;
    }

    let sizes = carrier.sizes();
    print_summary(cli, &stats, &sizes, started);
    write_summary_json(cli, &stats, &sizes, started)?;
    Ok(())
}

fn log_progress(stats: &ScanStats, sizes: &[usize; BAG_COUNT + 1], started: Instant) {
    let secs = started.elapsed().as_secs_f64();
    let rate = stats.scanned as f64 / secs.max(1e-9);
    println!(
        "t={secs:.1}s scanned={} pc={} failed={} pc_rate={:.1}% | \
         carrier[P1={} P2={} P3={} P4={}] | rate={rate:.1}/s",
        stats.scanned,
        stats.perfect_clears,
        stats.failures,
        stats.pc_rate() * 100.0,
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
    );
}

fn print_summary(
    cli: &DiscoverArgs,
    stats: &ScanStats,
    sizes: &[usize; BAG_COUNT + 1],
    started: Instant,
) {
    let secs = started.elapsed().as_secs_f64();
    println!();
    println!("--- summary ---");
    println!("scanned            = {}", stats.scanned);
    println!("perfect_clears     = {}", stats.perfect_clears);
    println!("failures           = {}", stats.failures);
    println!("rejected           = {}", stats.rejected);
    println!("pc_rate            = {:.2}%", stats.pc_rate() * 100.0);
    println!("carrier_deposited  = {}", stats.boards_deposited);
    println!(
        "carrier_sizes      = [P1={}, P2={}, P3={}, P4={}]",
        sizes[1], sizes[2], sizes[3], sizes[4]
    );
    println!("beam_width         = {}", cli.beam_width);
    println!("max_height         = {}", cli.max_height);
    println!("time               = {secs:.2}s");
    println!(
        "rate               = {:.2}/s",
        stats.scanned as f64 / secs.max(1e-9)
    );
}

fn write_summary_json(
    cli: &DiscoverArgs,
    stats: &ScanStats,
    sizes: &[usize; BAG_COUNT + 1],
    started: Instant,
) -> Result<()> {
    let path = format!("{}/phase_atlas_summary.json", cli.out_dir);
    let secs = started.elapsed().as_secs_f64();
    let json = format!(
        "{{\n  \"start\": {},\n  \"limit\": {},\n  \"beam_width\": {},\n  \"max_height\": {},\n  \
         \"scanned\": {},\n  \"perfect_clears\": {},\n  \"failures\": {},\n  \"rejected\": {},\n  \
         \"pc_rate\": {:.6},\n  \"carrier_deposited\": {},\n  \
         \"carrier_sizes\": [{}, {}, {}, {}, {}, {}],\n  \"elapsed_secs\": {:.3}\n}}\n",
        cli.start,
        cli.limit,
        cli.beam_width,
        cli.max_height,
        stats.scanned,
        stats.perfect_clears,
        stats.failures,
        stats.rejected,
        stats.pc_rate(),
        stats.boards_deposited,
        sizes[0],
        sizes[1],
        sizes[2],
        sizes[3],
        sizes[4],
        sizes[5],
        secs,
    );
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    println!("summary -> {path}");
    Ok(())
}

struct CsvLogger {
    file: std::io::BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> Result<Self> {
        let exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut file = std::io::BufWriter::new(file);
        if !exists {
            writeln!(
                file,
                "secs,scanned,perfect_clears,failures,pc_rate,p1,p2,p3,p4,rate_per_sec"
            )?;
        }
        Ok(Self { file })
    }

    fn write_row(
        &mut self,
        stats: &ScanStats,
        sizes: &[usize; BAG_COUNT + 1],
        started: Instant,
    ) -> Result<()> {
        let secs = started.elapsed().as_secs_f64();
        let rate = stats.scanned as f64 / secs.max(1e-9);
        writeln!(
            self.file,
            "{:.3},{},{},{},{:.6},{},{},{},{},{:.2}",
            secs,
            stats.scanned,
            stats.perfect_clears,
            stats.failures,
            stats.pc_rate(),
            sizes[1],
            sizes[2],
            sizes[3],
            sizes[4],
            rate,
        )?;
        self.file.flush()?;
        Ok(())
    }
}

// --- Adversarial closure certification -------------------------------------

/// State of the exact per-piece AND-OR search over one N-bag reset cycle.
///
/// `good(board, step, bag)` is true iff, from this state, the player can force
/// the board back to empty exactly at `total_steps`, against every adversarial
/// choice of which remaining bag piece is drawn next:
///
/// - AND over the pieces the adversary may draw (`bag.iter_next_states()`).
/// - OR over the player's legal hard-drop placements of that piece.
/// - terminal: at `total_steps`, the board must be empty.
///
/// Boards taller than `max_height` are inadmissible, and a perfect-clear
/// feasibility filter prunes boards whose cell count can never reach 0.
struct CertCtx {
    total_steps: usize,
    max_height: u32,
    node_budget: u64,
    memo: FxHashMap<(TetrisBoard, u16, u8), bool>,
    nodes: u64,
    terminal_hits: u64,
    budget_exceeded: bool,
}

impl CertCtx {
    fn new(total_steps: usize, max_height: u32, node_budget: u64) -> Self {
        Self {
            total_steps,
            max_height,
            node_budget,
            memo: FxHashMap::default(),
            nodes: 0,
            terminal_hits: 0,
            budget_exceeded: false,
        }
    }

    /// Recursive AND-OR evaluation. Short-circuits on the first adversary piece
    /// the player cannot answer (fast NO) and on the first winning placement
    /// (fast OR). Returns `false` conservatively once the node budget is hit,
    /// flagging `budget_exceeded` so the caller can report INCONCLUSIVE.
    fn good(&mut self, board: TetrisBoard, step: usize, bag: TetrisPieceBagState) -> bool {
        if step == self.total_steps {
            let win = board == TetrisBoard::EMPTY_BOARD;
            if win {
                self.terminal_hits += 1;
            }
            return win;
        }

        let key = (board, step as u16, u8::from(bag));
        if let Some(&cached) = self.memo.get(&key) {
            return cached;
        }

        if self.budget_exceeded {
            return false;
        }
        self.nodes += 1;
        if self.nodes > self.node_budget {
            self.budget_exceeded = true;
            return false;
        }

        // Perfect-clear feasibility: each line clear removes 10 cells, so the
        // board can only reach empty if the running cell count plus all future
        // cells is a multiple of 10.
        let remaining = (self.total_steps - step) as u32;
        if (board.count() + 4 * remaining) % 10 != 0 {
            self.memo.insert(key, false);
            return false;
        }

        let mut result = true;
        for (piece, next_bag) in bag.iter_next_states() {
            let mut answered = false;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child = board;
                let drop = child.apply_piece_placement(placement);
                if drop.is_lost == IsLost::LOST || child.height() > self.max_height {
                    continue;
                }
                if self.good(child, step + 1, next_bag) {
                    answered = true;
                    break;
                }
                if self.budget_exceeded {
                    return false;
                }
            }
            if !answered {
                result = false;
                break;
            }
        }

        if !self.budget_exceeded {
            self.memo.insert(key, result);
        }
        result
    }

    /// Per-first-piece coverage at the root: how many of the 7 opening pieces the
    /// player can answer with at least one placement leading to a winning state.
    fn root_coverage(&mut self, board: TetrisBoard) -> (u32, [bool; 7]) {
        let bag = TetrisPieceBagState::new();
        let mut covered = 0;
        let mut per_piece = [false; 7];
        for (piece, next_bag) in bag.iter_next_states() {
            let mut answered = false;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child = board;
                let drop = child.apply_piece_placement(placement);
                if drop.is_lost == IsLost::LOST || child.height() > self.max_height {
                    continue;
                }
                if self.good(child, 1, next_bag) {
                    answered = true;
                    break;
                }
            }
            per_piece[piece.index() as usize] = answered;
            if answered {
                covered += 1;
            }
        }
        (covered, per_piece)
    }
}

fn run_certify(args: &CertifyArgs) -> Result<()> {
    if args.bag_cycles == 0 || args.bag_cycles % BAG_COUNT != 0 {
        bail!("--bag-cycles must be a positive multiple of {BAG_COUNT}");
    }
    if args.max_height == 0 || args.max_height > StandardTetris::ROWS as u32 {
        bail!("--max-height must be in 1..={}", StandardTetris::ROWS);
    }
    fs::create_dir_all(&args.out_dir)?;

    let total_steps = args.bag_cycles * PIECES_PER_BAG;
    let lines_needed = (total_steps as u32 * 4) / 10;

    println!("=== tetris_phase_atlas: certify (adversarial) ===");
    println!("bag_cycles   = {}", args.bag_cycles);
    println!("total_steps  = {total_steps}");
    println!("lines_needed = {lines_needed}");
    println!("max_height   = {}", args.max_height);
    println!("node_budget  = {}", args.node_budget);
    println!("opponent     = per-piece online adversary (sees current piece, no lookahead)");
    println!();

    let started = Instant::now();
    let mut ctx = CertCtx::new(total_steps, args.max_height, args.node_budget);
    let root = TetrisBoard::EMPTY_BOARD;
    let bag = TetrisPieceBagState::new();
    let winning = ctx.good(root, 0, bag);
    let solve_secs = started.elapsed().as_secs_f64();

    // Root coverage (best-effort; reuses the memo populated above).
    let (covered, per_piece) = ctx.root_coverage(root);
    let elapsed = started.elapsed().as_secs_f64();

    let status = if ctx.budget_exceeded {
        "INCONCLUSIVE"
    } else if winning {
        "WINNING"
    } else {
        "NOT-WINNING"
    };

    println!("--- result ---");
    println!("status            = {status}");
    if !ctx.budget_exceeded {
        println!("winning           = {winning}");
    }
    println!("root_coverage     = {covered}/7");
    print!("opening_pieces    = [");
    for piece in TetrisPiece::all() {
        let ok = per_piece[piece.index() as usize];
        print!("{}:{} ", piece, if ok { "+" } else { "." });
    }
    println!("]");
    println!("nodes             = {}", ctx.nodes);
    println!("memo_entries      = {}", ctx.memo.len());
    println!("terminal_hits     = {}", ctx.terminal_hits);
    println!("budget_exceeded   = {}", ctx.budget_exceeded);
    println!("solve_time        = {solve_secs:.2}s");
    println!("total_time        = {elapsed:.2}s");

    let path = format!("{}/certify_summary.json", args.out_dir);
    let json = format!(
        "{{\n  \"bag_cycles\": {},\n  \"total_steps\": {},\n  \"max_height\": {},\n  \
         \"status\": \"{}\",\n  \"winning\": {},\n  \"root_coverage\": {},\n  \"nodes\": {},\n  \
         \"memo_entries\": {},\n  \"terminal_hits\": {},\n  \"budget_exceeded\": {},\n  \
         \"elapsed_secs\": {:.3}\n}}\n",
        args.bag_cycles,
        total_steps,
        args.max_height,
        status,
        winning && !ctx.budget_exceeded,
        covered,
        ctx.nodes,
        ctx.memo.len(),
        ctx.terminal_hits,
        ctx.budget_exceeded,
        elapsed,
    );
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    println!("summary -> {path}");
    Ok(())
}

// --- Closed safe-set growth ------------------------------------------------

/// Full 7-piece bag mask (bits 0..6 set).
const FULL_BAG_MASK: u8 = 0b0111_1111;

/// Growth state for the board-only safe-set closure search.
///
/// A board set `R` (at bag boundaries, where the bag is always fresh) is
/// **closed** iff from every `b in R` and every order in which the adversary may
/// reveal the 7 bag pieces, the player can hard-drop them — never exceeding the
/// height cap — and land on some board again in `R`. `good_bag` is the within-bag
/// per-piece AND-OR test of that property against the *current* `R`.
struct GrowCtx {
    max_height: u32,
    /// Memo of `(board, remaining_mask) -> can reach R`. Valid only for the
    /// current `R`; cleared every growth round.
    memo: FxHashMap<(TetrisBoard, u8), bool>,
    /// Boards the best-effort player is forced onto when `R` is insufficient;
    /// folded into `R` at the end of each round to grow the set.
    additions: FxHashSet<TetrisBoard>,
    /// Boards in `R` that have a piece with no admissible placement at all (a
    /// forced top-out): these can never be made safe at this height cap.
    hard_deaths: u64,
    nodes: u64,
}

impl GrowCtx {
    fn new(max_height: u32) -> Self {
        Self {
            max_height,
            memo: FxHashMap::default(),
            additions: FxHashSet::default(),
            hard_deaths: 0,
            nodes: 0,
        }
    }

    /// Within one bag from `board` with `remaining` pieces undrawn: can the
    /// player answer every adversarial reveal order and finish (all 7 placed) on
    /// a board in `R`? On a failing OR-node (no placement keeps the line in `R`)
    /// the player's lowest-`height_mse` fallback board is recorded in
    /// `additions` so the next round can grow `R` toward closure.
    fn good_bag(
        &mut self,
        board: TetrisBoard,
        remaining: u8,
        set: &FxHashSet<TetrisBoard>,
    ) -> bool {
        if remaining == 0 {
            return set.contains(&board);
        }
        if let Some(&cached) = self.memo.get(&(board, remaining)) {
            return cached;
        }
        self.nodes += 1;

        let mut result = true;
        // AND over the pieces the adversary may reveal next.
        for piece_idx in 0..7u8 {
            if remaining & (1 << piece_idx) == 0 {
                continue;
            }
            let piece = TetrisPiece::from_index(piece_idx);
            let next_remaining = remaining & !(1 << piece_idx);

            // OR over the player's placements of that piece.
            let mut answered = false;
            let mut best: Option<(f32, TetrisBoard)> = None;
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child = board;
                let drop = child.apply_piece_placement(placement);
                if drop.is_lost == IsLost::LOST || child.height() > self.max_height {
                    continue;
                }
                let dist = height_mse_distance_from_empty(child);
                if best.is_none_or(|(bd, _)| dist < bd) {
                    best = Some((dist, child));
                }
                if self.good_bag(child, next_remaining, set) {
                    answered = true;
                    break;
                }
            }

            if !answered {
                result = false;
                match best {
                    // Only fully-placed bags (a fresh boundary) are members of
                    // R; record the player's best boundary board to grow toward.
                    Some((_, fallback)) if next_remaining == 0 => {
                        self.additions.insert(fallback);
                    }
                    // Mid-bag dead end: propagate failure, nothing to add here.
                    Some(_) => {}
                    // No admissible placement at all: a forced top-out at the cap.
                    None => self.hard_deaths += 1,
                }
                // Keep scanning the other pieces to collect more additions.
            }
        }

        self.memo.insert((board, remaining), result);
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosureOutcome {
    /// `R` is closed: every board answers every bag order back into `R`.
    Closed,
    /// `R` exceeded the size cap before closing.
    Floor,
    /// `R` stopped growing but is still not closed (forced top-outs remain).
    Stuck,
    /// Ran out of growth rounds.
    RoundsExhausted,
}

fn run_closure(args: &ClosureArgs) -> Result<()> {
    if args.max_height == 0 || args.max_height > StandardTetris::ROWS as u32 {
        bail!("--max-height must be in 1..={}", StandardTetris::ROWS);
    }
    if args.beam_width == 0 {
        bail!("--beam-width must be > 0");
    }
    fs::create_dir_all(&args.out_dir)?;

    if args.bag_lookahead {
        return run_closure_bag(args);
    }

    println!("=== tetris_phase_atlas: closure (safe-set growth) ===");
    println!("max_height = {}", args.max_height);
    println!("max_boards = {}", args.max_boards);
    println!("max_rounds = {}", args.max_rounds);
    println!(
        "opponent   = per-piece online adversary; period = 1 bag; target = survival (bounded)"
    );
    println!();

    let started = Instant::now();
    let mut ctx = GrowCtx::new(args.max_height);
    let mut set: FxHashSet<TetrisBoard> = FxHashSet::default();
    set.insert(TetrisBoard::EMPTY_BOARD);

    let mut outcome = ClosureOutcome::RoundsExhausted;
    let mut rounds = 0u32;
    while rounds < args.max_rounds {
        rounds += 1;
        ctx.memo.clear();
        ctx.additions.clear();
        ctx.hard_deaths = 0;

        let mut all_closed = true;
        let mut unclosed = 0u64;
        for &board in &set {
            if !ctx.good_bag(board, FULL_BAG_MASK, &set) {
                all_closed = false;
                unclosed += 1;
            }
        }

        if all_closed {
            outcome = ClosureOutcome::Closed;
            break;
        }
        if ctx.additions.is_empty() {
            outcome = ClosureOutcome::Stuck;
            break;
        }

        let before = set.len();
        for &board in &ctx.additions {
            set.insert(board);
        }
        let added = set.len() - before;

        println!(
            "round={rounds} |R|={} unclosed={unclosed} added={added} hard_deaths={} nodes={} t={:.1}s",
            set.len(),
            ctx.hard_deaths,
            ctx.nodes,
            started.elapsed().as_secs_f64(),
        );

        if set.len() > args.max_boards {
            outcome = ClosureOutcome::Floor;
            break;
        }
    }

    let elapsed = started.elapsed().as_secs_f64();
    let max_h = set.iter().map(|b| b.height()).max().unwrap_or(0);
    let solved = outcome == ClosureOutcome::Closed && set.contains(&TetrisBoard::EMPTY_BOARD);

    // Distinct column-height profiles ("surfaces"). If this is far smaller than
    // the board count, a height-dominance basis could compress the safe set
    // (the hole-debt / WQO route): boards sharing a surface differ only in
    // buried holes.
    let surfaces: FxHashSet<[u32; StandardTetris::COLS]> =
        set.iter().map(|b| b.heights()).collect();

    println!();
    println!("--- result ---");
    println!("outcome        = {outcome:?}");
    println!("solved         = {solved}");
    println!("safe_set_size  = {}", set.len());
    println!("distinct_surfaces = {}", surfaces.len());
    println!(
        "empty_in_set   = {}",
        set.contains(&TetrisBoard::EMPTY_BOARD)
    );
    println!("max_height_seen= {max_h}");
    println!("rounds         = {rounds}");
    println!("nodes          = {}", ctx.nodes);
    println!("hard_deaths    = {}", ctx.hard_deaths);
    println!("time           = {elapsed:.2}s");
    if solved {
        println!();
        println!(
            "*** CLOSED SAFE SET FOUND: {} boards, height <= {}, empty included. ***",
            set.len(),
            args.max_height
        );
        println!("*** Infinite play under the per-piece 7-bag adversary (within the cap). ***");
    }

    let path = format!("{}/closure_summary.json", args.out_dir);
    let json = format!(
        "{{\n  \"max_height\": {},\n  \"outcome\": \"{:?}\",\n  \"solved\": {},\n  \
         \"safe_set_size\": {},\n  \"distinct_surfaces\": {},\n  \"empty_in_set\": {},\n  \
         \"max_height_seen\": {},\n  \"rounds\": {},\n  \"nodes\": {},\n  \"hard_deaths\": {},\n  \
         \"elapsed_secs\": {:.3}\n}}\n",
        args.max_height,
        outcome,
        solved,
        set.len(),
        surfaces.len(),
        set.contains(&TetrisBoard::EMPTY_BOARD),
        max_h,
        rounds,
        ctx.nodes,
        ctx.hard_deaths,
        elapsed,
    );
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    println!("summary -> {path}");
    Ok(())
}

/// Per-bag, full-lookahead within-bag landings. Places the 7 pieces of `bag`
/// *in the given order* (the adversary's reveal order), the player choosing
/// placements knowing the whole bag, via a beam ranked by `height_mse`.
/// Returns `(in_set, any)`:
/// - `in_set`: the lowest-`height_mse` reachable boundary board that is already
///   in `set` (so the player can stay in `R` — no growth), or `None`;
/// - `any`: the lowest-`height_mse` reachable boundary board overall, or `None`
///   if the bag cannot be placed at all under the cap (a forced top-out).
///
/// The BFS uses `in_set` when present and only falls back to `any` (growing
/// `R`) when forced — the "prefer to stay in R" policy that keeps `R` minimal.
fn within_bag_landings(
    start: TetrisBoard,
    bag: &ForcedBag,
    set: &FxHashSet<TetrisBoard>,
    max_height: u32,
    beam_width: usize,
) -> (Option<TetrisBoard>, Option<TetrisBoard>) {
    let mut beam: Vec<(f32, TetrisBoard)> = vec![(0.0, start)];
    let mut next: Vec<(f32, TetrisBoard)> = Vec::new();
    let mut seen: FxHashSet<TetrisBoard> = FxHashSet::default();

    for &piece in bag {
        next.clear();
        seen.clear();
        for &(_, board) in &beam {
            for &placement in TetrisPiecePlacement::all_from_piece(piece) {
                let mut child = board;
                let drop = child.apply_piece_placement(placement);
                if drop.is_lost == IsLost::LOST || child.height() > max_height {
                    continue;
                }
                if seen.insert(child) {
                    next.push((height_mse_distance_from_empty(child), child));
                }
            }
        }
        if next.is_empty() {
            return (None, None);
        }
        if next.len() > beam_width {
            next.select_nth_unstable_by(beam_width, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            next.truncate(beam_width);
        }
        std::mem::swap(&mut beam, &mut next);
    }

    let cmp = |a: &(f32, TetrisBoard), b: &(f32, TetrisBoard)| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    };
    let any = beam
        .iter()
        .min_by(|a, b| cmp(a, b))
        .map(|(_, board)| *board);
    let in_set = beam
        .iter()
        .filter(|(_, board)| set.contains(board))
        .min_by(|a, b| cmp(a, b))
        .map(|(_, board)| *board);
    (in_set, any)
}

/// Per-bag (full within-bag lookahead) closure via forward BFS under a fixed
/// policy. The policy is: for each of the 5040 bag orders, play the lowest-
/// `height_mse` landing. Starting from empty, BFS collects every landing board
/// the policy can produce. Because the landings *define* the reachable set `R`,
/// `R` is closed by construction; the only failure is a forced top-out (a bag
/// that cannot be placed under the cap). So:
///
/// - BFS completes with **no** top-out  →  `R` is a finite closed safe set and
///   the policy survives every bag forever (an infinite-play atlas).
/// - any top-out  →  the fixed policy fails at this cap (`Stuck`).
/// - `R` exceeds the cap  →  `Floor` (finite certificate not established here).
///
/// Each board is processed exactly once (unlike a fixpoint), and the 5040 orders
/// for a frontier batch are answered in parallel.
fn run_closure_bag(args: &ClosureArgs) -> Result<()> {
    println!("=== tetris_phase_atlas: closure (per-bag lookahead, forward BFS) ===");
    println!("max_height = {}", args.max_height);
    println!("max_boards = {}", args.max_boards);
    println!("beam_width = {}", args.beam_width);
    println!(
        "opponent   = per-bag adversary (5040 orders), full within-bag lookahead; survival target"
    );
    println!("policy     = lowest-height_mse landing per bag order");
    println!("threads    = {}", rayon::current_num_threads());
    println!();

    let started = Instant::now();
    let mut set: FxHashSet<TetrisBoard> = FxHashSet::default();
    set.insert(TetrisBoard::EMPTY_BOARD);
    let mut frontier: Vec<TetrisBoard> = vec![TetrisBoard::EMPTY_BOARD];

    let mut outcome = ClosureOutcome::Closed;
    let mut hard_deaths = 0u64;
    let mut processed = 0u64;

    while !frontier.is_empty() {
        let batch: Vec<TetrisBoard> = std::mem::take(&mut frontier);

        // Answer all 5040 bag orders for every board in the batch, in parallel.
        // Prefer staying in R; only add a new board when forced (no in-R landing).
        let results: Vec<(Vec<TetrisBoard>, u64)> = batch
            .par_iter()
            .map(|&board| {
                let mut additions = Vec::new();
                let mut deaths = 0u64;
                for bag in ALL_BAG_PERMUTATIONS.iter() {
                    let (in_set, any) =
                        within_bag_landings(board, bag, &set, args.max_height, args.beam_width);
                    if in_set.is_some() {
                        // Player stays in R — no growth, no death.
                    } else if let Some(landing) = any {
                        additions.push(landing);
                    } else {
                        deaths += 1;
                    }
                }
                (additions, deaths)
            })
            .collect();

        processed += batch.len() as u64;
        for (landings, deaths) in &results {
            hard_deaths += *deaths;
            for &landing in landings {
                if set.insert(landing) {
                    frontier.push(landing);
                }
            }
        }

        println!(
            "processed={processed} |R|={} frontier={} hard_deaths={hard_deaths} t={:.1}s",
            set.len(),
            frontier.len(),
            started.elapsed().as_secs_f64(),
        );

        if set.len() > args.max_boards {
            outcome = ClosureOutcome::Floor;
            break;
        }
    }

    if outcome != ClosureOutcome::Floor && hard_deaths > 0 {
        // Reachable but the policy tops out somewhere — not a survival proof.
        outcome = ClosureOutcome::Stuck;
    }

    let elapsed = started.elapsed().as_secs_f64();
    let surfaces: FxHashSet<[u32; StandardTetris::COLS]> =
        set.iter().map(|b| b.heights()).collect();
    let max_h = set.iter().map(|b| b.height()).max().unwrap_or(0);
    let solved = outcome == ClosureOutcome::Closed && set.contains(&TetrisBoard::EMPTY_BOARD);

    println!();
    println!("--- result ---");
    println!("outcome           = {outcome:?}");
    println!("solved            = {solved}");
    println!("safe_set_size     = {}", set.len());
    println!("distinct_surfaces = {}", surfaces.len());
    println!(
        "empty_in_set      = {}",
        set.contains(&TetrisBoard::EMPTY_BOARD)
    );
    println!("max_height_seen   = {max_h}");
    println!("boards_processed  = {processed}");
    println!("hard_deaths       = {hard_deaths}");
    println!("time              = {elapsed:.2}s");
    if solved {
        println!();
        println!(
            "*** CLOSED SAFE SET FOUND: {} boards, height <= {}, empty included. ***",
            set.len(),
            args.max_height
        );
        println!(
            "*** Infinite play under the per-bag (full-lookahead) 7-bag adversary (within cap). ***"
        );
    }

    let path = format!("{}/closure_bag_summary.json", args.out_dir);
    let json = format!(
        "{{\n  \"model\": \"bag_lookahead\",\n  \"max_height\": {},\n  \"beam_width\": {},\n  \
         \"outcome\": \"{:?}\",\n  \"solved\": {},\n  \"safe_set_size\": {},\n  \
         \"distinct_surfaces\": {},\n  \"empty_in_set\": {},\n  \"max_height_seen\": {},\n  \
         \"boards_processed\": {},\n  \"hard_deaths\": {},\n  \"elapsed_secs\": {:.3}\n}}\n",
        args.max_height,
        args.beam_width,
        outcome,
        solved,
        set.len(),
        surfaces.len(),
        set.contains(&TetrisBoard::EMPTY_BOARD),
        max_h,
        processed,
        hard_deaths,
        elapsed,
    );
    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    println!("summary -> {path}");
    Ok(())
}

// --- Bag-permutation enumeration (base-5040 tuples) -------------------------

const fn pow_u128(base: u128, exp: u32) -> u128 {
    let mut out = 1u128;
    let mut i = 0u32;
    while i < exp {
        out *= base;
        i += 1;
    }
    out
}

const fn advance_permutation(indices: &mut [u8; PIECES_PER_BAG]) -> bool {
    let mut pivot = PIECES_PER_BAG - 2;
    loop {
        if indices[pivot] < indices[pivot + 1] {
            break;
        }
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut successor = PIECES_PER_BAG - 1;
    while indices[pivot] >= indices[successor] {
        successor -= 1;
    }
    indices.swap(pivot, successor);
    let mut left = pivot + 1;
    let mut right = PIECES_PER_BAG - 1;
    while left < right {
        indices.swap(left, right);
        left += 1;
        right -= 1;
    }
    true
}

const fn generate_forced_bag_permutations() -> [ForcedBag; BAG_PERM_COUNT] {
    let mut permutations = [[TetrisPiece::O_PIECE; PIECES_PER_BAG]; BAG_PERM_COUNT];
    let mut indices = [0u8, 1, 2, 3, 4, 5, 6];
    let mut out_idx = 0usize;
    loop {
        let mut bag = [TetrisPiece::O_PIECE; PIECES_PER_BAG];
        let mut i = 0usize;
        while i < PIECES_PER_BAG {
            bag[i] = TetrisPiece::from_index(indices[i]);
            i += 1;
        }
        permutations[out_idx] = bag;
        out_idx += 1;
        if !advance_permutation(&mut indices) {
            break;
        }
    }
    permutations
}

fn tuple_from_ordinal(mut ordinal: u128) -> [usize; BAG_COUNT] {
    let mut tuple = [0usize; BAG_COUNT];
    let base = BAG_PERM_COUNT as u128;
    let mut i = 0;
    while i < BAG_COUNT {
        tuple[i] = (ordinal % base) as usize;
        ordinal /= base;
        i += 1;
    }
    tuple
}

fn fill_sequence(tuple: [usize; BAG_COUNT]) -> [TetrisPiece; TOTAL_PIECES] {
    let mut pieces = [TetrisPiece::O_PIECE; TOTAL_PIECES];
    for (bag_idx, &perm_idx) in tuple.iter().enumerate() {
        let bag = ALL_BAG_PERMUTATIONS[perm_idx];
        let start = bag_idx * PIECES_PER_BAG;
        pieces[start..start + PIECES_PER_BAG].copy_from_slice(&bag);
    }
    pieces
}

// --- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_table_endpoints() {
        assert_eq!(ALL_BAG_PERMUTATIONS.len(), BAG_PERM_COUNT);
        assert_eq!(
            ALL_BAG_PERMUTATIONS[0],
            [
                TetrisPiece::O_PIECE,
                TetrisPiece::I_PIECE,
                TetrisPiece::S_PIECE,
                TetrisPiece::Z_PIECE,
                TetrisPiece::T_PIECE,
                TetrisPiece::L_PIECE,
                TetrisPiece::J_PIECE,
            ]
        );
        let unique: FxHashSet<_> = ALL_BAG_PERMUTATIONS.iter().collect();
        assert_eq!(unique.len(), BAG_PERM_COUNT);
    }

    #[test]
    fn ordinal_is_base_5040() {
        assert_eq!(tuple_from_ordinal(0), [0, 0, 0, 0, 0]);
        assert_eq!(tuple_from_ordinal(1), [1, 0, 0, 0, 0]);
        assert_eq!(tuple_from_ordinal(BAG_PERM_COUNT as u128), [0, 1, 0, 0, 0]);
    }

    #[test]
    fn fill_sequence_lays_out_bags() {
        let pieces = fill_sequence([0, 1, 2, 3, 4]);
        assert_eq!(&pieces[0..PIECES_PER_BAG], &ALL_BAG_PERMUTATIONS[0]);
        assert_eq!(
            &pieces[PIECES_PER_BAG..2 * PIECES_PER_BAG],
            &ALL_BAG_PERMUTATIONS[1]
        );
        for bag in pieces.chunks(PIECES_PER_BAG) {
            let unique: FxHashSet<_> = bag.iter().collect();
            assert_eq!(unique.len(), PIECES_PER_BAG);
        }
    }

    #[test]
    fn pc_arithmetic() {
        assert_eq!(LINES_FOR_PC, 14);
        assert_eq!(TOTAL_PIECES, 35);
        assert_eq!(TOTAL_PIECES as u32 * 4, LINES_FOR_PC * 10);
    }

    /// Greedily place five O-pieces into the five column pairs to fill and clear
    /// two rows — a hand-built perfect clear validating the engine usage and the
    /// replay/verification path.
    #[test]
    fn five_o_pieces_perfect_clear() {
        let o = TetrisPiece::O_PIECE;
        let placements = TetrisPiecePlacement::all_from_piece(o);
        let mut board = TetrisBoard::EMPTY_BOARD;
        let mut total_lines = 0u32;
        for _ in 0..5 {
            let mut best: Option<(u32, TetrisBoard, u32)> = None;
            for &placement in placements {
                let mut candidate = board;
                let result = candidate.apply_piece_placement(placement);
                if result.is_lost == IsLost::LOST {
                    continue;
                }
                let key = candidate.height();
                if best.as_ref().is_none_or(|(h, _, _)| key < *h) {
                    best = Some((key, candidate, result.lines_cleared));
                }
            }
            let (_, candidate, lines) = best.expect("O always placeable on a low board");
            board = candidate;
            total_lines += lines;
        }
        assert_eq!(board, TetrisBoard::EMPTY_BOARD);
        assert_eq!(total_lines, 2);
    }

    #[test]
    fn carrier_deposit() {
        let mut carrier = PhaseCarrier::new();
        // Anchor phases never store boards.
        let empty = TetrisBoard::EMPTY_BOARD;
        assert!(!carrier.deposit(0, empty));
        assert!(!carrier.deposit(BAG_COUNT, empty));

        let mut b = TetrisBoard::EMPTY_BOARD;
        b.apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0]);
        assert!(carrier.deposit(2, b));
        assert!(!carrier.deposit(2, b)); // idempotent
        assert_eq!(carrier.sizes()[2], 1);
        assert_eq!(carrier.sizes()[1], 0);
    }

    /// The solver must never report an unverifiable perfect clear, and any PC it
    /// reports must replay to the empty board with 14 lines cleared.
    #[test]
    fn solver_successes_are_valid() {
        for ordinal in 0..24u128 {
            let pieces = fill_sequence(tuple_from_ordinal(ordinal));
            if let Some(node) = search_pc(&pieces, 1024, StandardTetris::ROWS as u32) {
                assert!(
                    verify_perfect_clear(&pieces, &node.placements, StandardTetris::ROWS as u32),
                    "reported PC must replay (ordinal {ordinal})"
                );
                // The final boundary board is the empty board.
                assert_eq!(node.boundaries[BAG_COUNT - 1], TetrisBoard::EMPTY_BOARD);
            }
        }
    }

    fn one_o_board() -> TetrisBoard {
        let mut b: TetrisBoard = TetrisBoard::EMPTY_BOARD;
        b.apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0]);
        b
    }

    #[test]
    fn certify_terminal_base_cases() {
        let mut ctx = CertCtx::new(TOTAL_PIECES, 6, 1_000_000);
        // At the terminal step only the empty board is a win.
        assert!(ctx.good(
            TetrisBoard::EMPTY_BOARD,
            TOTAL_PIECES,
            TetrisPieceBagState::new()
        ));
        assert!(!ctx.good(one_o_board(), TOTAL_PIECES, TetrisPieceBagState::new()));
    }

    #[test]
    fn certify_feasibility_prune() {
        // 4 cells with 1 piece remaining: 4 + 4 = 8, not a multiple of 10, so
        // empty is unreachable and the state is pruned to false.
        let mut ctx = CertCtx::new(TOTAL_PIECES, 6, 1_000_000);
        assert!(!ctx.good(one_o_board(), TOTAL_PIECES - 1, TetrisPieceBagState::new()));
    }

    #[test]
    fn good_bag_terminal_membership() {
        let mut ctx = GrowCtx::new(6);
        let mut set = FxHashSet::default();
        set.insert(TetrisBoard::EMPTY_BOARD);
        assert!(ctx.good_bag(TetrisBoard::EMPTY_BOARD, 0, &set));
        assert!(!ctx.good_bag(one_o_board(), 0, &set));
    }

    #[test]
    fn good_bag_records_fallback() {
        // With only the O piece left to place and R = {empty}, no placement ends
        // in R (the O leaves four cells), so the player's lowest-`height_mse`
        // fallback board is recorded for the next growth round. (Single-piece
        // mask keeps the within-bag DAG tiny and the test fast.)
        let mut ctx = GrowCtx::new(6);
        let mut set = FxHashSet::default();
        set.insert(TetrisBoard::EMPTY_BOARD);
        let mask = 1u8 << TetrisPiece::O_PIECE.index();
        assert!(!ctx.good_bag(TetrisBoard::EMPTY_BOARD, mask, &set));
        assert!(
            !ctx.additions.is_empty(),
            "growth must record a fallback board"
        );
    }

    #[test]
    fn within_bag_landings_basics() {
        let mut set = FxHashSet::default();
        set.insert(TetrisBoard::EMPTY_BOARD);
        let bag = ALL_BAG_PERMUTATIONS[0];
        let (in_set, any) = within_bag_landings(TetrisBoard::EMPTY_BOARD, &bag, &set, 6, 256);
        // One bag (28 cells) can never return to empty, so no in-R landing.
        assert!(in_set.is_none());
        // But a low landing board is always reachable at height cap 6.
        let landing = any.expect("a landing board exists");
        assert_ne!(landing, TetrisBoard::EMPTY_BOARD);
        assert!(landing.height() <= 6);
    }

    #[test]
    fn certify_deterministic_and_terminates() {
        let run = || {
            let mut ctx = CertCtx::new(TOTAL_PIECES, 3, 50_000_000);
            let winning = ctx.good(TetrisBoard::EMPTY_BOARD, 0, TetrisPieceBagState::new());
            (winning, ctx.budget_exceeded)
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "certify must be deterministic");
        assert!(!a.1, "cap=3 5-bag search should finish within budget");
    }
}
