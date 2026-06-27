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
use clap::Parser;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tetris_game::{
    IsLost, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPiecePlacement,
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
#[command(about = "Phase-layered 5-bag perfect-clear atlas (cooperative discovery)")]
struct Cli {
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_cli(&cli)?;
    fs::create_dir_all(&cli.out_dir)?;

    println!("=== tetris_phase_atlas (cooperative discovery) ===");
    println!("total_pieces         = {TOTAL_PIECES}");
    println!("lines_for_pc         = {LINES_FOR_PC}");
    println!("bag_perm_count       = {BAG_PERM_COUNT}");
    println!("total_5bag_sequences = {TOTAL_FIVE_BAG_SEQUENCES}");
    println!("start                = {}", cli.start);
    println!("limit                = {}", cli.limit);
    println!("beam_width           = {}", cli.beam_width);
    println!("max_height           = {}", cli.max_height);
    println!("chunk                = {}", cli.chunk);
    println!("out_dir              = {}", cli.out_dir);
    println!("threads              = {}", rayon::current_num_threads());
    println!();

    run_scan(&cli)
}

fn validate_cli(cli: &Cli) -> Result<()> {
    if cli.start >= TOTAL_FIVE_BAG_SEQUENCES {
        bail!("--start must be < {TOTAL_FIVE_BAG_SEQUENCES}");
    }
    if cli.beam_width == 0 {
        bail!("--beam-width must be > 0");
    }
    if cli.chunk == 0 {
        bail!("--chunk must be > 0");
    }
    if cli.max_height == 0 || cli.max_height > StandardTetris::ROWS as u32 {
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

fn run_scan(cli: &Cli) -> Result<()> {
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

fn print_summary(cli: &Cli, stats: &ScanStats, sizes: &[usize; BAG_COUNT + 1], started: Instant) {
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
    cli: &Cli,
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
}
