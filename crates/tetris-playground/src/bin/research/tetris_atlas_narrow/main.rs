#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Tetris Atlas Builder — Narrow (4-column, 3-piece) with Backtracking
//!
//! A reduced-complexity atlas builder with **death propagation**:
//! - **4 columns**, **3 pieces** (L, J, O)
//! - Forward expansion stores ALL valid placements (not just beam winner)
//! - When a state dies, propagates backward trying alternatives
//! - Can prove solvability OR unsolvability of the constrained game

use clap::{Parser, Subcommand};
use crossbeam_channel::{Sender, bounded};
use crossbeam_skiplist::SkipMap;
use dashmap::mapref::entry::Entry;
use dashmap::{DashMap, DashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard as GenericTetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement,
};
use tetris_utils::repeat_idx_unroll;

// --- Narrow Game Constraints ---

const ALLOWED_PIECES: [TetrisPiece; 3] = [
    TetrisPiece::L_PIECE,
    TetrisPiece::J_PIECE,
    TetrisPiece::O_PIECE,
];

const NARROW_BAG_FULL: TetrisPieceBagState = TetrisPieceBagState::from_pieces(ALLOWED_PIECES);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct NarrowTetris;

impl TetrisGameConfig for NarrowTetris {
    const ROWS: usize = 20;
    const COLS: usize = 4;
    const PIECE_SET: TetrisPieceBagState = NARROW_BAG_FULL;
}

type TetrisBoard = GenericTetrisBoard<NarrowTetris>;

fn is_allowed_piece(piece: TetrisPiece) -> bool {
    ALLOWED_PIECES.contains(&piece)
}

/// Strip forbidden pieces from `bag`. If no allowed pieces remain, refill
/// with `NARROW_BAG_FULL`. Keeps the bag state invariant: only ever
/// contains a non-empty subset of `ALLOWED_PIECES`.
///
/// The "strip" step is a bitwise AND against the full narrow mask — drops
/// any piece not in `ALLOWED_PIECES` in one op, no per-piece loop.
fn normalize_narrow_bag(bag: TetrisPieceBagState) -> TetrisPieceBagState {
    let allowed_mask = u8::from(NARROW_BAG_FULL);
    let stripped = TetrisPieceBagState::from(u8::from(bag) & allowed_mask);
    if stripped.is_empty() {
        NARROW_BAG_FULL
    } else {
        stripped
    }
}

/// Narrow-aware drop-in replacement for `TetrisPieceBagState::iter_next_states`.
/// The stock iterator refills to the *full 7-piece* bag when empty, which
/// re-introduces forbidden pieces after one cycle and eventually wedges the
/// atlas. This wrapper:
/// 1. normalizes the input bag (drop forbidden + refill if empty), so the
///    iteration only yields allowed pieces;
/// 2. normalizes each `next_bag`, so stored `result_bag`s only ever contain
///    pieces from `ALLOWED_PIECES`.
fn narrow_iter_next_states(
    bag: TetrisPieceBagState,
) -> impl Iterator<Item = (TetrisPiece, TetrisPieceBagState)> {
    let bag = normalize_narrow_bag(bag);
    // Collect to break the lifetime tie to the local `bag` — `iter_next_states`
    // borrows `&self`. At most 3 entries (one per allowed piece in bag), so
    // the allocation cost is negligible.
    let states: Vec<(TetrisPiece, TetrisPieceBagState)> = bag
        .iter_next_states()
        .filter(|(piece, _)| is_allowed_piece(*piece))
        .map(|(piece, next_bag)| (piece, normalize_narrow_bag(next_bag)))
        .collect();
    states.into_iter()
}

/// Board-level filter: reject boards that are too tall or have too many holes.
const fn board_filter(board: TetrisBoard) -> bool {
    let height = board.height();
    let holes = board.total_holes();
    height <= 10 && holes <= 10
}

fn placement_fits(placement: TetrisPiecePlacement) -> bool {
    let width = usize::from(placement.piece.width(placement.orientation.rotation));
    placement.orientation.column.index() + width <= NarrowTetris::COLS
}

// --- Channel & Threading ---
const CHANNEL_CAPACITY: usize = 256;

// --- Persistence ---
const LOG_EVERY_SECS: u64 = 3;
const SAVE_EVERY_SECS: u64 = 600;
const FIRST_SAVE_DELAY_SECS: u64 = 3600;
const SAVE_FILE: &str = "tetris_atlas_narrow.bin";
const CSV_FILE: &str = "tetris_atlas_narrow.csv";

// --- Data Structures ---

/// A lookup entry stores the chosen placement.
/// Alternatives are regenerated on demand via `generate_scored_placements`.
#[derive(Clone)]
struct PlacementEntry {
    chosen: TetrisPieceOrientation,
    result_board: TetrisBoard,
    result_bag: TetrisPieceBagState,
    depth: u16,
}

/// Reverse index: result_board → list of parents pointing to it.
#[derive(Clone, Copy)]
struct ParentRef {
    parent_board: TetrisBoard,
    parent_piece: TetrisPiece,
    result_bag: TetrisPieceBagState,
}

#[derive(Parser)]
#[command(name = "tetris-atlas-narrow")]
#[command(about = "Narrow Tetris Atlas (4-column, L/J/O) with backtracking death propagation")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
    #[arg(short, long, default_value = ".")]
    path: String,
}

#[derive(Subcommand)]
enum Mode {
    /// Create a new atlas with backtracking
    Create,
    /// Explore an existing atlas interactively
    Explore,
}

// --- Frontier ---

#[derive(Clone, Copy)]
struct FrontierValue {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    depth: u16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    depth: u16,
    hash: u64,
}

impl FrontierKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState, depth: u16) -> Self {
        Self {
            depth,
            hash: hash_board_bag(board, bag),
        }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.depth
            .cmp(&other.depth)
            .then(self.hash.cmp(&other.hash))
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let board_bytes: [u32; NarrowTetris::COLS] = board.as_limbs();
    let mut hash = FNV_OFFSET;
    repeat_idx_unroll!(NarrowTetris::COLS, I, {
        hash ^= board_bytes[I] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    });
    hash ^= (u8::from(bag) as u64) * 0x0101010101010101;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash
}

struct PriorityFrontier {
    map: Arc<SkipMap<FrontierKey, FrontierValue>>,
    enqueued_count: Arc<AtomicU64>,
}

impl PriorityFrontier {
    fn new(enqueued_count: Arc<AtomicU64>) -> Self {
        Self {
            map: Arc::new(SkipMap::new()),
            enqueued_count,
        }
    }

    fn push(&self, board: TetrisBoard, bag: TetrisPieceBagState, depth: u16) {
        let key = FrontierKey::new(board, bag, depth);
        if !self.map.contains_key(&key) {
            self.map.insert(key, FrontierValue { board, bag, depth });
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn pop(&self) -> Option<(TetrisBoard, TetrisPieceBagState, u16)> {
        self.map
            .pop_front()
            .map(|e| (e.value().board, e.value().bag, e.value().depth))
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

// --- Shared State ---

type LookupMap = DashMap<(TetrisBoard, TetrisPiece), PlacementEntry>;
/// Reverse index keyed by result_board only — O(1) parent lookup for death cascade.
type ParentMap = DashMap<TetrisBoard, Vec<ParentRef>>;
type DeadPieceSet = DashSet<(TetrisBoard, TetrisPiece)>;
type DeadBoardSet = DashSet<TetrisBoard>;

#[derive(Clone)]
struct SharedState {
    lookup: Arc<LookupMap>,
    parents: Arc<ParentMap>,
    dead_pieces: Arc<DeadPieceSet>,
    dead_boards: Arc<DeadBoardSet>,
    frontier: Arc<PriorityFrontier>,
    frontier_enqueued: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    shutdown_requested: Arc<AtomicBool>,
    pause_dispatch: Arc<AtomicBool>,
    processing_count: Arc<AtomicU64>,
}

impl SharedState {
    fn new() -> Self {
        let frontier_enqueued = Arc::new(AtomicU64::new(0));
        Self {
            lookup: Arc::new(DashMap::new()),
            parents: Arc::new(DashMap::new()),
            dead_pieces: Arc::new(DashSet::new()),
            dead_boards: Arc::new(DashSet::new()),
            frontier: Arc::new(PriorityFrontier::new(frontier_enqueued.clone())),
            frontier_enqueued,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            pause_dispatch: Arc::new(AtomicBool::new(false)),
            processing_count: Arc::new(AtomicU64::new(0)),
        }
    }

    fn seed_starting_state(&self) {
        self.frontier.push(TetrisBoard::new(), NARROW_BAG_FULL, 0);
    }

    fn is_root_dead(&self) -> bool {
        self.is_board_doomed(&TetrisBoard::new())
    }

    /// Mark a (board, piece) as dead. When the board becomes doomed,
    /// find all parent entries pointing to it via the reverse index,
    /// remove their stale lookup entries, and re-frontier those parents.
    fn mark_piece_dead(&self, board: TetrisBoard, piece: TetrisPiece, stats: &SharedStats) {
        if !self.dead_pieces.insert((board, piece)) {
            return;
        }
        stats.states_died.fetch_add(1, Ordering::Relaxed);

        let all_dead = ALLOWED_PIECES
            .iter()
            .all(|&p| self.dead_pieces.contains(&(board, p)));
        if all_dead {
            self.dead_boards.insert(board);
        }

        // O(1) parent lookup via reverse index.
        // Clone the list so we don't hold the shard lock while re-frontiering.
        let parent_refs: Vec<ParentRef> = match self.parents.get(&board) {
            Some(entry) => entry.value().clone(),
            None => return,
        };

        for ParentRef {
            parent_board,
            parent_piece,
            result_bag,
        } in parent_refs
        {
            if self.dead_pieces.contains(&(parent_board, parent_piece)) {
                continue;
            }

            // Atomically remove only if the entry's result is still doomed.
            let removed = self
                .lookup
                .remove_if(&(parent_board, parent_piece), |_, entry| {
                    self.is_board_doomed(&entry.result_board)
                });
            let Some((_, removed_entry)) = removed else {
                continue;
            };
            stats.backtrack_retries.fetch_add(1, Ordering::Relaxed);

            // Clean up the parent reference from the doomed board's parent list.
            // Prevents unbounded growth of stale entries in `parents`.
            if let Some(mut plist) = self.parents.get_mut(&removed_entry.result_board) {
                plist.value_mut().retain(|r| {
                    !(r.parent_board == parent_board && r.parent_piece == parent_piece)
                });
            }

            let parent_bag =
                TetrisPieceBagState::from(u8::from(result_bag) | u8::from(parent_piece));
            let parent_depth = removed_entry.depth.saturating_sub(1);

            self.frontier.push(parent_board, parent_bag, parent_depth);
        }
    }

    fn is_board_doomed(&self, board: &TetrisBoard) -> bool {
        if self.dead_boards.contains(board) {
            return true;
        }
        ALLOWED_PIECES
            .iter()
            .any(|&p| self.dead_pieces.contains(&(*board, p)))
    }

    fn is_result_dead(&self, result_board: &TetrisBoard) -> bool {
        self.is_board_doomed(result_board)
    }
}

// --- Stats ---

struct SharedStats {
    boards_expanded: AtomicU64,
    lookup_hits: AtomicU64,
    lookup_misses: AtomicU64,
    lookup_inserts: AtomicU64,
    games_lost: AtomicU64,
    states_died: AtomicU64,
    backtrack_retries: AtomicU64,
    backtrack_successes: AtomicU64,
    death_propagations: AtomicU64,
}

impl SharedStats {
    fn new() -> Self {
        Self {
            boards_expanded: AtomicU64::new(0),
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
            lookup_inserts: AtomicU64::new(0),
            games_lost: AtomicU64::new(0),
            states_died: AtomicU64::new(0),
            backtrack_retries: AtomicU64::new(0),
            backtrack_successes: AtomicU64::new(0),
            death_propagations: AtomicU64::new(0),
        }
    }
}

// --- Placement Generation ---

/// Generate all valid placements for `(board, piece)`, ranked by a deterministic
/// one-ply heuristic. Fewer holes, lower height, and lower roughness are preferred;
/// line clears and the orientation index provide stable tie-breakers.
fn generate_scored_placements(
    board: TetrisBoard,
    piece: TetrisPiece,
) -> Vec<TetrisPieceOrientation> {
    let mut scored: Vec<(TetrisPieceOrientation, u32, u32, u32, u32)> = Vec::new();
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        if !placement_fits(placement) {
            continue;
        }
        let mut test_board = board;
        let result = test_board.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            continue;
        }
        if !board_filter(test_board) {
            continue;
        }
        scored.push((
            placement.orientation,
            test_board.total_holes(),
            test_board.height(),
            test_board.roughness(),
            result.lines_cleared,
        ));
    }

    scored.sort_unstable_by(|a, b| {
        a.1.cmp(&b.1)
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(b.4.cmp(&a.4))
            .then(a.0.index().cmp(&b.0.index()))
    });
    scored.into_iter().map(|(o, _, _, _, _)| o).collect()
}

/// Apply an orientation to a board, return the resulting board or None if lost
/// or if the result violates the board filter.
fn apply_orientation(
    board: TetrisBoard,
    piece: TetrisPiece,
    orientation: TetrisPieceOrientation,
) -> Option<TetrisBoard> {
    let placement = TetrisPiecePlacement { piece, orientation };
    if !placement_fits(placement) {
        return None;
    }
    let mut new_board = board;
    let result = new_board.apply_piece_placement(placement);
    if result.is_lost == IsLost::LOST {
        return None;
    }
    if !board_filter(new_board) {
        return None;
    }
    Some(new_board)
}

// --- Forward Expansion ---

fn process_board_state(
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    depth: u16,
    state: &SharedState,
    stats: &SharedStats,
) -> Vec<(TetrisBoard, TetrisPieceBagState, u16)> {
    let mut new_states = Vec::new();
    let child_depth = depth + 1;

    stats.boards_expanded.fetch_add(1, Ordering::Relaxed);

    // If board is already doomed, skip entirely
    if state.is_board_doomed(&board) {
        return new_states;
    }

    for (piece, next_bag) in narrow_iter_next_states(bag) {
        // Skip if already known dead at piece level
        if state.dead_pieces.contains(&(board, piece)) {
            continue;
        }

        // Re-check: a previous piece in this loop might have just died,
        // making this board doomed
        if state.is_board_doomed(&board) {
            return new_states;
        }

        // Check existing entry — if it points to a doomed board, remove it
        // so we re-evaluate with a new placement.
        let mut is_re_eval = false;
        let mut old_result_board: Option<TetrisBoard> = None;
        if state.lookup.contains_key(&(board, piece)) {
            let removed = state.lookup.remove_if(&(board, piece), |_, entry| {
                state.is_board_doomed(&entry.result_board)
            });
            if let Some((_, entry)) = removed {
                is_re_eval = true;
                old_result_board = Some(entry.result_board);
            } else if state.lookup.contains_key(&(board, piece)) {
                stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
                continue;
            } else {
                is_re_eval = true;
            }
        }
        stats.lookup_misses.fetch_add(1, Ordering::Relaxed);

        let all_orientations = generate_scored_placements(board, piece);
        if all_orientations.is_empty() {
            stats.games_lost.fetch_add(1, Ordering::Relaxed);
            state.mark_piece_dead(board, piece, stats);
            continue;
        }

        let mut chosen_idx = None;
        for (i, &orientation) in all_orientations.iter().enumerate() {
            let Some(result_board) = apply_orientation(board, piece, orientation) else {
                continue;
            };
            if !state.is_result_dead(&result_board) {
                chosen_idx = Some(i);
                break;
            }
        }

        let Some(idx) = chosen_idx else {
            stats.games_lost.fetch_add(1, Ordering::Relaxed);
            state.mark_piece_dead(board, piece, stats);
            continue;
        };

        let chosen = all_orientations[idx];
        let result_board = apply_orientation(board, piece, chosen).expect("already validated");

        match state.lookup.entry((board, piece)) {
            Entry::Vacant(vacant) => {
                if state.dead_pieces.contains(&(board, piece)) {
                    continue;
                }
                vacant.insert(PlacementEntry {
                    chosen,
                    result_board,
                    result_bag: next_bag,
                    depth: child_depth,
                });
                stats.lookup_inserts.fetch_add(1, Ordering::Relaxed);
                if is_re_eval {
                    stats.backtrack_successes.fetch_add(1, Ordering::Relaxed);
                }

                // Clean up old parent ref if this is a re-eval pointing elsewhere
                if let Some(old_rb) = old_result_board {
                    if old_rb != result_board {
                        if let Some(mut plist) = state.parents.get_mut(&old_rb) {
                            plist
                                .value_mut()
                                .retain(|r| !(r.parent_board == board && r.parent_piece == piece));
                        }
                    }
                }

                state
                    .parents
                    .entry(result_board)
                    .or_default()
                    .push(ParentRef {
                        parent_board: board,
                        parent_piece: piece,
                        result_bag: next_bag,
                    });

                new_states.push((result_board, next_bag, child_depth));
            }
            Entry::Occupied(_) => {}
        }
    }

    new_states
}

// --- Dispatcher ---

fn dispatcher_thread(
    state: SharedState,
    sender: Sender<(TetrisBoard, TetrisPieceBagState, u16)>,
    frontier_consumed: Arc<AtomicU64>,
    stats: Arc<SharedStats>,
) {
    eprintln!("Dispatcher: started");

    // Convergence detection: require multiple consecutive stable snapshots.
    let mut stable_checks = 0u32;
    let mut last_snapshot = (0u64, 0u64); // (lookup_inserts, states_died)
    const REQUIRED_STABLE_CHECKS: u32 = 10;

    while !state.shutdown_requested.load(Ordering::Relaxed) {
        state.processing_count.fetch_add(1, Ordering::SeqCst);

        if state.pause_dispatch.load(Ordering::SeqCst) {
            state.processing_count.fetch_sub(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(10));
            stable_checks = 0;
            continue;
        }

        // Dispatcher only sends frontier items. Workers drain death queue directly.
        if let Some((board, bag, depth)) = state.frontier.pop() {
            frontier_consumed.fetch_add(1, Ordering::Relaxed);
            if sender.send((board, bag, depth)).is_err() {
                state.processing_count.fetch_sub(1, Ordering::Relaxed);
                break;
            }
            stable_checks = 0;
        } else {
            state.processing_count.fetch_sub(1, Ordering::Relaxed);

            let frontier_empty = state.frontier.len() == 0;
            let death_empty = true; // no death queue in re-frontier approach
            let no_inflight = state.processing_count.load(Ordering::SeqCst) == 0;
            let snapshot = (
                stats.lookup_inserts.load(Ordering::Relaxed),
                stats.states_died.load(Ordering::Relaxed),
            );

            if frontier_empty && death_empty && no_inflight && snapshot == last_snapshot {
                stable_checks += 1;
            } else {
                stable_checks = 0;
            }
            last_snapshot = snapshot;

            if stable_checks >= REQUIRED_STABLE_CHECKS {
                eprintln!("Dispatcher: converged ({REQUIRED_STABLE_CHECKS} stable checks)");
                state.shutdown_requested.store(true, Ordering::Relaxed);
                break;
            }

            thread::sleep(Duration::from_millis(50));
        }
    }

    drop(sender);
    eprintln!("Dispatcher: exiting");
}

// --- Logger ---

struct LoggerContext {
    lookup: Arc<LookupMap>,
    parents: Arc<ParentMap>,
    frontier: Arc<PriorityFrontier>,
    stats: Arc<SharedStats>,
    dead_pieces: Arc<DeadPieceSet>,
    dead_boards: Arc<DeadBoardSet>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    shutdown_requested: Arc<AtomicBool>,
    start: Instant,
}

struct CsvLogger {
    file: BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        let file_exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);
        if !file_exists {
            writeln!(
                writer,
                "timestamp_secs,boards_expanded,lookup_size,frontier_size,dead_states,\
                boards_per_sec,lookup_inserts,games_lost,states_died,\
                backtrack_retries,backtrack_successes,death_propagations"
            )?;
        }
        Ok(Self { file: writer })
    }

    fn log(&mut self, ctx: &LoggerContext) -> std::io::Result<()> {
        let secs = ctx.start.elapsed().as_secs_f64();
        let boards = ctx.stats.boards_expanded.load(Ordering::Relaxed);
        writeln!(
            self.file,
            "{:.3},{},{},{},{},{:.2},{},{},{},{},{},{}",
            secs,
            boards,
            ctx.lookup.len(),
            ctx.frontier.len(),
            ctx.dead_pieces.len(),
            boards as f64 / secs.max(1e-9),
            ctx.stats.lookup_inserts.load(Ordering::Relaxed),
            ctx.stats.games_lost.load(Ordering::Relaxed),
            ctx.stats.states_died.load(Ordering::Relaxed),
            ctx.stats.backtrack_retries.load(Ordering::Relaxed),
            ctx.stats.backtrack_successes.load(Ordering::Relaxed),
            ctx.stats.death_propagations.load(Ordering::Relaxed),
        )?;
        self.file.flush()
    }
}

fn print_stats(ctx: &LoggerContext, last_sample_at: &mut u64) {
    let secs = ctx.start.elapsed().as_secs_f64().max(1e-9);
    let boards = ctx.stats.boards_expanded.load(Ordering::Relaxed);
    let lookup_size = ctx.lookup.len();
    let frontier_size = ctx.frontier.len();
    let dead = ctx.stats.states_died.load(Ordering::Relaxed);
    let retries = ctx.stats.backtrack_retries.load(Ordering::Relaxed);
    let bt_ok = ctx.stats.backtrack_successes.load(Ordering::Relaxed);
    let propagations = ctx.stats.death_propagations.load(Ordering::Relaxed);
    let inserts = ctx.stats.lookup_inserts.load(Ordering::Relaxed);
    let lost = ctx.stats.games_lost.load(Ordering::Relaxed);

    let death_q = 0usize; // no death queue
    let hits = ctx.stats.lookup_hits.load(Ordering::Relaxed);
    let misses = ctx.stats.lookup_misses.load(Ordering::Relaxed);
    let cache_pct = if hits + misses > 0 {
        100.0 * hits as f64 / (hits + misses) as f64
    } else {
        0.0
    };

    // Find max depth in frontier
    let max_depth = ctx
        .frontier
        .map
        .back()
        .map(|e| e.value().depth)
        .unwrap_or(0);

    println!(
        "t={secs:.1}s boards={boards} lookup={lookup_size} frontier={frontier_size} \
        dead={dead} depth={max_depth} cache={cache_pct:.2}% hits={hits} misses={misses} | rate={:.1}/s inserts={inserts} lost={lost} | \
        retries={retries} bt_ok={bt_ok} propagations={propagations}",
        boards as f64 / secs,
    );

    let _ = last_sample_at;
}

/// Compact hex ID from board column limbs — uniquely identifies a board.
fn board_id(board: &TetrisBoard) -> String {
    let limbs = board.as_limbs();
    limbs
        .iter()
        .map(|l| format!("{l:05x}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// Compact single-line board representation showing only occupied rows.
fn board_oneliner(board: &TetrisBoard) -> String {
    let h = board.height() as usize;
    if h == 0 {
        return "[empty]".to_string();
    }
    let mut rows = Vec::with_capacity(h);
    for row in (0..h).rev() {
        let mut s = String::with_capacity(NarrowTetris::COLS);
        for col in 0..NarrowTetris::COLS {
            s.push(if board.get_bit(col, row) { '#' } else { '.' });
        }
        rows.push(s);
    }
    format!("[{}]", rows.join("|"))
}

/// Walk parent pointers backwards from a (board, bag) to reconstruct
/// how we got there. Returns vec of (piece_played, resulting_board)
/// from most recent to oldest (up to max_depth).
fn reconstruct_lineage(
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    parents: &ParentMap,
    _lookup: &LookupMap,
    max_depth: usize,
) -> Vec<(TetrisPiece, TetrisBoard, TetrisPieceBagState)> {
    use std::collections::HashSet;
    let mut lineage = Vec::new();
    let mut cur_board = board;
    let mut cur_bag = bag;
    let mut visited: HashSet<(TetrisBoard, TetrisPieceBagState)> = HashSet::new();
    visited.insert((cur_board, cur_bag));

    for _ in 0..max_depth {
        let parent_ref = match parents.get(&cur_board) {
            Some(entry) => {
                let list = entry.value();
                // STRICT: only accept a parent_ref whose result_bag matches cur_bag.
                // Other refs reached cur_board via different bag paths.
                // Among matching refs, prefer non-self, non-visited.
                let candidates: Vec<&ParentRef> =
                    list.iter().filter(|r| r.result_bag == cur_bag).collect();
                if candidates.is_empty() {
                    break; // no valid parent for this exact (board, bag)
                }
                // Prefer non-self, non-visited from the matching set
                let next_state_of = |r: &ParentRef| -> (TetrisBoard, TetrisPieceBagState) {
                    (
                        r.parent_board,
                        TetrisPieceBagState::from(
                            u8::from(r.result_bag) | u8::from(r.parent_piece),
                        ),
                    )
                };
                let best = candidates
                    .iter()
                    .find(|r| r.parent_board != cur_board && !visited.contains(&next_state_of(r)))
                    .or_else(|| candidates.iter().find(|r| r.parent_board != cur_board))
                    .or_else(|| candidates.first())
                    .copied()
                    .copied();
                match best {
                    Some(r) => r,
                    None => break,
                }
            }
            None => break,
        };

        lineage.push((parent_ref.parent_piece, cur_board, cur_bag));

        let parent_bag = TetrisPieceBagState::from(
            u8::from(parent_ref.result_bag) | u8::from(parent_ref.parent_piece),
        );
        let next_state = (parent_ref.parent_board, parent_bag);

        // Stop on cycle detection
        if !visited.insert(next_state) {
            break;
        }

        cur_board = next_state.0;
        cur_bag = next_state.1;

        if cur_board == TetrisBoard::new() {
            break;
        }
    }

    lineage
}

fn logger_thread(ctx: LoggerContext, state: SharedState, save_file: String, csv_file: String) {
    let mut last_logged = Instant::now();
    let mut last_saved = Instant::now();
    let mut first_save_done = false;
    let mut last_sample_at = 0u64;

    let mut csv_logger = CsvLogger::new(&csv_file).ok();
    if csv_logger.is_some() {
        println!("CSV logging to: {csv_file}");
    }

    while !ctx.shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(1));

        if last_logged.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            print_stats(&ctx, &mut last_sample_at);

            if let Some(ref mut logger) = csv_logger {
                let _ = logger.log(&ctx);
            }

            // Don't check is_root_dead() here — concurrent workers may
            // mark pieces dead temporarily during backtracking. Root death
            // is only authoritative after the drain completes.

            last_logged = Instant::now();
        }

        let save_interval = if !first_save_done {
            FIRST_SAVE_DELAY_SECS
        } else {
            SAVE_EVERY_SECS
        };
        if !ctx.shutdown_requested.load(Ordering::Relaxed)
            && last_saved.elapsed() >= Duration::from_secs(save_interval)
        {
            // Save is complex with new data structures — skip for now
            // TODO: implement save/load for PlacementEntry + parents + dead_states
            last_saved = Instant::now();
            first_save_done = true;
        }
    }

    println!("Logger: final stats...");
    print_stats(&ctx, &mut last_sample_at);
    if let Some(ref mut logger) = csv_logger {
        let _ = logger.log(&ctx);
    }
    println!("Logger: exiting");
}

// --- Main ---

#[derive(Parser)]
#[command(name = "tetris-atlas-narrow")]
#[command(about = "Narrow Tetris Atlas with backtracking")]
struct Cli2 {
    #[command(subcommand)]
    mode: Mode2,
    #[arg(short, long, default_value = ".")]
    path: String,
}

#[derive(Subcommand)]
enum Mode2 {
    Create,
}

fn main() {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Create => run_create(&cli.path),
        Mode::Explore => run_explore(&cli.path),
    }
}

fn run_create(base_path: &str) {
    let num_workers = num_cpus::get().saturating_sub(3).max(1); // -3 for dispatcher, logger, death_propagation
    let csv_file = format!("{}/{}", base_path, CSV_FILE);

    println!("=== Narrow Backtracking Atlas Builder ===");
    println!("  Board: {}×{}", NarrowTetris::COLS, NarrowTetris::ROWS);
    println!("  Pieces: L, J, O (3)");
    println!("  Workers: {num_workers}");
    println!("  Channel: {CHANNEL_CAPACITY}");
    println!("  Placement ordering: deterministic one-ply heuristic");
    println!("Press Ctrl+C to stop gracefully...");

    let state = SharedState::new();
    state.seed_starting_state();

    let frontier = state.frontier.clone();
    let frontier_enqueued = state.frontier_enqueued.clone();
    let frontier_consumed = Arc::new(AtomicU64::new(0));
    let stats = Arc::new(SharedStats::new());
    let shutdown = state.shutdown.clone();
    let shutdown_requested = state.shutdown_requested.clone();
    let pause_dispatch = state.pause_dispatch.clone();
    let processing_count = state.processing_count.clone();
    let start = Instant::now();

    let (tx, rx) = bounded::<(TetrisBoard, TetrisPieceBagState, u16)>(CHANNEL_CAPACITY);

    // Ctrl+C
    let sr = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C — shutting down...");
        sr.store(true, Ordering::Relaxed);
    })
    .ok();

    // Dispatcher
    let d_state = state.clone();
    let d_consumed = frontier_consumed.clone();
    let d_stats = stats.clone();
    let dispatcher = thread::spawn(move || {
        dispatcher_thread(d_state, tx, d_consumed, d_stats);
    });

    // Logger
    let logger_ctx = LoggerContext {
        lookup: state.lookup.clone(),
        parents: state.parents.clone(),
        frontier: frontier.clone(),
        stats: stats.clone(),
        dead_pieces: state.dead_pieces.clone(),
        dead_boards: state.dead_boards.clone(),
        frontier_enqueued: frontier_enqueued.clone(),
        frontier_consumed: frontier_consumed.clone(),
        shutdown: shutdown.clone(),
        shutdown_requested: shutdown_requested.clone(),
        start,
    };
    let logger_state = state.clone();
    let logger = thread::spawn(move || {
        logger_thread(logger_ctx, logger_state, String::new(), csv_file);
    });

    // Workers
    println!("Starting {num_workers} workers...");
    rayon::scope(|s| {
        for worker_id in 0..num_workers {
            let rx = rx.clone();
            let state = state.clone();
            let stats = stats.clone();
            let processing_count = processing_count.clone();

            s.spawn(move |_| {
                println!("Worker {worker_id}: started");
                while let Ok((board, bag, depth)) = rx.recv() {
                    if !state.is_board_doomed(&board) {
                        let new_states = process_board_state(board, bag, depth, &state, &stats);
                        for (new_board, new_bag, new_depth) in new_states {
                            state.frontier.push(new_board, new_bag, new_depth);
                        }
                    }
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                }
                println!("Worker {worker_id}: exiting");
            });
        }
    });

    println!("All workers finished. Shutting down...");
    shutdown.store(true, Ordering::Relaxed);

    dispatcher.join().ok();
    logger.join().ok();

    // Final drain: process any remaining frontier items synchronously.
    // mark_piece_dead pushes to frontier, so we drain until stable.
    println!("\n--- Final drain ---");
    let mut drain_iters = 0u64;
    loop {
        let Some((board, bag, depth)) = state.frontier.pop() else {
            break;
        };
        if state.is_board_doomed(&board) {
            continue;
        }
        let new_states = process_board_state(board, bag, depth, &state, &stats);
        for (new_board, new_bag, new_depth) in new_states {
            state.frontier.push(new_board, new_bag, new_depth);
        }
        drain_iters += 1;
        if drain_iters % 10000 == 0 {
            println!(
                "  drain: iters={drain_iters} lookup={} dead_pieces={} frontier={}",
                state.lookup.len(),
                state.dead_pieces.len(),
                state.frontier.len(),
            );
        }
    }
    println!("Final drain complete ({drain_iters} iterations)");

    println!("\n=== Final ===");
    println!("Lookup: {}", state.lookup.len());
    println!("Dead states: {}", state.dead_pieces.len());
    println!("Root dead: {}", state.is_root_dead());
    println!("Time: {:.1}s", start.elapsed().as_secs_f64());

    if state.is_root_dead() {
        println!("\n*** RESULT: UNSOLVABLE — no infinite play strategy exists ***");
    } else if state.frontier.len() == 0 {
        println!("\n--- Running verification ---");
        let result = verify_atlas(&state);
        match result {
            VerifyResult::Closed {
                reachable,
                entries_checked,
            } => {
                println!("*** VERIFIED SOLVED: atlas is closed ***");
                println!("  reachable states: {reachable}");
                println!("  lookup entries checked: {entries_checked}");
            }
            VerifyResult::Broken {
                board,
                piece,
                reason,
            } => {
                println!("*** VERIFICATION FAILED ***");
                println!("  board: {board}");
                println!("  piece: {piece}");
                println!("  reason: {reason}");
            }
        }
    } else {
        println!("\n*** RESULT: INCOMPLETE — stopped before convergence ***");
    }

    // Save lookup to disk
    let save_file = format!("{}/{}", base_path, SAVE_FILE);
    if let Err(e) = save_lookup(&state.lookup, &save_file) {
        eprintln!("Failed to save: {e}");
    } else {
        println!("Saved {} entries to {save_file}", state.lookup.len());
    }
}

// --- Save / Load ---

fn save_lookup(lookup: &LookupMap, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let count = lookup.len() as u64;
    w.write_all(&count.to_le_bytes())?;

    for entry in lookup.iter() {
        let (board, piece) = *entry.key();
        let pe = entry.value();
        let board_bytes: [u8; NarrowTetris::BOARD_BYTES] = board.into();
        w.write_all(&board_bytes)?;
        w.write_all(&[piece.into()])?;
        w.write_all(&[pe.chosen.index()])?;
        let result_bytes: [u8; NarrowTetris::BOARD_BYTES] = pe.result_board.into();
        w.write_all(&result_bytes)?;
        w.write_all(&[pe.result_bag.into()])?;
    }
    w.flush()?;
    Ok(())
}

fn load_lookup(path: &str) -> Result<LookupMap, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut count_buf = [0u8; 8];
    r.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let lookup = DashMap::with_capacity(count);
    const BOARD_BYTES: usize = NarrowTetris::BOARD_BYTES;

    for _ in 0..count {
        let mut board_bytes = [0u8; BOARD_BYTES];
        r.read_exact(&mut board_bytes)?;
        let mut piece_byte = [0u8; 1];
        r.read_exact(&mut piece_byte)?;
        let mut orient_byte = [0u8; 1];
        r.read_exact(&mut orient_byte)?;
        let mut result_bytes = [0u8; BOARD_BYTES];
        r.read_exact(&mut result_bytes)?;
        let mut bag_byte = [0u8; 1];
        r.read_exact(&mut bag_byte)?;

        let board = TetrisBoard::from(board_bytes);
        let piece = TetrisPiece::from(piece_byte[0]);
        let orientation = TetrisPieceOrientation::from_index(orient_byte[0]);
        let result_board = TetrisBoard::from(result_bytes);
        let result_bag = TetrisPieceBagState::from(bag_byte[0]);

        lookup.insert(
            (board, piece),
            PlacementEntry {
                chosen: orientation,
                result_board,
                result_bag,
                depth: 0,
            },
        );
    }

    Ok(lookup)
}

// --- Explore ---

fn run_explore(base_path: &str) {
    use crossterm::{
        event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{
        Terminal,
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        widgets::{Block, Borders, List, ListItem, Paragraph},
    };
    use std::io;

    let save_file = format!("{}/{}", base_path, SAVE_FILE);
    let lookup = match load_lookup(&save_file) {
        Ok(l) => {
            if l.is_empty() {
                eprintln!("Atlas is empty! Run 'create' first.");
                std::process::exit(1);
            }
            println!("Loaded {} lookup entries", l.len());
            Arc::new(l)
        }
        Err(e) => {
            eprintln!("Failed to load atlas: {e}");
            std::process::exit(1);
        }
    };

    #[derive(Clone)]
    struct Timeline {
        entries: Vec<TimelineEntry>,
        current_position: usize,
    }

    #[derive(Clone)]
    struct TimelineEntry {
        board: TetrisBoard,
        bag: TetrisPieceBagState,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
        lines_cleared: usize,
    }

    impl Timeline {
        fn new() -> Self {
            Self {
                entries: vec![],
                current_position: 0,
            }
        }
        fn current_board(&self) -> TetrisBoard {
            if self.current_position == 0 {
                TetrisBoard::new()
            } else {
                self.entries[self.current_position - 1].board
            }
        }
        fn current_bag(&self) -> TetrisPieceBagState {
            // Always return a normalized narrow bag — defensive in case the
            // stored bag came from an older save file (pre-normalization).
            let raw = if self.current_position == 0 {
                NARROW_BAG_FULL
            } else {
                self.entries[self.current_position - 1].bag
            };
            normalize_narrow_bag(raw)
        }
        fn total_lines_cleared(&self) -> usize {
            if self.current_position == 0 {
                0
            } else {
                self.entries[self.current_position - 1].lines_cleared
            }
        }
        fn push(
            &mut self,
            board: TetrisBoard,
            bag: TetrisPieceBagState,
            piece: TetrisPiece,
            orientation: TetrisPieceOrientation,
            lines_cleared: usize,
        ) {
            self.entries.truncate(self.current_position);
            self.entries.push(TimelineEntry {
                board,
                bag: normalize_narrow_bag(bag),
                piece,
                orientation,
                lines_cleared,
            });
            self.current_position = self.entries.len();
        }
        fn go_back(&mut self) {
            if self.current_position > 0 {
                self.current_position -= 1;
            }
        }
        fn go_forward(&mut self) {
            if self.current_position < self.entries.len() {
                self.current_position += 1;
            }
        }
        fn position_string(&self) -> String {
            format!("{}/{}", self.current_position, self.entries.len())
        }
    }

    let mut timeline = Timeline::new();

    enable_raw_mode().expect("Failed to enable raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).ok();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let board = timeline.current_board();
            let bag = timeline.current_bag();
            let filled_cells = board.count();
            let height = board.height();
            let holes: u32 = board.holes().iter().sum();

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(0), Constraint::Length(3)])
                    .split(f.area());
                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(chunks[0]);

                // Board
                let board_widget = Paragraph::new(format!("{board}"))
                    .block(Block::default().borders(Borders::ALL).title("Board"));
                f.render_widget(board_widget, main_chunks[0]);

                // Stats + Bag
                let right_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(8), Constraint::Min(0)])
                    .split(main_chunks[1]);

                let stats_text = format!(
                    "Position: {}\nFilled Cells: {}\nHeight: {}\nHoles: {}\nLines Cleared: {}",
                    timeline.position_string(),
                    filled_cells,
                    height,
                    holes,
                    timeline.total_lines_cleared(),
                );
                let stats_widget = Paragraph::new(stats_text)
                    .block(Block::default().borders(Borders::ALL).title("Statistics"));
                f.render_widget(stats_widget, right_chunks[0]);

                // Piece list
                let mut piece_items: Vec<ListItem> = vec![];
                for (idx, &piece) in ALLOWED_PIECES.iter().enumerate() {
                    let in_bag = bag.contains(piece);
                    let in_db = in_bag && lookup.contains_key(&(board, piece));
                    let style = if !in_bag {
                        Style::default().fg(Color::DarkGray)
                    } else if in_db {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::Red)
                    };
                    let status = if !in_bag {
                        " (not in bag)"
                    } else if in_db {
                        " (in DB)"
                    } else {
                        " (NOT in DB)"
                    };
                    piece_items.push(
                        ListItem::new(format!("{}. {}{}", idx + 1, piece, status)).style(style),
                    );
                }
                let bag_widget = List::new(piece_items)
                    .block(Block::default().borders(Borders::ALL).title("Select Piece"));
                f.render_widget(bag_widget, right_chunks[1]);

                let controls =
                    Paragraph::new("← → Navigate | 1-4 or I/O/L/J Select | R Random | Q Quit")
                        .block(Block::default().borders(Borders::ALL).title("Controls"));
                f.render_widget(controls, chunks[1]);
            })?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Left => timeline.go_back(),
                        KeyCode::Right => timeline.go_forward(),
                        KeyCode::Char(c) => {
                            let piece_opt = match c {
                                '1' | 'i' | 'I' => Some(TetrisPiece::I_PIECE),
                                '2' | 'o' | 'O' => Some(TetrisPiece::O_PIECE),
                                '3' | 'l' | 'L' => Some(TetrisPiece::L_PIECE),
                                '4' | 'j' | 'J' => Some(TetrisPiece::J_PIECE),
                                'r' | 'R' => {
                                    let available: Vec<TetrisPiece> = ALLOWED_PIECES
                                        .iter()
                                        .copied()
                                        .filter(|&p| {
                                            bag.contains(p) && lookup.contains_key(&(board, p))
                                        })
                                        .collect();
                                    if !available.is_empty() {
                                        use rand::Rng;
                                        let mut rng = rand::rng();
                                        Some(available[rng.random_range(0..available.len())])
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            };

                            if let Some(piece) = piece_opt {
                                if bag.contains(piece) {
                                    if let Some(entry) = lookup.get(&(board, piece)) {
                                        let placement = TetrisPiecePlacement {
                                            piece,
                                            orientation: entry.chosen,
                                        };
                                        let mut new_board = board;
                                        let result = new_board.apply_piece_placement(placement);
                                        if result.is_lost != IsLost::LOST {
                                            let lines_cleared = timeline.total_lines_cleared()
                                                + result.lines_cleared as usize;
                                            timeline.push(
                                                new_board,
                                                entry.result_bag,
                                                piece,
                                                entry.chosen,
                                                lines_cleared,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    })();

    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    terminal.show_cursor().ok();

    if let Err(e) = result {
        eprintln!("Error: {e}");
    }
}

// --- Verification ---

enum VerifyResult {
    /// Atlas is closed: every reachable (board, bag) has all pieces covered,
    /// and every result board is also reachable.
    Closed {
        reachable: usize,
        entries_checked: usize,
    },
    /// Found a hole: some (board, piece) entry leads to an uncovered state.
    Broken {
        board: TetrisBoard,
        piece: TetrisPiece,
        reason: String,
    },
}

/// Walk every reachable (board, bag) from the root and verify:
/// 1. For each piece in the bag, there's a lookup entry
/// 2. The result board + next bag is also reachable (has entries for all its pieces)
/// 3. No result leads to a dead state
fn verify_atlas(state: &SharedState) -> VerifyResult {
    use std::collections::{HashSet, VecDeque};

    let root_board = TetrisBoard::new();
    let root_bag = NARROW_BAG_FULL;

    let mut visited: HashSet<(TetrisBoard, TetrisPieceBagState)> = HashSet::new();
    let mut queue: VecDeque<(TetrisBoard, TetrisPieceBagState)> = VecDeque::new();
    let mut entries_checked = 0usize;

    queue.push_back((root_board, root_bag));
    visited.insert((root_board, root_bag));

    while let Some((board, bag)) = queue.pop_front() {
        // Check: current board is not doomed
        if state.is_board_doomed(&board) {
            return VerifyResult::Broken {
                board,
                piece: ALLOWED_PIECES[0],
                reason: "reached a doomed board (piece dead on it)".to_string(),
            };
        }

        for (piece, next_bag) in narrow_iter_next_states(bag) {
            // Check: piece is not in dead_pieces (zombie entry detection)
            if state.dead_pieces.contains(&(board, piece)) {
                return VerifyResult::Broken {
                    board,
                    piece,
                    reason: "piece is in dead_pieces but board is reachable".to_string(),
                };
            }

            // Check: lookup entry exists
            let entry = match state.lookup.get(&(board, piece)) {
                Some(e) => e,
                None => {
                    return VerifyResult::Broken {
                        board,
                        piece,
                        reason: "no lookup entry for (board, piece)".to_string(),
                    };
                }
            };
            entries_checked += 1;

            let result_board = entry.result_board;
            let result_bag = entry.result_bag;

            // Check: result is not dead
            if state.is_result_dead(&result_board) {
                return VerifyResult::Broken {
                    board,
                    piece,
                    reason: format!(
                        "lookup entry leads to dead state (cells={}, height={})",
                        result_board.count(),
                        result_board.height(),
                    ),
                };
            }

            // Check: result board passes filter
            if !board_filter(result_board) {
                return VerifyResult::Broken {
                    board,
                    piece,
                    reason: format!(
                        "result board fails filter (height={}, holes={})",
                        result_board.height(),
                        result_board.total_holes(),
                    ),
                };
            }

            // Enqueue the result state for further checking
            if visited.insert((result_board, result_bag)) {
                queue.push_back((result_board, result_bag));
            }
        }
    }

    VerifyResult::Closed {
        reachable: visited.len(),
        entries_checked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_config_has_expected_geometry_and_piece_set() {
        assert_eq!(NarrowTetris::COLS, 4);
        assert_eq!(NarrowTetris::ROWS, 20);
        assert_eq!(NarrowTetris::PIECE_SET, NARROW_BAG_FULL);
        assert_eq!(narrow_iter_next_states(NARROW_BAG_FULL).count(), 3);
    }

    #[test]
    fn full_narrow_row_clears_across_four_columns() {
        let mut board = TetrisBoard::new();
        for col in 0..NarrowTetris::COLS {
            board.set_bit(col, 0);
        }

        assert_eq!(board.clear_filled_rows(), 1);
        assert_eq!(board, TetrisBoard::new());
    }

    #[test]
    fn placement_ranking_returns_only_legal_in_carrier_moves() {
        let board = TetrisBoard::new();
        let orientations = generate_scored_placements(board, TetrisPiece::O_PIECE);

        assert!(!orientations.is_empty());
        for orientation in orientations {
            assert!(apply_orientation(board, TetrisPiece::O_PIECE, orientation).is_some());
        }
    }
}
