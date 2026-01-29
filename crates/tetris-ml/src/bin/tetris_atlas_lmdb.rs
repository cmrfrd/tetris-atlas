#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(impl_trait_in_bindings)]
#![feature(const_index)]
//! # Tetris Atlas Builder - LMDB (heed) Implementation
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using beam search. The goal is to map board states to optimal
//! piece placements such that the entire state space becomes self-referential.
//!
//! ## Approach
//!
//! 1. **Parallel Workers**: Multiple worker threads claim batches of board states from a frontier queue
//! 2. **Beam Search**: For each board state, uses multi-beam search to find optimal piece placements
//! 3. **LMDB Storage**: Stores the lookup table and frontier in a persistent LMDB environment via `heed`
//! 4. **Crash Recovery**: Supports resuming from crashes by tracking claimed-but-incomplete work
//!
//! ## Output
//!
//! - **LMDB Environment**: `tetris_atlas_lmdb/` - Persistent lookup table with board → placements
//! - **Statistics CSV**: `tetris_atlas_stats_lmdb.csv` - Real-time metrics (expansion rate, cache hits, etc.)
//!
//! ## Performance Optimizations
//!
//! - Large map size and reader count
//! - Batched write transactions
//! - Partitioned workers scanning disjoint key ranges
//! - Optional LMDB flags for relaxed durability (NO_SYNC / WRITE_MAP / MAP_ASYNC)

use clap::{Parser, Subcommand};
use heed::{Database, Env, EnvFlags, EnvOpenOptions, types::Bytes};
use std::io::Write;
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    repeat_idx_unroll,
};
use tetris_ml::beam_search::{BeamTetrisState, MultiBeamSearch};
use tetris_ml::set_global_threadpool;

const N: usize = 64;
const TOP_N_PER_BEAM: usize = 32;
const BEAM_WIDTH: usize = 64;
const MAX_DEPTH: usize = 7;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

const STATS_CSV_PATH: &str = "tetris_atlas_stats_lmdb.csv";
const LOG_EVERY_SECS: u64 = 3;

const NUM_WORKERS: usize = 16;
const WORKER_FRONTIER_BATCH_SIZE: usize = 2048;

const DEFAULT_MAP_SIZE_GB: u64 = 16;
const DEFAULT_MAX_READERS: u32 = 1024;
const DEFAULT_NO_SYNC: bool = true;
const DEFAULT_NO_META_SYNC: bool = true;
const DEFAULT_WRITE_MAP: bool = true;
const DEFAULT_MAP_ASYNC: bool = true;

// Database names
const DB_LOOKUP: &str = "lookup";
const DB_FRONTIER: &str = "frontier";
const DB_META: &str = "meta";

#[derive(Parser)]
#[command(name = "tetris-atlas-lmdb")]
#[command(about = "Tetris Atlas (LMDB) - Build and explore optimal Tetris lookup tables")]
struct Cli {
    /// Path to the LMDB environment directory
    #[arg(short, long)]
    db_path: String,

    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Create a new atlas database by exhaustively exploring the state space
    Create,
    /// Explore an existing atlas database interactively
    Explore,
}

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
    // FNV-1a hash: fast, good distribution, no state needed
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let board_bytes: [u32; TetrisBoard::WIDTH] = board.as_limbs();
    let mut hash = FNV_OFFSET;
    repeat_idx_unroll!(TetrisBoard::WIDTH, I, {
        hash ^= board_bytes[I] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    });
    hash ^= (u8::from(bag) as u64) * 0x0101010101010101;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash
}

#[derive(Clone, Copy)]
struct TetrisAtlasLookupKeyValue {
    board: TetrisBoard,
    piece: TetrisPiece,
    orientation: TetrisPieceOrientation,
}

impl TetrisAtlasLookupKeyValue {
    fn new(board: TetrisBoard, piece: TetrisPiece, orientation: TetrisPieceOrientation) -> Self {
        Self {
            board,
            piece,
            orientation,
        }
    }

    #[inline(always)]
    const fn key_bytes(&self) -> [u8; 41] {
        let board_bytes: [u8; 40] = self.board.into();
        let piece_byte: u8 = self.piece.into();
        let mut key = [0u8; 41];
        key[0..40].copy_from_slice(&board_bytes);
        key[40] = piece_byte;
        key
    }

    #[inline(always)]
    const fn value_bytes(&self) -> [u8; 1] {
        [self.orientation.index()]
    }
}

// ----------------------------- Frontier Queue ----------------------------------

#[derive(Clone, Copy)]
struct TetrisAtlasFrontierKeyValue {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl std::fmt::Display for TetrisAtlasFrontierKeyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FrontierKeyValue(board={}, bag={})",
            self.board, self.bag
        )
    }
}

impl TetrisAtlasFrontierKeyValue {
    #[inline(always)]
    const fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }

    #[inline(always)]
    const fn key_bytes(&self) -> [u8; 9] {
        let height = self.board.height() as u8;
        let hash = hash_board_bag(self.board, self.bag);
        let mut key = [0u8; 9];
        key[0] = height;
        key[1..9].copy_from_slice(&hash.to_be_bytes());
        key
    }

    #[inline(always)]
    const fn value_bytes(&self) -> [u8; 41] {
        let board_bytes: [u8; 40] = self.board.into();
        let mut out = [0u8; 41];
        out[0..40].copy_from_slice(&board_bytes);
        out[40] = u8::from(self.bag);
        out
    }

    #[inline(always)]
    const fn from_value_bytes(v: &[u8; 41]) -> Self {
        let board_bytes: [u8; 40] = unsafe { *(v.as_ptr() as *const [u8; 40]) };
        let bag_byte = v[40];
        Self {
            board: TetrisBoard::from(board_bytes),
            bag: TetrisPieceBagState::from(bag_byte),
        }
    }
}

#[derive(Clone, Copy)]
struct LookupTask {
    board: TetrisBoard,
    piece: TetrisPiece,
    bag_state: TetrisPieceBagState,
}

struct LookupWrite {
    key: [u8; 41],
    value: [u8; 1],
}

struct FrontierWrite {
    key: [u8; 9],
    value: [u8; 41],
}

// ----------------------------- DB Wrapper ----------------------------------

pub struct TetrisAtlasDB {
    env: Arc<Env>,
    lookup: Database<Bytes, Bytes>,
    frontier: Database<Bytes, Bytes>,
    meta: Database<Bytes, Bytes>,
    stop: Arc<AtomicBool>,

    // Performance counters
    lookup_hits: Arc<AtomicU64>,
    lookup_misses: Arc<AtomicU64>,
    lookup_inserts: Arc<AtomicU64>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    frontier_deleted: Arc<AtomicU64>,
    games_lost: Arc<AtomicU64>,
    boards_expanded: Arc<AtomicU64>,
}

impl TetrisAtlasDB {
    pub fn new(
        path: &str,
        map_size_gb: u64,
        max_readers: u32,
        flags: EnvFlags,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(path)?;

        let map_size_bytes = map_size_gb.saturating_mul(1024 * 1024 * 1024) as usize;
        let mut options = EnvOpenOptions::new();
        options.max_dbs(3);
        options.max_readers(max_readers);
        options.map_size(map_size_bytes);
        unsafe {
            options.flags(flags);
        }

        let env = unsafe { options.open(path)? };

        let mut wtxn = env.write_txn()?;
        let lookup: Database<Bytes, Bytes> = env.create_database(&mut wtxn, Some(DB_LOOKUP))?;
        let frontier: Database<Bytes, Bytes> = env.create_database(&mut wtxn, Some(DB_FRONTIER))?;
        let meta: Database<Bytes, Bytes> = env.create_database(&mut wtxn, Some(DB_META))?;
        wtxn.commit()?;

        Ok(Self {
            env: Arc::new(env),
            lookup,
            frontier,
            meta,
            stop: Arc::new(AtomicBool::new(false)),
            lookup_hits: Arc::new(AtomicU64::new(0)),
            lookup_misses: Arc::new(AtomicU64::new(0)),
            lookup_inserts: Arc::new(AtomicU64::new(0)),
            frontier_enqueued: Arc::new(AtomicU64::new(0)),
            frontier_consumed: Arc::new(AtomicU64::new(0)),
            frontier_deleted: Arc::new(AtomicU64::new(0)),
            games_lost: Arc::new(AtomicU64::new(0)),
            boards_expanded: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    pub fn flush(&self) {
        let _ = self.env.force_sync();
    }

    /// Check if a board+piece combination exists in the lookup table
    pub fn lookup_exists(&self, board: TetrisBoard, piece: TetrisPiece) -> bool {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();
        let txn = match self.env.read_txn() {
            Ok(txn) => txn,
            Err(_) => return false,
        };
        self.lookup.get(&txn, &key).ok().flatten().is_some()
    }

    /// Get the optimal orientation for a board+piece combination
    pub fn lookup_orientation(
        &self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Option<TetrisPieceOrientation> {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();
        let txn = self.env.read_txn().ok()?;
        match self.lookup.get(&txn, &key) {
            Ok(Some(value)) => {
                if value.len() == 1 {
                    Some(TetrisPieceOrientation::from_index(value[0]))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn seed_starting_state(&self) -> Result<(), Box<dyn std::error::Error>> {
        let empty_board = TetrisBoard::new();
        let bag = TetrisPieceBagState::new();
        let frontier_item = TetrisAtlasFrontierKeyValue::new(empty_board, bag);
        let mut wtxn = self.env.write_txn()?;
        self.frontier.put(
            &mut wtxn,
            &frontier_item.key_bytes(),
            &frontier_item.value_bytes(),
        )?;
        wtxn.commit()?;
        self.frontier_enqueued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn run_worker_loop(&self, num_workers: usize) -> Vec<JoinHandle<()>> {
        let env = Arc::clone(&self.env);
        let lookup = self.lookup;
        let frontier = self.frontier;
        let stop = Arc::clone(&self.stop);
        let lookup_hits = Arc::clone(&self.lookup_hits);
        let lookup_misses = Arc::clone(&self.lookup_misses);
        let lookup_inserts = Arc::clone(&self.lookup_inserts);
        let frontier_enqueued = Arc::clone(&self.frontier_enqueued);
        let frontier_consumed = Arc::clone(&self.frontier_consumed);
        let frontier_deleted = Arc::clone(&self.frontier_deleted);
        let games_lost = Arc::clone(&self.games_lost);
        let boards_expanded = Arc::clone(&self.boards_expanded);

        // Spawn N dedicated workers, each scanning its own partition of the key space
        (0..num_workers)
            .map(|worker_id| {
                let env = Arc::clone(&env);
                let lookup = lookup;
                let frontier = frontier;
                let stop = Arc::clone(&stop);
                let lookup_hits = Arc::clone(&lookup_hits);
                let lookup_misses = Arc::clone(&lookup_misses);
                let lookup_inserts = Arc::clone(&lookup_inserts);
                let frontier_enqueued = Arc::clone(&frontier_enqueued);
                let frontier_consumed = Arc::clone(&frontier_consumed);
                let frontier_deleted = Arc::clone(&frontier_deleted);
                let games_lost = Arc::clone(&games_lost);
                let boards_expanded = Arc::clone(&boards_expanded);

                std::thread::spawn(move || {
                    const BASE_SEED: u64 = 42;

                    let mut beam_search = MultiBeamSearch::<
                        BeamTetrisState,
                        N,
                        TOP_N_PER_BEAM,
                        BEAM_WIDTH,
                        MAX_DEPTH,
                        MAX_MOVES,
                    >::new();

                    let mut game = tetris_game::TetrisGame::new();

                    // Calculate this worker's partition of the key space (based on first byte of hash)
                    // Key structure: [height(1), hash(8)] = 9 bytes total
                    // Partitioning 256 values (0..=255) across num_workers based on hash's first byte
                    let partition_start_u16 = (worker_id as u16 * 256) / num_workers as u16;
                    let partition_end_u16 = ((worker_id + 1) as u16 * 256) / num_workers as u16;

                    let partition_start = partition_start_u16 as u8;
                    let partition_end = partition_end_u16 as u8;

                    // Keys are now [u8; 9]: [height, hash_byte_0, ..., hash_byte_7]
                    // lower_bound: start at height=0, partition_start hash
                    let lower_bound = [0u8, partition_start, 0, 0, 0, 0, 0, 0, 0].to_vec();

                    // upper_bound: scan up to max height (0xFF), partition_end hash
                    // For last worker, partition_end = 256 (doesn't fit in u8), so use max sentinel
                    let upper_bound = if partition_end_u16 > 255 {
                        vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
                    } else {
                        [0xFF, partition_end, 0, 0, 0, 0, 0, 0, 0].to_vec()
                    };

                    while !stop.load(Ordering::Acquire) {
                        let mut items_processed = 0;
                        let mut lookup_tasks: Vec<LookupTask> = Vec::new();
                        let mut lookup_writes: Vec<LookupWrite> = Vec::new();
                        let mut frontier_writes: Vec<FrontierWrite> = Vec::new();
                        let mut frontier_keys_to_delete: Vec<Vec<u8>> = Vec::new();

                        {
                            let rtxn = env.read_txn().expect("failed to open read transaction");
                            let range = (
                                Bound::Included(lower_bound.as_slice()),
                                Bound::Excluded(upper_bound.as_slice()),
                            );
                            let iter = frontier
                                .range(&rtxn, &range)
                                .expect("failed to create range iterator");

                            for item in iter.take(WORKER_FRONTIER_BATCH_SIZE) {
                                let (key, value) = item.expect("failed to iterate frontier");
                                items_processed += 1;

                                frontier_keys_to_delete.push(key.to_vec());

                                if value.len() != 41 {
                                    continue;
                                }

                                let mut value_bytes = [0u8; 41];
                                value_bytes.copy_from_slice(value);
                                let current_frontier_item =
                                    TetrisAtlasFrontierKeyValue::from_value_bytes(&value_bytes);

                                frontier_consumed.fetch_add(1, Ordering::Relaxed);
                                boards_expanded.fetch_add(1, Ordering::Relaxed);

                                // Process all pieces for this board state
                                for (piece, bag_state) in
                                    current_frontier_item.bag.iter_next_states()
                                {
                                    let lookup_key = TetrisAtlasLookupKeyValue::new(
                                        current_frontier_item.board,
                                        piece,
                                        TetrisPieceOrientation::default(),
                                    )
                                    .key_bytes();

                                    match lookup.get(&rtxn, &lookup_key) {
                                        Ok(Some(_)) => {
                                            lookup_hits.fetch_add(1, Ordering::Relaxed);
                                        }
                                        Ok(None) => {
                                            lookup_misses.fetch_add(1, Ordering::Relaxed);
                                            lookup_tasks.push(LookupTask {
                                                board: current_frontier_item.board,
                                                piece,
                                                bag_state,
                                            });
                                        }
                                        Err(_) => {}
                                    }
                                }
                            }
                        }

                        for task in lookup_tasks {
                            game.board = task.board;
                            game.set_bag_piece_seeded(task.bag_state, task.piece, BASE_SEED);

                            let best_placement = match beam_search.search_with_seeds(
                                BeamTetrisState::new(game),
                                BASE_SEED,
                                MAX_DEPTH,
                            ) {
                                Some(result) => result,
                                None => {
                                    games_lost.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            };

                            if game.apply_placement(best_placement).is_lost == IsLost::LOST {
                                games_lost.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            let new_lookup_kv = TetrisAtlasLookupKeyValue::new(
                                task.board,
                                task.piece,
                                best_placement.orientation,
                            );
                            lookup_writes.push(LookupWrite {
                                key: new_lookup_kv.key_bytes(),
                                value: new_lookup_kv.value_bytes(),
                            });
                            lookup_inserts.fetch_add(1, Ordering::Relaxed);

                            let new_frontier_kv =
                                TetrisAtlasFrontierKeyValue::new(game.board, task.bag_state);
                            frontier_writes.push(FrontierWrite {
                                key: new_frontier_kv.key_bytes(),
                                value: new_frontier_kv.value_bytes(),
                            });
                            frontier_enqueued.fetch_add(1, Ordering::Relaxed);
                        }

                        if !lookup_writes.is_empty()
                            || !frontier_writes.is_empty()
                            || !frontier_keys_to_delete.is_empty()
                        {
                            let mut wtxn =
                                env.write_txn().expect("failed to open write transaction");

                            for entry in lookup_writes {
                                lookup
                                    .put(&mut wtxn, &entry.key, &entry.value)
                                    .expect("failed to write lookup entry");
                            }

                            for entry in frontier_writes {
                                frontier
                                    .put(&mut wtxn, &entry.key, &entry.value)
                                    .expect("failed to write frontier entry");
                            }

                            for key in &frontier_keys_to_delete {
                                let _ = frontier.delete(&mut wtxn, key);
                            }

                            wtxn.commit().expect("failed to commit write batch");
                            frontier_deleted
                                .fetch_add(frontier_keys_to_delete.len() as u64, Ordering::Relaxed);
                        }

                        if items_processed == 0 {
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                })
            })
            .collect()
    }

    /// Check if the system is idle (no items in frontier queue)
    pub fn is_idle(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let count = self.frontier_size()?;
        Ok(count == 0)
    }

    /// Get frontier queue size
    fn frontier_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let txn = self.env.read_txn()?;
        let count = self.frontier.stat(&txn)?.entries;
        Ok(count)
    }

    /// Get lookup table size
    fn lookup_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let txn = self.env.read_txn()?;
        let count = self.lookup.stat(&txn)?.entries;
        Ok(count)
    }

    /// Get all performance statistics
    #[allow(clippy::type_complexity)]
    pub fn stats(
        &self,
    ) -> Result<
        (
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64, // App counters
        ),
        Box<dyn std::error::Error>,
    > {
        let lookup_hits = self.lookup_hits.load(Ordering::Relaxed);
        let lookup_misses = self.lookup_misses.load(Ordering::Relaxed);
        let lookup_inserts = self.lookup_inserts.load(Ordering::Relaxed);
        let frontier_enqueued = self.frontier_enqueued.load(Ordering::Relaxed);
        let frontier_consumed = self.frontier_consumed.load(Ordering::Relaxed);
        let frontier_deleted = self.frontier_deleted.load(Ordering::Relaxed);
        let games_lost = self.games_lost.load(Ordering::Relaxed);
        let boards_expanded = self.boards_expanded.load(Ordering::Relaxed);

        eprintln!("=== LMDB Stats ===");
        eprintln!("App Counters:");
        eprintln!("  Lookup hits: {}", lookup_hits);
        eprintln!("  Lookup misses: {}", lookup_misses);
        eprintln!("  Lookup inserts: {}", lookup_inserts);
        eprintln!("  Frontier enqueued: {}", frontier_enqueued);
        eprintln!("  Frontier consumed: {}", frontier_consumed);
        eprintln!("  Frontier deleted: {}", frontier_deleted);
        eprintln!("  Games lost: {}", games_lost);
        eprintln!("  Boards expanded: {}", boards_expanded);
        eprintln!("===================");

        Ok((
            lookup_hits,
            lookup_misses,
            lookup_inserts,
            frontier_enqueued,
            frontier_consumed,
            frontier_deleted,
            games_lost,
            boards_expanded,
        ))
    }
}

// ----------------------------- Board Statistics ----------------------------------

#[derive(Clone, Copy, Debug)]
struct BoardStats {
    filled_cells: usize,
    height: usize,
    holes: usize,
    lines_cleared: usize,
}

impl BoardStats {
    fn calculate(board: &TetrisBoard, lines_cleared: usize) -> Self {
        let mut filled_cells = 0;
        let mut height = 0;
        let mut holes = 0;

        // Calculate filled cells and height
        for y in 0..TetrisBoard::HEIGHT {
            let mut row_has_filled = false;
            for x in 0..TetrisBoard::WIDTH {
                if board.get_bit(x, y) {
                    filled_cells += 1;
                    row_has_filled = true;
                }
            }
            if row_has_filled {
                height = y + 1;
            }
        }

        // Calculate holes (empty cells with filled cells above them)
        for x in 0..TetrisBoard::WIDTH {
            let mut found_filled = false;
            for y in (0..TetrisBoard::HEIGHT).rev() {
                if board.get_bit(x, y) {
                    found_filled = true;
                } else if found_filled {
                    holes += 1;
                }
            }
        }

        Self {
            filled_cells,
            height,
            holes,
            lines_cleared,
        }
    }
}

// ----------------------------- Timeline Tracking ----------------------------------

#[derive(Clone)]
struct TimelineEntry {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    piece: TetrisPiece,
    orientation: TetrisPieceOrientation,
    lines_cleared: usize,
}

struct Timeline {
    entries: Vec<TimelineEntry>,
    current_position: usize,
}

impl Timeline {
    fn new(_board: TetrisBoard, _bag: TetrisPieceBagState) -> Self {
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
        if self.current_position == 0 {
            TetrisPieceBagState::new()
        } else {
            self.entries[self.current_position - 1].bag
        }
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
            bag,
            piece,
            orientation,
            lines_cleared,
        });
        self.current_position = self.entries.len();
    }

    fn go_back(&mut self) -> bool {
        if self.current_position > 0 {
            self.current_position -= 1;
            true
        } else {
            false
        }
    }

    fn go_forward(&mut self) -> bool {
        if self.current_position < self.entries.len() {
            self.current_position += 1;
            true
        } else {
            false
        }
    }

    fn position_string(&self) -> String {
        format!("{}/{}", self.current_position, self.entries.len())
    }
}

// ----------------------------- Explore Mode ----------------------------------

pub fn run_tetris_atlas_explore(cli: &Cli) {
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

    let atlas = TetrisAtlasDB::new(
        &cli.db_path,
        DEFAULT_MAP_SIZE_GB,
        DEFAULT_MAX_READERS,
        default_env_flags(),
    )
    .expect("Failed to open atlas database");
    let mut timeline = Timeline::new(TetrisBoard::new(), TetrisPieceBagState::new());
    let mut game = tetris_game::TetrisGame::new();

    enable_raw_mode().expect("Failed to enable raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).expect("Failed to setup terminal");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let board = timeline.current_board();
            let bag = timeline.current_bag();
            let stats = BoardStats::calculate(&board, timeline.total_lines_cleared());

            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(0), Constraint::Length(3)])
                    .split(f.area());

                let main_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(chunks[0]);

                let board_str = format!("{}", board);
                let board_widget = Paragraph::new(board_str)
                    .block(Block::default().borders(Borders::ALL).title("Board"));
                f.render_widget(board_widget, main_chunks[0]);

                let right_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(8), Constraint::Min(0)])
                    .split(main_chunks[1]);

                let stats_text = format!(
                    "Position: {}\nFilled Cells: {}\nHeight: {}\nHoles: {}\nLines Cleared: {}",
                    timeline.position_string(),
                    stats.filled_cells,
                    stats.height,
                    stats.holes,
                    stats.lines_cleared,
                );
                let stats_widget = Paragraph::new(stats_text)
                    .block(Block::default().borders(Borders::ALL).title("Statistics"));
                f.render_widget(stats_widget, right_chunks[0]);

                let mut piece_items: Vec<ListItem> = vec![];

                for (idx, piece) in [
                    TetrisPiece::O_PIECE,
                    TetrisPiece::I_PIECE,
                    TetrisPiece::L_PIECE,
                    TetrisPiece::J_PIECE,
                    TetrisPiece::S_PIECE,
                    TetrisPiece::Z_PIECE,
                    TetrisPiece::T_PIECE,
                ]
                .iter()
                .enumerate()
                {
                    let in_bag = bag.contains(*piece);
                    let in_db = if in_bag {
                        atlas.lookup_exists(board, *piece)
                    } else {
                        false
                    };

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
                        ListItem::new(format!("{}. {:?}{}", idx + 1, piece, status)).style(style),
                    );
                }

                let bag_widget = List::new(piece_items)
                    .block(Block::default().borders(Borders::ALL).title("Select Piece"));
                f.render_widget(bag_widget, right_chunks[1]);

                let controls = Paragraph::new(
                    "← → Navigate | 1-7 or O/I/L/J/S/Z/T Select Piece | R Random | Q Quit",
                )
                .block(Block::default().borders(Borders::ALL).title("Controls"));
                f.render_widget(controls, chunks[1]);
            })?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Left => {
                            timeline.go_back();
                        }
                        KeyCode::Right => {
                            timeline.go_forward();
                        }
                        KeyCode::Char(c) => {
                            let piece_opt = match c {
                                '1' => Some(TetrisPiece::O_PIECE),
                                '2' => Some(TetrisPiece::I_PIECE),
                                '3' => Some(TetrisPiece::L_PIECE),
                                '4' => Some(TetrisPiece::J_PIECE),
                                '5' => Some(TetrisPiece::S_PIECE),
                                '6' => Some(TetrisPiece::Z_PIECE),
                                '7' => Some(TetrisPiece::T_PIECE),
                                'o' | 'O' => Some(TetrisPiece::O_PIECE),
                                'i' | 'I' => Some(TetrisPiece::I_PIECE),
                                'l' | 'L' => Some(TetrisPiece::L_PIECE),
                                'j' | 'J' => Some(TetrisPiece::J_PIECE),
                                's' | 'S' => Some(TetrisPiece::S_PIECE),
                                'z' | 'Z' => Some(TetrisPiece::Z_PIECE),
                                't' | 'T' => Some(TetrisPiece::T_PIECE),
                                'r' | 'R' => {
                                    let available: Vec<TetrisPiece> = [
                                        TetrisPiece::O_PIECE,
                                        TetrisPiece::I_PIECE,
                                        TetrisPiece::L_PIECE,
                                        TetrisPiece::J_PIECE,
                                        TetrisPiece::S_PIECE,
                                        TetrisPiece::Z_PIECE,
                                        TetrisPiece::T_PIECE,
                                    ]
                                    .iter()
                                    .copied()
                                    .filter(|&p| bag.contains(p) && atlas.lookup_exists(board, p))
                                    .collect();

                                    if !available.is_empty() {
                                        use rand::Rng;
                                        let mut rng = rand::rng();
                                        let idx = rng.random_range(0..available.len());
                                        Some(available[idx])
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            };

                            if let Some(piece) = piece_opt {
                                if bag.contains(piece) && atlas.lookup_exists(board, piece) {
                                    if let Some(orientation) =
                                        atlas.lookup_orientation(board, piece)
                                    {
                                        game.board = board;
                                        game.set_bag_piece_seeded(bag, piece, 42);

                                        let placement = tetris_game::TetrisPiecePlacement {
                                            piece,
                                            orientation,
                                        };
                                        let result = game.apply_placement(placement);
                                        if result.is_lost != IsLost::LOST {
                                            let mut new_bag = bag;
                                            new_bag.remove(piece);
                                            if new_bag.count() == 0 {
                                                new_bag.fill();
                                            }
                                            let lines_cleared = timeline.total_lines_cleared()
                                                + result.lines_cleared as usize;
                                            timeline.push(
                                                game.board,
                                                new_bag,
                                                piece,
                                                orientation,
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

    disable_raw_mode().expect("Failed to disable raw mode");
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .expect("Failed to restore terminal");
    terminal.show_cursor().expect("Failed to show cursor");

    if let Err(e) = result {
        eprintln!("Error: {}", e);
    }
}

fn default_env_flags() -> EnvFlags {
    let mut flags = EnvFlags::empty();
    if DEFAULT_NO_SYNC {
        flags |= EnvFlags::NO_SYNC;
    }
    if DEFAULT_NO_META_SYNC {
        flags |= EnvFlags::NO_META_SYNC;
    }
    if DEFAULT_WRITE_MAP {
        flags |= EnvFlags::WRITE_MAP;
    }
    if DEFAULT_MAP_ASYNC {
        flags |= EnvFlags::MAP_ASYNC;
    }
    flags
}

fn main() {
    let cli = Cli::parse();

    match cli.mode {
        Mode::Create => run_tetris_atlas_create(&cli),
        Mode::Explore => run_tetris_atlas_explore(&cli),
    }
}

pub fn run_tetris_atlas_create(cli: &Cli) {
    set_global_threadpool();

    let start = Instant::now();

    let atlas = TetrisAtlasDB::new(
        &cli.db_path,
        DEFAULT_MAP_SIZE_GB,
        DEFAULT_MAX_READERS,
        default_env_flags(),
    )
    .expect("Failed to create atlas");
    atlas
        .seed_starting_state()
        .expect("Failed to seed starting state");

    let handles: Vec<JoinHandle<()>> = atlas.run_worker_loop(NUM_WORKERS);

    {
        let shutdown_requested = Arc::clone(&atlas.stop);
        ctrlc::set_handler(move || {
            if !shutdown_requested.swap(true, Ordering::Release) {
                eprintln!("\nSIGINT received, shutting down gracefully...");
            }
        })
        .expect("Error setting Ctrl-C handler");
    }

    let csv_file = std::fs::File::create(STATS_CSV_PATH).expect("Failed to create CSV file");
    let mut csv_writer = std::io::BufWriter::new(csv_file);
    writeln!(
        csv_writer,
        "timestamp_secs,boards_expanded,lookup_hits,lookup_misses,lookup_inserts,\
        frontier_enqueued,frontier_consumed,frontier_deleted,games_lost,lookup_size,frontier_size,\
        boards_per_sec,cache_hit_rate,total_lookups,\
        frontier_consumption_rate,frontier_expansion_rate,frontier_expansion_ratio"
    )
    .expect("Failed to write CSV header");
    csv_writer.flush().expect("Failed to flush CSV header");

    let mut last_log = Instant::now();

    loop {
        thread::sleep(Duration::from_millis(50));

        if atlas.is_stopped() {
            eprintln!("Shutdown initiated, waiting for workers to finish...");
            break;
        }

        if last_log.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let (
                lookup_hits,
                lookup_misses,
                lookup_inserts,
                frontier_enqueued,
                frontier_consumed,
                frontier_deleted,
                games_lost,
                boards_expanded,
            ) = atlas.stats().expect("Failed to get stats");

            let frontier_size = atlas.frontier_size().unwrap_or(0);
            let lookup_size = atlas.lookup_size().unwrap_or(0);

            let boards_per_sec = boards_expanded as f64 / secs;

            let total_lookups = lookup_hits + lookup_misses;
            let cache_hit_rate = if total_lookups > 0 {
                (lookup_hits as f64 / total_lookups as f64) * 100.0
            } else {
                0.0
            };

            let frontier_consumption_rate = if secs > 0.0 {
                frontier_consumed as f64 / secs
            } else {
                0.0
            };

            let frontier_expansion_rate = if secs > 0.0 {
                frontier_enqueued as f64 / secs
            } else {
                0.0
            };

            let frontier_expansion_ratio =
                frontier_enqueued as f64 / frontier_consumed.max(1) as f64;

            println!(
                "t={secs:.1}s boards={boards_expanded} lookup={lookup_size} frontier={frontier_size} | \
                boards_rate={boards_per_sec:.1}/s inserts={lookup_inserts} lost={games_lost} | \
                cache_hit={cache_hit_rate:.1}% ({lookup_hits}/{total_lookups}) | \
                f_consume={frontier_consumption_rate:.1}/s f_expand={frontier_expansion_rate:.1}/s f_ratio={frontier_expansion_ratio:.2}"
            );

            writeln!(
                csv_writer,
                "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6}",
                secs,
                boards_expanded,
                lookup_hits,
                lookup_misses,
                lookup_inserts,
                frontier_enqueued,
                frontier_consumed,
                frontier_deleted,
                games_lost,
                lookup_size,
                frontier_size,
                boards_per_sec,
                cache_hit_rate,
                total_lookups,
                frontier_consumption_rate,
                frontier_expansion_rate,
                frontier_expansion_ratio
            )
            .expect("Failed to write CSV row");
            csv_writer.flush().expect("Failed to flush CSV");

            last_log = Instant::now();

            atlas.flush();
        }
    }

    for h in handles {
        let _ = h.join();
    }

    eprintln!("All workers stopped.");
    println!("Finished in {:.2} seconds", start.elapsed().as_secs_f64());
}
