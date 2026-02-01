#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Builder - In-Memory Implementation
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using beam search. Uses a channel-based architecture for
//! strict priority ordering.
//!
//! ## Approach
//!
//! 1. **Dispatcher Thread**: Single thread pops from priority frontier in strict height order
//! 2. **Bounded Channel**: Connects dispatcher to workers with backpressure
//! 3. **Worker Pool**: rayon::scope workers process items in parallel
//! 4. **Priority Frontier**: Lower-height boards processed first (strict ordering)
//! 5. **In-Memory Storage**: DashMap for lookup table, SkipMap for priority frontier
//!
//! ## Benefits
//!
//! - Strict priority ordering (dispatcher controls pop order)
//! - No contention on priority queue pop (single dispatcher)
//! - Natural backpressure via bounded channel
//! - Parallel processing via rayon workers

use clap::{Parser, Subcommand};
use crossbeam_channel::{Sender, bounded};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    repeat_idx_unroll,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch};

// --- Beam Search Parameters ---
const N: usize = 16;
const TOP_N_PER_BEAM: usize = 32;
const BEAM_WIDTH: usize = 64;
const MAX_DEPTH: usize = 4;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

// --- Channel & Threading ---
const CHANNEL_CAPACITY: usize = 64;

// --- Persistence ---
const LOG_EVERY_SECS: u64 = 3;
const SAVE_EVERY_SECS: u64 = 600;
const SAVE_FILE: &str = "tetris_atlas_inmemory.bin";
const CSV_FILE: &str = "tetris_atlas_inmemory.csv";
const BASE_SEED: u64 = 42;

#[derive(Parser)]
#[command(name = "tetris-atlas-inmemory")]
#[command(about = "Tetris Atlas (In-Memory) - Build and explore optimal Tetris lookup tables")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Create a new atlas by exhaustively exploring the state space
    Create,
    /// Explore an existing atlas interactively
    Explore,
}

/// Frontier value (similar to RocksDB implementation)
#[derive(Clone, Copy)]
struct FrontierValue {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl FrontierValue {
    #[inline(always)]
    const fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }
}

/// Frontier key that orders by height (lowest first), then by hash for uniqueness
#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    cost: u32,
    hash: u64,
}

impl FrontierKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        let cost = board.height() + board.total_holes();
        let hash = hash_board_bag(board, bag);
        Self { cost, hash }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by height (lower is better)
        match self.cost.cmp(&other.cost) {
            std::cmp::Ordering::Equal => {
                // If heights are equal, compare by hash for deterministic ordering
                self.hash.cmp(&other.hash)
            }
            other => other,
        }
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Hash function for board+bag to create unique keys
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

/// Priority frontier using SkipMap for concurrent sorted access
/// Lower heights are processed first for efficient state space exploration
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

    /// Push a (board, bag) pair into the frontier if not already present
    fn push(&self, board: TetrisBoard, bag: TetrisPieceBagState) {
        let key = FrontierKey::new(board, bag);
        let value = FrontierValue::new(board, bag);

        if !self.map.contains_key(&key) {
            self.map.insert(key, value);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Pop one entry from the front of the map
    /// Returns the item with the lowest height
    fn pop(&self) -> Option<(TetrisBoard, TetrisPieceBagState)> {
        self.map.pop_front().map(|entry| {
            let value = entry.value();
            (value.board, value.bag)
        })
    }

    /// Get total size
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// Shared state for save/load and core data structures
#[derive(Clone)]
struct SharedState {
    /// Concurrent hashmap for (board, piece) -> orientation lookup
    lookup: Arc<DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>>,
    /// Priority frontier: height-indexed sorted set for (board, bag) states
    frontier: Arc<PriorityFrontier>,
    /// Counter for frontier enqueued (used by PriorityFrontier)
    frontier_enqueued: Arc<AtomicU64>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Shutdown request (Ctrl+C)
    shutdown_requested: Arc<AtomicBool>,
}

impl SharedState {
    fn new() -> Self {
        let frontier_enqueued = Arc::new(AtomicU64::new(0));
        Self {
            lookup: Arc::new(DashMap::new()),
            frontier: Arc::new(PriorityFrontier::new(frontier_enqueued.clone())),
            frontier_enqueued,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    fn seed_starting_state(&self) {
        let empty_board = TetrisBoard::new();
        let bag = TetrisPieceBagState::new();
        self.frontier.push(empty_board, bag);
    }

    fn lookup_size(&self) -> usize {
        self.lookup.len()
    }

    /// Save both lookup table and frontier to disk
    fn save_to_disk(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        println!("Saving state to {}...", path);

        // Collect lookup table entries and convert to bytes
        let lookup_entries: Vec<(Vec<u8>, u8, u8)> = self
            .lookup
            .iter()
            .map(|entry| {
                let ((board, piece), orientation) = (*entry.key(), *entry.value());
                let board_bytes: [u8; 40] = board.into();
                let piece_byte: u8 = piece.into();
                let orientation_byte: u8 = orientation.index();
                (board_bytes.to_vec(), piece_byte, orientation_byte)
            })
            .collect();

        // Collect frontier entries from SkipMap
        let frontier_entries: Vec<(Vec<u8>, u8)> = self
            .frontier
            .map
            .iter()
            .map(|entry_ref| {
                let value: &FrontierValue = entry_ref.value();
                let board_bytes: [u8; 40] = value.board.into();
                let bag_byte: u8 = value.bag.into();
                (board_bytes.to_vec(), bag_byte)
            })
            .collect();

        // Package both together
        let snapshot = (lookup_entries, frontier_entries);

        // Serialize to disk using bincode
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &snapshot)?;

        println!(
            "Saved {} lookup entries and {} frontier items in {:.2}s",
            snapshot.0.len(),
            snapshot.1.len(),
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Load both lookup table and frontier from disk (if exists)
    fn load_from_disk(&self, path: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        if !Path::new(path).exists() {
            return Ok((0, 0));
        }

        let start = Instant::now();
        println!("Loading state from {}...", path);

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let (lookup_entries, frontier_entries): (Vec<(Vec<u8>, u8, u8)>, Vec<(Vec<u8>, u8)>) =
            bincode::deserialize_from(reader)?;

        // Insert all lookup entries into the DashMap (converting from bytes)
        let lookup_count = lookup_entries.len();
        for (board_vec, piece_byte, orientation_byte) in lookup_entries {
            let board_bytes: [u8; 40] = board_vec.try_into().map_err(|_| "Invalid board bytes")?;
            let board = TetrisBoard::from(board_bytes);
            let piece = TetrisPiece::from(piece_byte);
            let orientation = TetrisPieceOrientation::from_index(orientation_byte);
            self.lookup.insert((board, piece), orientation);
        }

        // Insert all frontier entries into the priority frontier (converting from bytes)
        let frontier_count = frontier_entries.len();
        for (board_vec, bag_byte) in frontier_entries {
            let board_bytes: [u8; 40] = board_vec.try_into().map_err(|_| "Invalid board bytes")?;
            let board = TetrisBoard::from(board_bytes);
            let bag = TetrisPieceBagState::from(bag_byte);
            self.frontier.push(board, bag);
        }

        println!(
            "Loaded {} lookup entries and {} frontier items in {:.2}s",
            lookup_count,
            frontier_count,
            start.elapsed().as_secs_f64()
        );
        Ok((lookup_count, frontier_count))
    }
}

/// Dispatcher thread: pops from priority frontier in strict order, sends to channel
fn dispatcher_thread(
    frontier: Arc<PriorityFrontier>,
    sender: Sender<(TetrisBoard, TetrisPieceBagState)>,
    shutdown_requested: Arc<AtomicBool>,
    frontier_consumed: Arc<AtomicU64>,
) {
    println!("Dispatcher: started");

    while !shutdown_requested.load(Ordering::Relaxed) {
        match frontier.pop() {
            Some((board, bag)) => {
                frontier_consumed.fetch_add(1, Ordering::Relaxed);
                // Send to workers - blocks if channel is full (backpressure)
                if sender.send((board, bag)).is_err() {
                    break; // Channel closed
                }
            }
            None => {
                // Frontier empty - brief sleep to avoid spinning
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    // Close channel to signal workers to exit
    drop(sender);
    println!("Dispatcher: exiting");
}

/// Process a single (board, bag) state - returns new states to add to frontier
fn process_board_state(
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    lookup: &DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>,
    beam_search: &mut MultiBeamSearch<
        BeamTetrisState,
        N,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    >,
    stats: &SharedStats,
) -> Vec<(TetrisBoard, TetrisPieceBagState)> {
    let mut new_states = Vec::new();
    let mut game = tetris_game::TetrisGame::new();

    stats.boards_expanded.fetch_add(1, Ordering::Relaxed);

    for (piece, bag_state) in bag.iter_next_states() {
        // Check if already computed
        if lookup.contains_key(&(board, piece)) {
            stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        stats.lookup_misses.fetch_add(1, Ordering::Relaxed);

        // Setup game state
        game.board = board;
        game.set_bag_piece_seeded(bag_state, piece, BASE_SEED);

        // Run beam search
        let game_state = BeamTetrisState::new(game);
        let best_placement = beam_search.search_with_seeds(game_state, BASE_SEED, MAX_DEPTH);
        let Some(placement) = best_placement else {
            stats.games_lost.fetch_add(1, Ordering::Relaxed);
            continue;
        };

        // Apply placement
        if game.apply_placement(placement).is_lost == IsLost::LOST {
            stats.games_lost.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // Try to insert - only add to frontier if we won the race
        match lookup.entry((board, piece)) {
            Entry::Vacant(vacant) => {
                vacant.insert(placement.orientation);
                stats.lookup_inserts.fetch_add(1, Ordering::Relaxed);
                new_states.push((game.board, bag_state));
            }
            Entry::Occupied(_) => {
                // Another worker beat us - skip
            }
        }
    }

    new_states
}

/// Shared statistics for workers
struct SharedStats {
    lookup_hits: AtomicU64,
    lookup_misses: AtomicU64,
    lookup_inserts: AtomicU64,
    games_lost: AtomicU64,
    boards_expanded: AtomicU64,
}

impl SharedStats {
    fn new() -> Self {
        Self {
            lookup_hits: AtomicU64::new(0),
            lookup_misses: AtomicU64::new(0),
            lookup_inserts: AtomicU64::new(0),
            games_lost: AtomicU64::new(0),
            boards_expanded: AtomicU64::new(0),
        }
    }
}

/// Logger context for statistics printing
struct LoggerContext {
    lookup: Arc<DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>>,
    frontier: Arc<PriorityFrontier>,
    stats: Arc<SharedStats>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    shutdown_requested: Arc<AtomicBool>,
    start: Instant,
}

/// CSV writer for statistics logging
struct CsvLogger {
    file: BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        let file_exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        // Write header if file is new
        if !file_exists {
            writeln!(
                writer,
                "timestamp_secs,boards_expanded,lookup_size,frontier_size,\
                boards_per_sec,lookup_inserts,games_lost,lookup_hits,lookup_misses,\
                cache_hit_rate,frontier_enqueued,frontier_consumed,\
                frontier_in_rate,frontier_out_rate,frontier_ratio"
            )?;
        }

        Ok(Self { file: writer })
    }

    fn log(&mut self, ctx: &LoggerContext) -> std::io::Result<()> {
        let secs = ctx.start.elapsed().as_secs_f64();

        let lookup_hits = ctx.stats.lookup_hits.load(Ordering::Relaxed);
        let lookup_misses = ctx.stats.lookup_misses.load(Ordering::Relaxed);
        let lookup_inserts = ctx.stats.lookup_inserts.load(Ordering::Relaxed);
        let frontier_enqueued = ctx.frontier_enqueued.load(Ordering::Relaxed);
        let frontier_consumed = ctx.frontier_consumed.load(Ordering::Relaxed);
        let games_lost = ctx.stats.games_lost.load(Ordering::Relaxed);
        let boards_expanded = ctx.stats.boards_expanded.load(Ordering::Relaxed);

        let frontier_size = ctx.frontier.len();
        let lookup_size = ctx.lookup.len();

        let boards_per_sec = boards_expanded as f64 / secs.max(1e-9);

        let total_lookups = lookup_hits + lookup_misses;
        let cache_hit_rate = if total_lookups > 0 {
            (lookup_hits as f64 / total_lookups as f64) * 100.0
        } else {
            0.0
        };

        let frontier_in_rate = frontier_enqueued as f64 / secs.max(1e-9);
        let frontier_out_rate = frontier_consumed as f64 / secs.max(1e-9);
        let frontier_ratio = frontier_enqueued as f64 / frontier_consumed.max(1) as f64;

        writeln!(
            self.file,
            "{:.3},{},{},{},{:.2},{},{},{},{},{:.2},{},{},{:.2},{:.2},{:.4}",
            secs,
            boards_expanded,
            lookup_size,
            frontier_size,
            boards_per_sec,
            lookup_inserts,
            games_lost,
            lookup_hits,
            lookup_misses,
            cache_hit_rate,
            frontier_enqueued,
            frontier_consumed,
            frontier_in_rate,
            frontier_out_rate,
            frontier_ratio
        )?;

        self.file.flush()?;
        Ok(())
    }
}

/// Logger thread: periodically prints statistics, logs to CSV, and saves to disk
fn logger_thread(ctx: LoggerContext, state_for_save: SharedState) {
    let mut last_logged = Instant::now();
    let mut last_saved = Instant::now();

    // Initialize CSV logger
    let mut csv_logger = match CsvLogger::new(CSV_FILE) {
        Ok(logger) => {
            println!("CSV logging to: {CSV_FILE}");
            Some(logger)
        }
        Err(e) => {
            eprintln!("Failed to create CSV logger: {e}");
            None
        }
    };

    while !ctx.shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(1));

        if last_logged.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            print_stats(&ctx);

            // Log to CSV
            if let Some(ref mut logger) = csv_logger {
                if let Err(e) = logger.log(&ctx) {
                    eprintln!("Failed to write CSV: {e}");
                }
            }

            last_logged = Instant::now();
        }

        // Periodic save to disk
        if !ctx.shutdown_requested.load(Ordering::Relaxed)
            && last_saved.elapsed() >= Duration::from_secs(SAVE_EVERY_SECS)
        {
            if let Err(e) = state_for_save.save_to_disk(SAVE_FILE) {
                eprintln!("Failed to save to disk: {e}");
            }
            last_saved = Instant::now();
        }
    }

    println!("Logger: final stats...");
    print_stats(&ctx);

    // Final CSV log
    if let Some(ref mut logger) = csv_logger {
        if let Err(e) = logger.log(&ctx) {
            eprintln!("Failed to write final CSV: {e}");
        }
    }

    println!("Logger: exiting");
}

/// Print current statistics
fn print_stats(ctx: &LoggerContext) {
    let secs = ctx.start.elapsed().as_secs_f64().max(1e-9);

    let lookup_hits = ctx.stats.lookup_hits.load(Ordering::Relaxed);
    let lookup_misses = ctx.stats.lookup_misses.load(Ordering::Relaxed);
    let lookup_inserts = ctx.stats.lookup_inserts.load(Ordering::Relaxed);
    let frontier_enqueued = ctx.frontier_enqueued.load(Ordering::Relaxed);
    let frontier_consumed = ctx.frontier_consumed.load(Ordering::Relaxed);
    let games_lost = ctx.stats.games_lost.load(Ordering::Relaxed);
    let boards_expanded = ctx.stats.boards_expanded.load(Ordering::Relaxed);

    let frontier_size = ctx.frontier.len();
    let lookup_size = ctx.lookup.len();

    let boards_per_sec = boards_expanded as f64 / secs;

    // Calculate rates and ratios
    let total_lookups = lookup_hits + lookup_misses;
    let cache_hit_rate = if total_lookups > 0 {
        (lookup_hits as f64 / total_lookups as f64) * 100.0
    } else {
        0.0
    };

    let frontier_consumption_rate = frontier_consumed as f64 / secs;
    let frontier_expansion_rate = frontier_enqueued as f64 / secs;
    let frontier_expansion_ratio = frontier_enqueued as f64 / frontier_consumed.max(1) as f64;

    println!(
        "t={secs:.1}s boards={boards_expanded} lookup={lookup_size} frontier={frontier_size} | \
        rate={boards_per_sec:.1}/s inserts={lookup_inserts} lost={games_lost} | \
        cache={cache_hit_rate:.1}% | \
        f_in={frontier_expansion_rate:.1}/s f_out={frontier_consumption_rate:.1}/s ratio={frontier_expansion_ratio:.2}"
    );
}

/// Entry point
fn main() {
    let cli = Cli::parse();

    match cli.mode {
        Mode::Create => run_tetris_atlas_create(),
        Mode::Explore => run_tetris_atlas_explore(),
    }
}

/// Run the in-memory atlas builder with channel-based priority dispatch
pub fn run_tetris_atlas_create() {
    let num_workers = num_cpus::get().saturating_sub(2).max(1);

    println!("Starting atlas builder with channel-based priority dispatch:");
    println!("  Workers: {num_workers}");
    println!("  Channel capacity: {CHANNEL_CAPACITY}");
    println!("  Beam: WIDTH={BEAM_WIDTH}, DEPTH={MAX_DEPTH}");
    println!("Press Ctrl+C to stop gracefully...");

    // Create shared state (for save/load compatibility)
    let state = SharedState::new();

    // Try to load from disk if file exists
    match state.load_from_disk(SAVE_FILE) {
        Ok((lookup_count, frontier_count)) => {
            if lookup_count > 0 || frontier_count > 0 {
                println!(
                    "Resumed from checkpoint: {} lookups, {} frontier items",
                    lookup_count, frontier_count
                );
            } else {
                println!("No checkpoint found, starting fresh");
                state.seed_starting_state();
            }
        }
        Err(e) => {
            eprintln!("Failed to load from disk: {}, starting fresh", e);
            state.seed_starting_state();
        }
    }

    // Extract components for the new architecture
    let lookup = state.lookup.clone();
    let frontier = state.frontier.clone();
    let frontier_enqueued = state.frontier_enqueued.clone();
    let frontier_consumed = Arc::new(AtomicU64::new(0));
    let stats = Arc::new(SharedStats::new());
    let shutdown = state.shutdown.clone();
    let shutdown_requested = state.shutdown_requested.clone();
    let start = Instant::now();

    // Create bounded channel
    let (tx, rx) = bounded::<(TetrisBoard, TetrisPieceBagState)>(CHANNEL_CAPACITY);

    // Setup Ctrl+C handler
    let shutdown_requested_clone = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received! Initiating graceful shutdown...");
        shutdown_requested_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    // Spawn dispatcher thread
    let dispatcher_frontier = frontier.clone();
    let dispatcher_shutdown = shutdown_requested.clone();
    let dispatcher_consumed = frontier_consumed.clone();
    let dispatcher_handle = thread::spawn(move || {
        dispatcher_thread(
            dispatcher_frontier,
            tx,
            dispatcher_shutdown,
            dispatcher_consumed,
        );
    });

    // Spawn logger thread
    let logger_ctx = LoggerContext {
        lookup: lookup.clone(),
        frontier: frontier.clone(),
        stats: stats.clone(),
        frontier_enqueued: frontier_enqueued.clone(),
        frontier_consumed: frontier_consumed.clone(),
        shutdown: shutdown.clone(),
        shutdown_requested: shutdown_requested.clone(),
        start,
    };
    let logger_state = state.clone();
    let logger_handle = thread::spawn(move || {
        logger_thread(logger_ctx, logger_state);
    });

    // Run workers via rayon::scope
    println!("Starting {num_workers} workers...");
    rayon::scope(|s| {
        for worker_id in 0..num_workers {
            let rx = rx.clone();
            let lookup = lookup.clone();
            let frontier = frontier.clone();
            let stats = stats.clone();
            let shutdown = shutdown.clone();

            s.spawn(move |_| {
                println!("Worker {worker_id}: started");

                // Thread-local beam search
                let mut beam_search = MultiBeamSearch::<
                    BeamTetrisState,
                    N,
                    TOP_N_PER_BEAM,
                    BEAM_WIDTH,
                    MAX_DEPTH,
                    MAX_MOVES,
                >::new();

                // Process items from channel
                while let Ok((board, bag)) = rx.recv() {
                    let new_states =
                        process_board_state(board, bag, &lookup, &mut beam_search, &stats);

                    // Push new states to frontier (maintains priority)
                    for (new_board, new_bag) in new_states {
                        frontier.push(new_board, new_bag);
                    }
                }

                println!("Worker {worker_id}: exiting");
            });
        }
    });

    // Workers finished - shutdown dispatcher and logger
    println!("All workers finished. Shutting down...");
    shutdown.store(true, Ordering::Relaxed);

    dispatcher_handle.join().expect("Dispatcher panicked");
    logger_handle.join().expect("Logger panicked");

    // Save final state to disk
    println!("\nSaving final state to disk...");
    if let Err(e) = state.save_to_disk(SAVE_FILE) {
        eprintln!("Failed to save final state: {}", e);
    }

    println!("\nAll threads exited cleanly.");
    println!("Final lookup table size: {}", state.lookup_size());
    println!("Total time: {:.2}s", start.elapsed().as_secs_f64());
}

/// Explore an existing atlas interactively using a TUI
pub fn run_tetris_atlas_explore() {
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

    // Load lookup table from disk
    let state = SharedState::new();
    match state.load_from_disk(SAVE_FILE) {
        Ok((lookup_count, _)) => {
            if lookup_count == 0 {
                eprintln!("No atlas found! Please run 'create' mode first.");
                std::process::exit(1);
            }
            println!("Loaded {} lookup entries", lookup_count);
        }
        Err(e) => {
            eprintln!("Failed to load atlas: {}", e);
            std::process::exit(1);
        }
    }

    // TUI state
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

    let mut timeline = Timeline::new();
    let mut game = tetris_game::TetrisGame::new();

    // Setup terminal
    enable_raw_mode().expect("Failed to enable raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).expect("Failed to setup terminal");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("Failed to create terminal");

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let board = timeline.current_board();
            let bag = timeline.current_bag();

            // Calculate stats
            let mut filled_cells = 0;
            let mut height = 0;

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

                // Left panel: Board visualization
                let board_str = format!("{}", board);
                let board_widget = Paragraph::new(board_str)
                    .block(Block::default().borders(Borders::ALL).title("Board"));
                f.render_widget(board_widget, main_chunks[0]);

                // Right panel: Stats and Bag
                let right_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(8), Constraint::Min(0)])
                    .split(main_chunks[1]);

                // Stats
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

                // Bag and piece selection
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
                        state.lookup.contains_key(&(board, *piece))
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

                // Controls at bottom
                let controls = Paragraph::new(
                    "← → Navigate | 1-7 or O/I/L/J/S/Z/T Select Piece | R Random | Q Quit",
                )
                .block(Block::default().borders(Borders::ALL).title("Controls"));
                f.render_widget(controls, chunks[1]);
            })?;

            // Handle input
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
                                    // Random piece from available pieces in DB
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
                                    .filter(|&p| {
                                        bag.contains(p) && state.lookup.contains_key(&(board, p))
                                    })
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
                                // Check if piece is valid (in bag and in DB)
                                if bag.contains(piece) {
                                    if let Some(&orientation) =
                                        state.lookup.get(&(board, piece)).as_deref()
                                    {
                                        // Apply the move
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
                                            // Auto-refill bag if empty
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

    // Restore terminal
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
