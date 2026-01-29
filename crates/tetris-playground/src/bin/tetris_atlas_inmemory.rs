#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Builder - In-Memory Implementation
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using tiered beam search. Unlike the RocksDB version,
//! this keeps everything in memory for maximum speed.
//!
//! ## Approach
//!
//! 1. **Parallel Workers**: Multiple worker threads claim board states from a height-prioritized frontier queue
//! 2. **Tiered Beam Search**: Uses different beam parameters (Low/Med/High) based on board height
//! 3. **Priority Frontier**: Processes lower-height boards first for optimal exploration
//! 4. **In-Memory Storage**: DashMap for lookup table, height-indexed queues for frontier - no serialization overhead
//! 5. **Lock-Free Concurrency**: All data structures are lock-free for maximum throughput
//!
//! ## Performance Benefits
//!
//! - No disk I/O overhead
//! - No serialization/deserialization
//! - Lock-free concurrent data structures
//! - Better cache locality
//! - Height-based prioritization for efficient space exploration
//!
//! ## Limitations
//!
//! - Limited by RAM (no persistence)
//! - State lost on crash (no recovery)

use clap::{Parser, Subcommand};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use std::fs::File;
use std::io::{BufReader, BufWriter};
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
use tetris_search::set_global_threadpool;

// --- Tiered Beam Search Parameters ---
const LOW_N: usize = 16;
const LOW_TOP_N_PER_BEAM: usize = 16;
const LOW_BEAM_WIDTH: usize = 16;
const LOW_MAX_DEPTH: usize = 4;

const MED_N: usize = 32;
const MED_TOP_N_PER_BEAM: usize = 8;
const MED_BEAM_WIDTH: usize = 32;
const MED_MAX_DEPTH: usize = 7;

const HIGH_N: usize = 64;
const HIGH_TOP_N_PER_BEAM: usize = 32;
const HIGH_BEAM_WIDTH: usize = 64;
const HIGH_MAX_DEPTH: usize = 7;

const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
const NUM_WORKERS: usize = 8;
const LOG_EVERY_SECS: u64 = 3;
const SAVE_EVERY_SECS: u64 = 600;
const SAVE_FILE: &str = "tetris_atlas_inmemory.bin";
const BASE_SEED: u64 = 42;
// ---------------

#[derive(Clone, Copy, Debug)]
enum BeamTier {
    Low,
    Med,
    High,
}

#[inline(always)]
fn select_tier(height: u32) -> BeamTier {
    match height {
        0..=2 => BeamTier::Low,
        3..=4 => BeamTier::Med,
        _ => BeamTier::High,
    }
}

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
    height: u32,
    hash: u64,
}

impl FrontierKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        let height = board.height();
        let hash = hash_board_bag(board, bag);
        Self { height, hash }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First compare by height (lower is better)
        match self.height.cmp(&other.height) {
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

/// Shared state between all threads
#[derive(Clone)]
struct SharedState {
    /// Concurrent hashmap for (board, piece) -> orientation lookup
    lookup: Arc<DashMap<(TetrisBoard, TetrisPiece), TetrisPieceOrientation>>,
    /// Priority frontier: height-indexed sorted set for (board, bag) states
    frontier: Arc<PriorityFrontier>,

    // Performance counters
    lookup_hits: Arc<AtomicU64>,
    lookup_misses: Arc<AtomicU64>,
    lookup_inserts: Arc<AtomicU64>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    games_lost: Arc<AtomicU64>,
    boards_expanded: Arc<AtomicU64>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Start time
    start: Instant,
}

impl SharedState {
    fn new() -> Self {
        let frontier_enqueued = Arc::new(AtomicU64::new(0));
        Self {
            lookup: Arc::new(DashMap::new()),
            frontier: Arc::new(PriorityFrontier::new(frontier_enqueued.clone())),
            lookup_hits: Arc::new(AtomicU64::new(0)),
            lookup_misses: Arc::new(AtomicU64::new(0)),
            lookup_inserts: Arc::new(AtomicU64::new(0)),
            frontier_enqueued,
            frontier_consumed: Arc::new(AtomicU64::new(0)),
            games_lost: Arc::new(AtomicU64::new(0)),
            boards_expanded: Arc::new(AtomicU64::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            start: Instant::now(),
        }
    }

    fn seed_starting_state(&self) {
        let empty_board = TetrisBoard::new();
        let bag = TetrisPieceBagState::new();
        self.frontier.push(empty_board, bag);
    }

    fn frontier_size(&self) -> usize {
        self.frontier.len()
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

/// Worker thread: processes frontier items using tiered beam search
fn worker_thread(state: SharedState, worker_id: usize) {
    // Create three beam searchers for different height tiers
    let mut beam_search_low = MultiBeamSearch::<
        BeamTetrisState,
        LOW_N,
        LOW_TOP_N_PER_BEAM,
        LOW_BEAM_WIDTH,
        LOW_MAX_DEPTH,
        MAX_MOVES,
    >::new();
    let mut beam_search_med = MultiBeamSearch::<
        BeamTetrisState,
        MED_N,
        MED_TOP_N_PER_BEAM,
        MED_BEAM_WIDTH,
        MED_MAX_DEPTH,
        MAX_MOVES,
    >::new();
    let mut beam_search_high = MultiBeamSearch::<
        BeamTetrisState,
        HIGH_N,
        HIGH_TOP_N_PER_BEAM,
        HIGH_BEAM_WIDTH,
        HIGH_MAX_DEPTH,
        MAX_MOVES,
    >::new();

    let mut game = tetris_game::TetrisGame::new();
    let mut last_work_time = Instant::now();
    const IDLE_TIMEOUT: Duration = Duration::from_secs(15);

    println!("Worker {worker_id}: started");

    while !state.shutdown.load(Ordering::Relaxed) {
        // Try to pop one item from the frontier
        let Some((board, bag)) = state.frontier.pop() else {
            // No work available - check if we've been idle too long
            if last_work_time.elapsed() > IDLE_TIMEOUT {
                println!("Worker {worker_id}: no work for {IDLE_TIMEOUT:?}, exiting");
                break;
            }
            // Backoff to avoid CPU spinning
            thread::sleep(Duration::from_millis(100));
            continue;
        };

        // Found work - reset idle timer
        last_work_time = Instant::now();
        state.frontier_consumed.fetch_add(1, Ordering::Relaxed);
        state.boards_expanded.fetch_add(1, Ordering::Relaxed);

        // Process all pieces for this board state
        for (piece, bag_state) in bag.iter_next_states() {
            // Check if we've already computed this (board, piece) combination
            if state.lookup.contains_key(&(board, piece)) {
                state.lookup_hits.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Cache miss - run tiered beam search
            state.lookup_misses.fetch_add(1, Ordering::Relaxed);

            game.board = board;
            game.set_bag_piece_seeded(bag_state, piece, BASE_SEED);

            let tier = select_tier(board.height());
            let game_state = BeamTetrisState::new(game);
            let search_result = match tier {
                BeamTier::Low => {
                    beam_search_low.search_with_seeds(game_state, BASE_SEED, LOW_MAX_DEPTH)
                }
                BeamTier::Med => {
                    beam_search_med.search_with_seeds(game_state, BASE_SEED, MED_MAX_DEPTH)
                }
                BeamTier::High => {
                    beam_search_high.search_with_seeds(game_state, BASE_SEED, HIGH_MAX_DEPTH)
                }
            };

            let best_placement = match search_result {
                Some(result) => result,
                None => {
                    // Beam search failed (no valid placements at all)
                    state.games_lost.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            // Apply placement and check if game is lost
            if game.apply_placement(best_placement).is_lost == IsLost::LOST {
                state.games_lost.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Try to insert the new lookup entry - only add to frontier if we actually inserted
            match state.lookup.entry((board, piece)) {
                Entry::Vacant(vacant) => {
                    // We won the race - insert our result
                    vacant.insert(best_placement.orientation);
                    state.lookup_inserts.fetch_add(1, Ordering::Relaxed);

                    // Add resulting board to frontier (only for new entries!)
                    state.frontier.push(game.board, bag_state);
                }
                Entry::Occupied(_) => {
                    // Another worker already inserted this entry - don't add to frontier
                    // (the other worker already added it)
                }
            }
        }
    }

    println!("Worker {worker_id}: exiting cleanly");
}

/// Logger thread: periodically prints statistics and saves to disk
fn logger_thread(state: SharedState) {
    let mut last_logged = Instant::now();
    let mut last_saved = Instant::now();

    loop {
        thread::sleep(Duration::from_secs(1));

        // Check shutdown signal
        if state.shutdown.load(Ordering::Relaxed) {
            println!("Logger thread: shutdown signal received, printing final stats...");
            print_stats(&state);
            break;
        }

        if last_logged.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            print_stats(&state);
            last_logged = Instant::now();
        }

        // Periodic save to disk
        if last_saved.elapsed() >= Duration::from_secs(SAVE_EVERY_SECS) {
            if let Err(e) = state.save_to_disk(SAVE_FILE) {
                eprintln!("Failed to save to disk: {}", e);
            }
            last_saved = Instant::now();
        }
    }

    println!("Logger thread: exiting cleanly");
}

/// Print current statistics
fn print_stats(state: &SharedState) {
    let secs = state.start.elapsed().as_secs_f64().max(1e-9);

    let lookup_hits = state.lookup_hits.load(Ordering::Relaxed);
    let lookup_misses = state.lookup_misses.load(Ordering::Relaxed);
    let lookup_inserts = state.lookup_inserts.load(Ordering::Relaxed);
    let frontier_enqueued = state.frontier_enqueued.load(Ordering::Relaxed);
    let frontier_consumed = state.frontier_consumed.load(Ordering::Relaxed);
    let games_lost = state.games_lost.load(Ordering::Relaxed);
    let boards_expanded = state.boards_expanded.load(Ordering::Relaxed);

    let frontier_size = state.frontier_size();
    let lookup_size = state.lookup_size();

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
        boards_rate={boards_per_sec:.1}/s inserts={lookup_inserts} lost={games_lost} | \
        cache_hit={cache_hit_rate:.1}% ({lookup_hits}/{total_lookups}) | \
        f_consume={frontier_consumption_rate:.1}/s f_expand={frontier_expansion_rate:.1}/s f_ratio={frontier_expansion_ratio:.2}"
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

/// Run the in-memory atlas builder with parallel workers and multi-beam search
pub fn run_tetris_atlas_create() {
    set_global_threadpool();

    println!("Starting in-memory atlas builder with tiered beam search:");
    println!("  NUM_WORKERS: {NUM_WORKERS}");
    println!("  Low tier (height 0-1): N={LOW_N}, WIDTH={LOW_BEAM_WIDTH}, DEPTH={LOW_MAX_DEPTH}");
    println!("  Med tier (height 2): N={MED_N}, WIDTH={MED_BEAM_WIDTH}, DEPTH={MED_MAX_DEPTH}");
    println!(
        "  High tier (height 3+): N={HIGH_N}, WIDTH={HIGH_BEAM_WIDTH}, DEPTH={HIGH_MAX_DEPTH}"
    );
    println!("Press Ctrl+C to stop gracefully...");

    // Create shared state
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
                // No checkpoint found, seed with starting state
                println!("No checkpoint found, starting fresh");
                state.seed_starting_state();
            }
        }
        Err(e) => {
            eprintln!("Failed to load from disk: {}, starting fresh", e);
            state.seed_starting_state();
        }
    }

    // Setup Ctrl+C handler
    let shutdown_state = state.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received! Initiating graceful shutdown...");
        shutdown_state.shutdown.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    // Spawn worker threads
    let mut worker_handles = Vec::new();
    for worker_id in 0..NUM_WORKERS {
        let worker_state = state.clone();
        let handle = thread::spawn(move || {
            worker_thread(worker_state, worker_id);
        });
        worker_handles.push(handle);
    }

    // Spawn logger thread
    let logger_state = state.clone();
    let logger_handle = thread::spawn(move || {
        logger_thread(logger_state);
    });

    // Wait for workers to finish (either via Ctrl+C or natural completion after 10s idle)
    println!("Main thread: waiting for worker threads to finish...");
    for (i, handle) in worker_handles.into_iter().enumerate() {
        handle.join().expect("Worker thread panicked");
        println!("Main thread: worker {i} finished");
    }

    // All workers finished - trigger shutdown for logger
    if !state.shutdown.load(Ordering::Relaxed) {
        println!("All workers finished naturally. Shutting down logger...");
        state.shutdown.store(true, Ordering::Relaxed);
    }

    println!("Waiting for logger thread to finish...");
    logger_handle.join().expect("Logger thread panicked");

    // Save final state to disk
    println!("\nSaving final state to disk...");
    if let Err(e) = state.save_to_disk(SAVE_FILE) {
        eprintln!("Failed to save final state: {}", e);
    }

    println!("\nAll threads exited cleanly.");
    println!("Final lookup table size: {}", state.lookup_size());
    println!("Total time: {:.2}s", state.start.elapsed().as_secs_f64());
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
