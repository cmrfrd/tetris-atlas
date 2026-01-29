#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(impl_trait_in_bindings)]
#![feature(const_index)]
//! # Tetris Atlas Builder - FeoxDB Implementation
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using beam search. The goal is to map board states to optimal
//! piece placements such that the entire state space becomes self-referential.
//!
//! ## Approach
//!
//! 1. **Parallel Workers**: Multiple worker threads claim batches of board states from a frontier queue
//! 2. **Beam Search**: For each board state, uses multi-beam search to find optimal piece placements
//! 3. **FeoxDB Storage**: Stores the lookup table and frontier in a persistent embedded database
//! 4. **Crash Recovery**: Supports resuming from crashes by tracking claimed-but-incomplete work
//!
//! ## Output
//!
//! - **FeoxDB Database**: `tetris_atlas_db/` - Persistent lookup table with board → placements
//! - **Statistics CSV**: `tetris_atlas_stats.csv` - Real-time metrics (expansion rate, cache hits, etc.)
//!
//! ## Performance Optimizations
//!
//! - Partitioned workers scanning disjoint key ranges (no coordination overhead)
//! - Thread-local beam search reuse (no allocation per board)
//! - Write buffer with periodic flushes
//! - Atomic counters for performance tracking

use clap::{Parser, Subcommand};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    repeat_idx_unroll,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch};
use tetris_search::set_global_threadpool;

const BEAM_WIDTH: usize = 32;
const MAX_DEPTH: usize = 8;
const NUM_SEARCHES: usize = 8;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

// Thread-local beam search instance for rayon worker threads
thread_local! {
    static THREAD_BEAM_SEARCH: std::cell::RefCell<Option<MultiBeamSearch<BeamTetrisState, NUM_SEARCHES, 1, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>>> = std::cell::RefCell::new(None);
}

const STATS_CSV_PATH: &str = "tetris_atlas_stats.csv";
const LOG_EVERY_SECS: u64 = 3;
const IDLE_GRACE_SECS: u64 = 15;
const WORKER_RECV_TIMEOUT_MS: u64 = 100;

const NUM_WORKERS: usize = 16;
const WORKER_FRONTIER_BATCH_SIZE: usize = 1024;

const LOOKUP_TABLE_BYTE: u8 = b'L';
const FRONTIER_QUEUE_BYTE: u8 = b'F';

#[derive(Parser)]
#[command(name = "tetris-atlas")]
#[command(about = "Tetris Atlas - Build and explore optimal Tetris lookup tables")]
struct Cli {
    /// Path to the database directory
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
    const fn key_bytes(&self) -> [u8; 42] {
        let board_bytes: [u8; 40] = self.board.into();
        let piece_byte: u8 = self.piece.into();
        let mut key = [0u8; 42];
        key[0] = LOOKUP_TABLE_BYTE;
        key[1..41].copy_from_slice(&board_bytes);
        key[41] = piece_byte;
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
        let hash: [u8; 8] = hash_board_bag(self.board, self.bag).to_be_bytes();
        let mut out = [0u8; 9];
        out[0] = FRONTIER_QUEUE_BYTE;
        out[1..9].copy_from_slice(&hash);
        out
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

// ----------------------------- DB Wrapper ----------------------------------

pub struct TetrisAtlasDB {
    store: Arc<feoxdb::FeoxStore>,
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
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        const MB: u64 = feoxdb::constants::MB as u64;
        const GB: u64 = feoxdb::constants::GB as u64;

        // let store = feoxdb::FeoxStore::builder()
        //     .device_path(path)
        //     .file_size(1 * GB)
        //     .enable_caching(false)
        //     .hash_bits(16)
        //     .enable_ttl(false)
        //     .build()?;
        let store = feoxdb::FeoxStore::new(Some(path.to_string()))?;

        Ok(Self {
            store: Arc::new(store),
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
        self.store.flush_all();
    }

    /// Check if a board+piece combination exists in the lookup table
    pub fn lookup_exists(&self, board: TetrisBoard, piece: TetrisPiece) -> bool {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();
        self.store.contains_key(&key)
    }

    /// Get the optimal orientation for a board+piece combination
    pub fn lookup_orientation(
        &self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Option<TetrisPieceOrientation> {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();

        match self.store.get(&key) {
            Ok(value) => {
                if value.len() == 1 {
                    Some(TetrisPieceOrientation::from_index(value[0]))
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    fn seed_starting_state(&self) -> Result<(), Box<dyn std::error::Error>> {
        let empty_board = TetrisBoard::new();
        let bag = TetrisPieceBagState::new();
        let frontier_item = TetrisAtlasFrontierKeyValue::new(empty_board, bag);
        self.store
            .insert(&frontier_item.key_bytes(), &frontier_item.value_bytes())
            .expect("Failed to insert starting state");
        self.frontier_enqueued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn run_worker_loop(&self, num_workers: usize) -> Vec<JoinHandle<()>> {
        let store = Arc::clone(&self.store);
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
                let store = Arc::clone(&store);
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
                        NUM_SEARCHES,
                        1,
                        BEAM_WIDTH,
                        MAX_DEPTH,
                        MAX_MOVES,
                    >::new();

                    let mut game = tetris_game::TetrisGame::new();

                    // Calculate this worker's partition of the key space
                    // Keys are [FRONTIER_QUEUE_BYTE, hash_byte_0, hash_byte_1, ...]
                    // We partition on hash_byte_0 (second byte of the key)
                    let partition_start = ((worker_id as u16 * 256) / num_workers as u16) as u8;

                    // For the last worker, use 0xFF + 1 as the end (scanning to the end of the partition)
                    let start_key = [FRONTIER_QUEUE_BYTE, partition_start];
                    let end_key = if worker_id == num_workers - 1 {
                        // Last worker: scan from partition_start to end of byte range
                        [FRONTIER_QUEUE_BYTE + 1, 0] // Next prefix
                    } else {
                        let partition_end =
                            (((worker_id + 1) as u16 * 256) / num_workers as u16) as u8;
                        [FRONTIER_QUEUE_BYTE, partition_end]
                    };

                    while !stop.load(Ordering::Acquire) {
                        // Scan this worker's partition for frontier items and process in parallel
                        let count = store
                            .range_query_iter(&start_key, &end_key, WORKER_FRONTIER_BATCH_SIZE)
                            .expect("Failed to query frontier partition")
                            .map(|entry| {
                                let key = entry.key();
                                let record = entry.value();
                                let value = if let Some(val) = record.get_value() {
                                    val.to_vec()
                                } else {
                                    store
                                        .load_value_from_disk(record)
                                        .expect("Failed to load value from disk")
                                };

                                let current_frontier_item =
                                    TetrisAtlasFrontierKeyValue::from_value_bytes(&unsafe {
                                        *(value.as_ptr() as *const [u8; 41])
                                    });

                                frontier_consumed.fetch_add(1, Ordering::Relaxed);
                                boards_expanded.fetch_add(1, Ordering::Relaxed);

                                // Process all pieces for this board state
                                for (piece, bag_state) in
                                    current_frontier_item.bag.iter_next_states()
                                {
                                    game.board = current_frontier_item.board;
                                    game.set_bag_piece_seeded(bag_state, piece, BASE_SEED);

                                    let best_placement = beam_search
                                        .search_with_seeds(
                                            BeamTetrisState::new(game),
                                            BASE_SEED,
                                            MAX_DEPTH,
                                        )
                                        .expect("Failed to search");

                                    // Skip if game is lost
                                    if game.apply_placement(best_placement).is_lost == IsLost::LOST
                                    {
                                        games_lost.fetch_add(1, Ordering::Relaxed);
                                        continue;
                                    }

                                    // Try to insert the new lookup entry
                                    let new_lookup_kv = TetrisAtlasLookupKeyValue::new(
                                        current_frontier_item.board,
                                        piece,
                                        best_placement.orientation,
                                    );
                                    let new_lookup_key = new_lookup_kv.key_bytes();
                                    let new_lookup_value = new_lookup_kv.value_bytes();

                                    if !store.contains_key(&new_lookup_key) {
                                        // Cache miss - insert new frontier entry
                                        lookup_misses.fetch_add(1, Ordering::Relaxed);
                                        lookup_inserts.fetch_add(1, Ordering::Relaxed);

                                        let _ = store.insert(&new_lookup_key, &new_lookup_value);

                                        let new_frontier_kv =
                                            TetrisAtlasFrontierKeyValue::new(game.board, bag_state);
                                        let _ = store.insert(
                                            &new_frontier_kv.key_bytes(),
                                            &new_frontier_kv.value_bytes(),
                                        );
                                        frontier_enqueued.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        // Cache hit - entry already exists
                                        lookup_hits.fetch_add(1, Ordering::Relaxed);
                                    }
                                }

                                // Delete the processed frontier item
                                let _ = store.delete(&key);
                                frontier_deleted.fetch_add(1, Ordering::Relaxed);

                                1 // Count processed item
                            })
                            .sum::<usize>();

                        if count == 0 {
                            // No work in this partition, sleep briefly
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                })
            })
            .collect()
    }

    /// Check if the system is idle (no items in frontier queue)
    pub fn is_idle(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let start_key = [FRONTIER_QUEUE_BYTE, 0];
        let end_key = [FRONTIER_QUEUE_BYTE + 1, 0];
        let mut iter = self
            .store
            .range_query_iter(&start_key, &end_key, 1)
            .expect("Failed to query frontier");
        Ok(iter.next().is_none())
    }

    /// Get all performance statistics including FeoxDB stats
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
            u32,
            usize, // Store metrics
            u64,
            u64,
            u64,
            u64,
            u64,
            u64, // Operations
            u64,
            u64,
            u64, // Latencies
            u64,
            u64,
            f64,
            u64,
            usize, // Cache
            u64,
            u64,
            u64,
            u64, // Write buffer
            u64,
            u64,
            u64,
            u64, // Disk I/O
            u64,
            u64,
            u64, // Errors
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

        // Get feoxdb stats
        let db_stats = self.store.stats();

        // Print ALL feoxdb stats to stderr for visibility
        println!("{}", db_stats.format());

        Ok((
            // App counters
            lookup_hits,
            lookup_misses,
            lookup_inserts,
            frontier_enqueued,
            frontier_consumed,
            frontier_deleted,
            games_lost,
            boards_expanded,
            // Store metrics
            db_stats.record_count,
            db_stats.memory_usage,
            // Operations
            db_stats.total_operations,
            db_stats.total_gets,
            db_stats.total_inserts,
            db_stats.total_updates,
            db_stats.total_deletes,
            db_stats.total_range_queries,
            // Latencies
            db_stats.avg_get_latency_ns,
            db_stats.avg_insert_latency_ns,
            db_stats.avg_delete_latency_ns,
            // Cache
            db_stats.cache_hits,
            db_stats.cache_misses,
            db_stats.cache_hit_rate,
            db_stats.cache_evictions,
            db_stats.cache_memory,
            // Write buffer
            db_stats.writes_buffered,
            db_stats.writes_flushed,
            db_stats.write_failures,
            db_stats.flush_count,
            // Disk I/O
            db_stats.disk_reads,
            db_stats.disk_writes,
            db_stats.disk_bytes_read,
            db_stats.disk_bytes_written,
            // Errors
            db_stats.key_not_found_errors,
            db_stats.out_of_memory_errors,
            db_stats.io_errors,
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
    fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
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
        // Truncate timeline at current position (discard future if we branched)
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

pub fn run_tetris_atlas_explore(db_path: &str) {
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

    let atlas = TetrisAtlasDB::new(db_path).expect("Failed to open atlas database");
    let mut timeline = Timeline::new(TetrisBoard::new(), TetrisPieceBagState::new());
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
                    stats.filled_cells,
                    stats.height,
                    stats.holes,
                    stats.lines_cleared,
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
                                // Check if piece is valid (in bag and in DB)
                                if bag.contains(piece) && atlas.lookup_exists(board, piece) {
                                    if let Some(orientation) =
                                        atlas.lookup_orientation(board, piece)
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
                                            // Auto-refill bag if empty (matches create mode behavior)
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

fn main() {
    let cli = Cli::parse();

    match cli.mode {
        Mode::Create => run_tetris_atlas_create(&cli.db_path),
        Mode::Explore => run_tetris_atlas_explore(&cli.db_path),
    }
}

pub fn run_tetris_atlas_create(db_path: &str) {
    set_global_threadpool();

    let start = Instant::now();

    let atlas = TetrisAtlasDB::new(db_path).expect("Failed to create atlas");
    atlas
        .seed_starting_state()
        .expect("Failed to seed starting state");

    let handles: Vec<JoinHandle<()>> = atlas.run_worker_loop(NUM_WORKERS);

    {
        let shutdown_requested = Arc::clone(&atlas.stop);
        ctrlc::set_handler(move || {
            // Only print message once
            if !shutdown_requested.swap(true, Ordering::Release) {
                eprintln!("\nSIGINT received, shutting down gracefully...");
            }
        })
        .expect("Error setting Ctrl-C handler");
    }

    // Initialize CSV file with headers
    let csv_file = std::fs::File::create(STATS_CSV_PATH).expect("Failed to create CSV file");
    let mut csv_writer = std::io::BufWriter::new(csv_file);
    writeln!(
        csv_writer,
        "timestamp_secs,boards_expanded,lookup_hits,lookup_misses,lookup_inserts,\
        frontier_enqueued,frontier_consumed,frontier_deleted,games_lost,lookup_size,frontier_size,\
        boards_per_sec,cache_hit_rate,total_lookups,\
        frontier_consumption_rate,frontier_expansion_rate,frontier_expansion_ratio,\
        db_record_count,db_memory_usage,\
        db_total_operations,db_total_gets,db_total_inserts,db_total_updates,db_total_deletes,db_total_range_queries,\
        db_avg_get_latency_ns,db_avg_insert_latency_ns,db_avg_delete_latency_ns,\
        db_cache_hits,db_cache_misses,db_cache_hit_rate,db_cache_evictions,db_cache_memory,\
        db_writes_buffered,db_writes_flushed,db_write_failures,db_flush_count,\
        db_disk_reads,db_disk_writes,db_disk_bytes_read,db_disk_bytes_written,\
        db_key_not_found_errors,db_out_of_memory_errors,db_io_errors"
    )
    .expect("Failed to write CSV header");
    csv_writer.flush().expect("Failed to flush CSV header");

    let mut last_log = Instant::now();

    loop {
        thread::sleep(Duration::from_millis(50));

        // Check for shutdown first
        if atlas.is_stopped() {
            eprintln!("Shutdown initiated, waiting for workers to finish...");
            break;
        }

        if last_log.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let (
                // App counters
                lookup_hits,
                lookup_misses,
                lookup_inserts,
                frontier_enqueued,
                frontier_consumed,
                frontier_deleted,
                games_lost,
                boards_expanded,
                // Store metrics
                db_record_count,
                db_memory_usage,
                // Operations
                db_total_operations,
                db_total_gets,
                db_total_inserts,
                db_total_updates,
                db_total_deletes,
                db_total_range_queries,
                // Latencies
                db_avg_get_latency_ns,
                db_avg_insert_latency_ns,
                db_avg_delete_latency_ns,
                // Cache
                db_cache_hits,
                db_cache_misses,
                db_cache_hit_rate,
                db_cache_evictions,
                db_cache_memory,
                // Write buffer
                db_writes_buffered,
                db_writes_flushed,
                db_write_failures,
                db_flush_count,
                // Disk I/O
                db_disk_reads,
                db_disk_writes,
                db_disk_bytes_read,
                db_disk_bytes_written,
                // Errors
                db_key_not_found_errors,
                db_out_of_memory_errors,
                db_io_errors,
            ) = atlas.stats().expect("Failed to get stats");

            // Calculate sizes from counters
            let lookup_size = lookup_inserts;
            let frontier_size = frontier_enqueued.saturating_sub(frontier_deleted);

            let boards_per_sec = boards_expanded as f64 / secs;

            // Calculate rates and ratios
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
                f_consume={frontier_consumption_rate:.1}/s f_expand={frontier_expansion_rate:.1}/s f_ratio={frontier_expansion_ratio:.2} f_deleted={frontier_deleted}"
            );

            // Write to CSV with ALL stats
            writeln!(
                csv_writer,
                "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
                frontier_expansion_ratio,
                // Store metrics
                db_record_count,
                db_memory_usage,
                // Operations
                db_total_operations,
                db_total_gets,
                db_total_inserts,
                db_total_updates,
                db_total_deletes,
                db_total_range_queries,
                // Latencies
                db_avg_get_latency_ns,
                db_avg_insert_latency_ns,
                db_avg_delete_latency_ns,
                // Cache
                db_cache_hits,
                db_cache_misses,
                db_cache_hit_rate,
                db_cache_evictions,
                db_cache_memory,
                // Write buffer
                db_writes_buffered,
                db_writes_flushed,
                db_write_failures,
                db_flush_count,
                // Disk I/O
                db_disk_reads,
                db_disk_writes,
                db_disk_bytes_read,
                db_disk_bytes_written,
                // Errors
                db_key_not_found_errors,
                db_out_of_memory_errors,
                db_io_errors
            )
            .expect("Failed to write CSV row");
            csv_writer.flush().expect("Failed to flush CSV");

            // Flush database write buffer to disk periodically
            atlas.flush();

            last_log = Instant::now();
        }
    }

    // Wait for all worker threads to finish
    for h in handles {
        let _ = h.join();
    }

    eprintln!("All workers stopped.");

    // Final flush to ensure all data is persisted to disk
    eprintln!("Flushing database to disk...");
    atlas.flush();
    eprintln!("Database flushed.");

    println!("Finished in {:.2} seconds", start.elapsed().as_secs_f64());
}
