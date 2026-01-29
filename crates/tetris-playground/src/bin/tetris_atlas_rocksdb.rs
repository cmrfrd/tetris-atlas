#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(impl_trait_in_bindings)]
#![feature(const_index)]
//! # Tetris Atlas Builder - RocksDB Implementation
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using beam search. The goal is to map board states to optimal
//! piece placements such that the entire state space becomes self-referential.
//!
//! ## Approach
//!
//! 1. **Parallel Workers**: Multiple worker threads claim batches of board states from a frontier queue
//! 2. **Beam Search**: For each board state, uses multi-beam search to find optimal piece placements
//! 3. **RocksDB Storage**: Stores the lookup table and frontier in a persistent embedded database with column families
//! 4. **Crash Recovery**: Supports resuming from crashes by tracking claimed-but-incomplete work
//!
//! ## Output
//!
//! - **RocksDB Database**: `tetris_atlas_rocksdb/` - Persistent lookup table with board → placements
//! - **Statistics CSV**: `tetris_atlas_stats.csv` - Real-time metrics (expansion rate, cache hits, etc.)
//!
//! ## Performance Optimizations
//!
//! - Optimistic transactions with write batches
//! - Column families for lookup table and frontier queue
//! - Bloom filters and block caching
//! - Partitioned workers scanning disjoint key ranges

use clap::{Parser, Subcommand};
use rocksdb::{
    ColumnFamilyDescriptor, IteratorMode, MultiThreaded, OptimisticTransactionDB, Options,
    ReadOptions, WriteBatchWithTransaction, WriteOptions,
};
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

const LOW_N: usize = 16;
const LOW_TOP_N_PER_BEAM: usize = 16;
const LOW_BEAM_WIDTH: usize = 16;
const LOW_MAX_DEPTH: usize = 4;

const MED_N: usize = 64;
const MED_TOP_N_PER_BEAM: usize = 8;
const MED_BEAM_WIDTH: usize = 64;
const MED_MAX_DEPTH: usize = 7;

const HIGH_N: usize = 256;
const HIGH_TOP_N_PER_BEAM: usize = 64;
const HIGH_BEAM_WIDTH: usize = 256;
const HIGH_MAX_DEPTH: usize = 7;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

const STATS_CSV_PATH: &str = "tetris_atlas_stats_rocksdb.csv";
const LOG_EVERY_SECS: u64 = 3;

const NUM_WORKERS: usize = 16;
const WORKER_FRONTIER_BATCH_SIZE: usize = 64;
const WORKER_STACK_BYTES: usize = 256 * 1024 * 1024;

// Column family names
const CF_LOOKUP: &str = "lookup";
const CF_FRONTIER: &str = "frontier";
const CF_META: &str = "meta";

#[derive(Clone, Copy, Debug)]
enum BeamTier {
    Low,
    Med,
    High,
}

#[inline(always)]
fn select_tier(height: u32) -> BeamTier {
    match height {
        0..=1 => BeamTier::Low,
        2..=2 => BeamTier::Med,
        _ => BeamTier::High,
    }
}

#[derive(Parser)]
#[command(name = "tetris-atlas-rocksdb")]
#[command(about = "Tetris Atlas (RocksDB) - Build and explore optimal Tetris lookup tables")]
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

// ----------------------------- DB Wrapper ----------------------------------

pub struct TetrisAtlasDB {
    db: Arc<OptimisticTransactionDB<MultiThreaded>>,
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
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Performance tuning
        opts.set_max_background_jobs(num_cpus::get() as i32);
        opts.set_write_buffer_size(256 * 1024 * 1024); // 256MB
        opts.set_max_write_buffer_number(6); // Increased from 4 for better write throughput
        opts.set_min_write_buffer_number_to_merge(2);
        opts.set_target_file_size_base(256 * 1024 * 1024);
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_compression_type(rocksdb::DBCompressionType::None);

        // Optimize for bulk writes
        opts.set_allow_concurrent_memtable_write(true);
        opts.set_enable_write_thread_adaptive_yield(true);

        // Optimize compaction
        opts.set_max_bytes_for_level_base(256 * 1024 * 1024);
        opts.set_max_bytes_for_level_multiplier(4.0);

        // Block cache optimization
        let cache = rocksdb::Cache::new_lru_cache(2 * 1024 * 1024 * 1024); // 2GB cache
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_size(16 * 1024);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_cache(&cache);
        opts.set_block_based_table_factory(&block_opts);

        // Lookup CF: Optimized for random reads, mostly inserts
        let mut cf_opts_lookup = Options::default();
        cf_opts_lookup.set_write_buffer_size(512 * 1024 * 1024); // Larger buffer for inserts
        cf_opts_lookup.set_compaction_style(rocksdb::DBCompactionStyle::Level); // Default, good for random reads
        let mut block_opts_lookup = rocksdb::BlockBasedOptions::default();
        block_opts_lookup.set_block_cache(&cache);
        block_opts_lookup.set_bloom_filter(10.0, false);
        cf_opts_lookup.set_block_based_table_factory(&block_opts_lookup);

        // Frontier CF: Optimized for sequential access and high delete rate
        let mut cf_opts_frontier = Options::default();
        cf_opts_frontier.set_write_buffer_size(128 * 1024 * 1024); // Smaller buffer
        cf_opts_frontier.set_level_compaction_dynamic_level_bytes(true); // Better for deletes
        cf_opts_frontier.set_compaction_style(rocksdb::DBCompactionStyle::Universal); // Good for queue pattern

        let cf_opts_meta = Options::default();
        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_LOOKUP, cf_opts_lookup),
            ColumnFamilyDescriptor::new(CF_FRONTIER, cf_opts_frontier),
            ColumnFamilyDescriptor::new(CF_META, cf_opts_meta),
        ];

        let db: OptimisticTransactionDB<MultiThreaded> =
            OptimisticTransactionDB::open_cf_descriptors(&opts, path, cfs)?;

        Ok(Self {
            db: Arc::new(db),
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

    #[inline(always)]
    fn cf_handle(&self, name: &str) -> Arc<rocksdb::BoundColumnFamily<'_>> {
        self.db
            .cf_handle(name)
            .unwrap_or_else(|| panic!("missing {} cf", name))
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    pub fn flush(&self) {
        let _ = self.db.flush();
    }

    /// Check if a board+piece combination exists in the lookup table
    pub fn lookup_exists(&self, board: TetrisBoard, piece: TetrisPiece) -> bool {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();
        let cf_lookup = self.cf_handle(CF_LOOKUP);
        self.db.get_cf(&cf_lookup, &key).ok().flatten().is_some()
    }

    /// Get the optimal orientation for a board+piece combination
    pub fn lookup_orientation(
        &self,
        board: TetrisBoard,
        piece: TetrisPiece,
    ) -> Option<TetrisPieceOrientation> {
        let key = TetrisAtlasLookupKeyValue::new(board, piece, TetrisPieceOrientation::default())
            .key_bytes();
        let cf_lookup = self.cf_handle(CF_LOOKUP);

        match self.db.get_cf(&cf_lookup, &key) {
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

        let cf_frontier = self.cf_handle(CF_FRONTIER);
        self.db.put_cf(
            &cf_frontier,
            &frontier_item.key_bytes(),
            &frontier_item.value_bytes(),
        )?;
        self.frontier_enqueued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn run_worker_loop(&self, num_workers: usize) -> Vec<JoinHandle<()>> {
        let db = Arc::clone(&self.db);
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
                let db = Arc::clone(&db);
                let stop = Arc::clone(&stop);
                let lookup_hits = Arc::clone(&lookup_hits);
                let lookup_misses = Arc::clone(&lookup_misses);
                let lookup_inserts = Arc::clone(&lookup_inserts);
                let frontier_enqueued = Arc::clone(&frontier_enqueued);
                let frontier_consumed = Arc::clone(&frontier_consumed);
                let frontier_deleted = Arc::clone(&frontier_deleted);
                let games_lost = Arc::clone(&games_lost);
                let boards_expanded = Arc::clone(&boards_expanded);

                let mut builder =
                    std::thread::Builder::new().name(format!("atlas-worker-{worker_id}"));
                builder = builder.stack_size(WORKER_STACK_BYTES);

                builder
                    .spawn(move || {
                        const BASE_SEED: u64 = 42;

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
                            vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF] // Beyond any real 9-byte key
                        } else {
                            [0xFF, partition_end, 0, 0, 0, 0, 0, 0, 0].to_vec()
                        };

                        let cf_frontier = db.cf_handle(CF_FRONTIER).expect("missing frontier cf");
                        let cf_lookup = db.cf_handle(CF_LOOKUP).expect("missing lookup cf");

                        // Optimized read options for lookup table queries
                        let mut lookup_read_opts = ReadOptions::default();
                        lookup_read_opts.set_verify_checksums(true); // Verify correctness for lookups
                        lookup_read_opts.fill_cache(true); // Keep hot keys in cache

                        // Optimized write options for batch writes
                        let mut write_opts = WriteOptions::default();
                        write_opts.disable_wal(true); // Disable WAL for speed (can rebuild on crash)
                        write_opts.set_sync(false); // Don't wait for fsync

                        while !stop.load(Ordering::Acquire) {
                            // Create ReadOptions for frontier iteration with optimizations
                            let mut read_opts = ReadOptions::default();
                            read_opts.set_total_order_seek(true);
                            read_opts.set_iterate_lower_bound(lower_bound.clone());
                            read_opts.set_iterate_upper_bound(upper_bound.clone());

                            // Optimize for sequential frontier scans
                            read_opts.fill_cache(false); // Don't pollute cache with one-time reads
                            read_opts.set_readahead_size(256 * 1024); // 256KB readahead for sequential access
                            read_opts.set_verify_checksums(false); // Skip checksums for speed

                            let iter =
                                db.iterator_cf_opt(&cf_frontier, read_opts, IteratorMode::Start);

                            let mut wb: WriteBatchWithTransaction<true> =
                                WriteBatchWithTransaction::<true>::default();

                            let mut items_processed = 0;
                            // Process items and track the last key using .last()
                            let _: Option<_> = iter
                                .take(WORKER_FRONTIER_BATCH_SIZE)
                                .filter_map(|item| {
                                    if let Ok((key, value)) = item {
                                        items_processed += 1;

                                        let current_frontier_item =
                                            TetrisAtlasFrontierKeyValue::from_value_bytes(
                                                &unsafe {
                                                    *(value.as_ref().as_ptr() as *const [u8; 41])
                                                },
                                            );

                                        frontier_consumed.fetch_add(1, Ordering::Relaxed);
                                        let board_index =
                                            boards_expanded.fetch_add(1, Ordering::Relaxed);

                                        // Process all pieces for this board state
                                        for (piece, bag_state) in
                                            current_frontier_item.bag.iter_next_states()
                                        {
                                            // Check if we've already computed this (board, piece) combination
                                            let lookup_key = TetrisAtlasLookupKeyValue::new(
                                                current_frontier_item.board,
                                                piece,
                                                TetrisPieceOrientation::default(),
                                            )
                                            .key_bytes();

                                            // Skip beam search if already in lookup table (cache hit)
                                            let should_skip = match db.key_may_exist_cf_opt_value(
                                                &cf_lookup,
                                                &lookup_key,
                                                &lookup_read_opts,
                                            ) {
                                                // definitely exists with value
                                                (true, Some(_)) => {
                                                    lookup_hits.fetch_add(1, Ordering::Relaxed);
                                                    true
                                                }
                                                // might exist, need to check
                                                (true, None) => {
                                                    if db
                                                        .get_cf_opt(
                                                            &cf_lookup,
                                                            &lookup_key,
                                                            &lookup_read_opts,
                                                        )
                                                        .ok()
                                                        .flatten()
                                                        .is_some()
                                                    {
                                                        lookup_hits.fetch_add(1, Ordering::Relaxed);
                                                        true
                                                    } else {
                                                        false
                                                    }
                                                }
                                                // definitely doesn't exist
                                                (false, _) => false,
                                            };

                                            if should_skip {
                                                continue;
                                            }

                                            // Cache miss - run beam search
                                            lookup_misses.fetch_add(1, Ordering::Relaxed);

                                            game.board = current_frontier_item.board;
                                            game.set_bag_piece_seeded(bag_state, piece, BASE_SEED);

                                            let tier =
                                                select_tier(current_frontier_item.board.height());
                                            let state = BeamTetrisState::new(game);
                                            let search_result = match tier {
                                                BeamTier::Low => beam_search_low.search_with_seeds(
                                                    state,
                                                    BASE_SEED,
                                                    LOW_MAX_DEPTH,
                                                ),
                                                BeamTier::Med => beam_search_med.search_with_seeds(
                                                    state,
                                                    BASE_SEED,
                                                    MED_MAX_DEPTH,
                                                ),
                                                BeamTier::High => beam_search_high
                                                    .search_with_seeds(
                                                        state,
                                                        BASE_SEED,
                                                        HIGH_MAX_DEPTH,
                                                    ),
                                            };

                                            let best_placement = match search_result {
                                                Some(result) => result,
                                                None => {
                                                    // Beam search failed (no valid placements at all)
                                                    games_lost.fetch_add(1, Ordering::Relaxed);
                                                    continue;
                                                }
                                            };

                                            // Skip if game is lost
                                            if game.apply_placement(best_placement).is_lost
                                                == IsLost::LOST
                                            {
                                                games_lost.fetch_add(1, Ordering::Relaxed);
                                                continue;
                                            }

                                            // Insert the new lookup entry
                                            let new_lookup_kv = TetrisAtlasLookupKeyValue::new(
                                                current_frontier_item.board,
                                                piece,
                                                best_placement.orientation,
                                            );
                                            wb.put_cf(
                                                &cf_lookup,
                                                new_lookup_kv.key_bytes(),
                                                new_lookup_kv.value_bytes(),
                                            );
                                            lookup_inserts.fetch_add(1, Ordering::Relaxed);

                                            // Add resulting board to frontier
                                            let new_frontier_kv = TetrisAtlasFrontierKeyValue::new(
                                                game.board, bag_state,
                                            );
                                            wb.put_cf(
                                                &cf_frontier,
                                                new_frontier_kv.key_bytes(),
                                                new_frontier_kv.value_bytes(),
                                            );
                                            frontier_enqueued.fetch_add(1, Ordering::Relaxed);
                                        }

                                        // Delete this frontier item from the queue after processing
                                        wb.delete_cf(&cf_frontier, &key);
                                        frontier_deleted.fetch_add(1, Ordering::Relaxed);

                                        Some(key)
                                    } else {
                                        None
                                    }
                                })
                                .last();

                            let _ = db
                                .write_opt(wb, &write_opts)
                                .expect("failed to write batch");

                            // Backoff when frontier is empty to avoid CPU spinning
                            if items_processed == 0 {
                                std::thread::sleep(Duration::from_millis(100));
                            }
                        }
                    })
                    .expect("Failed to spawn worker thread")
            })
            .collect()
    }

    /// Check if the system is idle (no items in frontier queue)
    pub fn is_idle(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let cf_frontier = self.cf_handle(CF_FRONTIER);
        let count = self
            .db
            .property_int_value_cf(&cf_frontier, "rocksdb.estimate-num-keys")?
            .unwrap_or(0);
        Ok(count == 0)
    }

    /// Get frontier queue size
    fn frontier_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let cf_frontier = self.cf_handle(CF_FRONTIER);
        let count = self
            .db
            .property_int_value_cf(&cf_frontier, "rocksdb.estimate-num-keys")?
            .unwrap_or(0) as usize;
        Ok(count)
    }

    /// Get lookup table size
    fn lookup_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let cf_lookup = self.cf_handle(CF_LOOKUP);
        let count = self
            .db
            .property_int_value_cf(&cf_lookup, "rocksdb.estimate-num-keys")?
            .unwrap_or(0) as usize;
        Ok(count)
    }

    /// Get all performance statistics (RocksDB doesn't have as many built-in stats as FeoxDB)
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

        // Print stats to stderr for visibility
        eprintln!("=== RocksDB Stats ===");
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
        frontier_consumption_rate,frontier_expansion_rate,frontier_expansion_ratio"
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
                f_consume={frontier_consumption_rate:.1}/s f_expand={frontier_expansion_rate:.1}/s f_ratio={frontier_expansion_ratio:.2}"
            );

            // Write to CSV
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

    // Wait for all worker threads to finish
    for h in handles {
        let _ = h.join();
    }

    eprintln!("All workers stopped.");
    println!("Finished in {:.2} seconds", start.elapsed().as_secs_f64());
}
