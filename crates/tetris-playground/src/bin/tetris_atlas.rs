#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(impl_trait_in_bindings)]
#![feature(const_index)]
//! # Tetris Atlas Builder
//!
//! This binary builds a comprehensive lookup table (atlas) of optimal Tetris moves by exhaustively
//! exploring the game state space using beam search. The goal is to map board states to optimal
//! piece placements such that the entire state space becomes self-referential.
//!
//! ## Approach
//!
//! 1. **Parallel Workers**: Multiple worker threads claim batches of board states from a frontier queue
//! 2. **Beam Search**: For each board state, uses multi-beam search to find optimal piece placements
//! 3. **RocksDB Storage**: Stores the lookup table and frontier in a persistent embedded database
//! 4. **Crash Recovery**: Supports resuming from crashes by tracking claimed-but-incomplete work
//!
//! ## Output
//!
//! - **RocksDB Database**: `tetris_atlas_rocksdb/` - Persistent lookup table with board → placements
//! - **Statistics CSV**: `tetris_atlas_stats.csv` - Real-time metrics (expansion rate, cache hits, etc.)
//!
//! ## Performance Optimizations
//!
//! - Parallel multi_get_cf for batched RocksDB reads
//! - Thread-local beam search reuse (no allocation per board)
//! - Chunked rayon processing to minimize TLS overhead
//! - Atomic batch operations with optimistic transactions

use rayon::prelude::*;
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

const BEAM_WIDTH: usize = 8;
const MAX_DEPTH: usize = 8;
const NUM_SEARCHES: usize = 8;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

// Thread-local beam search instance for rayon worker threads
thread_local! {
    static THREAD_BEAM_SEARCH: std::cell::RefCell<Option<MultiBeamSearch<BeamTetrisState, NUM_SEARCHES, 1, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>>> = std::cell::RefCell::new(None);
}

const DB_PATH: &str = "tetris_atlas_db";
const STATS_CSV_PATH: &str = "tetris_atlas_stats.csv";
const LOG_EVERY_SECS: u64 = 3;
const IDLE_GRACE_SECS: u64 = 15;
const WORKER_RECV_TIMEOUT_MS: u64 = 100;

const NUM_WORKERS: usize = 16;
const WORKER_FRONTIER_BATCH_SIZE: usize = 2048;
const WORKER_CHUNK_SIZE: usize = 256;
const QUEUE_SIZE: usize = 1024;

const WORK_ITEM_BYTE: u8 = b'W';
const LOOKUP_ITEM_BYTE: u8 = b'L';
const EMPTY_BYTES: &[u8] = &[];

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
    // FNV-1a hash: fast, good distribution, no state needed
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let board_bytes: [u8; 40] = board.into();

    let mut hash = FNV_OFFSET;
    repeat_idx_unroll!(40, I, {
        hash ^= board_bytes[I] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    });
    hash ^= u8::from(bag) as u64;
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
        key[0] = WORK_ITEM_BYTE;
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
        out[0] = LOOKUP_ITEM_BYTE;
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

struct TetrisAtlasDB {
    store: Arc<feoxdb::FeoxStore>,
    stop: Arc<AtomicBool>,

    // Performance counters
    lookup_hits: Arc<AtomicU64>,
    lookup_misses: Arc<AtomicU64>,
    lookup_inserts: Arc<AtomicU64>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    games_lost: Arc<AtomicU64>,
    boards_expanded: Arc<AtomicU64>,
}

impl TetrisAtlasDB {
    fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        const MB: u64 = feoxdb::constants::MB as u64;
        const GB: u64 = feoxdb::constants::GB as u64;

        let store = feoxdb::FeoxStore::builder()
            .device_path(path)
            // .file_size(10 * GB)
            // .max_memory(8 * GB as usize)
            // .enable_caching(true)
            // .enable_caching(false)
            // .hash_bits(16)
            // .enable_ttl(false)
            .build()?;

        Ok(Self {
            store: Arc::new(store),
            stop: Arc::new(AtomicBool::new(false)),
            lookup_hits: Arc::new(AtomicU64::new(0)),
            lookup_misses: Arc::new(AtomicU64::new(0)),
            lookup_inserts: Arc::new(AtomicU64::new(0)),
            frontier_enqueued: Arc::new(AtomicU64::new(0)),
            frontier_consumed: Arc::new(AtomicU64::new(0)),
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
                    // Keys are [LOOKUP_ITEM_BYTE, hash_byte_0, hash_byte_1, ...]
                    // We partition on hash_byte_0 (second byte of the key)
                    let partition_start = ((worker_id as u16 * 256) / num_workers as u16) as u8;

                    // For the last worker, use 0xFF + 1 as the end (scanning to the end of the partition)
                    let start_key = [LOOKUP_ITEM_BYTE, partition_start];
                    let end_key = if worker_id == num_workers - 1 {
                        // Last worker: scan from partition_start to end of byte range
                        [LOOKUP_ITEM_BYTE + 1, 0] // Next prefix
                    } else {
                        let partition_end =
                            (((worker_id + 1) as u16 * 256) / num_workers as u16) as u8;
                        [LOOKUP_ITEM_BYTE, partition_end]
                    };

                    while !stop.load(Ordering::Acquire) {
                        // Scan this worker's partition for frontier items and process in parallel
                        let count = store
                            .range_query_iter(&start_key, &end_key, 1024)
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

                                    let existing = store.contains_key(&new_lookup_key);

                                    if !existing {
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

    /// Get frontier queue size by counting all LOOKUP_ITEM_BYTE prefixed entries
    fn frontier_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let start_key = [LOOKUP_ITEM_BYTE, 0];
        let end_key = [LOOKUP_ITEM_BYTE + 1, 0];
        let count = self
            .store
            .range_query_iter(&start_key, &end_key, 10000)
            .expect("Failed to query frontier")
            .count();
        Ok(count)
    }

    /// Get lookup table size by counting all WORK_ITEM_BYTE prefixed entries
    fn lookup_size(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let start_key = [WORK_ITEM_BYTE, 0];
        let end_key = [WORK_ITEM_BYTE + 1, 0];
        let count = self
            .store
            .range_query_iter(&start_key, &end_key, 10000)
            .expect("Failed to query lookup")
            .count();
        Ok(count)
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
        let games_lost = self.games_lost.load(Ordering::Relaxed);
        let boards_expanded = self.boards_expanded.load(Ordering::Relaxed);

        // Get feoxdb stats
        let db_stats = self.store.stats();

        // Print ALL feoxdb stats to stderr for visibility
        eprintln!("=== FeoxDB Stats ===");
        eprintln!("Store Metrics:");
        eprintln!("  Record count: {}", db_stats.record_count);
        eprintln!("  Memory usage: {} bytes", db_stats.memory_usage);
        eprintln!("Operations:");
        eprintln!("  Total operations: {}", db_stats.total_operations);
        eprintln!("  Total gets: {}", db_stats.total_gets);
        eprintln!("  Total inserts: {}", db_stats.total_inserts);
        eprintln!("  Total updates: {}", db_stats.total_updates);
        eprintln!("  Total deletes: {}", db_stats.total_deletes);
        eprintln!("  Total range queries: {}", db_stats.total_range_queries);
        eprintln!("Latencies:");
        eprintln!("  Avg get latency: {} ns", db_stats.avg_get_latency_ns);
        eprintln!(
            "  Avg insert latency: {} ns",
            db_stats.avg_insert_latency_ns
        );
        eprintln!(
            "  Avg delete latency: {} ns",
            db_stats.avg_delete_latency_ns
        );
        eprintln!("Cache:");
        eprintln!("  Cache hits: {}", db_stats.cache_hits);
        eprintln!("  Cache misses: {}", db_stats.cache_misses);
        eprintln!("  Cache hit rate: {:.2}%", db_stats.cache_hit_rate);
        eprintln!("  Cache evictions: {}", db_stats.cache_evictions);
        eprintln!("  Cache memory: {} bytes", db_stats.cache_memory);
        eprintln!("Write Buffer:");
        eprintln!("  Writes buffered: {}", db_stats.writes_buffered);
        eprintln!("  Writes flushed: {}", db_stats.writes_flushed);
        eprintln!("  Write failures: {}", db_stats.write_failures);
        eprintln!("  Flush count: {}", db_stats.flush_count);
        eprintln!("Disk I/O:");
        eprintln!("  Disk reads: {}", db_stats.disk_reads);
        eprintln!("  Disk writes: {}", db_stats.disk_writes);
        eprintln!("  Disk bytes read: {}", db_stats.disk_bytes_read);
        eprintln!("  Disk bytes written: {}", db_stats.disk_bytes_written);
        eprintln!("Errors:");
        eprintln!("  Key not found errors: {}", db_stats.key_not_found_errors);
        eprintln!("  Out of memory errors: {}", db_stats.out_of_memory_errors);
        eprintln!("  I/O errors: {}", db_stats.io_errors);
        eprintln!("===================");

        Ok((
            // App counters
            lookup_hits,
            lookup_misses,
            lookup_inserts,
            frontier_enqueued,
            frontier_consumed,
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

// struct AtlasDb {
//     db: Arc<OptimisticTransactionDB>,

//     // counters
//     retry_count: AtomicU64,
//     lookup_hits: AtomicU64,
//     lookup_misses: AtomicU64,
//     frontier_consumed: AtomicU64,
//     frontier_enqueued: AtomicU64,
// }

// impl AtlasDb {
//     fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
//         let mut opts = Options::default();
//         opts.create_if_missing(true);
//         opts.create_missing_column_families(true);

//         // Performance tuning
//         opts.set_max_background_jobs(num_cpus::get() as i32);
//         opts.set_write_buffer_size(256 * 1024 * 1024); // 256MB
//         opts.set_max_write_buffer_number(4);
//         opts.set_target_file_size_base(256 * 1024 * 1024);
//         opts.set_level_zero_file_num_compaction_trigger(4);
//         opts.set_compression_type(rocksdb::DBCompressionType::None);

//         let mut block_opts = rocksdb::BlockBasedOptions::default();
//         block_opts.set_block_size(16 * 1024);
//         block_opts.set_cache_index_and_filter_blocks(true);
//         block_opts.set_bloom_filter(10.0, false);
//         opts.set_block_based_table_factory(&block_opts);

//         // Lookup CF gets larger write buffer and merge operator
//         let mut cf_opts_lookup = Options::default();
//         cf_opts_lookup.set_write_buffer_size(512 * 1024 * 1024);

//         let cf_opts = Options::default();
//         let cfs = vec![
//             ColumnFamilyDescriptor::new(CF_LOOKUP, cf_opts_lookup),
//             ColumnFamilyDescriptor::new(CF_FQUEUE, cf_opts.clone()),
//             ColumnFamilyDescriptor::new(CF_META, cf_opts),
//         ];

//         let db: OptimisticTransactionDB<MultiThreaded> =
//             OptimisticTransactionDB::open_cf_descriptors(&opts, path, cfs)?;

//         Ok(Self {
//             db: Arc::new(db),
//             retry_count: AtomicU64::new(0),
//             lookup_hits: AtomicU64::new(0),
//             lookup_misses: AtomicU64::new(0),
//             frontier_consumed: AtomicU64::new(0),
//             frontier_enqueued: AtomicU64::new(0),
//         })
//     }

//     #[inline(always)]
//     fn cf_handle(&self, name: &str) -> Arc<rocksdb::BoundColumnFamily<'_>> {
//         self.db
//             .cf_handle(name)
//             .expect(&format!("missing {} cf", name))
//     }

//     /// Helper to retry a transaction operation with exponential backoff
//     fn retry_txn<F, T>(&self, mut f: F) -> Result<T, Box<dyn std::error::Error>>
//     where
//         F: FnMut() -> Result<T, Box<dyn std::error::Error>>,
//     {
//         let mut attempt = 0;
//         backoff::retry(
//             ExponentialBackoff::default(),
//             || -> Result<T, backoff::Error<Box<dyn std::error::Error>>> {
//                 if attempt > 0 {
//                     self.retry_count.fetch_add(1, Ordering::Relaxed);
//                 }
//                 attempt += 1;
//                 f().map_err(|e| backoff::Error::transient(e))
//             },
//         )
//         .map_err(|e| match e {
//             backoff::Error::Permanent(err) => err,
//             backoff::Error::Transient { err, .. } => err,
//         })
//     }

//     // ----------------------- Frontier (durable queue) ----------------------

//     fn insert_worker_results_batch(
//         &self,
//         worker_results_iter: impl Iterator<Item = WorkerResult>,
//     ) -> Result<usize, Box<dyn std::error::Error>> {
//         let cf_lookup = self.cf_handle(CF_LOOKUP);
//         let cf_fqueue = self.cf_handle(CF_FQUEUE);

//         // Stateless hash function - no allocations needed per iteration
//         #[inline(always)]
//         fn hash_board_bag(board_bytes: &[u8; 40], bag_byte: u8) -> u64 {
//             // FNV-1a hash: fast, good distribution, no state needed
//             const FNV_OFFSET: u64 = 0xcbf29ce484222325;
//             const FNV_PRIME: u64 = 0x100000001b3;

//             let mut hash = FNV_OFFSET;
//             repeat_idx_unroll!(40, I, {
//                 hash ^= board_bytes[I] as u64;
//                 hash = hash.wrapping_mul(FNV_PRIME);
//             });
//             hash ^= bag_byte as u64;
//             hash = hash.wrapping_mul(FNV_PRIME);
//             hash
//         }

//         let mut wb: WriteBatchWithTransaction<true> = WriteBatchWithTransaction::<true>::default();
//         worker_results_iter.for_each(|worker_result| {
//             let lookup_update = worker_result.lookup_update;
//             let lookup_key = lookup_update.key();

//             let may_exist =
//                 self.db
//                     .key_may_exist_cf_opt_value(&cf_lookup, lookup_key, &ReadOptions::default());
//             match may_exist {
//                 // definately doesn't exist
//                 (false, _) => {
//                     wb.put_cf(cf_lookup, lookup_key, lookup_update.value());
//                 }
//             };

//             let (key, value) = worker_result.lookup_update.into_key_value();
//             wb.merge_cf(&cf_lookup, key.0, value.to_bytes());

//             worker_result
//                 .new_frontier_items
//                 .iter()
//                 .for_each(|new_frontier_item| {
//                     let board_bytes: [u8; 40] = new_frontier_item.board.into();
//                     let bag_byte = u8::from(new_frontier_item.bag);
//                     let hash = hash_board_bag(&board_bytes, bag_byte);

//                     let fqueue_key: [u8; 8] = FrontierQueueKey(hash).to_bytes();
//                     let fqueue_value: [u8; 41] = (*new_frontier_item).into();
//                     wb.put_cf(&cf_fqueue, fqueue_key, fqueue_value);
//                 });
//         });

//         let wb_len = wb.len();
//         self.frontier_enqueued
//             .fetch_add(wb_len as u64, Ordering::Relaxed);
//         self.db.write_opt(wb, &WriteOptions::default())?;
//         Ok(wb_len)
//     }

//     fn frontier_pop_batch(
//         &self,
//         worker_id: u8,
//         num_workers: u8,
//         read_buf: &mut HeaplessVec<FrontierQueueValue, WORKER_FRONTIER_BATCH_SIZE>,
//     ) -> Result<(), Box<dyn std::error::Error>> {
//         let mut wb: WriteBatchWithTransaction<true> = WriteBatchWithTransaction::<true>::default();
//         let cf_fqueue = self.cf_handle(CF_FQUEUE)?;

//         let worker_start_byte = ((worker_id as u16 * 256) / num_workers as u16) as u8;
//         let worker_end_byte = (((worker_id + 1) as u16 * 256) / num_workers as u16) as u8;
//         let worker_db_iter = self.db.prefix_iterator_cf(&cf_fqueue, &[worker_start_byte]);

//         read_buf.clear();
//         worker_db_iter
//             .take(WORKER_FRONTIER_BATCH_SIZE)
//             .take_while(|item| {
//                 let key_first_byte = item.as_ref().unwrap().0[0];
//                 key_first_byte >= worker_start_byte && key_first_byte < worker_end_byte
//             })
//             .for_each(|item| {
//                 let (key, value) = item.unwrap();
//                 let frontier_queue_item = FrontierQueueValue::from(value.as_ref());
//                 let _ = read_buf.try_push(frontier_queue_item);
//                 wb.delete_cf(&cf_fqueue, key.as_ref());
//             });

//         if !read_buf.is_empty() {
//             self.db.write_opt(wb, &WriteOptions::default())?;
//         }

//         Ok(())
//     }

//     fn frontier_count(&self) -> Result<usize, Box<dyn std::error::Error>> {
//         let cf_fqueue = self.cf_handle(CF_FQUEUE)?;
//         let count = self
//             .db
//             .property_int_value_cf(&cf_fqueue, "rocksdb.estimate-num-keys")?
//             .unwrap_or(0) as usize;
//         Ok(count)
//     }

//     fn lookup_count(&self) -> Result<usize, Box<dyn std::error::Error>> {
//         let cf_lookup = self.cf_handle(CF_LOOKUP)?;
//         let count = self
//             .db
//             .property_int_value_cf(&cf_lookup, "rocksdb.estimate-num-keys")?
//             .unwrap_or(0) as usize;
//         Ok(count)
//     }

//     fn get_stats(&self) -> (u64, u64, u64, u64, u64) {
//         (
//             self.retry_count.load(Ordering::Relaxed),
//             self.lookup_hits.load(Ordering::Relaxed),
//             self.lookup_misses.load(Ordering::Relaxed),
//             self.frontier_consumed.load(Ordering::Relaxed),
//             self.frontier_enqueued.load(Ordering::Relaxed),
//         )
//     }
// }

// // ----------------------------- TetrisAtlas ---------------------------------

// pub struct TetrisAtlas {
//     db: AtlasDb,
//     stop: AtomicBool,
//     in_flight: AtomicU64,
//     expanded: AtomicU64,
//     discovered: AtomicU64,
//     games_lost: AtomicU64,
// }

// impl TetrisAtlas {
//     pub fn new(db_path: String) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
//         let db = AtlasDb::open(&db_path)?;
//         Ok(Arc::new(Self {
//             db,
//             stop: AtomicBool::new(false),
//             in_flight: AtomicU64::new(0),
//             expanded: AtomicU64::new(0),
//             discovered: AtomicU64::new(0),
//             games_lost: AtomicU64::new(0),
//         }))
//     }

//     pub fn seed_starting_state(&self) -> Result<(), Box<dyn std::error::Error>> {
//         let empty_board = TetrisBoard::new();
//         let bag = TetrisPieceBagState::new();

//         // Enqueue starting state
//         let cf_fqueue = self.db.cf_handle(CF_FQUEUE)?;
//         let fqueue_key: [u8; 8] = FrontierQueueKey(0).to_bytes();
//         let fqueue_value: [u8; 41] = FrontierQueueValue::from((empty_board, bag)).into();
//         self.db.db.put_cf(&cf_fqueue, fqueue_key, fqueue_value)?;

//         self.discovered.fetch_add(1, Ordering::Relaxed);
//         Ok(())
//     }

//     pub fn stop(&self) {
//         self.stop.store(true, Ordering::Release);
//     }

//     pub fn is_idle(&self) -> Result<bool, Box<dyn std::error::Error>> {
//         if self.in_flight.load(Ordering::Relaxed) > 0 {
//             return Ok(false);
//         }

//         let count = self.db.frontier_count()?;
//         Ok(count == 0)
//     }

//     pub fn stats(
//         &self,
//     ) -> Result<(u64, u64, u64, usize, usize, u64, u64, u64, u64, u64), Box<dyn std::error::Error>>
//     {
//         let expanded = self.expanded.load(Ordering::Relaxed);
//         let discovered = self.discovered.load(Ordering::Relaxed);
//         let in_flight = self.in_flight.load(Ordering::Relaxed);

//         let table_len = self.db.lookup_count()?;
//         let frontier_size = self.db.frontier_count()?;

//         let (retry_count, lookup_hits, lookup_misses, frontier_consumed, frontier_enqueued) =
//             self.db.get_stats();

//         Ok((
//             expanded,
//             discovered,
//             in_flight,
//             table_len,
//             frontier_size,
//             retry_count,
//             lookup_hits,
//             lookup_misses,
//             frontier_consumed,
//             frontier_enqueued,
//         ))
//     }

//     pub fn launch_worker(self: &Arc<Self>, worker_id: u64) -> JoinHandle<()> {
//         let atlas = Arc::clone(self);
//         thread::spawn(move || {
//             if let Err(e) = atlas.worker_loop(worker_id) {
//                 eprintln!("Worker {} error: {}", worker_id, e);
//             }
//         })
//     }

//     fn worker_loop(&self, worker_id: u64) -> Result<(), Box<dyn std::error::Error>> {
//         let mut local_ctr: u64 = 0;
//         let mut batch_buf: HeaplessVec<FrontierQueueValue, WORKER_FRONTIER_BATCH_SIZE> =
//             HeaplessVec::new();

//         while !self.stop.load(Ordering::Acquire) {
//             self.db
//                 .frontier_pop_batch(worker_id as u8, NUM_WORKERS as u8, &mut batch_buf)?;

//             if batch_buf.is_empty() {
//                 thread::sleep(Duration::from_millis(10));
//                 continue;
//             }

//             let work_results = batch_buf
//                 .to_slice()
//                 .par_iter()
//                 .chunks(WORKER_CHUNK_SIZE)
//                 .flat_map_iter(|chunk| {
//                     // Take beam search ONCE per chunk (not per board!)
//                     let mut thread_search = THREAD_BEAM_SEARCH.with(|tls| {
//                         tls.borrow_mut().take().unwrap_or_else(|| {
//                             MultiBeamSearch::<
//                                 BeamTetrisState,
//                                 NUM_SEARCHES,
//                                 BEAM_WIDTH,
//                                 MAX_DEPTH,
//                                 MAX_MOVES,
//                             >::new()
//                         })
//                     });

//                     const BASE_SEED: u64 = 42;
//                     let mut base_game = tetris_game::TetrisGame::new();

//                     let chunk_results = CHUNK_RESULTS_BUF.with(|chunk_buf_cell| {
//                         let mut chunk_buf = chunk_buf_cell.borrow_mut();
//                         chunk_buf.clear();

//                         let mut local_games_lost = 0u64;
//                         for &item in chunk.iter() {
//                             for (piece, bag_state) in item.bag.iter_next_states() {
//                                 base_game.board = item.board;
//                                 base_game.set_bag_piece_seeded(bag_state, piece, BASE_SEED);

//                                 let (best_placement, _score) = thread_search
//                                     .search_with_seeds(
//                                         BeamTetrisState::new(base_game),
//                                         BASE_SEED,
//                                         MAX_DEPTH,
//                                     )
//                                     .expect("Failed to search");

//                                 if base_game.apply_placement(best_placement).is_lost != IsLost::LOST
//                                 {
//                                     chunk_buf.push(WorkerResult {
//                                         lookup_update: LookupUpdate::new(
//                                             item.board,
//                                             piece,
//                                             best_placement.orientation,
//                                         ),
//                                         new_frontier: FrontierQueueValue::from((
//                                             base_game.board,
//                                             bag_state,
//                                         )),
//                                     });
//                                 } else {
//                                     local_games_lost += 1;
//                                 }
//                             }
//                         }
//                         if local_games_lost > 0 {
//                             self.games_lost
//                                 .fetch_add(local_games_lost, Ordering::Relaxed);
//                         }

//                         // Return owned slice iterator (no clone!)
//                         chunk_buf.to_slice().iter().copied().collect::<Vec<_>>()
//                     });

//                     // Return beam search to TLS AFTER processing the entire chunk
//                     THREAD_BEAM_SEARCH.with(|tls| {
//                         *tls.borrow_mut() = Some(thread_search);
//                     });

//                     chunk_results.into_iter()
//                 });

//             // Collect parallel iterator into thread-local Vec (reuses allocation)
//             COLLECT_BUF.with(|buf_cell| {
//                 let mut buf = buf_cell.borrow_mut();
//                 buf.clear();
//                 buf.par_extend(work_results);

//                 self.db.insert_worker_results_batch(buf.iter().copied())
//             })?;
//         }

//         Ok(())
//     }
// }

fn main() {
    run_tetris_atlas();
}

pub fn run_tetris_atlas() {
    set_global_threadpool();

    let start = Instant::now();

    let atlas = TetrisAtlasDB::new(DB_PATH).expect("Failed to create atlas");
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
        frontier_enqueued,frontier_consumed,games_lost,lookup_size,frontier_size,\
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

            // Write to CSV with ALL stats
            writeln!(
                csv_writer,
                "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{}",
                secs,
                boards_expanded,
                lookup_hits,
                lookup_misses,
                lookup_inserts,
                frontier_enqueued,
                frontier_consumed,
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
