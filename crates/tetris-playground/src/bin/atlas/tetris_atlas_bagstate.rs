#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(const_convert)]
#![feature(const_trait_impl)]
#![feature(impl_trait_in_bindings)]
#![feature(const_index)]
#![feature(const_cmp)]
//! # Tetris Atlas Builder — Known-Sequence (Bag-State) Edition
//!
//! The player sees the full ordered 7-piece bag before placing anything.
//! Atlas key = `(board, perm_index)`, value = `[placement_index; 7]`.
//!
//! For each (board, bag_permutation), beam search plans all 7 placements
//! looking ahead at the full sequence. The atlas stores the complete plan.

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
    IsLost, Major, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, constants,
};
use tetris_search::{BeamTetrisState, TetrisMultiBeamSearch, height_mse_beam_tetris_score};
use tetris_utils::repeat_idx_unroll;

// --- Beam Search Parameters ---
const N: usize = 8;
const TOP_N_PER_BEAM: usize = 64;
const BEAM_WIDTH: usize = 64;
const MAX_DEPTH: usize = 6;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;

const fn filter_fn(state: &BeamTetrisState) -> bool {
    let height = state.0.board.height();
    let holes = state.0.board.total_holes();
    height <= 6 && holes <= 4
}

// --- Threading ---
const CHANNEL_CAPACITY: usize = 256;
const NUM_WORKERS: usize = 16;
const WORKER_FRONTIER_BATCH_SIZE: usize = 128;
const WORKER_STACK_BYTES: usize = 256 * 1024 * 1024;

// --- Persistence ---
const LOG_EVERY_SECS: u64 = 3;
const STATS_CSV_PATH: &str = "tetris_atlas_bagstate_stats.csv";
const BASE_SEED: u64 = 42;

const CF_LOOKUP: &str = "lookup";
const CF_FRONTIER: &str = "frontier";
const CF_SEEDED: &str = "seeded"; // tracks which boards have been seeded with 5040 perms
const CF_META: &str = "meta";

// ---------------------------------------------------------------------------
// Compile-time permutation generation
// ---------------------------------------------------------------------------

const NUM_PERMS: usize = 5040;
type Bag = [TetrisPiece; 7];

const fn advance_permutation(indices: &mut [u8; 7]) -> bool {
    let n = 7;
    if n < 2 {
        return false;
    }
    let mut pivot = n - 2;
    loop {
        if indices[pivot] < indices[pivot + 1] {
            break;
        }
        if pivot == 0 {
            return false;
        }
        pivot -= 1;
    }
    let mut successor = n - 1;
    while indices[pivot] >= indices[successor] {
        successor -= 1;
    }
    let tmp = indices[pivot];
    indices[pivot] = indices[successor];
    indices[successor] = tmp;
    let mut left = pivot + 1;
    let mut right = n - 1;
    while left < right {
        let tmp = indices[left];
        indices[left] = indices[right];
        indices[right] = tmp;
        left += 1;
        right -= 1;
    }
    true
}

const fn generate_all_permutations() -> [Bag; NUM_PERMS] {
    let mut permutations = [[TetrisPiece::O_PIECE; 7]; NUM_PERMS];
    let mut indices = [0u8, 1, 2, 3, 4, 5, 6];
    let mut out_idx = 0usize;
    loop {
        let mut bag = [TetrisPiece::O_PIECE; 7];
        let mut i = 0usize;
        while i < 7 {
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

/// All 5040 bag permutations, computed at compile time.
static ALL_PERMUTATIONS: [Bag; NUM_PERMS] = generate_all_permutations();

// ---------------------------------------------------------------------------
// Key/Value types
// ---------------------------------------------------------------------------

/// Lookup key: (board, perm_index) = 42 bytes.
/// Value: [placement_index; 7] = 7 bytes — the full 7-move plan for this bag.
const LOOKUP_KEY_LEN: usize = 42;

#[inline(always)]
const fn hash_board_perm(board: TetrisBoard, perm_index: u16) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let board_bytes: [u32; StandardTetris::COLS] = board.as_limbs();
    let mut hash = FNV_OFFSET;
    repeat_idx_unroll!(StandardTetris::COLS, I, {
        hash ^= board_bytes[I] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    });
    hash ^= perm_index as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash
}

fn make_lookup_key(board: TetrisBoard, perm_index: u16) -> [u8; LOOKUP_KEY_LEN] {
    let board_bytes = board.to_bytes(Major::Col);
    let mut key = [0u8; LOOKUP_KEY_LEN];
    key[0..40].copy_from_slice(&board_bytes);
    key[40..42].copy_from_slice(&perm_index.to_le_bytes());
    key
}

/// Frontier item: (board, perm_index). Serialized as 42 bytes.
/// The frontier key includes a priority prefix for ordered processing.
fn make_frontier_key(board: TetrisBoard, perm_index: u16) -> [u8; 9] {
    let height = board.height();
    let holes = board.total_holes();
    let cost = (height + holes).min(255) as u8;
    let hash = hash_board_perm(board, perm_index);
    let mut key = [0u8; 9];
    key[0] = cost;
    key[1..9].copy_from_slice(&hash.to_be_bytes());
    key
}

fn make_frontier_value(board: TetrisBoard, perm_index: u16) -> [u8; LOOKUP_KEY_LEN] {
    make_lookup_key(board, perm_index) // same encoding
}

fn parse_frontier_value(v: &[u8]) -> (TetrisBoard, u16) {
    let board_bytes: [u8; 40] = v[0..40].try_into().unwrap_or([0u8; 40]);
    let perm_index = u16::from_le_bytes([v[40], v[41]]);
    (TetrisBoard::from_bytes(board_bytes, Major::Col), perm_index)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "tetris-atlas-bagstate")]
#[command(about = "Bag-State Tetris Atlas — key=(board, perm), value=[placement; 7]")]
struct Cli {
    #[arg(short, long)]
    db_path: String,
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    Create,
    Explore,
}

// ---------------------------------------------------------------------------
// DB wrapper
// ---------------------------------------------------------------------------

pub struct AtlasDB {
    db: Arc<OptimisticTransactionDB<MultiThreaded>>,
    stop: Arc<AtomicBool>,
    lookup_hits: Arc<AtomicU64>,
    lookup_misses: Arc<AtomicU64>,
    lookup_inserts: Arc<AtomicU64>,
    frontier_enqueued: Arc<AtomicU64>,
    frontier_consumed: Arc<AtomicU64>,
    frontier_deleted: Arc<AtomicU64>,
    games_lost: Arc<AtomicU64>,
    boards_expanded: Arc<AtomicU64>,
}

impl AtlasDB {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        opts.set_max_background_jobs(num_cpus::get() as i32);
        opts.set_write_buffer_size(256 * 1024 * 1024);
        opts.set_max_write_buffer_number(6);
        opts.set_min_write_buffer_number_to_merge(2);
        opts.set_target_file_size_base(256 * 1024 * 1024);
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_allow_concurrent_memtable_write(true);

        let cache = rocksdb::Cache::new_lru_cache(2 * 1024 * 1024 * 1024);
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_size(16 * 1024);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_cache(&cache);
        opts.set_block_based_table_factory(&block_opts);

        let mut cf_lookup = Options::default();
        cf_lookup.set_write_buffer_size(512 * 1024 * 1024);
        let mut blk_lookup = rocksdb::BlockBasedOptions::default();
        blk_lookup.set_block_cache(&cache);
        blk_lookup.set_bloom_filter(10.0, false);
        cf_lookup.set_block_based_table_factory(&blk_lookup);

        let mut cf_frontier = Options::default();
        cf_frontier.set_write_buffer_size(128 * 1024 * 1024);
        cf_frontier.set_compaction_style(rocksdb::DBCompactionStyle::Universal);

        let mut cf_seeded = Options::default();
        cf_seeded.set_write_buffer_size(64 * 1024 * 1024);
        let mut blk_seeded = rocksdb::BlockBasedOptions::default();
        blk_seeded.set_bloom_filter(10.0, false);
        blk_seeded.set_block_cache(&cache);
        cf_seeded.set_block_based_table_factory(&blk_seeded);

        let cfs = vec![
            ColumnFamilyDescriptor::new(CF_LOOKUP, cf_lookup),
            ColumnFamilyDescriptor::new(CF_FRONTIER, cf_frontier),
            ColumnFamilyDescriptor::new(CF_SEEDED, cf_seeded),
            ColumnFamilyDescriptor::new(CF_META, Options::default()),
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

    fn cf_handle(&self, name: &str) -> Arc<rocksdb::BoundColumnFamily<'_>> {
        self.db.cf_handle(name).expect("missing column family")
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

    /// Seed the empty board with all 5040 bag permutations.
    fn seed_board(&self, board: TetrisBoard) -> Result<bool, Box<dyn std::error::Error>> {
        let board_bytes = board.to_bytes(Major::Col);
        let cf_seeded = self.cf_handle(CF_SEEDED);
        let cf_frontier = self.cf_handle(CF_FRONTIER);

        // Check if already seeded (exact check, not bloom filter)
        if self.db.get_cf(&cf_seeded, &board_bytes)?.is_some() {
            return Ok(false); // already seeded
        }

        // Mark as seeded FIRST (prevents races)
        self.db.put_cf(&cf_seeded, &board_bytes, &[1u8])?;

        // Add all 5040 permutations to frontier
        for perm_idx in 0..NUM_PERMS as u16 {
            let fk = make_frontier_key(board, perm_idx);
            let fv = make_frontier_value(board, perm_idx);
            self.db.put_cf(&cf_frontier, fk, fv)?;
        }
        self.frontier_enqueued
            .fetch_add(NUM_PERMS as u64, Ordering::Relaxed);
        Ok(true)
    }

    fn seed_starting_state(&self) -> Result<(), Box<dyn std::error::Error>> {
        let empty = TetrisBoard::EMPTY_BOARD;
        println!("Seeding {} permutations from empty board...", NUM_PERMS);
        self.seed_board(empty)?;
        println!("Seeded.");
        Ok(())
    }

    fn is_board_seeded(&self, board: TetrisBoard) -> bool {
        let board_bytes = board.to_bytes(Major::Col);
        let cf_seeded = self.cf_handle(CF_SEEDED);
        self.db
            .get_cf(&cf_seeded, &board_bytes)
            .ok()
            .flatten()
            .is_some()
    }

    fn seeded_count(&self) -> usize {
        let cf = self.cf_handle(CF_SEEDED);
        self.db
            .property_int_value_cf(&cf, "rocksdb.estimate-num-keys")
            .ok()
            .flatten()
            .unwrap_or(0) as usize
    }

    fn frontier_size(&self) -> usize {
        let cf = self.cf_handle(CF_FRONTIER);
        self.db
            .property_int_value_cf(&cf, "rocksdb.estimate-num-keys")
            .ok()
            .flatten()
            .unwrap_or(0) as usize
    }

    fn lookup_size(&self) -> usize {
        let cf = self.cf_handle(CF_LOOKUP);
        self.db
            .property_int_value_cf(&cf, "rocksdb.estimate-num-keys")
            .ok()
            .flatten()
            .unwrap_or(0) as usize
    }

    // -----------------------------------------------------------------------
    // Worker loop
    // -----------------------------------------------------------------------

    fn run_worker_loop(&self, num_workers: usize) -> Vec<JoinHandle<()>> {
        (0..num_workers)
            .map(|worker_id| {
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

                thread::Builder::new()
                    .name(format!("atlas-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn(move || {
                        let mut beam_search =
                            TetrisMultiBeamSearch::<
                                N,
                                TOP_N_PER_BEAM,
                                BEAM_WIDTH,
                                MAX_DEPTH,
                                MAX_MOVES,
                            >::new(height_mse_beam_tetris_score);

                        let mut game = tetris_game::TetrisGame::new();

                        // Partition frontier key space across workers
                        let part_start = (worker_id as u16 * 256 / num_workers as u16) as u8;
                        let part_end_u16 = ((worker_id + 1) as u16 * 256) / num_workers as u16;
                        let lower = [0u8, part_start, 0, 0, 0, 0, 0, 0, 0].to_vec();
                        let upper = if part_end_u16 > 255 {
                            vec![0xFF; 9]
                        } else {
                            [0xFF, part_end_u16 as u8, 0, 0, 0, 0, 0, 0, 0].to_vec()
                        };

                        let cf_frontier = db.cf_handle(CF_FRONTIER).expect("missing frontier cf");
                        let cf_lookup = db.cf_handle(CF_LOOKUP).expect("missing lookup cf");

                        let mut lookup_ropts = ReadOptions::default();
                        lookup_ropts.set_verify_checksums(true);
                        lookup_ropts.fill_cache(true);

                        let mut write_opts = WriteOptions::default();
                        write_opts.disable_wal(true);
                        write_opts.set_sync(false);

                        while !stop.load(Ordering::Acquire) {
                            let mut read_opts = ReadOptions::default();
                            read_opts.set_total_order_seek(true);
                            read_opts.set_iterate_lower_bound(lower.clone());
                            read_opts.set_iterate_upper_bound(upper.clone());
                            read_opts.fill_cache(false);
                            read_opts.set_readahead_size(256 * 1024);
                            read_opts.set_verify_checksums(false);

                            let iter =
                                db.iterator_cf_opt(&cf_frontier, read_opts, IteratorMode::Start);

                            let mut wb = WriteBatchWithTransaction::<true>::default();
                            let mut items_processed = 0;

                            let _: Option<_> = iter
                                .take(WORKER_FRONTIER_BATCH_SIZE)
                                .filter_map(|item| {
                                    let (key, value) = match item {
                                        Ok(kv) => kv,
                                        Err(_) => return None,
                                    };
                                    items_processed += 1;
                                    frontier_consumed.fetch_add(1, Ordering::Relaxed);
                                    boards_expanded.fetch_add(1, Ordering::Relaxed);

                                    let (board, perm_idx) = parse_frontier_value(value.as_ref());
                                    let perm = &ALL_PERMUTATIONS[perm_idx as usize];

                                    // Check if already in lookup
                                    let lk = make_lookup_key(board, perm_idx);
                                    let skip = match db.key_may_exist_cf_opt_value(
                                        &cf_lookup,
                                        lk,
                                        &lookup_ropts,
                                    ) {
                                        (true, Some(_)) => {
                                            lookup_hits.fetch_add(1, Ordering::Relaxed);
                                            true
                                        }
                                        (true, None) => {
                                            if db
                                                .get_cf_opt(&cf_lookup, lk, &lookup_ropts)
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
                                        (false, _) => false,
                                    };

                                    if !skip {
                                        lookup_misses.fetch_add(1, Ordering::Relaxed);

                                        // Plan all 7 placements via sequential beam search
                                        let mut plan = [0u8; 7];
                                        let mut sim_board = board;
                                        let mut plan_ok = true;

                                        for step in 0..7 {
                                            let piece = perm[step];
                                            // Build bag state for remaining pieces
                                            let mut bag_mask = 0u8;
                                            for j in step..7 {
                                                bag_mask |= u8::from(perm[j]);
                                            }
                                            let bag = TetrisPieceBagState::from(bag_mask);
                                            let mut next_bag = bag;
                                            next_bag.remove(piece);
                                            if next_bag.count() == 0 {
                                                next_bag.fill();
                                            }

                                            game.board = sim_board;
                                            game.set_bag_piece_seeded(next_bag, piece, BASE_SEED);

                                            let state = BeamTetrisState::new(game);
                                            let result = beam_search.search_with_seeds(
                                                state, BASE_SEED, MAX_DEPTH, BEAM_WIDTH,
                                            );

                                            match result {
                                                Some(placement) => {
                                                    game.board = sim_board;
                                                    let r = game.apply_placement(placement);
                                                    if r.is_lost == IsLost::LOST {
                                                        games_lost.fetch_add(1, Ordering::Relaxed);
                                                        plan_ok = false;
                                                        break;
                                                    }
                                                    // Filter: reject boards that violate constraints
                                                    let next_state = BeamTetrisState::new(game);
                                                    if !filter_fn(&next_state) {
                                                        games_lost.fetch_add(1, Ordering::Relaxed);
                                                        plan_ok = false;
                                                        break;
                                                    }
                                                    plan[step] = placement.orientation.index();
                                                    sim_board = game.board;
                                                }
                                                None => {
                                                    games_lost.fetch_add(1, Ordering::Relaxed);
                                                    plan_ok = false;
                                                    break;
                                                }
                                            }
                                        }

                                        if plan_ok {
                                            // Store (board, perm) → [placement; 7]
                                            wb.put_cf(&cf_lookup, &lk, &plan);
                                            lookup_inserts.fetch_add(1, Ordering::Relaxed);

                                            // After completing a bag, seed the resulting
                                            // board with all 5040 new permutations.
                                            // Uses the CF_SEEDED column family for
                                            // exact dedup (no false positives).
                                            // Only seed boards that pass the filter
                                            let seed_state = BeamTetrisState::new(game);
                                            game.board = sim_board; // restore after BeamTetrisState consumed it
                                            let passes_filter = filter_fn(&seed_state);

                                            let board_bytes = sim_board.to_bytes(Major::Col);
                                            let cf_seeded =
                                                db.cf_handle(CF_SEEDED).expect("missing seeded cf");
                                            let already = db
                                                .get_cf(&cf_seeded, &board_bytes)
                                                .ok()
                                                .flatten()
                                                .is_some();
                                            if passes_filter && !already {
                                                // Mark seeded first to prevent races
                                                wb.put_cf(&cf_seeded, &board_bytes, &[1u8]);
                                                for next_perm in 0..NUM_PERMS as u16 {
                                                    let nfk =
                                                        make_frontier_key(sim_board, next_perm);
                                                    let nfv =
                                                        make_frontier_value(sim_board, next_perm);
                                                    wb.put_cf(&cf_frontier, nfk, nfv);
                                                }
                                                frontier_enqueued
                                                    .fetch_add(NUM_PERMS as u64, Ordering::Relaxed);
                                            }
                                        }
                                    }

                                    // Delete from frontier
                                    wb.delete_cf(&cf_frontier, &key);
                                    frontier_deleted.fetch_add(1, Ordering::Relaxed);

                                    Some(key)
                                })
                                .last();

                            db.write_opt(wb, &write_opts)
                                .expect("failed to write batch");

                            if items_processed == 0 {
                                thread::sleep(Duration::from_millis(100));
                            }
                        }
                    })
                    .expect("Failed to spawn worker")
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Create mode
// ---------------------------------------------------------------------------

fn run_create(db_path: &str) {
    tetris_search::set_global_threadpool();
    let start = Instant::now();

    let atlas = AtlasDB::new(db_path).expect("Failed to create atlas");
    atlas.seed_starting_state().expect("Failed to seed");

    let handles = atlas.run_worker_loop(NUM_WORKERS);

    {
        let stop = Arc::clone(&atlas.stop);
        ctrlc::set_handler(move || {
            if !stop.swap(true, Ordering::Release) {
                eprintln!("\nShutting down...");
            }
        })
        .expect("Error setting Ctrl-C handler");
    }

    let csv_file = std::fs::File::create(STATS_CSV_PATH).expect("Failed to create CSV");
    let mut csv = std::io::BufWriter::new(csv_file);
    writeln!(
        csv,
        "t,boards,lookup,frontier,rate,inserts,lost,hits,misses"
    )
    .expect("CSV header");
    csv.flush().expect("flush");

    let mut last_log = Instant::now();
    loop {
        thread::sleep(Duration::from_millis(50));
        if atlas.is_stopped() {
            break;
        }
        if last_log.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let boards = atlas.boards_expanded.load(Ordering::Relaxed);
            let inserts = atlas.lookup_inserts.load(Ordering::Relaxed);
            let lost = atlas.games_lost.load(Ordering::Relaxed);
            let hits = atlas.lookup_hits.load(Ordering::Relaxed);
            let misses = atlas.lookup_misses.load(Ordering::Relaxed);
            let lsize = atlas.lookup_size();
            let fsize = atlas.frontier_size();
            let ssize = atlas.seeded_count();
            let rate = boards as f64 / secs;

            println!(
                "t={secs:.1}s boards={boards} lookup={lsize} frontier={fsize} seeded={ssize} rate={rate:.1}/s inserts={inserts} lost={lost} hits={hits} misses={misses}"
            );
            writeln!(
                csv,
                "{secs:.1},{boards},{lsize},{fsize},{rate:.1},{inserts},{lost},{hits},{misses}"
            )
            .expect("CSV write");
            csv.flush().expect("flush");
            last_log = Instant::now();
            atlas.flush();
        }
    }

    for h in handles {
        let _ = h.join();
    }
    println!("Finished in {:.2}s", start.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------------------
// Explore mode (simplified)
// ---------------------------------------------------------------------------

fn run_explore(db_path: &str) {
    let atlas = AtlasDB::new(db_path).expect("Failed to open atlas");
    let cf_lookup = atlas.cf_handle(CF_LOOKUP);

    println!("Atlas explore mode. Lookup size ≈ {}", atlas.lookup_size());
    println!("Checking empty board coverage...");

    let empty = TetrisBoard::EMPTY_BOARD;
    let mut covered = 0u32;
    for perm_idx in 0..NUM_PERMS as u16 {
        let lk = make_lookup_key(empty, perm_idx);
        if atlas.db.get_cf(&cf_lookup, lk).ok().flatten().is_some() {
            covered += 1;
        }
    }
    println!(
        "Empty board: {covered}/{NUM_PERMS} permutations covered ({:.1}%)",
        covered as f64 / NUM_PERMS as f64 * 100.0
    );

    // Show a sample plan
    if covered > 0 {
        for perm_idx in 0..NUM_PERMS as u16 {
            let lk = make_lookup_key(empty, perm_idx);
            if let Ok(Some(value)) = atlas.db.get_cf(&cf_lookup, lk) {
                let perm = &ALL_PERMUTATIONS[perm_idx as usize];
                println!("\nSample plan for perm {perm_idx}:");
                print!("  Pieces: ");
                for p in perm {
                    print!("{p} ");
                }
                println!();
                print!("  Placements: ");
                for (i, &orient_idx) in value.iter().enumerate() {
                    if i < 7 {
                        print!("{orient_idx} ");
                    }
                }
                println!();
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Create => run_create(&cli.db_path),
        Mode::Explore => run_explore(&cli.db_path),
    }
}
