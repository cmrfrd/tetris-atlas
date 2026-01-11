use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry as DashEntry;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tetris_atlas::beam_search::{BeamSearch, BeamTetrisState};
use tetris_atlas::set_global_threadpool;
use tetris_atlas::tetris::{
    IsLost, PlacementResult, TetrisBoard, TetrisGame, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement,
};

/*
python3 -c "import matplotlib.pyplot as plt; import numpy as np; data = [line.strip().split(',') for line in open('/Users/cmrfrd/Desktop/repos/cmrfrd/tetris-atlas/beam_search_output.csv') if line.strip() and ',' in line]; data = [(int(row[0]), int(row[1])) for row in data if len(row) >= 2]; x, y = zip(*data); x, y = np.array(x), np.array(y); ratios = y / x; diffs_y = np.diff(y); diffs_x = np.diff(x); rates = diffs_y / diffs_x; fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12)); ax1.plot(x, ratios, 'b-', linewidth=2); ax1.set_xlabel('Number of Pieces Placed'); ax1.set_ylabel('Ratio (Unique Boards / Pieces)'); ax1.set_title('Board Uniqueness Ratio Over Pieces Placed'); ax1.grid(True, alpha=0.3); ax1.text(0.95, 0.05, f'{ratios[-1]:.4f}', transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), ha='right'); ax2.plot(x, y, 'g-', linewidth=3, label='Unique Boards'); ax2.plot(x, x, 'r-', linewidth=3, label='Total Pieces'); ax2.set_xlabel('Number of Pieces Placed'); ax2.set_ylabel('Count'); ax2.set_title('Unique Boards vs Total Pieces Placed'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.text(0.95, 0.05, f'{y[-1]:,}', transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), ha='right'); ax3.plot(x[1:], rates, 'm-', linewidth=1, marker='o', markersize=2); ax3.set_xlabel('Number of Pieces Placed'); ax3.set_ylabel('Discovery Rate (New Unique / Pieces per Interval)'); ax3.set_title('Rate of New Unique Board Discovery'); ax3.grid(True, alpha=0.3); ax3.text(0.95, 0.05, f'{rates[-1]:.4f}', transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8), ha='right'); plt.tight_layout(); plt.show()"
 */

const BEAM_WIDTH: usize = 4;
const MAX_DEPTH: usize = 8;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS; // >= max placements per piece (usually ~40)
const CSV_PATH: &str = "beam_lookup_progress.csv";
const LOOKUP_BIN_PATH: &str = "beam_lookup_table.bin";
const LOG_EVERY_SECS: u64 = 1;
const IDLE_GRACE_SECS: u64 = 3;
const NUM_WORKERS: usize = 8;
const SAMPLES: usize = 128;
const BATCH_SIZE: usize = 64;

type Key = (TetrisBoard, TetrisPieceBagState);
type Row = [Option<TetrisPiecePlacement>; 7];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct LookupEntrySer {
    board: [u32; 10],
    bag: u8,
    placements: [Option<u8>; 7],
}

pub struct TetrisAtlas {
    pub frontier: SegQueue<Key>,
    pub lookup_table: DashMap<Key, Row>,

    stop: AtomicBool,
    in_flight: AtomicUsize,
    expanded: AtomicU64,
    discovered: AtomicU64,
}

impl TetrisAtlas {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            frontier: SegQueue::new(),
            lookup_table: DashMap::with_shard_amount(32),
            stop: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            expanded: AtomicU64::new(0),
            discovered: AtomicU64::new(0),
        })
    }

    pub fn seed_starting_state(&self) {
        let empty_board = TetrisBoard::new();
        let bag = TetrisPieceBagState::new();
        let key = (empty_board, bag);
        // mark discovered via table presence
        self.lookup_table.insert(key, [None; 7]);
        self.frontier.push(key);
        self.discovered.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub fn is_idle(&self) -> bool {
        self.frontier.is_empty() && (self.in_flight.load(Ordering::Relaxed) == 0)
    }

    pub fn stats(&self) -> (u64, u64, usize, usize, bool) {
        let expanded = self.expanded.load(Ordering::Relaxed);
        let discovered = self.discovered.load(Ordering::Relaxed);
        let in_flight = self.in_flight.load(Ordering::Relaxed);
        let table_len = self.lookup_table.len();
        let frontier_empty = self.frontier.is_empty();
        (expanded, discovered, in_flight, table_len, frontier_empty)
    }

    pub fn launch_worker(self: &Arc<Self>, worker_id: u64) -> JoinHandle<()> {
        let atlas = Arc::clone(self);
        thread::spawn(move || {
            let mut search = BeamSearch::<BeamTetrisState, BEAM_WIDTH, MAX_DEPTH, MAX_MOVES>::new();
            let mut local_ctr: u64 = 0;

            while !atlas.stop.load(Ordering::Relaxed) {
                // Pop a batch of keys to reduce contention on the shared queue/map.
                let mut batch: Vec<Key> = Vec::with_capacity(BATCH_SIZE);
                let Some(first) = atlas.frontier.pop() else {
                    // Avoid hot spinning when the queue is empty.
                    thread::sleep(Duration::from_millis(1));
                    continue;
                };
                batch.push(first);
                while batch.len() < BATCH_SIZE {
                    let Some(k) = atlas.frontier.pop() else {
                        break;
                    };
                    batch.push(k);
                }

                let batch_len = batch.len();
                atlas.in_flight.fetch_add(batch_len, Ordering::Relaxed);
                atlas
                    .expanded
                    .fetch_add(batch_len as u64, Ordering::Relaxed);

                for (board, bag_state) in batch {
                    let key: Key = (board, bag_state);
                    atlas.lookup_table.entry(key).or_insert([None; 7]);

                    let row_snapshot: Row = *atlas
                        .lookup_table
                        .get(&key)
                        .expect("lookup_table entry just inserted must exist");

                    let mut made_progress = false;
                    let mut missing_any = false;
                    let mut updates: Vec<(usize, TetrisPiecePlacement)> = Vec::new();
                    let mut successors: Vec<Key> = Vec::new();

                    for piece in TetrisPiece::all() {
                        if !bag_state.contains(piece) {
                            continue;
                        }
                        let p_idx = piece.index() as usize;
                        if row_snapshot[p_idx].is_some() {
                            continue;
                        }

                        let mut bag_after_draw = bag_state;
                        bag_after_draw.remove(piece);
                        if bag_after_draw.count() == 0 {
                            bag_after_draw.fill();
                        }

                        // Unique-ish seed stream per worker + per-task.
                        let base_seed = (worker_id << 48) ^ local_ctr ^ rand::rng().next_u64();
                        local_ctr = local_ctr.wrapping_add(1);

                        let mut agg: HashMap<TetrisPiecePlacement, (f32, u32)> = HashMap::new();
                        for s in 0..SAMPLES {
                            let game = TetrisGame::new_with_board_bag_piece_seeded(
                                board,
                                bag_after_draw,
                                piece,
                                base_seed.wrapping_add(s as u64),
                            );
                            let Some(scored) = search.search_first_action_with_state(
                                BeamTetrisState::new(game),
                                MAX_DEPTH,
                            ) else {
                                continue;
                            };
                            let Some(first) = scored.first_action else {
                                continue;
                            };
                            let e = agg.entry(first).or_insert((0.0, 0));
                            e.0 += scored.score;
                            e.1 += 1;
                        }

                        let Some(best_placement) = agg
                            .into_iter()
                            .max_by(|a, b| {
                                let (sa, ca) = a.1;
                                let (sb, cb) = b.1;
                                // Prefer the placement that wins most often across rollouts
                                // (majority vote). Break ties by mean score.
                                ca.cmp(&cb)
                                    .then_with(|| (sa / ca as f32).total_cmp(&(sb / cb as f32)))
                            })
                            .map(|(p, _)| p)
                        else {
                            missing_any = true;
                            continue;
                        };

                        let mut next_board = board;
                        let PlacementResult { is_lost, .. } =
                            next_board.apply_piece_placement(best_placement);
                        if is_lost == IsLost::LOST {
                            missing_any = true;
                            continue;
                        }

                        made_progress = true;
                        updates.push((p_idx, best_placement));
                        successors.push((next_board, bag_after_draw));
                    }

                    if !updates.is_empty() {
                        if let Some(mut row) = atlas.lookup_table.get_mut(&key) {
                            for (idx, placement) in updates {
                                row[idx] = Some(placement);
                            }
                        }
                    }

                    for succ in successors {
                        match atlas.lookup_table.entry(succ) {
                            DashEntry::Vacant(v) => {
                                v.insert([None; 7]);
                                atlas.discovered.fetch_add(1, Ordering::Relaxed);
                                atlas.frontier.push(succ);
                            }
                            DashEntry::Occupied(_) => {}
                        }
                    }

                    if missing_any && made_progress {
                        atlas.frontier.push(key);
                    }
                }

                atlas.in_flight.fetch_sub(batch_len, Ordering::Relaxed);
            }
        })
    }
}

/// Entry point for the `tetris_beam` binary.
fn main() {
    run_tetris_beam();
}

/// Run a beam-search-driven Tetris game, continually planning and applying placements.
///
/// This is intentionally argument-free: tweak the consts below while iterating.
pub fn run_tetris_beam() {
    set_global_threadpool();

    let start = Instant::now();

    let file = File::create(CSV_PATH).expect("Failed to create CSV output file");
    let mut csv = BufWriter::new(file);
    writeln!(
        csv,
        "secs,expanded,discovered,in_flight,lookup_table_size,frontier_empty,expanded_per_sec"
    )
    .expect("Failed to write CSV header");

    let atlas = TetrisAtlas::new();
    atlas.seed_starting_state();

    let mut handles: Vec<JoinHandle<()>> = Vec::new();
    for w in 0..NUM_WORKERS {
        handles.push(atlas.launch_worker(w as u64));
    }

    let mut last_log = Instant::now();
    let mut idle_since: Option<Instant> = None;

    loop {
        thread::sleep(Duration::from_millis(50));

        if last_log.elapsed() >= Duration::from_secs(LOG_EVERY_SECS) {
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let (expanded, discovered, in_flight, table_len, frontier_empty) = atlas.stats();
            let rate = expanded as f64 / secs;

            println!(
                "secs={secs:.1} expanded={expanded} discovered={discovered} in_flight={in_flight} table={table_len} frontier_empty={frontier_empty} expanded_per_sec={rate:.1}"
            );

            writeln!(
                csv,
                "{secs},{expanded},{discovered},{in_flight},{table_len},{frontier_empty},{rate}"
            )
            .expect("Failed to write CSV row");
            csv.flush().expect("Failed to flush CSV output");

            last_log = Instant::now();
        }

        if atlas.is_idle() {
            let t = idle_since.get_or_insert_with(Instant::now);
            if t.elapsed() >= Duration::from_secs(IDLE_GRACE_SECS) {
                break;
            }
        } else {
            idle_since = None;
        }
    }

    atlas.stop();
    for h in handles {
        let _ = h.join();
    }

    // Serialize the lookup table to a compact bincode blob.
    // Format: Vec<LookupEntrySer>, each entry is (board limbs, bag mask, placement indices).
    let snapshot: Vec<LookupEntrySer> = atlas
        .lookup_table
        .iter()
        .map(|kv| {
            let (board, bag_state) = *kv.key();
            let row = *kv.value();
            LookupEntrySer {
                board: board.as_limbs(),
                bag: u8::from(bag_state),
                placements: row.map(|opt| opt.map(|p| p.index())),
            }
        })
        .collect();

    let bytes = bincode::serialize(&snapshot).expect("Failed to bincode serialize lookup table");
    std::fs::write(LOOKUP_BIN_PATH, bytes).expect("Failed to write lookup table bincode file");

    csv.flush().expect("Failed to flush CSV output");
    println!("Finished in {:.2} seconds", start.elapsed().as_secs_f64());
}
