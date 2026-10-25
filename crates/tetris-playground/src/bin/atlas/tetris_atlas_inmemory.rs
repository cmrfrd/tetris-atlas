#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Builder - In-Memory Implementation
//!
//! This binary builds a *provably surviving* atlas with one **continuous, fully-parallel
//! AND-OR engine**: expansion and backward death-propagation run as a single interleaved
//! stream of tasks across all workers, then a parallel commit at quiescence. Rather than
//! greedily committing one beam move per state (which can bake in moves that later top out),
//! it grows the AND-OR state graph and propagates "death" backward until a closed safe set
//! remains.
//!
//! ## Why one continuous engine is sound
//!
//! SAFE is a *greatest* fixpoint (a "safe" label can be retracted), so the committed policy is
//! only finalized at global quiescence. But DEATH is the dual *least* fixpoint and is
//! **monotone**: once a state is proven dead it is dead forever. Monotone work can run
//! concurrently with expansion without unsound retraction, so expansion (`process_expand`) and
//! death (`process_die`) interleave as one task stream; only the commit waits for quiescence.
//!
//! ## Engine
//!
//! - **Workers** (`worker_loop`, hosted on a `rayon::scope` pool) pull two task kinds,
//!   **death-first**: drain the `SegQueue` death tasks, else pop the next state from the
//!   priority frontier and expand it (skipping any already proven dead — the pruning win).
//! - **`process_expand`** enumerates a state's in-carrier placements, publishes its forward
//!   edges + per-piece live counts under the node lock, then appends reverse edges to each
//!   successor (publish-under-lock resolves the append-vs-die race), and seeds death if any
//!   forced piece has zero surviving placements.
//! - **`process_die`** marks a state dead (monotone) and decrements each predecessor's live
//!   count, dying parents whose last placement for a forced piece is gone — propagating backward.
//! - **Termination** is an `in_flight` counter (incremented before every push, decremented
//!   after every task); quiescence is `in_flight == 0` with both queues empty.
//! - **Commit** (`commit_surviving_policy`, parallel via `par_iter`) writes the surviving
//!   `(board, bag, piece) -> orientation` policy once DEAD is final.
//!
//! The committed lookup is a *closed safe set*: from any committed state, every forced piece
//! has a move to another committed state, i.e. provably infinite survival within the carrier.

use clap::{Parser, Subcommand};
use crossbeam::queue::SegQueue;
use crossbeam::utils::Backoff;
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, Major, StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPieceOrientation, TetrisPiecePlacement,
};

// --- Carrier boundary ---
// A placement is only a valid move if the resulting board stays inside this carrier.
// This bounds the reachable state graph so the death-propagation fixpoint terminates over a
// finite set. Empty board (holes 0, height 0, roughness 0) is inside.
//
// `MAX_CARRIER_ROUGHNESS` is the primary tuning knob: surface roughness (total bumpiness =
// sum of adjacent column-height differences) is the dimension that blows up the state count.
// Capping it keeps the universe small enough to enumerate while still admitting the jaggedness
// that survival needs. Lower it to make the run finish; raise it toward survivability.
const MAX_CARRIER_HOLES: u32 = 1;
const MAX_CARRIER_HEIGHT: u32 = 4;
const MAX_CARRIER_ROUGHNESS: u32 = 16;

// --- Optional memory safety cap ---
// The frontier (`SkipMap`) and node map grow unboundedly during expansion; a loose carrier can
// intern hundreds of millions of `(board, bag)` states and OOM-kill the process before the death
// fixpoint ever reaches the seed. `create --max-states N` bounds the interned-state count: once it
// is hit the logger requests a graceful early stop (the same conservative finalize path as Ctrl+C),
// so the run yields a sound conservative result and a saved atlas instead of being SIGKILLed with
// nothing. OMITTED = no cap (unbounded, may OOM). Empirically ~19.7M states cost ~5.5GB, so e.g.
// `--max-states 20000000` keeps a run under ~6GB.

/// Total surface roughness (bumpiness): sum over adjacent columns of |height[i] - height[i+1]|.
#[inline]
const fn surface_roughness(board: &TetrisBoard) -> u32 {
    let heights = board.heights();
    let mut rough = 0u32;
    let mut i = 0;
    while i + 1 < StandardTetris::COLS {
        rough += heights[i].abs_diff(heights[i + 1]);
        i += 1;
    }
    rough
}

#[inline]
const fn carrier_contains(board: &TetrisBoard) -> bool {
    board.total_holes() <= MAX_CARRIER_HOLES
        && board.height() <= MAX_CARRIER_HEIGHT
        && surface_roughness(board) <= MAX_CARRIER_ROUGHNESS
}

// --- Concurrency / hashing ---
// The concurrent maps hash 40-byte boards on the hottest path (every interned successor and
// every lookup), so we use FxHash instead of DashMap's default SipHash, and a generous shard
// count to cut lock contention across workers. Shard count must be a power of two.
type FxDashMap<K, V> = DashMap<K, V, FxBuildHasher>;
const DASHMAP_SHARDS: usize = 256;

#[inline]
fn new_fx_dashmap<K: std::hash::Hash + Eq, V>() -> FxDashMap<K, V> {
    DashMap::with_hasher_and_shard_amount(FxBuildHasher, DASHMAP_SHARDS)
}

// --- Persistence ---
const LOG_EVERY_SECS: u64 = 3;
const SAVE_FILE: &str = "tetris_atlas_inmemory.bin";
const CSV_FILE: &str = "tetris_atlas_inmemory.csv";

#[derive(Parser)]
#[command(name = "tetris-atlas-inmemory")]
#[command(about = "Tetris Atlas (In-Memory) - Build and explore optimal Tetris lookup tables")]
struct Cli {
    #[command(subcommand)]
    mode: Mode,
    /// Directory path where files will be saved (default: current directory)
    #[arg(short, long, default_value = ".")]
    path: String,
}

#[derive(Subcommand)]
enum Mode {
    /// Create a new atlas by exhaustively exploring the state space
    Create {
        /// Optional cap on interned states: request a graceful early stop (conservative finalize)
        /// once the graph reaches this many `(board, bag)` states, instead of risking an OOM.
        /// Omit for no cap (unbounded). ~19.7M states ≈ ~5.5GB, so e.g. `--max-states 20000000`.
        #[arg(long)]
        max_states: Option<usize>,
    },
    /// Explore an existing atlas interactively
    Explore,
    /// Independently cross-check whether the empty board is forced to lose (a separate referee).
    /// The carrier defaults to the compile-time constants but can be overridden per-run so you
    /// can sweep tightness without recompiling.
    Verify {
        /// Max number of pieces to look ahead.
        #[arg(short, long, default_value_t = 14)]
        depth: u32,
        /// Carrier hole cap.
        #[arg(long, default_value_t = MAX_CARRIER_HOLES)]
        holes: u32,
        /// Carrier height cap.
        #[arg(long, default_value_t = MAX_CARRIER_HEIGHT)]
        height: u32,
        /// Carrier roughness cap.
        #[arg(long, default_value_t = MAX_CARRIER_ROUGHNESS)]
        rough: u32,
    },
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

/// Frontier key that orders by height (lowest first), then by interned `StateId` for a unique,
/// deterministic total order. The `StateId` is unique per state, so every push inserts (no
/// dedup needed — the graph interner already guarantees each state is enqueued once).
#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    priority: u32,
    id: u32,
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority frontier using SkipMap for concurrent sorted access.
/// Lower heights are processed first for efficient state space exploration.
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

    /// Push an interned `(board, bag)` state. Always inserts (the `id` makes the key unique).
    fn push(&self, board: TetrisBoard, bag: TetrisPieceBagState, id: u32) {
        let key = FrontierKey {
            priority: board.count(),
            id,
        };
        self.map.insert(key, FrontierValue::new(board, bag));
        self.enqueued_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop the lowest-height entry.
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

/// Dense identifier for an interned `(board, bag)` AND-OR state.
type StateId = u32;

/// One reverse edge: `parent` reached the child state by placing piece `piece_idx`.
#[derive(Clone, Copy)]
struct PredecessorRef {
    parent: StateId,
    piece_idx: u8,
}

/// One AND-OR state node. ALL mutation of a node happens through the `Graph.nodes` DashMap
/// shard lock for that id, so the predecessor-append vs death race serializes per node.
struct Node {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    /// Surviving-placement edges per forced piece, written once at expansion.
    forward: [Vec<(TetrisPieceOrientation, StateId)>; 7],
    /// Reverse edges: parents that reach this node. Appended concurrently; drained on death.
    preds: Vec<PredecessorRef>,
    /// Live (non-dead) placement count per forced piece. Only forced pieces are meaningful;
    /// published once at expansion, then only ever decremented.
    live: [i32; 7],
    /// True once `forward`/`live` are fully enumerated.
    expanded: bool,
    /// Monotone: set once in `process_die`, never cleared.
    dead: bool,
}

impl Node {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self {
            board,
            bag,
            forward: Default::default(),
            preds: Vec::new(),
            live: [0; 7],
            expanded: false,
            dead: false,
        }
    }
}

/// The continuous AND-OR engine's shared graph: an interner + node map + a lock-free death
/// queue + a global in-flight counter for quiescence. Expansion and death propagation run as
/// one interleaved task stream across all workers (see `process_expand` / `process_die`).
struct Graph {
    state_to_id: FxDashMap<(TetrisBoard, TetrisPieceBagState), StateId>,
    nodes: FxDashMap<StateId, Node>,
    /// High-priority death tasks (drained before expansion by every worker).
    death_q: SegQueue<StateId>,
    next_id: AtomicU32,
    edge_count: AtomicU64,
    dead_count: AtomicU64,
    expanded_count: AtomicU64,
    /// Outstanding tasks (queued expansions + queued deaths + in-progress). Quiescence == 0.
    in_flight: AtomicU64,
    /// Set the instant the seed (empty board, `StateId 0`) is proven dead. Because death is
    /// monotone this is final — the run has provably failed, so workers stop themselves.
    seed_dead: AtomicBool,
}

impl Graph {
    fn new() -> Self {
        Self {
            state_to_id: new_fx_dashmap(),
            nodes: new_fx_dashmap(),
            death_q: SegQueue::new(),
            next_id: AtomicU32::new(0),
            edge_count: AtomicU64::new(0),
            dead_count: AtomicU64::new(0),
            expanded_count: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            seed_dead: AtomicBool::new(false),
        }
    }

    /// Intern a `(board, bag)` state, returning its id and whether it was newly created.
    /// The DashMap entry lock makes id assignment + node creation race-free across workers.
    fn intern(&self, board: TetrisBoard, bag: TetrisPieceBagState) -> (StateId, bool) {
        match self.state_to_id.entry((board, bag)) {
            Entry::Occupied(o) => (*o.get(), false),
            Entry::Vacant(v) => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                // Insert the `Node` BEFORE publishing the id into `state_to_id`. The `VacantEntry`
                // `v` is still alive here, so the entry's shard lock is held across `nodes.insert`;
                // a concurrent intern of the same key blocks until we publish via `v.insert(id)`,
                // by which point the node already exists. This guarantees a `StateId` is observable
                // only AFTER its `Node` — closing the window where `nodes.get_mut(id)` returns
                // `None` for a freshly-interned id and silently drops a death decrement.
                self.nodes.insert(id, Node::new(board, bag));
                v.insert(id);
                (id, true)
            }
        }
    }

    fn is_dead(&self, id: StateId) -> bool {
        self.nodes.get(&id).map(|n| n.dead).unwrap_or(false)
    }

    /// Enqueue a death task (increments in-flight *before* the push, so quiescence is sound).
    fn enqueue_death(&self, id: StateId) {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.death_q.push(id);
    }

    /// THE single death condition: a not-yet-dead, fully-expanded node is dead the moment some
    /// forced piece in its bag has no surviving placement left (`live == 0`). Used as both the
    /// base case (right after expansion — catches dead-ends that were born stuck) and the
    /// inductive step (after a decrement — catches deaths that propagate from a dying successor).
    fn dies_now(node: &Node) -> bool {
        !node.dead
            && node.expanded
            && node
                .bag
                .iter_next_states()
                .any(|(piece, _)| node.live[piece.index() as usize] == 0)
    }

    /// Inductive step: a successor died (or a dead-at-append compensation), so decrement `id`'s
    /// live count for that piece; if that makes `id` itself dead, enqueue it.
    /// Locks exactly one node — never held alongside any other node lock.
    fn decrement_live(&self, id: StateId, piece_idx: usize) {
        let trigger = {
            let Some(mut node) = self.nodes.get_mut(&id) else {
                return;
            };
            if node.dead {
                false
            } else {
                node.live[piece_idx] -= 1;
                // Tripwire: a live count must never go below zero. If it does, some edge was
                // decremented twice — a counting bug that could wrongly declare a position lost.
                debug_assert!(
                    node.live[piece_idx] >= 0,
                    "live[{piece_idx}] went negative on state {id} — double-decrement bug"
                );
                Self::dies_now(node.value())
            }
        };
        if trigger {
            self.enqueue_death(id);
        }
    }

    /// Base case: evaluate `id`'s death right after it is fully expanded. This is where the death
    /// wave *originates* — a state born with a forced piece that has no placement at all (so no
    /// successor will ever die to trigger `decrement_live`) is detected here.
    fn check_death(&self, id: StateId) {
        let should_die = self
            .nodes
            .get(&id)
            .map(|node| Self::dies_now(node.value()))
            .unwrap_or(false);
        if should_die {
            self.enqueue_death(id);
        }
    }

    fn state_count(&self) -> usize {
        self.next_id.load(Ordering::Relaxed) as usize
    }

    fn edges(&self) -> u64 {
        self.edge_count.load(Ordering::Relaxed)
    }
}

/// Shared state: the committed lookup table, the expansion frontier, and control flags.
#[derive(Clone)]
struct SharedState {
    /// Committed `(board, bag, piece) -> orientation` policy (written in the commit pass).
    lookup: Arc<FxDashMap<(TetrisBoard, TetrisPieceBagState, TetrisPiece), TetrisPieceOrientation>>,
    /// Priority frontier: height-indexed expansion queue for `(board, bag)` states.
    frontier: Arc<PriorityFrontier>,
    /// Counter for frontier enqueued (used by PriorityFrontier).
    frontier_enqueued: Arc<AtomicU64>,
    /// Stop the logger thread.
    shutdown: Arc<AtomicBool>,
    /// Ctrl+C: end expansion early (yields a conservative safe set).
    shutdown_requested: Arc<AtomicBool>,
}

impl SharedState {
    fn new() -> Self {
        let frontier_enqueued = Arc::new(AtomicU64::new(0));
        Self {
            lookup: Arc::new(new_fx_dashmap()),
            frontier: Arc::new(PriorityFrontier::new(frontier_enqueued.clone())),
            frontier_enqueued,
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    fn lookup_size(&self) -> usize {
        self.lookup.len()
    }

    /// Save the committed lookup table to disk.
    /// Format: `[u64 count]` then count × `[40B board][1B bag][1B piece][1B orientation]`.
    fn save_to_disk(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let lookup_count = self.lookup.len();
        println!("Saving {lookup_count} committed entries to {path}...");

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&(lookup_count as u64).to_le_bytes())?;
        for entry in self.lookup.iter() {
            let ((board, bag, piece), orientation) = (*entry.key(), *entry.value());
            writer.write_all(&board.to_bytes(Major::Col))?;
            writer.write_all(&[bag.into()])?;
            writer.write_all(&[piece.into()])?;
            writer.write_all(&[orientation.index()])?;
        }
        writer.flush()?;

        println!(
            "Saved {lookup_count} entries in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Load the committed lookup table from disk (if present). Returns the entry count.
    fn load_from_disk(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        if !Path::new(path).exists() {
            return Ok(0);
        }

        let start = Instant::now();
        println!("Loading committed atlas from {path}...");

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut count_buf = [0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let lookup_count = u64::from_le_bytes(count_buf) as usize;

        for _ in 0..lookup_count {
            let mut board_bytes = [0u8; 40];
            reader.read_exact(&mut board_bytes)?;
            let mut bag_byte = [0u8; 1];
            reader.read_exact(&mut bag_byte)?;
            let mut piece_byte = [0u8; 1];
            reader.read_exact(&mut piece_byte)?;
            let mut orientation_byte = [0u8; 1];
            reader.read_exact(&mut orientation_byte)?;

            let board = TetrisBoard::from_bytes(board_bytes, Major::Col);
            let bag = TetrisPieceBagState::from(bag_byte[0]);
            let piece = TetrisPiece::from(piece_byte[0]);
            let orientation = TetrisPieceOrientation::from_index(orientation_byte[0]);
            self.lookup.insert((board, bag, piece), orientation);
        }

        println!(
            "Loaded {lookup_count} entries in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Ok(lookup_count)
    }
}

/// Summary of the merged engine's result.
#[derive(Default)]
struct SolveSummary {
    total_states: usize,
    dead_states: usize,
    safe_states: usize,
    committed_moves: usize,
}

/// Final verdict of a `create` run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunVerdict {
    /// The seed (empty board) is proven dead — no atlas can exist from the start.
    Failed,
    /// The seed is alive and the committed policy is a non-empty closed safe set.
    Success,
    /// The seed is alive but the committed policy is empty or not closed.
    Inconclusive,
}

/// Decide a run's verdict from its final state. A DEAD seed is ALWAYS `Failed`, even if a
/// disconnected safe cycle committed moves — the atlas must cover the START (empty board). This
/// specifically guards the finalize path: conservative killing of unexpanded survivors can make
/// the seed dead *after* the initial `is_dead(0)` check, so the SUCCESS gate must re-consult it
/// rather than trusting `committed_moves > 0 && violations == 0` alone.
fn decide_verdict(
    seed_dead: bool,
    committed_moves: usize,
    closure_violations: usize,
) -> RunVerdict {
    if seed_dead {
        RunVerdict::Failed
    } else if committed_moves > 0 && closure_violations == 0 {
        RunVerdict::Success
    } else {
        RunVerdict::Inconclusive
    }
}

/// Whether the engine should stop expanding early because the interned-state count reached the
/// optional memory cap. `None` means no cap (never stop for this reason); `Some(c)` is an inclusive
/// ceiling (stop at or above `c`).
fn should_request_stop(total_states: usize, cap: Option<usize>) -> bool {
    matches!(cap, Some(c) if total_states >= c)
}

/// States actually expanded and not yet proven dead — a BOUNDED "alive so far" count. Unlike
/// `states_total - dead` (which counts unexamined frontier states as "alive"), this never exceeds
/// the number of states the engine has actually evaluated for death.
fn evaluated_alive(expanded: u64, dead: u64) -> u64 {
    expanded.saturating_sub(dead)
}

/// Expand task: enumerate one state's edges, seed any immediate death, and append reverse
/// edges to successors. Interleaves with `process_die` across all workers.
///
/// Concurrency: `live` is computed locally and **published once** (so death decrements only
/// ever *reduce* it — never racing an increment). Predecessor edges are appended **after**
/// publish, under each successor's node lock, with a compensating decrement if the successor
/// already died — the publish-under-lock that resolves the append-vs-die race.
fn process_expand(
    graph: &Graph,
    frontier: &PriorityFrontier,
    board: TetrisBoard,
    bag: TetrisPieceBagState,
) {
    let (id, _) = graph.intern(board, bag);

    // Death-first pruning: never expand a state already proven dead.
    if graph.is_dead(id) {
        return;
    }

    let mut edges: [Vec<(TetrisPieceOrientation, StateId)>; 7] = Default::default();
    let mut live: [i32; 7] = [0; 7];
    let mut pending: Vec<(StateId, u8)> = Vec::new(); // (succ, piece_idx) reverse edges to add
    let mut new_succs: Vec<(TetrisBoard, TetrisPieceBagState, StateId)> = Vec::new();

    for (piece, next_bag) in bag.iter_next_states() {
        let piece_idx = piece.index() as usize;
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next_board = board;
            let result = next_board.apply_piece_placement(placement);
            // Drop moves that top out or leave the carrier (keeps the graph closed + finite).
            if result.is_lost == IsLost::LOST || !carrier_contains(&next_board) {
                continue;
            }
            let (succ_id, is_new) = graph.intern(next_board, next_bag);
            edges[piece_idx].push((placement.orientation, succ_id));
            live[piece_idx] += 1;
            pending.push((succ_id, piece_idx as u8));
            graph.edge_count.fetch_add(1, Ordering::Relaxed);
            if is_new {
                new_succs.push((next_board, next_bag, succ_id));
            }
        }
    }

    // Publish forward edges + live counts + `expanded` under this node's lock (single point;
    // after this, concurrent successor deaths may only decrement `live`).
    if let Some(mut node) = graph.nodes.get_mut(&id) {
        node.forward = edges;
        node.live = live;
        node.expanded = true;
    }
    graph.expanded_count.fetch_add(1, Ordering::Relaxed);

    // Enqueue newly discovered successors for expansion (in-flight first, then push).
    for (nb, nbag, sid) in new_succs {
        graph.in_flight.fetch_add(1, Ordering::SeqCst);
        frontier.push(nb, nbag, sid);
    }

    // Append reverse edges; compensate for successors that are already dead at append time.
    for (succ_id, piece_idx) in pending {
        let succ_dead = {
            let Some(mut snode) = graph.nodes.get_mut(&succ_id) else {
                continue;
            };
            if snode.dead {
                true
            } else {
                snode.preds.push(PredecessorRef {
                    parent: id,
                    piece_idx,
                });
                false
            }
        };
        if succ_dead {
            graph.decrement_live(id, piece_idx as usize);
        }
    }

    // Base-case death check: any forced piece left with zero surviving placements kills this
    // state right now (the origin of the death wave for genuinely-stuck states).
    graph.check_death(id);
}

/// Die task: mark a state dead (monotone, idempotent) and propagate the loss backward to its
/// predecessors. Holds exactly one node lock at a time (snapshot preds, release, then touch
/// each parent), so there is no lock-order deadlock.
fn process_die(graph: &Graph, id: StateId) {
    let preds = {
        let Some(mut node) = graph.nodes.get_mut(&id) else {
            return;
        };
        if node.dead {
            return; // already dead — idempotent
        }
        node.dead = true;
        std::mem::take(&mut node.preds)
    };
    graph.dead_count.fetch_add(1, Ordering::Relaxed);

    // The seed (empty board) is `StateId 0`. Its death is the definitive "run failed" signal —
    // monotone, so it never reverts; the search can stop the instant it proves this.
    if id == 0 {
        graph.seed_dead.store(true, Ordering::SeqCst);
    }

    for PredecessorRef { parent, piece_idx } in preds {
        graph.decrement_live(parent, piece_idx as usize);
    }
}

/// Final commit (run once at quiescence, in parallel): for every safe state and forced piece,
/// commit a surviving placement into `lookup`, choosing the most optimal successor (fewest
/// holes, then lowest height, then fewest cells).
///
/// To avoid nested DashMap reads during the parallel pass (which would be a deadlock hazard if
/// any writer existed), the dead flags and per-node board heuristic are snapshotted into plain
/// `Vec`s first; the parallel pass then holds at most one node lock at a time.
fn commit_surviving_policy(
    graph: &Graph,
    lookup: &FxDashMap<(TetrisBoard, TetrisPieceBagState, TetrisPiece), TetrisPieceOrientation>,
) -> SolveSummary {
    let n = graph.state_count();

    // Snapshot: dead flag + (holes, height, cells) per state id — no locks held during commit.
    let mut dead = vec![false; n];
    let mut meta = vec![(u32::MAX, u32::MAX, u32::MAX); n];
    for e in graph.nodes.iter() {
        let idx = *e.key() as usize;
        let node = e.value();
        dead[idx] = node.dead;
        meta[idx] = (
            node.board.total_holes(),
            node.board.height(),
            node.board.count(),
        );
    }
    let dead_states = dead.iter().filter(|&&d| d).count();
    let committed = AtomicUsize::new(0);

    let safe_ids: Vec<StateId> = (0..n as StateId).filter(|&id| !dead[id as usize]).collect();
    safe_ids.par_iter().for_each(|&id| {
        let Some(node) = graph.nodes.get(&id) else {
            return;
        };
        let board = node.board;
        let bag = node.bag;
        for (piece, _next) in bag.iter_next_states() {
            let piece_idx = piece.index() as usize;
            // Pick a surviving successor edge by the heuristic — reads only the `meta`/`dead`
            // snapshots, so there is no second DashMap lock acquired while `node` is held.
            let best = node.forward[piece_idx]
                .iter()
                .filter(|(_, succ)| !dead[*succ as usize])
                .min_by_key(|(_, succ)| meta[*succ as usize]);
            if let Some((orientation, _succ)) = best {
                lookup.insert((board, bag, piece), *orientation);
                committed.fetch_add(1, Ordering::Relaxed);
            } else {
                debug_assert!(
                    false,
                    "safe state {id} has no surviving placement for {piece:?}"
                );
            }
        }
    });

    SolveSummary {
        total_states: n,
        dead_states,
        safe_states: n - dead_states,
        committed_moves: committed.load(Ordering::Relaxed),
    }
}

/// Logger context for the continuous engine's live statistics.
struct LoggerContext {
    graph: Arc<Graph>,
    frontier: Arc<PriorityFrontier>,
    shutdown: Arc<AtomicBool>,
    /// Set by the logger to request a graceful early stop when the state cap is hit.
    shutdown_requested: Arc<AtomicBool>,
    /// Optional interned-state cap (`None` = unbounded); enforced by the logger.
    state_cap: Option<usize>,
    start: Instant,
}

/// CSV writer for the continuous engine's statistics.
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
                "timestamp_secs,expanded,states_total,edges,dead,alive,death_sat_pct,seed_dead,\
                frontier_size,death_q,in_flight,expand_rate"
            )?;
        }

        Ok(Self { file: writer })
    }

    fn log(&mut self, ctx: &LoggerContext) -> std::io::Result<()> {
        let secs = ctx.start.elapsed().as_secs_f64();
        let g = &ctx.graph;

        let expanded = g.expanded_count.load(Ordering::Relaxed);
        let states_total = g.state_count();
        let edges = g.edges();
        let dead = g.dead_count.load(Ordering::Relaxed);
        let frontier_size = ctx.frontier.len();
        let death_q = g.death_q.len();
        let in_flight = g.in_flight.load(Ordering::Relaxed);
        let expand_rate = expanded as f64 / secs.max(1e-9);

        // `alive` = expanded-minus-dead (BOUNDED evaluated survivors), NOT states_total-minus-dead:
        // the latter counts the unexamined frontier as "alive". Unexamined states = `frontier_size`.
        let alive = evaluated_alive(expanded, dead);
        let death_sat = 100.0 * dead as f64 / expanded.max(1) as f64;
        let seed_dead = u8::from(g.is_dead(0));

        writeln!(
            self.file,
            "{:.3},{},{},{},{},{},{:.2},{},{},{},{},{:.2}",
            secs,
            expanded,
            states_total,
            edges,
            dead,
            alive,
            death_sat,
            seed_dead,
            frontier_size,
            death_q,
            in_flight,
            expand_rate
        )?;

        self.file.flush()?;
        Ok(())
    }
}

/// Logger thread: periodically prints engine statistics and logs to CSV.
/// No disk save happens here — the committed policy is written once after quiescence.
fn logger_thread(ctx: LoggerContext, csv_file: String) {
    let mut last_logged = Instant::now();
    let mut seed_death_announced = false;
    let mut cap_stop_announced = false;

    // Initialize CSV logger
    let mut csv_logger = match CsvLogger::new(&csv_file) {
        Ok(logger) => {
            println!("CSV logging to: {}", csv_file);
            Some(logger)
        }
        Err(e) => {
            eprintln!("Failed to create CSV logger: {e}");
            None
        }
    };

    while !ctx.shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(1));

        // Optional memory-safety cap (only if `--max-states` was given): if the interned-state
        // count reaches the ceiling, request a graceful early stop (the conservative finalize path)
        // instead of letting the process OOM-kill.
        if !cap_stop_announced && should_request_stop(ctx.graph.state_count(), ctx.state_cap) {
            let secs = ctx.start.elapsed().as_secs_f64();
            let cap = ctx.state_cap.unwrap_or(0);
            println!(
                "\n⚠️  STATE CAP HIT at t={secs:.1}s: {} interned states >= --max-states {cap}. \
                Stopping expansion early and finalizing a conservative safe set (raise \
                --max-states if you have RAM, or tighten the carrier to finish within it).\n",
                ctx.graph.state_count()
            );
            ctx.shutdown_requested.store(true, Ordering::Relaxed);
            cap_stop_announced = true;
        }

        // Definitive "we lost" alarm: the moment the empty board (seed, StateId 0) dies, the
        // safe set cannot contain the start. Fire once, as soon as it happens.
        if !seed_death_announced && ctx.graph.is_dead(0) {
            let secs = ctx.start.elapsed().as_secs_f64();
            println!(
                "\n⚠️  SEED DEAD at t={secs:.1}s: the empty board is forced to lose within this \
                carrier — the run has FAILED. The engine is stopping itself now.\n"
            );
            seed_death_announced = true;
        }

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

/// Print current engine statistics. Field meanings: `interned` = total distinct states discovered
/// (most are UNEXAMINED — they sit in `frontier`); `eval_alive` = expanded-minus-dead, the BOUNDED
/// count of states actually evaluated and not (yet) proven dead — this is the honest "alive so far"
/// (do NOT read `interned - dead` as alive: that counts the unexamined frontier); `dead% of exp` =
/// death saturation among expanded states; `seed` = whether the empty board (`StateId 0`) is still
/// alive (once DEAD the safe set cannot contain the start). Death numbers only sharpen as the
/// frontier drains; `eval_alive` is sound (a true safe-set upper bound) only at full quiescence.
fn print_stats(ctx: &LoggerContext) {
    let secs = ctx.start.elapsed().as_secs_f64().max(1e-9);
    let g = &ctx.graph;

    let expanded = g.expanded_count.load(Ordering::Relaxed);
    let states_total = g.state_count();
    let dead = g.dead_count.load(Ordering::Relaxed);
    let frontier_size = ctx.frontier.len();
    let death_q = g.death_q.len();
    let expand_rate = expanded as f64 / secs;

    // Bounded evaluated-survivors count; see `evaluated_alive`. The unexamined states are the
    // separate `frontier` field — do NOT read `interned - dead` as "alive".
    let alive = evaluated_alive(expanded, dead);
    let death_sat = 100.0 * dead as f64 / expanded.max(1) as f64;
    let seed = if g.is_dead(0) { "DEAD" } else { "alive" };

    println!(
        "[engine] t={secs:.1}s exp={expanded} interned={states_total} eval_alive={alive} dead={dead} \
        ({death_sat:.1}% of exp) | frontier={frontier_size} death_q={death_q} seed={seed} \
        rate={expand_rate:.1}/s"
    );
}

/// Entry point
fn main() {
    let cli = Cli::parse();

    match cli.mode {
        Mode::Create { max_states } => run_tetris_atlas_create(&cli.path, max_states),
        Mode::Explore => run_tetris_atlas_explore(&cli.path),
        Mode::Verify {
            depth,
            holes,
            height,
            rough,
        } => run_referee(depth, holes, height, rough),
    }
}

/// Verify the committed atlas is closed: every committed move lands on a state that is itself
/// fully committed (has a move for every forced piece). Returns the number of violations.
fn verify_closure(
    lookup: &FxDashMap<(TetrisBoard, TetrisPieceBagState, TetrisPiece), TetrisPieceOrientation>,
) -> usize {
    let mut violations = 0usize;
    for entry in lookup.iter() {
        let ((board, bag, piece), orientation) = (*entry.key(), *entry.value());

        let mut next_board = board;
        let result = next_board.apply_piece_placement(TetrisPiecePlacement { piece, orientation });
        if result.is_lost == IsLost::LOST {
            violations += 1; // a committed move must never top out
            continue;
        }

        // Successor bag = current bag minus the placed piece (auto-refilled when empty).
        let mut next_bag = bag;
        next_bag.remove(piece);
        if next_bag.is_empty() {
            next_bag.fill();
        }

        // Closure: the successor state must have a committed move for every forced piece.
        for (next_piece, _) in next_bag.iter_next_states() {
            if !lookup.contains_key(&(next_board, next_bag, next_piece)) {
                violations += 1;
            }
        }
    }
    violations
}

/// Independent "referee": a from-scratch, single-threaded, depth-bounded AND-OR minimax that
/// re-derives whether `(board, bag)` is **forced to lose within `depth` more pieces** — sharing
/// NONE of the engine's interner / edge graph / death-propagation / concurrency code. Used to
/// cross-check the engine's verdict (especially the "wrongly declared lost" direction, which the
/// `audit_leaks` safety check cannot see).
///
/// Returns true iff the adversary (who picks the next piece) can force a loss: there is a piece
/// in the bag for which the player has no placement that is legal, stays in-carrier, and avoids
/// a forced loss in the remaining `depth - 1` pieces. `in_carrier` is passed in so tests can
/// supply their own boundary independently of the global carrier.
fn forced_loss_within(
    board: TetrisBoard,
    bag: TetrisPieceBagState,
    depth: u32,
    in_carrier: &impl Fn(&TetrisBoard) -> bool,
    memo: &mut HashMap<(TetrisBoard, TetrisPieceBagState, u32), bool>,
) -> bool {
    if depth == 0 {
        return false; // survived the horizon without being forced to lose
    }
    if let Some(&cached) = memo.get(&(board, bag, depth)) {
        return cached;
    }

    let mut forced = false;
    for (piece, next_bag) in bag.iter_next_states() {
        // The player needs ANY placement of this piece that survives; if none, the adversary
        // simply hands them this piece and they lose.
        let mut survivable = false;
        for &placement in TetrisPiecePlacement::all_from_piece(piece) {
            let mut next_board = board;
            let result = next_board.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST || !in_carrier(&next_board) {
                continue; // tops out or leaves the carrier — not a usable move
            }
            if !forced_loss_within(next_board, next_bag, depth - 1, in_carrier, memo) {
                survivable = true;
                break;
            }
        }
        if !survivable {
            forced = true;
            break;
        }
    }

    memo.insert((board, bag, depth), forced);
    forced
}

/// Smallest piece-count at which the empty board is forced to lose within the given carrier,
/// or `None` if it survives at least `cap` pieces.
fn referee_forced_loss_depth(cap: u32, in_carrier: &impl Fn(&TetrisBoard) -> bool) -> Option<u32> {
    let mut memo = HashMap::new();
    let board = TetrisBoard::EMPTY_BOARD;
    let bag = TetrisPieceBagState::new();
    (1..=cap).find(|&k| forced_loss_within(board, bag, k, in_carrier, &mut memo))
}

/// `verify` subcommand: run the independent referee from the empty board and print its verdict,
/// to cross-check the engine's SUCCESS/FAILED result. The carrier is supplied at runtime so you
/// can sweep tightness without recompiling.
pub fn run_referee(cap: u32, holes: u32, height: u32, rough: u32) {
    let in_carrier = |b: &TetrisBoard| {
        b.total_holes() <= holes && b.height() <= height && surface_roughness(b) <= rough
    };

    println!("Referee: independent depth-bounded forced-loss check from the empty board");
    println!("  Carrier: holes <= {holes}, height <= {height}, roughness <= {rough}");
    println!("  Max depth: {cap} pieces");

    let start = Instant::now();
    let verdict = referee_forced_loss_depth(cap, &in_carrier);
    let secs = start.elapsed().as_secs_f64();

    match verdict {
        Some(k) => println!(
            "\nReferee verdict: FORCED LOSS — the empty board is forced to lose in {k} piece(s). \
             This independently CONFIRMS the engine's FAILED result. [{secs:.2}s]"
        ),
        None => println!(
            "\nReferee verdict: NO forced loss within {cap} pieces — the empty board survives at \
             least {cap} pieces here. If the engine reported FAILED, the forced loss is deeper \
             than {cap} (raise --depth) or there is a disagreement to investigate. [{secs:.2}s]"
        ),
    }
}

/// Safety audit: count "leak" states — non-dead states that have a forced piece with zero
/// surviving (non-dead) successor edges. Such a state SHOULD have died, so a nonzero count
/// means the death fixpoint missed deaths (a concurrency bug). This is the graph-level closure
/// invariant, and unlike `verify_closure` it is meaningful even when the committed map is empty.
/// Prints a few examples (live count vs actual surviving edges) when it trips.
fn audit_leaks(graph: &Graph) -> usize {
    let n = graph.state_count();
    let mut dead = vec![false; n];
    for e in graph.nodes.iter() {
        dead[*e.key() as usize] = e.value().dead;
    }
    let mut leaks = 0usize;
    let mut shown = 0;
    for e in graph.nodes.iter() {
        let node = e.value();
        if node.dead {
            continue;
        }
        for (piece, _) in node.bag.iter_next_states() {
            let pi = piece.index() as usize;
            let surviving = node.forward[pi]
                .iter()
                .filter(|(_, s)| !dead[*s as usize])
                .count();
            if surviving == 0 {
                leaks += 1;
                if shown < 8 {
                    println!(
                        "LEAK id={} piece={piece:?} live={} edges={} expanded={}",
                        e.key(),
                        node.live[pi],
                        node.forward[pi].len(),
                        node.expanded
                    );
                    shown += 1;
                }
                break;
            }
        }
    }
    leaks
}

/// One worker's loop in the continuous engine: drain death tasks first (death-first
/// scheduling), else expand the next frontier state, else exit when globally quiescent.
/// Also self-terminates the instant the seed is proven dead — the run has provably failed.
fn worker_loop(graph: &Graph, frontier: &PriorityFrontier, shutdown_requested: &AtomicBool) {
    let backoff = Backoff::new();
    loop {
        if graph.seed_dead.load(Ordering::Acquire) {
            break; // the search proved the empty board is dead => the run failed; stop now.
        }
        if shutdown_requested.load(Ordering::Relaxed) {
            break; // Ctrl+C: stop early (a conservative finalize runs after the scope).
        }
        if let Some(d) = graph.death_q.pop() {
            process_die(graph, d);
            graph.in_flight.fetch_sub(1, Ordering::SeqCst);
            backoff.reset();
        } else if let Some((board, bag)) = frontier.pop() {
            process_expand(graph, frontier, board, bag);
            graph.in_flight.fetch_sub(1, Ordering::SeqCst);
            backoff.reset();
        } else if graph.in_flight.load(Ordering::SeqCst) == 0 {
            break; // global quiescence: no queued work and nothing in progress
        } else {
            backoff.snooze(); // work exists on another worker; wait for it to surface
        }
    }
}

/// Run the atlas builder as one continuous, fully-parallel AND-OR engine: expansion and
/// death-propagation interleave across all workers, then a single parallel commit at quiescence.
pub fn run_tetris_atlas_create(base_path: &str, max_states: Option<usize>) {
    let num_workers = num_cpus::get().saturating_sub(2).max(1);

    let save_file = format!("{}/{}", base_path, SAVE_FILE);
    let csv_file = format!("{}/{}", base_path, CSV_FILE);

    println!("Starting continuous AND-OR atlas engine:");
    println!("  Workers: {num_workers}");
    println!(
        "  Carrier: holes <= {MAX_CARRIER_HOLES}, height <= {MAX_CARRIER_HEIGHT}, roughness <= {MAX_CARRIER_ROUGHNESS}"
    );
    match max_states {
        Some(cap) => {
            println!("  State cap: {cap} interned states (graceful early stop when reached)")
        }
        None => {
            println!("  State cap: none (unbounded — may OOM; pass --max-states N to bound it)")
        }
    }
    println!("  Save path: {base_path}");
    println!("Press Ctrl+C to stop early (yields a conservative safe set)...");

    // SharedState owns the committed lookup (written at quiescence) + the frontier + flags.
    let state = SharedState::new();
    let graph = Arc::new(Graph::new());

    // Seed the empty board: intern it, count it in-flight, push to the frontier.
    let (seed_id, _) = graph.intern(TetrisBoard::EMPTY_BOARD, TetrisPieceBagState::new());
    graph.in_flight.fetch_add(1, Ordering::SeqCst);
    state.frontier.push(
        TetrisBoard::EMPTY_BOARD,
        TetrisPieceBagState::new(),
        seed_id,
    );

    let frontier = state.frontier.clone();
    let shutdown = state.shutdown.clone();
    let shutdown_requested = state.shutdown_requested.clone();
    let start = Instant::now();

    // Ctrl+C requests an early stop.
    let shutdown_requested_clone = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received! Stopping engine early...");
        shutdown_requested_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    // Logger thread (live stats; no disk save during the run). It also enforces the state cap by
    // setting `shutdown_requested` (the same graceful early-stop the Ctrl+C handler triggers).
    let logger_ctx = LoggerContext {
        graph: graph.clone(),
        frontier: frontier.clone(),
        shutdown: shutdown.clone(),
        shutdown_requested: shutdown_requested.clone(),
        state_cap: max_states,
        start,
    };
    let logger_handle = thread::spawn(move || {
        logger_thread(logger_ctx, csv_file);
    });

    // --- Continuous engine: all workers pull expand + die tasks until quiescent ---
    println!("Engine: expanding + propagating death across {num_workers} workers...");
    rayon::scope(|s| {
        for _ in 0..num_workers {
            let graph = graph.clone();
            let frontier = frontier.clone();
            let shutdown_requested = shutdown_requested.clone();
            s.spawn(move |_| worker_loop(&graph, &frontier, &shutdown_requested));
        }
    });

    // Stop the logger.
    shutdown.store(true, Ordering::Relaxed);
    logger_handle.join().expect("Logger panicked");

    let seed_dead = graph.seed_dead.load(Ordering::SeqCst);
    println!(
        "\nEngine {}: {} states, {} edges, {} dead in {:.2}s",
        if seed_dead {
            "STOPPED (seed dead)"
        } else {
            "quiesced"
        },
        graph.state_count(),
        graph.edges(),
        graph.dead_count.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );

    // The empty board (StateId 0) is the start of every game. If it is dead, no atlas can exist
    // from the start — the run has provably FAILED. Death is monotone so this verdict is final;
    // the search already stopped itself, and the finalize/commit below would be pointless.
    if graph.is_dead(0) {
        println!(
            "\nRESULT: FAILED — the empty board (start) is forced to lose within this carrier \
             (holes<={MAX_CARRIER_HOLES}, height<={MAX_CARRIER_HEIGHT}, roughness<={MAX_CARRIER_ROUGHNESS})."
        );
        println!(
            "  No safe atlas exists from the start. Loosen the carrier (raise roughness/height) \
             to search for a survivable region."
        );
        println!("Total time: {:.2}s", start.elapsed().as_secs_f64());
        return;
    }

    // --- Finalize (only non-trivial on Ctrl+C): conservatively kill any unexpanded survivor,
    // then drain the death queue to fixpoint. At full quiescence this is a no-op. ---
    let unexpanded: Vec<StateId> = graph
        .nodes
        .iter()
        .filter(|e| !e.value().dead && !e.value().expanded)
        .map(|e| *e.key())
        .collect();
    if !unexpanded.is_empty() {
        println!(
            "Finalize: conservatively killing {} unexpanded states (early stop)...",
            unexpanded.len()
        );
        for id in unexpanded {
            graph.death_q.push(id);
        }
        while let Some(d) = graph.death_q.pop() {
            process_die(&graph, d);
        }
    }

    // Safety audit: the dead-set must be a genuine greatest fixpoint (no missed deaths).
    let leaks = audit_leaks(&graph);
    if leaks == 0 {
        println!("Safety audit: PASSED (no leak states — death fixpoint is complete)");
    } else {
        eprintln!("Safety audit: FAILED with {leaks} leak state(s) — death fixpoint missed deaths");
    }

    // --- Commit the surviving policy in parallel ---
    println!("Committing surviving policy...");
    let summary = commit_surviving_policy(&graph, &state.lookup);
    let safe_fraction = if summary.total_states > 0 {
        100.0 * summary.safe_states as f64 / summary.total_states as f64
    } else {
        0.0
    };
    println!(
        "\nEngine summary:\n  \
         total states : {}\n  \
         dead states  : {}\n  \
         safe states  : {} ({:.2}%)\n  \
         committed    : {} moves",
        summary.total_states,
        summary.dead_states,
        summary.safe_states,
        safe_fraction,
        summary.committed_moves
    );

    // Closure verification: every committed move must land on a committed (safe) state.
    let violations = verify_closure(&state.lookup);

    // The seed can become dead DURING finalize: conservative killing of unexpanded survivors (early
    // stop) can propagate death all the way back to StateId 0 *after* the initial `is_dead(0)` check
    // above. So re-consult it here — a committed cycle that doesn't cover the START is NOT a success.
    let seed_dead_final = graph.is_dead(0);
    match decide_verdict(seed_dead_final, summary.committed_moves, violations) {
        RunVerdict::Success => println!(
            "\nRESULT: SUCCESS — the empty board (start) is safe; the committed atlas is a closed \
             safe set ({} moves over {} safe states).",
            summary.committed_moves, summary.safe_states
        ),
        RunVerdict::Failed => println!(
            "\nRESULT: FAILED — the empty board (start) died during finalize; the committed set does \
             not cover the start (committed={} closure_violations={}). No atlas exists from the start.",
            summary.committed_moves, violations
        ),
        RunVerdict::Inconclusive => eprintln!(
            "\nRESULT: INCONCLUSIVE — seed alive but committed={} closure_violations={} (investigate).",
            summary.committed_moves, violations
        ),
    }

    // Persist the committed policy.
    println!("\nSaving committed atlas to disk...");
    if let Err(e) = state.save_to_disk(&save_file) {
        eprintln!("Failed to save final state: {e}");
    }

    println!("Done. Committed atlas size: {}", state.lookup_size());
    println!("Total time: {:.2}s", start.elapsed().as_secs_f64());
}

/// Explore an existing atlas interactively using a TUI
pub fn run_tetris_atlas_explore(base_path: &str) {
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

    // Construct full file path
    let save_file = format!("{}/{}", base_path, SAVE_FILE);

    // Load lookup table from disk
    let state = SharedState::new();
    match state.load_from_disk(&save_file) {
        Ok(lookup_count) => {
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
                TetrisBoard::EMPTY_BOARD
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
            for y in 0..StandardTetris::ROWS {
                let mut row_has_filled = false;
                for x in 0..StandardTetris::COLS {
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
                        state.lookup.contains_key(&(board, bag, *piece))
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
                                        bag.contains(p)
                                            && state.lookup.contains_key(&(board, bag, p))
                                    })
                                    .collect();

                                    if !available.is_empty() {
                                        use rand::RngExt;
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
                                        state.lookup.get(&(board, bag, piece)).as_deref()
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

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::TetrisPiece;

    const ALL_PIECES: [TetrisPiece; 7] = [
        TetrisPiece::O_PIECE,
        TetrisPiece::I_PIECE,
        TetrisPiece::L_PIECE,
        TetrisPiece::J_PIECE,
        TetrisPiece::S_PIECE,
        TetrisPiece::Z_PIECE,
        TetrisPiece::T_PIECE,
    ];

    /// A bag containing exactly the given pieces (its AND-obligations).
    fn bag_of(pieces: &[TetrisPiece]) -> TetrisPieceBagState {
        let mut bag = TetrisPieceBagState::new(); // full 7-bag
        for p in ALL_PIECES {
            if !pieces.contains(&p) {
                bag.remove(p);
            }
        }
        bag
    }

    /// A distinct synthetic board per `k` (its byte pattern differs), so `(board, bag)` keys do
    /// not collide in the committed lookup. Not a "valid" board — only used as a distinct key.
    fn make_board(k: u8) -> TetrisBoard {
        let mut bytes = [0u8; 40];
        bytes[0] = k;
        TetrisBoard::from_bytes(bytes, Major::Col)
    }

    /// Insert node `id` with the given board + bag (optionally already expanded).
    fn make_node(
        g: &Graph,
        id: StateId,
        board: TetrisBoard,
        bag: TetrisPieceBagState,
        expanded: bool,
    ) {
        let mut node = Node::new(board, bag);
        node.expanded = expanded;
        g.nodes.insert(id, node);
        let want = id + 1;
        if g.next_id.load(Ordering::Relaxed) < want {
            g.next_id.store(want, Ordering::Relaxed);
        }
    }

    /// Add a forward edge `parent --piece(ori)--> succ` plus its reverse edge, updating `live`.
    fn link(g: &Graph, parent: StateId, succ: StateId, piece_idx: usize, ori_idx: u8) {
        let ori = TetrisPieceOrientation::from_index(ori_idx);
        {
            let mut p = g.nodes.get_mut(&parent).unwrap();
            p.forward[piece_idx].push((ori, succ));
            p.live[piece_idx] += 1;
        }
        {
            let mut c = g.nodes.get_mut(&succ).unwrap();
            c.preds.push(PredecessorRef {
                parent,
                piece_idx: piece_idx as u8,
            });
        }
    }

    /// Run the engine's death logic over a pre-built graph: seed deaths, then drain to fixpoint.
    fn solve(g: &Graph) -> Vec<bool> {
        let n = g.state_count();
        for id in 0..n as StateId {
            g.check_death(id);
        }
        while let Some(d) = g.death_q.pop() {
            process_die(g, d);
        }
        (0..n as StateId).map(|id| g.is_dead(id)).collect()
    }

    /// Death must propagate backward through chains while sparing states that retain a
    /// surviving placement (including a self-sustaining cycle).
    #[test]
    fn death_propagates_backward_but_spares_a_surviving_cycle() {
        let i = TetrisPiece::I_PIECE.index() as usize;
        let bag = bag_of(&[TetrisPiece::I_PIECE]);

        let g = Graph::new();
        for id in 0u32..6 {
            make_node(&g, id, make_board(id as u8), bag, true);
        }

        // 4: dead leaf (no surviving placement). Single chains 3->4 and 2->3 must die.
        link(&g, 3, 4, i, 0);
        link(&g, 2, 3, i, 0);
        // 5: self-loop (surviving cycle). 1 has a dead branch (->2) and a live branch (->5).
        link(&g, 5, 5, i, 0);
        link(&g, 1, 2, i, 0);
        link(&g, 1, 5, i, 0);
        // 0 -> 1 survives because 1 survives.
        link(&g, 0, 1, i, 0);

        let dead = solve(&g);

        assert!(!dead[0], "0 should survive (reaches surviving 1)");
        assert!(!dead[1], "1 should survive (live branch to 5)");
        assert!(dead[2], "2 should die (only route 2->3 dies)");
        assert!(dead[3], "3 should die (only route 3->4 dies)");
        assert!(dead[4], "4 should die (no surviving placement)");
        assert!(!dead[5], "5 should survive (self-sustaining cycle)");
    }

    /// AND over pieces: if any one forced piece has no surviving placement, the state dies.
    #[test]
    fn state_dies_when_any_forced_piece_is_unsatisfiable() {
        // Bag = {I, T}: two obligations. I has a self-loop placement; T has none.
        let i = TetrisPiece::I_PIECE.index() as usize;
        let bag = bag_of(&[TetrisPiece::I_PIECE, TetrisPiece::T_PIECE]);

        let g = Graph::new();
        make_node(&g, 0, make_board(0), bag, true);
        link(&g, 0, 0, i, 0); // satisfy I; T slot stays empty (live[T] == 0)

        let dead = solve(&g);
        assert!(
            dead[0],
            "state must die: forced piece T has no surviving placement"
        );
    }

    /// Invariant #3: a `live` count hitting 0 must NOT kill a parent until it is `expanded`.
    /// This is what keeps a transient mid-expansion zero from falsely killing a state.
    #[test]
    fn death_by_exhaustion_waits_for_expanded() {
        let i = TetrisPiece::I_PIECE.index() as usize;
        let bag = bag_of(&[TetrisPiece::I_PIECE]);

        let g = Graph::new();
        make_node(&g, 0, make_board(0), bag, false); // parent NOT yet expanded
        make_node(&g, 1, make_board(1), bag, true); // child (will die)
        link(&g, 0, 1, i, 0); // parent's only I-placement goes to the child

        // Kill the child: this decrements parent.live[I] to 0, but parent is not expanded yet.
        process_die(&g, 1);
        assert!(
            !g.is_dead(0),
            "unexpanded parent must not die on a transient zero"
        );

        // Once expanded, the base-case death check observes live[I] == 0 and kills it.
        g.nodes.get_mut(&0).unwrap().expanded = true;
        g.check_death(0);
        while let Some(d) = g.death_q.pop() {
            process_die(&g, d);
        }
        assert!(
            g.is_dead(0),
            "expanded parent with no surviving placement must die"
        );
    }

    /// Commit must write exactly one move per (safe state, forced piece), choosing a SURVIVING
    /// successor edge (not a dead one), and must write nothing for dead states.
    #[test]
    fn commit_writes_surviving_edges_and_skips_dead_states() {
        let i = TetrisPiece::I_PIECE.index() as usize;
        let bag = bag_of(&[TetrisPiece::I_PIECE]);

        let g = Graph::new();
        for id in 0u32..6 {
            make_node(&g, id, make_board(id as u8), bag, true);
        }
        // Same survivor topology as the death test, but with DISTINCT orientations per edge so we
        // can assert which edge was committed. Survivors: {0,1,5}; dead: {2,3,4}.
        link(&g, 0, 1, i, 1); // 0 -> 1 (only edge, ori 1)
        link(&g, 1, 2, i, 2); // 1 -> 2 (DEAD branch, ori 2)
        link(&g, 1, 5, i, 3); // 1 -> 5 (LIVE branch, ori 3) -> commit must pick this
        link(&g, 5, 5, i, 4); // 5 -> 5 self-loop (ori 4)
        link(&g, 2, 3, i, 0);
        link(&g, 3, 4, i, 0);

        let dead = solve(&g);
        assert_eq!(dead, vec![false, false, true, true, true, false]);

        let lookup = new_fx_dashmap();
        let summary = commit_surviving_policy(&g, &lookup);

        assert_eq!(summary.safe_states, 3);
        assert_eq!(summary.dead_states, 3);
        assert_eq!(summary.committed_moves, 3);
        assert_eq!(
            lookup.len(),
            3,
            "one entry per safe state (distinct boards)"
        );

        let key = |k: u8| (make_board(k), bag, TetrisPiece::I_PIECE);
        // State 0: only surviving edge is ->1 (ori 1).
        assert_eq!(lookup.get(&key(0)).unwrap().index(), 1);
        // State 1: the DEAD edge ->2 (ori 2) must be filtered out; commit picks ->5 (ori 3).
        assert_eq!(lookup.get(&key(1)).unwrap().index(), 3);
        // State 5: self-loop (ori 4).
        assert_eq!(lookup.get(&key(5)).unwrap().index(), 4);
        // Dead states commit nothing.
        for k in [2u8, 3, 4] {
            assert!(
                !lookup.contains_key(&key(k)),
                "dead state {k} must not commit"
            );
        }
    }

    /// `verify_closure` must flag an open (non-closed) policy and accept the empty policy.
    #[test]
    fn verify_closure_detects_open_policy() {
        // Empty policy is trivially closed.
        let empty: FxDashMap<
            (TetrisBoard, TetrisPieceBagState, TetrisPiece),
            TetrisPieceOrientation,
        > = new_fx_dashmap();
        assert_eq!(verify_closure(&empty), 0);

        // A single real committed move whose successor state has no committed moves is OPEN.
        let lookup = new_fx_dashmap();
        let board = TetrisBoard::EMPTY_BOARD;
        let bag = TetrisPieceBagState::new();
        let piece = TetrisPiece::I_PIECE;
        let placement = TetrisPiecePlacement::all_from_piece(piece)[0];
        lookup.insert((board, bag, piece), placement.orientation);

        let violations = verify_closure(&lookup);
        assert!(
            violations > 0,
            "an uncovered successor must be reported as a closure violation"
        );
    }

    // --- Independent referee (forced_loss_within) logic, validated with custom carriers so the
    // tests don't depend on the global carrier constants. ---

    /// Depth 0 means "we stopped looking" — never a forced loss.
    #[test]
    fn referee_depth_zero_is_never_forced_loss() {
        let mut memo = HashMap::new();
        assert!(!forced_loss_within(
            TetrisBoard::EMPTY_BOARD,
            TetrisPieceBagState::new(),
            0,
            &|_| true,
            &mut memo
        ));
    }

    /// If NOTHING is in-carrier, every placement is rejected, so the very first piece the
    /// adversary hands you is already a forced loss.
    #[test]
    fn referee_immediate_loss_when_nothing_fits() {
        let mut memo = HashMap::new();
        assert!(forced_loss_within(
            TetrisBoard::EMPTY_BOARD,
            TetrisPieceBagState::new(),
            1,
            &|_| false,
            &mut memo
        ));
    }

    /// With an unrestricted carrier the player can always place a piece, so a short look-ahead
    /// finds no forced loss.
    #[test]
    fn referee_no_loss_when_everything_fits() {
        let mut memo = HashMap::new();
        assert!(!forced_loss_within(
            TetrisBoard::EMPTY_BOARD,
            TetrisPieceBagState::new(),
            5,
            &|_| true,
            &mut memo
        ));
    }

    // --- Run-verdict logic (guards the false-SUCCESS-after-finalize bug) ---

    /// THE bug guard: a dead seed must be FAILED even when a disconnected safe cycle committed
    /// moves and passed closure. Reporting SUCCESS here would falsely claim the empty board is safe.
    #[test]
    fn verdict_dead_seed_is_failed_even_with_committed_closed_cycle() {
        assert_eq!(decide_verdict(true, 5, 0), RunVerdict::Failed);
        assert_eq!(decide_verdict(true, 0, 0), RunVerdict::Failed);
        assert_eq!(decide_verdict(true, 5, 3), RunVerdict::Failed);
    }

    /// A live seed with a non-empty, closure-clean policy is the only SUCCESS case.
    #[test]
    fn verdict_live_seed_with_closed_policy_is_success() {
        assert_eq!(decide_verdict(false, 5, 0), RunVerdict::Success);
    }

    /// A live seed but empty policy or closure violations is INCONCLUSIVE, not SUCCESS.
    #[test]
    fn verdict_live_seed_open_or_empty_is_inconclusive() {
        assert_eq!(decide_verdict(false, 0, 0), RunVerdict::Inconclusive);
        assert_eq!(decide_verdict(false, 5, 2), RunVerdict::Inconclusive);
    }

    // --- Early-stop cap decision ---

    /// No cap (`None`) never stops; a `Some` cap is an inclusive ceiling.
    #[test]
    fn cap_requests_stop_only_when_set_and_reached() {
        // Omitted cap => unbounded => never stop, no matter how large the graph grows.
        assert!(!should_request_stop(0, None));
        assert!(!should_request_stop(usize::MAX, None));
        // Explicit cap => stop at or above it, keep going below it.
        assert!(!should_request_stop(999, Some(1000)));
        assert!(should_request_stop(1000, Some(1000)));
        assert!(should_request_stop(1001, Some(1000)));
    }

    // --- Bounded "alive" metric ---

    /// `evaluated_alive` counts only expanded-minus-dead and never underflows.
    #[test]
    fn evaluated_alive_is_bounded_and_saturating() {
        assert_eq!(evaluated_alive(100, 30), 70);
        assert_eq!(evaluated_alive(100, 100), 0);
        assert_eq!(evaluated_alive(30, 100), 0); // saturating: never wraps
    }

    // --- Persistence round-trip (the graceful early-stop relies on committing + saving whatever
    // conservative safe set exists, so the on-disk format must round-trip exactly). ---

    /// `save_to_disk` then `load_from_disk` must reproduce the committed lookup exactly:
    /// same entry count, same (board, bag, piece) keys, same orientation values.
    #[test]
    fn save_load_roundtrip_preserves_committed_lookup() {
        let bag = bag_of(&[TetrisPiece::I_PIECE, TetrisPiece::T_PIECE]);
        let (b0, p0, o0) = (
            make_board(0),
            TetrisPiece::I_PIECE,
            TetrisPieceOrientation::from_index(2),
        );
        let (b1, p1, o1) = (
            make_board(9),
            TetrisPiece::T_PIECE,
            TetrisPieceOrientation::from_index(0),
        );

        let state = SharedState::new();
        state.lookup.insert((b0, bag, p0), o0);
        state.lookup.insert((b1, bag, p1), o1);

        let path = std::env::temp_dir().join("tetris_atlas_inmemory_roundtrip_test.bin");
        let path = path.to_str().unwrap();
        state.save_to_disk(path).expect("save");

        let loaded = SharedState::new();
        let n = loaded.load_from_disk(path).expect("load");
        assert_eq!(n, 2, "loaded entry count");
        assert_eq!(loaded.lookup.len(), 2);
        assert_eq!(
            loaded.lookup.get(&(b0, bag, p0)).unwrap().index(),
            o0.index()
        );
        assert_eq!(
            loaded.lookup.get(&(b1, bag, p1)).unwrap().index(),
            o1.index()
        );

        let _ = std::fs::remove_file(path);
    }
}
