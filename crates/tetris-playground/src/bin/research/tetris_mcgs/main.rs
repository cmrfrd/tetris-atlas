#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Monte Carlo Graph Search (MCGS) Cycle Finder
//!
//! Explores the Tetris game graph using AND-OR MCGS to discover closed cycles
//! proving infinite play under adversarial 7-bag piece selection.
//!
//! ## Storage backends
//!
//! - **In-memory** (default): fast, bounded by `--max-nodes`
//! - **SQLite** (`--db path.db`): unbounded, persistent, resumable

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
    mpsc::{self, Receiver, TryRecvError},
};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use rusqlite::{Connection, OpenFlags, TransactionBehavior, params};
use rustc_hash::FxHashMap;
use tetris_game::{
    IsLost, Major, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPiecePlacement,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "tetris_mcgs")]
#[command(about = "Monte Carlo Graph Search cycle finder for proving infinite Tetris play")]
struct Cli {
    /// Maximum board height during intermediate placements.
    #[arg(long, default_value_t = 6)]
    max_height: u32,

    /// Maximum holes during intermediate placements.
    #[arg(long, default_value_t = 3)]
    max_holes: u32,

    /// Maximum roughness during intermediate placements.
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,

    /// MCGS iterations.
    #[arg(long, default_value_t = 500_000)]
    iterations: u64,

    /// PUCT exploration constant.
    #[arg(long, default_value_t = 1.4)]
    exploration: f32,

    /// Reward for returning to already-visited/recurrent graph states.
    #[arg(long, default_value_t = 0.25)]
    recurrence_weight: f32,

    /// Penalty for moves that create or depend on novel graph states.
    #[arg(long, default_value_t = 0.10)]
    novelty_cost: f32,

    /// Softmax temperature for placement priors.
    #[arg(long, default_value_t = 2.0)]
    prior_temperature: f32,

    /// Maximum depth in bag-cycles before heuristic cutoff.
    #[arg(long, default_value_t = 3)]
    max_bags: usize,

    /// Maximum graph nodes (in-memory mode only).
    #[arg(long, default_value_t = 2_000_000)]
    max_nodes: usize,

    /// Minimum visits for a full-bag node to become a cycle candidate.
    #[arg(long, default_value_t = 20)]
    cycle_visit_threshold: u32,

    /// How often to refresh the candidate cycle set (in iterations).
    #[arg(long, default_value_t = 10_000)]
    candidate_refresh: u64,

    /// Maximum candidate boards in the cycle set.
    #[arg(long, default_value_t = 200)]
    max_candidates: usize,

    /// Log interval (iterations).
    #[arg(long, default_value_t = 1_000)]
    log_every: u64,

    /// SQLite database path. If set, uses disk-backed storage (no OOM, resumable).
    #[arg(long)]
    db: Option<String>,

    /// Flush SQLite-backed node metadata every N iterations. 0 = final flush only.
    #[arg(long, default_value_t = 100_000)]
    db_flush_every: u64,

    /// How often to run full-bag verification on candidates (in iterations). 0 = never.
    #[arg(long, default_value_t = 100_000)]
    verify_every: u64,

    /// Log interval inside full-bag candidate verification, counted in permutation checks. 0 = silent.
    #[arg(long, default_value_t = 1_000)]
    verify_log_every: u64,

    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

// ---------------------------------------------------------------------------
// State key
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct StateKey {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl StateKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }

    fn empty() -> Self {
        Self {
            board: TetrisBoard::EMPTY_BOARD,
            bag: TetrisPieceBagState::new(),
        }
    }

    fn is_full_bag(&self) -> bool {
        self.bag == TetrisPieceBagState::new()
    }

    fn board_bytes(&self) -> [u8; 40] {
        let limbs = self.board.as_limbs();
        let mut bytes = [0u8; 40];
        for (i, limb) in limbs.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
        }
        bytes
    }

    fn from_board_bytes(bytes: &[u8], bag: u8) -> Self {
        let mut limbs = [0u32; 10];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        Self {
            board: TetrisBoard::from_bytes(
                unsafe { std::mem::transmute::<[u32; 10], [u8; 40]>(limbs) },
                Major::Col,
            ),
            bag: TetrisPieceBagState::from(bag),
        }
    }
}

type NodeIdx = u32;
const INVALID_IDX: NodeIdx = u32::MAX;
const DEAD_VALUE: f32 = -1.0;
const NODE_CAP_VALUE: f32 = -0.25;

// ---------------------------------------------------------------------------
// Graph node types (in-memory representation)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct PredecessorEdge {
    parent: NodeIdx,
    piece_group: usize,
    placement: usize,
}

#[derive(Clone)]
struct Edge {
    child: NodeIdx,
    child_key: StateKey,
    visits: u32,
    value_sum: f32,
    prior: f32,
    dead: bool,
}

/// A candidate placement not yet materialized as an edge.
/// Sorted by cost (ascending). Promoted to Edge on demand.
#[derive(Clone)]
struct PendingPlacement {
    child_key: StateKey,
    prior: f32,
}

#[derive(Clone)]
struct PieceGroup {
    piece: TetrisPiece,
    visits: u32,
    value_sum: f32,
    prior: f32,
    dead: bool,
    total_placements: u16,
    invalid_placements: u16,
    /// Active edges (already materialized children).
    placements: Vec<Edge>,
    /// Pending placements sorted by prior descending. Promoted to `placements` lazily.
    pending: Vec<PendingPlacement>,
}

/// How many OR children to materialize: C * visits^alpha, minimum `min_children`.
const PW_C: f32 = 1.5;
const PW_ALPHA: f32 = 0.5;
const PW_MIN: usize = 2;

fn progressive_width(visits: u32) -> usize {
    let w = (PW_C * (visits as f32).powf(PW_ALPHA)).ceil() as usize;
    w.max(PW_MIN)
}

struct Node {
    key: StateKey,
    visits: u32,
    value_sum: f32,
    expanded: bool,
    dead: bool,
    piece_groups: Vec<PieceGroup>,
    predecessors: Vec<PredecessorEdge>,
    bags_deep: u16,
}

impl Node {
    fn new(key: StateKey, bags_deep: u16) -> Self {
        Self {
            key,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            dead: false,
            piece_groups: Vec::new(),
            predecessors: Vec::new(),
            bags_deep,
        }
    }

    fn avg_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Board heuristics
// ---------------------------------------------------------------------------

fn has_too_many_holes(board: &TetrisBoard, max_holes: u32) -> bool {
    let limbs = board.as_limbs();
    let mut sum = 0u32;
    for col in &limbs {
        let h = u32::BITS - col.leading_zeros();
        sum += h - col.count_ones();
        if sum > max_holes {
            return true;
        }
    }
    false
}

fn board_ok(board: &TetrisBoard, max_height: u32, max_holes: u32, max_roughness: u32) -> bool {
    board.height() <= max_height
        && (max_holes == u32::MAX || !has_too_many_holes(board, max_holes))
        && (max_roughness == u32::MAX || board.roughness() <= max_roughness)
}

fn score_board(board: &TetrisBoard) -> f32 {
    let cells = board.count();
    if cells == 0 {
        return 0.0;
    }
    let limbs = board.as_limbs();
    let mut holes = 0u32;
    let mut col_heights = [0u32; 10];
    for (i, col) in limbs.iter().enumerate() {
        col_heights[i] = u32::BITS - col.leading_zeros();
        holes += col_heights[i] - col.count_ones();
    }
    let mut roughness = 0u32;
    for i in 0..9 {
        roughness += col_heights[i].abs_diff(col_heights[i + 1]);
    }
    cells as f32 + holes as f32 * 4.0 + roughness as f32
}

fn heuristic_value(board: &TetrisBoard) -> f32 {
    let cost = score_board(board);
    (1.0 - cost / 30.0).clamp(-1.0, 1.0)
}

fn recurrence_score(visits: u32) -> f32 {
    ((visits as f32 + 1.0).ln() / 8.0).min(1.0)
}

fn novelty_score(visits: u32) -> f32 {
    if visits == 0 {
        1.0
    } else {
        1.0 / (visits as f32 + 1.0).sqrt()
    }
}

fn edge_value_adjustment(
    store: &dyn GraphStore,
    edge: &Edge,
    recurrence_weight: f32,
    novelty_cost: f32,
) -> f32 {
    if edge.child == INVALID_IDX {
        return heuristic_value(&edge.child_key.board) * 0.25 - novelty_cost;
    }

    let Some(child) = store.get_node(edge.child) else {
        return DEAD_VALUE;
    };
    if child.dead {
        DEAD_VALUE
    } else {
        let bag_boundary_bonus = if child.key.is_full_bag() { 0.05 } else { 0.0 };
        recurrence_weight * recurrence_score(child.visits) + bag_boundary_bonus
            - novelty_cost * novelty_score(child.visits)
    }
}

fn leaf_value(
    store: &dyn GraphStore,
    node_idx: NodeIdx,
    recurrence_weight: f32,
    novelty_cost: f32,
) -> f32 {
    let Some(node) = store.get_node(node_idx) else {
        return DEAD_VALUE;
    };
    if node.dead {
        return DEAD_VALUE;
    }
    (heuristic_value(&node.key.board) + recurrence_weight * recurrence_score(node.visits)
        - novelty_cost * novelty_score(node.visits))
    .clamp(DEAD_VALUE, 1.0)
}

fn piece_group_has_possible_response(pg: &PieceGroup) -> bool {
    !pg.pending.is_empty() || pg.placements.iter().any(|edge| !edge.dead)
}

#[derive(Clone, Copy, Default)]
struct ExpansionSummary {
    lost_placements: u64,
    condition_violations: u64,
    immediate_dead_piece_groups: u64,
}

#[derive(Clone, Copy, Default)]
struct McgsStats {
    lost_placements: u64,
    condition_violations: u64,
    hard_dead_edges: u64,
    hard_dead_piece_groups: u64,
    hard_dead_states: u64,
    node_cap_hits: u64,
}

#[derive(Default)]
struct ClosureSnapshot {
    core_states: u64,
    covered: u64,
    required: u64,
    open: u64,
    weighted_covered: u64,
    weighted_required: u64,
}

impl ClosureSnapshot {
    fn closure(&self) -> f64 {
        if self.required == 0 {
            0.0
        } else {
            self.covered as f64 / self.required as f64
        }
    }

    fn weighted_closure(&self) -> f64 {
        if self.weighted_required == 0 {
            0.0
        } else {
            self.weighted_covered as f64 / self.weighted_required as f64
        }
    }
}

fn all_piece_mask() -> u8 {
    TetrisPiece::all()
        .iter()
        .fold(0u8, |mask, &piece| mask | u8::from(piece))
}

// ---------------------------------------------------------------------------
// Graph storage trait
// ---------------------------------------------------------------------------

trait GraphStore {
    fn node_count(&self) -> usize;
    fn get_node(&self, idx: NodeIdx) -> Option<&Node>;
    fn get_node_mut(&mut self, idx: NodeIdx) -> Option<&mut Node>;
    fn lookup(&self, key: &StateKey) -> Option<NodeIdx>;
    fn get_or_create(&mut self, key: StateKey, bags_deep: u16) -> NodeIdx;
    fn can_grow(&self) -> bool;
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

fn add_predecessor(store: &mut dyn GraphStore, child_idx: NodeIdx, pred: PredecessorEdge) {
    if child_idx == INVALID_IDX {
        return;
    }
    let Some(child) = store.get_node_mut(child_idx) else {
        return;
    };
    if !child.predecessors.contains(&pred) {
        child.predecessors.push(pred);
    }
}

fn mark_state_dead(store: &mut dyn GraphStore, node_idx: NodeIdx, stats: &mut McgsStats) {
    let predecessors = {
        let Some(node) = store.get_node_mut(node_idx) else {
            return;
        };
        if node.dead {
            return;
        }
        node.dead = true;
        stats.hard_dead_states += 1;
        node.predecessors.clone()
    };

    for pred in predecessors {
        mark_edge_dead(store, pred.parent, pred.piece_group, pred.placement, stats);
    }
}

fn mark_edge_dead(
    store: &mut dyn GraphStore,
    parent_idx: NodeIdx,
    pg_idx: usize,
    edge_idx: usize,
    stats: &mut McgsStats,
) {
    let should_mark_parent_dead = {
        let Some(parent) = store.get_node_mut(parent_idx) else {
            return;
        };
        if parent.dead || pg_idx >= parent.piece_groups.len() {
            return;
        }
        let pg = &mut parent.piece_groups[pg_idx];
        if edge_idx >= pg.placements.len() {
            return;
        }
        if !pg.placements[edge_idx].dead {
            pg.placements[edge_idx].dead = true;
            stats.hard_dead_edges += 1;
        }
        if !pg.dead && !piece_group_has_possible_response(pg) {
            pg.dead = true;
            stats.hard_dead_piece_groups += 1;
            true
        } else {
            false
        }
    };

    if should_mark_parent_dead {
        mark_state_dead(store, parent_idx, stats);
    }
}

fn refresh_node_deadness(store: &mut dyn GraphStore, node_idx: NodeIdx, stats: &mut McgsStats) {
    let should_mark_dead = {
        let Some(node) = store.get_node_mut(node_idx) else {
            return;
        };
        if node.dead {
            return;
        }

        let mut has_dead_piece = false;
        for pg in &mut node.piece_groups {
            if !pg.dead && !piece_group_has_possible_response(pg) {
                pg.dead = true;
                stats.hard_dead_piece_groups += 1;
            }
            has_dead_piece |= pg.dead;
        }
        has_dead_piece
    };

    if should_mark_dead {
        mark_state_dead(store, node_idx, stats);
    }
}

fn sync_piece_group_dead_edges(
    store: &mut dyn GraphStore,
    node_idx: NodeIdx,
    pg_idx: usize,
    stats: &mut McgsStats,
) {
    let child_edges: Vec<(usize, NodeIdx)> = {
        let Some(node) = store.get_node(node_idx) else {
            return;
        };
        if node.dead || pg_idx >= node.piece_groups.len() {
            return;
        }
        node.piece_groups[pg_idx]
            .placements
            .iter()
            .enumerate()
            .filter(|(_, edge)| !edge.dead && edge.child != INVALID_IDX)
            .map(|(idx, edge)| (idx, edge.child))
            .collect()
    };

    for (edge_idx, child_idx) in child_edges {
        if store.get_node(child_idx).is_some_and(|child| child.dead) {
            mark_edge_dead(store, node_idx, pg_idx, edge_idx, stats);
        }
    }
    refresh_node_deadness(store, node_idx, stats);
}

fn materialize_edge_child(
    store: &mut dyn GraphStore,
    parent_idx: NodeIdx,
    pg_idx: usize,
    edge_idx: usize,
) -> NodeIdx {
    let (child_idx, child_key, parent_depth) = {
        let Some(parent) = store.get_node(parent_idx) else {
            return INVALID_IDX;
        };
        if parent.dead || pg_idx >= parent.piece_groups.len() {
            return INVALID_IDX;
        }
        let pg = &parent.piece_groups[pg_idx];
        if edge_idx >= pg.placements.len() {
            return INVALID_IDX;
        }
        let edge = &pg.placements[edge_idx];
        if edge.dead {
            return INVALID_IDX;
        }
        if edge.child != INVALID_IDX {
            return edge.child;
        }
        (edge.child, edge.child_key, parent.bags_deep)
    };

    if child_idx != INVALID_IDX {
        return child_idx;
    }

    let child_depth = if child_key.is_full_bag() {
        parent_depth + 1
    } else {
        parent_depth
    };
    let created_idx = store.get_or_create(child_key, child_depth);
    if created_idx == INVALID_IDX {
        return INVALID_IDX;
    }

    let child_dead = store.get_node(created_idx).is_some_and(|child| child.dead);

    if let Some(parent) = store.get_node_mut(parent_idx) {
        if pg_idx < parent.piece_groups.len() {
            let pg = &mut parent.piece_groups[pg_idx];
            if edge_idx < pg.placements.len() {
                pg.placements[edge_idx].child = created_idx;
                pg.placements[edge_idx].dead = child_dead;
            }
        }
    }

    add_predecessor(
        store,
        created_idx,
        PredecessorEdge {
            parent: parent_idx,
            piece_group: pg_idx,
            placement: edge_idx,
        },
    );

    created_idx
}

// ---------------------------------------------------------------------------
// In-memory store
// ---------------------------------------------------------------------------

struct MemStore {
    nodes: Vec<Node>,
    index: FxHashMap<StateKey, NodeIdx>,
    max_nodes: usize,
}

impl MemStore {
    fn new(max_nodes: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(max_nodes.min(10_000_000)),
            index: FxHashMap::default(),
            max_nodes,
        }
    }
}

impl GraphStore for MemStore {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn get_node(&self, idx: NodeIdx) -> Option<&Node> {
        self.nodes.get(idx as usize)
    }

    fn get_node_mut(&mut self, idx: NodeIdx) -> Option<&mut Node> {
        self.nodes.get_mut(idx as usize)
    }

    fn lookup(&self, key: &StateKey) -> Option<NodeIdx> {
        self.index.get(key).copied()
    }

    fn get_or_create(&mut self, key: StateKey, bags_deep: u16) -> NodeIdx {
        if let Some(&idx) = self.index.get(&key) {
            return idx;
        }
        if self.nodes.len() >= self.max_nodes {
            return INVALID_IDX;
        }
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(Node::new(key, bags_deep));
        self.index.insert(key, idx);
        idx
    }

    fn can_grow(&self) -> bool {
        self.nodes.len() < self.max_nodes
    }
}

// ---------------------------------------------------------------------------
// SQLite store
// ---------------------------------------------------------------------------

struct SqliteStore {
    /// In-memory node arena.
    nodes: Vec<Node>,
    /// Dedup index.
    index: FxHashMap<StateKey, NodeIdx>,
    /// SQLite connection for persistence.
    conn: Connection,
    /// Node indices modified since last flush.
    dirty: Vec<bool>,
    /// Max nodes.
    max_nodes: usize,
}

impl SqliteStore {
    fn new(path: &str, max_nodes: usize) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = OFF;
             PRAGMA cache_size = -128000;
             PRAGMA mmap_size = 2147483648;
             PRAGMA temp_store = MEMORY;
             PRAGMA page_size = 8192;
             PRAGMA wal_autocheckpoint = 10000;
             PRAGMA locking_mode = EXCLUSIVE;",
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                board BLOB NOT NULL,
                bag INTEGER NOT NULL,
                visits INTEGER NOT NULL DEFAULT 0,
                value_sum REAL NOT NULL DEFAULT 0.0,
                expanded INTEGER NOT NULL DEFAULT 0,
                bags_deep INTEGER NOT NULL DEFAULT 0
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_state ON nodes(board, bag);
            CREATE INDEX IF NOT EXISTS idx_nodes_fullbag ON nodes(bag, visits) WHERE bag = 127;",
        )?;

        // Load existing nodes into memory
        let mut nodes = Vec::new();
        let mut index = FxHashMap::default();

        let count: u64 = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        if count > 0 {
            let mut stmt = conn.prepare("SELECT id, board, bag, visits, value_sum, expanded, bags_deep FROM nodes ORDER BY id")?;
            let rows = stmt.query_map([], |row| {
                let id: u32 = row.get(0)?;
                let board_bytes: Vec<u8> = row.get(1)?;
                let bag: u8 = row.get(2)?;
                let visits: u32 = row.get(3)?;
                let value_sum: f64 = row.get(4)?;
                let expanded: bool = row.get(5)?;
                let bags_deep: u16 = row.get(6)?;
                Ok((
                    id,
                    board_bytes,
                    bag,
                    visits,
                    value_sum as f32,
                    expanded,
                    bags_deep,
                ))
            })?;

            for row in rows {
                let (id, board_bytes, bag, visits, value_sum, _expanded, bags_deep) = row?;
                let key = StateKey::from_board_bytes(&board_bytes, bag);
                let node = Node {
                    key,
                    visits,
                    value_sum,
                    expanded: false,
                    dead: false,
                    piece_groups: Vec::new(), // loaded separately if needed
                    predecessors: Vec::new(),
                    bags_deep,
                };
                let idx = nodes.len() as NodeIdx;
                assert_eq!(idx, id, "node IDs must be sequential");
                nodes.push(node);
                index.insert(key, idx);
            }

            eprintln!("loaded {count} nodes from {path}");
        }

        let n = nodes.len();
        Ok(Self {
            nodes,
            index,
            conn,
            dirty: vec![false; n],
            max_nodes,
        })
    }

    fn flush(&mut self) -> Result<()> {
        let dirty_count: usize = self.dirty.iter().filter(|&&d| d).count();
        if dirty_count == 0 {
            return Ok(());
        }

        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut insert = tx.prepare_cached(
                "INSERT OR REPLACE INTO nodes (id, board, bag, visits, value_sum, expanded, bags_deep)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;

            for (idx, node) in self.nodes.iter().enumerate() {
                if !self.dirty[idx] {
                    continue;
                }
                let board_bytes = node.key.board_bytes();
                insert.execute(params![
                    idx as u32,
                    &board_bytes[..],
                    u8::from(node.key.bag),
                    node.visits,
                    node.value_sum as f64,
                    node.expanded,
                    node.bags_deep,
                ])?;
            }
        }
        tx.commit()?;

        // Clear dirty flags
        for d in &mut self.dirty {
            *d = false;
        }

        eprintln!("  flushed {dirty_count} dirty nodes to sqlite");
        Ok(())
    }
}

impl GraphStore for SqliteStore {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn get_node(&self, idx: NodeIdx) -> Option<&Node> {
        self.nodes.get(idx as usize)
    }

    fn get_node_mut(&mut self, idx: NodeIdx) -> Option<&mut Node> {
        let i = idx as usize;
        if i < self.dirty.len() {
            self.dirty[i] = true;
        }
        self.nodes.get_mut(i)
    }

    fn lookup(&self, key: &StateKey) -> Option<NodeIdx> {
        self.index.get(key).copied()
    }

    fn get_or_create(&mut self, key: StateKey, bags_deep: u16) -> NodeIdx {
        if let Some(&idx) = self.index.get(&key) {
            return idx;
        }
        if self.nodes.len() >= self.max_nodes {
            return INVALID_IDX;
        }
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(Node::new(key, bags_deep));
        self.dirty.push(true);
        self.index.insert(key, idx);
        idx
    }

    fn can_grow(&self) -> bool {
        self.nodes.len() < self.max_nodes
    }

    fn flush(&mut self) -> Result<()> {
        SqliteStore::flush(self)
    }
}

// ---------------------------------------------------------------------------
// Board pool with reachability tracking
// ---------------------------------------------------------------------------

/// A board in the pool with its cycle-forming metrics.
#[derive(Clone)]
struct PoolBoard {
    board: TetrisBoard,
    /// Bitmask of first adversarial pieces from this full-bag board that have
    /// sampled a return to another pool board.
    covered_root_pieces: u8,
    /// How many times this board was used as a search root.
    starts: u32,
    /// How many times a search starting from this board reached another pool board.
    outgoing: u32,
    /// How many times a search starting elsewhere reached this board.
    incoming: u32,
    /// Cycle score: min(incoming, outgoing) — high means mutual reachability.
    cycle_score: f32,
}

struct McgsSearch {
    /// The board pool: boards we're testing for cycle membership.
    pool: Vec<PoolBoard>,
    /// Fast lookup: board → pool index.
    pool_index: FxHashMap<TetrisBoard, usize>,
    /// Total cycle closes detected.
    cycle_closes: u64,
    /// Reachability matrix: (source_pool_idx, target_pool_idx) → count.
    reachability: FxHashMap<(usize, usize), u32>,
    /// Hard-death and admissibility counters.
    stats: McgsStats,
}

impl McgsSearch {
    fn new() -> Self {
        let empty = TetrisBoard::EMPTY_BOARD;
        let mut pool_index = FxHashMap::default();
        pool_index.insert(empty, 0);

        Self {
            pool: vec![PoolBoard {
                board: empty,
                covered_root_pieces: 0,
                starts: 0,
                outgoing: 0,
                incoming: 0,
                cycle_score: 0.0,
            }],
            pool_index,
            cycle_closes: 0,
            reachability: FxHashMap::default(),
            stats: McgsStats::default(),
        }
    }

    fn pool_size(&self) -> usize {
        self.pool.len()
    }

    fn is_in_pool(&self, board: &TetrisBoard) -> bool {
        self.pool_index.contains_key(board)
    }

    /// Add a board to the pool if not already present.
    fn add_to_pool(&mut self, board: TetrisBoard, max_pool: usize) {
        if self.pool_index.contains_key(&board) || self.pool.len() >= max_pool {
            return;
        }
        let idx = self.pool.len();
        self.pool.push(PoolBoard {
            board,
            covered_root_pieces: 0,
            starts: 0,
            outgoing: 0,
            incoming: 0,
            cycle_score: 0.0,
        });
        self.pool_index.insert(board, idx);
    }

    /// Pick a start board from the pool, weighted by exploration need.
    /// Boards with low starts relative to cycle_score get priority.
    fn sample_start(&self, store: &dyn GraphStore, rng: &mut impl rand::Rng) -> TetrisBoard {
        if self.pool.len() <= 1 {
            return self.pool[0].board;
        }

        // Weight recurrent/cyclic boards up, but remove proven-dead starts.
        let weights: Vec<f32> = self
            .pool
            .iter()
            .map(|pb| {
                let key = StateKey::new(pb.board, TetrisPieceBagState::new());
                let Some(idx) = store.lookup(&key) else {
                    return 1.0 / (pb.starts as f32 + 1.0);
                };
                let Some(node) = store.get_node(idx) else {
                    return 0.0;
                };
                if node.dead {
                    return 0.0;
                }
                let recurrence = recurrence_score(node.visits);
                (pb.cycle_score + 1.0 + recurrence * 4.0) / (pb.starts as f32 + 1.0)
            })
            .collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return self.pool[0].board;
        }
        let mut r = rng.random_range(0.0f32..total);
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.pool[i].board;
            }
        }
        self.pool[self.pool.len() - 1].board
    }

    /// Record that a search from `source` reached `target`.
    fn record_reach(
        &mut self,
        source: TetrisBoard,
        target: TetrisBoard,
        root_piece: Option<TetrisPiece>,
    ) {
        if let (Some(&si), Some(&ti)) = (self.pool_index.get(&source), self.pool_index.get(&target))
        {
            if let Some(piece) = root_piece {
                self.pool[si].covered_root_pieces |= u8::from(piece);
            }
            self.pool[si].outgoing += 1;
            self.pool[ti].incoming += 1;
            *self.reachability.entry((si, ti)).or_insert(0) += 1;
            self.cycle_closes += 1;

            // Update cycle scores
            self.pool[si].cycle_score =
                (self.pool[si].incoming as f32).min(self.pool[si].outgoing as f32);
            self.pool[ti].cycle_score =
                (self.pool[ti].incoming as f32).min(self.pool[ti].outgoing as f32);
        }
    }

    fn closure_snapshot(&self, store: &dyn GraphStore) -> ClosureSnapshot {
        let mut snapshot = ClosureSnapshot::default();
        let all_piece_mask = all_piece_mask();

        for pb in &self.pool {
            let key = StateKey::new(pb.board, TetrisPieceBagState::new());
            let is_alive_core = store
                .lookup(&key)
                .and_then(|idx| store.get_node(idx))
                .is_some_and(|node| !node.dead);
            if !is_alive_core {
                continue;
            }

            let covered = (pb.covered_root_pieces & all_piece_mask).count_ones() as u64;
            let required = TetrisPiece::NUM_PIECES as u64;
            let open = required.saturating_sub(covered);
            let weight = u64::from(pb.starts.max(1));

            snapshot.core_states += 1;
            snapshot.covered += covered;
            snapshot.required += required;
            snapshot.open += open;
            snapshot.weighted_covered += covered * weight;
            snapshot.weighted_required += required * weight;
        }

        snapshot
    }

    /// Discover new pool candidates from the graph's most-visited full-bag nodes.
    fn discover_candidates(&mut self, store: &dyn GraphStore, min_visits: u32, max_pool: usize) {
        let mut full_bag_nodes: Vec<(TetrisBoard, u32)> = Vec::new();
        for i in 0..store.node_count() {
            if let Some(n) = store.get_node(i as NodeIdx) {
                if n.key.is_full_bag()
                    && !n.dead
                    && n.visits >= min_visits
                    && !self.pool_index.contains_key(&n.key.board)
                {
                    full_bag_nodes.push((n.key.board, n.visits));
                }
            }
        }
        full_bag_nodes.sort_by(|a, b| b.1.cmp(&a.1));

        let old_size = self.pool.len();
        for (board, _) in full_bag_nodes
            .into_iter()
            .take(max_pool.saturating_sub(self.pool.len()))
        {
            self.add_to_pool(board, max_pool);
        }
        if self.pool.len() != old_size {
            eprintln!(
                "  pool: {} boards (+{})",
                self.pool.len(),
                self.pool.len() - old_size
            );
        }
    }

    /// Prune pool boards with zero cycle potential after enough exploration.
    fn prune_pool(&mut self, store: &dyn GraphStore, min_starts: u32) {
        let to_remove: Vec<usize> = self
            .pool
            .iter()
            .enumerate()
            .filter(|(i, pb)| {
                if *i == 0 {
                    return false;
                }
                let key = StateKey::new(pb.board, TetrisPieceBagState::new());
                let node_dead = store
                    .lookup(&key)
                    .and_then(|idx| store.get_node(idx))
                    .is_some_and(|node| node.dead);
                node_dead || (pb.starts >= min_starts && pb.incoming == 0 && pb.outgoing == 0)
            })
            .map(|(i, _)| i)
            .collect();

        if to_remove.is_empty() {
            return;
        }

        // Remove from back to front to preserve indices
        for &idx in to_remove.iter().rev() {
            let board = self.pool[idx].board;
            self.pool_index.remove(&board);
            self.pool.swap_remove(idx);
            // Fix swapped index
            if idx < self.pool.len() {
                self.pool_index.insert(self.pool[idx].board, idx);
            }
        }
        eprintln!(
            "  pruned {} dead boards, pool now {}",
            to_remove.len(),
            self.pool.len()
        );
    }

    /// Run one MCGS iteration from a sampled start board.
    fn run_iteration(
        &mut self,
        store: &mut dyn GraphStore,
        start_board: TetrisBoard,
        exploration: f32,
        recurrence_weight: f32,
        novelty_cost: f32,
        max_bags: usize,
        max_height: u32,
        max_holes: u32,
        max_roughness: u32,
        prior_temperature: f32,
    ) {
        // Record that we started from this board
        if let Some(&si) = self.pool_index.get(&start_board) {
            self.pool[si].starts += 1;
        }

        // Find or create root node for this start board
        let root_key = StateKey::new(start_board, TetrisPieceBagState::new());
        let root_idx = store.get_or_create(root_key, 0);
        if root_idx == INVALID_IDX {
            return;
        }

        let mut path: Vec<(NodeIdx, usize, usize)> = Vec::with_capacity(64);
        let mut current = root_idx;
        // Track piece steps in THIS iteration's path (not stored on nodes).
        // Each AND→OR→AND transition = 1 piece placed.
        // A full bag = 7 pieces. We only count pool hits after >= 7 steps.
        let mut pieces_placed: usize = 0;
        let mut root_piece: Option<TetrisPiece> = None;

        let leaf_value = loop {
            let (node_bag, node_dead) = match store.get_node(current) {
                Some(n) => (n.key.bag, n.dead),
                None => break 0.0,
            };
            if node_dead {
                break DEAD_VALUE;
            }

            // Pool hit: only after at least 7 pieces (one full bag traversal)
            // and only at full-bag boundaries (bag just refilled).
            if pieces_placed >= 7 && node_bag == TetrisPieceBagState::new() {
                if let Some(n) = store.get_node(current) {
                    if self.is_in_pool(&n.key.board) && n.key.board != start_board {
                        let target = n.key.board;
                        self.record_reach(start_board, target, root_piece);
                        break 1.0; // real cycle reward
                    }
                }
            }

            // Depth limit (in pieces, not bags)
            if pieces_placed >= max_bags * 7 {
                break leaf_value(store, current, recurrence_weight, novelty_cost);
            }

            // Expand if needed
            let needs_expand = store.get_node(current).is_some_and(|n| !n.expanded);
            if needs_expand {
                if store.can_grow() {
                    let summary = expand(
                        store,
                        current,
                        max_height,
                        max_holes,
                        max_roughness,
                        prior_temperature,
                    );
                    self.stats.lost_placements += summary.lost_placements;
                    self.stats.condition_violations += summary.condition_violations;
                    self.stats.hard_dead_piece_groups += summary.immediate_dead_piece_groups;
                    refresh_node_deadness(store, current, &mut self.stats);
                } else {
                    self.stats.node_cap_hits += 1;
                    break NODE_CAP_VALUE;
                }
            }

            let node = match store.get_node(current) {
                Some(n) => n,
                None => break -1.0,
            };

            if node.dead {
                break DEAD_VALUE;
            }
            if node.piece_groups.is_empty() {
                break DEAD_VALUE;
            }
            if node.piece_groups.iter().any(|pg| pg.dead) {
                mark_state_dead(store, current, &mut self.stats);
                break DEAD_VALUE;
            }

            // AND selection (adversary picks piece)
            let sqrt_pv = (node.visits.max(1) as f32).sqrt();
            let mut best_pg = 0usize;
            let mut best_and_ucb = f32::NEG_INFINITY;
            for (i, pg) in node.piece_groups.iter().enumerate() {
                let q = if pg.visits > 0 {
                    pg.value_sum / pg.visits as f32
                } else {
                    0.0
                };
                let ucb = -q + exploration * pg.prior * sqrt_pv / (1.0 + pg.visits as f32);
                if ucb > best_and_ucb {
                    best_and_ucb = ucb;
                    best_pg = i;
                }
            }

            // Progressive widening
            promote_pending(store, current, best_pg);
            sync_piece_group_dead_edges(store, current, best_pg, &mut self.stats);

            if store.get_node(current).is_some_and(|node| node.dead) {
                break DEAD_VALUE;
            }

            let node = match store.get_node(current) {
                Some(n) => n,
                None => break -1.0,
            };

            let pg = &node.piece_groups[best_pg];
            if pieces_placed == 0 && current == root_idx {
                root_piece = Some(pg.piece);
            }
            if pg.dead || !piece_group_has_possible_response(pg) {
                mark_state_dead(store, current, &mut self.stats);
                break DEAD_VALUE;
            }
            let pg_sqrt = (pg.visits.max(1) as f32).sqrt();
            let mut best_pl = None;
            let mut best_or_ucb = f32::NEG_INFINITY;
            for (i, e) in pg.placements.iter().enumerate() {
                if e.dead {
                    continue;
                }
                let q = if e.visits > 0 {
                    e.value_sum / e.visits as f32
                } else {
                    0.0
                };
                let recurrence_adjustment =
                    edge_value_adjustment(store, e, recurrence_weight, novelty_cost);
                let ucb = q
                    + recurrence_adjustment
                    + exploration * e.prior * pg_sqrt / (1.0 + e.visits as f32);
                if ucb > best_or_ucb {
                    best_or_ucb = ucb;
                    best_pl = Some(i);
                }
            }

            let Some(best_pl) = best_pl else {
                refresh_node_deadness(store, current, &mut self.stats);
                if store.get_node(current).is_some_and(|node| node.dead) {
                    break DEAD_VALUE;
                }
                break leaf_value(store, current, recurrence_weight, novelty_cost);
            };

            let child_idx = materialize_edge_child(store, current, best_pg, best_pl);

            if child_idx == INVALID_IDX {
                path.push((current, best_pg, best_pl));
                self.stats.node_cap_hits += 1;
                break NODE_CAP_VALUE;
            }

            if store.get_node(child_idx).is_some_and(|child| child.dead) {
                mark_edge_dead(store, current, best_pg, best_pl, &mut self.stats);
                path.push((current, best_pg, best_pl));
                break DEAD_VALUE;
            }

            path.push((current, best_pg, best_pl));
            current = child_idx;
            pieces_placed += 1; // one AND+OR step = one piece placed
        };

        // Backup
        if let Some(node) = store.get_node_mut(current) {
            node.visits += 1;
            node.value_sum += leaf_value;
        }
        for &(ni, pgi, pli) in path.iter().rev() {
            if let Some(node) = store.get_node_mut(ni) {
                node.visits += 1;
                node.value_sum += leaf_value;
                if pgi < node.piece_groups.len() {
                    let pg = &mut node.piece_groups[pgi];
                    pg.visits += 1;
                    pg.value_sum += leaf_value;
                    if pli < pg.placements.len() {
                        pg.placements[pli].visits += 1;
                        pg.placements[pli].value_sum += leaf_value;
                    }
                }
            }
        }
    }
}

/// Promote pending placements to active edges if the piece group has enough visits.
fn promote_pending(store: &mut dyn GraphStore, node_idx: NodeIdx, pg_idx: usize) {
    // Extract what we need from the node without holding the borrow
    let (visits, n_active, n_live_active, has_pending) = {
        let node = match store.get_node(node_idx) {
            Some(n) => n,
            None => return,
        };
        if pg_idx >= node.piece_groups.len() {
            return;
        }
        let pg = &node.piece_groups[pg_idx];
        (
            pg.visits,
            pg.placements.len(),
            pg.placements.iter().filter(|edge| !edge.dead).count(),
            !pg.pending.is_empty(),
        )
    };

    let target_width = progressive_width(visits);
    if n_live_active >= target_width || !has_pending {
        return;
    }

    // Collect pending items to promote
    let to_promote: Vec<PendingPlacement> = {
        let node = match store.get_node(node_idx) {
            Some(n) => n,
            None => return,
        };
        let pg = &node.piece_groups[pg_idx];
        let n_to_add = (target_width - n_live_active).min(pg.pending.len());
        pg.pending[..n_to_add].to_vec()
    };

    let mut new_edges: Vec<Edge> = Vec::with_capacity(to_promote.len());
    for pp in &to_promote {
        new_edges.push(Edge {
            child: INVALID_IDX,
            child_key: pp.child_key,
            visits: 0,
            value_sum: 0.0,
            prior: pp.prior,
            dead: false,
        });
    }

    // Now add edges and remove from pending
    if let Some(node) = store.get_node_mut(node_idx) {
        if pg_idx < node.piece_groups.len() {
            let pg = &mut node.piece_groups[pg_idx];
            let n_promoted = new_edges.len();
            pg.placements.extend(new_edges);
            pg.pending.drain(..n_promoted);
        }
    }
}

fn expand(
    store: &mut dyn GraphStore,
    node_idx: NodeIdx,
    max_height: u32,
    max_holes: u32,
    max_roughness: u32,
    prior_temperature: f32,
) -> ExpansionSummary {
    let key = match store.get_node(node_idx) {
        Some(n) => n.key,
        None => return ExpansionSummary::default(),
    };
    let mut summary = ExpansionSummary::default();

    let n_pieces = key.bag.count() as f32;
    let and_prior = 1.0 / n_pieces.max(1.0);

    let mut piece_groups: Vec<PieceGroup> = Vec::with_capacity(key.bag.count() as usize);

    for (piece, next_bag) in key.bag.iter_next_states() {
        let placements = TetrisPiecePlacement::all_from_piece(piece);
        let mut candidates: Vec<(TetrisPiecePlacement, TetrisBoard, TetrisPieceBagState, f32)> =
            Vec::new();
        let mut invalid_placements = 0u16;

        for &placement in placements {
            let mut next_board = key.board;
            let result = next_board.apply_piece_placement(placement);
            if result.is_lost == IsLost::LOST {
                summary.lost_placements += 1;
                invalid_placements += 1;
                continue;
            }
            if !board_ok(&next_board, max_height, max_holes, max_roughness) {
                summary.condition_violations += 1;
                invalid_placements += 1;
                continue;
            }
            candidates.push((placement, next_board, next_bag, score_board(&next_board)));
        }

        if candidates.is_empty() {
            summary.immediate_dead_piece_groups += 1;
            piece_groups.push(PieceGroup {
                piece,
                visits: 0,
                value_sum: 0.0,
                prior: and_prior,
                dead: true,
                total_placements: placements.len() as u16,
                invalid_placements,
                placements: Vec::new(),
                pending: Vec::new(),
            });
            continue;
        }

        // Sort by cost ascending (best first)
        candidates.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax priors
        let min_cost = candidates[0].3;
        let logits: Vec<f32> = candidates
            .iter()
            .map(|c| -(c.3 - min_cost) / prior_temperature)
            .collect();
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();

        // Materialize top PW_MIN edges, keep rest as pending
        let initial_width = PW_MIN.min(candidates.len());
        let mut edges = Vec::with_capacity(initial_width);
        let mut pending = Vec::with_capacity(candidates.len().saturating_sub(initial_width));

        for (i, (_, next_board, nb, _)) in candidates.iter().enumerate() {
            let prior = ((logits[i] - max_l).exp() / exp_sum).max(1e-6);
            let child_key = StateKey::new(*next_board, *nb);

            if i < initial_width {
                edges.push(Edge {
                    child: INVALID_IDX,
                    child_key,
                    visits: 0,
                    value_sum: 0.0,
                    prior,
                    dead: false,
                });
            } else {
                pending.push(PendingPlacement { child_key, prior });
            }
        }

        piece_groups.push(PieceGroup {
            piece,
            visits: 0,
            value_sum: 0.0,
            prior: and_prior,
            dead: false,
            total_placements: placements.len() as u16,
            invalid_placements,
            placements: edges,
            pending,
        });
    }

    if let Some(node) = store.get_node_mut(node_idx) {
        node.expanded = true;
        node.piece_groups = piece_groups;
    }

    summary
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("tetris_mcgs (Monte Carlo Graph Search cycle finder)");
    println!("max_height         = {}", cli.max_height);
    println!("max_holes          = {}", cli.max_holes);
    println!("max_roughness      = {}", cli.max_roughness);
    println!("iterations         = {}", cli.iterations);
    println!("exploration        = {}", cli.exploration);
    println!("recurrence_weight  = {}", cli.recurrence_weight);
    println!("novelty_cost       = {}", cli.novelty_cost);
    println!("prior_temperature  = {}", cli.prior_temperature);
    println!("max_bags           = {}", cli.max_bags);
    if cli.db.is_some() {
        println!(
            "storage            = sqlite ({})",
            cli.db.as_deref().unwrap_or("?")
        );
    } else {
        println!(
            "storage            = in-memory (max_nodes={})",
            cli.max_nodes
        );
    }
    println!("cycle_visit_thresh = {}", cli.cycle_visit_threshold);
    println!("candidate_refresh  = {}", cli.candidate_refresh);
    println!("max_candidates     = {}", cli.max_candidates);
    println!("db_flush_every     = {}", cli.db_flush_every);
    println!("verify_every       = {}", cli.verify_every);
    println!("verify_log_every   = {}", cli.verify_log_every);
    println!("seed               = {}", cli.seed);
    println!();

    let t0 = Instant::now();
    let shutdown = AtomicBool::new(false);
    {
        let p = &shutdown as *const AtomicBool as usize;
        let _ = ctrlc::set_handler(move || {
            let s = unsafe { &*(p as *const AtomicBool) };
            eprintln!("\nShutting down...");
            s.store(true, AtomicOrdering::Relaxed);
        });
    }

    let mut search = McgsSearch::new();

    if let Some(ref db_path) = cli.db {
        // SQLite mode
        let mut store = SqliteStore::new(db_path, cli.max_nodes)?;
        if store.node_count() == 0 {
            store.get_or_create(StateKey::empty(), 0);
        }

        run_main_loop(&mut search, &mut store, &cli, &t0, &shutdown);
        store.flush()?;
        print_results(&store, &search, &t0);
    } else {
        let mut store = MemStore::new(cli.max_nodes);
        store.get_or_create(StateKey::empty(), 0);

        run_main_loop(&mut search, &mut store, &cli, &t0, &shutdown);
        print_results(&store, &search, &t0);
    }

    Ok(())
}

fn run_main_loop(
    search: &mut McgsSearch,
    store: &mut dyn GraphStore,
    cli: &Cli,
    t0: &Instant,
    shutdown: &AtomicBool,
) {
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(cli.seed);
    let mut cycle_found = false;
    let mut running_verification: Option<RunningVerification> = None;

    for iter in 1..=cli.iterations {
        if shutdown.load(AtomicOrdering::Relaxed) {
            eprintln!("interrupted at iteration {iter}");
            break;
        }

        if let Some(report) = poll_verification(&mut running_verification) {
            if report_verification(&report) {
                println!(
                    "*** CYCLE FOUND — ALL {} CANDIDATES VERIFIED ***",
                    report.candidates
                );
                cycle_found = true;
                break;
            }
        }

        // Discover new pool candidates from graph
        if cli.candidate_refresh > 0 && iter % cli.candidate_refresh == 0 {
            search.discover_candidates(store, cli.cycle_visit_threshold, cli.max_candidates);
        }

        // Prune dead pool boards periodically
        if cli.candidate_refresh > 0 && iter % (cli.candidate_refresh * 10) == 0 {
            search.prune_pool(store, 50);
        }

        // Sample a start board from the weighted pool
        let start = search.sample_start(store, &mut rng);

        search.run_iteration(
            store,
            start,
            cli.exploration,
            cli.recurrence_weight,
            cli.novelty_cost,
            cli.max_bags,
            cli.max_height,
            cli.max_holes,
            cli.max_roughness,
            cli.prior_temperature,
        );

        if cli.db_flush_every > 0 && iter % cli.db_flush_every == 0 {
            if let Err(err) = store.flush() {
                eprintln!("  [db] flush failed at iter {iter}: {err:#}");
            }
        }

        // Logging
        if cli.log_every > 0 && iter % cli.log_every == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let n_edges = search.reachability.len();

            // Find best cycle score in pool
            let best_cs = search
                .pool
                .iter()
                .map(|pb| pb.cycle_score)
                .fold(0.0f32, f32::max);
            let closure = search.closure_snapshot(store);

            println!(
                "[{iter:>8}] nodes={:>8} pool={} edges={n_edges} cycles={} best_cs={best_cs:.0} closure={:.3} wclosure={:.3} covered={}/{} open={} core={} dead_s={} dead_e={} cond={} rate={:.0}/s t={elapsed:.1}s",
                store.node_count(),
                search.pool_size(),
                search.cycle_closes,
                closure.closure(),
                closure.weighted_closure(),
                closure.covered,
                closure.required,
                closure.open,
                closure.core_states,
                search.stats.hard_dead_states,
                search.stats.hard_dead_edges,
                search.stats.condition_violations,
                iter as f64 / elapsed,
            );
        }

        // Periodic verification of most promising subgraph. This is intentionally
        // non-blocking: verification runs on a snapshot while MCGS keeps sampling.
        if cli.verify_every > 0 && iter % cli.verify_every == 0 {
            if running_verification.is_some() {
                if let Some(verifier) = &running_verification {
                    println!(
                        "  [verify] previous verifier from iter {} still running; skipping new launch at iter {iter}",
                        verifier.started_iter
                    );
                }
            } else {
                // Find boards with cycle_score > 0 (mutual reachability)
                let cycle_candidates: Vec<TetrisBoard> = search
                    .pool
                    .iter()
                    .filter(|pb| pb.cycle_score > 0.0)
                    .map(|pb| pb.board)
                    .collect();

                if !cycle_candidates.is_empty() {
                    println!();
                    println!(
                        "=== Launching background verification of {} cycle candidates at iter {iter} ===",
                        cycle_candidates.len()
                    );

                    running_verification = Some(launch_verification(
                        cycle_candidates,
                        iter,
                        cli.max_height,
                        cli.max_holes,
                        cli.max_roughness,
                        cli.verify_log_every,
                    ));
                    println!("  [verify] sampling continues while verification runs");
                    println!();
                }
            }
        }
    }

    if !cycle_found {
        if let Some(report) = poll_verification(&mut running_verification) {
            if report_verification(&report) {
                println!(
                    "*** CYCLE FOUND — ALL {} CANDIDATES VERIFIED ***",
                    report.candidates
                );
                cycle_found = true;
            }
        } else if let Some(verifier) = &running_verification {
            println!(
                "  [verify] verifier from iter {} still running; not waiting for it at shutdown",
                verifier.started_iter
            );
        }
    }

    // Final report: top pool boards by cycle score
    let mut ranked: Vec<&PoolBoard> = search.pool.iter().collect();
    ranked.sort_by(|a, b| {
        b.cycle_score
            .partial_cmp(&a.cycle_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!();
    println!("top pool boards by cycle score:");
    for (i, pb) in ranked.iter().take(15).enumerate() {
        println!(
            "  [{i}] cells={} h={} rough={} starts={} in={} out={} cs={:.0}",
            pb.board.count(),
            pb.board.height(),
            pb.board.roughness(),
            pb.starts,
            pb.incoming,
            pb.outgoing,
            pb.cycle_score,
        );
    }

    if cycle_found {
        println!("\nInfinite play PROVEN for the verified cycle boards.");
    }
}

fn print_results(store: &dyn GraphStore, search: &McgsSearch, t0: &Instant) {
    let elapsed = t0.elapsed().as_secs_f64();
    let root_val = store.get_node(0).map_or(0.0, |n| n.avg_value());
    let graph_health = graph_health_counts(store);
    let closure = search.closure_snapshot(store);

    println!();
    println!("--- done ---");
    println!("graph_nodes        = {}", store.node_count());
    println!("root_value         = {root_val:+.4}");
    println!("cycle_closes       = {}", search.cycle_closes);
    println!("pool_size          = {}", search.pool_size());
    println!("reachability_edges = {}", search.reachability.len());
    println!("closure            = {:.4}", closure.closure());
    println!("weighted_closure   = {:.4}", closure.weighted_closure());
    println!(
        "closure_covered    = {}/{}",
        closure.covered, closure.required
    );
    println!("closure_open       = {}", closure.open);
    println!("closure_core       = {}", closure.core_states);
    println!("dead_states        = {}", graph_health.dead_states);
    println!("dead_edges         = {}", graph_health.dead_edges);
    println!("dead_piece_groups  = {}", graph_health.dead_piece_groups);
    println!(
        "placement_slots    = {}",
        graph_health.total_placement_slots
    );
    println!(
        "invalid_slots      = {}",
        graph_health.invalid_placement_slots
    );
    println!("lost_placements    = {}", search.stats.lost_placements);
    println!("condition_violated = {}", search.stats.condition_violations);
    println!("node_cap_hits      = {}", search.stats.node_cap_hits);
    println!("total_time         = {elapsed:.1}s");
}

#[derive(Default)]
struct GraphHealthCounts {
    dead_states: u64,
    dead_edges: u64,
    dead_piece_groups: u64,
    total_placement_slots: u64,
    invalid_placement_slots: u64,
}

fn graph_health_counts(store: &dyn GraphStore) -> GraphHealthCounts {
    let mut counts = GraphHealthCounts::default();
    for idx in 0..store.node_count() {
        let Some(node) = store.get_node(idx as NodeIdx) else {
            continue;
        };
        if node.dead {
            counts.dead_states += 1;
        }
        for pg in &node.piece_groups {
            if pg.dead {
                counts.dead_piece_groups += 1;
            }
            counts.total_placement_slots += u64::from(pg.total_placements);
            counts.invalid_placement_slots += u64::from(pg.invalid_placements);
            counts.dead_edges += pg.placements.iter().filter(|edge| edge.dead).count() as u64;
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// Background verification
// ---------------------------------------------------------------------------

struct RunningVerification {
    started_iter: u64,
    receiver: Receiver<VerificationReport>,
    handle: JoinHandle<()>,
}

struct VerificationReport {
    started_iter: u64,
    candidates: usize,
    results: Vec<VerifyResult>,
    elapsed_secs: f64,
}

fn launch_verification(
    candidates: Vec<TetrisBoard>,
    started_iter: u64,
    max_height: u32,
    max_holes: u32,
    max_roughness: u32,
    log_every: u64,
) -> RunningVerification {
    let candidate_count = candidates.len();
    let (sender, receiver) = mpsc::channel();
    let handle = thread::spawn(move || {
        let started_at = Instant::now();
        let results = verify_candidates_full_bag(
            &candidates,
            max_height,
            max_holes,
            max_roughness,
            log_every,
        );
        let elapsed_secs = started_at.elapsed().as_secs_f64();
        let _ = sender.send(VerificationReport {
            started_iter,
            candidates: candidate_count,
            results,
            elapsed_secs,
        });
    });

    RunningVerification {
        started_iter,
        receiver,
        handle,
    }
}

fn poll_verification(running: &mut Option<RunningVerification>) -> Option<VerificationReport> {
    let verifier = running.as_ref()?;
    match verifier.receiver.try_recv() {
        Ok(report) => {
            let verifier = running.take().expect("verification existed");
            let _ = verifier.handle.join();
            Some(report)
        }
        Err(TryRecvError::Empty) => None,
        Err(TryRecvError::Disconnected) => {
            let verifier = running.take().expect("verification existed");
            let _ = verifier.handle.join();
            eprintln!(
                "  [verify] background verifier from iter {} exited without a report",
                verifier.started_iter
            );
            None
        }
    }
}

fn report_verification(report: &VerificationReport) -> bool {
    println!();
    println!(
        "=== Verification result from iter {}: {} candidates in {:.1}s ===",
        report.started_iter, report.candidates, report.elapsed_secs
    );

    let mut all_pass = true;
    for vr in &report.results {
        let total = vr.perms_ok + vr.perms_fail;
        let pct = if total == 0 {
            0.0
        } else {
            vr.perms_ok as f32 / total as f32 * 100.0
        };
        if vr.perms_fail == 0 {
            println!(
                "  PASS cells={} h={} rough={} {}/{total}",
                vr.board.count(),
                vr.board.height(),
                vr.board.roughness(),
                vr.perms_ok,
            );
        } else {
            println!(
                "  FAIL cells={} h={} rough={} {}/{total} ({pct:.1}%)",
                vr.board.count(),
                vr.board.height(),
                vr.board.roughness(),
                vr.perms_ok,
            );
            all_pass = false;
        }
    }
    println!();

    all_pass
}

// ---------------------------------------------------------------------------
// Verification: check if candidate boards form a closed cycle
// ---------------------------------------------------------------------------

/// Full bag-level verification: from a candidate board, for each of 5040
/// permutations, can the player place all 7 pieces and land on another candidate?
fn verify_candidates_full_bag(
    candidates: &[TetrisBoard],
    max_height: u32,
    max_holes: u32,
    max_roughness: u32,
    log_every: u64,
) -> Vec<VerifyResult> {
    use rayon::prelude::*;
    use rustc_hash::FxHashSet;

    // Generate all 5040 permutations
    let perms = generate_permutations();
    let target_hashes: FxHashSet<u64> = candidates.iter().map(|b| board_hash_fn(b)).collect();
    let total_checks = candidates.len() as u64 * perms.len() as u64;
    let started_at = Instant::now();
    let completed_checks = AtomicU64::new(0);
    let completed_candidates = AtomicU64::new(0);

    if log_every > 0 {
        println!(
            "  [verify] total_perm_checks={total_checks} candidates={} perms_per_candidate={} log_every={log_every}",
            candidates.len(),
            perms.len(),
        );
    }

    candidates
        .par_iter()
        .enumerate()
        .map(|(candidate_idx, &board)| {
            if log_every > 0 {
                println!(
                    "  [verify] candidate {}/{} start cells={} h={} rough={}",
                    candidate_idx + 1,
                    candidates.len(),
                    board.count(),
                    board.height(),
                    board.roughness(),
                );
            }

            let mut perms_ok = 0u32;
            let mut perms_fail = 0u32;

            for perm in &perms {
                if can_reach_any_target(
                    &board,
                    perm,
                    0,
                    max_height,
                    max_holes,
                    max_roughness,
                    &target_hashes,
                ) {
                    perms_ok += 1;
                } else {
                    perms_fail += 1;
                }

                if log_every > 0 {
                    let done = completed_checks.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    if done % log_every == 0 || done == total_checks {
                        let elapsed = started_at.elapsed().as_secs_f64().max(1e-9);
                        let pct = done as f64 / total_checks as f64 * 100.0;
                        println!(
                            "  [verify] progress {done}/{total_checks} ({pct:.1}%) candidates_done={} rate={:.0}/s t={elapsed:.1}s",
                            completed_candidates.load(AtomicOrdering::Relaxed),
                            done as f64 / elapsed,
                        );
                    }
                }
            }

            let finished = completed_candidates.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if log_every > 0 {
                let elapsed = started_at.elapsed().as_secs_f64();
                let total = perms_ok + perms_fail;
                let pct = if total == 0 {
                    0.0
                } else {
                    perms_ok as f32 / total as f32 * 100.0
                };
                println!(
                    "  [verify] candidate {}/{} done ok={perms_ok}/{total} ({pct:.1}%) finished={finished}/{} t={elapsed:.1}s",
                    candidate_idx + 1,
                    candidates.len(),
                    candidates.len(),
                );
            }

            VerifyResult {
                board,
                perms_ok,
                perms_fail,
            }
        })
        .collect()
}

struct VerifyResult {
    board: TetrisBoard,
    perms_ok: u32,
    perms_fail: u32,
}

/// DFS: can placing pieces[step..] from `board` reach any target board?
fn can_reach_any_target(
    board: &TetrisBoard,
    pieces: &[tetris_game::TetrisPiece; 7],
    step: usize,
    max_height: u32,
    max_holes: u32,
    max_roughness: u32,
    targets: &rustc_hash::FxHashSet<u64>,
) -> bool {
    if step == 7 {
        return targets.contains(&board_hash_fn(board));
    }

    let piece = pieces[step];
    for &placement in TetrisPiecePlacement::all_from_piece(piece) {
        let mut next = *board;
        let result = next.apply_piece_placement(placement);
        if result.is_lost == IsLost::LOST {
            continue;
        }
        if !board_ok(&next, max_height, max_holes, max_roughness) {
            continue;
        }
        if can_reach_any_target(
            &next,
            pieces,
            step + 1,
            max_height,
            max_holes,
            max_roughness,
            targets,
        ) {
            return true;
        }
    }
    false
}

fn board_hash_fn(board: &TetrisBoard) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    board.hash(&mut hasher);
    hasher.finish()
}

fn generate_permutations() -> Vec<[tetris_game::TetrisPiece; 7]> {
    let pieces = tetris_game::TetrisPiece::all();
    let mut perms = Vec::with_capacity(5040);
    let mut indices: [usize; 7] = [0, 1, 2, 3, 4, 5, 6];
    let mut c = [0usize; 7];
    perms.push(std::array::from_fn(|i| pieces[indices[i]]));
    let mut i = 0;
    while i < 7 {
        if c[i] < i {
            if i % 2 == 0 {
                indices.swap(0, i);
            } else {
                indices.swap(c[i], i);
            }
            perms.push(std::array::from_fn(|j| pieces[indices[j]]));
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    perms
}

// ---------------------------------------------------------------------------
// BagExt
// ---------------------------------------------------------------------------

trait BagExt {
    fn is_full(&self) -> bool;
}

impl BagExt for TetrisPieceBagState {
    fn is_full(&self) -> bool {
        *self == TetrisPieceBagState::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_game::TetrisPiece;

    fn make_mem_store() -> MemStore {
        let mut store = MemStore::new(100_000);
        store.get_or_create(StateKey::empty(), 0);
        store
    }

    #[test]
    fn state_key_dedup() {
        let k1 = StateKey::empty();
        let k2 = StateKey::new(TetrisBoard::EMPTY_BOARD, TetrisPieceBagState::new());
        assert_eq!(k1, k2);
        assert!(k1.is_full_bag());
    }

    #[test]
    fn state_key_different_bags() {
        let mut bag2 = TetrisPieceBagState::new();
        bag2.remove(TetrisPiece::all()[0]);
        let k1 = StateKey::empty();
        let k2 = StateKey::new(TetrisBoard::EMPTY_BOARD, bag2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn mem_store_dedup() {
        let mut store = make_mem_store();
        let idx1 = store.get_or_create(StateKey::empty(), 0);
        let idx2 = store.get_or_create(StateKey::empty(), 0);
        assert_eq!(idx1, idx2);
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn mem_store_cap() {
        let mut store = MemStore::new(5);
        for i in 0..10 {
            let mut board = TetrisBoard::EMPTY_BOARD;
            if i > 0 {
                let p = TetrisPiecePlacement::all_from_piece(TetrisPiece::all()[i % 7])[0];
                let _ = board.apply_piece_placement(p);
            }
            store.get_or_create(StateKey::new(board, TetrisPieceBagState::new()), 0);
        }
        assert!(store.node_count() <= 5);
    }

    #[test]
    fn expand_root_creates_7_groups() {
        let mut store = make_mem_store();
        expand(&mut store, 0, 6, 3, u32::MAX, 2.0);
        let root = store.get_node(0).unwrap();
        assert!(root.expanded);
        assert_eq!(root.piece_groups.len(), 7);
        for pg in &root.piece_groups {
            assert!(!pg.placements.is_empty());
        }
    }

    #[test]
    fn expand_keeps_child_nodes_lazy() {
        let mut store = make_mem_store();
        expand(&mut store, 0, 6, 3, u32::MAX, 2.0);

        let root = store.get_node(0).unwrap();
        assert_eq!(store.node_count(), 1);
        assert!(
            root.piece_groups
                .iter()
                .flat_map(|pg| &pg.placements)
                .all(|edge| edge.child == INVALID_IDX)
        );

        let child = materialize_edge_child(&mut store, 0, 0, 0);
        assert_ne!(child, INVALID_IDX);
        assert_eq!(store.node_count(), 2);
    }

    #[test]
    fn condition_violation_marks_state_dead() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();
        search.run_iteration(
            &mut store,
            TetrisBoard::EMPTY_BOARD,
            1.4,
            0.25,
            0.10,
            1,
            0,
            0,
            u32::MAX,
            2.0,
        );

        let root = store.get_node(0).unwrap();
        assert!(root.dead);
        assert!(root.piece_groups.iter().all(|pg| pg.dead));
        assert!(search.stats.condition_violations > 0);
        assert!(search.stats.hard_dead_states > 0);
    }

    #[test]
    fn dead_child_backprops_to_parent_edge() {
        let mut store = make_mem_store();
        let mut child_board = TetrisBoard::EMPTY_BOARD;
        let _ = child_board
            .apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0]);
        let child_idx =
            store.get_or_create(StateKey::new(child_board, TetrisPieceBagState::new()), 0);

        {
            let parent = store.get_node_mut(0).unwrap();
            parent.expanded = true;
            parent.piece_groups = vec![PieceGroup {
                piece: TetrisPiece::O_PIECE,
                visits: 0,
                value_sum: 0.0,
                prior: 1.0,
                dead: false,
                total_placements: 1,
                invalid_placements: 0,
                placements: vec![Edge {
                    child: child_idx,
                    child_key: StateKey::new(child_board, TetrisPieceBagState::new()),
                    visits: 0,
                    value_sum: 0.0,
                    prior: 1.0,
                    dead: false,
                }],
                pending: Vec::new(),
            }];
        }
        add_predecessor(
            &mut store,
            child_idx,
            PredecessorEdge {
                parent: 0,
                piece_group: 0,
                placement: 0,
            },
        );

        let mut stats = McgsStats::default();
        mark_state_dead(&mut store, child_idx, &mut stats);

        let parent = store.get_node(0).unwrap();
        assert!(parent.dead);
        assert!(parent.piece_groups[0].dead);
        assert!(parent.piece_groups[0].placements[0].dead);
        assert_eq!(stats.hard_dead_edges, 1);
        assert_eq!(stats.hard_dead_states, 2);
    }

    #[test]
    fn progressive_widening_replaces_dead_active_edges() {
        let mut store = make_mem_store();
        let mut pending_board = TetrisBoard::EMPTY_BOARD;
        let _ = pending_board
            .apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)[0]);

        {
            let parent = store.get_node_mut(0).unwrap();
            parent.expanded = true;
            parent.piece_groups = vec![PieceGroup {
                piece: TetrisPiece::I_PIECE,
                visits: 0,
                value_sum: 0.0,
                prior: 1.0,
                dead: false,
                total_placements: 2,
                invalid_placements: 0,
                placements: vec![Edge {
                    child: INVALID_IDX,
                    child_key: StateKey::new(pending_board, TetrisPieceBagState::new()),
                    visits: 0,
                    value_sum: 0.0,
                    prior: 1.0,
                    dead: true,
                }],
                pending: vec![PendingPlacement {
                    child_key: StateKey::new(pending_board, TetrisPieceBagState::new()),
                    prior: 0.5,
                }],
            }];
        }

        promote_pending(&mut store, 0, 0);

        let parent = store.get_node(0).unwrap();
        assert_eq!(parent.piece_groups[0].placements.len(), 2);
        assert!(!parent.piece_groups[0].placements[1].dead);
        assert!(parent.piece_groups[0].pending.is_empty());
    }

    #[test]
    fn closure_snapshot_tracks_covered_root_pieces() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();

        let mut target_board = TetrisBoard::EMPTY_BOARD;
        let _ = target_board
            .apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0]);
        store.get_or_create(StateKey::new(target_board, TetrisPieceBagState::new()), 0);
        search.add_to_pool(target_board, 10);

        search.record_reach(
            TetrisBoard::EMPTY_BOARD,
            target_board,
            Some(TetrisPiece::O_PIECE),
        );

        let closure = search.closure_snapshot(&store);
        assert_eq!(closure.core_states, 2);
        assert_eq!(closure.required, 14);
        assert_eq!(closure.covered, 1);
        assert_eq!(closure.open, 13);
        assert!(closure.closure() > 0.0);
    }

    #[test]
    fn iteration_increments_visits() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();
        for _ in 0..100 {
            search.run_iteration(
                &mut store,
                TetrisBoard::EMPTY_BOARD,
                1.4,
                0.25,
                0.10,
                2,
                6,
                3,
                u32::MAX,
                2.0,
            );
        }
        assert_eq!(store.get_node(0).unwrap().visits, 100);
    }

    #[test]
    fn piece_group_visits_sum() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();
        for _ in 0..200 {
            search.run_iteration(
                &mut store,
                TetrisBoard::EMPTY_BOARD,
                1.4,
                0.25,
                0.10,
                2,
                6,
                3,
                u32::MAX,
                2.0,
            );
        }
        let root = store.get_node(0).unwrap();
        let total: u32 = root.piece_groups.iter().map(|pg| pg.visits).sum();
        assert_eq!(total, root.visits);
    }

    #[test]
    fn edge_visits_sum_to_group() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();
        for _ in 0..500 {
            search.run_iteration(
                &mut store,
                TetrisBoard::EMPTY_BOARD,
                1.4,
                0.25,
                0.10,
                2,
                6,
                3,
                u32::MAX,
                2.0,
            );
        }
        let root = store.get_node(0).unwrap();
        for pg in &root.piece_groups {
            if pg.visits > 0 {
                let edge_total: u32 = pg.placements.iter().map(|e| e.visits).sum();
                assert_eq!(edge_total, pg.visits);
            }
        }
    }

    #[test]
    fn no_false_cycles_shallow() {
        let mut store = make_mem_store();
        let mut search = McgsSearch::new();
        for _ in 0..1000 {
            search.run_iteration(
                &mut store,
                TetrisBoard::EMPTY_BOARD,
                1.4,
                0.25,
                0.10,
                1,
                6,
                3,
                u32::MAX,
                2.0,
            );
        }
        assert_eq!(search.cycle_closes, 0);
    }

    #[test]
    fn heuristic_bounds() {
        assert_eq!(heuristic_value(&TetrisBoard::EMPTY_BOARD), 1.0);
        let mut b = TetrisBoard::EMPTY_BOARD;
        let _ =
            b.apply_piece_placement(TetrisPiecePlacement::all_from_piece(TetrisPiece::all()[0])[0]);
        let v = heuristic_value(&b);
        assert!((-1.0..=1.0).contains(&v));
    }

    #[test]
    fn board_bytes_roundtrip() {
        let key = StateKey::empty();
        let bytes = key.board_bytes();
        let bag_u8 = u8::from(key.bag);
        let restored = StateKey::from_board_bytes(&bytes, bag_u8);
        assert_eq!(key.board, restored.board);
        assert_eq!(key.bag, restored.bag);
    }

    #[test]
    fn verify_progress_logging_handles_empty_candidates() {
        let results = verify_candidates_full_bag(&[], 6, 3, u32::MAX, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn background_verification_returns_without_blocking_sampler_state() {
        let mut running = Some(launch_verification(Vec::new(), 123, 6, 3, u32::MAX, 0));
        let started_at = Instant::now();

        let report = loop {
            if let Some(report) = poll_verification(&mut running) {
                break report;
            }
            assert!(started_at.elapsed() < std::time::Duration::from_secs(5));
            std::thread::yield_now();
        };

        assert_eq!(report.started_iter, 123);
        assert_eq!(report.candidates, 0);
        assert!(report.results.is_empty());
        assert!(running.is_none());
    }

    #[test]
    fn sqlite_roundtrip() {
        let db_path = "/tmp/test_mcgs_roundtrip.db";
        let _ = std::fs::remove_file(db_path);

        {
            let mut store = SqliteStore::new(db_path, 100_000).unwrap();
            store.get_or_create(StateKey::empty(), 0);
            let mut search = McgsSearch::new();
            for _ in 0..50 {
                search.run_iteration(
                    &mut store,
                    TetrisBoard::EMPTY_BOARD,
                    1.4,
                    0.25,
                    0.10,
                    2,
                    6,
                    3,
                    u32::MAX,
                    2.0,
                );
            }
            store.flush().unwrap();
            assert!(store.node_count() > 1);
            assert_eq!(store.get_node(0).unwrap().visits, 50);
        }

        // Reload and verify
        {
            let store = SqliteStore::new(db_path, 100_000).unwrap();
            assert!(store.node_count() > 1);
            assert_eq!(store.get_node(0).unwrap().visits, 50);
        }

        let _ = std::fs::remove_file(db_path);
    }
}
