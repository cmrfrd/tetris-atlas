#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Search With Backtracking
//!
//! In-memory atlas search that keeps alternatives per (board, bag, piece) decision and
//! backtracks when a child state is proven dead. This version interns states to compact IDs and
//! uses an advisory (board, piece) prior cache to improve candidate ordering.

use clap::Parser;
use crossbeam_channel::{Receiver, Sender, bounded};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    TetrisPiecePlacement, repeat_idx_unroll,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch, OrientationCounts};

// --- Beam Search Parameters ---
const N: usize = 8;
const TOP_N_PER_BEAM: usize = 32;
const BEAM_WIDTH: usize = 64;
const MAX_DEPTH: usize = 4;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
const TOP_K_CANDIDATES: usize = 2;

// --- Channel & Threading ---
const CHANNEL_CAPACITY: usize = 256;
const LOG_EVERY_SECS: u64 = 3;
const BASE_SEED: u64 = 42;
const CSV_FILE: &str = "tetris_atlas_search_with_backtracking.csv";
const PRIOR_CACHE_MAX_LEN: usize = 8;

type StateId = u32;
type PriorCache = DashMap<(TetrisBoard, TetrisPiece), Vec<TetrisPieceOrientation>>;

#[derive(Parser)]
#[command(name = "tetris-atlas-search-with-backtracking")]
#[command(about = "Tetris Atlas search with decision backtracking and dead-state propagation")]
struct Cli {
    /// Optional hard cap on frontier pops (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_expansions: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateKey {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl StateKey {
    const fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DecisionKey {
    state_id: StateId,
    piece: TetrisPiece,
}

impl DecisionKey {
    const fn new(state_id: StateId, piece: TetrisPiece) -> Self {
        Self { state_id, piece }
    }
}

struct StateInterner {
    to_id: DashMap<StateKey, StateId>,
    from_id: RwLock<Vec<StateKey>>,
    next_id: AtomicU32,
}

impl StateInterner {
    fn new() -> Self {
        Self {
            to_id: DashMap::new(),
            from_id: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn intern(&self, key: StateKey) -> StateId {
        if let Some(id) = self.to_id.get(&key) {
            return *id;
        }

        let mut write_guard = self.from_id.write().expect("state interner poisoned");
        if let Some(id) = self.to_id.get(&key) {
            return *id;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        write_guard.push(key);
        self.to_id.insert(key, id);
        id
    }

    fn get(&self, id: StateId) -> Option<StateKey> {
        let read_guard = self.from_id.read().expect("state interner poisoned");
        read_guard.get(id as usize).copied()
    }

    fn len(&self) -> usize {
        self.next_id.load(Ordering::Relaxed) as usize
    }
}

#[derive(Clone, Copy)]
struct FrontierValue {
    state_id: StateId,
}

impl FrontierValue {
    const fn new(state_id: StateId) -> Self {
        Self { state_id }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    priority: u32,
    hash: u64,
}

impl FrontierKey {
    fn new(state: StateKey) -> Self {
        let priority = state.board.count();
        let hash = hash_board_bag(state.board, state.bag);
        Self { priority, hash }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => self.hash.cmp(&other.hash),
            other => other,
        }
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
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

    fn push(&self, state_id: StateId, state_key: StateKey) {
        let key = FrontierKey::new(state_key);
        let value = FrontierValue::new(state_id);
        if !self.map.contains_key(&key) {
            self.map.insert(key, value);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn pop(&self) -> Option<StateId> {
        self.map.pop_front().map(|entry| entry.value().state_id)
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    orientation: TetrisPieceOrientation,
    child_id: StateId,
    votes: u32,
    dead: bool,
}

#[derive(Clone)]
struct DecisionRecord {
    candidates: Vec<Candidate>,
    selected_idx: Option<usize>,
    exhausted: bool,
}

impl DecisionRecord {
    fn new(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates,
            selected_idx: None,
            exhausted: false,
        }
    }
}

#[derive(Clone, Copy)]
struct ParentRef {
    decision_key: DecisionKey,
    candidate_idx: usize,
}

struct GraphStore {
    decisions: DashMap<DecisionKey, DecisionRecord>,
    parents: DashMap<StateId, Vec<ParentRef>>,
    dead_states: DashMap<StateId, ()>,
    state_first_decision_seq: DashMap<StateId, u64>,
}

impl GraphStore {
    fn new() -> Self {
        Self {
            decisions: DashMap::new(),
            parents: DashMap::new(),
            dead_states: DashMap::new(),
            state_first_decision_seq: DashMap::new(),
        }
    }

    fn is_state_dead(&self, state_id: StateId) -> bool {
        self.dead_states.contains_key(&state_id)
    }
}

#[derive(Default)]
struct Stats {
    frontier_consumed: AtomicU64,
    boards_expanded: AtomicU64,
    decisions_created: AtomicU64,
    decisions_exhausted: AtomicU64,
    dead_states_marked: AtomicU64,
    backtrack_switches: AtomicU64,
    candidate_losses_immediate: AtomicU64,
    lookup_checks: AtomicU64,
    lookup_hits: AtomicU64,
    candidate_total_raw: AtomicU64,
    candidate_unique_child_total: AtomicU64,
    duplicate_candidates_total: AtomicU64,
    dead_lag_decisions_sum: AtomicU64,
    dead_lag_samples: AtomicU64,
    prior_lookups: AtomicU64,
    prior_used: AtomicU64,
    prior_top1_match: AtomicU64,
}

struct CsvLogger {
    file: BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        let file_exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        if !file_exists {
            writeln!(
                writer,
                "timestamp_secs,boards_expanded,frontier_size,in_flight,\
                 frontier_enqueued,frontier_consumed,decisions_created,decisions_total,\
                 decisions_exhausted,dead_states,switches,immediate_losses,lookup_hits,\
                 lookup_checks,lookup_hit_rate,\
                 boards_per_sec,frontier_in_rate,frontier_out_rate,frontier_ratio,\
                 decision_per_expanded,hit_rate,avg_unique_children_per_decision,\
                 duplicate_candidate_rate,avg_dead_lag_decisions,dead_lag_samples,\
                 prior_lookups,prior_used,prior_hit_rate,prior_top1_match,prior_top1_rate,interned_states"
            )?;
        }

        Ok(Self { file: writer })
    }

    fn log(
        &mut self,
        frontier: &PriorityFrontier,
        stats: &Stats,
        processing_count: &AtomicU64,
        graph: &GraphStore,
        interner: &StateInterner,
        start: Instant,
    ) -> std::io::Result<()> {
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let frontier_size = frontier.len();
        let in_flight = processing_count.load(Ordering::Relaxed);
        let frontier_enqueued = frontier.enqueued_count.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_total = graph.decisions.len() as u64;
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = graph.dead_states.len() as u64;
        let switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let lookup_checks = stats.lookup_checks.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let candidate_total_raw = stats.candidate_total_raw.load(Ordering::Relaxed);
        let candidate_unique_child_total =
            stats.candidate_unique_child_total.load(Ordering::Relaxed);
        let duplicate_candidates_total = stats.duplicate_candidates_total.load(Ordering::Relaxed);
        let dead_lag_decisions_sum = stats.dead_lag_decisions_sum.load(Ordering::Relaxed);
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);
        let prior_lookups = stats.prior_lookups.load(Ordering::Relaxed);
        let prior_used = stats.prior_used.load(Ordering::Relaxed);
        let prior_top1_match = stats.prior_top1_match.load(Ordering::Relaxed);
        let interned_states = interner.len() as u64;

        let boards_per_sec = boards_expanded as f64 / secs;
        let frontier_in_rate = frontier_enqueued as f64 / secs;
        let frontier_out_rate = frontier_consumed as f64 / secs;
        let frontier_ratio = frontier_enqueued as f64 / frontier_consumed.max(1) as f64;
        let decision_per_expanded = decisions_created as f64 / boards_expanded.max(1) as f64;
        let lookup_hit_rate = lookup_hits as f64 / lookup_checks.max(1) as f64;
        let hit_rate = lookup_hit_rate;
        let avg_unique_children_per_decision =
            candidate_unique_child_total as f64 / decisions_created.max(1) as f64;
        let duplicate_candidate_rate =
            duplicate_candidates_total as f64 / candidate_total_raw.max(1) as f64;
        let avg_dead_lag_decisions = dead_lag_decisions_sum as f64 / dead_lag_samples.max(1) as f64;
        let prior_hit_rate = prior_used as f64 / prior_lookups.max(1) as f64;
        let prior_top1_rate = prior_top1_match as f64 / prior_used.max(1) as f64;

        writeln!(
            self.file,
            "{:.3},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            secs,
            boards_expanded,
            frontier_size,
            in_flight,
            frontier_enqueued,
            frontier_consumed,
            decisions_created,
            decisions_total,
            decisions_exhausted,
            dead_states,
            switches,
            immediate_losses,
            lookup_hits,
            lookup_checks,
            lookup_hit_rate,
            boards_per_sec,
            frontier_in_rate,
            frontier_out_rate,
            frontier_ratio,
            decision_per_expanded,
            hit_rate,
            avg_unique_children_per_decision,
            duplicate_candidate_rate,
            avg_dead_lag_decisions,
            dead_lag_samples,
            prior_lookups,
            prior_used,
            prior_hit_rate,
            prior_top1_match,
            prior_top1_rate,
            interned_states
        )?;
        self.file.flush()?;
        Ok(())
    }
}

struct WorkerPieceEval {
    piece: TetrisPiece,
    candidates: Vec<Candidate>,
}

struct WorkerResult {
    state_id: StateId,
    per_piece: Vec<WorkerPieceEval>,
}

fn dispatcher_thread(
    frontier: Arc<PriorityFrontier>,
    sender: Sender<StateId>,
    shutdown_requested: Arc<AtomicBool>,
    processing_count: Arc<AtomicU64>,
    stats: Arc<Stats>,
) {
    while !shutdown_requested.load(Ordering::Relaxed) {
        processing_count.fetch_add(1, Ordering::SeqCst);
        match frontier.pop() {
            Some(state_id) => {
                stats.frontier_consumed.fetch_add(1, Ordering::Relaxed);
                if sender.send(state_id).is_err() {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    break;
                }
            }
            None => {
                processing_count.fetch_sub(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
    drop(sender);
}

fn orientation_candidates_desc(counts: OrientationCounts) -> Vec<(TetrisPieceOrientation, u32)> {
    let mut out: Vec<(TetrisPieceOrientation, u32)> = counts.nonzero_orientations().collect();
    out.sort_by_key(|(_, votes)| Reverse(*votes));
    if out.len() > TOP_K_CANDIDATES {
        out.truncate(TOP_K_CANDIDATES);
    }
    out
}

fn apply_prior_order(
    ranked: &mut [(TetrisPieceOrientation, u32)],
    prior: &[TetrisPieceOrientation],
) {
    ranked.sort_by(|(lhs_ori, lhs_votes), (rhs_ori, rhs_votes)| {
        let lhs_pos = prior.iter().position(|x| x == lhs_ori);
        let rhs_pos = prior.iter().position(|x| x == rhs_ori);
        match (lhs_pos, rhs_pos) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => rhs_votes.cmp(lhs_votes),
        }
    });
}

fn evaluate_state(
    state_id: StateId,
    interner: &StateInterner,
    prior_cache: &PriorCache,
    beam_search: &mut MultiBeamSearch<
        BeamTetrisState,
        N,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    >,
    stats: &Stats,
) -> Option<WorkerResult> {
    let state = interner.get(state_id)?;
    stats.boards_expanded.fetch_add(1, Ordering::Relaxed);

    let mut game = tetris_game::TetrisGame::new();
    let mut per_piece = Vec::new();
    let (depth, beam_width) = (MAX_DEPTH, BEAM_WIDTH);

    for (piece, next_bag) in state.bag.iter_next_states() {
        game.board = state.board;
        game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);

        let counts = beam_search.search_count_actions_with_seeds(
            BeamTetrisState::new(game),
            BASE_SEED,
            depth,
            beam_width,
        );
        let mut ranked = orientation_candidates_desc(counts);
        stats.prior_lookups.fetch_add(1, Ordering::Relaxed);
        if let Some(prior) = prior_cache.get(&(state.board, piece)) {
            stats.prior_used.fetch_add(1, Ordering::Relaxed);
            if let Some((top_ori, _)) = ranked.first() {
                if prior.first().is_some_and(|p| p == top_ori) {
                    stats.prior_top1_match.fetch_add(1, Ordering::Relaxed);
                }
            }
            apply_prior_order(&mut ranked, &prior);
        }

        let mut candidates = Vec::with_capacity(ranked.len());
        for (orientation, votes) in ranked {
            game.board = state.board;
            game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);
            let placement = TetrisPiecePlacement { piece, orientation };
            let result = game.apply_placement(placement);
            if result.is_lost == IsLost::LOST {
                stats
                    .candidate_losses_immediate
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let child_key = StateKey::new(game.board, next_bag);
            let child_id = interner.intern(child_key);
            candidates.push(Candidate {
                orientation,
                child_id,
                votes,
                dead: false,
            });
        }
        per_piece.push(WorkerPieceEval { piece, candidates });
    }

    Some(WorkerResult {
        state_id,
        per_piece,
    })
}

fn should_process_state(
    graph: &GraphStore,
    interner: &StateInterner,
    stats: &Stats,
    state_id: StateId,
) -> bool {
    if graph.is_state_dead(state_id) {
        return false;
    }

    let Some(state) = interner.get(state_id) else {
        return false;
    };

    state.bag.iter_pieces().any(|piece| {
        stats.lookup_checks.fetch_add(1, Ordering::Relaxed);
        let has_decision = graph
            .decisions
            .contains_key(&DecisionKey::new(state_id, piece));
        if has_decision {
            stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
        }
        !has_decision
    })
}

fn mark_state_dead_and_propagate(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    interner: &StateInterner,
    stats: &Stats,
    initial_dead_state: StateId,
) {
    let mut queue = VecDeque::new();
    queue.push_back(initial_dead_state);

    while let Some(dead_state_id) = queue.pop_front() {
        if graph.dead_states.insert(dead_state_id, ()).is_some() {
            continue;
        }
        stats.dead_states_marked.fetch_add(1, Ordering::Relaxed);
        if let Some(seq_ref) = graph.state_first_decision_seq.get(&dead_state_id) {
            let created_now = stats.decisions_created.load(Ordering::Relaxed);
            let lag = created_now.saturating_sub(*seq_ref);
            stats
                .dead_lag_decisions_sum
                .fetch_add(lag, Ordering::Relaxed);
            stats.dead_lag_samples.fetch_add(1, Ordering::Relaxed);
        }

        let parent_refs = graph
            .parents
            .get(&dead_state_id)
            .map(|r| r.clone())
            .unwrap_or_default();

        for parent_ref in parent_refs {
            let mut should_enqueue_parent_dead = false;
            let mut newly_selected_child: Option<StateId> = None;
            let mut switched = false;

            if let Some(mut dec) = graph.decisions.get_mut(&parent_ref.decision_key) {
                if parent_ref.candidate_idx < dec.candidates.len() {
                    dec.candidates[parent_ref.candidate_idx].dead = true;
                }

                let selected_is_dead = dec.selected_idx.is_none_or(|idx| {
                    dec.candidates
                        .get(idx)
                        .is_none_or(|c| c.dead || graph.is_state_dead(c.child_id))
                });

                if selected_is_dead {
                    let next = dec
                        .candidates
                        .iter()
                        .enumerate()
                        .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child_id))
                        .map(|(idx, c)| (idx, c.child_id));

                    match next {
                        Some((next_idx, child_id)) => {
                            switched = dec.selected_idx != Some(next_idx);
                            dec.selected_idx = Some(next_idx);
                            dec.exhausted = false;
                            newly_selected_child = Some(child_id);
                        }
                        None => {
                            if !dec.exhausted {
                                dec.exhausted = true;
                                should_enqueue_parent_dead = true;
                                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

            if switched {
                stats.backtrack_switches.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(child_id) = newly_selected_child
                && let Some(child_key) = interner.get(child_id)
            {
                frontier.push(child_id, child_key);
            }
            if should_enqueue_parent_dead {
                queue.push_back(parent_ref.decision_key.state_id);
            }
        }
    }
}

fn update_prior_cache(
    prior_cache: &PriorCache,
    board: TetrisBoard,
    piece: TetrisPiece,
    preferred_orientation: TetrisPieceOrientation,
) {
    prior_cache
        .entry((board, piece))
        .and_modify(|v| {
            if let Some(pos) = v.iter().position(|x| *x == preferred_orientation) {
                if pos != 0 {
                    v.remove(pos);
                    v.insert(0, preferred_orientation);
                }
            } else {
                v.insert(0, preferred_orientation);
                if v.len() > PRIOR_CACHE_MAX_LEN {
                    v.truncate(PRIOR_CACHE_MAX_LEN);
                }
            }
        })
        .or_insert_with(|| vec![preferred_orientation]);
}

fn coordinator_handle_result(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    interner: &StateInterner,
    prior_cache: &PriorCache,
    stats: &Stats,
    result: WorkerResult,
) {
    if graph.is_state_dead(result.state_id) {
        return;
    }
    let Some(state_key) = interner.get(result.state_id) else {
        return;
    };

    for piece_eval in result.per_piece {
        if graph.is_state_dead(result.state_id) {
            break;
        }

        let dkey = DecisionKey::new(result.state_id, piece_eval.piece);
        stats.lookup_checks.fetch_add(1, Ordering::Relaxed);
        if graph.decisions.contains_key(&dkey) {
            stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let mut decision = DecisionRecord::new(piece_eval.candidates);
        let created_seq = stats.decisions_created.fetch_add(1, Ordering::Relaxed) + 1;
        graph
            .state_first_decision_seq
            .entry(result.state_id)
            .or_insert(created_seq);

        let raw_count = decision.candidates.len() as u64;
        let unique_count = decision
            .candidates
            .iter()
            .map(|c| c.child_id)
            .collect::<HashSet<_>>()
            .len() as u64;
        let dup_count = raw_count.saturating_sub(unique_count);
        stats
            .candidate_total_raw
            .fetch_add(raw_count, Ordering::Relaxed);
        stats
            .candidate_unique_child_total
            .fetch_add(unique_count, Ordering::Relaxed);
        stats
            .duplicate_candidates_total
            .fetch_add(dup_count, Ordering::Relaxed);

        for (idx, candidate) in decision.candidates.iter().enumerate() {
            let pref = ParentRef {
                decision_key: dkey,
                candidate_idx: idx,
            };
            graph
                .parents
                .entry(candidate.child_id)
                .and_modify(|v| v.push(pref))
                .or_insert_with(|| vec![pref]);
        }

        let selected = decision
            .candidates
            .iter()
            .enumerate()
            .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child_id))
            .map(|(idx, c)| (idx, c.child_id, c.orientation, c.votes));

        match selected {
            Some((idx, child_id, orientation, _votes)) => {
                decision.selected_idx = Some(idx);
                if let Some(child_key) = interner.get(child_id) {
                    frontier.push(child_id, child_key);
                }
                update_prior_cache(prior_cache, state_key.board, piece_eval.piece, orientation);
            }
            None => {
                decision.exhausted = true;
                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
            }
        }

        let exhausted = decision.exhausted;
        graph.decisions.insert(dkey, decision);
        if exhausted {
            mark_state_dead_and_propagate(graph, frontier, interner, stats, result.state_id);
            if graph.is_state_dead(result.state_id) {
                break;
            }
        }
    }
}

fn coordinator_thread(
    result_rx: Receiver<WorkerResult>,
    graph: Arc<GraphStore>,
    frontier: Arc<PriorityFrontier>,
    interner: Arc<StateInterner>,
    prior_cache: Arc<PriorCache>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
) {
    while let Ok(result) = result_rx.recv() {
        coordinator_handle_result(&graph, &frontier, &interner, &prior_cache, &stats, result);
        processing_count.fetch_sub(1, Ordering::Relaxed);
    }
}

fn logger_thread(
    frontier: Arc<PriorityFrontier>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
    graph: Arc<GraphStore>,
    interner: Arc<StateInterner>,
    shutdown: Arc<AtomicBool>,
    start: Instant,
    csv_file: String,
) {
    let mut csv_logger = match CsvLogger::new(&csv_file) {
        Ok(logger) => {
            println!("CSV logging to: {csv_file}");
            Some(logger)
        }
        Err(e) => {
            eprintln!("Failed to open CSV logger at {csv_file}: {e}");
            None
        }
    };

    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(LOG_EVERY_SECS));

        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = stats.dead_states_marked.load(Ordering::Relaxed);
        let backtrack_switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let lookup_checks = stats.lookup_checks.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let avg_unique_children_per_decision =
            stats.candidate_unique_child_total.load(Ordering::Relaxed) as f64
                / decisions_created.max(1) as f64;
        let duplicate_candidate_rate = stats.duplicate_candidates_total.load(Ordering::Relaxed)
            as f64
            / stats.candidate_total_raw.load(Ordering::Relaxed).max(1) as f64;
        let avg_dead_lag_decisions = stats.dead_lag_decisions_sum.load(Ordering::Relaxed) as f64
            / stats.dead_lag_samples.load(Ordering::Relaxed).max(1) as f64;
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);
        let prior_lookups = stats.prior_lookups.load(Ordering::Relaxed);
        let prior_used = stats.prior_used.load(Ordering::Relaxed);
        let lookup_hit_rate = lookup_hits as f64 / lookup_checks.max(1) as f64;
        let prior_hit_rate = prior_used as f64 / prior_lookups.max(1) as f64;
        let prior_top1_rate =
            stats.prior_top1_match.load(Ordering::Relaxed) as f64 / prior_used.max(1) as f64;

        println!(
            "t={secs:.1}s expanded={boards_expanded} frontier={} in_flight={} consumed={} | \
             decisions={} exhausted={} dead_states={} switches={} immediate_losses={} \
             hits={}/{} ({lookup_hit_rate:.3}) prior_hits={}/{} ({prior_hit_rate:.3}) | \
             uniq/dec={avg_unique_children_per_decision:.3} dup_rate={duplicate_candidate_rate:.3} \
             dead_lag={avg_dead_lag_decisions:.2} n_dead_lag={dead_lag_samples} prior_top1={prior_top1_rate:.3} states={}",
            frontier.len(),
            processing_count.load(Ordering::Relaxed),
            frontier_consumed,
            decisions_created,
            decisions_exhausted,
            dead_states,
            backtrack_switches,
            immediate_losses,
            lookup_hits,
            lookup_checks,
            prior_used,
            prior_lookups,
            interner.len()
        );

        if let Some(ref mut logger) = csv_logger
            && let Err(e) = logger.log(
                &frontier,
                &stats,
                &processing_count,
                &graph,
                &interner,
                start,
            )
        {
            eprintln!("Failed to write CSV row: {e}");
        }
    }

    if let Some(ref mut logger) = csv_logger
        && let Err(e) = logger.log(
            &frontier,
            &stats,
            &processing_count,
            &graph,
            &interner,
            start,
        )
    {
        eprintln!("Failed to write final CSV row: {e}");
    }
}

fn main() {
    let cli = Cli::parse();
    run_tetris_atlas_search_with_backtracking(cli.max_expansions);
}

pub fn run_tetris_atlas_search_with_backtracking(max_expansions: u64) {
    let num_workers = num_cpus::get().saturating_sub(2).max(1);
    println!("Starting tetris_atlas_search_with_backtracking");
    println!("Workers: {num_workers} | Channel capacity: {CHANNEL_CAPACITY}");
    if max_expansions > 0 {
        println!("Max expansions: {max_expansions}");
    }
    println!("CSV file: {CSV_FILE}");

    let interner = Arc::new(StateInterner::new());
    let prior_cache = Arc::new(PriorCache::new());
    let graph = Arc::new(GraphStore::new());
    let frontier_enqueued = Arc::new(AtomicU64::new(0));
    let frontier = Arc::new(PriorityFrontier::new(frontier_enqueued));
    let stats = Arc::new(Stats::default());
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));
    let processing_count = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let root_key = StateKey::new(TetrisBoard::new(), TetrisPieceBagState::new());
    let root_id = interner.intern(root_key);
    frontier.push(root_id, root_key);

    let (work_tx, work_rx) = bounded::<StateId>(CHANNEL_CAPACITY);
    let (result_tx, result_rx) = bounded::<WorkerResult>(CHANNEL_CAPACITY);

    let shutdown_requested_clone = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received, stopping...");
        shutdown_requested_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    let dispatcher = {
        let frontier = frontier.clone();
        let shutdown_requested = shutdown_requested.clone();
        let processing_count = processing_count.clone();
        let stats = stats.clone();
        thread::spawn(move || {
            dispatcher_thread(
                frontier,
                work_tx,
                shutdown_requested,
                processing_count,
                stats,
            );
        })
    };

    let coordinator = {
        let graph = graph.clone();
        let frontier = frontier.clone();
        let interner = interner.clone();
        let prior_cache = prior_cache.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        thread::spawn(move || {
            coordinator_thread(
                result_rx,
                graph,
                frontier,
                interner,
                prior_cache,
                stats,
                processing_count,
            );
        })
    };

    let logger = {
        let frontier = frontier.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        let graph = graph.clone();
        let interner = interner.clone();
        let shutdown = shutdown.clone();
        let csv_file = CSV_FILE.to_string();
        thread::spawn(move || {
            logger_thread(
                frontier,
                stats,
                processing_count,
                graph,
                interner,
                shutdown,
                start,
                csv_file,
            );
        })
    };

    let mut worker_handles = Vec::with_capacity(num_workers);
    for _worker_id in 0..num_workers {
        let rx = work_rx.clone();
        let tx = result_tx.clone();
        let graph = graph.clone();
        let interner = interner.clone();
        let prior_cache = prior_cache.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        worker_handles.push(thread::spawn(move || {
            let mut beam_search = MultiBeamSearch::<
                BeamTetrisState,
                N,
                TOP_N_PER_BEAM,
                BEAM_WIDTH,
                MAX_DEPTH,
                MAX_MOVES,
            >::new();

            while let Ok(state_id) = rx.recv() {
                if !should_process_state(&graph, &interner, &stats, state_id) {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    continue;
                }
                let Some(result) =
                    evaluate_state(state_id, &interner, &prior_cache, &mut beam_search, &stats)
                else {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    continue;
                };
                if tx.send(result).is_err() {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    break;
                }
            }
        }));
    }

    drop(result_tx);

    let mut last_progress_consumed = 0u64;
    loop {
        if shutdown_requested.load(Ordering::Relaxed) {
            break;
        }

        let consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        if max_expansions > 0 && consumed >= max_expansions {
            println!("Reached max_expansions={max_expansions}");
            break;
        }

        if frontier.len() == 0 && processing_count.load(Ordering::Relaxed) == 0 {
            if consumed == last_progress_consumed {
                break;
            }
            last_progress_consumed = consumed;
        }

        thread::sleep(Duration::from_millis(100));
    }

    shutdown_requested.store(true, Ordering::Relaxed);
    dispatcher.join().expect("dispatcher panicked");
    for handle in worker_handles {
        handle.join().expect("worker panicked");
    }
    coordinator.join().expect("coordinator panicked");

    shutdown.store(true, Ordering::Relaxed);
    logger.join().expect("logger panicked");

    println!("\nDone.");
    println!(
        "Expanded: {}",
        stats.boards_expanded.load(Ordering::Relaxed)
    );
    println!("Decisions: {}", graph.decisions.len());
    println!("Dead states: {}", graph.dead_states.len());
    println!("Frontier remaining: {}", frontier.len());
    println!("Interned states: {}", interner.len());
    println!("Elapsed: {:.2}s", start.elapsed().as_secs_f64());
}
/*
#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Search With Backtracking
//!
//! In-memory atlas search that keeps alternatives per (board, bag, piece) decision and
//! backtracks when a child state is proven dead. This version interns states to compact IDs and
//! uses an advisory (board, piece) prior cache to improve candidate ordering.

use clap::Parser;
use crossbeam_channel::{Receiver, Sender, bounded};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    TetrisPiecePlacement, repeat_idx_unroll,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch, OrientationCounts};

// --- Beam Search Parameters ---
const N: usize = 8;
const TOP_N_PER_BEAM: usize = 32;
const BEAM_WIDTH: usize = 1024;
const MAX_DEPTH: usize = 4;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
const TOP_K_CANDIDATES: usize = 3;

// --- Channel & Threading ---
const CHANNEL_CAPACITY: usize = 256;
const LOG_EVERY_SECS: u64 = 3;
const BASE_SEED: u64 = 42;
const CSV_FILE: &str = "tetris_atlas_search_with_backtracking.csv";
const PRIOR_CACHE_MAX_LEN: usize = 8;

type StateId = u32;
type PriorCache = DashMap<(TetrisBoard, TetrisPiece), Vec<TetrisPieceOrientation>>;

#[derive(Parser)]
#[command(name = "tetris-atlas-search-with-backtracking")]
#[command(about = "Tetris Atlas search with decision backtracking and dead-state propagation")]
struct Cli {
    /// Optional hard cap on frontier pops (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_expansions: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateKey {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl StateKey {
    const fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DecisionKey {
    state_id: StateId,
    piece: TetrisPiece,
}

impl DecisionKey {
    const fn new(state_id: StateId, piece: TetrisPiece) -> Self {
        Self { state_id, piece }
    }
}

struct StateInterner {
    to_id: DashMap<StateKey, StateId>,
    from_id: RwLock<Vec<StateKey>>,
    next_id: AtomicU32,
}

impl StateInterner {
    fn new() -> Self {
        Self {
            to_id: DashMap::new(),
            from_id: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    fn intern(&self, key: StateKey) -> StateId {
        if let Some(id) = self.to_id.get(&key) {
            return *id;
        }

        let mut write_guard = self.from_id.write().expect("state interner poisoned");
        if let Some(id) = self.to_id.get(&key) {
            return *id;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        write_guard.push(key);
        self.to_id.insert(key, id);
        id
    }

    fn get(&self, id: StateId) -> Option<StateKey> {
        let read_guard = self.from_id.read().expect("state interner poisoned");
        read_guard.get(id as usize).copied()
    }

    fn len(&self) -> usize {
        self.next_id.load(Ordering::Relaxed) as usize
    }
}

#[derive(Clone, Copy)]
struct FrontierValue {
    state_id: StateId,
}

impl FrontierValue {
    const fn new(state_id: StateId) -> Self {
        Self { state_id }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    priority: u32,
    hash: u64,
}

impl FrontierKey {
    fn new(state: StateKey) -> Self {
        let priority = state.board.count();
        let hash = hash_board_bag(state.board, state.bag);
        Self { priority, hash }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => self.hash.cmp(&other.hash),
            other => other,
        }
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
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

    fn push(&self, state_id: StateId, state_key: StateKey) {
        let key = FrontierKey::new(state_key);
        let value = FrontierValue::new(state_id);
        if !self.map.contains_key(&key) {
            self.map.insert(key, value);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn pop(&self) -> Option<StateId> {
        self.map.pop_front().map(|entry| entry.value().state_id)
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    orientation: TetrisPieceOrientation,
    child_id: StateId,
    votes: u32,
    dead: bool,
}

#[derive(Clone)]
struct DecisionRecord {
    candidates: Vec<Candidate>,
    selected_idx: Option<usize>,
    exhausted: bool,
}

impl DecisionRecord {
    fn new(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates,
            selected_idx: None,
            exhausted: false,
        }
    }
}

#[derive(Clone, Copy)]
struct ParentRef {
    decision_key: DecisionKey,
    candidate_idx: usize,
}

struct GraphStore {
    decisions: DashMap<DecisionKey, DecisionRecord>,
    parents: DashMap<StateId, Vec<ParentRef>>,
    dead_states: DashMap<StateId, ()>,
    state_first_decision_seq: DashMap<StateId, u64>,
}

impl GraphStore {
    fn new() -> Self {
        Self {
            decisions: DashMap::new(),
            parents: DashMap::new(),
            dead_states: DashMap::new(),
            state_first_decision_seq: DashMap::new(),
        }
    }

    fn is_state_dead(&self, state_id: StateId) -> bool {
        self.dead_states.contains_key(&state_id)
    }
}

#[derive(Default)]
struct Stats {
    frontier_consumed: AtomicU64,
    boards_expanded: AtomicU64,
    decisions_created: AtomicU64,
    decisions_exhausted: AtomicU64,
    dead_states_marked: AtomicU64,
    backtrack_switches: AtomicU64,
    candidate_losses_immediate: AtomicU64,
    lookup_hits: AtomicU64,
    candidate_total_raw: AtomicU64,
    candidate_unique_child_total: AtomicU64,
    duplicate_candidates_total: AtomicU64,
    dead_lag_decisions_sum: AtomicU64,
    dead_lag_samples: AtomicU64,
    prior_used: AtomicU64,
    prior_top1_match: AtomicU64,
}

struct CsvLogger {
    file: BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        let file_exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        if !file_exists {
            writeln!(
                writer,
                "timestamp_secs,boards_expanded,frontier_size,in_flight,\
                 frontier_enqueued,frontier_consumed,decisions_created,decisions_total,\
                 decisions_exhausted,dead_states,switches,immediate_losses,lookup_hits,\
                 boards_per_sec,frontier_in_rate,frontier_out_rate,frontier_ratio,\
                 decision_per_expanded,hit_rate,avg_unique_children_per_decision,\
                 duplicate_candidate_rate,avg_dead_lag_decisions,dead_lag_samples,\
                 prior_used,prior_top1_match,prior_top1_rate,interned_states"
            )?;
        }

        Ok(Self { file: writer })
    }

    fn log(
        &mut self,
        frontier: &PriorityFrontier,
        stats: &Stats,
        processing_count: &AtomicU64,
        graph: &GraphStore,
        interner: &StateInterner,
        start: Instant,
    ) -> std::io::Result<()> {
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let frontier_size = frontier.len();
        let in_flight = processing_count.load(Ordering::Relaxed);
        let frontier_enqueued = frontier.enqueued_count.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_total = graph.decisions.len() as u64;
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = graph.dead_states.len() as u64;
        let switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let candidate_total_raw = stats.candidate_total_raw.load(Ordering::Relaxed);
        let candidate_unique_child_total =
            stats.candidate_unique_child_total.load(Ordering::Relaxed);
        let duplicate_candidates_total = stats.duplicate_candidates_total.load(Ordering::Relaxed);
        let dead_lag_decisions_sum = stats.dead_lag_decisions_sum.load(Ordering::Relaxed);
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);
        let prior_used = stats.prior_used.load(Ordering::Relaxed);
        let prior_top1_match = stats.prior_top1_match.load(Ordering::Relaxed);
        let interned_states = interner.len() as u64;

        let boards_per_sec = boards_expanded as f64 / secs;
        let frontier_in_rate = frontier_enqueued as f64 / secs;
        let frontier_out_rate = frontier_consumed as f64 / secs;
        let frontier_ratio = frontier_enqueued as f64 / frontier_consumed.max(1) as f64;
        let decision_per_expanded = decisions_created as f64 / boards_expanded.max(1) as f64;
        let hit_rate = lookup_hits as f64 / decisions_created.max(1) as f64;
        let avg_unique_children_per_decision =
            candidate_unique_child_total as f64 / decisions_created.max(1) as f64;
        let duplicate_candidate_rate =
            duplicate_candidates_total as f64 / candidate_total_raw.max(1) as f64;
        let avg_dead_lag_decisions = dead_lag_decisions_sum as f64 / dead_lag_samples.max(1) as f64;
        let prior_top1_rate = prior_top1_match as f64 / prior_used.max(1) as f64;

        writeln!(
            self.file,
            "{:.3},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.6},{:.6},{:.4},{},{},{},{:.4},{}",
            secs,
            boards_expanded,
            frontier_size,
            in_flight,
            frontier_enqueued,
            frontier_consumed,
            decisions_created,
            decisions_total,
            decisions_exhausted,
            dead_states,
            switches,
            immediate_losses,
            lookup_hits,
            boards_per_sec,
            frontier_in_rate,
            frontier_out_rate,
            frontier_ratio,
            decision_per_expanded,
            hit_rate,
            avg_unique_children_per_decision,
            duplicate_candidate_rate,
            avg_dead_lag_decisions,
            dead_lag_samples,
            prior_used,
            prior_top1_match,
            prior_top1_rate,
            interned_states
        )?;
        self.file.flush()?;
        Ok(())
    }
}

struct WorkerPieceEval {
    piece: TetrisPiece,
    candidates: Vec<Candidate>,
}

struct WorkerResult {
    state_id: StateId,
    per_piece: Vec<WorkerPieceEval>,
}

fn dispatcher_thread(
    frontier: Arc<PriorityFrontier>,
    sender: Sender<StateId>,
    shutdown_requested: Arc<AtomicBool>,
    processing_count: Arc<AtomicU64>,
    stats: Arc<Stats>,
) {
    while !shutdown_requested.load(Ordering::Relaxed) {
        processing_count.fetch_add(1, Ordering::SeqCst);
        match frontier.pop() {
            Some(state_id) => {
                stats.frontier_consumed.fetch_add(1, Ordering::Relaxed);
                if sender.send(state_id).is_err() {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    break;
                }
            }
            None => {
                processing_count.fetch_sub(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
    drop(sender);
}

fn orientation_candidates_desc(counts: OrientationCounts) -> Vec<(TetrisPieceOrientation, u32)> {
    let mut out: Vec<(TetrisPieceOrientation, u32)> = counts.nonzero_orientations().collect();
    out.sort_by_key(|(_, votes)| Reverse(*votes));
    if out.len() > TOP_K_CANDIDATES {
        out.truncate(TOP_K_CANDIDATES);
    }
    out
}

fn apply_prior_order(
    ranked: &mut [(TetrisPieceOrientation, u32)],
    prior: &[TetrisPieceOrientation],
) {
    ranked.sort_by(|(lhs_ori, lhs_votes), (rhs_ori, rhs_votes)| {
        let lhs_pos = prior.iter().position(|x| x == lhs_ori);
        let rhs_pos = prior.iter().position(|x| x == rhs_ori);
        match (lhs_pos, rhs_pos) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => rhs_votes.cmp(lhs_votes),
        }
    });
}

fn evaluate_state(
    state_id: StateId,
    interner: &StateInterner,
    prior_cache: &PriorCache,
    beam_search: &mut MultiBeamSearch<
        BeamTetrisState,
        N,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    >,
    stats: &Stats,
) -> Option<WorkerResult> {
    let state = interner.get(state_id)?;
    stats.boards_expanded.fetch_add(1, Ordering::Relaxed);

    let mut game = tetris_game::TetrisGame::new();
    let mut per_piece = Vec::new();
    let (depth, beam_width) = (MAX_DEPTH, BEAM_WIDTH);

    for (piece, next_bag) in state.bag.iter_next_states() {
        game.board = state.board;
        game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);

        let counts = beam_search.search_count_actions_with_seeds(
            BeamTetrisState::new(game),
            BASE_SEED,
            depth,
            beam_width,
        );
        let mut ranked = orientation_candidates_desc(counts);
        if let Some(prior) = prior_cache.get(&(state.board, piece)) {
            stats.prior_used.fetch_add(1, Ordering::Relaxed);
            if let Some((top_ori, _)) = ranked.first() {
                if prior.first().is_some_and(|p| p == top_ori) {
                    stats.prior_top1_match.fetch_add(1, Ordering::Relaxed);
                }
            }
            apply_prior_order(&mut ranked, &prior);
        }

        let mut candidates = Vec::with_capacity(ranked.len());
        for (orientation, votes) in ranked {
            game.board = state.board;
            game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);
            let placement = TetrisPiecePlacement { piece, orientation };
            let result = game.apply_placement(placement);
            if result.is_lost == IsLost::LOST {
                stats
                    .candidate_losses_immediate
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let child_key = StateKey::new(game.board, next_bag);
            let child_id = interner.intern(child_key);
            candidates.push(Candidate {
                orientation,
                child_id,
                votes,
                dead: false,
            });
        }
        per_piece.push(WorkerPieceEval { piece, candidates });
    }

    Some(WorkerResult { state_id, per_piece })
}

fn mark_state_dead_and_propagate(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    interner: &StateInterner,
    stats: &Stats,
    initial_dead_state: StateId,
) {
    let mut queue = VecDeque::new();
    queue.push_back(initial_dead_state);

    while let Some(dead_state_id) = queue.pop_front() {
        if graph.dead_states.insert(dead_state_id, ()).is_some() {
            continue;
        }
        stats.dead_states_marked.fetch_add(1, Ordering::Relaxed);
        if let Some(seq_ref) = graph.state_first_decision_seq.get(&dead_state_id) {
            let created_now = stats.decisions_created.load(Ordering::Relaxed);
            let lag = created_now.saturating_sub(*seq_ref);
            stats
                .dead_lag_decisions_sum
                .fetch_add(lag, Ordering::Relaxed);
            stats.dead_lag_samples.fetch_add(1, Ordering::Relaxed);
        }

        let parent_refs = graph
            .parents
            .get(&dead_state_id)
            .map(|r| r.clone())
            .unwrap_or_default();

        for parent_ref in parent_refs {
            let mut should_enqueue_parent_dead = false;
            let mut newly_selected_child: Option<StateId> = None;
            let mut switched = false;

            if let Some(mut dec) = graph.decisions.get_mut(&parent_ref.decision_key) {
                if parent_ref.candidate_idx < dec.candidates.len() {
                    dec.candidates[parent_ref.candidate_idx].dead = true;
                }

                let selected_is_dead = dec.selected_idx.is_none_or(|idx| {
                    dec.candidates
                        .get(idx)
                        .is_none_or(|c| c.dead || graph.is_state_dead(c.child_id))
                });

                if selected_is_dead {
                    let next = dec
                        .candidates
                        .iter()
                        .enumerate()
                        .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child_id))
                        .map(|(idx, c)| (idx, c.child_id));

                    match next {
                        Some((next_idx, child_id)) => {
                            switched = dec.selected_idx != Some(next_idx);
                            dec.selected_idx = Some(next_idx);
                            dec.exhausted = false;
                            newly_selected_child = Some(child_id);
                        }
                        None => {
                            if !dec.exhausted {
                                dec.exhausted = true;
                                should_enqueue_parent_dead = true;
                                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

            if switched {
                stats.backtrack_switches.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(child_id) = newly_selected_child
                && let Some(child_key) = interner.get(child_id)
            {
                frontier.push(child_id, child_key);
            }
            if should_enqueue_parent_dead {
                queue.push_back(parent_ref.decision_key.state_id);
            }
        }
    }
}

fn update_prior_cache(
    prior_cache: &PriorCache,
    board: TetrisBoard,
    piece: TetrisPiece,
    preferred_orientation: TetrisPieceOrientation,
) {
    prior_cache
        .entry((board, piece))
        .and_modify(|v| {
            if let Some(pos) = v.iter().position(|x| *x == preferred_orientation) {
                if pos != 0 {
                    v.remove(pos);
                    v.insert(0, preferred_orientation);
                }
            } else {
                v.insert(0, preferred_orientation);
                if v.len() > PRIOR_CACHE_MAX_LEN {
                    v.truncate(PRIOR_CACHE_MAX_LEN);
                }
            }
        })
        .or_insert_with(|| vec![preferred_orientation]);
}

fn coordinator_handle_result(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    interner: &StateInterner,
    prior_cache: &PriorCache,
    stats: &Stats,
    result: WorkerResult,
) {
    if graph.is_state_dead(result.state_id) {
        return;
    }
    let Some(state_key) = interner.get(result.state_id) else {
        return;
    };

    for piece_eval in result.per_piece {
        let dkey = DecisionKey::new(result.state_id, piece_eval.piece);
        if graph.decisions.contains_key(&dkey) {
            stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let mut decision = DecisionRecord::new(piece_eval.candidates);
        let created_seq = stats.decisions_created.fetch_add(1, Ordering::Relaxed) + 1;
        graph
            .state_first_decision_seq
            .entry(result.state_id)
            .or_insert(created_seq);

        let raw_count = decision.candidates.len() as u64;
        let unique_count = decision
            .candidates
            .iter()
            .map(|c| c.child_id)
            .collect::<HashSet<_>>()
            .len() as u64;
        let dup_count = raw_count.saturating_sub(unique_count);
        stats
            .candidate_total_raw
            .fetch_add(raw_count, Ordering::Relaxed);
        stats
            .candidate_unique_child_total
            .fetch_add(unique_count, Ordering::Relaxed);
        stats
            .duplicate_candidates_total
            .fetch_add(dup_count, Ordering::Relaxed);

        for (idx, candidate) in decision.candidates.iter().enumerate() {
            let pref = ParentRef {
                decision_key: dkey,
                candidate_idx: idx,
            };
            graph
                .parents
                .entry(candidate.child_id)
                .and_modify(|v| v.push(pref))
                .or_insert_with(|| vec![pref]);
        }

        let selected = decision
            .candidates
            .iter()
            .enumerate()
            .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child_id))
            .map(|(idx, c)| (idx, c.child_id, c.orientation, c.votes));

        match selected {
            Some((idx, child_id, orientation, _votes)) => {
                decision.selected_idx = Some(idx);
                if let Some(child_key) = interner.get(child_id) {
                    frontier.push(child_id, child_key);
                }
                update_prior_cache(prior_cache, state_key.board, piece_eval.piece, orientation);
            }
            None => {
                decision.exhausted = true;
                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
            }
        }

        let exhausted = decision.exhausted;
        graph.decisions.insert(dkey, decision);
        if exhausted {
            mark_state_dead_and_propagate(graph, frontier, interner, stats, result.state_id);
        }
    }
}

fn coordinator_thread(
    result_rx: Receiver<WorkerResult>,
    graph: Arc<GraphStore>,
    frontier: Arc<PriorityFrontier>,
    interner: Arc<StateInterner>,
    prior_cache: Arc<PriorCache>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
) {
    while let Ok(result) = result_rx.recv() {
        coordinator_handle_result(&graph, &frontier, &interner, &prior_cache, &stats, result);
        processing_count.fetch_sub(1, Ordering::Relaxed);
    }
}

fn logger_thread(
    frontier: Arc<PriorityFrontier>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
    graph: Arc<GraphStore>,
    interner: Arc<StateInterner>,
    shutdown: Arc<AtomicBool>,
    start: Instant,
    csv_file: String,
) {
    let mut csv_logger = match CsvLogger::new(&csv_file) {
        Ok(logger) => {
            println!("CSV logging to: {csv_file}");
            Some(logger)
        }
        Err(e) => {
            eprintln!("Failed to open CSV logger at {csv_file}: {e}");
            None
        }
    };

    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(LOG_EVERY_SECS));

        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = stats.dead_states_marked.load(Ordering::Relaxed);
        let backtrack_switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let avg_unique_children_per_decision =
            stats.candidate_unique_child_total.load(Ordering::Relaxed) as f64
                / decisions_created.max(1) as f64;
        let duplicate_candidate_rate = stats.duplicate_candidates_total.load(Ordering::Relaxed)
            as f64
            / stats.candidate_total_raw.load(Ordering::Relaxed).max(1) as f64;
        let avg_dead_lag_decisions = stats.dead_lag_decisions_sum.load(Ordering::Relaxed) as f64
            / stats.dead_lag_samples.load(Ordering::Relaxed).max(1) as f64;
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);
        let prior_used = stats.prior_used.load(Ordering::Relaxed);
        let prior_top1_rate =
            stats.prior_top1_match.load(Ordering::Relaxed) as f64 / prior_used.max(1) as f64;

        println!(
            "t={secs:.1}s expanded={boards_expanded} frontier={} in_flight={} consumed={} | \
             decisions={} exhausted={} dead_states={} switches={} immediate_losses={} hits={} | \
             uniq/dec={avg_unique_children_per_decision:.3} dup_rate={duplicate_candidate_rate:.3} \
             dead_lag={avg_dead_lag_decisions:.2} n_dead_lag={dead_lag_samples} prior_top1={prior_top1_rate:.3} states={}",
            frontier.len(),
            processing_count.load(Ordering::Relaxed),
            frontier_consumed,
            decisions_created,
            decisions_exhausted,
            dead_states,
            backtrack_switches,
            immediate_losses,
            lookup_hits,
            interner.len()
        );

        if let Some(ref mut logger) = csv_logger
            && let Err(e) = logger.log(&frontier, &stats, &processing_count, &graph, &interner, start)
        {
            eprintln!("Failed to write CSV row: {e}");
        }
    }

    if let Some(ref mut logger) = csv_logger
        && let Err(e) = logger.log(&frontier, &stats, &processing_count, &graph, &interner, start)
    {
        eprintln!("Failed to write final CSV row: {e}");
    }
}

fn main() {
    let cli = Cli::parse();
    run_tetris_atlas_search_with_backtracking(cli.max_expansions);
}

pub fn run_tetris_atlas_search_with_backtracking(max_expansions: u64) {
    let num_workers = num_cpus::get().saturating_sub(2).max(1);
    println!("Starting tetris_atlas_search_with_backtracking");
    println!("Workers: {num_workers} | Channel capacity: {CHANNEL_CAPACITY}");
    if max_expansions > 0 {
        println!("Max expansions: {max_expansions}");
    }
    println!("CSV file: {CSV_FILE}");

    let interner = Arc::new(StateInterner::new());
    let prior_cache = Arc::new(PriorCache::new());
    let graph = Arc::new(GraphStore::new());
    let frontier_enqueued = Arc::new(AtomicU64::new(0));
    let frontier = Arc::new(PriorityFrontier::new(frontier_enqueued));
    let stats = Arc::new(Stats::default());
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));
    let processing_count = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let root_key = StateKey::new(TetrisBoard::new(), TetrisPieceBagState::new());
    let root_id = interner.intern(root_key);
    frontier.push(root_id, root_key);

    let (work_tx, work_rx) = bounded::<StateId>(CHANNEL_CAPACITY);
    let (result_tx, result_rx) = bounded::<WorkerResult>(CHANNEL_CAPACITY);

    let shutdown_requested_clone = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received, stopping...");
        shutdown_requested_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    let dispatcher = {
        let frontier = frontier.clone();
        let shutdown_requested = shutdown_requested.clone();
        let processing_count = processing_count.clone();
        let stats = stats.clone();
        thread::spawn(move || {
            dispatcher_thread(
                frontier,
                work_tx,
                shutdown_requested,
                processing_count,
                stats,
            );
        })
    };

    let coordinator = {
        let graph = graph.clone();
        let frontier = frontier.clone();
        let interner = interner.clone();
        let prior_cache = prior_cache.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        thread::spawn(move || {
            coordinator_thread(
                result_rx,
                graph,
                frontier,
                interner,
                prior_cache,
                stats,
                processing_count,
            );
        })
    };

    let logger = {
        let frontier = frontier.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        let graph = graph.clone();
        let interner = interner.clone();
        let shutdown = shutdown.clone();
        let csv_file = CSV_FILE.to_string();
        thread::spawn(move || {
            logger_thread(
                frontier,
                stats,
                processing_count,
                graph,
                interner,
                shutdown,
                start,
                csv_file,
            );
        })
    };

    let mut worker_handles = Vec::with_capacity(num_workers);
    for _worker_id in 0..num_workers {
        let rx = work_rx.clone();
        let tx = result_tx.clone();
        let interner = interner.clone();
        let prior_cache = prior_cache.clone();
        let stats = stats.clone();
        worker_handles.push(thread::spawn(move || {
            let mut beam_search = MultiBeamSearch::<
                BeamTetrisState,
                N,
                TOP_N_PER_BEAM,
                BEAM_WIDTH,
                MAX_DEPTH,
                MAX_MOVES,
            >::new();

            while let Ok(state_id) = rx.recv() {
                let Some(result) =
                    evaluate_state(state_id, &interner, &prior_cache, &mut beam_search, &stats)
                else {
                    continue;
                };
                if tx.send(result).is_err() {
                    break;
                }
            }
        }));
    }

    drop(result_tx);

    let mut last_progress_consumed = 0u64;
    loop {
        if shutdown_requested.load(Ordering::Relaxed) {
            break;
        }

        let consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        if max_expansions > 0 && consumed >= max_expansions {
            println!("Reached max_expansions={max_expansions}");
            break;
        }

        if frontier.len() == 0 && processing_count.load(Ordering::Relaxed) == 0 {
            if consumed == last_progress_consumed {
                break;
            }
            last_progress_consumed = consumed;
        }

        thread::sleep(Duration::from_millis(100));
    }

    shutdown_requested.store(true, Ordering::Relaxed);
    dispatcher.join().expect("dispatcher panicked");
    for handle in worker_handles {
        handle.join().expect("worker panicked");
    }
    coordinator.join().expect("coordinator panicked");

    shutdown.store(true, Ordering::Relaxed);
    logger.join().expect("logger panicked");

    println!("\nDone.");
    println!(
        "Expanded: {}",
        stats.boards_expanded.load(Ordering::Relaxed)
    );
    println!("Decisions: {}", graph.decisions.len());
    println!("Dead states: {}", graph.dead_states.len());
    println!("Frontier remaining: {}", frontier.len());
    println!("Interned states: {}", interner.len());
    println!("Elapsed: {:.2}s", start.elapsed().as_secs_f64());
}
#![feature(const_convert)]
#![feature(const_trait_impl)]
//! # Tetris Atlas Search With Backtracking
//!
//! In-memory atlas search that keeps alternatives per (board, bag, piece) decision and
//! backtracks when a child state is proven dead. This converges toward a closed surviving subgraph.

use clap::Parser;
use crossbeam_channel::{Receiver, Sender, bounded};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tetris_game::{
    IsLost, TetrisBoard, TetrisPiece, TetrisPieceBagState, TetrisPieceOrientation,
    TetrisPiecePlacement, repeat_idx_unroll,
};
use tetris_search::{BeamTetrisState, MultiBeamSearch, OrientationCounts};

// --- Beam Search Parameters ---
const N: usize = 8;
const TOP_N_PER_BEAM: usize = 32;
const BEAM_WIDTH: usize = 1024;
const MAX_DEPTH: usize = 4;
const MAX_MOVES: usize = TetrisPieceOrientation::TOTAL_NUM_ORIENTATIONS;
const TOP_K_CANDIDATES: usize = 3;

// --- Channel & Threading ---
const CHANNEL_CAPACITY: usize = 256;
const LOG_EVERY_SECS: u64 = 3;
const BASE_SEED: u64 = 42;
const CSV_FILE: &str = "tetris_atlas_search_with_backtracking.csv";

#[derive(Parser)]
#[command(name = "tetris-atlas-search-with-backtracking")]
#[command(about = "Tetris Atlas search with decision backtracking and dead-state propagation")]
struct Cli {
    /// Optional hard cap on frontier pops (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    max_expansions: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateKey {
    board: TetrisBoard,
    bag: TetrisPieceBagState,
}

impl StateKey {
    const fn new(board: TetrisBoard, bag: TetrisPieceBagState) -> Self {
        Self { board, bag }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DecisionKey {
    state: StateKey,
    piece: TetrisPiece,
}

impl DecisionKey {
    const fn new(state: StateKey, piece: TetrisPiece) -> Self {
        Self { state, piece }
    }
}

#[derive(Clone, Copy)]
struct FrontierValue {
    state: StateKey,
}

impl FrontierValue {
    const fn new(state: StateKey) -> Self {
        Self { state }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FrontierKey {
    priority: u32,
    hash: u64,
}

impl FrontierKey {
    fn new(state: StateKey) -> Self {
        let priority = state.board.count();
        let hash = hash_board_bag(state.board, state.bag);
        Self { priority, hash }
    }
}

impl Ord for FrontierKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => self.hash.cmp(&other.hash),
            other => other,
        }
    }
}

impl PartialOrd for FrontierKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline(always)]
const fn hash_board_bag(board: TetrisBoard, bag: TetrisPieceBagState) -> u64 {
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

    fn push(&self, state: StateKey) {
        let key = FrontierKey::new(state);
        let value = FrontierValue::new(state);
        if !self.map.contains_key(&key) {
            self.map.insert(key, value);
            self.enqueued_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn pop(&self) -> Option<StateKey> {
        self.map.pop_front().map(|entry| entry.value().state)
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    orientation: TetrisPieceOrientation,
    child: StateKey,
    votes: u32,
    dead: bool,
}

#[derive(Clone)]
struct DecisionRecord {
    candidates: Vec<Candidate>,
    selected_idx: Option<usize>,
    exhausted: bool,
}

impl DecisionRecord {
    fn new(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates,
            selected_idx: None,
            exhausted: false,
        }
    }
}

#[derive(Clone, Copy)]
struct ParentRef {
    decision_key: DecisionKey,
    candidate_idx: usize,
}

struct GraphStore {
    decisions: DashMap<DecisionKey, DecisionRecord>,
    parents: DashMap<StateKey, Vec<ParentRef>>,
    dead_states: DashMap<StateKey, ()>,
    state_first_decision_seq: DashMap<StateKey, u64>,
}

impl GraphStore {
    fn new() -> Self {
        Self {
            decisions: DashMap::new(),
            parents: DashMap::new(),
            dead_states: DashMap::new(),
            state_first_decision_seq: DashMap::new(),
        }
    }

    fn is_state_dead(&self, state: StateKey) -> bool {
        self.dead_states.contains_key(&state)
    }
}

#[derive(Default)]
struct Stats {
    frontier_consumed: AtomicU64,
    boards_expanded: AtomicU64,
    decisions_created: AtomicU64,
    decisions_exhausted: AtomicU64,
    dead_states_marked: AtomicU64,
    backtrack_switches: AtomicU64,
    candidate_losses_immediate: AtomicU64,
    lookup_hits: AtomicU64,
    candidate_total_raw: AtomicU64,
    candidate_unique_child_total: AtomicU64,
    duplicate_candidates_total: AtomicU64,
    dead_lag_decisions_sum: AtomicU64,
    dead_lag_samples: AtomicU64,
}

struct CsvLogger {
    file: BufWriter<File>,
}

impl CsvLogger {
    fn new(path: &str) -> std::io::Result<Self> {
        let file_exists = Path::new(path).exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut writer = BufWriter::new(file);

        if !file_exists {
            writeln!(
                writer,
                "timestamp_secs,boards_expanded,frontier_size,in_flight,\
                 frontier_enqueued,frontier_consumed,decisions_created,decisions_total,\
                 decisions_exhausted,dead_states,switches,immediate_losses,lookup_hits,\
                 boards_per_sec,frontier_in_rate,frontier_out_rate,frontier_ratio,\
                 decision_per_expanded,hit_rate,avg_unique_children_per_decision,\
                 duplicate_candidate_rate,avg_dead_lag_decisions,dead_lag_samples"
            )?;
        }

        Ok(Self { file: writer })
    }

    fn log(
        &mut self,
        frontier: &PriorityFrontier,
        stats: &Stats,
        processing_count: &AtomicU64,
        graph: &GraphStore,
        start: Instant,
    ) -> std::io::Result<()> {
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let frontier_size = frontier.len();
        let in_flight = processing_count.load(Ordering::Relaxed);
        let frontier_enqueued = frontier.enqueued_count.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_total = graph.decisions.len() as u64;
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = graph.dead_states.len() as u64;
        let switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let candidate_total_raw = stats.candidate_total_raw.load(Ordering::Relaxed);
        let candidate_unique_child_total =
            stats.candidate_unique_child_total.load(Ordering::Relaxed);
        let duplicate_candidates_total = stats.duplicate_candidates_total.load(Ordering::Relaxed);
        let dead_lag_decisions_sum = stats.dead_lag_decisions_sum.load(Ordering::Relaxed);
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);

        let boards_per_sec = boards_expanded as f64 / secs;
        let frontier_in_rate = frontier_enqueued as f64 / secs;
        let frontier_out_rate = frontier_consumed as f64 / secs;
        let frontier_ratio = frontier_enqueued as f64 / frontier_consumed.max(1) as f64;
        let decision_per_expanded = decisions_created as f64 / boards_expanded.max(1) as f64;
        let hit_rate = lookup_hits as f64 / decisions_created.max(1) as f64;
        let avg_unique_children_per_decision =
            candidate_unique_child_total as f64 / decisions_created.max(1) as f64;
        let duplicate_candidate_rate =
            duplicate_candidates_total as f64 / candidate_total_raw.max(1) as f64;
        let avg_dead_lag_decisions = dead_lag_decisions_sum as f64 / dead_lag_samples.max(1) as f64;

        writeln!(
            self.file,
            "{:.3},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.6},{:.6},{:.4},{}",
            secs,
            boards_expanded,
            frontier_size,
            in_flight,
            frontier_enqueued,
            frontier_consumed,
            decisions_created,
            decisions_total,
            decisions_exhausted,
            dead_states,
            switches,
            immediate_losses,
            lookup_hits,
            boards_per_sec,
            frontier_in_rate,
            frontier_out_rate,
            frontier_ratio,
            decision_per_expanded,
            hit_rate,
            avg_unique_children_per_decision,
            duplicate_candidate_rate,
            avg_dead_lag_decisions,
            dead_lag_samples
        )?;
        self.file.flush()?;
        Ok(())
    }
}

struct WorkerPieceEval {
    piece: TetrisPiece,
    candidates: Vec<Candidate>,
}

struct WorkerResult {
    state: StateKey,
    per_piece: Vec<WorkerPieceEval>,
}

fn dispatcher_thread(
    frontier: Arc<PriorityFrontier>,
    sender: Sender<StateKey>,
    shutdown_requested: Arc<AtomicBool>,
    processing_count: Arc<AtomicU64>,
    stats: Arc<Stats>,
) {
    while !shutdown_requested.load(Ordering::Relaxed) {
        processing_count.fetch_add(1, Ordering::SeqCst);
        match frontier.pop() {
            Some(state) => {
                stats.frontier_consumed.fetch_add(1, Ordering::Relaxed);
                if sender.send(state).is_err() {
                    processing_count.fetch_sub(1, Ordering::Relaxed);
                    break;
                }
            }
            None => {
                processing_count.fetch_sub(1, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
    drop(sender);
}

fn orientation_candidates_desc(counts: OrientationCounts) -> Vec<(TetrisPieceOrientation, u32)> {
    let mut out: Vec<(TetrisPieceOrientation, u32)> = counts.nonzero_orientations().collect();
    out.sort_by_key(|(_, votes)| Reverse(*votes));
    if out.len() > TOP_K_CANDIDATES {
        out.truncate(TOP_K_CANDIDATES);
    }
    out
}

fn evaluate_state(
    state: StateKey,
    beam_search: &mut MultiBeamSearch<
        BeamTetrisState,
        N,
        TOP_N_PER_BEAM,
        BEAM_WIDTH,
        MAX_DEPTH,
        MAX_MOVES,
    >,
    stats: &Stats,
) -> WorkerResult {
    stats.boards_expanded.fetch_add(1, Ordering::Relaxed);

    let mut game = tetris_game::TetrisGame::new();
    let mut per_piece = Vec::new();
    // let (depth, beam_width) = match state.board.count() {
    //     0..=8 => (2, 64),
    //     10..=12 => (4, 256),
    //     14..=16 => (6, 512),
    //     18..=22 => (8, 1024),
    //     _ => (10, 2048),
    // };
    let (depth, beam_width) = (MAX_DEPTH, BEAM_WIDTH);

    for (piece, next_bag) in state.bag.iter_next_states() {
        game.board = state.board;
        game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);

        let counts = beam_search.search_count_actions_with_seeds(
            BeamTetrisState::new(game),
            BASE_SEED,
            depth,
            beam_width,
        );
        let ranked = orientation_candidates_desc(counts);
        let mut candidates = Vec::with_capacity(ranked.len());

        for (orientation, votes) in ranked {
            game.board = state.board;
            game.set_bag_piece_seeded(state.bag, piece, BASE_SEED);
            let placement = TetrisPiecePlacement { piece, orientation };
            let result = game.apply_placement(placement);
            if result.is_lost == IsLost::LOST {
                stats
                    .candidate_losses_immediate
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            candidates.push(Candidate {
                orientation,
                child: StateKey::new(game.board, next_bag),
                votes,
                dead: false,
            });
        }

        per_piece.push(WorkerPieceEval { piece, candidates });
    }

    WorkerResult { state, per_piece }
}

fn mark_state_dead_and_propagate(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    stats: &Stats,
    initial_dead_state: StateKey,
) {
    let mut queue = VecDeque::new();
    queue.push_back(initial_dead_state);

    while let Some(dead_state) = queue.pop_front() {
        if graph.dead_states.insert(dead_state, ()).is_some() {
            continue;
        }
        stats.dead_states_marked.fetch_add(1, Ordering::Relaxed);
        if let Some(seq_ref) = graph.state_first_decision_seq.get(&dead_state) {
            let created_now = stats.decisions_created.load(Ordering::Relaxed);
            let lag = created_now.saturating_sub(*seq_ref);
            stats
                .dead_lag_decisions_sum
                .fetch_add(lag, Ordering::Relaxed);
            stats.dead_lag_samples.fetch_add(1, Ordering::Relaxed);
        }

        let parent_refs = graph
            .parents
            .get(&dead_state)
            .map(|r| r.clone())
            .unwrap_or_default();

        for parent_ref in parent_refs {
            let mut should_enqueue_parent_dead = false;
            let mut newly_selected_child: Option<StateKey> = None;
            let mut switched = false;

            if let Some(mut dec) = graph.decisions.get_mut(&parent_ref.decision_key) {
                if parent_ref.candidate_idx < dec.candidates.len() {
                    dec.candidates[parent_ref.candidate_idx].dead = true;
                }

                let selected_is_dead = dec.selected_idx.is_none_or(|idx| {
                    dec.candidates
                        .get(idx)
                        .is_none_or(|c| c.dead || graph.is_state_dead(c.child))
                });

                if selected_is_dead {
                    let next = dec
                        .candidates
                        .iter()
                        .enumerate()
                        .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child))
                        .map(|(idx, c)| (idx, c.child));

                    match next {
                        Some((next_idx, child)) => {
                            switched = dec.selected_idx != Some(next_idx);
                            dec.selected_idx = Some(next_idx);
                            dec.exhausted = false;
                            newly_selected_child = Some(child);
                        }
                        None => {
                            if !dec.exhausted {
                                dec.exhausted = true;
                                should_enqueue_parent_dead = true;
                                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

            if switched {
                stats.backtrack_switches.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(child) = newly_selected_child {
                frontier.push(child);
            }
            if should_enqueue_parent_dead {
                queue.push_back(parent_ref.decision_key.state);
            }
        }
    }
}

fn coordinator_handle_result(
    graph: &GraphStore,
    frontier: &PriorityFrontier,
    stats: &Stats,
    result: WorkerResult,
) {
    if graph.is_state_dead(result.state) {
        return;
    }

    for piece_eval in result.per_piece {
        let dkey = DecisionKey::new(result.state, piece_eval.piece);

        if graph.decisions.contains_key(&dkey) {
            stats.lookup_hits.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let mut decision = DecisionRecord::new(piece_eval.candidates);
        let created_seq = stats.decisions_created.fetch_add(1, Ordering::Relaxed) + 1;
        graph
            .state_first_decision_seq
            .entry(result.state)
            .or_insert(created_seq);

        let raw_count = decision.candidates.len() as u64;
        let unique_count = decision
            .candidates
            .iter()
            .map(|c| c.child)
            .collect::<HashSet<_>>()
            .len() as u64;
        let dup_count = raw_count.saturating_sub(unique_count);
        stats
            .candidate_total_raw
            .fetch_add(raw_count, Ordering::Relaxed);
        stats
            .candidate_unique_child_total
            .fetch_add(unique_count, Ordering::Relaxed);
        stats
            .duplicate_candidates_total
            .fetch_add(dup_count, Ordering::Relaxed);

        for (idx, candidate) in decision.candidates.iter().enumerate() {
            let pref = ParentRef {
                decision_key: dkey,
                candidate_idx: idx,
            };
            graph
                .parents
                .entry(candidate.child)
                .and_modify(|v| v.push(pref))
                .or_insert_with(|| vec![pref]);
        }

        let selected = decision
            .candidates
            .iter()
            .enumerate()
            .find(|(_, c)| !c.dead && !graph.is_state_dead(c.child))
            .map(|(idx, c)| (idx, c.child));

        match selected {
            Some((idx, child)) => {
                decision.selected_idx = Some(idx);
                frontier.push(child);
            }
            None => {
                decision.exhausted = true;
                stats.decisions_exhausted.fetch_add(1, Ordering::Relaxed);
            }
        }

        let exhausted = decision.exhausted;
        graph.decisions.insert(dkey, decision);
        if exhausted {
            mark_state_dead_and_propagate(graph, frontier, stats, result.state);
        }
    }
}

fn coordinator_thread(
    result_rx: Receiver<WorkerResult>,
    graph: Arc<GraphStore>,
    frontier: Arc<PriorityFrontier>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
) {
    while let Ok(result) = result_rx.recv() {
        coordinator_handle_result(&graph, &frontier, &stats, result);
        processing_count.fetch_sub(1, Ordering::Relaxed);
    }
}

fn logger_thread(
    frontier: Arc<PriorityFrontier>,
    stats: Arc<Stats>,
    processing_count: Arc<AtomicU64>,
    graph: Arc<GraphStore>,
    shutdown: Arc<AtomicBool>,
    start: Instant,
    csv_file: String,
) {
    let mut csv_logger = match CsvLogger::new(&csv_file) {
        Ok(logger) => {
            println!("CSV logging to: {csv_file}");
            Some(logger)
        }
        Err(e) => {
            eprintln!("Failed to open CSV logger at {csv_file}: {e}");
            None
        }
    };

    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_secs(LOG_EVERY_SECS));

        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let boards_expanded = stats.boards_expanded.load(Ordering::Relaxed);
        let decisions_created = stats.decisions_created.load(Ordering::Relaxed);
        let decisions_exhausted = stats.decisions_exhausted.load(Ordering::Relaxed);
        let dead_states = stats.dead_states_marked.load(Ordering::Relaxed);
        let backtrack_switches = stats.backtrack_switches.load(Ordering::Relaxed);
        let immediate_losses = stats.candidate_losses_immediate.load(Ordering::Relaxed);
        let frontier_consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        let lookup_hits = stats.lookup_hits.load(Ordering::Relaxed);
        let avg_unique_children_per_decision =
            stats.candidate_unique_child_total.load(Ordering::Relaxed) as f64
                / decisions_created.max(1) as f64;
        let duplicate_candidate_rate = stats.duplicate_candidates_total.load(Ordering::Relaxed)
            as f64
            / stats.candidate_total_raw.load(Ordering::Relaxed).max(1) as f64;
        let avg_dead_lag_decisions = stats.dead_lag_decisions_sum.load(Ordering::Relaxed) as f64
            / stats.dead_lag_samples.load(Ordering::Relaxed).max(1) as f64;
        let dead_lag_samples = stats.dead_lag_samples.load(Ordering::Relaxed);

        println!(
            "t={secs:.1}s expanded={boards_expanded} frontier={} in_flight={} consumed={} | \
             decisions={} exhausted={} dead_states={} switches={} immediate_losses={} hits={} | \
             uniq/dec={avg_unique_children_per_decision:.3} dup_rate={duplicate_candidate_rate:.3} \
             dead_lag={avg_dead_lag_decisions:.2} n_dead_lag={dead_lag_samples}",
            frontier.len(),
            processing_count.load(Ordering::Relaxed),
            frontier_consumed,
            decisions_created,
            decisions_exhausted,
            dead_states,
            backtrack_switches,
            immediate_losses,
            lookup_hits
        );

        if frontier.len() == 0 && processing_count.load(Ordering::Relaxed) == 0 {
            let _ = graph.decisions.len();
        }

        if let Some(ref mut logger) = csv_logger {
            if let Err(e) = logger.log(&frontier, &stats, &processing_count, &graph, start) {
                eprintln!("Failed to write CSV row: {e}");
            }
        }
    }

    if let Some(ref mut logger) = csv_logger {
        if let Err(e) = logger.log(&frontier, &stats, &processing_count, &graph, start) {
            eprintln!("Failed to write final CSV row: {e}");
        }
    }
}

fn main() {
    let cli = Cli::parse();
    run_tetris_atlas_search_with_backtracking(cli.max_expansions);
}

pub fn run_tetris_atlas_search_with_backtracking(max_expansions: u64) {
    let num_workers = num_cpus::get().saturating_sub(2).max(1);
    println!("Starting tetris_atlas_search_with_backtracking");
    println!("Workers: {num_workers} | Channel capacity: {CHANNEL_CAPACITY}");
    if max_expansions > 0 {
        println!("Max expansions: {max_expansions}");
    }
    println!("CSV file: {CSV_FILE}");

    let graph = Arc::new(GraphStore::new());
    let frontier_enqueued = Arc::new(AtomicU64::new(0));
    let frontier = Arc::new(PriorityFrontier::new(frontier_enqueued));
    let stats = Arc::new(Stats::default());
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));
    let processing_count = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    // Seed empty state.
    frontier.push(StateKey::new(
        TetrisBoard::new(),
        TetrisPieceBagState::new(),
    ));

    let (work_tx, work_rx) = bounded::<StateKey>(CHANNEL_CAPACITY);
    let (result_tx, result_rx) = bounded::<WorkerResult>(CHANNEL_CAPACITY);

    let shutdown_requested_clone = shutdown_requested.clone();
    ctrlc::set_handler(move || {
        println!("\nCtrl+C received, stopping...");
        shutdown_requested_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl+C handler");

    let dispatcher = {
        let frontier = frontier.clone();
        let shutdown_requested = shutdown_requested.clone();
        let processing_count = processing_count.clone();
        let stats = stats.clone();
        thread::spawn(move || {
            dispatcher_thread(
                frontier,
                work_tx,
                shutdown_requested,
                processing_count,
                stats,
            );
        })
    };

    let coordinator = {
        let graph = graph.clone();
        let frontier = frontier.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        thread::spawn(move || {
            coordinator_thread(result_rx, graph, frontier, stats, processing_count);
        })
    };

    let logger = {
        let frontier = frontier.clone();
        let stats = stats.clone();
        let processing_count = processing_count.clone();
        let graph = graph.clone();
        let shutdown = shutdown.clone();
        let csv_file = CSV_FILE.to_string();
        thread::spawn(move || {
            logger_thread(
                frontier,
                stats,
                processing_count,
                graph,
                shutdown,
                start,
                csv_file,
            );
        })
    };

    let mut worker_handles = Vec::with_capacity(num_workers);
    for _worker_id in 0..num_workers {
        let rx = work_rx.clone();
        let tx = result_tx.clone();
        let stats = stats.clone();
        worker_handles.push(thread::spawn(move || {
            let mut beam_search = MultiBeamSearch::<
                BeamTetrisState,
                N,
                TOP_N_PER_BEAM,
                BEAM_WIDTH,
                MAX_DEPTH,
                MAX_MOVES,
            >::new();

            while let Ok(state) = rx.recv() {
                let result = evaluate_state(state, &mut beam_search, &stats);
                if tx.send(result).is_err() {
                    break;
                }
            }
        }));
    }

    drop(result_tx);

    let mut last_progress_consumed = 0u64;
    loop {
        if shutdown_requested.load(Ordering::Relaxed) {
            break;
        }

        let consumed = stats.frontier_consumed.load(Ordering::Relaxed);
        if max_expansions > 0 && consumed >= max_expansions {
            println!("Reached max_expansions={max_expansions}");
            break;
        }

        if frontier.len() == 0 && processing_count.load(Ordering::Relaxed) == 0 {
            if consumed == last_progress_consumed {
                break;
            }
            last_progress_consumed = consumed;
        }

        thread::sleep(Duration::from_millis(100));
    }

    shutdown_requested.store(true, Ordering::Relaxed);
    dispatcher.join().expect("dispatcher panicked");
    for handle in worker_handles {
        handle.join().expect("worker panicked");
    }
    coordinator.join().expect("coordinator panicked");

    shutdown.store(true, Ordering::Relaxed);
    logger.join().expect("logger panicked");

    println!("\nDone.");
    println!(
        "Expanded: {}",
        stats.boards_expanded.load(Ordering::Relaxed)
    );
    println!("Decisions: {}", graph.decisions.len());
    println!("Dead states: {}", graph.dead_states.len());
    println!("Frontier remaining: {}", frontier.len());
    println!("Elapsed: {:.2}s", start.elapsed().as_secs_f64());
}
*/
