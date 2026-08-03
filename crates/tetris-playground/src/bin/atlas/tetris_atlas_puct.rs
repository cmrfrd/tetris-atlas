#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
//! # Tetris Atlas PUCT — integer-UCT graph search + gfp safe-set extraction
//!
//! Grows a canonical AND-OR state graph over complete Tetris states
//! `(board, remaining_bag, current_piece)`: an explicit expansion frontier
//! guarantees coverage, integer-only UCT rollouts prioritize promising and
//! dangerous regions, and a periodic greatest-fixpoint closure pass extracts
//! the candidate safe set
//! `K = { s | ∃ placement a, ∀ next piece q ∈ bag(s): T(s,a,q) ∈ K }`.
//!
//! Design spec: docs/superpowers/specs/2026-07-13-tetris-atlas-puct-design.md
//!
//! Memory is paid up-front: three preallocated arenas (~5 KB per state at the
//! configured `--max-states`) hold fixed-size records addressed by dense u32
//! ids. The DashMap is used only for interning; rollouts traverse cached ids.
//! The whole search is deterministic — no RNG anywhere; only thread scheduling
//! perturbs visit statistics.

use clap::Parser;
use crossbeam::utils::Backoff;
use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use tetris_game::{
    StandardTetris, TetrisBoard, TetrisGameConfig, TetrisPiece, TetrisPieceBagState,
    TetrisPiecePlacement,
};

// ============================================================================
// Config & CLI
// ============================================================================

#[derive(Parser, Debug, Clone)]
#[command(name = "tetris-atlas-puct")]
#[command(about = "PUCT-style graph search + gfp closure for a candidate Tetris safe set")]
struct Args {
    /// Worker threads (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Number of epochs to run (default: until Ctrl-C).
    #[arg(long)]
    epochs: Option<u64>,
    /// Rollouts per epoch.
    #[arg(long, default_value_t = 100_000)]
    rollouts_per_epoch: u64,
    /// Frontier states fully expanded per epoch.
    #[arg(long, default_value_t = 10_000)]
    expansions_per_epoch: u64,
    /// Run a closure pass every N epochs (0 is treated as 1).
    #[arg(long, default_value_t = 10)]
    closure_every: u64,
    /// Rollout horizon in pieces. Statistics are only comparable within one horizon.
    #[arg(long, default_value_t = 128)]
    horizon: usize,
    /// Depth-credit saturation, in pieces (clamped to horizon). A node's
    /// backed-up credit is min(continued_depth, cap) and all means normalize
    /// by the cap, so "survived ≥ cap deeper" scores 1.0 — keeping the value
    /// term on the same scale as the c=1.0 exploration/uncertainty terms.
    #[arg(long, default_value_t = 16)]
    credit_cap: u32,
    /// State arena capacity. ~5 KB of up-front arena memory per state.
    #[arg(long, default_value_t = 500_000)]
    max_states: u32,
    /// Placement arena capacity (default: max_states * 34).
    #[arg(long)]
    max_placements: Option<u32>,
    /// Branch arena capacity (default: placements * 7).
    #[arg(long)]
    max_branches: Option<u32>,
    /// UCT exploration constant, fixed-point with 20 fractional bits (1_048_576 = 1.0).
    #[arg(long, default_value_t = 1_048_576)]
    c_uct: u64,
    /// LCB uncertainty constant, fixed-point with 20 fractional bits.
    #[arg(long, default_value_t = 1_048_576)]
    c_lcb: u64,
    /// Carrier: max total holes (u32::MAX = unbounded).
    #[arg(long, default_value_t = u32::MAX)]
    max_holes: u32,
    /// Carrier: max stack height.
    #[arg(long, default_value_t = u32::MAX)]
    max_height: u32,
    /// Carrier: max surface roughness (sum of adjacent |Δheight|).
    #[arg(long, default_value_t = u32::MAX)]
    max_roughness: u32,
    /// Expansion policy: `all` queues every interned state; `candidate` queues on first rollout visit.
    #[arg(long, value_enum, default_value_t = ExpansionPolicyArg::All)]
    expansion_policy: ExpansionPolicyArg,
    /// Stop the epoch's frontier-expansion batch once any arena's occupancy
    /// reaches this percent of capacity, reserving the remaining headroom for
    /// rollout-driven lazy edge expansion (100 = no throttle).
    #[arg(long, default_value_t = 70)]
    expand_occupancy: u32,
    /// Candidate mode: admit a state to the expansion frontier on its Nth
    /// rollout visit (1 = first touch; roots are always admitted).
    #[arg(long, default_value_t = 1)]
    admit_after: u64,
    /// Compact at an epoch barrier when any arena's occupancy reaches this
    /// percent of capacity (also fires after a mid-epoch exhaustion).
    /// 100 = only on exhaustion.
    #[arg(long, default_value_t = 90)]
    compact_at: u32,
    /// Survivor target after compaction, as a percent of max-states capacity
    /// (roots and the current K are always kept, even above target).
    /// A proof-throughput dial: 30-40 evicts harder, reaching more fresh
    /// territory per epoch (+17-27% ledger growth) at the cost of K0
    /// coverage (-20-37%); 50 favors closing K; ≥70 leaves no room to
    /// explore and collapses both.
    #[arg(long, default_value_t = 50)]
    keep_frac: u32,
    /// Aging: right-shift applied to every copied visit/reward counter per
    /// compaction (1 = halve; 0 = pure cumulative LFU). Default 0: when the
    /// arena is tight, compaction fires every epoch and any nonzero shift
    /// halves ALL evidence every epoch — cumulative LFU keeps the survivor
    /// set stable so death proofs compound (+33% dead on the reference
    /// benchmark vs shift 1).
    #[arg(long, default_value_t = 0)]
    decay_shift: u32,
    /// Disable compaction: restore terminal-exhaustion exit (A/B comparisons).
    #[arg(long, default_value_t = false)]
    no_compact: bool,
    /// Percent of each rollout chunk seeded at the 7 roots; the rest start
    /// at near-miss boundary states (when any exist). 100 = roots only.
    /// Default 75: root-heavy sampling reaches fresh territory each
    /// generation (A/B 25/50/75 — 75 grew the dead ledger 9% faster and the
    /// near-miss boundary 6% thicker than 50; 25 was worst on both).
    #[arg(long, default_value_t = 75)]
    root_seed_pct: u32,
    /// Optional CSV metrics output path.
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Persistent death-ledger path: loaded at startup (if present) and saved
    /// at exit. Death facts depend only on the carrier + rules, so runs with
    /// the SAME carrier accumulate proofs across invocations; a mismatched
    /// carrier header is refused (soundness).
    #[arg(long)]
    ledger: Option<PathBuf>,
    /// Autosave the ledger every N epochs (0 = only at exit). Written to a
    /// .tmp sibling then renamed, so a crash never corrupts the ledger.
    #[arg(long, default_value_t = 25)]
    ledger_save_every: u64,
    /// Rollout piece selection: worst-case adversary (proof steering) or
    /// balanced canonical dynamics (play-strength measurement).
    #[arg(long, value_enum, default_value_t = PiecePolicyArg::Adversarial)]
    piece_policy: PiecePolicyArg,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ExpansionPolicyArg {
    All,
    Candidate,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum PiecePolicyArg {
    /// Worst-case: the adversary plays the min-LCB piece (proof-steering).
    Adversarial,
    /// Canonical-ish: deterministic least-visited piece (balanced bag
    /// coverage without RNG — the binary stays fully deterministic).
    Balanced,
}

// ============================================================================
// Domain (Task 1)
// ============================================================================

/// Edge/child sentinels. A valid arena `StateId` never equals a sentinel.
const UNEXPANDED: u32 = u32::MAX;
const EXPANDING: u32 = u32::MAX - 1;
/// Branch stamp: the (computed) child key is in the run-lifetime death
/// ledger, but the child is NOT materialized. Refutation and the death lfp
/// treat it as a proven-dead child at zero arena cost; resolving the edge
/// (rollout traversal) claims it like UNEXPANDED and materializes the child,
/// which the ledger then marks PROOF_DEAD at intern.
const DEAD_LEDGER: u32 = u32::MAX - 2;
const MAX_VALID_STATE_ID: u32 = u32::MAX - 3;

/// Game-truth proof status: facts about Tetris itself, not the current graph.
/// They never retract and survive compaction. DEAD = the adversary can force a
/// loss (least fixpoint); SAFE = gfp-certified member of a closed safe set.
const PROOF_UNKNOWN: u8 = 0;
const PROOF_DEAD: u8 = 1;
const PROOF_SAFE: u8 = 2;

/// A complete Tetris decision state: board, remaining-bag mask, and the visible
/// current piece. 42 bytes, `Copy`, no heap.
///
/// Canonical form (matches `TetrisPieceBagState::iter_next_states`): refill
/// happens at draw time, so `bag_bits` is never empty. Invariant:
/// `bag.contains(piece)  ⟹  bag == FULL_MASK` (a fresh refill legitimately
/// contains the just-drawn current piece; otherwise the draw removed it).
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct StateKey {
    board: TetrisBoard,
    bag_bits: u8,
    piece_idx: u8,
}

impl StateKey {
    fn new(board: TetrisBoard, bag: TetrisPieceBagState, piece: TetrisPiece) -> Self {
        // Canonical invariant: the current piece is absent from the remaining
        // bag unless a refill just restored the full bag. Real construction
        // sites (root_keys, expansion via child_bag) always satisfy this;
        // synthetic test helpers bypass `new` via a struct literal.
        debug_assert!(
            !bag.contains(piece) || u8::from(bag) == TetrisPieceBagState::FULL_MASK,
            "non-canonical StateKey: current piece in a non-full remaining bag"
        );
        Self {
            board,
            bag_bits: u8::from(bag),
            piece_idx: piece.index(),
        }
    }

    fn bag(&self) -> TetrisPieceBagState {
        TetrisPieceBagState::from(self.bag_bits)
    }

    fn piece(&self) -> TetrisPiece {
        TetrisPiece::from_index(self.piece_idx)
    }
}

/// Optional bound on the board universe. Restricting the *player's* placements
/// to stay inside the carrier is sound: it only weakens the player, so any safe
/// set found is a safe set of the unrestricted game.
#[derive(Clone, Copy, Debug)]
struct Carrier {
    max_holes: u32,
    max_height: u32,
    max_roughness: u32,
}

impl Carrier {
    const UNBOUNDED: Self = Self {
        max_holes: u32::MAX,
        max_height: u32::MAX,
        max_roughness: u32::MAX,
    };

    fn contains(&self, board: &TetrisBoard) -> bool {
        board.total_holes() <= self.max_holes
            && board.height() <= self.max_height
            && surface_roughness(board) <= self.max_roughness
    }
}

/// Total surface roughness (bumpiness): sum over adjacent columns of
/// |height[i] - height[i+1]|. Same definition as tetris_atlas_inmemory, for
/// comparability across runs.
fn surface_roughness(board: &TetrisBoard) -> u32 {
    let heights = board.heights();
    let mut rough = 0u32;
    for i in 0..StandardTetris::COLS - 1 {
        rough += heights[i].abs_diff(heights[i + 1]);
    }
    rough
}

/// Apply a placement. `None` means the move is not available to the player:
/// either it tops out or the resulting board leaves the carrier.
///
/// The result is independent of the *next* piece draw, so loss is decided per
/// placement, before piece branching — stored branches always point at live states.
fn apply_move(
    board: &TetrisBoard,
    placement: TetrisPiecePlacement,
    carrier: &Carrier,
) -> Option<TetrisBoard> {
    let mut next = *board;
    let result = next.apply_piece_placement(placement);
    if bool::from(result.is_lost) || !carrier.contains(&next) {
        return None;
    }
    Some(next)
}

/// The canonical bag transition: draw `piece`, refill if the bag empties.
/// Mirrors `TetrisPieceBagState::iter_next_states` for a single piece.
fn child_bag(bag: TetrisPieceBagState, piece: TetrisPiece) -> TetrisPieceBagState {
    let mut next = bag;
    next.remove(piece);
    if next.is_empty() {
        next.fill();
    }
    next
}

/// The seven initial states: empty board, full bag minus the first draw.
/// "init ∈ K" means all seven are in the extracted safe set.
fn root_keys() -> [StateKey; 7] {
    let pieces = TetrisPiece::all();
    pieces.map(|piece| {
        let mut bag = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        bag.remove(piece);
        StateKey::new(TetrisBoard::EMPTY_BOARD, bag, piece)
    })
}

/// Left-right mirror of a piece index: S↔Z, L↔J; O/I/T are self-mirror.
/// (Indices: O=0, I=1, S=2, Z=3, T=4, L=5, J=6.)
const fn mirror_piece_idx(idx: u8) -> u8 {
    match idx {
        2 => 3,
        3 => 2,
        5 => 6,
        6 => 5,
        x => x,
    }
}

/// Left-right mirror of a bag mask: swap the S/Z and L/J bits.
const fn mirror_bag_bits(bits: u8) -> u8 {
    let fixed = bits & 0b1001_0011;
    let s = (bits >> 2) & 1;
    let z = (bits >> 3) & 1;
    let l = (bits >> 5) & 1;
    let j = (bits >> 6) & 1;
    fixed | (z << 2) | (s << 3) | (j << 5) | (l << 6)
}

/// Left-right mirror of a complete state. Tetris is mirror-equivariant
/// (placement sets, gravity, and line clears all commute with the reflection
/// — pinned by `engine_is_mirror_equivariant`), so `dead(k) ⟺ dead(mirror(k))`.
fn mirror_key(key: &StateKey) -> StateKey {
    let limbs = key.board.as_limbs();
    let mut mirrored = limbs;
    mirrored.reverse();
    StateKey {
        board: TetrisBoard::from_limbs(mirrored),
        bag_bits: mirror_bag_bits(key.bag_bits),
        piece_idx: mirror_piece_idx(key.piece_idx),
    }
}

/// Canonical representative of a key's mirror pair (the Ord-smaller one).
/// The dead ledger stores and queries ONLY canonical keys: every proof
/// covers both orientations, and mirror-pair entries deduplicate.
fn ledger_canon(key: StateKey) -> StateKey {
    let m = mirror_key(&key);
    if m < key { m } else { key }
}

// ============================================================================
// Arenas (Task 2)
// ============================================================================

/// Structured capacity exhaustion — never silently wrap ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // all variants are naturally `*Exhausted`
enum CapacityError {
    StatesExhausted,
    PlacementsExhausted,
    BranchesExhausted,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let which = match self {
            Self::StatesExhausted => "state arena exhausted",
            Self::PlacementsExhausted => "placement arena exhausted",
            Self::BranchesExhausted => "branch arena exhausted",
        };
        f.write_str(which)
    }
}

impl std::error::Error for CapacityError {}

/// Preallocated write-once slot arena. `reserve` bumps atomically; each slot is
/// published exactly once via `OnceLock::set` and read via `get` only after its
/// id has been made reachable (interner map / branch child / frontier).
struct Arena<T> {
    slots: Box<[OnceLock<T>]>,
    next: AtomicU32,
}

impl<T> Arena<T> {
    fn new(capacity: u32) -> Self
    where
        T: Send,
    {
        debug_assert!(capacity <= MAX_VALID_STATE_ID);
        // Parallel init: this runs once per compaction GENERATION (fresh
        // graph), and the placement arena alone is millions of slots.
        let slots: Vec<OnceLock<T>> = (0..capacity)
            .into_par_iter()
            .map(|_| OnceLock::new())
            .collect();
        Self {
            slots: slots.into_boxed_slice(),
            next: AtomicU32::new(0),
        }
    }

    /// Reserve `count` contiguous slots; returns the base index, or `None` when
    /// the arena cannot satisfy the request. Overshoot of the bump pointer past
    /// capacity is harmless: the miss-path exhaustion fast-fail in `intern`
    /// bounds it to in-flight threads, and compaction discards the old arena
    /// (and its bump pointer) entirely.
    fn reserve(&self, count: u32) -> Option<u32> {
        let base = self.next.fetch_add(count, Ordering::Relaxed);
        if (base as u64) + (count as u64) > self.slots.len() as u64 {
            return None;
        }
        Some(base)
    }

    fn set(&self, idx: u32, value: T) {
        let published = self.slots[idx as usize].set(value).is_ok();
        debug_assert!(published, "arena slot {idx} double-set");
    }

    /// INVARIANT: callers only hold an id after its slot was published
    /// (payload-before-id, see SearchGraph::intern). The expect documents that
    /// invariant; it is the single non-test expect in this binary.
    fn get(&self, idx: u32) -> &T {
        self.slots[idx as usize]
            .get()
            .expect("arena slot read before publication")
    }

    /// Number of reserved slots (occupancy), capped at capacity.
    fn len(&self) -> u32 {
        self.next
            .load(Ordering::Acquire)
            .min(self.slots.len() as u32)
    }

    /// Total slot capacity (occupancy is `len`).
    fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }
}

// ============================================================================
// Records (Task 3)
// ============================================================================

const STATUS_QUEUED: u8 = 0;
const STATUS_EXPANDING: u8 = 1;
const STATUS_FULLY_EXPANDED: u8 = 2;

/// One canonical state. Structurally complete (placements enumerated, branch
/// blocks reserved) the moment it is published — `expansion_status` tracks only
/// child interning.
struct StateRecord {
    key: StateKey,
    visits: AtomicU64,
    reward_sum: AtomicU64,
    /// Base slot of `placement_len` contiguous PlacementRecords (surviving
    /// placements only; dead/out-of-carrier placements are not stored).
    placement_base: u32,
    /// 0 ⟹ terminally dead state (no placement keeps the board alive in-carrier).
    placement_len: u8,
    expansion_status: AtomicU8,
    /// 0/1: guarantees the frontier sees each state at most once.
    frontier_queued: AtomicU8,
    /// Game-truth proof byte: PROOF_UNKNOWN / PROOF_DEAD / PROOF_SAFE.
    /// Initialized PROOF_UNKNOWN; never retracted; persists across compaction.
    proof: AtomicU8,
}

struct PlacementRecord {
    placement: TetrisPiecePlacement,
    /// 7 contiguous branch slots; the slot for piece `q` is `branch_base + q.index()`.
    /// Slots for pieces not in `key.bag` stay UNEXPANDED forever.
    branch_base: u32,
    visits: AtomicU64,
    reward_sum: AtomicU64,
}

/// All-atomic 16-byte branch record (spec §16.12): plain pre-initialized arena,
/// no OnceLock. Saturating depth-reward accumulator — saturation under-counts,
/// which is pessimistic and therefore sound for LCB.
struct PieceBranchRecord {
    child: AtomicU32,
    reward_sum: AtomicU32,
    visits: AtomicU64,
}

// The 16-byte all-atomic layout is load-bearing for the arena memory budget
// (branch records dominate ~40:1). Catch any accidental field reorder/padding.
const _: () = assert!(core::mem::size_of::<PieceBranchRecord>() == 16);

impl PieceBranchRecord {
    fn new() -> Self {
        Self {
            child: AtomicU32::new(UNEXPANDED),
            reward_sum: AtomicU32::new(0),
            visits: AtomicU64::new(0),
        }
    }

    fn saturating_add_reward(&self, amount: u32) {
        let _ = self
            .reward_sum
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                v.checked_add(amount)
            });
    }
}

/// Branch arena: every slot is pre-initialized at construction (child =
/// UNEXPANDED), so publication needs no OnceLock — all fields are atomics.
struct BranchArena {
    slots: Box<[PieceBranchRecord]>,
    next: AtomicU32,
}

impl BranchArena {
    fn new(capacity: u32) -> Self {
        debug_assert!(capacity <= MAX_VALID_STATE_ID);
        // Parallel init: ~48M slots (≈762 MB of stores) per compaction
        // generation at default capacities — the single biggest turnover cost.
        let slots: Vec<PieceBranchRecord> = (0..capacity)
            .into_par_iter()
            .map(|_| PieceBranchRecord::new())
            .collect();
        Self {
            slots: slots.into_boxed_slice(),
            next: AtomicU32::new(0),
        }
    }

    fn reserve(&self, count: u32) -> Option<u32> {
        let base = self.next.fetch_add(count, Ordering::Relaxed);
        if (base as u64) + (count as u64) > self.slots.len() as u64 {
            return None;
        }
        Some(base)
    }

    fn get(&self, idx: u32) -> &PieceBranchRecord {
        &self.slots[idx as usize]
    }

    /// Number of reserved slots (occupancy), capped at capacity.
    fn len(&self) -> u32 {
        self.next
            .load(Ordering::Acquire)
            .min(self.slots.len() as u32)
    }

    /// Total slot capacity (occupancy is `len`).
    fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }
}

// ============================================================================
// Graph: interning + expansion (Tasks 3-4)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpansionPolicy {
    AllDiscovered,
    CandidateOnly,
}

#[derive(Default)]
struct Counters {
    rollouts: AtomicU64,
    deaths: AtomicU64,
    horizons: AtomicU64,
    cycles: AtomicU64,
    aborted: AtomicU64,
    /// Rollouts short-circuited by a gfp-certified PROOF_SAFE state.
    certified: AtomicU64,
    edges_expanded: AtomicU64,
    full_expansions: AtomicU64,
    max_rollout_depth: AtomicU32,
    rollout_depth_sum: AtomicU64,
}

struct InternResult {
    id: u32,
    newly_created: bool,
}

/// Number of DashMap shards (power of two). Hashing 42-byte keys is the hot
/// intern path; FxHash + generous sharding cuts contention (same configuration
/// as tetris_atlas_inmemory).
const DASHMAP_SHARDS: usize = 256;

struct SearchGraph {
    /// Interning only. Hot traversal never touches this map.
    ids: DashMap<StateKey, u32, FxBuildHasher>,
    states: Arena<StateRecord>,
    placements: Arena<PlacementRecord>,
    branches: BranchArena,
    frontier: ArrayQueue<u32>,
    policy: ExpansionPolicy,
    carrier: Carrier,
    exhausted: AtomicBool,
    counters: Counters,
    /// Run-lifetime death ledger, shared across compaction generations
    /// (compact clones the Arc). Death is game truth — it never retracts —
    /// so an evicted dead state's proof survives here (~50 B/key vs ~5 KB of
    /// arena) and a re-interned dead key materializes as PROOF_DEAD with
    /// ZERO placements: no re-proving, no placement/branch arena spend, and
    /// rollouts steer away on sight.
    dead_keys: Arc<DashMap<StateKey, (), FxBuildHasher>>,
}

impl SearchGraph {
    fn new(
        max_states: u32,
        max_placements: u32,
        max_branches: u32,
        carrier: Carrier,
        policy: ExpansionPolicy,
    ) -> Self {
        Self {
            ids: DashMap::with_hasher_and_shard_amount(FxBuildHasher, DASHMAP_SHARDS),
            states: Arena::new(max_states),
            placements: Arena::new(max_placements),
            branches: BranchArena::new(max_branches),
            frontier: ArrayQueue::new(max_states.max(1) as usize),
            policy,
            carrier,
            exhausted: AtomicBool::new(false),
            counters: Counters::default(),
            dead_keys: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher,
                DASHMAP_SHARDS,
            )),
        }
    }

    /// Insert a death fact (canonicalized: covers both mirror orientations).
    fn ledger_insert(&self, key: StateKey) {
        self.dead_keys.insert(ledger_canon(key), ());
    }

    /// Is this key (or its mirror) a recorded death fact?
    fn ledger_contains(&self, key: &StateKey) -> bool {
        self.dead_keys.contains_key(&ledger_canon(*key))
    }

    fn note_exhausted(&self, e: CapacityError) -> CapacityError {
        self.exhausted.store(true, Ordering::SeqCst);
        e
    }

    /// Canonical interning: one StateId per StateKey, all statistics shared
    /// across histories. The record is fully built and published into the arena
    /// BEFORE its id enters the map (payload-before-id — a consumer that can
    /// see the id can always `get` the record).
    fn intern(&self, key: StateKey) -> Result<InternResult, CapacityError> {
        if let Some(existing) = self.ids.get(&key) {
            return Ok(InternResult {
                id: *existing,
                newly_created: false,
            });
        }
        // Terminal-exhaustion fast-fail — MISS PATH ONLY. Existing keys must
        // always resolve (rollouts keep traversing the interior after
        // exhaustion, and expansions whose children all already exist can
        // still complete). This avoids shard locking and arena bump inflation
        // (up to 238 branch slots per failed intern) from doomed misses.
        if self.exhausted.load(Ordering::SeqCst) {
            return Err(CapacityError::StatesExhausted);
        }
        let id = match self.ids.entry(key) {
            Entry::Occupied(entry) => {
                return Ok(InternResult {
                    id: *entry.get(),
                    newly_created: false,
                });
            }
            Entry::Vacant(entry) => {
                let record = self
                    .build_state_record(key)
                    .map_err(|e| self.note_exhausted(e))?;
                let id = self
                    .states
                    .reserve(1)
                    .ok_or_else(|| self.note_exhausted(CapacityError::StatesExhausted))?;
                self.states.set(id, record);
                entry.insert(id);
                id
            }
        };
        if self.policy == ExpansionPolicy::AllDiscovered {
            self.mark_queued_and_push(id);
        }
        Ok(InternResult {
            id,
            newly_created: true,
        })
    }

    /// Structural enumeration: filter the current piece's placements through
    /// `apply_move`, storing only survivors, each with 7 UNEXPANDED branch slots.
    ///
    /// Ledger short-circuit: a key already proven dead this run materializes
    /// as PROOF_DEAD with zero placements — its placements/branches are never
    /// consulted again (rollouts die on the proof byte; the death lfp seeds
    /// from it), so re-proving and arena spend are both skipped.
    fn build_state_record(&self, key: StateKey) -> Result<StateRecord, CapacityError> {
        if self.ledger_contains(&key) {
            return Ok(StateRecord {
                key,
                visits: AtomicU64::new(0),
                reward_sum: AtomicU64::new(0),
                placement_base: 0,
                placement_len: 0,
                expansion_status: AtomicU8::new(STATUS_FULLY_EXPANDED),
                frontier_queued: AtomicU8::new(1), // never expand: nothing to do
                proof: AtomicU8::new(PROOF_DEAD),
            });
        }
        let mut surviving =
            [TetrisPiecePlacement::from_index(0); TetrisPiecePlacement::MAX_PIECE_PLACEMENT_COUNT];
        let mut n: u32 = 0;
        for &placement in TetrisPiecePlacement::all_from_piece(key.piece()) {
            if apply_move(&key.board, placement, &self.carrier).is_some() {
                surviving[n as usize] = placement;
                n += 1;
            }
        }
        if n == 0 {
            // Natural death (no surviving placement): a game-truth fact —
            // ledger it NOW so same-epoch expansions stamp against it,
            // instead of waiting for the next barrier's lfp to find it.
            self.ledger_insert(key);
        }
        let placement_base = if n == 0 {
            0
        } else {
            let base = self
                .placements
                .reserve(n)
                .ok_or(CapacityError::PlacementsExhausted)?;
            let branch_base_all = self
                .branches
                .reserve(n * 7)
                .ok_or(CapacityError::BranchesExhausted)?;
            for i in 0..n {
                self.placements.set(
                    base + i,
                    PlacementRecord {
                        placement: surviving[i as usize],
                        branch_base: branch_base_all + i * 7,
                        visits: AtomicU64::new(0),
                        reward_sum: AtomicU64::new(0),
                    },
                );
            }
            base
        };
        Ok(StateRecord {
            key,
            visits: AtomicU64::new(0),
            reward_sum: AtomicU64::new(0),
            placement_base,
            placement_len: n as u8,
            expansion_status: AtomicU8::new(STATUS_QUEUED),
            frontier_queued: AtomicU8::new(0),
            // Natural deaths are proven on sight (keeps the invariant that
            // the in-graph record of a ledgered key is always PROOF_DEAD).
            proof: AtomicU8::new(if n == 0 { PROOF_DEAD } else { PROOF_UNKNOWN }),
        })
    }

    /// Queue a state for full expansion at most once, ever. Zero-placement
    /// states are dead ends — expanding them is a no-op, so they never enter
    /// the frontier (the queued-once flag is set to keep the invariant that
    /// a state is CONSIDERED at most once).
    fn mark_queued_and_push(&self, id: u32) {
        let state = self.states.get(id);
        if state.placement_len == 0 {
            state.frontier_queued.store(1, Ordering::Release);
            return;
        }
        if state
            .frontier_queued
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Queued-once + capacity == max_states makes push infallible; a
            // failure would indicate a broken invariant, so count it visibly.
            if self.frontier.push(id).is_err() {
                self.counters.aborted.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Claim-and-publish one `(s, a, q)` edge. Loop shape handles all races:
    /// valid child -> return it; UNEXPANDED -> try to claim; EXPANDING (or lost
    /// claim) -> back off and re-read. On capacity error the claim is rolled
    /// back to UNEXPANDED so no edge is ever stuck EXPANDING.
    fn expand_edge(
        &self,
        branch: &PieceBranchRecord,
        child_board: TetrisBoard,
        child_bag_state: TetrisPieceBagState,
        piece: TetrisPiece,
    ) -> Result<u32, CapacityError> {
        let backoff = Backoff::new();
        loop {
            let current = branch.child.load(Ordering::Acquire);
            if current <= MAX_VALID_STATE_ID {
                return Ok(current);
            }
            if current == DEAD_LEDGER {
                // Ledgered dead child: terminal fact, nothing to expand.
                return Ok(DEAD_LEDGER);
            }
            if current == UNEXPANDED
                && branch
                    .child
                    .compare_exchange(current, EXPANDING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                let key = StateKey::new(child_board, child_bag_state, piece);
                // Ledger short-circuit: a known-dead child never materializes.
                // The stamp refutes the owning placement (adversary wins by
                // playing this piece) at zero state/placement/branch cost —
                // accumulated proofs directly prune the NEW graph's frontier.
                if self.ledger_contains(&key) {
                    branch.child.store(DEAD_LEDGER, Ordering::Release);
                    return Ok(DEAD_LEDGER);
                }
                return match self.intern(key) {
                    Ok(result) => {
                        branch.child.store(result.id, Ordering::Release);
                        self.counters.edges_expanded.fetch_add(1, Ordering::Relaxed);
                        Ok(result.id)
                    }
                    Err(e) => {
                        // Roll back to what we claimed from (a DEAD_LEDGER
                        // stamp is a persistent fact — never erase it).
                        branch.child.store(current, Ordering::Release);
                        Err(self.note_exhausted(e))
                    }
                };
            }
            backoff.snooze();
        }
    }

    /// Recompute the successor board for a stored (surviving) placement and
    /// resolve the branch child, expanding the edge on first touch.
    fn resolve_child(
        &self,
        state: &StateRecord,
        pr: &PlacementRecord,
        br: &PieceBranchRecord,
        piece: TetrisPiece,
    ) -> Result<u32, CapacityError> {
        let cached = br.child.load(Ordering::Acquire);
        if cached <= MAX_VALID_STATE_ID || cached == DEAD_LEDGER {
            return Ok(cached);
        }
        // Stored placements survived structural enumeration; recompute is
        // deterministic, so this cannot fail. Skip defensively if it somehow
        // does (an unexpanded branch is treated as unsafe everywhere).
        let Some(child_board) = apply_move(&state.key.board, pr.placement, &self.carrier) else {
            debug_assert!(false, "stored placement failed re-application");
            return Err(self.note_exhausted(CapacityError::StatesExhausted));
        };
        self.expand_edge(br, child_board, child_bag(state.key.bag(), piece), piece)
    }

    /// Full expansion: intern every `(a, q ∈ bag)` child. One function, two
    /// drivers (frontier batches and rollouts). Losers of the status CAS wait
    /// until the winner publishes FULLY_EXPANDED. On capacity error the status
    /// is still published so waiters never spin; the FULLY_EXPANDED label
    /// persists across generations with unexpanded branches that heal lazily
    /// via rollouts; closure never trusts the label — it re-checks every
    /// branch child, so a partially expanded state simply fails the ∀ and is
    /// pruned.
    fn ensure_fully_expanded(&self, id: u32) -> Result<(), CapacityError> {
        let state = self.states.get(id);
        match state.expansion_status.compare_exchange(
            STATUS_QUEUED,
            STATUS_EXPANDING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                let mut result = Ok(());
                'placements: for slot in 0..state.placement_len as u32 {
                    let pr = self.placements.get(state.placement_base + slot);
                    let Some(child_board) =
                        apply_move(&state.key.board, pr.placement, &self.carrier)
                    else {
                        debug_assert!(false, "stored placement failed re-application");
                        continue;
                    };
                    for (piece, bag) in state.key.bag().iter_next_states() {
                        let br = self.branches.get(pr.branch_base + piece.index() as u32);
                        if let Err(e) = self.expand_edge(br, child_board, bag, piece) {
                            result = Err(e);
                            break 'placements;
                        }
                    }
                }
                state
                    .expansion_status
                    .store(STATUS_FULLY_EXPANDED, Ordering::Release);
                self.counters
                    .full_expansions
                    .fetch_add(1, Ordering::Relaxed);
                // Inline single-level death inference: if every placement now
                // has a proven-dead (materialized or DEAD_LEDGER-stamped) bag
                // branch, the state is dead by the same rule as the barrier
                // lfp — record it NOW so same-epoch rollouts steer away and
                // same-epoch expansions of its parents stamp against it.
                // Proof bytes are monotone (UNKNOWN→DEAD) and the adversarial
                // proof-kill pass already tolerates concurrent proof writes.
                if result.is_ok()
                    && state.placement_len > 0
                    && state.proof.load(Ordering::Acquire) == PROOF_UNKNOWN
                {
                    let all_refuted = (0..state.placement_len as u32).all(|slot| {
                        let pr = self.placements.get(state.placement_base + slot);
                        state.key.bag().iter_pieces().any(|piece| {
                            let child = self
                                .branches
                                .get(pr.branch_base + piece.index() as u32)
                                .child
                                .load(Ordering::Acquire);
                            child == DEAD_LEDGER
                                || (child <= MAX_VALID_STATE_ID
                                    && self.states.get(child).proof.load(Ordering::Acquire)
                                        == PROOF_DEAD)
                        })
                    });
                    if all_refuted {
                        state.proof.store(PROOF_DEAD, Ordering::Release);
                        self.ledger_insert(state.key);
                    }
                }
                result
            }
            Err(STATUS_EXPANDING) => {
                let backoff = Backoff::new();
                while state.expansion_status.load(Ordering::Acquire) != STATUS_FULLY_EXPANDED {
                    if self.exhausted.load(Ordering::SeqCst) {
                        return Err(CapacityError::StatesExhausted);
                    }
                    backoff.snooze();
                }
                Ok(())
            }
            Err(_) => Ok(()), // already FULLY_EXPANDED
        }
    }
}

// ============================================================================
// Integer scoring (Task 5)
// ============================================================================

/// Fixed-point scale: 20 fractional bits. A probability of 1.0 is SCORE_ONE.
/// No floating-point types exist anywhere in search code — only u64/u128 arithmetic.
const SCORE_SHIFT: u32 = 20;
const SCORE_ONE: u64 = 1 << SCORE_SHIFT;

/// successes/visits in fixed point. u128 intermediates: no overflow for any u64 inputs.
fn mean_fixed(successes: u64, visits: u64) -> u64 {
    if visits == 0 {
        return 0;
    }
    (((successes as u128) << SCORE_SHIFT) / (visits as u128)) as u64
}

/// c * sqrt(log2(parent+1) / action_visits), all fixed point.
/// log2 replaces ln (the base change is absorbed by c); std u128::isqrt is the
/// exact floor root, so sqrt(log << 2*SHIFT) is the root in SHIFT fixed point.
///
/// The `action_visits == 0` arm returns `u64::MAX / 4` as a fully defensive
/// sentinel — with FPU, no production caller passes 0 (unvisited placements are
/// scored via the FPU branch in `choose_search_placement` instead). The arm is
/// retained for safety in case future callers change this invariant.
fn exploration_bonus(parent_visits: u64, action_visits: u64, c: u64) -> u64 {
    if action_visits == 0 {
        return u64::MAX / 4;
    }
    let log = parent_visits.saturating_add(1).ilog2() as u128;
    let radicand = (log << (2 * SCORE_SHIFT)) / (action_visits as u128);
    let sqrt_fixed = radicand.isqrt() as u64;
    (((c as u128) * (sqrt_fixed as u128)) >> SCORE_SHIFT) as u64
}

/// Lower-confidence bound on a piece branch: depth-reward mean minus uncertainty,
/// floored at 0. Unvisited branches score 0 — unknown is never treated as safe.
/// The parent count for the uncertainty term is the owning placement's visits.
/// `credit_cap` is the depth-credit saturation: per-visit credit is capped
/// there, so the mean normalizes to 1.0 at "always survived ≥ cap deeper".
fn branch_lcb(
    branch_visits: u64,
    branch_reward: u64,
    placement_visits: u64,
    c_lcb: u64,
    credit_cap: u64,
) -> u64 {
    if branch_visits == 0 {
        return 0;
    }
    mean_fixed(
        branch_reward,
        branch_visits.saturating_mul(credit_cap.max(1)),
    )
    .saturating_sub(exploration_bonus(placement_visits, branch_visits, c_lcb))
}

// ============================================================================
// Selection (Task 6)
// ============================================================================

/// Per-run search parameters (immutable after startup).
struct SearchParams {
    horizon: usize,
    /// Depth-credit saturation (1..=horizon). Backup credit and every mean
    /// normalization use this, NOT the horizon.
    credit_cap: u64,
    c_uct: u64,
    c_lcb: u64,
    admit_after: u64,
    piece_policy: PiecePolicyArg,
}

/// A placement is REFUTED iff some bag piece leads to a proven-dead child —
/// the adversary plays that piece; the placement is a theorem-level loss.
/// Unknown/unexpanded children never refute (conservative).
fn placement_refuted(graph: &SearchGraph, state: &StateRecord, pr: &PlacementRecord) -> bool {
    for piece in state.key.bag().iter_pieces() {
        let child = graph
            .branches
            .get(pr.branch_base + piece.index() as u32)
            .child
            .load(Ordering::Acquire);
        if child == DEAD_LEDGER
            || (child <= MAX_VALID_STATE_ID
                && graph.states.get(child).proof.load(Ordering::Acquire) == PROOF_DEAD)
        {
            return true;
        }
    }
    false
}

/// Conservative placement value: the weakest bag-piece branch (adversary picks
/// the piece). Unexpanded/unvisited branches contribute 0 via branch_lcb.
/// Single fused scan (this is the selection hot path — no separate
/// refutation pre-pass re-reading the same child/proof atomics).
///
/// Proof bounds override statistics in both directions:
/// - child DEAD → the adversary plays that piece; the placement is REFUTED
///   → `None` (a theorem-level loss, never selectable).
/// - child SAFE → this branch contributes SCORE_ONE (certainty; skip the LCB).
/// - else → branch_lcb as usual.
fn placement_value(
    graph: &SearchGraph,
    state: &StateRecord,
    pr: &PlacementRecord,
    c_lcb: u64,
    credit_cap: u64,
) -> Option<u64> {
    let placement_visits = pr.visits.load(Ordering::Relaxed);
    let mut min = u64::MAX;
    for piece in state.key.bag().iter_pieces() {
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let child = br.child.load(Ordering::Acquire);
        if child == DEAD_LEDGER {
            return None; // refuted: ledgered dead child, unmaterialized
        }
        let branch_val = if child <= MAX_VALID_STATE_ID {
            let child_proof = graph.states.get(child).proof.load(Ordering::Acquire);
            if child_proof == PROOF_DEAD {
                return None; // refuted: adversary plays this piece
            }
            if child_proof == PROOF_SAFE {
                // Proven safe: certainty, no statistical uncertainty to penalize.
                SCORE_ONE
            } else {
                branch_lcb(
                    br.visits.load(Ordering::Relaxed),
                    br.reward_sum.load(Ordering::Relaxed) as u64,
                    placement_visits,
                    c_lcb,
                    credit_cap,
                )
            }
        } else {
            branch_lcb(
                br.visits.load(Ordering::Relaxed),
                br.reward_sum.load(Ordering::Relaxed) as u64,
                placement_visits,
                c_lcb,
                credit_cap,
            )
        };
        min = min.min(branch_val);
    }
    Some(min) // bag is never empty, so min was set
}

/// Does any bag-piece branch of this placement have a valid (expanded) child?
/// Post-exhaustion a placement with none is unsampleable even when visited —
/// compaction keeps high-visit placements while evicting their children
/// (edges reset to UNEXPANDED), and stepping into one can only abort.
fn placement_has_expanded_branch(
    graph: &SearchGraph,
    state: &StateRecord,
    pr: &PlacementRecord,
) -> bool {
    state.key.bag().iter_pieces().any(|piece| {
        graph
            .branches
            .get(pr.branch_base + piece.index() as u32)
            .child
            .load(Ordering::Acquire)
            <= MAX_VALID_STATE_ID
    })
}

/// UCT selection (exploration included). Argmax(value + bonus) with
/// deterministic ties: higher score > fewer visits > lower slot.
///
/// First-play urgency (FPU): unvisited placements compete at half the parent's
/// own depth-reward mean (tempered optimism) plus a first-visit bonus, instead
/// of force-winning selection. This concentrates rollouts on known-good lines
/// and lets UCT pull in untried placements only when the region's evidence or
/// the visit imbalance justifies it.
fn choose_search_placement(
    graph: &SearchGraph,
    state: &StateRecord,
    params: &SearchParams,
) -> Option<u32> {
    let n = state.placement_len as u32;
    if n == 0 {
        return None;
    }
    let parent_visits = state.visits.load(Ordering::Relaxed);
    let cap = params.credit_cap;
    // First-play urgency: unvisited placements compete at half the parent's
    // own depth-reward mean (tempered optimism) plus a first-visit bonus,
    // instead of force-winning selection. This concentrates rollouts on
    // known-good lines and lets UCT pull in untried placements only when the
    // region's evidence or the visit imbalance justifies it.
    let parent_mean = mean_fixed(
        state.reward_sum.load(Ordering::Relaxed),
        state
            .visits
            .load(Ordering::Relaxed)
            .saturating_mul(cap.max(1)),
    );
    // Post-exhaustion, an unvisited placement is unsampleable: none of its
    // branches are expanded, so stepping into it can only abort. Skipping
    // keeps rollouts inside the expanded subgraph (choose_piece applies the
    // matching gate at the piece level).
    let exhausted = graph.exhausted.load(Ordering::SeqCst);
    let mut best: Option<(u64, u64, u32)> = None; // (score, visits, slot)
    for slot in 0..n {
        let pr = graph.placements.get(state.placement_base + slot);
        let visits = pr.visits.load(Ordering::Relaxed);
        let score = if visits == 0 {
            if exhausted || placement_refuted(graph, state, pr) {
                continue;
            }
            (parent_mean / 2).saturating_add(exploration_bonus(parent_visits, 1, params.c_uct))
        } else {
            // Visited yet childless (compaction evicted the children):
            // unsampleable post-exhaustion — redirect to a live sibling.
            if exhausted && !placement_has_expanded_branch(graph, state, pr) {
                continue;
            }
            // Fused scan: None = refuted (never selectable).
            let Some(value) = placement_value(graph, state, pr, params.c_lcb, cap) else {
                continue;
            };
            value.saturating_add(exploration_bonus(parent_visits, visits, params.c_uct))
        };
        let better = match best {
            None => true,
            Some((bs, bv, _)) => score > bs || (score == bs && visits < bv),
        };
        if better {
            best = Some((score, visits, slot));
        }
    }
    best.map(|(_, _, slot)| slot)
}

/// Solver view: pure worst-case exploitation, no exploration term. This is the
/// placement a policy/witness would play.
fn choose_best_placement(
    graph: &SearchGraph,
    state: &StateRecord,
    params: &SearchParams,
) -> Option<u32> {
    let n = state.placement_len as u32;
    if n == 0 {
        return None;
    }
    let mut best: Option<(u64, u64, u32)> = None;
    for slot in 0..n {
        let pr = graph.placements.get(state.placement_base + slot);
        let visits = pr.visits.load(Ordering::Relaxed);
        let Some(value) = placement_value(graph, state, pr, params.c_lcb, params.credit_cap) else {
            continue; // refuted
        };
        let better = match best {
            None => true,
            Some((bs, bv, _)) => value > bs || (value == bs && visits < bv),
        };
        if better {
            best = Some((value, visits, slot));
        }
    }
    best.map(|(_, _, slot)| slot)
}

/// The search's own worst-case reward estimate at the roots, in permille of
/// certainty (0..=1000 fixed-point conversion): for each root, the value of
/// the solver's best placement (min-over-bag LCB); dead/unscored roots are 0.
/// Returns (min, mean) over the 7 roots — min is the binding constraint.
fn root_values_permille(
    graph: &SearchGraph,
    roots: &[u32; 7],
    params: &SearchParams,
) -> (u64, u64) {
    let mut values = [0u64; 7];
    for (i, &root) in roots.iter().enumerate() {
        let state = graph.states.get(root);
        if let Some(slot) = choose_best_placement(graph, state, params) {
            let pr = graph.placements.get(state.placement_base + slot);
            values[i] = (placement_value(graph, state, pr, params.c_lcb, params.credit_cap)
                .unwrap_or(0)
                * 1000)
                >> SCORE_SHIFT;
        }
    }
    let min = values.iter().copied().min().unwrap_or(0);
    let mean = values.iter().sum::<u64>() / 7;
    (min, mean)
}

/// Adversarial next-piece selection: (0) proof-kill — any bag branch with a
/// valid PROOF_DEAD child is an immediate win for the adversary; (1) first
/// unexpanded bag branch, (2) first unvisited, (3) minimum LCB — dangerous
/// and uncertain branches get the visits.
/// Deterministic: bag iteration order is piece-index order.
///
/// SAFE children flow through the LCB path (their high value makes the
/// adversary avoid them naturally — no special case needed here).
///
/// Note: pass 1 (unexpanded) and pass 2 (unvisited) collapse into a single loop
/// only because an unexpanded branch always has 0 visits — the `child` check
/// runs before the `visits` check per piece, and iteration order is fixed, so
/// precedence unexpanded > unvisited > min-LCB holds piece-by-piece. If an
/// earlier piece is unvisited and a later one unexpanded, the earlier unvisited
/// one returns first — that is acceptable because both cases mean "no
/// information; must sample", and the plan's ordering intent (never run UCT
/// before local coverage exists) is preserved.
///
/// Balanced next-piece selection: (1) first unexpanded branch (edge creation
/// required), then (2) least-visited branch with ties broken toward lower piece
/// index. No proof-kill pass and no LCB — canonical bags neither seek nor avoid
/// killers; a proven-dead piece can come up like any other (the rollout then
/// dies at the child, which is honest sampling of realistic dynamics).
fn choose_piece(
    graph: &SearchGraph,
    state: &StateRecord,
    pr: &PlacementRecord,
    c_lcb: u64,
    credit_cap: u64,
    policy: PiecePolicyArg,
) -> TetrisPiece {
    // Once the arena is exhausted, unexpanded branches can never be resolved:
    // choosing one only aborts the rollout and backs up an artifact death.
    // Post-exhaustion, both policies therefore steer through EXPANDED children
    // only, spending the remaining rollout budget refining interior statistics
    // instead of hammering an unresolvable frontier. If no branch of the
    // chosen placement is expanded, fall through to an unexpanded piece — the
    // rollout aborts there exactly as before (rare; pessimistic; sound).
    let exhausted = graph.exhausted.load(Ordering::SeqCst);
    match policy {
        PiecePolicyArg::Adversarial => {
            // Defense-in-depth: selection pre-filters refuted placements and
            // proofs are barrier-frozen, so this pass is normally unreachable
            // from rollouts — it guards any future concurrent proof writer and
            // is pinned directly by its unit test.
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                let child = br.child.load(Ordering::Acquire);
                if child == DEAD_LEDGER
                    || (child <= MAX_VALID_STATE_ID
                        && graph.states.get(child).proof.load(Ordering::Acquire) == PROOF_DEAD)
                {
                    return piece;
                }
            }
            let placement_visits = pr.visits.load(Ordering::Relaxed);
            let mut fallback = state.key.piece(); // unreachable in practice: bag is never empty
            let mut first_unexpanded: Option<TetrisPiece> = None;
            let mut best: Option<(u64, TetrisPiece)> = None;
            for (i, piece) in state.key.bag().iter_pieces().enumerate() {
                if i == 0 {
                    fallback = piece;
                }
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                if br.child.load(Ordering::Acquire) > MAX_VALID_STATE_ID {
                    if !exhausted {
                        return piece; // pass 1: unexpanded
                    }
                    if first_unexpanded.is_none() {
                        first_unexpanded = Some(piece);
                    }
                    continue; // exhausted: unknown children are unsampleable
                }
                let visits = br.visits.load(Ordering::Relaxed);
                if visits == 0 {
                    return piece; // pass 2: unvisited (valid because expansion precedes visits)
                }
                let lcb = branch_lcb(
                    visits,
                    br.reward_sum.load(Ordering::Relaxed) as u64,
                    placement_visits,
                    c_lcb,
                    credit_cap,
                );
                if best.is_none_or(|(bl, _)| lcb < bl) {
                    best = Some((lcb, piece));
                }
            }
            best.map(|(_, piece)| piece)
                .or(first_unexpanded)
                .unwrap_or(fallback)
        }
        PiecePolicyArg::Balanced => {
            // Pass 1: unexpanded branch — edge creation still required before
            // any statistics can be collected (skipped once the arena is
            // exhausted; see above).
            let mut first_unexpanded: Option<TetrisPiece> = None;
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                if br.child.load(Ordering::Acquire) > MAX_VALID_STATE_ID {
                    if !exhausted {
                        return piece;
                    }
                    if first_unexpanded.is_none() {
                        first_unexpanded = Some(piece);
                    }
                }
            }
            // Pass 2: least-visited branch, ties toward lower piece index.
            // No proof-kill and no LCB: canonical bags neither seek nor avoid
            // killers; a proven-dead piece can come up like any other (the
            // rollout then dies at the child, which is honest sampling).
            // Post-exhaustion only expanded branches compete.
            let mut fallback = state.key.piece(); // unreachable in practice: bag is never empty
            let mut best: Option<(u64, TetrisPiece)> = None;
            for (i, piece) in state.key.bag().iter_pieces().enumerate() {
                if i == 0 {
                    fallback = piece;
                }
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                if exhausted && br.child.load(Ordering::Acquire) > MAX_VALID_STATE_ID {
                    continue;
                }
                let visits = br.visits.load(Ordering::Relaxed);
                // Strict less-than: ties keep the earlier (lower-index) piece.
                if best.is_none_or(|(bv, _)| visits < bv) {
                    best = Some((visits, piece));
                }
            }
            best.map(|(_, piece)| piece)
                .or(first_unexpanded)
                .unwrap_or(fallback)
        }
    }
}

// ============================================================================
// Rollout (Task 7)
// ============================================================================

/// One step of a recorded rollout path: enough to find the state, placement,
/// and branch records again at backup time.
struct PathEntry {
    state: u32,
    placement_slot: u32,
    piece_idx: u8,
}

/// Per-worker reusable scratch. All capacity is allocated once; rollouts only
/// clear. (`for_each_init` may build one of these per rayon job split — the
/// no-allocation guarantee is per-rollout, not per-run.)
struct WorkerScratch {
    path: Vec<PathEntry>,
}

impl WorkerScratch {
    fn new(horizon: usize) -> Self {
        Self {
            path: Vec::with_capacity(horizon + 1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RolloutOutcome {
    /// Reached a state with no surviving placement.
    Death,
    /// Survived the full configured horizon.
    Horizon,
    /// Revisited a StateId within this rollout — scored as full-horizon reward
    /// (spec §9): a loop under adversarial-leaning piece selection is the
    /// strongest empirical signal available; counted separately in metrics.
    Cycle,
    /// Capacity exhaustion mid-walk; scored as a pessimistic death at the
    /// truncation point; no entry for the failed edge.
    Aborted,
    /// Reached a gfp-certified PROOF_SAFE state — proof, not statistics;
    /// earns full-horizon reward.
    Certified,
}

/// One rollout from `root`: UCT placement, adversarial piece, cached-id
/// traversal (edges expand on first touch). No allocation, no floats, no RNG.
fn rollout(
    graph: &SearchGraph,
    params: &SearchParams,
    root: u32,
    scratch: &mut WorkerScratch,
) -> RolloutOutcome {
    scratch.path.clear();
    let cap32 = params.credit_cap as u32;
    let mut current = root;
    loop {
        if scratch.path.len() >= params.horizon {
            backup(graph, &scratch.path, current, params.horizon as u32, cap32);
            return RolloutOutcome::Horizon;
        }
        // Cycle check: linear scan of the recorded path (≤ horizon entries,
        // contiguous u32 loads) — cheaper than per-step hashing at these
        // depths, and one less scratch structure to maintain.
        if scratch.path.iter().any(|e| e.state == current) {
            backup(graph, &scratch.path, current, params.horizon as u32, cap32);
            return RolloutOutcome::Cycle;
        }
        let state = graph.states.get(current);
        // Proof short-circuits: these fire before the placement_len == 0 check
        // because proofs subsume it (proofs are facts about Tetris, not the graph).
        let proof = state.proof.load(Ordering::Acquire);
        if proof == PROOF_SAFE {
            // Certified safe: proof beats statistics — full-horizon credit,
            // and the rollout budget moves to the uncertain boundary.
            backup(graph, &scratch.path, current, params.horizon as u32, cap32);
            return RolloutOutcome::Certified;
        }
        if proof == PROOF_DEAD {
            // Proven loss: credit only the depth survived so far.
            backup(
                graph,
                &scratch.path,
                current,
                scratch.path.len() as u32,
                cap32,
            );
            return RolloutOutcome::Death;
        }
        if state.placement_len == 0 {
            backup(
                graph,
                &scratch.path,
                current,
                scratch.path.len() as u32,
                cap32,
            );
            return RolloutOutcome::Death;
        }
        if graph.policy == ExpansionPolicy::CandidateOnly
            && state.visits.load(Ordering::Relaxed) + 1 >= params.admit_after
        {
            graph.mark_queued_and_push(current); // admission by search visit
        }
        let Some(slot) = choose_search_placement(graph, state, params) else {
            // Either all placements are proof-refuted (dead in truth; the next
            // barrier's death-lfp will mark it PROOF_DEAD — we do not write
            // proofs from the rollout hot path), or the arena is exhausted and
            // every placement is unsampleable. Both score as pessimistic death.
            backup(
                graph,
                &scratch.path,
                current,
                scratch.path.len() as u32,
                cap32,
            );
            return RolloutOutcome::Death;
        };
        let pr = graph.placements.get(state.placement_base + slot);
        let piece = choose_piece(
            graph,
            state,
            pr,
            params.c_lcb,
            params.credit_cap,
            params.piece_policy,
        );
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let child = match graph.resolve_child(state, pr, br, piece) {
            Ok(DEAD_LEDGER) => {
                // Ledgered dead successor (defensive: selection pre-filters
                // refuted placements, so this is unreachable in practice).
                // Score: died one step past `current`.
                scratch.path.push(PathEntry {
                    state: current,
                    placement_slot: slot,
                    piece_idx: piece.index(),
                });
                let depth = scratch.path.len() as u32;
                backup(graph, &scratch.path, current, depth, cap32);
                return RolloutOutcome::Death;
            }
            Ok(id) => id,
            Err(_) => {
                // Capacity: the edge cannot be resolved. Score the line as a
                // death at `current` (unknown is never safe) so post-exhaustion
                // rollouts still steer away from the unresolvable frontier.
                backup(
                    graph,
                    &scratch.path,
                    current,
                    scratch.path.len() as u32,
                    cap32,
                );
                return RolloutOutcome::Aborted;
            }
        };
        scratch.path.push(PathEntry {
            state: current,
            placement_slot: slot,
            piece_idx: piece.index(),
        });
        current = child;
    }
}

/// Single-pass backup along the recorded path plus the terminal state.
/// Relaxed ordering throughout: counters publish nothing.
///
/// Remaining-depth credit: the entry at path index `i` receives `reward - i` —
/// how much deeper the rollout continued FROM that node — not the whole-path
/// total. Uniform whole-path credit made near-terminal nodes of deep lines
/// look nearly safe (a depth-60 branch whose child dies immediately scored
/// mean 60/64), poisoning the adversary's min-LCB. `reward >= i` for every
/// produced outcome (death/abort reward == path.len(); horizon/cycle/certified
/// reward == horizon >= path.len()); saturating_sub is defensive only.
/// Credit saturates at `cap` (the scoring normalizer): "continued ≥ cap
/// deeper" earns full marks, keeping means on the [0, 1] certainty scale.
fn backup(graph: &SearchGraph, path: &[PathEntry], final_state: u32, reward: u32, cap: u32) {
    let terminal = graph.states.get(final_state);
    terminal.visits.fetch_add(1, Ordering::Relaxed);
    terminal.reward_sum.fetch_add(
        reward.saturating_sub(path.len() as u32).min(cap) as u64,
        Ordering::Relaxed,
    );
    for (i, entry) in path.iter().enumerate() {
        let credit = reward.saturating_sub(i as u32).min(cap);
        let state = graph.states.get(entry.state);
        state.visits.fetch_add(1, Ordering::Relaxed);
        state.reward_sum.fetch_add(credit as u64, Ordering::Relaxed);
        let pr = graph
            .placements
            .get(state.placement_base + entry.placement_slot);
        pr.visits.fetch_add(1, Ordering::Relaxed);
        pr.reward_sum.fetch_add(credit as u64, Ordering::Relaxed);
        let br = graph.branches.get(pr.branch_base + entry.piece_idx as u32);
        br.visits.fetch_add(1, Ordering::Relaxed);
        br.saturating_add_reward(credit);
    }
}

impl Counters {
    fn record_rollout(&self, outcome: RolloutOutcome, depth: usize) {
        self.rollouts.fetch_add(1, Ordering::Relaxed);
        let bucket = match outcome {
            RolloutOutcome::Death => &self.deaths,
            RolloutOutcome::Horizon => &self.horizons,
            RolloutOutcome::Cycle => &self.cycles,
            RolloutOutcome::Aborted => &self.aborted,
            RolloutOutcome::Certified => &self.certified,
        };
        bucket.fetch_add(1, Ordering::Relaxed);
        self.max_rollout_depth
            .fetch_max(depth as u32, Ordering::Relaxed);
        self.rollout_depth_sum
            .fetch_add(depth as u64, Ordering::Relaxed);
    }
}

// ============================================================================
// Closure (Task 8)
// ============================================================================

/// Result of one gfp pass over a quiescent graph.
struct ClosureReport {
    k_size: usize,
    iterations: u32,
    roots_in_k: [bool; 7],
    /// Per state id: the witness placement slot, or u8::MAX if the state ∉ K.
    witnesses: Vec<u8>,
    /// |K0| = alive ∧ FULLY_EXPANDED states before any pruning.
    k0_size: usize,
    /// States removed by each gfp pass (last entry is the zero-removal pass).
    pass_removed: Vec<usize>,
    /// States OUTSIDE the final K whose best placement fails the ∀ by exactly
    /// ONE bag branch — the thickness of the "almost closed" boundary.
    near_miss: usize,
    /// The near-miss states themselves (pre-compaction ids, ascending,
    /// capped at BOUNDARY_ID_CAP). First-class output: compaction protects
    /// them (they are the K-frontier) and the rollout seeder reads them
    /// directly instead of re-scanning the graph.
    boundary_ids: Vec<u32>,
    /// Total states marked PROOF_DEAD by the backward death propagation lfp
    /// (including those seeded from prior proof state).
    dead_total: usize,
}

/// Cap on ClosureReport::boundary_ids (protection + seeding). Bounds the
/// compaction-protection cost of a thick boundary; the count metric is
/// NOT capped.
const BOUNDARY_ID_CAP: usize = 16_384;

/// Does `id` have a placement whose every bag-piece successor is in `k`?
/// Unknown (UNEXPANDED/EXPANDING) branches fail the ∀ — never treated as safe.
/// Returns the first qualifying slot (deterministic witness).
fn closure_qualifies(graph: &SearchGraph, id: u32, k: &[bool]) -> Option<u8> {
    let state = graph.states.get(id);
    // An empty bag would make the ∀ vacuously true and certify a state with
    // zero checked successors — unsound. Canonical keys can never have an
    // empty bag (child_bag refills), so this only guards synthetic misuse.
    debug_assert!(
        state.key.bag_bits != 0,
        "closure_qualifies on empty-bag state"
    );
    'placements: for slot in 0..state.placement_len {
        let pr = graph.placements.get(state.placement_base + slot as u32);
        for piece in state.key.bag().iter_pieces() {
            let child = graph
                .branches
                .get(pr.branch_base + piece.index() as u32)
                .child
                .load(Ordering::Acquire);
            if child > MAX_VALID_STATE_ID || !k[child as usize] {
                continue 'placements;
            }
        }
        return Some(slot);
    }
    None
}

/// Boundary extraction: states outside K whose BEST placement fails the ∀ by
/// exactly one bag branch (one unexpanded/outside-K child away from closing).
/// These are where expansion effort moves K next. One fused scan returns the
/// full count AND the member ids (ascending, capped at BOUNDARY_ID_CAP for
/// the protection/seeding consumers). Proven-dead states are excluded — a
/// dead state can trivially have a one-failure placement, but it is not a
/// boundary the safe set can ever absorb.
fn near_miss_boundary(graph: &SearchGraph, k: &[bool]) -> (usize, Vec<u32>) {
    let ids: Vec<u32> = (0..k.len() as u32)
        .into_par_iter()
        .filter(|&id| {
            if k[id as usize] {
                return false;
            }
            let state = graph.states.get(id);
            if state.placement_len == 0
                || state.expansion_status.load(Ordering::Acquire) != STATUS_FULLY_EXPANDED
                || state.proof.load(Ordering::Acquire) == PROOF_DEAD
            {
                return false;
            }
            let mut min_failures = usize::MAX;
            for slot in 0..state.placement_len as u32 {
                let pr = graph.placements.get(state.placement_base + slot);
                let mut failures = 0usize;
                for piece in state.key.bag().iter_pieces() {
                    let child = graph
                        .branches
                        .get(pr.branch_base + piece.index() as u32)
                        .child
                        .load(Ordering::Acquire);
                    if child > MAX_VALID_STATE_ID || !k[child as usize] {
                        failures += 1;
                    }
                }
                min_failures = min_failures.min(failures);
            }
            min_failures == 1
        })
        .collect();
    let count = ids.len();
    let mut ids = ids;
    ids.truncate(BOUNDARY_ID_CAP);
    (count, ids)
}

/// Reverse adjacency in CSR form: for every VALID branch child, one
/// (child -> parent-state) edge. Built once per closure barrier and shared by
/// the death lfp and the gfp worklist — the alternative (full graph rescans
/// per fixpoint pass) reloads every placement of every state each pass.
struct ReverseEdges {
    offsets: Vec<u32>,
    parents: Vec<u32>,
}

fn build_reverse_edges(graph: &SearchGraph) -> ReverseEdges {
    let n = graph.states.len() as usize;
    let indeg: Vec<AtomicU32> = (0..n + 1).map(|_| AtomicU32::new(0)).collect();
    let for_each_valid_child = |f: &(dyn Fn(u32, u32) + Sync)| {
        (0..n as u32).into_par_iter().for_each(|id| {
            let state = graph.states.get(id);
            for slot in 0..state.placement_len as u32 {
                let pr = graph.placements.get(state.placement_base + slot);
                for piece in state.key.bag().iter_pieces() {
                    let child = graph
                        .branches
                        .get(pr.branch_base + piece.index() as u32)
                        .child
                        .load(Ordering::Acquire);
                    if child <= MAX_VALID_STATE_ID {
                        f(child, id);
                    }
                }
            }
        });
    };
    for_each_valid_child(&|child, _| {
        indeg[child as usize + 1].fetch_add(1, Ordering::Relaxed);
    });
    let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
    let mut acc = 0u32;
    for d in &indeg {
        acc += d.load(Ordering::Relaxed);
        offsets.push(acc);
    }
    // offsets[child] = start cursor for child's parents; offsets[n] = total.
    let cursors: Vec<AtomicU32> = offsets[..n].iter().map(|&o| AtomicU32::new(o)).collect();
    let total = offsets[n] as usize;
    let parents: Vec<AtomicU32> = (0..total).map(|_| AtomicU32::new(0)).collect();
    for_each_valid_child(&|child, parent| {
        let at = cursors[child as usize].fetch_add(1, Ordering::Relaxed);
        parents[at as usize].store(parent, Ordering::Relaxed);
    });
    let parents: Vec<u32> = parents
        .into_iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();
    ReverseEdges { offsets, parents }
}

impl ReverseEdges {
    fn parents_of(&self, id: u32) -> &[u32] {
        let lo = self.offsets[id as usize] as usize;
        let hi = self.offsets[id as usize + 1] as usize;
        &self.parents[lo..hi]
    }
}

/// Is every placement of `state` refuted under `dead` flags (plus stamps)?
/// A placement is refuted iff SOME bag piece leads to a proven-dead child or
/// carries a DEAD_LEDGER stamp. Unknown/unexpanded children NEVER support
/// death (conservative).
fn state_all_refuted(graph: &SearchGraph, id: u32, dead: &[bool]) -> bool {
    let state = graph.states.get(id);
    (0..state.placement_len as u32).all(|slot| {
        let pr = graph.placements.get(state.placement_base + slot);
        state.key.bag().iter_pieces().any(|piece| {
            let child = graph
                .branches
                .get(pr.branch_base + piece.index() as u32)
                .child
                .load(Ordering::Acquire);
            child == DEAD_LEDGER || (child <= MAX_VALID_STATE_ID && dead[child as usize])
        })
    })
}

/// Backward death propagation (least fixpoint), worklist form. A placement is
/// REFUTED iff some bag piece leads to a proven-dead child (the adversary
/// just plays it); a state is dead iff it has no placements or every
/// placement is refuted. Seeds: zero-placement / already-proven states, PLUS
/// every state with a refuted-by-stamp-or-dead-child placement set (checked
/// once, in parallel); then death flows strictly through reverse edges — a
/// parent is rechecked only when one of its children dies. The lfp is unique,
/// so the result is deterministic. Persists PROOF_DEAD marks (and the
/// run-lifetime ledger); returns the total proven-dead count. Barrier-only.
fn propagate_death(graph: &SearchGraph, rev: &ReverseEdges) -> usize {
    let n = graph.states.len() as usize;
    let mut dead: Vec<bool> = (0..n as u32)
        .into_par_iter()
        .map(|id| {
            let state = graph.states.get(id);
            state.placement_len == 0 || state.proof.load(Ordering::Acquire) == PROOF_DEAD
        })
        .collect();
    // One full evaluation pass (parallel): catches stamp-refuted states and
    // states whose dead children materialized since the last barrier.
    let newly: Vec<u32> = (0..n as u32)
        .into_par_iter()
        .filter(|&id| !dead[id as usize] && state_all_refuted(graph, id, &dead))
        .collect();
    let mut queue: Vec<u32> = Vec::with_capacity(newly.len() + 1024);
    for id in newly {
        dead[id as usize] = true;
        queue.push(id);
    }
    // Also propagate from the seeds themselves (their parents were judged
    // against pre-seed flags in the pass above only if seed flags were set —
    // they were, so parents of seeds are already handled; parents of NEWLY
    // dead states are not).
    let mut head = 0usize;
    while head < queue.len() {
        let died = queue[head];
        head += 1;
        for &parent in rev.parents_of(died) {
            if !dead[parent as usize] && state_all_refuted(graph, parent, &dead) {
                dead[parent as usize] = true;
                queue.push(parent);
            }
        }
    }
    // Persist newly proven dead states (parallel; count via reduction).
    (0..n)
        .into_par_iter()
        .filter(|&id| dead[id])
        .map(|id| {
            let state = graph.states.get(id as u32);
            let prior = state.proof.load(Ordering::Acquire);
            if prior != PROOF_DEAD {
                debug_assert!(
                    prior != PROOF_SAFE,
                    "SAFE state proven dead — soundness bug"
                );
                state.proof.store(PROOF_DEAD, Ordering::Release);
                // Run-lifetime ledger: the proof survives eviction.
                graph.ledger_insert(state.key);
            }
            1usize
        })
        .sum()
}

/// Greatest-fixpoint pruning:
///   K0    = { alive ∧ fully expanded ∧ not proven-dead }
///   K_n+1 = { s ∈ K_n | ∃a ∀q∈bag(s): child(a,q) ∈ K_n }
/// iterated (rayon-parallel scans) until no state is removed. Runs only
/// between epochs — the graph is quiescent, no snapshot machinery needed.
/// `iterations` counts every pass, including the final zero-removal pass.
fn extract_safe_set(graph: &SearchGraph, roots: &[u32; 7]) -> ClosureReport {
    // Build the reverse adjacency once; the death lfp (and later the gfp
    // worklist) both consume it. Death runs first so K0 exclusion and SAFE
    // marking see fresh facts within the same call.
    let rev = build_reverse_edges(graph);
    let dead_total = propagate_death(graph, &rev);

    let n = graph.states.len() as usize;
    let mut k: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|id| {
            let state = graph.states.get(id as u32);
            state.placement_len > 0
                && state.expansion_status.load(Ordering::Acquire) == STATUS_FULLY_EXPANDED
                && state.proof.load(Ordering::Acquire) != PROOF_DEAD
        })
        .collect();
    let k0_size = k.iter().filter(|in_k| **in_k).count();
    // Level-synchronous gfp worklist. Pass 1 evaluates every K0 member; each
    // later level re-evaluates ONLY the still-in-K parents (via the shared
    // reverse adjacency) of the previous level's removals — a member can
    // first fail at level L only if one of its children left K at level L-1.
    // Removals apply at level barriers, so the per-level evaluation snapshot
    // (and therefore pass_removed / iterations) is bit-identical to the old
    // full-rescan loop at a fraction of the loads.
    let mut iterations = 0u32;
    let mut pass_removed: Vec<usize> = Vec::new();
    let members: Vec<u32> = (0..n as u32).filter(|&id| k[id as usize]).collect();
    let mut removed_now: Vec<u32> = {
        iterations += 1;
        members
            .par_iter()
            .copied()
            .filter(|&id| closure_qualifies(graph, id, &k).is_none())
            .collect()
    };
    pass_removed.push(removed_now.len());
    while !removed_now.is_empty() {
        for &id in &removed_now {
            k[id as usize] = false;
        }
        iterations += 1;
        let mut candidates: Vec<u32> = removed_now
            .iter()
            .flat_map(|&id| rev.parents_of(id).iter().copied())
            .filter(|&p| k[p as usize])
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        removed_now = candidates
            .par_iter()
            .copied()
            .filter(|&id| closure_qualifies(graph, id, &k).is_none())
            .collect();
        pass_removed.push(removed_now.len());
    }
    let witnesses: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|id| {
            if k[id] {
                closure_qualifies(graph, id as u32, &k).map_or(u8::MAX, |s| s)
            } else {
                u8::MAX
            }
        })
        .collect();
    // Mark every K member SAFE.
    for id in 0..n {
        if k[id] {
            let state = graph.states.get(id as u32);
            let prior = state.proof.load(Ordering::Acquire);
            if prior != PROOF_SAFE {
                debug_assert!(prior != PROOF_DEAD, "dead state entered K — soundness bug");
                state.proof.store(PROOF_SAFE, Ordering::Release);
            }
        }
    }
    let k_size = k.iter().filter(|in_k| **in_k).count();
    let roots_in_k = roots.map(|r| k.get(r as usize).copied().unwrap_or(false));
    let (near_miss, boundary_ids) = near_miss_boundary(graph, &k);
    ClosureReport {
        k_size,
        iterations,
        roots_in_k,
        witnesses,
        k0_size,
        pass_removed,
        near_miss,
        boundary_ids,
        dead_total,
    }
}

// ============================================================================
// Compaction (Phase 2, Tasks C1-C2)
// ============================================================================

/// Barrier-only compaction parameters (from CLI; percentages are integers —
/// the float ban applies).
struct CompactConfig {
    /// Survivor target as a percentage of the states arena CAPACITY.
    keep_frac_pct: u32,
    /// Right-shift applied to every copied visit/reward counter (aging).
    decay_shift: u32,
}

/// Choose which states survive a compaction generation. Deterministic.
///
/// Protected (always survive): the 7 roots and every member of the current
/// candidate safe set K (`protect.witnesses[id] != u8::MAX`) — protecting K
/// preserves it exactly across compaction (K is closed under its witness
/// placements, so witness children are themselves in K and protected).
///
/// The rest of the budget — up to `keep_target = states.capacity() *
/// keep_frac_pct / 100` total survivors — goes to the highest-visit
/// non-protected states (ties broken toward lower id). Dead states
/// (placement_len == 0) compete like any other: a hot dead state is a useful
/// cached death; cold ones evaporate.
fn select_survivors(
    graph: &SearchGraph,
    protect: &ClosureReport,
    roots: &[u32; 7],
    keep_frac_pct: u32,
) -> Vec<bool> {
    let n = graph.states.len() as usize;
    let keep_target = (graph.states.capacity() as u64 * keep_frac_pct as u64 / 100) as usize;
    let mut survive = vec![false; n];
    let mut protected_count = 0usize;
    // Mark K members.
    for (id, w) in protect.witnesses.iter().enumerate() {
        if id < n && *w != u8::MAX {
            if !survive[id] {
                survive[id] = true;
                protected_count += 1;
            }
        }
    }
    // Mark SAFE-proof states (belt for prior-generation SAFE beyond the current report's K).
    for id in 0..n {
        if !survive[id] && graph.states.get(id as u32).proof.load(Ordering::Acquire) == PROOF_SAFE {
            survive[id] = true;
            protected_count += 1;
        }
    }
    // Mark roots.
    for r in roots {
        let id = *r as usize;
        if id < n && !survive[id] {
            survive[id] = true;
            protected_count += 1;
        }
    }
    // Mark the near-miss boundary: these are the states K grows through next.
    // Visit-ranked selection is recency-blind and evicts them (fresh boundary
    // states have few visits), throwing away exactly the K-frontier.
    for id in &protect.boundary_ids {
        let id = *id as usize;
        if id < n && !survive[id] {
            survive[id] = true;
            protected_count += 1;
        }
    }
    // Fill remaining budget with highest-visit non-protected states.
    if protected_count < keep_target {
        let remaining = keep_target - protected_count;
        // Dead states never compete for the budget: their refutations
        // survive as DEAD_LEDGER stamps on parent edges plus ledger entries,
        // so keeping the records only caches what the stamp already proves —
        // every budget slot goes to live work instead.
        let mut candidates: Vec<(u64, u32)> = (0..n as u32)
            .filter(|&id| {
                !survive[id as usize]
                    && graph.states.get(id).proof.load(Ordering::Acquire) != PROOF_DEAD
            })
            .map(|id| {
                let visits = graph.states.get(id).visits.load(Ordering::Relaxed);
                (visits, id)
            })
            .collect();
        // Top-`remaining` by (visits desc, id asc) — deterministic and O(n):
        // partition at the cut instead of fully sorting ~n·log n candidates
        // per generation (only the surviving side's MEMBERSHIP matters, not
        // its order).
        let key = |c: &(u64, u32)| (std::cmp::Reverse(c.0), c.1);
        if remaining < candidates.len() {
            candidates.select_nth_unstable_by_key(remaining, key);
            candidates.truncate(remaining);
        }
        for (_, id) in &candidates {
            survive[*id as usize] = true;
        }
    }
    // NOTE: dead branch-children of survivors are deliberately NOT protected.
    // The run-lifetime dead ledger subsumes that evidence: when a compaction
    // resets an edge to an evicted dead child, the next rollout through the
    // parent re-resolves it and the ledger materializes the child as
    // PROOF_DEAD for one state slot and zero placement/branch spend. Keeping
    // dead records in-arena instead starved live exploration (the protected
    // set crossed the expansion throttle).
    survive
}

struct CompactionStats {
    kept: u32,
    evicted: u32,
    protected_k: usize,
    wall_ms: u64,
}

/// One mark-compact generation. BARRIER-ONLY: caller guarantees no concurrent
/// access to `old` (epoch phases are joined). Copies survivors into a fresh
/// graph with a deterministic old→new id remap; evicted-pointing edges reset
/// to UNEXPANDED (they heal lazily via rollouts); FULLY_EXPANDED labels are
/// kept (closure checks branches, never labels); every copied visit/reward
/// counter is aged by `cfg.decay_shift`. Cumulative Counters transfer; the
/// `exhausted` flag deliberately does NOT (the new graph has fresh headroom).
fn compact(
    old: &SearchGraph,
    protect: &ClosureReport,
    roots: &[u32; 7],
    cfg: &CompactConfig,
) -> Result<(SearchGraph, CompactionStats), CapacityError> {
    let started = Instant::now();
    let survive = select_survivors(old, protect, roots, cfg.keep_frac_pct);
    let n = survive.len();

    // Build remap: ascending old-id order -> deterministic new ids.
    // Survivors get consecutive new ids; evicted entries get UNEXPANDED.
    let mut remap: Vec<u32> = vec![UNEXPANDED; n];
    let mut new_id_counter: u32 = 0;
    for old_id in 0..n {
        if survive[old_id] {
            remap[old_id] = new_id_counter;
            new_id_counter += 1;
        }
    }
    let kept = new_id_counter;

    let mut new = SearchGraph::new(
        old.states.capacity(),
        old.placements.capacity(),
        old.branches.capacity(),
        old.carrier,
        old.policy,
    );
    // The death ledger is run-lifetime state, not generation state.
    new.dead_keys = Arc::clone(&old.dead_keys);
    let new = new;

    // Transfer cumulative counters FIRST so loop-time increments (e.g. frontier-push aborts) add on top instead of being overwritten.
    new.counters.rollouts.store(
        old.counters.rollouts.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.deaths.store(
        old.counters.deaths.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.horizons.store(
        old.counters.horizons.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.cycles.store(
        old.counters.cycles.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.aborted.store(
        old.counters.aborted.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.certified.store(
        old.counters.certified.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.edges_expanded.store(
        old.counters.edges_expanded.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.full_expansions.store(
        old.counters.full_expansions.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.max_rollout_depth.store(
        old.counters.max_rollout_depth.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    new.counters.rollout_depth_sum.store(
        old.counters.rollout_depth_sum.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    // `new.exhausted` deliberately stays false — fresh headroom.

    let shift = cfg.decay_shift;
    // clamp: overshift would panic in debug / mask in release
    let decayed_u64 = |v: &AtomicU64| -> u64 {
        let s = shift.min(63);
        v.load(Ordering::Relaxed) >> s
    };
    // clamp: overshift would panic in debug / mask in release
    let decayed_u32 = |v: &AtomicU32| -> u32 {
        let s = shift.min(31);
        v.load(Ordering::Relaxed) >> s
    };

    // Deterministic bulk reservation: prefix-sum survivor placement counts in
    // ascending old-id order, reserve each arena ONCE, then fill records in
    // parallel (every slot index is precomputed, so writers never contend).
    // Survivor old-ids, ascending.
    let survivor_ids: Vec<u32> = (0..n as u32).filter(|&id| survive[id as usize]).collect();
    let mut placement_offset: Vec<u32> = Vec::with_capacity(survivor_ids.len());
    let mut total_placements: u32 = 0;
    for &old_id in &survivor_ids {
        placement_offset.push(total_placements);
        total_placements += old.states.get(old_id).placement_len as u32;
    }
    let pbase = if total_placements > 0 {
        new.placements
            .reserve(total_placements)
            .ok_or(CapacityError::PlacementsExhausted)?
    } else {
        0
    };
    let bbase = if total_placements > 0 {
        new.branches
            .reserve(total_placements * 7)
            .ok_or(CapacityError::BranchesExhausted)?
    } else {
        0
    };
    let sbase = if kept > 0 {
        new.states
            .reserve(kept)
            .ok_or(CapacityError::StatesExhausted)?
    } else {
        0
    };
    debug_assert_eq!(sbase, 0, "fresh graph: state ids start at 0");

    // Parallel fill; collect frontier requeues per survivor (indexed order
    // preserved) and push sequentially afterward for a deterministic queue.
    let requeue: Vec<Option<u32>> = survivor_ids
        .par_iter()
        .enumerate()
        .map(|(idx, &old_id)| {
            let old_state = old.states.get(old_id);
            let placement_len = old_state.placement_len as u32;
            let new_id = remap[old_id as usize];
            debug_assert_eq!(new_id, sbase + idx as u32, "remap and arena out of sync");
            let new_pbase = pbase + placement_offset[idx];

            for slot in 0..placement_len {
                let old_pr = old.placements.get(old_state.placement_base + slot);
                let new_bbase = bbase + (placement_offset[idx] + slot) * 7;

                // Copy branch records for each of the 7 piece slots.
                for piece_idx in 0..7u32 {
                    let old_br = old.branches.get(old_pr.branch_base + piece_idx);
                    let old_child = old_br.child.load(Ordering::Acquire);
                    debug_assert!(old_child != EXPANDING, "EXPANDING edge at barrier");
                    let new_child = if old_child <= MAX_VALID_STATE_ID {
                        let remapped = remap[old_child as usize];
                        if remapped == UNEXPANDED
                            && old.states.get(old_child).proof.load(Ordering::Acquire) == PROOF_DEAD
                        {
                            // Evicted PROOF_DEAD child: keep the refutation as
                            // a zero-cost stamp instead of forgetting it.
                            DEAD_LEDGER
                        } else {
                            remapped
                        }
                    } else if old_child == DEAD_LEDGER {
                        DEAD_LEDGER // persistent fact — carries over
                    } else {
                        UNEXPANDED
                    };
                    let new_br = new.branches.get(new_bbase + piece_idx);
                    new_br.child.store(new_child, Ordering::Release);
                    new_br
                        .visits
                        .store(decayed_u64(&old_br.visits), Ordering::Relaxed);
                    new_br
                        .reward_sum
                        .store(decayed_u32(&old_br.reward_sum), Ordering::Relaxed);
                }

                new.placements.set(
                    new_pbase + slot,
                    PlacementRecord {
                        placement: old_pr.placement,
                        branch_base: new_bbase,
                        visits: AtomicU64::new(decayed_u64(&old_pr.visits)),
                        reward_sum: AtomicU64::new(decayed_u64(&old_pr.reward_sum)),
                    },
                );
            }

            let expansion_status_val = old_state.expansion_status.load(Ordering::Acquire);
            debug_assert!(
                expansion_status_val != STATUS_EXPANDING,
                "EXPANDING state status at barrier"
            );
            let frontier_queued_val = old_state.frontier_queued.load(Ordering::Acquire);

            new.states.set(
                new_id,
                StateRecord {
                    key: old_state.key,
                    visits: AtomicU64::new(decayed_u64(&old_state.visits)),
                    reward_sum: AtomicU64::new(decayed_u64(&old_state.reward_sum)),
                    placement_base: if placement_len > 0 { new_pbase } else { 0 },
                    placement_len: old_state.placement_len,
                    expansion_status: AtomicU8::new(expansion_status_val),
                    frontier_queued: AtomicU8::new(frontier_queued_val),
                    proof: AtomicU8::new(old_state.proof.load(Ordering::Acquire)),
                },
            );
            new.ids.insert(old_state.key, new_id);

            // Re-queue pending expansions exactly once (pushed in order below).
            (expansion_status_val == STATUS_QUEUED && frontier_queued_val == 1).then_some(new_id)
        })
        .collect();
    for new_id in requeue.into_iter().flatten() {
        if new.frontier.push(new_id).is_err() {
            new.counters.aborted.fetch_add(1, Ordering::Relaxed);
        }
    }

    // NOTE: faithfulness is asserted by callers on real graphs —
    // run_search validates after every compaction in debug builds, and
    // compaction_remap_is_faithful_on_real_graph covers it in tests.
    // (No in-function assert: synthetic test graphs wire non-canonical
    // children, which is legitimate fixture behavior, not corruption.)

    let wall_ms = started.elapsed().as_millis() as u64;
    Ok((
        new,
        CompactionStats {
            kept,
            evicted: n as u32 - kept,
            protected_k: protect.k_size,
            wall_ms,
        },
    ))
}

/// True iff every valid branch child in the graph points at the canonical
/// successor state: recompute the expected child key from (parent key,
/// placement, piece) and compare DIRECTLY against the child's stored key.
/// This is the entire safety surface of the compaction remap — everything
/// else can only shrink K.
///
/// Only meaningful for graphs built through real expansion: synthetic test
/// graphs wire arbitrary children whose keys are legitimately non-canonical,
/// so tests assert this on real graphs only (see
/// `compaction_remap_is_faithful_on_real_graph`); `run_search` asserts it
/// after every compaction in debug builds (Task D).
fn compaction_is_faithful(graph: &SearchGraph) -> bool {
    for id in 0..graph.states.len() {
        let state = graph.states.get(id);
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            let Some(child_board) = apply_move(&state.key.board, pr.placement, &graph.carrier)
            else {
                return false; // stored placements must be survivors
            };
            for piece in state.key.bag().iter_pieces() {
                let child = graph
                    .branches
                    .get(pr.branch_base + piece.index() as u32)
                    .child
                    .load(Ordering::Acquire);
                if child <= MAX_VALID_STATE_ID {
                    let expected =
                        StateKey::new(child_board, child_bag(state.key.bag(), piece), piece);
                    if graph.states.get(child).key != expected {
                        return false;
                    }
                }
            }
        }
    }
    // Interner rebuild surface: every (key -> id) entry must point at a state
    // that actually carries that key.
    for entry in graph.ids.iter() {
        let id = *entry.value();
        if id > MAX_VALID_STATE_ID || graph.states.get(id).key != *entry.key() {
            return false;
        }
    }
    true
}

// ============================================================================
// Ledger persistence
// ============================================================================

/// Death-ledger file layout (little-endian): magic "TAPD", version u32,
/// carrier (max_holes, max_height, max_roughness) u32×3, count u64, then
/// count × 42-byte keys (board 10×u32, bag u8, piece u8). Sorted by key for
/// deterministic bytes.
const LEDGER_MAGIC: &[u8; 4] = b"TAPD";
const LEDGER_VERSION: u32 = 1;

fn save_ledger(graph: &SearchGraph, path: &PathBuf, carrier: &Carrier) -> anyhow::Result<usize> {
    let mut keys: Vec<StateKey> = graph.dead_keys.iter().map(|e| *e.key()).collect();
    keys.sort_unstable();
    let tmp = path.with_extension("tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(LEDGER_MAGIC)?;
    w.write_all(&LEDGER_VERSION.to_le_bytes())?;
    for v in [carrier.max_holes, carrier.max_height, carrier.max_roughness] {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(&(keys.len() as u64).to_le_bytes())?;
    for key in &keys {
        for limb in key.board.as_limbs() {
            w.write_all(&limb.to_le_bytes())?;
        }
        w.write_all(&[key.bag_bits, key.piece_idx])?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(keys.len())
}

fn load_ledger(graph: &SearchGraph, path: &PathBuf, carrier: &Carrier) -> anyhow::Result<usize> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == LEDGER_MAGIC, "not a ledger file: bad magic");
    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    anyhow::ensure!(
        u32::from_le_bytes(u32buf) == LEDGER_VERSION,
        "ledger version mismatch"
    );
    let mut header = [0u32; 3];
    for slot in &mut header {
        r.read_exact(&mut u32buf)?;
        *slot = u32::from_le_bytes(u32buf);
    }
    anyhow::ensure!(
        header == [carrier.max_holes, carrier.max_height, carrier.max_roughness],
        "ledger carrier mismatch: file {:?} vs run ({}, {}, {}) — death facts are \
         carrier-specific; refusing to load",
        header,
        carrier.max_holes,
        carrier.max_height,
        carrier.max_roughness
    );
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u64buf)?;
    let count = u64::from_le_bytes(u64buf) as usize;
    let mut loaded = 0usize;
    for _ in 0..count {
        let mut limbs = [0u32; 10];
        for limb in &mut limbs {
            r.read_exact(&mut u32buf)?;
            *limb = u32::from_le_bytes(u32buf);
        }
        let mut tail = [0u8; 2];
        r.read_exact(&mut tail)?;
        let (bag_bits, piece_idx) = (tail[0], tail[1]);
        // Skip corrupt entries rather than poisoning the run.
        if piece_idx >= 7 || bag_bits == 0 || bag_bits > TetrisPieceBagState::FULL_MASK {
            continue;
        }
        let key = StateKey {
            board: TetrisBoard::from_limbs(limbs),
            bag_bits,
            piece_idx,
        };
        graph.ledger_insert(key);
        loaded += 1;
    }
    Ok(loaded)
}

// ============================================================================
// Main / epochs (Task 9)
// ============================================================================

/// One compact barrier: protect closure → boundary-seed harvest → compact →
/// root re-resolution → faithfulness check. Callable at the epoch boundary
/// AND mid-epoch between rollout chunks (all pool work is joined at both
/// sites, so the barrier-only contract holds). Returns Ok(false) when the
/// livelock guard fires (occupancy cannot drop below the trigger) — the
/// caller must stop the run.
#[allow(clippy::too_many_arguments)]
fn compact_barrier(
    args: &Args,
    pool: &rayon::ThreadPool,
    keep_frac: u32,
    compact_at: u32,
    graph: &mut SearchGraph,
    roots: &mut [u32; 7],
    boundary_keys: &mut Vec<StateKey>,
    last_report: &mut Option<ClosureReport>,
    generations: &mut u64,
) -> anyhow::Result<bool> {
    let occupancy_hit =
        |len: u32, cap: u32| -> bool { (len as u64) * 100 >= (cap as u64) * (compact_at as u64) };
    let protect = {
        let g: &SearchGraph = graph;
        let r: &[u32; 7] = roots;
        pool.install(|| extract_safe_set(g, r))
    };
    // Harvest fresh boundary seeds straight from the closure report (its
    // boundary_ids index THIS graph, pre-compaction). Keys re-resolve through
    // the post-compaction interner (evicted seeds drop out).
    *boundary_keys = protect
        .boundary_ids
        .iter()
        .map(|&id| graph.states.get(id).key)
        .collect();
    let cfg = CompactConfig {
        keep_frac_pct: keep_frac,
        decay_shift: args.decay_shift,
    };
    // compact cannot fail by construction (new capacities == old,
    // survivors ⊆ old occupancy); `?` is defense, not a real path.
    let (new_graph, stats) = compact(graph, &protect, roots, &cfg)?;
    *graph = new_graph;
    // Roots are protected; re-resolve their ids in the new interner.
    for (i, key) in root_keys().into_iter().enumerate() {
        match graph.ids.get(&key) {
            Some(id) => roots[i] = *id,
            None => anyhow::bail!("compaction lost a protected root"),
        }
    }
    debug_assert!(compaction_is_faithful(graph));
    // K is preserved exactly across compaction, so the scalar fields of the
    // protection report stay valid on the new graph; only the witnesses
    // (which index pre-compaction ids) are stale. Keep the scalars so epoch
    // logs/CSV don't falsely report K 0 until the next periodic closure.
    // k0/near_miss go slightly stale across compaction (K itself stays
    // exact); refreshed at the next closure.
    *last_report = Some(ClosureReport {
        witnesses: Vec::new(), // stale ids dropped
        ..protect
    });
    *generations += 1;
    println!(
        "compact gen {generations} | kept {} (K {}) | evicted {} | {}ms",
        stats.kept, stats.protected_k, stats.evicted, stats.wall_ms
    );
    // Livelock guard: if the protected/hot set alone still fills any
    // arena past the trigger, compaction cannot make progress.
    if occupancy_hit(graph.states.len(), graph.states.capacity())
        || occupancy_hit(graph.placements.len(), graph.placements.capacity())
        || occupancy_hit(graph.branches.len(), graph.branches.capacity())
    {
        // When K + roots alone fills the arena past the trigger,
        // lowering --keep-frac cannot help — only a larger arena does.
        let protected_fills_trigger = (stats.protected_k as u64 + 7) >= (stats.kept as u64);
        if protected_fills_trigger {
            println!(
                "arena occupancy cannot drop below the trigger (kept {} of {}); \
                 the protected set (K + roots) alone fills the arena past the trigger \
                 — raise --max-states/--max-placements/--max-branches",
                graph.states.len(),
                graph.states.capacity()
            );
        } else {
            println!(
                "arena occupancy cannot drop below the trigger (kept {} of {}); \
                 stopping — raise --max-states/--max-placements/--max-branches or lower --keep-frac",
                graph.states.len(),
                graph.states.capacity()
            );
        }
        return Ok(false);
    }
    Ok(true)
}

/// Drain up to `limit` frontier states and fully expand them in parallel,
/// stopping early once any arena's occupancy reaches `occupancy_pct` percent
/// of capacity. Expansion interns children as it runs, so the batch drains in
/// chunks and re-checks occupancy between chunks (overshoot is bounded by one
/// chunk). Un-drained states stay queued for later epochs. The reserved
/// headroom is what lets rollouts extend lazily along UCT-chosen lines instead
/// of aborting on a breadth-first-exhausted arena.
/// Returns the number of states expanded.
fn expansion_batch(
    graph: &SearchGraph,
    pool: &rayon::ThreadPool,
    limit: u64,
    occupancy_pct: u32,
) -> usize {
    const CHUNK: u64 = 512;
    let occupancy_hit = |len: u32, cap: u32| -> bool {
        (len as u64) * 100 >= (cap as u64) * (occupancy_pct as u64)
    };
    let over_budget = || -> bool {
        graph.exhausted.load(Ordering::SeqCst)
            || occupancy_hit(graph.states.len(), graph.states.capacity())
            || occupancy_hit(graph.placements.len(), graph.placements.capacity())
            || occupancy_hit(graph.branches.len(), graph.branches.capacity())
    };
    let mut total: u64 = 0;
    let mut batch: Vec<u32> = Vec::with_capacity(CHUNK as usize);
    while total < limit && !over_budget() {
        batch.clear();
        while (batch.len() as u64) < CHUNK.min(limit - total) {
            match graph.frontier.pop() {
                Some(id) => batch.push(id),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        pool.install(|| {
            batch.par_iter().for_each(|&id| {
                let _ = graph.ensure_fully_expanded(id);
            });
        });
        total += batch.len() as u64;
    }
    total as usize
}

/// What a run produced, for main's summary and the integration tests.
struct RunOutcome {
    report: ClosureReport,
    generations: u64,
    epochs_run: u64,
    /// Per root: proof byte at exit. All-DEAD is a certified impossibility
    /// theorem for this carrier (no strategy survives forever inside it).
    roots_proof: [u8; 7],
}

/// Epoch loop: expansion barrier -> rollout barrier -> (periodic) closure ->
/// metrics. Returns the final closure report and run metadata. Deterministic
/// modulo scheduling.
fn run_search(args: &Args, stop: &AtomicBool) -> anyhow::Result<RunOutcome> {
    anyhow::ensure!(
        args.horizon >= 1 && args.horizon <= (u32::MAX / 2) as usize,
        "--horizon out of range"
    );
    anyhow::ensure!(args.credit_cap >= 1, "--credit-cap must be >= 1");
    // Default placement provisioning: measured mean STORED (surviving)
    // placements per state is ~2.9 across reference-carrier runs (dead
    // states store none; constrained states store few) — 8 per state gives
    // ~2.7× headroom at a quarter of the worst-case (34/state) footprint.
    // The branch arena is 7× this and pre-initialized EVERY generation, so
    // over-provisioning is paid in init stores and RSS, not just address
    // space; undershoot merely triggers the normal exhaustion→compact path.
    const PLACEMENTS_PER_STATE_DEFAULT: u32 = 8;
    let max_placements = args
        .max_placements
        .unwrap_or_else(|| args.max_states.saturating_mul(PLACEMENTS_PER_STATE_DEFAULT));
    let max_branches = args
        .max_branches
        .unwrap_or_else(|| max_placements.saturating_mul(7));
    let carrier = Carrier {
        max_holes: args.max_holes,
        max_height: args.max_height,
        max_roughness: args.max_roughness,
    };
    let policy = match args.expansion_policy {
        ExpansionPolicyArg::All => ExpansionPolicy::AllDiscovered,
        ExpansionPolicyArg::Candidate => ExpansionPolicy::CandidateOnly,
    };
    let params = SearchParams {
        horizon: args.horizon,
        credit_cap: (args.credit_cap as u64).min(args.horizon as u64),
        c_uct: args.c_uct,
        c_lcb: args.c_lcb,
        admit_after: args.admit_after,
        piece_policy: args.piece_policy,
    };

    // Sanitize compaction percentages once.
    let compact_at = args.compact_at.clamp(1, 100);
    let keep_frac = args.keep_frac.min(100);

    let mut graph = SearchGraph::new(
        args.max_states,
        max_placements,
        max_branches,
        carrier,
        policy,
    );
    if let Some(path) = &args.ledger {
        if path.exists() {
            let loaded = load_ledger(&graph, path, &carrier)?;
            println!("ledger loaded: {loaded} dead keys from {}", path.display());
        }
    }
    let mut roots = [0u32; 7];
    for (i, key) in root_keys().into_iter().enumerate() {
        roots[i] = graph.intern(key).map_err(anyhow::Error::from)?.id;
        if policy == ExpansionPolicy::CandidateOnly {
            graph.mark_queued_and_push(roots[i]); // roots are always admitted
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let mut csv = match &args.csv {
        Some(path) => {
            let mut file = std::fs::File::create(path)?;
            writeln!(
                file,
                "epoch,states,frontier,rollouts,deaths,horizons,cycles,aborted,\
                 edges,k_size,roots_in_k,elapsed_ms,gen,k0,near_miss,rootv_min_pm,rootv_mean_pm,avg_depth_tenths,dead_total,certified"
            )?;
            Some(file)
        }
        None => None,
    };

    let started = Instant::now();
    let mut epoch: u64 = 0;
    let mut generations: u64 = 0;
    let mut last_report: Option<ClosureReport> = None;
    // Near-miss boundary seeds (refreshed at periodic closures). Stored as
    // keys: compaction remaps ids, the interner re-resolves each epoch.
    let mut boundary_keys: Vec<StateKey> = Vec::new();
    let mut prev_rollouts: u64 = 0;
    let mut prev_depth_sum: u64 = 0;
    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if args.no_compact && graph.exhausted.load(Ordering::SeqCst) {
            break; // legacy terminal-exhaustion behavior
        }

        if let Some(max) = args.epochs {
            if epoch >= max {
                break;
            }
        }

        let occupancy_hit = |len: u32, cap: u32| -> bool {
            (len as u64) * 100 >= (cap as u64) * (compact_at as u64)
        };
        let should_compact = !args.no_compact
            && (graph.exhausted.load(Ordering::SeqCst)
                || occupancy_hit(graph.states.len(), graph.states.capacity())
                || occupancy_hit(graph.placements.len(), graph.placements.capacity())
                || occupancy_hit(graph.branches.len(), graph.branches.capacity()));
        if should_compact
            && !compact_barrier(
                args,
                &pool,
                keep_frac,
                compact_at,
                &mut graph,
                &mut roots,
                &mut boundary_keys,
                &mut last_report,
                &mut generations,
            )?
        {
            break;
        }

        let mut expanded = expansion_batch(
            &graph,
            &pool,
            args.expansions_per_epoch,
            args.expand_occupancy.clamp(1, 100),
        );

        // Boundary-seeded rollouts in CHUNKS: half of each chunk starts at
        // the 7 roots (keeps root statistics live), half at near-miss
        // boundary states — sampling density lands exactly where the closure
        // needs evidence. Between chunks, if the arena exhausted, run a
        // mid-epoch compact barrier so a large rollout budget keeps growing
        // the graph instead of re-treading a saturated interior. Keys
        // re-resolve through the interner (evicted seeds drop out). The seed
        // index is the GLOBAL epoch rollout index (base + i), so chunking
        // does not change the root/boundary rotation.
        const ROLLOUT_CHUNK: u64 = 50_000;
        let mut remaining = args.rollouts_per_epoch;
        let mut rollout_base: u64 = 0;
        let mut run_alive = true;
        while remaining > 0 {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            let chunk = remaining.min(ROLLOUT_CHUNK);
            let boundary_ids: Vec<u32> = boundary_keys
                .iter()
                .filter_map(|key| graph.ids.get(key).map(|entry| *entry))
                .collect();
            let base = rollout_base;
            let root_pct = args.root_seed_pct.min(100) as u64;
            pool.install(|| {
                (0..chunk).into_par_iter().for_each_init(
                    || WorkerScratch::new(params.horizon),
                    |scratch, i| {
                        let i = base + i;
                        // Deterministic interleave: rollout i goes to a root
                        // iff its residue class mod 100 is below root_pct.
                        let seed = if boundary_ids.is_empty() || i % 100 < root_pct {
                            roots[(i % 7) as usize]
                        } else {
                            boundary_ids[(i as usize) % boundary_ids.len()]
                        };
                        let outcome = rollout(&graph, &params, seed, scratch);
                        graph.counters.record_rollout(outcome, scratch.path.len());
                    },
                );
            });
            rollout_base += chunk;
            remaining -= chunk;
            if remaining > 0 && !args.no_compact && graph.exhausted.load(Ordering::SeqCst) {
                if !compact_barrier(
                    args,
                    &pool,
                    keep_frac,
                    compact_at,
                    &mut graph,
                    &mut roots,
                    &mut boundary_keys,
                    &mut last_report,
                    &mut generations,
                )? {
                    run_alive = false;
                    break;
                }
                expanded += expansion_batch(
                    &graph,
                    &pool,
                    args.expansions_per_epoch,
                    args.expand_occupancy.clamp(1, 100),
                );
            }
        }
        if !run_alive {
            break;
        }

        let cum_rollouts = graph.counters.rollouts.load(Ordering::Relaxed);
        let cum_depth_sum = graph.counters.rollout_depth_sum.load(Ordering::Relaxed);
        let epoch_rollouts = cum_rollouts - prev_rollouts;
        let epoch_depth_sum = cum_depth_sum - prev_depth_sum;
        prev_rollouts = cum_rollouts;
        prev_depth_sum = cum_depth_sum;
        // Average traversed depth this epoch, in tenths of a piece (integer-only).
        let avg_depth_tenths = if epoch_rollouts == 0 {
            0
        } else {
            epoch_depth_sum * 10 / epoch_rollouts
        };

        let (rootv_min, rootv_mean) = root_values_permille(&graph, &roots, &params);

        let closure = if epoch % args.closure_every.max(1) == 0 {
            Some(pool.install(|| extract_safe_set(&graph, &roots)))
        } else {
            None
        };

        if let Some(report) = closure.as_ref() {
            // Refresh the rollout boundary seeds from this closure's K view.
            boundary_keys = report
                .boundary_ids
                .iter()
                .map(|&id| graph.states.get(id).key)
                .collect();
            println!(
                "closure e{epoch} | K0 {} | prune {:?} | near-miss {} | dead {} (ledger {}) | K {} ({} iters)",
                report.k0_size,
                report.pass_removed,
                report.near_miss,
                report.dead_total,
                graph.dead_keys.len(),
                report.k_size,
                report.iterations
            );
        }

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let counters = &graph.counters;
        let (k_size, roots_in_k) = closure
            .as_ref()
            .map(|r| (r.k_size, r.roots_in_k.iter().filter(|b| **b).count()))
            .or_else(|| {
                last_report
                    .as_ref()
                    .map(|r| (r.k_size, r.roots_in_k.iter().filter(|b| **b).count()))
            })
            .unwrap_or((0, 0));
        let (csv_k0, csv_near_miss, csv_dead_total) = closure
            .as_ref()
            .map(|r| (r.k0_size, r.near_miss, r.dead_total))
            .or_else(|| {
                last_report
                    .as_ref()
                    .map(|r| (r.k0_size, r.near_miss, r.dead_total))
            })
            .unwrap_or((0, 0, 0));
        println!(
            "epoch {epoch} gen {generations} | states {} | frontier {} | expanded {expanded} | \
             rollouts(cum) {} (d {} / h {} / c {} / a {} / s {}) | depth {}.{} | edges(cum) {} | K {k_size} \
             (roots {roots_in_k}/7) | rootv {rootv_min}/{rootv_mean}pm | {elapsed_ms}ms",
            graph.states.len(),
            graph.frontier.len(),
            counters.rollouts.load(Ordering::Relaxed),
            counters.deaths.load(Ordering::Relaxed),
            counters.horizons.load(Ordering::Relaxed),
            counters.cycles.load(Ordering::Relaxed),
            counters.aborted.load(Ordering::Relaxed),
            counters.certified.load(Ordering::Relaxed),
            avg_depth_tenths / 10,
            avg_depth_tenths % 10,
            counters.edges_expanded.load(Ordering::Relaxed),
        );
        if let Some(file) = csv.as_mut() {
            writeln!(
                file,
                "{epoch},{},{},{},{},{},{},{},{},{k_size},{roots_in_k},{elapsed_ms},{generations},{csv_k0},{csv_near_miss},{rootv_min},{rootv_mean},{avg_depth_tenths},{csv_dead_total},{}",
                graph.states.len(),
                graph.frontier.len(),
                counters.rollouts.load(Ordering::Relaxed),
                counters.deaths.load(Ordering::Relaxed),
                counters.horizons.load(Ordering::Relaxed),
                counters.cycles.load(Ordering::Relaxed),
                counters.aborted.load(Ordering::Relaxed),
                counters.edges_expanded.load(Ordering::Relaxed),
                counters.certified.load(Ordering::Relaxed),
            )?;
        }
        if let Some(report) = closure {
            last_report = Some(report);
        }
        epoch += 1;
        if let Some(path) = &args.ledger {
            if args.ledger_save_every > 0 && epoch % args.ledger_save_every == 0 {
                let saved = save_ledger(&graph, path, &carrier)?;
                println!("ledger autosave: {saved} dead keys -> {}", path.display());
            }
        }
    }

    let final_report = pool.install(|| extract_safe_set(&graph, &roots));
    let life_rollouts = graph.counters.rollouts.load(Ordering::Relaxed);
    let life_depth_sum = graph.counters.rollout_depth_sum.load(Ordering::Relaxed);
    let life_avg_depth_tenths = if life_rollouts == 0 {
        0
    } else {
        life_depth_sum * 10 / life_rollouts
    };
    println!(
        "FINAL | states {} | K {} | K0 {} | near-miss {} | dead {} (ledger {}) | iterations {} | roots in K: {:?} | \
         max rollout depth {} | avg depth(life) {}.{} | full-exp {} | arenas s {} p {} b {} | exhausted: {} | \
         generations {generations} | certified {}",
        graph.states.len(),
        final_report.k_size,
        final_report.k0_size,
        final_report.near_miss,
        final_report.dead_total,
        graph.dead_keys.len(),
        final_report.iterations,
        final_report.roots_in_k,
        graph.counters.max_rollout_depth.load(Ordering::Relaxed),
        life_avg_depth_tenths / 10,
        life_avg_depth_tenths % 10,
        graph.counters.full_expansions.load(Ordering::Relaxed),
        graph.states.len(),
        graph.placements.len(),
        graph.branches.len(),
        graph.exhausted.load(Ordering::SeqCst),
        graph.counters.certified.load(Ordering::Relaxed),
    );
    if let Some(path) = &args.ledger {
        let saved = save_ledger(&graph, path, &carrier)?;
        println!("ledger saved: {saved} dead keys -> {}", path.display());
    }
    let roots_proof = roots.map(|r| graph.states.get(r).proof.load(Ordering::Acquire));
    Ok(RunOutcome {
        report: final_report,
        generations,
        epochs_run: epoch,
        roots_proof,
    })
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))?;
    }
    println!("tetris_atlas_puct starting: {args:?}");
    let outcome = run_search(&args, &stop)?;
    let roots_dead = outcome
        .roots_proof
        .iter()
        .filter(|p| **p == PROOF_DEAD)
        .count();
    if outcome.report.roots_in_k.iter().all(|b| *b) && outcome.report.k_size > 0 {
        println!(
            "SUCCESS: all 7 roots are in the candidate safe set (|K| = {}) | generations {}",
            outcome.report.k_size, outcome.generations
        );
    } else if roots_dead == 7 {
        println!(
            "IMPOSSIBLE: all 7 roots are PROOF_DEAD — no strategy survives forever inside \
             this carrier (holes<={}, height<={}, roughness<={}). Certified impossibility; \
             generations {}.",
            args.max_holes, args.max_height, args.max_roughness, outcome.generations
        );
    } else {
        println!(
            "No root-covering safe set yet (|K| = {}, roots dead {roots_dead}/7) | generations {}.",
            outcome.report.k_size, outcome.generations
        );
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // --- Task 1: domain ---

    #[test]
    fn key_identity_axes() {
        let full = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        // base uses the refill case (current piece present in a FULL bag) — canonical.
        let base = StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::O_PIECE);
        let same = StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::O_PIECE);
        assert_eq!(base, same);

        // board axis
        let mut other_board = TetrisBoard::EMPTY_BOARD;
        other_board.set_bit(0, 0);
        assert_ne!(base, StateKey::new(other_board, full, TetrisPiece::O_PIECE));

        // bag axis: a canonical bag missing the current piece (O drawn) — differs
        // from `full`, and keeps the invariant (O absent) so `new` accepts it.
        let mut dropped = full;
        dropped.remove(TetrisPiece::O_PIECE);
        dropped.remove(TetrisPiece::I_PIECE);
        assert_ne!(
            base,
            StateKey::new(TetrisBoard::EMPTY_BOARD, dropped, TetrisPiece::O_PIECE)
        );

        // piece axis
        assert_ne!(
            base,
            StateKey::new(TetrisBoard::EMPTY_BOARD, full, TetrisPiece::T_PIECE)
        );
    }

    #[test]
    fn root_keys_are_canonical() {
        let roots = root_keys();
        assert_eq!(roots.len(), 7);
        for (i, root) in roots.iter().enumerate() {
            assert_eq!(root.board, TetrisBoard::EMPTY_BOARD);
            assert_eq!(root.piece().index(), i as u8);
            // Current piece was removed from a full bag: 6 remain, piece not among them.
            assert_eq!(root.bag().count(), 6);
            assert!(!root.bag().contains(root.piece()));
        }
        // All roots distinct.
        for i in 0..7 {
            for j in (i + 1)..7 {
                assert_ne!(roots[i], roots[j]);
            }
        }
    }

    #[test]
    fn bag_transition_refills() {
        // Singleton bag {O}: drawing O empties it -> refill to FULL (which contains O).
        let mut bag = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        for p in TetrisPiece::all() {
            if p != TetrisPiece::O_PIECE {
                bag.remove(p);
            }
        }
        assert_eq!(bag.count(), 1);
        let next = child_bag(bag, TetrisPiece::O_PIECE);
        assert_eq!(u8::from(next), TetrisPieceBagState::FULL_MASK);
        // Mid-bag draw: no refill.
        let full = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        let after = child_bag(full, TetrisPiece::I_PIECE);
        assert_eq!(after.count(), 6);
        assert!(!after.contains(TetrisPiece::I_PIECE));
    }

    #[test]
    fn apply_move_detects_loss() {
        // Columns 0-8 stacked 19 high, column 9 EMPTY: no row can ever complete
        // (so nothing clears), and a vertical I on column 0 reaches row 22 > 20.
        let mut limbs = [(1u32 << 19) - 1; 10];
        limbs[9] = 0;
        let board = TetrisBoard::from_limbs(limbs);
        let vertical_i_col0 = TetrisPiecePlacement::all_from_piece(TetrisPiece::I_PIECE)
            .iter()
            .copied()
            .find(|p| {
                p.piece.width(p.orientation.rotation) == 1 && p.orientation.column.index() == 0
            })
            .unwrap();
        assert!(apply_move(&board, vertical_i_col0, &Carrier::UNBOUNDED).is_none());
        // Empty board: every placement survives.
        for &p in TetrisPiecePlacement::all_from_piece(TetrisPiece::T_PIECE) {
            assert!(apply_move(&TetrisBoard::EMPTY_BOARD, p, &Carrier::UNBOUNDED).is_some());
        }
    }

    #[test]
    fn apply_move_respects_carrier() {
        let tight = Carrier {
            max_holes: 0,
            max_height: 0,
            max_roughness: 0,
        };
        // Height cap 0: every placement leaves the carrier.
        for &p in TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE) {
            assert!(apply_move(&TetrisBoard::EMPTY_BOARD, p, &tight).is_none());
        }
        let loose = Carrier {
            max_holes: 0,
            max_height: 2,
            max_roughness: 4,
        };
        // Flat O placement at height 2 fits.
        let flat_o = TetrisPiecePlacement::all_from_piece(TetrisPiece::O_PIECE)[0];
        assert!(apply_move(&TetrisBoard::EMPTY_BOARD, flat_o, &loose).is_some());
    }

    #[test]
    fn surface_roughness_matches_reference() {
        assert_eq!(surface_roughness(&TetrisBoard::EMPTY_BOARD), 0);
        let mut board = TetrisBoard::EMPTY_BOARD;
        board.set_bit(3, 0); // one cell at column 3 -> |0-1| + |1-0| = 2
        assert_eq!(surface_roughness(&board), 2);
    }

    // --- shared test helper ---

    fn test_graph(max_states: u32, carrier: Carrier, policy: ExpansionPolicy) -> SearchGraph {
        SearchGraph::new(
            max_states,
            max_states * 34,
            max_states * 34 * 7,
            carrier,
            policy,
        )
    }

    // --- Task 2: arena ---

    #[test]
    fn arena_reserve_is_contiguous_and_bounded() {
        let arena: Arena<u64> = Arena::new(10);
        assert_eq!(arena.reserve(3), Some(0));
        assert_eq!(arena.reserve(4), Some(3));
        assert_eq!(arena.reserve(3), Some(7)); // exactly at capacity
        assert_eq!(arena.reserve(1), None); // exhausted
        assert_eq!(arena.len(), 10);
    }

    #[test]
    fn arena_set_get_roundtrip() {
        let arena: Arena<u64> = Arena::new(4);
        let base = arena.reserve(2).unwrap();
        arena.set(base, 42);
        arena.set(base + 1, 43);
        assert_eq!(*arena.get(base), 42);
        assert_eq!(*arena.get(base + 1), 43);
    }

    #[test]
    fn arena_concurrent_reserves_do_not_overlap() {
        let arena: Arena<u32> = Arena::new(8_000);
        let bases: Vec<u32> = (0..1_000u32)
            .into_par_iter()
            .map(|i| {
                let base = arena.reserve(8).unwrap();
                for j in 0..8 {
                    arena.set(base + j, i);
                }
                base
            })
            .collect();
        let mut sorted = bases.clone();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            assert!(pair[1] - pair[0] >= 8, "overlapping reservations");
        }
        // Every slot readable and tagged by exactly the reserving worker.
        for (i, &base) in bases.iter().enumerate() {
            for j in 0..8 {
                assert_eq!(*arena.get(base + j), i as u32);
            }
        }
    }

    // --- Task 3: interning ---

    #[test]
    fn intern_same_key_returns_same_id() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let first = graph.intern(key).unwrap();
        let second = graph.intern(key).unwrap();
        assert!(first.newly_created);
        assert!(!second.newly_created);
        assert_eq!(first.id, second.id);
        assert_eq!(graph.states.len(), 1);
    }

    #[test]
    fn concurrent_intern_one_record() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[3];
        let results: Vec<InternResult> = (0..1_000)
            .into_par_iter()
            .map(|_| graph.intern(key).unwrap())
            .collect();
        assert!(results.iter().all(|r| r.id == results[0].id));
        // Exactly one caller wins the Vacant-entry race and sees newly_created.
        assert_eq!(results.iter().filter(|r| r.newly_created).count(), 1);
        assert_eq!(graph.states.len(), 1);
        assert_eq!(graph.ids.len(), 1);
        assert_eq!(graph.frontier.len(), 1); // queued exactly once
    }

    #[test]
    fn structural_enumeration_filters_placements() {
        // Unbounded carrier, empty board: every placement of the current piece survives.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[TetrisPiece::T_PIECE.index() as usize];
        let id = graph.intern(key).unwrap().id;
        let state = graph.states.get(id);
        let all = TetrisPiecePlacement::all_from_piece(TetrisPiece::T_PIECE).len();
        assert_eq!(state.placement_len as usize, all);
        // Stored placements are exactly the surviving ones, in order, each with
        // 7 reserved branch slots all UNEXPANDED.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            assert_eq!(pr.placement.piece, TetrisPiece::T_PIECE);
            // Each stored placement must genuinely survive apply_move (survivors-only).
            assert!(
                apply_move(&key.board, pr.placement, &graph.carrier).is_some(),
                "stored placement must be a survivor"
            );
            for piece_slot in 0..7 {
                let br = graph.branches.get(pr.branch_base + piece_slot);
                assert_eq!(br.child.load(Ordering::Acquire), UNEXPANDED);
            }
        }

        // Partial carrier: prunes some but not all T placements, exercising the
        // filter loop in the 0 < placement_len < all region (not just all-survive).
        let partial = Carrier {
            max_holes: 0,
            max_height: 2,
            max_roughness: 4,
        };
        let pg = test_graph(100, partial, ExpansionPolicy::AllDiscovered);
        let pid = pg.intern(key).unwrap().id;
        let pstate = pg.states.get(pid);
        assert!(
            (pstate.placement_len as usize) > 0 && (pstate.placement_len as usize) < all,
            "partial carrier should prune some but not all T placements (got {})",
            pstate.placement_len
        );
        for slot in 0..pstate.placement_len as u32 {
            let pr = pg.placements.get(pstate.placement_base + slot);
            assert!(
                apply_move(&key.board, pr.placement, &pg.carrier).is_some(),
                "stored placement must survive the partial carrier"
            );
        }
    }

    #[test]
    fn dead_state_has_zero_placements() {
        // Carrier height 0: every placement leaves the carrier -> terminal death.
        let tight = Carrier {
            max_holes: 0,
            max_height: 0,
            max_roughness: 0,
        };
        let graph = test_graph(100, tight, ExpansionPolicy::AllDiscovered);
        let id = graph.intern(root_keys()[0]).unwrap().id;
        assert_eq!(graph.states.get(id).placement_len, 0);
        // Terminally-dead states never enter the frontier (no-op expansion),
        // but the queued-once flag is set so they are never reconsidered.
        assert_eq!(graph.frontier.len(), 0);
        assert_eq!(
            graph.states.get(id).frontier_queued.load(Ordering::Acquire),
            1
        );
    }

    #[test]
    fn candidate_policy_defers_queueing() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::CandidateOnly);
        let id = graph.intern(root_keys()[0]).unwrap().id;
        assert_eq!(graph.frontier.len(), 0);
        graph.mark_queued_and_push(id);
        graph.mark_queued_and_push(id); // second push is a no-op
        assert_eq!(graph.frontier.len(), 1);
    }

    // --- Task 4: expansion ---

    #[test]
    fn ensure_fully_expanded_covers_exactly_the_bag() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let id = graph.intern(key).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        let state = graph.states.get(id);
        assert_eq!(
            state.expansion_status.load(Ordering::Acquire),
            STATUS_FULLY_EXPANDED
        );
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            for piece in TetrisPiece::all() {
                let child = graph
                    .branches
                    .get(pr.branch_base + piece.index() as u32)
                    .child
                    .load(Ordering::Acquire);
                if key.bag().contains(piece) {
                    assert!(child <= MAX_VALID_STATE_ID, "bag piece must be expanded");
                    // Child key is the canonical successor.
                    let expect_board =
                        apply_move(&key.board, pr.placement, &graph.carrier).unwrap();
                    let expect = StateKey::new(expect_board, child_bag(key.bag(), piece), piece);
                    assert_eq!(graph.states.get(child).key, expect);
                } else {
                    assert_eq!(child, UNEXPANDED, "non-bag piece slot must stay unused");
                }
            }
        }
    }

    #[test]
    fn expand_edge_is_idempotent_and_canonical() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let id = graph.intern(root_keys()[1]).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        let state = graph.states.get(id);
        let pr = graph.placements.get(state.placement_base);
        let piece = state.key.bag().iter_pieces().next().unwrap();
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        let first = br.child.load(Ordering::Acquire);
        // Re-resolving the same edge returns the same cached child.
        let again = graph.resolve_child(state, pr, br, piece).unwrap();
        assert_eq!(first, again);
        // A second full expansion is a no-op.
        graph.ensure_fully_expanded(id).unwrap();
        assert_eq!(br.child.load(Ordering::Acquire), first);
    }

    #[test]
    fn concurrent_full_expansion_no_stuck_expanding() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let ids: Vec<u32> = root_keys()
            .iter()
            .map(|&k| graph.intern(k).unwrap().id)
            .collect();
        // Hammer all roots from many workers simultaneously.
        (0..64usize).into_par_iter().for_each(|i| {
            graph.ensure_fully_expanded(ids[i % 7]).unwrap();
        });
        for &id in &ids {
            let state = graph.states.get(id);
            assert_eq!(
                state.expansion_status.load(Ordering::Acquire),
                STATUS_FULLY_EXPANDED
            );
            for slot in 0..state.placement_len as u32 {
                let pr = graph.placements.get(state.placement_base + slot);
                for piece in TetrisPiece::all() {
                    let child = graph
                        .branches
                        .get(pr.branch_base + piece.index() as u32)
                        .child
                        .load(Ordering::Acquire);
                    if state.key.bag().contains(piece) {
                        assert!(
                            child <= MAX_VALID_STATE_ID,
                            "no EXPANDING/UNEXPANDED left behind"
                        );
                    } else {
                        assert_eq!(child, UNEXPANDED, "non-bag slot written under concurrency");
                    }
                }
            }
        }
        // Same (s,a,q) reached from different roots shares one child id:
        // canonicality means the map and arena agree.
        assert_eq!(graph.ids.len() as u32, graph.states.len());
    }

    #[test]
    fn capacity_exhaustion_rolls_back_edges() {
        // Arena too small to hold a root's children: expansion hits a capacity
        // error mid-way. The claim protocol must leave every branch either a
        // valid id or UNEXPANDED — never stuck EXPANDING — and still publish
        // FULLY_EXPANDED so waiters don't spin forever.
        let graph = SearchGraph::new(
            3,
            3 * 34,
            3 * 34 * 7,
            Carrier::UNBOUNDED,
            ExpansionPolicy::AllDiscovered,
        );
        let id = graph.intern(root_keys()[0]).unwrap().id;
        let result = graph.ensure_fully_expanded(id);
        assert!(
            result.is_err(),
            "3-state arena cannot hold the O-root expansion"
        );
        assert!(graph.exhausted.load(Ordering::SeqCst));
        let state = graph.states.get(id);
        assert_eq!(
            state.expansion_status.load(Ordering::Acquire),
            STATUS_FULLY_EXPANDED
        );
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            for piece in TetrisPiece::all() {
                let child = graph
                    .branches
                    .get(pr.branch_base + piece.index() as u32)
                    .child
                    .load(Ordering::Acquire);
                assert_ne!(
                    child, EXPANDING,
                    "edge stuck EXPANDING after capacity error"
                );
            }
        }
    }

    // --- Task 5: integer scoring ---

    #[test]
    fn mean_fixed_basics() {
        assert_eq!(mean_fixed(0, 0), 0); // no visits -> 0, no division
        assert_eq!(mean_fixed(5, 5), SCORE_ONE); // certainty == 1.0
        assert_eq!(mean_fixed(1, 2), SCORE_ONE / 2);
        assert_eq!(mean_fixed(u64::MAX, u64::MAX), SCORE_ONE); // u128 path, no overflow
    }

    #[test]
    fn isqrt_property_holds() {
        // std u128::isqrt is the exact floor root; verify the contract we rely on.
        for n in [
            0u128,
            1,
            2,
            3,
            4,
            5,
            24,
            25,
            26,
            u64::MAX as u128,
            u128::MAX,
        ] {
            let r = n.isqrt();
            assert!(r * r <= n);
            assert!((r + 1).checked_mul(r + 1).is_none_or(|sq| sq > n));
        }
        // Pseudo-random walk (deterministic; no RNG in this crate).
        let mut x: u128 = 0x9E3779B97F4A7C15;
        for _ in 0..1_000 {
            x = x.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
            let r = x.isqrt();
            assert!(r * r <= x && (r + 1).checked_mul(r + 1).is_none_or(|sq| sq > x));
        }
    }

    #[test]
    fn exploration_bonus_shape() {
        let c = SCORE_ONE;
        // Unvisited actions get flat priority.
        assert_eq!(exploration_bonus(100, 0, c), u64::MAX / 4);
        // parent = 0 gives zero pressure (log2(1) = 0), NOT the unvisited sentinel.
        assert_eq!(exploration_bonus(0, 1, c), 0);
        // More action visits -> less pressure.
        assert!(exploration_bonus(1_000, 1, c) > exploration_bonus(1_000, 100, c));
        // More parent visits -> more pressure (log grows).
        assert!(exploration_bonus(1 << 20, 10, c) > exploration_bonus(1 << 4, 10, c));
        // Scales linearly in c — exact here because c = SCORE_ONE = 2^SHIFT
        // makes the low SHIFT bits of c*sqrt_fixed always zero (odd c would
        // truncate differently on the two sides).
        assert_eq!(
            exploration_bonus(1_000, 10, 2 * SCORE_ONE),
            2 * exploration_bonus(1_000, 10, SCORE_ONE)
        );
    }

    #[test]
    fn exploration_bonus_matches_float_reference() {
        // Floats allowed ONLY in tests. Reference: c * sqrt(log2(N+1) / n).
        for (parent, action) in [(10u64, 3u64), (1_000, 7), (1 << 30, 999), (5, 1)] {
            let got = exploration_bonus(parent, action, SCORE_ONE) as f64 / SCORE_ONE as f64;
            let want = (((parent + 1) as f64).log2().floor() / action as f64).sqrt();
            assert!(
                (got - want).abs() <= want * 0.01 + 0.002,
                "parent={parent} action={action}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn branch_lcb_is_pessimistic() {
        // Unvisited branch -> 0 (never optimistic about unknowns).
        assert_eq!(branch_lcb(0, 0, 100, SCORE_ONE, 1), 0);
        // Uncertainty subtracts (saturating at 0). With horizon=1: mean_fixed(50, 100*1)=mean_fixed(50,100).
        let mean = mean_fixed(50, 100);
        assert!(branch_lcb(100, 50, 200, SCORE_ONE, 1) < mean);
        // Huge uncertainty saturates to 0, never wraps.
        assert_eq!(branch_lcb(1, 1, u64::MAX / 2, u64::MAX / 2, 1), 0);
        // More evidence at the same mean -> higher LCB.
        assert!(
            branch_lcb(10_000, 5_000, 20_000, SCORE_ONE, 1)
                > branch_lcb(100, 50, 20_000, SCORE_ONE, 1)
        );
        // Horizon scales the denominator: visits=10, reward=320, horizon=32 -> mean=1.0.
        assert_eq!(mean_fixed(320, 10 * 32), SCORE_ONE);
    }

    // --- Task 6: selection ---

    fn expanded_root(graph: &SearchGraph, root_idx: usize) -> u32 {
        let id = graph.intern(root_keys()[root_idx]).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        id
    }

    #[test]
    fn fpu_fresh_state_prefers_first_slot_at_zero_information() {
        // FPU semantics at a completely fresh root (parent visits = 0).
        //
        // All slots unvisited, parent visits = 0:
        //   parent_mean = mean_fixed(0, 0·horizon) = 0
        //   FPU score   = (0/2) + exploration_bonus(0, 1, SCORE_ONE)
        //               = 0 + c·sqrt(log2(1)/1) = 0 + 0 = 0
        // Every slot scores 0 → deterministic tie-break gives Some(0).
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // All unvisited, parent visits = 0 → all score 0 → slot 0 wins on tie-break.
        assert_eq!(choose_search_placement(&graph, state, &params), Some(0));
        // Give slot 0 one placement visit (parent visits still 0, no reward).
        // slot 0 (visited, visits=1):
        //   value = placement_value = 0 (branches unvisited → branch_lcb=0)
        //   bonus = exploration_bonus(0, 1, SCORE_ONE) = 0
        //   score = 0
        // slot 1+ (unvisited):
        //   FPU score = (0/2) + exploration_bonus(0, 1, SCORE_ONE) = 0
        // All slots still score 0. Tie-break: slot 1 has 0 visits < slot 0's 1 visit
        // → slot 1 wins on the fewer-visits rung.
        graph
            .placements
            .get(state.placement_base)
            .visits
            .fetch_add(1, Ordering::Relaxed);
        assert_eq!(choose_search_placement(&graph, state, &params), Some(1));
    }

    #[test]
    fn selection_none_when_dead() {
        let tight = Carrier {
            max_holes: 0,
            max_height: 0,
            max_roughness: 0,
        };
        let graph = test_graph(100, tight, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let id = graph.intern(root_keys()[0]).unwrap().id;
        let state = graph.states.get(id);
        assert_eq!(choose_search_placement(&graph, state, &params), None);
        assert_eq!(choose_best_placement(&graph, state, &params), None);
    }

    #[test]
    fn best_placement_tracks_min_branch_and_ignores_exploration() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // Give every placement/branch one visit, then make slot 2 clearly best:
        // all its branches survive; every other placement has a 0-survival branch.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            pr.visits.fetch_add(1_000, Ordering::Relaxed);
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                br.visits.fetch_add(1_000, Ordering::Relaxed);
                if slot == 2 {
                    br.reward_sum.fetch_add(8_000, Ordering::Relaxed);
                }
            }
        }
        assert_eq!(choose_best_placement(&graph, state, &params), Some(2));
        // placement_value is the min over branches: zero out ONE branch of slot 2.
        let pr2 = graph.placements.get(state.placement_base + 2);
        let piece = state.key.bag().iter_pieces().next().unwrap();
        let br = graph.branches.get(pr2.branch_base + piece.index() as u32);
        // visits 2000, survivals still 1000 -> mean 0.5 on that branch.
        br.visits.fetch_add(1_000, Ordering::Relaxed);
        let value = placement_value(&graph, state, pr2, 0, params.horizon as u64);
        assert_eq!(value, Some(SCORE_ONE / 2));
    }

    #[test]
    fn adversarial_piece_prefers_unexpanded_then_unvisited_then_min_lcb() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        let pr = graph.placements.get(state.placement_base);
        let bag_pieces: Vec<TetrisPiece> = state.key.bag().iter_pieces().collect();
        // Fully expanded: no UNEXPANDED among bag pieces -> first unvisited wins.
        assert_eq!(
            choose_piece(&graph, state, pr, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            bag_pieces[0]
        );
        // Visit all but one branch; the unvisited one must be chosen.
        for &piece in &bag_pieces[..bag_pieces.len() - 1] {
            let br = graph.branches.get(pr.branch_base + piece.index() as u32);
            br.visits.fetch_add(10, Ordering::Relaxed);
            br.reward_sum.fetch_add(80, Ordering::Relaxed);
        }
        assert_eq!(
            choose_piece(&graph, state, pr, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            bag_pieces[bag_pieces.len() - 1]
        );
        // All visited: min LCB wins. Make bag_pieces[1] the weakest (0 reward).
        let last = graph
            .branches
            .get(pr.branch_base + bag_pieces[bag_pieces.len() - 1].index() as u32);
        last.visits.fetch_add(10, Ordering::Relaxed);
        last.reward_sum.fetch_add(80, Ordering::Relaxed);
        let weak = graph
            .branches
            .get(pr.branch_base + bag_pieces[1].index() as u32);
        weak.visits.fetch_add(1_000, Ordering::Relaxed); // mean collapses toward 0
        assert_eq!(
            choose_piece(&graph, state, pr, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            bag_pieces[1]
        );
    }

    #[test]
    fn tie_break_is_deterministic_lowest_slot() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: 0,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // Identical statistics everywhere -> equal scores -> lowest slot wins.
        for slot in 0..state.placement_len as u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            pr.visits.fetch_add(5, Ordering::Relaxed);
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                br.visits.fetch_add(5, Ordering::Relaxed);
                br.reward_sum.fetch_add(40, Ordering::Relaxed);
            }
        }
        assert_eq!(choose_search_placement(&graph, state, &params), Some(0));
        assert_eq!(choose_best_placement(&graph, state, &params), Some(0));
    }

    #[test]
    fn tie_break_prefers_fewer_visits_at_equal_score() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: 0,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let id = expanded_root(&graph, 0);
        let state = graph.states.get(id);
        // Slots 0 and 1 get identical branch stats (equal value with c_lcb = 0)
        // but different placement visits. The middle tie-break rung must prefer
        // the less-visited slot 1. (Only choose_best_placement is assertable
        // here: choose_search_placement would short-circuit to an unvisited
        // slot 2+ first.)
        for slot in 0..2u32 {
            let pr = graph.placements.get(state.placement_base + slot);
            pr.visits
                .fetch_add(if slot == 0 { 500 } else { 100 }, Ordering::Relaxed);
            for piece in state.key.bag().iter_pieces() {
                let br = graph.branches.get(pr.branch_base + piece.index() as u32);
                br.visits.fetch_add(10, Ordering::Relaxed);
                br.reward_sum.fetch_add(80, Ordering::Relaxed);
            }
        }
        assert_eq!(choose_best_placement(&graph, state, &params), Some(1));
    }

    // --- synthetic graph helpers (Tasks 7-8) ---

    /// Hand-built state bypassing the interner: `marker` differentiates keys,
    /// `bag_bits` sets the AND branching, `n_placements` placements each get 7
    /// UNEXPANDED branch slots. Status starts FULLY_EXPANDED (wire children
    /// yourself; leave a branch unwired to model an unknown edge).
    fn synth_state(graph: &SearchGraph, marker: u32, bag_bits: u8, n_placements: u8) -> u32 {
        let mut limbs = [0u32; 10];
        limbs[0] = marker;
        let key = StateKey {
            board: TetrisBoard::from_limbs(limbs),
            bag_bits,
            piece_idx: 0,
        };
        let placement_base = if n_placements == 0 {
            0
        } else {
            let base = graph.placements.reserve(n_placements as u32).unwrap();
            let branch_base = graph.branches.reserve(n_placements as u32 * 7).unwrap();
            for i in 0..n_placements as u32 {
                graph.placements.set(
                    base + i,
                    PlacementRecord {
                        placement: TetrisPiecePlacement::from_index(0),
                        branch_base: branch_base + i * 7,
                        visits: AtomicU64::new(0),
                        reward_sum: AtomicU64::new(0),
                    },
                );
            }
            base
        };
        let id = graph.states.reserve(1).unwrap();
        graph.states.set(
            id,
            StateRecord {
                key,
                visits: AtomicU64::new(0),
                reward_sum: AtomicU64::new(0),
                placement_base,
                placement_len: n_placements,
                expansion_status: AtomicU8::new(STATUS_FULLY_EXPANDED),
                frontier_queued: AtomicU8::new(1),
                proof: AtomicU8::new(PROOF_UNKNOWN),
            },
        );
        id
    }

    fn wire(graph: &SearchGraph, parent: u32, slot: u32, piece: TetrisPiece, child: u32) {
        let state = graph.states.get(parent);
        let pr = graph.placements.get(state.placement_base + slot);
        graph
            .branches
            .get(pr.branch_base + piece.index() as u32)
            .child
            .store(child, Ordering::Release);
    }

    // --- Task 7: rollout ---

    const O_BAG: u8 = 0b0000_0001; // bag containing only the O piece

    #[test]
    fn rollout_death_on_dead_state() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // A -> B where B has no placements (terminal death).
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 0);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(
            rollout(&graph, &params, a, &mut scratch),
            RolloutOutcome::Death
        ));
        // Each traversed node updated exactly once. Remaining-depth credit:
        // the dying terminal b earns 0 (no future survival from b); a earns
        // 1 (play continued one step from a).
        assert_eq!(graph.states.get(a).visits.load(Ordering::Relaxed), 1);
        assert_eq!(graph.states.get(b).visits.load(Ordering::Relaxed), 1);
        assert_eq!(graph.states.get(b).reward_sum.load(Ordering::Relaxed), 0);
        assert_eq!(graph.states.get(a).reward_sum.load(Ordering::Relaxed), 1);
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        assert_eq!(pr.visits.load(Ordering::Relaxed), 1);
        assert_eq!(pr.reward_sum.load(Ordering::Relaxed), 1);
        let br = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        assert_eq!(br.visits.load(Ordering::Relaxed), 1);
        assert_eq!(br.reward_sum.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rollout_survives_exact_horizon() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // Self-loop; horizon 1 stops before the cycle check fires twice.
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        let params = SearchParams {
            horizon: 1,
            credit_cap: 1,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(
            rollout(&graph, &params, a, &mut scratch),
            RolloutOutcome::Horizon
        ));
        assert_eq!(graph.states.get(a).reward_sum.load(Ordering::Relaxed), 1);
        // ^ +1 as path entry (remaining depth 1), +0 as final state (remaining
        // depth 0 at the horizon); visits counted both times.
        assert_eq!(graph.states.get(a).visits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn rollout_cycle_detected_and_scored_as_survival() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        let params = SearchParams {
            horizon: 100,
            credit_cap: 100,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        assert!(matches!(
            rollout(&graph, &params, a, &mut scratch),
            RolloutOutcome::Cycle
        ));
        // Reward credit on the traversed placement/branch.
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        assert_eq!(pr.reward_sum.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn rollout_scratch_does_not_grow() {
        let graph = test_graph(10_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let params = SearchParams {
            horizon: 16,
            credit_cap: 16,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let roots: Vec<u32> = root_keys()
            .iter()
            .map(|&k| graph.intern(k).unwrap().id)
            .collect();
        let mut scratch = WorkerScratch::new(params.horizon);
        let path_cap = scratch.path.capacity();
        for i in 0..200 {
            let _ = rollout(&graph, &params, roots[i % 7], &mut scratch);
        }
        assert_eq!(scratch.path.capacity(), path_cap, "path reallocated");
    }

    // --- Task 8: closure ---

    fn closure_of(graph: &SearchGraph) -> ClosureReport {
        // Roots don't matter for the synthetic tests; state 0 always exists.
        extract_safe_set(graph, &[0, 0, 0, 0, 0, 0, 0])
    }

    #[test]
    fn safe_cycle_is_retained() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, c);
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a);
        let report = extract_safe_set(&graph, &[a, b, c, a, b, c, a]);
        assert_eq!(report.k_size, 3);
        assert_eq!(report.witnesses[a as usize], 0);
        // Already a fixpoint: one pass proves it (final zero-removal pass counted).
        assert_eq!(report.iterations, 1);
        assert!(report.roots_in_k.iter().all(|&in_k| in_k));
    }

    #[test]
    fn poisoned_chain_death_propagates_backward() {
        // a -> b -> dead, c -> a: a linear CHAIN (not a cycle).
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, O_BAG, 1);
        let dead = synth_state(&graph, 4, O_BAG, 0); // no placements: not in K0
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, dead); // poison at the chain head
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a);
        let report = closure_of(&graph);
        assert_eq!(
            report.k_size, 0,
            "death propagates back through the whole chain"
        );
        // propagate_death (lfp) now runs before K0 construction, so all states are
        // excluded from K0; the GFP loop runs a single zero-removal pass.
        assert_eq!(report.iterations, 1);
        // a == state 0, so closure_of's [0; 7] roots all report dead.
        assert!(report.roots_in_k.iter().all(|&in_k| !in_k));
        assert_eq!(
            graph.states.get(a).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "a must be PROOF_DEAD (death propagated backward)"
        );
        assert_eq!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "b must be PROOF_DEAD (death propagated backward)"
        );
        assert_eq!(
            graph.states.get(c).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "c must be PROOF_DEAD (death propagated backward)"
        );
        assert_eq!(
            graph.states.get(dead).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "dead must be PROOF_DEAD (zero placements)"
        );
    }

    #[test]
    fn poisoned_cycle_is_rejected() {
        // A real cycle a -> b -> c -> a stays fully wired, but c's bag is
        // {O, I} and its I branch leads to a dead state: the ∀ kills c even
        // though its O branch continues the loop, and death then propagates
        // around the intact cycle edges.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, OI_BAG, 1);
        let dead = synth_state(&graph, 4, O_BAG, 0);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, c);
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a); // cycle edge stays live
        wire(&graph, c, 0, TetrisPiece::I_PIECE, dead); // the poison
        let report = closure_of(&graph);
        assert_eq!(
            report.k_size, 0,
            "one poisoned AND-branch kills the whole cycle"
        );
        // propagate_death (lfp) now runs before K0 construction, so all states are
        // excluded from K0; the GFP loop runs a single zero-removal pass.
        assert_eq!(report.iterations, 1);
        assert_eq!(
            graph.states.get(a).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "a must be PROOF_DEAD (cycle fully poisoned)"
        );
        assert_eq!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "b must be PROOF_DEAD (cycle fully poisoned)"
        );
        assert_eq!(
            graph.states.get(c).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "c must be PROOF_DEAD (I branch hits dead)"
        );
        assert_eq!(
            graph.states.get(dead).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "dead must be PROOF_DEAD (zero placements)"
        );
    }

    #[test]
    fn unknown_blocked_cascade_requires_multi_pass_gfp() {
        // No state is dead (death-lfp cannot fire): z is merely BLOCKED by an
        // unexpanded I branch, and its removal must cascade back through y
        // then x across successive gfp passes — the one pruning mechanism
        // death propagation cannot replace.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let x = synth_state(&graph, 1, O_BAG, 1);
        let y = synth_state(&graph, 2, O_BAG, 1);
        let z = synth_state(&graph, 3, OI_BAG, 1);
        let k_leaf = synth_state(&graph, 4, O_BAG, 1);
        wire(&graph, x, 0, TetrisPiece::O_PIECE, y);
        wire(&graph, y, 0, TetrisPiece::O_PIECE, z);
        wire(&graph, z, 0, TetrisPiece::O_PIECE, k_leaf); // z's O branch is fine...
        // ...but z's I branch stays UNEXPANDED — z fails the ∀ without being dead.
        wire(&graph, k_leaf, 0, TetrisPiece::O_PIECE, k_leaf); // the K anchor
        let report = extract_safe_set(&graph, &[k_leaf; 7]);
        assert_eq!(report.dead_total, 0, "nothing is provably dead here");
        assert_eq!(report.k_size, 1, "only the self-loop anchor closes");
        // K0 = {x,y,z,k_leaf}; pass 1 removes z (unknown branch), pass 2 removes
        // y (child z left K), pass 3 removes x, pass 4 confirms the fixpoint.
        assert_eq!(report.iterations, 4);
        assert_eq!(report.pass_removed, vec![1, 1, 1, 0]);
        // x, y, z each fail by exactly one branch against the final K.
        assert_eq!(report.near_miss, 3);
        // Proof separation: blocked-but-alive states stay UNKNOWN, never DEAD.
        assert_eq!(
            graph.states.get(k_leaf).proof.load(Ordering::Acquire),
            PROOF_SAFE
        );
        for id in [x, y, z] {
            assert_eq!(
                graph.states.get(id).proof.load(Ordering::Acquire),
                PROOF_UNKNOWN
            );
        }
    }

    #[test]
    fn unknown_branch_blocks_certification() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011; // bag {O, I}: two required branches
        let a = synth_state(&graph, 1, OI_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, b); // b: safe self-loop
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        // a's I branch stays UNEXPANDED -> a fails the ∀; b survives alone.
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 1);
        assert_eq!(report.witnesses[b as usize], 0);
        assert_eq!(report.witnesses[a as usize], u8::MAX);
        // Positive control: wiring the unknown I branch brings `a` into K —
        // proving the ∀ genuinely spans both bag pieces.
        wire(&graph, a, 0, TetrisPiece::I_PIECE, b);
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 2);
        assert_eq!(report.witnesses[a as usize], 0);
    }

    #[test]
    fn witness_skips_failing_placement() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2); // two placements
        let dead = synth_state(&graph, 2, O_BAG, 0);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead); // slot 0 fails
        wire(&graph, s, 1, TetrisPiece::O_PIECE, s); // slot 1 keeps s safe
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 1);
        assert_eq!(
            report.witnesses[s as usize], 1,
            "witness is the qualifying slot"
        );
    }

    #[test]
    fn k0_requires_fully_expanded() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        // Force status back to QUEUED: a would be safe, but is not fully expanded.
        graph
            .states
            .get(a)
            .expansion_status
            .store(STATUS_QUEUED, Ordering::Release);
        let report = closure_of(&graph);
        assert_eq!(report.k_size, 0);
    }

    // --- Task 9: epochs / integration ---

    fn drain_expand_universe(graph: &SearchGraph, pool: &rayon::ThreadPool) {
        loop {
            let drained = expansion_batch(graph, pool, u64::MAX, 100);
            if drained == 0 {
                break;
            }
        }
    }

    #[test]
    fn expansion_topology_is_thread_count_invariant() {
        // Tiny bounded universe: carrier holes 0 / height 1 / roughness 2.
        // Height 1 is very restrictive (most pieces create height ≥ 2) so the
        // universe is small enough to fit in the arena and be fully explored.
        let carrier = Carrier {
            max_holes: 0,
            max_height: 1,
            max_roughness: 2,
        };
        let collect_keys = |threads: usize| -> Vec<StateKey> {
            let graph = test_graph(50_000, carrier, ExpansionPolicy::AllDiscovered);
            for key in root_keys() {
                graph.intern(key).unwrap();
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            drain_expand_universe(&graph, &pool);
            // Verify we didn't hit capacity (would cause non-determinism).
            assert!(
                !graph.exhausted.load(Ordering::SeqCst),
                "hit capacity — universe larger than max_states; tighten the carrier"
            );
            let mut keys: Vec<StateKey> = (0..graph.states.len())
                .map(|id| graph.states.get(id).key)
                .collect();
            keys.sort_unstable();
            keys
        };
        let single = collect_keys(1);
        let multi = collect_keys(4);
        assert!(!single.is_empty());
        assert_eq!(
            single, multi,
            "fully expanded topology must not depend on scheduling"
        );
    }

    #[test]
    fn smoke_run_bounded() {
        let args = Args {
            threads: 2,
            epochs: Some(3),
            rollouts_per_epoch: 2_000,
            expansions_per_epoch: 2_000,
            closure_every: 1,
            horizon: 32,
            credit_cap: 32,
            max_states: 50_000,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::All,
            expand_occupancy: 100,
            admit_after: 1,
            compact_at: 90,
            keep_frac: 50,
            decay_shift: 1,
            no_compact: false,
            root_seed_pct: 75,
            csv: None,
            ledger: None,
            ledger_save_every: 0,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let stop = AtomicBool::new(false);
        let outcome = run_search(&args, &stop).unwrap();
        assert_eq!(outcome.report.roots_in_k.len(), 7);
        // One witness entry per interned state; at minimum the 7 roots exist.
        assert!(outcome.report.witnesses.len() >= 7);
    }

    #[test]
    fn smoke_run_bounded_candidate_policy() {
        let args = Args {
            threads: 2,
            epochs: Some(3),
            rollouts_per_epoch: 2_000,
            expansions_per_epoch: 2_000,
            closure_every: 1,
            horizon: 32,
            credit_cap: 32,
            max_states: 50_000,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::Candidate,
            expand_occupancy: 100,
            admit_after: 1,
            compact_at: 90,
            keep_frac: 50,
            decay_shift: 1,
            no_compact: false,
            root_seed_pct: 75,
            csv: None,
            ledger: None,
            ledger_save_every: 0,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let stop = AtomicBool::new(false);
        let outcome = run_search(&args, &stop).unwrap();
        assert_eq!(outcome.report.roots_in_k.len(), 7);
        // One witness entry per interned state; at minimum the 7 roots exist.
        assert!(outcome.report.witnesses.len() >= 7);
        // Candidate mode drains admitted states through the frontier across
        // epochs — the frontier must not still hold everything at the end.
        // (Roots are force-admitted; rollout-visited states are admitted on
        // first visit; the expansion batch is the sole FULLY_EXPANDED driver.)
        assert!(
            outcome.report.witnesses.len() > 7,
            "candidate mode interned states beyond the 7 roots"
        );
    }

    // --- Task AB: intern fast-fail reorder, rollout pessimistic death, --admit-after ---

    #[test]
    fn intern_existing_key_resolves_after_exhaustion() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let id = graph.intern(key).unwrap().id;
        graph.exhausted.store(true, Ordering::SeqCst);
        // Existing keys still resolve after exhaustion (miss-path-only fast-fail).
        let again = graph.intern(key).unwrap();
        assert_eq!(again.id, id);
        assert!(!again.newly_created);
        // Genuinely new keys still fail fast.
        assert!(graph.intern(root_keys()[1]).is_err());
    }

    #[test]
    fn rollout_unresolvable_edge_scores_death() {
        // 1-state arena: the root interns, but its first child cannot.
        let graph = SearchGraph::new(
            1,
            34,
            238,
            Carrier::UNBOUNDED,
            ExpansionPolicy::AllDiscovered,
        );
        let root = graph.intern(root_keys()[0]).unwrap().id;
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        let outcome = rollout(&graph, &params, root, &mut scratch);
        assert!(matches!(outcome, RolloutOutcome::Aborted));
        // The truncation was scored as a death at the root (visit, no survival).
        let state = graph.states.get(root);
        assert_eq!(state.visits.load(Ordering::Relaxed), 1);
        assert_eq!(state.reward_sum.load(Ordering::Relaxed), 0);
        assert!(graph.exhausted.load(Ordering::SeqCst));
    }

    #[test]
    fn candidate_admission_respects_admit_after() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::CandidateOnly);
        // Synth self-loop; each Cycle rollout bumps its visits by 2
        // (path entry + terminal), so with admit_after = 3 the check
        // (visits + 1 >= 3) fails on rollout 1 (0+1) and passes on rollout 2 (2+1).
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        graph
            .states
            .get(a)
            .frontier_queued
            .store(0, Ordering::Release); // synth defaults to 1
        let params = SearchParams {
            horizon: 50,
            credit_cap: 50,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 3,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        let _ = rollout(&graph, &params, a, &mut scratch);
        assert_eq!(graph.frontier.len(), 0, "below threshold: not admitted");
        let _ = rollout(&graph, &params, a, &mut scratch);
        assert_eq!(
            graph.frontier.len(),
            1,
            "threshold crossed: admitted exactly once"
        );
        let _ = rollout(&graph, &params, a, &mut scratch);
        assert_eq!(
            graph.frontier.len(),
            1,
            "queued-once flag prevents re-admission"
        );
    }

    // --- Task 10: hygiene ---

    #[test]
    fn no_floats_outside_tests() {
        // The search engine is integer-only by spec. Floats may appear only in
        // this test module (reference UCT). Scan our own source up to `mod tests`.
        let source = include_str!("tetris_atlas_puct.rs");
        let (engine, _tests) = source
            .split_once("mod tests")
            .expect("tests module marker present");
        for token in ["f32", "f64"] {
            assert!(
                !engine.contains(token),
                "float type `{token}` found in search engine code"
            );
        }
    }

    // --- Task C1: survivor selection ---

    fn empty_closure_report(n: usize) -> ClosureReport {
        ClosureReport {
            k_size: 0,
            iterations: 0,
            roots_in_k: [false; 7],
            witnesses: vec![u8::MAX; n],
            k0_size: 0,
            pass_removed: Vec::new(),
            near_miss: 0,
            boundary_ids: Vec::new(),
            dead_total: 0,
        }
    }

    #[test]
    fn select_survivors_protects_roots_and_k() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // States 0..=3: a safe self-loop (in K), and junk.
        let a = synth_state(&graph, 1, O_BAG, 1); // 0: K member
        let b = synth_state(&graph, 2, O_BAG, 1); // 1: junk
        let c = synth_state(&graph, 3, O_BAG, 1); // 2: junk with visits
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        graph
            .states
            .get(c)
            .visits
            .fetch_add(1_000, Ordering::Relaxed);
        let mut protect = empty_closure_report(graph.states.len() as usize);
        protect.witnesses[a as usize] = 0; // a ∈ K
        // keep_frac 0: survivors are exactly the protected set.
        let roots = [a, a, a, a, a, a, a]; // synthetic roots alias a
        let survive = select_survivors(&graph, &protect, &roots, 0);
        assert!(survive[a as usize], "K member survives");
        assert!(!survive[b as usize], "junk evicted");
        assert!(
            !survive[c as usize],
            "even hot junk evicted at keep 0 when not protected"
        );
    }

    #[test]
    fn select_survivors_takes_top_by_visits_deterministically() {
        // Capacity 10 so keep_frac 50 -> keep_target 5.
        let graph = test_graph(10, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let ids: Vec<u32> = (0..8)
            .map(|i| synth_state(&graph, i + 1, O_BAG, 1))
            .collect();
        // Visits: ids[0]=root-alias (protected anyway), others graded.
        let visits = [0u64, 5, 10, 10, 3, 0, 7, 1];
        for (i, &id) in ids.iter().enumerate() {
            graph
                .states
                .get(id)
                .visits
                .fetch_add(visits[i], Ordering::Relaxed);
        }
        let protect = empty_closure_report(graph.states.len() as usize);
        let roots = [ids[0]; 7];
        let survive = select_survivors(&graph, &protect, &roots, 50);
        // Protected: ids[0]. Remaining budget 4 -> top visits: 10,10,7,5
        // (ids[2], ids[3] tie at 10 — both fit; then 6 at 7, then 1 at 5).
        assert!(survive[ids[0] as usize], "root protected despite 0 visits");
        assert!(survive[ids[2] as usize] && survive[ids[3] as usize]);
        assert!(survive[ids[6] as usize]);
        assert!(survive[ids[1] as usize]);
        assert!(
            !survive[ids[4] as usize] && !survive[ids[5] as usize] && !survive[ids[7] as usize]
        );
        assert_eq!(survive.iter().filter(|s| **s).count(), 5);
    }

    #[test]
    fn select_survivors_tie_breaks_toward_lower_id() {
        let graph = test_graph(10, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let ids: Vec<u32> = (0..4)
            .map(|i| synth_state(&graph, i + 1, O_BAG, 1))
            .collect();
        // All equal visits; keep_target 2 (capacity 10 * 20% = 2); roots alias ids[0].
        for &id in &ids {
            graph.states.get(id).visits.fetch_add(5, Ordering::Relaxed);
        }
        let protect = empty_closure_report(graph.states.len() as usize);
        let roots = [ids[0]; 7];
        let survive = select_survivors(&graph, &protect, &roots, 20);
        // ids[0] protected; 1 slot left -> lowest id among equals = ids[1].
        assert!(survive[ids[0] as usize] && survive[ids[1] as usize]);
        assert!(!survive[ids[2] as usize] && !survive[ids[3] as usize]);
    }

    // --- Task C2: compaction copy/remap + faithfulness validator ---

    #[test]
    fn compaction_preserves_k_and_evicts_junk() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // Safe 3-cycle (will be K) + zero-visit junk.
        let a = synth_state(&graph, 1, O_BAG, 1);
        let b = synth_state(&graph, 2, O_BAG, 1);
        let c = synth_state(&graph, 3, O_BAG, 1);
        let junk1 = synth_state(&graph, 4, O_BAG, 1);
        let junk2 = synth_state(&graph, 5, O_BAG, 0);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, c);
        wire(&graph, c, 0, TetrisPiece::O_PIECE, a);
        wire(&graph, junk1, 0, TetrisPiece::O_PIECE, junk2);
        let roots = [a; 7];
        let protect = extract_safe_set(&graph, &roots);
        assert_eq!(protect.k_size, 3);
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 1,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        // Only the protected K survived; junk gone.
        assert_eq!(stats.kept, 3);
        assert_eq!(stats.evicted, 2);
        assert_eq!(new.states.len(), 3);
        // K is preserved EXACTLY on the new graph (monotone across compaction).
        let new_a = *new.ids.get(&graph.states.get(a).key).unwrap();
        let new_report = extract_safe_set(&new, &[new_a; 7]);
        assert_eq!(new_report.k_size, 3);
        assert!(new_report.roots_in_k.iter().all(|&x| x));
    }

    #[test]
    fn compaction_remap_is_faithful_on_real_graph() {
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let mut roots = [0u32; 7];
        for (i, key) in root_keys().into_iter().enumerate() {
            roots[i] = graph.intern(key).unwrap().id;
        }
        for &r in &roots {
            graph.ensure_fully_expanded(r).unwrap();
        }
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        for i in 0..200 {
            let _ = rollout(&graph, &params, roots[i % 7], &mut scratch);
        }
        let protect = extract_safe_set(&graph, &roots);
        let cfg = CompactConfig {
            keep_frac_pct: 50,
            decay_shift: 1,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert!(stats.kept > 7, "roots plus hot states survive");
        assert!(
            compaction_is_faithful(&new),
            "every remapped edge is canonical"
        );
        assert_eq!(
            new.ids.len() as u32,
            new.states.len(),
            "interner and arena agree"
        );
        // Roots survived and are resolvable by key.
        for key in root_keys() {
            assert!(new.ids.get(&key).is_some());
        }

        // Second compaction: roots-only survive (keep_frac 0) — exercises the
        // evicted-edge-reset path on a post-compaction graph.
        let new_root_ids: [u32; 7] = root_keys().map(|key| *new.ids.get(&key).unwrap());
        let new_protect = extract_safe_set(&new, &new_root_ids);
        let cfg2 = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (newer, _) = compact(&new, &new_protect, &new_root_ids, &cfg2).unwrap();

        // (a) The strict faithfulness validator must hold even when most children were evicted.
        assert!(
            compaction_is_faithful(&newer),
            "faithfulness must hold after aggressive eviction"
        );

        // (b) At least one root placement's branch was reset to UNEXPANDED
        //     (evicted-edge reset actually happened).
        let newer_root_ids: [u32; 7] = root_keys().map(|key| *newer.ids.get(&key).unwrap());
        let mut found_unexpanded = false;
        'outer: for &rid in &newer_root_ids {
            let state = newer.states.get(rid);
            for slot in 0..state.placement_len as u32 {
                let pr = newer.placements.get(state.placement_base + slot);
                for piece_idx in 0..7u32 {
                    let br = newer.branches.get(pr.branch_base + piece_idx);
                    if br.child.load(Ordering::Acquire) == UNEXPANDED {
                        found_unexpanded = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(
            found_unexpanded,
            "at least one root branch should be UNEXPANDED after aggressive eviction"
        );

        // (c) interner and arena agree on the new graph.
        assert_eq!(
            newer.ids.len() as u32,
            newer.states.len(),
            "interner and arena agree after second compaction"
        );
    }

    #[test]
    fn compaction_decays_stats() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        graph.states.get(a).visits.fetch_add(100, Ordering::Relaxed);
        graph
            .states
            .get(a)
            .reward_sum
            .fetch_add(40, Ordering::Relaxed);
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        pr.visits.fetch_add(60, Ordering::Relaxed);
        pr.reward_sum.fetch_add(20, Ordering::Relaxed);
        let br = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br.visits.fetch_add(30, Ordering::Relaxed);
        br.reward_sum.fetch_add(10, Ordering::Relaxed);
        let roots = [a; 7];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 1,
        }; // roots protect a
        let (new, _) = compact(&graph, &protect, &roots, &cfg).unwrap();
        let na = *new.ids.get(&graph.states.get(a).key).unwrap();
        let ns = new.states.get(na);
        assert_eq!(ns.visits.load(Ordering::Relaxed), 50);
        assert_eq!(ns.reward_sum.load(Ordering::Relaxed), 20);
        let npr = new.placements.get(ns.placement_base);
        assert_eq!(npr.visits.load(Ordering::Relaxed), 30);
        assert_eq!(npr.reward_sum.load(Ordering::Relaxed), 10); // placement reward_sum >> 1
        let nbr = new
            .branches
            .get(npr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        assert_eq!(nbr.visits.load(Ordering::Relaxed), 15);
        assert_eq!(nbr.reward_sum.load(Ordering::Relaxed), 5); // branch reward_sum >> 1
        // shift 0 = no decay
        let cfg0 = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new0, _) = compact(&graph, &protect, &roots, &cfg0).unwrap();
        let na0 = *new0.ids.get(&graph.states.get(a).key).unwrap();
        assert_eq!(new0.states.get(na0).visits.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn compaction_resets_evicted_edges_keeps_label() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let parent = synth_state(&graph, 1, O_BAG, 1); // FULLY_EXPANDED by synth default
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, parent, 0, TetrisPiece::O_PIECE, child);
        graph
            .states
            .get(parent)
            .visits
            .fetch_add(100, Ordering::Relaxed);
        // child has 0 visits -> evicted (only roots-protected parent survives).
        let roots = [parent; 7];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert_eq!(stats.kept, 1);
        let np = *new.ids.get(&graph.states.get(parent).key).unwrap();
        let ns = new.states.get(np);
        // Label kept, edge reset.
        assert_eq!(
            ns.expansion_status.load(Ordering::Acquire),
            STATUS_FULLY_EXPANDED
        );
        let nbr = new.branches.get(
            new.placements.get(ns.placement_base).branch_base + TetrisPiece::O_PIECE.index() as u32,
        );
        assert_eq!(nbr.child.load(Ordering::Acquire), UNEXPANDED);
    }

    #[test]
    fn compaction_requeues_queued_survivors_once() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let queued = synth_state(&graph, 1, O_BAG, 1);
        let expanded = synth_state(&graph, 2, O_BAG, 1);
        // A third state: STATUS_QUEUED but frontier_queued == 0 (never admitted).
        // It has high visits so it survives the budget cut, but must NOT be requeued
        // because frontier_queued == 0 means it was never admitted to the frontier.
        let unadmitted = synth_state(&graph, 3, O_BAG, 1);
        graph
            .states
            .get(unadmitted)
            .expansion_status
            .store(STATUS_QUEUED, Ordering::Release);
        graph
            .states
            .get(unadmitted)
            .frontier_queued
            .store(0, Ordering::Release); // never admitted
        wire(&graph, expanded, 0, TetrisPiece::O_PIECE, expanded);
        // queued: pending expansion (status QUEUED, flag 1 = was admitted).
        graph
            .states
            .get(queued)
            .expansion_status
            .store(STATUS_QUEUED, Ordering::Release);
        graph
            .states
            .get(queued)
            .visits
            .fetch_add(50, Ordering::Relaxed);
        graph
            .states
            .get(expanded)
            .visits
            .fetch_add(50, Ordering::Relaxed);
        graph
            .states
            .get(unadmitted)
            .visits
            .fetch_add(50, Ordering::Relaxed);
        // Use roots array to guarantee all three survive regardless of budget.
        let roots = [
            queued, expanded, unadmitted, queued, expanded, unadmitted, queued,
        ];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 100,
            decay_shift: 0,
        };
        let (new, _) = compact(&graph, &protect, &roots, &cfg).unwrap();
        // Exactly the QUEUED+admitted survivor is on the new frontier, exactly once.
        // The unadmitted QUEUED state must NOT be spuriously requeued.
        assert_eq!(new.frontier.len(), 1);
        let nq = *new.ids.get(&graph.states.get(queued).key).unwrap();
        assert_eq!(new.frontier.pop(), Some(nq));
    }

    #[test]
    fn compaction_transfers_counters_and_resets_exhausted() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        graph.counters.rollouts.fetch_add(1234, Ordering::Relaxed);
        graph.counters.deaths.fetch_add(999, Ordering::Relaxed);
        graph.counters.horizons.fetch_add(11, Ordering::Relaxed);
        graph.counters.cycles.fetch_add(22, Ordering::Relaxed);
        graph.counters.aborted.fetch_add(33, Ordering::Relaxed);
        graph
            .counters
            .edges_expanded
            .fetch_add(77, Ordering::Relaxed);
        graph
            .counters
            .full_expansions
            .fetch_add(44, Ordering::Relaxed);
        graph
            .counters
            .max_rollout_depth
            .fetch_max(42, Ordering::Relaxed);
        graph
            .counters
            .rollout_depth_sum
            .fetch_add(555, Ordering::Relaxed);
        graph.exhausted.store(true, Ordering::SeqCst);
        let roots = [a; 7];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 100,
            decay_shift: 1,
        };
        let (new, _) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert_eq!(new.counters.rollouts.load(Ordering::Relaxed), 1234);
        assert_eq!(new.counters.deaths.load(Ordering::Relaxed), 999);
        assert_eq!(new.counters.horizons.load(Ordering::Relaxed), 11);
        assert_eq!(new.counters.cycles.load(Ordering::Relaxed), 22);
        assert_eq!(new.counters.aborted.load(Ordering::Relaxed), 33);
        assert_eq!(new.counters.edges_expanded.load(Ordering::Relaxed), 77);
        assert_eq!(new.counters.full_expansions.load(Ordering::Relaxed), 44);
        assert_eq!(new.counters.max_rollout_depth.load(Ordering::Relaxed), 42);
        assert_eq!(new.counters.rollout_depth_sum.load(Ordering::Relaxed), 555);
        assert!(
            !new.exhausted.load(Ordering::SeqCst),
            "fresh headroom: exhausted resets"
        );
    }

    #[test]
    fn record_rollout_accumulates_depth() {
        let counters = Counters::default();
        counters.record_rollout(RolloutOutcome::Death, 7);
        counters.record_rollout(RolloutOutcome::Horizon, 32);
        counters.record_rollout(RolloutOutcome::Cycle, 11);
        assert_eq!(counters.rollout_depth_sum.load(Ordering::Relaxed), 50);
        assert_eq!(counters.rollouts.load(Ordering::Relaxed), 3);
        assert_eq!(counters.max_rollout_depth.load(Ordering::Relaxed), 32);
    }

    #[test]
    fn compaction_preserves_k_with_evicted_nonwitness_branch() {
        // Synth graph: K member `s` with TWO placements.
        //   slot 0 (O piece) -> s  (safe self-loop, witness)
        //   slot 1 (O piece) -> junk  (0 visits, will be evicted)
        // Junk is evicted (keep_frac 0, protect from extract_safe_set on old graph).
        // After compact: s survives; K must still contain s; slot 1's child must be UNEXPANDED.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2); // two placements
        // OI bag: junk fails the closure by TWO unexpanded branches, so it is
        // NOT near-miss (boundary protection would otherwise keep it alive).
        const JUNK_BAG: u8 = 0b0000_0011;
        let junk = synth_state(&graph, 2, JUNK_BAG, 1);

        // slot 0: O -> s (self-loop, makes s a K member via witness slot 0)
        wire(&graph, s, 0, TetrisPiece::O_PIECE, s);
        // slot 1: O -> junk (non-witness placement; junk has 0 visits)
        wire(&graph, s, 1, TetrisPiece::O_PIECE, junk);
        // Wire junk to itself so it's nominally expandable (placement_len=1), but
        // it won't be protected: protect comes from extract_safe_set which finds
        // only s in K (junk's O branch = junk which is not in K0 since it's not
        // FULLY_EXPANDED in closure sense -- actually synth_state sets FULLY_EXPANDED
        // but junk's self-loop isn't wired, so it fails the ∀). Leave junk unwired.

        // Verify: old graph extract_safe_set finds s in K (slot 0 self-loop) but not junk.
        let roots = [s; 7];
        let protect = extract_safe_set(&graph, &roots);
        assert_eq!(
            protect.k_size, 1,
            "only s is in K (junk is not wired into a cycle)"
        );
        assert_eq!(
            protect.witnesses[s as usize], 0,
            "witness is slot 0 (self-loop)"
        );

        // Compact: keep_frac 0 -> only protected (s) survives; junk is evicted.
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert_eq!(stats.kept, 1, "only s survived");

        // s survives under its new id.
        let new_s = *new.ids.get(&graph.states.get(s).key).unwrap();

        // Fresh extract_safe_set on new graph: K must still have s with k_size == 1.
        let new_roots = [new_s; 7];
        let new_report = extract_safe_set(&new, &new_roots);
        assert_eq!(
            new_report.k_size, 1,
            "K preserved: s is still in K after compaction"
        );
        // Witness must still be slot 0 (the self-loop).
        assert_eq!(
            new_report.witnesses[new_s as usize], 0,
            "witness slot 0 preserved across compaction"
        );

        // Slot 1's branch child must be UNEXPANDED (evicted-edge reset).
        let ns = new.states.get(new_s);
        let pr1 = new.placements.get(ns.placement_base + 1);
        let br1 = new
            .branches
            .get(pr1.branch_base + TetrisPiece::O_PIECE.index() as u32);
        assert_eq!(
            br1.child.load(Ordering::Acquire),
            UNEXPANDED,
            "slot 1's branch (to evicted junk) must be reset to UNEXPANDED"
        );
    }

    #[test]
    fn smoke_run_with_compaction() {
        // Arena far too small for the carrier: without compaction this exits
        // in ~1 epoch (exhausted). With compaction it must run ALL epochs,
        // compacting as needed.
        let args = Args {
            threads: 2,
            epochs: Some(6),
            rollouts_per_epoch: 2_000,
            expansions_per_epoch: 2_000,
            closure_every: 1,
            horizon: 32,
            credit_cap: 32,
            max_states: 2_000,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::All,
            expand_occupancy: 100,
            admit_after: 1,
            compact_at: 60,
            keep_frac: 30,
            decay_shift: 1,
            no_compact: false,
            root_seed_pct: 75,
            csv: None,
            ledger: None,
            ledger_save_every: 0,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let stop = AtomicBool::new(false);
        let outcome = run_search(&args, &stop).unwrap();
        assert_eq!(
            outcome.epochs_run, 6,
            "compaction keeps the run alive to the epoch limit"
        );
        assert!(
            outcome.generations >= 1,
            "the tiny arena must have forced a compaction"
        );
        assert!(outcome.report.witnesses.len() >= 7);
    }

    #[test]
    fn livelock_guard_exits_gracefully() {
        // compact_at 1 with 7 roots in a 100-state arena trips the trigger
        // immediately; keep_frac 100 keeps everything, so occupancy cannot
        // drop below the trigger -> the guard must stop the run cleanly.
        let args = Args {
            threads: 1,
            epochs: Some(10),
            rollouts_per_epoch: 100,
            expansions_per_epoch: 100,
            closure_every: 1,
            horizon: 8,
            credit_cap: 8,
            max_states: 100,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::All,
            expand_occupancy: 100,
            admit_after: 1,
            compact_at: 1,
            keep_frac: 100,
            decay_shift: 0,
            no_compact: false,
            root_seed_pct: 75,
            csv: None,
            ledger: None,
            ledger_save_every: 0,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let stop = AtomicBool::new(false);
        let outcome = run_search(&args, &stop).unwrap();
        assert_eq!(
            outcome.generations, 1,
            "one compaction attempt, then the guard fires"
        );
        assert_eq!(
            outcome.epochs_run, 0,
            "guard fires at the first barrier, before epoch 0 completes"
        );
    }

    #[test]
    fn no_compact_restores_terminal_exhaustion_exit() {
        // Tiny arena + big carrier universe: the arena exhausts quickly.
        // With --no-compact the run must break out early (legacy behavior),
        // never compacting.
        let args = Args {
            threads: 2,
            epochs: Some(50),
            rollouts_per_epoch: 2_000,
            expansions_per_epoch: 2_000,
            closure_every: 1,
            horizon: 32,
            credit_cap: 32,
            max_states: 500,
            max_placements: None,
            max_branches: None,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            max_holes: 1,
            max_height: 4,
            max_roughness: 16,
            expansion_policy: ExpansionPolicyArg::All,
            expand_occupancy: 100,
            admit_after: 1,
            compact_at: 90,
            keep_frac: 50,
            decay_shift: 1,
            no_compact: true,
            root_seed_pct: 75,
            csv: None,
            ledger: None,
            ledger_save_every: 0,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let stop = AtomicBool::new(false);
        let outcome = run_search(&args, &stop).unwrap();
        assert_eq!(outcome.generations, 0, "no-compact must never compact");
        assert!(
            outcome.epochs_run < 50,
            "exhaustion must terminate the run early (got {})",
            outcome.epochs_run
        );
    }

    #[test]
    fn closure_profile_and_near_miss() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        // b: safe self-loop (the K). s1: one branch into K, one UNEXPANDED ->
        // near-miss. s2: both branches unexpanded -> 2 failures, NOT near-miss.
        let b = synth_state(&graph, 1, O_BAG, 1);
        let s1 = synth_state(&graph, 2, OI_BAG, 1);
        let s2 = synth_state(&graph, 3, OI_BAG, 1);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, s1, 0, TetrisPiece::O_PIECE, b);
        let report = extract_safe_set(&graph, &[b; 7]);
        assert_eq!(report.k_size, 1);
        assert_eq!(report.k0_size, 3, "b, s1, s2 are alive and FULLY");
        // Pass 1 removes s1 and s2; pass 2 removes nothing (fixpoint).
        assert_eq!(report.pass_removed, vec![2, 0]);
        assert_eq!(report.iterations, 2);
        assert_eq!(
            report.near_miss, 1,
            "s1 is one branch from closing; s2 is two"
        );
    }

    #[test]
    fn root_values_reflect_search_estimates() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // a: fully confident self-loop stats; z: unvisited (value 0).
        let a = synth_state(&graph, 1, O_BAG, 1);
        let z = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        wire(&graph, z, 0, TetrisPiece::O_PIECE, z);
        let pr = graph.placements.get(graph.states.get(a).placement_base);
        pr.visits.fetch_add(100, Ordering::Relaxed);
        let br = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br.visits.fetch_add(100, Ordering::Relaxed);
        br.reward_sum.fetch_add(800, Ordering::Relaxed);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // Roots: 4x a (value 1000pm at c_lcb=0), 3x z (unvisited branches -> 0).
        let (min, mean) = root_values_permille(&graph, &[a, z, a, z, a, z, a], &params);
        assert_eq!(min, 0);
        assert_eq!(mean, (4 * 1000) / 7);
    }

    // --- Task 2a: proof crystallization ---

    #[test]
    fn death_propagates_backward_through_forced_lines() {
        // Chain: a → b → dead_end (dead_end has no placements).
        // O_BAG: single piece, so every placement has exactly one branch slot
        // and a forced adversary choice; all states must become PROOF_DEAD.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let dead_end = synth_state(&graph, 3, O_BAG, 0); // placement_len = 0
        let b = synth_state(&graph, 2, O_BAG, 1);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, dead_end);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, b);

        let report = extract_safe_set(&graph, &[a; 7]);
        assert_eq!(report.dead_total, 3, "all three states must be proven dead");
        assert_eq!(
            graph.states.get(dead_end).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "dead_end must be PROOF_DEAD"
        );
        assert_eq!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "b must be PROOF_DEAD"
        );
        assert_eq!(
            graph.states.get(a).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "a must be PROOF_DEAD"
        );
        assert_eq!(report.k_size, 0, "K must be empty (all dead)");
        assert_eq!(
            report.k0_size, 0,
            "K0 must be empty (dead states excluded from K0)"
        );
    }

    #[test]
    fn multi_placement_state_escapes_death() {
        // State s has 2 placements: slot 0 → dead_end, slot 1 → s (self-loop).
        // Slot 1 is unrefuted (s is not dead), so s escapes death and enters K.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let dead_end = synth_state(&graph, 2, O_BAG, 0); // no placements
        let s = synth_state(&graph, 1, O_BAG, 2); // two placements
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead_end); // slot 0 refuted
        wire(&graph, s, 1, TetrisPiece::O_PIECE, s); // slot 1: self-loop, unrefuted

        let report = extract_safe_set(&graph, &[s; 7]);
        assert_eq!(
            graph.states.get(s).proof.load(Ordering::Acquire),
            PROOF_SAFE,
            "s is a K member (slot-1 self-loop witness) and must be marked SAFE"
        );
        // s has a self-loop in slot 1 → it must be in K (witness slot 1).
        assert!(
            report.witnesses.get(s as usize).copied().unwrap_or(u8::MAX) != u8::MAX,
            "s must be in K (self-loop via slot 1)"
        );
    }

    #[test]
    fn unknown_children_never_support_death() {
        // State with one O_BAG placement whose branch is left UNEXPANDED.
        // The branch is unknown, so death is NOT supported → proof stays UNKNOWN.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 1);
        // Do NOT wire the branch child → it stays UNEXPANDED (u32::MAX).

        let report = extract_safe_set(&graph, &[s; 7]);
        assert_eq!(
            report.dead_total, 0,
            "no states should be proven dead (unknown child)"
        );
        assert_eq!(
            graph.states.get(s).proof.load(Ordering::Acquire),
            PROOF_UNKNOWN,
            "s proof must remain UNKNOWN (unknown children never support death)"
        );
        assert_eq!(
            report.k_size, 0,
            "unknown branch fails closure — s must not enter K"
        );
    }

    #[test]
    fn safe_and_dead_marks_persist_and_stay_disjoint() {
        // b: safe self-loop; x → dead_end (dead chain).
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let b = synth_state(&graph, 1, O_BAG, 1);
        let dead_end = synth_state(&graph, 3, O_BAG, 0);
        let x = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, b, 0, TetrisPiece::O_PIECE, b);
        wire(&graph, x, 0, TetrisPiece::O_PIECE, dead_end);

        let report1 = extract_safe_set(&graph, &[b; 7]);
        assert_eq!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_SAFE,
            "first extract: b must be PROOF_SAFE"
        );
        assert_eq!(
            graph.states.get(x).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "first extract: x must be PROOF_DEAD"
        );
        assert_eq!(
            graph.states.get(dead_end).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "first extract: dead_end must be PROOF_DEAD"
        );

        // Second extract: marks must be idempotent.
        let report2 = extract_safe_set(&graph, &[b; 7]);
        assert_eq!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_SAFE,
            "second extract: b still PROOF_SAFE"
        );
        assert_eq!(
            graph.states.get(x).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "second extract: x still PROOF_DEAD"
        );
        assert_eq!(
            report2.dead_total, report1.dead_total,
            "dead_total must be stable across extracts"
        );
        assert_eq!(
            report2.k_size, report1.k_size,
            "k_size must be stable across extracts"
        );
        // SAFE and DEAD disjointness: b is SAFE, x/dead_end are DEAD.
        assert_ne!(
            graph.states.get(b).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "SAFE state must not be DEAD"
        );
        assert_ne!(
            graph.states.get(x).proof.load(Ordering::Acquire),
            PROOF_SAFE,
            "DEAD state must not be SAFE"
        );
    }

    // --- Task 2b: crystallization consumers ---

    #[test]
    fn rollout_certified_at_safe_state() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1);
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        graph
            .states
            .get(a)
            .proof
            .store(PROOF_SAFE, Ordering::Release);
        let params = SearchParams {
            horizon: 32,
            credit_cap: 32,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        let outcome = rollout(&graph, &params, a, &mut scratch);
        assert!(matches!(outcome, RolloutOutcome::Certified));
        let s = graph.states.get(a);
        assert_eq!(s.visits.load(Ordering::Relaxed), 1);
        assert_eq!(
            s.reward_sum.load(Ordering::Relaxed),
            32,
            "full-horizon certified credit"
        );
    }

    #[test]
    fn rollout_dies_instantly_at_proven_dead() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let a = synth_state(&graph, 1, O_BAG, 1); // has placements — proof, not len==0, must fire
        wire(&graph, a, 0, TetrisPiece::O_PIECE, a);
        graph
            .states
            .get(a)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let params = SearchParams {
            horizon: 32,
            credit_cap: 32,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        let outcome = rollout(&graph, &params, a, &mut scratch);
        assert!(matches!(outcome, RolloutOutcome::Death));
        assert!(scratch.path.is_empty(), "no placement was played");
        let s = graph.states.get(a);
        assert_eq!(s.visits.load(Ordering::Relaxed), 1);
        assert_eq!(s.reward_sum.load(Ordering::Relaxed), 0, "died at depth 0");
    }

    #[test]
    fn refuted_placement_scores_zero_despite_stats() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 1);
        let dead_child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead_child);
        graph
            .states
            .get(dead_child)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        pr.visits.fetch_add(1_000, Ordering::Relaxed);
        let br = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br.visits.fetch_add(1_000, Ordering::Relaxed);
        br.saturating_add_reward(8_000); // juicy stats: mean 1.0 at horizon 8
        assert_eq!(
            placement_value(&graph, st, pr, 0, 8),
            None,
            "proof overrides stats: refuted, never selectable"
        );
    }

    #[test]
    fn all_safe_placement_scores_one_without_visits() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let s = synth_state(&graph, 1, OI_BAG, 1);
        let safe1 = synth_state(&graph, 2, O_BAG, 1);
        let safe2 = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, safe1);
        wire(&graph, s, 0, TetrisPiece::I_PIECE, safe2);
        graph
            .states
            .get(safe1)
            .proof
            .store(PROOF_SAFE, Ordering::Release);
        graph
            .states
            .get(safe2)
            .proof
            .store(PROOF_SAFE, Ordering::Release);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        assert_eq!(
            placement_value(&graph, st, pr, SCORE_ONE, 8),
            Some(SCORE_ONE)
        );
    }

    #[test]
    fn adversary_plays_proven_dead_piece_on_sight() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let s = synth_state(&graph, 1, OI_BAG, 1);
        let safe_child = synth_state(&graph, 2, O_BAG, 1);
        let dead_child = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, safe_child);
        wire(&graph, s, 0, TetrisPiece::I_PIECE, dead_child);
        graph
            .states
            .get(safe_child)
            .proof
            .store(PROOF_SAFE, Ordering::Release);
        graph
            .states
            .get(dead_child)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        assert_eq!(
            choose_piece(&graph, st, pr, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            TetrisPiece::I_PIECE,
            "proof-perfect kill beats all other piece-selection passes"
        );
    }

    #[test]
    fn compaction_carries_proofs_and_protects_safe() {
        // sfe: safe self-loop, ZERO visits (would be evicted by visits budget alone).
        // junk: high visits, no proof.
        // dead_st: DEAD, zero visits.
        // Compact with keep_frac_pct=0 and empty_closure_report as protect
        // (so K-witness protection does NOT cover sfe).
        // Roots aliased to junk so root protection covers junk.
        let graph = test_graph(200, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let sfe = synth_state(&graph, 1, O_BAG, 1);
        let junk = synth_state(&graph, 2, O_BAG, 1);
        let dead_st = synth_state(&graph, 3, O_BAG, 0); // no placements → will be DEAD

        // Wire sfe as a safe self-loop.
        wire(&graph, sfe, 0, TetrisPiece::O_PIECE, sfe);
        // Wire junk to itself too (has a valid cycle but won't be protected by K via empty report).
        wire(&graph, junk, 0, TetrisPiece::O_PIECE, junk);

        // Give junk high visits so it wins the budget competition.
        graph.states.get(junk).visits.store(9999, Ordering::Relaxed);

        // First extract: marks sfe SAFE, dead_st DEAD.
        let _ = extract_safe_set(&graph, &[sfe; 7]);
        assert_eq!(
            graph.states.get(sfe).proof.load(Ordering::Acquire),
            PROOF_SAFE,
            "sfe must be PROOF_SAFE after first extract"
        );
        assert_eq!(
            graph.states.get(dead_st).proof.load(Ordering::Acquire),
            PROOF_DEAD,
            "dead_st must be PROOF_DEAD after first extract"
        );

        // Compact: pass empty_closure_report so K-witnesses don't protect sfe.
        // Roots = junk (so junk is root-protected). keep_frac_pct = 0 (no budget survivors).
        let protect = empty_closure_report(graph.states.len() as usize);
        let roots = [junk; 7];
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new_graph, _stats) = compact(&graph, &protect, &roots, &cfg).unwrap();

        // sfe must survive (proof-based protection).
        assert!(
            new_graph.ids.get(&graph.states.get(sfe).key).is_some(),
            "sfe must survive compaction via proof protection"
        );
        // Proof must be carried across compaction.
        let new_sfe_id = *new_graph.ids.get(&graph.states.get(sfe).key).unwrap();
        assert_eq!(
            new_graph
                .states
                .get(new_sfe_id)
                .proof
                .load(Ordering::Acquire),
            PROOF_SAFE,
            "sfe proof must be PROOF_SAFE in the new graph"
        );

        // junk survives (root protection).
        assert!(
            new_graph.ids.get(&graph.states.get(junk).key).is_some(),
            "junk must survive compaction via root protection"
        );

        // dead_st with 0 visits must NOT survive (no proof protection for DEAD, no root, no K).
        assert!(
            new_graph.ids.get(&graph.states.get(dead_st).key).is_none(),
            "dead_st must be evicted (DEAD states get no proof protection)"
        );
    }

    #[test]
    fn expansion_stamps_ledgered_children_without_intern() {
        // Precompute one canonical child key of root 0, put it in the ledger,
        // THEN expand. The branch must come out DEAD_LEDGER with no state
        // interned for it, the owning placement must be refuted, and a
        // rollout from the root must never abort on it.
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let key = root_keys()[0];
        let piece = key.bag().iter_pieces().next().unwrap();
        let first_placement = TetrisPiecePlacement::all_from_piece(key.piece())
            .iter()
            .copied()
            .find(|&p| apply_move(&key.board, p, &graph.carrier).is_some())
            .unwrap();
        let child_board = apply_move(&key.board, first_placement, &graph.carrier).unwrap();
        let child_key = StateKey::new(child_board, child_bag(key.bag(), piece), piece);
        graph.ledger_insert(child_key);

        let id = graph.intern(key).unwrap().id;
        graph.ensure_fully_expanded(id).unwrap();
        assert!(
            graph.ids.get(&child_key).is_none(),
            "ledgered child never materialized"
        );
        let state = graph.states.get(id);
        let pr = graph.placements.get(state.placement_base);
        assert_eq!(
            pr.placement, first_placement,
            "slot 0 is the first survivor"
        );
        let br = graph.branches.get(pr.branch_base + piece.index() as u32);
        assert_eq!(br.child.load(Ordering::Acquire), DEAD_LEDGER);
        assert!(placement_refuted(&graph, state, pr), "stamp refutes");
        // resolve_child returns the stamp without claiming/materializing.
        assert_eq!(
            graph.resolve_child(state, pr, br, piece).unwrap(),
            DEAD_LEDGER
        );
        assert!(graph.ids.get(&child_key).is_none());
        // Rollouts still work from the root (selection avoids the refuted slot).
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        let mut scratch = WorkerScratch::new(params.horizon);
        let outcome = rollout(&graph, &params, id, &mut scratch);
        assert!(!matches!(outcome, RolloutOutcome::Aborted));
    }

    #[test]
    fn compaction_stamps_evicted_dead_children() {
        // parent (root-protected) --O--> dead_child (PROOF_DEAD, evicted at
        // keep 0). The copy must stamp DEAD_LEDGER instead of UNEXPANDED so
        // the refutation survives at zero arena cost, the death lfp proves
        // the parent dead on the NEW graph, and resolving the edge lazily
        // re-materializes the child as PROOF_DEAD via the ledger.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let parent = synth_state(&graph, 1, O_BAG, 1);
        let dead_child = synth_state(&graph, 2, O_BAG, 0);
        wire(&graph, parent, 0, TetrisPiece::O_PIECE, dead_child);
        graph
            .states
            .get(dead_child)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        graph.ledger_insert(graph.states.get(dead_child).key);
        let roots = [parent; 7];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert_eq!(stats.kept, 1, "dead child evicted");
        let np = *new.ids.get(&graph.states.get(parent).key).unwrap();
        let ns = new.states.get(np);
        let npr = new.placements.get(ns.placement_base);
        let nbr = new
            .branches
            .get(npr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        assert_eq!(
            nbr.child.load(Ordering::Acquire),
            DEAD_LEDGER,
            "evicted dead child leaves a stamp, not a hole"
        );
        assert!(placement_refuted(&new, ns, npr), "stamp refutes");
        assert_eq!(
            placement_value(&new, ns, npr, SCORE_ONE, 8),
            None,
            "stamped placement never selectable"
        );
        // Death flows through the stamp: the parent is proven dead on the
        // new graph even though the child record is gone.
        let dead = propagate_death(&new, &build_reverse_edges(&new));
        assert!(dead >= 1);
        assert_eq!(ns.proof.load(Ordering::Acquire), PROOF_DEAD);
    }

    #[test]
    fn engine_is_mirror_equivariant() {
        // The soundness of mirror-canonical ledger keys rests on: for every
        // placement p of piece π on board B, there EXISTS a placement p' of
        // mirror(π) on mirror(B) with result mirror(apply(B, p)) and the
        // same loss flag. With bag-transition equivariance (a bit swap,
        // trivially true), dead(k) ⟺ dead(mirror(k)) follows by induction.
        let mirror_board = |b: &TetrisBoard| {
            let mut limbs = b.as_limbs();
            limbs.reverse();
            TetrisBoard::from_limbs(limbs)
        };
        // Boards: empty, plus a few reachable stacks built by placements.
        let mut boards = vec![TetrisBoard::EMPTY_BOARD];
        let mut b = TetrisBoard::EMPTY_BOARD;
        for (i, &piece) in [
            TetrisPiece::S_PIECE,
            TetrisPiece::L_PIECE,
            TetrisPiece::T_PIECE,
            TetrisPiece::J_PIECE,
            TetrisPiece::Z_PIECE,
        ]
        .iter()
        .enumerate()
        {
            let placements = TetrisPiecePlacement::all_from_piece(piece);
            let _ = b.apply_piece_placement(placements[(i * 3) % placements.len()]);
            boards.push(b);
        }
        for board in &boards {
            let mboard = mirror_board(board);
            for piece in TetrisPiece::all() {
                let mpiece = TetrisPiece::from_index(mirror_piece_idx(piece.index()));
                for &pl in TetrisPiecePlacement::all_from_piece(piece) {
                    let mut a = *board;
                    let res = a.apply_piece_placement(pl);
                    let want = mirror_board(&a);
                    let found = TetrisPiecePlacement::all_from_piece(mpiece)
                        .iter()
                        .any(|&mpl| {
                            let mut c = mboard;
                            let mres = c.apply_piece_placement(mpl);
                            bool::from(mres.is_lost) == bool::from(res.is_lost) && c == want
                        });
                    assert!(
                        found,
                        "no mirror placement reproduces mirror(apply(B, {pl:?}))"
                    );
                }
            }
        }
    }

    #[test]
    fn ledger_covers_both_mirror_orientations() {
        // Inserting a key makes BOTH it and its mirror hit, one entry total;
        // expansion stamps a branch whose child key is only ledgered in
        // mirrored form.
        let graph = test_graph(50_000, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // An asymmetric key: S piece on an asymmetric board.
        let mut board = TetrisBoard::EMPTY_BOARD;
        board.set_bit(0, 0);
        let mut bag = TetrisPieceBagState::from(TetrisPieceBagState::FULL_MASK);
        bag.remove(TetrisPiece::S_PIECE);
        let key = StateKey::new(board, bag, TetrisPiece::S_PIECE);
        let mkey = mirror_key(&key);
        assert_ne!(key, mkey, "fixture must be asymmetric");
        assert_eq!(mirror_key(&mkey), key, "mirror is an involution");
        graph.ledger_insert(key);
        assert_eq!(graph.dead_keys.len(), 1);
        assert!(graph.ledger_contains(&key));
        assert!(graph.ledger_contains(&mkey), "mirror orientation covered");
        graph.ledger_insert(mkey);
        assert_eq!(graph.dead_keys.len(), 1, "mirror pair deduplicates");
        // Interning the MIRROR key materializes PROOF_DEAD from the ledger.
        let id = graph.intern(mkey).unwrap().id;
        assert_eq!(
            graph.states.get(id).proof.load(Ordering::Acquire),
            PROOF_DEAD
        );
        assert_eq!(graph.states.get(id).placement_len, 0);
    }

    #[test]
    fn ledger_file_roundtrip_and_carrier_guard() {
        let carrier = Carrier {
            max_holes: 1,
            max_height: 5,
            max_roughness: 16,
        };
        let graph = SearchGraph::new(
            100,
            100 * 34,
            100 * 34 * 7,
            carrier,
            ExpansionPolicy::AllDiscovered,
        );
        // Two canonical dead keys.
        let k1 = root_keys()[0];
        let k2 = root_keys()[3];
        graph.ledger_insert(k1);
        graph.ledger_insert(k2);
        let path = std::env::temp_dir().join(format!(
            "tetris_puct_ledger_test_{}.bin",
            std::process::id()
        ));
        let saved = save_ledger(&graph, &path, &carrier).unwrap();
        assert_eq!(saved, 2);
        // Round-trip into a fresh graph.
        let fresh = SearchGraph::new(
            100,
            100 * 34,
            100 * 34 * 7,
            carrier,
            ExpansionPolicy::AllDiscovered,
        );
        let loaded = load_ledger(&fresh, &path, &carrier).unwrap();
        assert_eq!(loaded, 2);
        assert!(fresh.ledger_contains(&k1) && fresh.ledger_contains(&k2));
        // Interning a ledgered key materializes the proof.
        let id = fresh.intern(k1).unwrap().id;
        assert_eq!(
            fresh.states.get(id).proof.load(Ordering::Acquire),
            PROOF_DEAD
        );
        // Carrier mismatch refused.
        let other = Carrier {
            max_holes: 0,
            max_height: 5,
            max_roughness: 16,
        };
        let wrong = SearchGraph::new(
            100,
            100 * 34,
            100 * 34 * 7,
            other,
            ExpansionPolicy::AllDiscovered,
        );
        assert!(load_ledger(&wrong, &path, &other).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dead_ledger_survives_eviction_and_reintern() {
        // x -> dead_end proves x dead; compact evicts BOTH (nothing protects
        // them); re-interning x's key must come back PROOF_DEAD with zero
        // placements, straight from the run-lifetime ledger.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let anchor = synth_state(&graph, 1, O_BAG, 1); // root-protected survivor
        wire(&graph, anchor, 0, TetrisPiece::O_PIECE, anchor);
        // A REAL key for x (root 0's key) so re-interning is meaningful.
        let x_key = root_keys()[0];
        let x = graph.intern(x_key).unwrap().id;
        // Force x dead: overwrite its record semantics via a dead child on
        // every placement is overkill — instead prove death by the lfp on a
        // zero-placement synthetic... x is a real state with placements, so
        // mark it dead the honest way: put its key in the ledger via
        // propagate_death on a graph where x has no placements is not
        // possible. Simplest honest path: insert directly, as propagate_death
        // does on transition.
        graph
            .states
            .get(x)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        graph.ledger_insert(x_key);
        // Compact with keep 0: only the root-aliased anchor survives.
        let roots = [anchor; 7];
        let protect = empty_closure_report(graph.states.len() as usize);
        let cfg = CompactConfig {
            keep_frac_pct: 0,
            decay_shift: 0,
        };
        let (new, stats) = compact(&graph, &protect, &roots, &cfg).unwrap();
        assert_eq!(stats.kept, 1, "x and its record are gone");
        assert!(new.ids.get(&x_key).is_none());
        // Re-intern: the ledger resurrects the proof, not the subtree.
        let again = new.intern(x_key).unwrap();
        assert!(again.newly_created);
        let st = new.states.get(again.id);
        assert_eq!(st.proof.load(Ordering::Acquire), PROOF_DEAD);
        assert_eq!(st.placement_len, 0, "no placement/branch arena spend");
        assert_eq!(
            st.expansion_status.load(Ordering::Acquire),
            STATUS_FULLY_EXPANDED
        );
        // And the death lfp seeds from it: propagate_death counts it.
        let dead = propagate_death(&new, &build_reverse_edges(&new));
        assert!(dead >= 1);
    }

    #[test]
    fn refuted_placement_is_never_selected() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        // Slot 0: UNVISITED and refuted (would win the old unvisited-priority);
        // slot 1: visited, unrefuted, modest stats. Selection must skip slot 0.
        let s = synth_state(&graph, 1, O_BAG, 2);
        let dead_child = synth_state(&graph, 2, O_BAG, 1);
        let live_child = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead_child);
        wire(&graph, s, 1, TetrisPiece::O_PIECE, live_child);
        graph
            .states
            .get(dead_child)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let st = graph.states.get(s);
        let pr1 = graph.placements.get(st.placement_base + 1);
        pr1.visits.fetch_add(10, Ordering::Relaxed);
        let br1 = graph
            .branches
            .get(pr1.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br1.visits.fetch_add(10, Ordering::Relaxed);
        br1.saturating_add_reward(40);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        assert_eq!(choose_search_placement(&graph, st, &params), Some(1));
        assert_eq!(choose_best_placement(&graph, st, &params), Some(1));
    }

    #[test]
    fn all_refuted_state_dies_mid_epoch() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 1);
        let dead_child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead_child);
        graph
            .states
            .get(dead_child)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let st = graph.states.get(s);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: SCORE_ONE,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // Selection has nothing legal to offer...
        assert_eq!(choose_search_placement(&graph, st, &params), None);
        assert_eq!(choose_best_placement(&graph, st, &params), None);
        // ...and a rollout entering s dies immediately (depth 0), without the
        // state itself being PROOF_DEAD yet (that happens at the next barrier).
        let mut scratch = WorkerScratch::new(params.horizon);
        let outcome = rollout(&graph, &params, s, &mut scratch);
        assert!(matches!(outcome, RolloutOutcome::Death));
        assert!(scratch.path.is_empty());
        assert_eq!(st.visits.load(Ordering::Relaxed), 1);
        assert_eq!(st.reward_sum.load(Ordering::Relaxed), 0);
        assert_eq!(st.proof.load(Ordering::Acquire), PROOF_UNKNOWN);
    }

    #[test]
    fn fpu_strong_sibling_beats_unvisited() {
        // Old behavior returned the unvisited slot unconditionally; with FPU a
        // proven-strong sibling wins (c_uct = 0 isolates the value comparison).
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, child);
        wire(&graph, s, 1, TetrisPiece::O_PIECE, child);
        let st = graph.states.get(s);
        // Parent: mean 1.0 (visits 100, reward 100*8 at horizon 8).
        st.visits.fetch_add(100, Ordering::Relaxed);
        st.reward_sum.fetch_add(800, Ordering::Relaxed);
        // Slot 0: visited and perfect (value 1.0 at c_lcb = 0). Slot 1: unvisited.
        let pr0 = graph.placements.get(st.placement_base);
        pr0.visits.fetch_add(100, Ordering::Relaxed);
        let br0 = graph
            .branches
            .get(pr0.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br0.visits.fetch_add(100, Ordering::Relaxed);
        br0.saturating_add_reward(800);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: 0,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // FPU: slot 1 scores parent_mean/2 = 0.5; slot 0 scores 1.0 -> slot 0 wins.
        assert_eq!(choose_search_placement(&graph, st, &params), Some(0));
    }

    #[test]
    fn fpu_unvisited_wins_in_weak_region() {
        // When the visited sibling is worthless and the parent has visits, the
        // first-visit bonus pulls the unvisited placement in.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, child);
        wire(&graph, s, 1, TetrisPiece::O_PIECE, child);
        let st = graph.states.get(s);
        st.visits.fetch_add(100, Ordering::Relaxed); // parent mean 0 (no reward)
        let pr0 = graph.placements.get(st.placement_base);
        pr0.visits.fetch_add(100, Ordering::Relaxed); // slot 0 visited, mean 0
        let br0 = graph
            .branches
            .get(pr0.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br0.visits.fetch_add(100, Ordering::Relaxed); // 0 reward
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // Slot 1 (unvisited): 0 + bonus(100, 1) > slot 0: 0 + bonus(100, 100).
        assert_eq!(choose_search_placement(&graph, st, &params), Some(1));
    }

    #[test]
    fn balanced_policy_picks_least_visited_piece() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let s = synth_state(&graph, 1, OI_BAG, 1);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, child);
        wire(&graph, s, 0, TetrisPiece::I_PIECE, child);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        // O branch heavily visited with TERRIBLE value; I branch lightly
        // visited with PERFECT value. Adversarial picks O (min LCB);
        // balanced picks I (least visited) — opposite choices, one fixture.
        let bro = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        bro.visits.fetch_add(100, Ordering::Relaxed); // reward 0 -> LCB 0
        let bri = graph
            .branches
            .get(pr.branch_base + TetrisPiece::I_PIECE.index() as u32);
        bri.visits.fetch_add(10, Ordering::Relaxed);
        bri.saturating_add_reward(80); // mean 1.0 at horizon 8
        pr.visits.fetch_add(110, Ordering::Relaxed);
        assert_eq!(
            choose_piece(&graph, st, pr, 0, 8, PiecePolicyArg::Adversarial),
            TetrisPiece::O_PIECE
        );
        assert_eq!(
            choose_piece(&graph, st, pr, 0, 8, PiecePolicyArg::Balanced),
            TetrisPiece::I_PIECE
        );
    }

    #[test]
    fn exhausted_selection_skips_unvisited_placements() {
        // Slot 0: visited with an expanded branch. Slot 1: unvisited (FPU bait).
        // Pre-exhaustion FPU pulls the unvisited slot in a weak region;
        // post-exhaustion it is unsampleable and slot 0 must win.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, child);
        let st = graph.states.get(s);
        st.visits.fetch_add(100, Ordering::Relaxed); // parent visited, mean 0
        let pr0 = graph.placements.get(st.placement_base);
        pr0.visits.fetch_add(100, Ordering::Relaxed);
        let br0 = graph
            .branches
            .get(pr0.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br0.visits.fetch_add(100, Ordering::Relaxed);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        // Weak region: unvisited slot 1 wins pre-exhaustion (first-visit bonus).
        assert_eq!(choose_search_placement(&graph, st, &params), Some(1));
        graph.exhausted.store(true, Ordering::SeqCst);
        assert_eq!(
            choose_search_placement(&graph, st, &params),
            Some(0),
            "post-exhaustion: unvisited placements are unsampleable"
        );
        // Fully fresh state post-exhaustion: nothing sampleable -> None.
        let s2 = synth_state(&graph, 3, O_BAG, 1);
        assert_eq!(
            choose_search_placement(&graph, graph.states.get(s2), &params),
            None
        );
    }

    #[test]
    fn exhausted_selection_skips_childless_visited_placements() {
        // Slot 0: heavily visited but ALL branches UNEXPANDED (compaction
        // evicted its children). Slot 1: visited with an expanded child.
        // Post-exhaustion slot 0 is unsampleable regardless of stats.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        let s = synth_state(&graph, 1, O_BAG, 2);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 1, TetrisPiece::O_PIECE, child);
        let st = graph.states.get(s);
        st.visits.fetch_add(200, Ordering::Relaxed);
        let pr0 = graph.placements.get(st.placement_base);
        pr0.visits.fetch_add(100, Ordering::Relaxed);
        pr0.reward_sum.fetch_add(800, Ordering::Relaxed); // juicy but childless
        let pr1 = graph.placements.get(st.placement_base + 1);
        pr1.visits.fetch_add(100, Ordering::Relaxed);
        let br1 = graph
            .branches
            .get(pr1.branch_base + TetrisPiece::O_PIECE.index() as u32);
        br1.visits.fetch_add(100, Ordering::Relaxed);
        let params = SearchParams {
            horizon: 8,
            credit_cap: 8,
            c_uct: SCORE_ONE,
            c_lcb: 0,
            admit_after: 1,
            piece_policy: PiecePolicyArg::Adversarial,
        };
        graph.exhausted.store(true, Ordering::SeqCst);
        assert_eq!(
            choose_search_placement(&graph, st, &params),
            Some(1),
            "childless visited placement skipped post-exhaustion"
        );
    }

    #[test]
    fn exhausted_piece_selection_skips_unexpanded() {
        // Bag {O, I}: O branch expanded + visited (terrible LCB), I branch
        // UNEXPANDED. Pre-exhaustion both policies demand the unexpanded I
        // (coverage first). Post-exhaustion I is unsampleable — both policies
        // must steer through the expanded O instead of aborting on I.
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let s = synth_state(&graph, 1, OI_BAG, 1);
        let child = synth_state(&graph, 2, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, child);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        let bro = graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32);
        bro.visits.fetch_add(100, Ordering::Relaxed); // visited, reward 0
        pr.visits.fetch_add(100, Ordering::Relaxed);
        for policy in [PiecePolicyArg::Adversarial, PiecePolicyArg::Balanced] {
            assert_eq!(
                choose_piece(&graph, st, pr, SCORE_ONE, 8, policy),
                TetrisPiece::I_PIECE,
                "pre-exhaustion: unexpanded branch demanded first ({policy:?})"
            );
        }
        graph.exhausted.store(true, Ordering::SeqCst);
        for policy in [PiecePolicyArg::Adversarial, PiecePolicyArg::Balanced] {
            assert_eq!(
                choose_piece(&graph, st, pr, SCORE_ONE, 8, policy),
                TetrisPiece::O_PIECE,
                "post-exhaustion: expanded branch wins ({policy:?})"
            );
        }
        // All-unexpanded placement still falls through to an unexpanded piece
        // (the rollout aborts there — rare and pessimistically sound).
        let s2 = synth_state(&graph, 3, OI_BAG, 1);
        let st2 = graph.states.get(s2);
        let pr2 = graph.placements.get(st2.placement_base);
        assert_eq!(
            choose_piece(&graph, st2, pr2, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            TetrisPiece::O_PIECE,
            "no expanded branch: fall back to first unexpanded"
        );
    }

    #[test]
    fn balanced_policy_ignores_proof_kill() {
        let graph = test_graph(100, Carrier::UNBOUNDED, ExpansionPolicy::AllDiscovered);
        const OI_BAG: u8 = 0b0000_0011;
        let s = synth_state(&graph, 1, OI_BAG, 1);
        let live = synth_state(&graph, 2, O_BAG, 1);
        let dead = synth_state(&graph, 3, O_BAG, 1);
        wire(&graph, s, 0, TetrisPiece::O_PIECE, dead);
        wire(&graph, s, 0, TetrisPiece::I_PIECE, live);
        graph
            .states
            .get(dead)
            .proof
            .store(PROOF_DEAD, Ordering::Release);
        let st = graph.states.get(s);
        let pr = graph.placements.get(st.placement_base);
        // O (dead child) more visited than I: balanced picks I by visits,
        // NOT O by proof-kill; adversarial picks O (proof-perfect kill).
        graph
            .branches
            .get(pr.branch_base + TetrisPiece::O_PIECE.index() as u32)
            .visits
            .fetch_add(5, Ordering::Relaxed);
        assert_eq!(
            choose_piece(&graph, st, pr, SCORE_ONE, 8, PiecePolicyArg::Adversarial),
            TetrisPiece::O_PIECE
        );
        assert_eq!(
            choose_piece(&graph, st, pr, SCORE_ONE, 8, PiecePolicyArg::Balanced),
            TetrisPiece::I_PIECE
        );
    }
}
