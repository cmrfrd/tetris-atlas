# Tetris Atlas PUCT — Design

**Date:** 2026-07-13
**Binary:** `tetris_atlas_puct` — `crates/tetris-playground/src/bin/atlas/tetris_atlas_puct.rs`
**Status:** Approved (single-file revision)

## 1. Objective

A massively parallel graph-search binary that combines integer-only UCT-style (PUCT)
rollouts with an explicit expansion frontier to grow a canonical AND-OR state graph of
Tetris, then extracts a **candidate safe set** via greatest-fixpoint (gfp) pruning:

```
K = { s | ∃ placement a, ∀ next piece q ∈ bag(s): T(s,a,q) ∈ K }
```

The search *discovers* the atlas; the closure pass determines whether the discovered
graph contains a genuine closed safe set. Success criterion: all seven root states
(empty board, each first draw) survive closure with nonempty K.

## 2. Shape

One flat file, matching the sibling atlas bins (`tetris_atlas_inmemory.rs`, etc.).
No required features (no TUI): plain log lines + CSV. All dependencies already in
`tetris-playground`: `dashmap`, `rayon`, `crossbeam-queue`, `rustc-hash`, `clap`,
`ctrlc`. Internal organization as commented sections:

```
Config & CLI | Domain | Arenas | Records | Graph | Integer scoring
| Selection | Rollout | Closure | Main/epochs | Tests
```

Cargo.toml gets one `[[bin]]` entry, no `required-features`.

## 3. Domain model

### StateKey

```rust
struct StateKey {
    board: TetrisBoard,          // [u32; 10], Copy
    bag:   TetrisPieceBagState,  // u8 mask — pieces remaining after current draw
    piece: TetrisPiece,          // the visible current piece
}
```

42 bytes, `Copy + Hash + Eq`, no heap. **Canonical form** follows the repo's
`iter_next_states()` convention: refill happens at draw time, so `bag` is never
empty in a key.

**Invariant:** `bag.contains(piece) ⟹ bag == FULL_MASK` (immediately after a
refill the current piece legitimately appears in the fresh bag; otherwise the
current piece has been removed).

**Required piece mask** for the AND quantifier is `key.bag` itself — no separate
field. Successor draws are exactly `key.bag.iter_next_states()`.

### Roots

Seven: `(EMPTY_BOARD, FULL \ {p}, p)` for each piece `p`. Rollouts round-robin
across roots. "init ∈ K" means all seven roots ∈ K.

### StateId and sentinels

```rust
struct StateId(u32);
const UNEXPANDED: u32 = u32::MAX;
const EXPANDING:  u32 = u32::MAX - 1;
const MAX_VALID_STATE_ID: u32 = u32::MAX - 2;
```

No `LOST` sentinel is needed (see §4): branches always point at real states.

### Move application and carrier

```rust
fn apply_move(board: &TetrisBoard, p: TetrisPiecePlacement, carrier: &Carrier)
    -> Option<TetrisBoard>   // None = top-out OR out-of-carrier
```

`Carrier { max_holes, max_height, max_roughness }` from CLI flags, default
`u32::MAX` each (unbounded — the pure-plan configuration). Restricting the
player's placements to a carrier is sound: it only weakens the player, so any
safe set found is a safe set of the unrestricted game.

**Key structural fact:** `board' = apply(board, a)` is independent of the next
piece `q`. Loss / carrier-exit is decided per placement, *before* piece
branching. Therefore only surviving placements are stored, and every stored
branch child is a live state.

## 4. Graph memory

### Arenas

```rust
struct Arena<T> { slots: Box<[OnceLock<T>]>, next: AtomicU32 }
impl<T> Arena<T> { fn reserve(&self, count: u32) -> Result<Range<u32>, CapacityError>; }
enum CapacityError { StatesExhausted, PlacementsExhausted, BranchesExhausted }
```

Three arenas: states, placements, branches. Capacity exhaustion is a structured
error that ends the run gracefully (final closure pass still runs); IDs never wrap.
Safe Rust only in v1 (`OnceLock` slots; `MaybeUninit` is a Phase-7 option).

### Records

```rust
struct StateRecord {
    key: StateKey,
    visits: AtomicU64,
    horizon_survivals: AtomicU64,
    placement_base: u32,          // plain fields: set before publication (§5)
    placement_len: u8,            // 0 ⇒ terminally dead state
    expansion_status: AtomicU8,   // Queued=0 / Expanding=1 / FullyExpanded=2
}
struct PlacementRecord {
    placement: TetrisPiecePlacement,   // 2 bytes
    branch_base: u32,                  // 7 contiguous branch slots
    visits: AtomicU64,
    horizon_survivals: AtomicU64,
}
struct PieceBranchRecord {
    child: AtomicU32,                  // UNEXPANDED / EXPANDING / StateId
    horizon_survivals: AtomicU32,      // saturating (pessimistic on overflow)
    visits: AtomicU64,
}   // exactly 16 bytes, no padding
```

Branch slot for piece `q` = `branch_base + q.index()`. Pieces not in `key.bag`
leave their slot unused. 7 fixed slots per placement (per plan); popcount-rank
compression to `|bag|` slots is a Phase-7 optimization.

Branch records are **all-atomic**, so they need no `OnceLock` publication: the
branch arena is a plain pre-initialized `Box<[PieceBranchRecord]>` (every slot
starts `child = UNEXPANDED`) with an atomic bump pointer. States and placements
keep `OnceLock` slots (they contain non-atomic fields). The plan's per-branch
`max_survival_depth` becomes a single global diagnostic counter (it feeds no
scoring or closure decision).

**Memory budget (paid up-front at arena construction):** ≈ 80 B/state +
34 placements × 32 B + 34×7 branches × 16 B ≈ **5 KB/state**. Default
`--max-states 500_000` ≈ 2.5 GB at startup; placements/branches capacities
derived (`states × 34`, `× 7`) unless overridden. All capacities are CLI knobs.

## 5. Interning

`DashMap<StateKey, StateId, FxBuildHasher>` with 256 shards (inmemory-proven
configuration), used **only** for interning. Hot rollout traversal follows cached
`StateId`s through the arenas.

Protocol (incorporates the inmemory lost-decrement race lesson — publish payload
before id):

1. Lock-free `get` fast path → hit returns existing id.
2. `entry(key)`:
   - `Occupied` → return existing id.
   - `Vacant` → reserve state slot + placement range + branch ranges; run
     **structural enumeration** (≤34 `apply_move` calls, storing surviving
     placements, all branches `UNEXPANDED`); `OnceLock::set` the fully-built
     record; **then** insert id into the map, push frontier (policy-dependent),
     return `{ id, newly_created: true }`.

Reserving inside the vacant branch (unlike the plan's reserve-before-entry
pseudo-code) leaks no arena slots on races, at the cost of ~0.5 µs of work under
one of 256 shard locks.

Because structural enumeration completes before publication, `placement_base`
/ `placement_len` are plain fields and a visible `StateRecord` is always
structurally complete. `expansion_status` tracks only child interning.

## 6. Expansion

One function, two drivers:

```
ensure_fully_expanded(id):
    CAS expansion_status Queued→Expanding (loser: Backoff-spin until FullyExpanded)
    for each stored placement a:            # board' recomputed, cheap bitops
        for each (q, bag') in key.bag.iter_next_states():
            intern (board', bag', q); store child id with Release
    expansion_status = FullyExpanded (Release)
```

- **Frontier driver:** `ArrayQueue<StateId>` (capacity = max_states); each epoch
  drains ≤ `--expansions-per-epoch` states via `par_iter`.
- **Rollout driver:** a rollout reaching an unexpanded edge lazily expands
  only that single `(a, q)` edge via the per-edge CAS (`resolve_child`); it
  does NOT trigger full expansion. The frontier is the sole driver of
  `FullyExpanded` in both policies (deviation #13 — keeps rollout latency
  flat; under `CandidateOnly` a rollout-visited state becomes K-eligible only
  after a later epoch's expansion batch drains it).

Per-edge CAS `UNEXPANDED→EXPANDING→id` (AcqRel/Acquire publication) exists for
`CandidateOnly` mode, where edges intern lazily.

**Expansion policy** (CLI): `AllDiscovered` (default; every interned state is
queued — completeness) vs `CandidateOnly` (queue on first rollout visit — focus).

## 7. Integer scoring

No `f32`/`f64` anywhere in search code (enforced by a source-grep test; a float
UCT reference exists only inside `#[cfg(test)]`).

- Fixed point: `SCORE_SHIFT = 20`, `SCORE_ONE = 1 << 20`.
- `mean_fixed(successes, visits)` via `u128` intermediates.
- `ilog2` = `u64::ilog2`; base change absorbed into the exploration constant.
- Exact floor sqrt = std `u128::isqrt()` (nightly toolchain), property-tested
  `r² ≤ n < (r+1)²`.
- Exploration bonus: `c · isqrt((log2(N_parent+1) << 2·SHIFT) / N_action) >> SHIFT`;
  `N_action = 0` ⇒ `u64::MAX / 4`.
- **Branch LCB:** `L(s,a,q) = mean(branch) − c_lcb·bonus(N(s,a), N(s,a,q))`
  (saturating subtraction; parent count = placement visits).
- **Placement value:** `min over q ∈ key.bag of L(s,a,q)`; an unexpanded or
  unvisited branch contributes 0 (pessimistic — never optimistic about unknowns).
- **Search score:** `placement_value + c_uct·bonus(N(s), N(s,a))`.

`c_uct`, `c_lcb` are fixed-point CLI args (default `SCORE_ONE`,
i.e. 1.0). The entire search is **deterministic** — no RNG exists; only thread
scheduling perturbs visit statistics.

## 8. Selection

- `choose_search_placement`: (1) any unvisited placement first (by index);
  (2) else argmax of search score. Tie-break: score, then fewer visits, then
  lower placement index.
- `choose_best_placement` (solver/witness view): same exploitation term, **no**
  exploration bonus. Separate function per plan.
- `choose_piece` (adversarial AND side): among `q ∈ key.bag` —
  (1) unexpanded branch first; (2) unvisited branch; (3) else min `L(s,a,q)`.
  Deterministic tie-break by piece index.

## 9. Rollout and backup

Per-worker scratch via `for_each_init`: path `Vec<PathEntry>` with capacity =
horizon, pre-reserved `FxHashSet<StateId>` for cycle detection, local counters.
Zero allocation after warmup (asserted in tests).

```
rollout(root):
    current = root
    loop (≤ horizon):
        if placement_len == 0: backup(death); return
        if current already seen this rollout: backup(survival); count cycle; return
        ensure_fully_expanded(current) if needed
        a = choose_search_placement; q = choose_piece
        push PathEntry { state, placement_slot, piece }
        current = branch child id
    backup(survival)   # horizon reached
```

`PathEntry { state: StateId, placement_slot: u32, piece: u8 }`. Backup walks the
path once: `Relaxed` visit increment on state/placement/branch; on survival also
`horizon_survivals`; `max_survival_depth.fetch_max(depth)` on branches.

**Decision (approved):** a within-rollout cycle is backed up as a survival — a
loop under adversarial-leaning piece selection is the strongest empirical signal
available; cycles are also counted separately in metrics. One fixed horizon per
run; statistics are not comparable across horizons (documented on the CLI flag).

## 10. Epoch executor

One explicitly built Rayon pool (`--threads`). Phases are barriers, so the
closure pass always sees a quiescent graph — no snapshot machinery:

```
per epoch:
    1. parallel expansion batch (≤ expansions-per-epoch frontier states)
    2. parallel rollout batch (rollouts-per-epoch, round-robin roots)
    3. if epoch % closure-every == 0: gfp closure pass
    4. metrics snapshot → log line + CSV row
```

Stop conditions: `--epochs` reached, Ctrl-C (clean stop after current epoch),
or capacity exhaustion. A final closure pass always runs before exit.

## 11. gfp closure

- `K₀` = { alive (placement_len > 0) ∧ FullyExpanded }.
- Iterate `K_{n+1} = { s ∈ K_n | ∃a ∀q∈bag(s): child(a,q) is a valid id ∧ ∈ K_n }`
  with two ping-pong bitsets (`Vec<AtomicU64>`) over the dense id range,
  rayon-parallel scans, atomic removal counter; fixpoint when removals = 0.
- `UNEXPANDED`/`EXPANDING` branches fail the ∀ — unknown is never safe.
- Final pass records a witness placement per surviving state (first qualifying
  placement, i.e. deterministic).
- Report: |K|, iterations to fixpoint, per-root membership, wall time.

Certificate serialization / independent verifier: **deferred** (out of scope for
v1 per user decision). The closure result and witnesses drive logging only.

## 11b. Generational compaction (Phase 2 addition, approved 2026-07-14)

Arena exhaustion is no longer terminal by default. At an epoch barrier, when any
arena's occupancy reaches `--compact-at` percent of capacity (or after a
mid-epoch exhaustion), the executor runs a closure pass and then a
**mark-compact generation**:

- **Survivors** = roots ∪ current K (always protected), topped up with the
  highest-visit remaining states **up to a total of `--keep-frac`% of
  `max_states`** (the protected sets are kept even when they alone exceed the
  target). Everything else is evicted.
- **Copy** survivors into fresh arenas with a deterministic old→new id remap;
  branch edges to evicted states reset to `UNEXPANDED` (they heal lazily via
  rollout piece selection, which already prefers unexpanded branches);
  `FULLY_EXPANDED` labels are kept (closure checks branches, never labels).
  All visit/survival counters are aged by `--decay-shift` (right-shift; default
  halving) so "density" tracks current interest, not seniority. Cumulative
  `Counters` transfer unchanged.
- **Rebuild** the interner and frontier; re-resolve roots; drop stale reports.
- **Livelock guard:** if occupancy is still at/above the trigger after
  compaction, exit gracefully with an explanatory report.

Soundness: the post-compaction graph is an information-subgraph and closure
treats unknown as unsafe, so eviction can never fabricate safety — any policy
bug only shrinks K. Protecting roots ∪ K preserves K exactly (K is closed under
its witness placements, so the copied K remains closed; gfp ⊇ copied K ⇒ |K| is
monotone non-decreasing across generations). The single safety-critical
mechanism is remap faithfulness (a rewritten branch id points at the same
`StateKey`), enforced by a debug-mode validator that recomputes every child key
via `apply_move` + `child_bag`.

Candidate-mode admission gains `--admit-after N` (admit a state to the frontier
on its Nth rollout visit; roots exempt) so first-touch wandering does not flood
the frontier before UCT concentrates. Rollouts hitting an unresolvable edge
(capacity) back up a pessimistic death instead of silently aborting, and the
intern fast-fail applies only to the miss path (existing keys always resolve).

## 11c. Graded depth reward + proof crystallization (Phase 4, approved 2026-07-14)

**Graded reward.** Binary horizon-survival credit starves the search (observed: 2.7B
rollouts with zero horizon hits ⇒ all means 0 ⇒ unguided play). Backup now credits
every path entry with the rollout's depth: death/truncation at depth d earns d,
horizon and cycle earn the full horizon. The stat fields become `reward_sum`
accumulators; scoring normalizes as `mean = reward_sum / (visits × horizon)` (fixed
point), so every rollout carries gradient and deeper play is always preferred.
Full-path credit is unbiased for within-state placement comparisons (the prefix depth
is a common additive term). Statistics remain comparable only within one horizon.
Closure/proofs never read statistics — unaffected.

**Proof crystallization.** A `proof` byte per state (UNKNOWN/DEAD/SAFE) records
game-truths that never retract and survive compaction:
- DEAD via backward death propagation (lfp at the closure barrier): a placement is
  refuted iff some bag piece leads to a proven-dead child; a state is dead iff
  placement_len == 0 or every placement is refuted. Unknown children never support
  death (conservative).
- SAFE: every K member is gfp-certified and marked permanently; SAFE ⊆ K stays
  invariant (K is monotone across compaction), and select_survivors protects SAFE
  independently of the current report.
Consumers: rollouts short-circuit at crystals (DEAD → death backup at the truncation
depth; SAFE → full-horizon reward, counted as a distinct `Certified` outcome);
scoring gets hard bounds (any-DEAD-child placement scores 0 and is never selected;
all-SAFE placement scores 1.0; the adversary plays a proven-dead piece on sight);
K0 excludes proven-dead states. This creates the bootstrapping flywheel: death proofs
prune losing lines without needing survival signal; once K seeds, certified credit
propagates to everything that can reach it.

## 12. CLI

```
tetris_atlas_puct
  --threads N                 (default: num_cpus)
  --epochs N                  (default: run until Ctrl-C)
  --rollouts-per-epoch N      (default: 100_000)
  --expansions-per-epoch N    (default: 10_000)
  --closure-every N           (default: 10)
  --horizon N                 (default: 128)
  --max-states N              (default: 500_000; placement/branch caps derived)
  --c-uct FIXED / --c-lcb FIXED   (default: 1 << 20)
  --max-holes / --max-height / --max-roughness   (default: unbounded)
  --expansion-policy all|candidate   (default: all)
  --csv PATH                  (optional metrics CSV)
```

## 13. Metrics

`Counters` struct of atomics: states interned, arena occupancy per arena,
frontier length, rollouts (total / deaths / horizon-survivals / cycles),
expansions, epochs. Per-epoch log line + optional CSV row: epoch, states,
frontier, rollouts/s, expansions/s, |K|, roots-in-K, elapsed. Runs report seed-free
determinism note, commit hash, and parameters (repo reproducibility policy).

## 14. Testing

`#[cfg(test)]` mod at the bottom; runs via
`cargo test -p tetris-playground --bin tetris_atlas_puct`.

- **Domain:** key identity on all three axes; refill canonical form;
  `bag.contains(piece) ⟹ FULL` invariant; seven roots well-formed.
- **Arena:** contiguous non-overlapping reservations; exact boundary; exhaustion
  error; sentinels never allocated.
- **Interner:** many rayon workers × one key ⇒ one id and arena count 1;
  randomized mixed-key stress; equal keys share statistics.
- **Expansion:** same (s,a,q) ⇒ same child; concurrent edge expansion publishes
  one child; FullyExpanded only after all required edges; no stuck EXPANDING.
- **Integer math:** isqrt property incl. 0/1/`u128::MAX`/random; mean/bonus
  monotonicity properties; float-reference comparison (test-only floats).
- **Selection:** unvisited-first, adversarial min, both selectors' exploration
  split, deterministic ties.
- **Rollout:** death on placement_len 0; survival at exact horizon; each path
  node updated exactly once; cycle terminates; scratch capacity does not grow.
- **Closure (synthetic arenas, no Tetris):** safe 3-cycle retained; poisoned
  cycle rejected; unknown branch blocks certification; multi-placement witness.
- **Float ban:** grep-style test over the source file's non-test sections.
- **Parallel:** 1-thread vs N-thread runs produce identical fully-expanded
  topology (visit counts may differ).

## 15. Implementation phases

Each phase lands compiling with its tests green (`cargo fmt`, `clippy -D warnings`,
`cargo test`):

1. **Domain + arenas + interner** — StateKey/StateId/sentinels, bag transition,
   `Arena<T>`, concurrent intern with structural enumeration.
2. **Records + expansion** — full/edge expansion, frontier, policies.
3. **Integer scoring** — fixed-point math + property tests.
4. **Single-thread rollout** — selection, path, cycle detection, backup.
5. **Parallel epochs** — pool, batches, stress invariants, Ctrl-C.
6. **gfp closure** — bitset fixpoint, witnesses, root membership, synthetic tests.
7. **CLI/metrics polish + profiling** — evidence-driven optimization only
   (candidates: branch-block rank compression, MaybeUninit arenas, custom hasher).

## 16. Deviations from the original plan text (all approved)

1. Single flat file instead of module tree (matches sibling atlas bins).
2. `GameModel` trait dropped → `apply_move` free function with carrier folded in.
3. Structural enumeration at intern time → `placement_base/len` non-atomic;
   `expansion_status` keeps exactly the plan's three values.
4. Only surviving placements stored → no `LOST` sentinel; branches always point
   at real states; `placement_len == 0` marks terminal death.
5. `required_piece_mask` = `key.bag` (no extra field).
6. std `u128::isqrt` instead of hand-rolled.
7. Reserve-inside-vacant interning → no leaked slots on races.
8. Optional CLI carrier (sound player restriction), default unbounded.
9. Certificate serialization/verifier deferred; closure stats only.
10. Cycle-in-rollout backed up as survival (counted separately).
11. No TUI; plain logging + CSV.
12. 16-byte all-atomic branch records in a plain pre-initialized arena (no
    OnceLock); per-branch `max_survival_depth` replaced by one global
    diagnostic counter. Rationale: arena memory is paid at startup, and
    branch records dominate it ~40:1 over state records.
13. Rollouts expand single edges only (`resolve_child` per-edge CAS), never
    full states; the frontier is the sole driver of `FullyExpanded` in both
    policies (matches the original plan §17 "if edge is unknown: claim and
    expand edge"; keeps rollout latency flat; closure requires
    `FullyExpanded`, so soundness is unaffected).
14. (Phase 2) Arena exhaustion is no longer terminal by default: generational
    mark-compact at epoch barriers (§11b) evicts low-visit states and
    continues; `--no-compact` restores the original terminal behavior.
    Unresolvable rollout edges back up pessimistic deaths; `--admit-after`
    throttles candidate-mode admission.

## 17. Non-goals (v1)

Certificate persistence, Lean export, loom-based model checking, branch-block
compression, NUMA partitioning, graph checkpoint/resume.
