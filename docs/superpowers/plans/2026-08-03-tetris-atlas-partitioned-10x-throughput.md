# Partitioned Tetris Atlas 10x Throughput Plan

> **For implementers:** Execute this plan task-by-task. Every performance stage has a
> correctness gate, a same-checkpoint performance gate, and an explicit rollback point.
> Estimated gains below are hypotheses until measured.

**Goal:** Make the partitioned RocksDB atlas builder process at least 10 times as many
commands per second as the old prod4 implementation on the same database checkpoint,
hardware, writer count, RAM budget, and workload phase, without weakening atlas
correctness or adding a persistent policy flag.

**Architecture:** Replace the scalar read path with a batch plan. After claiming work, a
writer predicts every command-node and candidate-node read that the batch can require,
generates each deterministic Tetris transition once, rejects exact-known-dead and
definitely-absent states in memory, performs one sorted/deduplicated RocksDB MultiGet,
and then applies commands serially through the existing overlay. A bounded read-ahead
layer subsequently amortizes work-queue claim and commit overhead. All new scratch
storage is allocated once per worker and reused.

**Primary file:**
`crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

**Tech stack:** Rust nightly, `tetris-game`, `rocksdb` 0.22 / RocksDB 8.10,
`tetris_utils::HeaplessVec`, `rustc_hash::FxHashMap`, fixed-size atomics, JSONL
telemetry, and the existing RocksDB column families.

## Executive Estimate

There is no isolated change that responsibly promises a 10x end-to-end improvement.
The credible no-migration attempt consists of four overlapping changes:

| Stage | Main effect | Low | Expected | High |
|---|---|---:|---:|---:|
| Already implemented command-node MultiGet | Removes scalar primary node calls | 1.2x | 1.8x | 3.0x |
| Whole-batch candidate planning and MultiGet | Removes scalar candidate calls | 1.5x | 2.5x | 4.0x |
| Exact dead cache plus node-presence Bloom | Avoids redundant or definitely-missing reads | 1.2x | 1.6x | 2.5x |
| Work read-ahead and fuller commits | Amortizes claim/commit | 1.1x | 1.4x | 2.0x |
| Transition cache, only if CPU-bound afterward | Avoids repeated placement/scoring work | 1.0x | 1.2x | 1.7x |

These factors overlap and **must not be multiplied as independent guarantees**. The
working expectation is:

- **4-6x:** conservative outcome versus old prod4.
- **6-9x:** reasonable outcome if point-lookup overhead is the dominant cost.
- **10-12x:** stretch outcome if the combined plan raises effective batch occupancy and
  the dead-state working set has useful temporal locality.
- **More than 12x:** likely requires the fresh-database storage redesign in the
  contingency section.

The plan therefore targets 10x, but its acceptance gate reports the measured result even
if it falls short.

---

## Provisional Prod4 Baseline

Snapshot source:
`artifacts/databases/partitioned-prod-4-events-4.jsonl`, latest progress near
291,962 elapsed seconds on 2026-08-03.

This is a sizing baseline, not yet a reproducible benchmark claim: the running prod4
binary predates the current untracked source edits. Task 0 creates a commit-identified,
same-checkpoint baseline before any result is accepted.

| Metric | Prod4 cumulative value |
|---|---:|
| Commands | 18,301,641,552 |
| Committed batches | 69,983,814 |
| Average commands per batch | 261.51 |
| Processing time per command | 116.10 us |
| Claim plus commit time per command | 11.02 us |
| Processing share | 91.33% |
| All node DB Gets | 23,371,072,428 |
| Candidate DB Gets | 7,991,817,214 |
| Candidate reuse probes | 10,703,661,794 |
| Dead records verified by a DB read | 7,165,643,390 |
| Distinct dead-state insertions | 2,645,851,468 |
| Candidate states classified definitely new | 9,711,348,935 |
| Dead Bloom occupancy | 70.03% |
| Peak RSS | 13.85 GB |
| RocksDB block-cache usage | 4.88 GB |
| RocksDB data-block hit rate | about 18.9% |
| Pending compaction / write stalls | 0 / 0 |

### What these totals imply

1. Primary, non-candidate DB Gets were approximately:

   ```text
   23.371B all Gets - 7.992B candidate Gets = 15.379B primary Gets
   15.379B / 69.984M batches = 219.75 primary Gets per batch
   ```

   The command-node prefetch already written specifically targets this group.

2. Candidate classification issued approximately:

   ```text
   7.992B candidate Gets + 10.704B reuse probes = 18.695B scalar probes
   18.695B / 69.984M batches = 267.14 candidate probes per batch
   ```

   After command-node prefetch, this is expected to become the dominant scalar
   RocksDB path.

3. At least 4.520B dead verifications were repeats:

   ```text
   7.166B verified-dead reads - 2.646B distinct deaths = at least 4.520B repeats
   ```

   This is a lower bound because not every dead state must first be encountered through
   candidate classification.

4. Approximately 9.0B negative-branch reuse probes ultimately found a definitely
   absent node. A node-presence filter can reject most of those before RocksDB.

5. A larger dead Bloom alone cannot produce a major gain. About 89.7% of current
   dead-Bloom-positive checks verify a genuinely dead node. Removing every Bloom false
   positive would eliminate only about 10.3% of candidate Gets.

6. With the current average batch size held constant, claim and commit alone consume
   almost the entire 10x time budget. Processing cannot be the only optimized stage.

### Quantitative 10x budget

Current measured worker time is approximately:

```text
116.10 us processing + 11.02 us claim/commit = 127.12 us per command
```

A 10x target is therefore at most about `12.7 us/command`. The intended budget is:

| Component | Target |
|---|---:|
| Batch planning, lookup, and serial processing | <= 8.0 us/command |
| Claim and commit | <= 4.5 us/command |
| Total | <= 12.5 us/command |
| Effective batch occupancy | >= 768 commands when backlog permits |

The effective-batch target is important: merely making processing 10x faster would
yield only about 5.6x overall if claim and commit remained unchanged.

---

## Global Constraints

- **No migration for Tasks 0-9.** The existing prod4 database must remain resumable.
- Do not change `SCHEMA_VERSION`, state-key bytes, work-key bytes, reverse-edge bytes,
  column-family names, owner mapping, or persisted metadata in the main plan.
- Do not add a runtime policy flag for candidate preference. Existing states continue to
  be preferred; exact batch reads replace uncertain `MayExist` hints with exact
  `Existing` or `DefinitelyNew` knowledge.
- Commands remain serially applied in their existing `WorkKey` order within each owner.
  Prefetch may be speculative; mutation and output ordering may not be speculative.
- Earlier same-batch mutations must override prefetched DB values through the existing
  node overlay.
- Probabilistic structures may prove absence only through their no-false-negative
  invariant. They may never prove death or existence.
- The dead hot cache must compare the complete lossless carrier identity. Hash or slot
  collisions may cause misses only, never false hits.
- Publish a newly transitioned dead state to the shared exact cache only after its
  RocksDB write batch commits successfully. A precommit dead Bloom insertion is merely a
  hint that is verified; a precommit exact-dead insertion could let another owner commit
  work based on a death that is not durable.
- No per-command or per-candidate heap allocation. Large bounded buffers are allocated
  once in `WorkerScratch` because the worst-case candidate plan is too large for a thread
  stack.
- All RAM consumers are charged against `--ram-mib`; do not silently grow RSS outside
  the configured application budget.
- Keep strict death/edge/expand priority unless the mixed-priority experiment in Task 8
  passes its closure-quality ablation.
- Preserve graceful Ctrl-C/restart behavior and atomic RocksDB batch semantics.
- Do not stage or commit unrelated dirty-worktree files. Logical commit boundaries in
  this plan require explicit user authorization before performing Git writes.
- Do not claim a speedup from prod4's changing three-second intervals. Compare cloned
  checkpoints, fixed run specifications, medians, and dispersion.

## Required Validation Commands

Run after every implementation task that changes Rust:

```sh
cargo fmt --all -- --check
cargo test --release -p tetris-playground --features rocksdb-backend \
  --bin tetris_atlas_partitioned
cargo clippy -p tetris-playground --features rocksdb-backend \
  --bin tetris_atlas_partitioned -- -D warnings
```

Before accepting the entire plan, also run:

```sh
cargo test --workspace
```

If the full workspace test is blocked by unrelated dirty-tree failures, record the exact
failure and still run the focused release tests and strict bin Clippy command.

---

### Task 0: Freeze a reproducible baseline and rollback point

**Files:**

- Inspect: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`
- Read: `artifacts/databases/partitioned-prod-4-events-4.jsonl`
- Create locally: `artifacts/benchmarks/<baseline-id>/...` (gitignored)

**Purpose:** Separate the old running binary, current source, and future candidate so
that every speed claim names an executable and database checkpoint.

- [ ] **Step 1: Record source and machine identity**

  Save the full commit hash, `git status --short`, `rustc -Vv`, macOS version, CPU,
  physical RAM, binary inode, binary modification time, CLI, writers, and RAM budget.

- [ ] **Step 2: Gracefully pause a checkpoint source database**

  Use the program's supported signal path. Verify all workers joined and RocksDB closed
  before copying. Never copy the live writable database directory as an ordinary
  recursive copy.

- [ ] **Step 3: Create two filesystem-cloned checkpoint directories**

  One clone is the baseline input and one is the candidate input. Preserve the original
  database untouched. Validate `CURRENT`, the referenced manifest, metadata status,
  writer count, state count, work count, and seed state in both clones.

- [ ] **Step 4: Define the comparison window**

  Use the same starting checkpoint, eight writers, 8192 MiB, hardware, release profile,
  feature set, and run duration. Discard warm-up, then report at least 30 progress
  windows with median, p25, p75, p95, and standard deviation.

- [ ] **Step 5: Preserve baseline artifacts**

  Store logs, command line, commit, binary hash, and summarized metrics under
  `artifacts/benchmarks/<baseline-id>/`. Do not commit the database or logs.

**Acceptance:** A second baseline execution from the other clone is within 10% of the
first baseline's median commands/sec, or the noise source is understood before coding.

**Rollback:** No source or persistent schema changes occur in this task.

---

### Task 1: Add metrics that expose every avoided and remaining read

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

**New counters:**

```text
prepared_choice_requests
prepared_choice_unique
prepared_placements
prepared_choices_unused
candidate_prefetch_calls
candidate_prefetch_keys_planned
candidate_prefetch_keys_unique
candidate_prefetch_keys_filtered_dead_cache
candidate_prefetch_keys_filtered_absent
candidate_prefetch_db_found
candidate_prefetch_db_missing
candidate_unplanned_db_reads
dead_hot_cache_checks
dead_hot_cache_hits
dead_hot_cache_inserts
dead_hot_cache_replacements
presence_filter_checks
presence_filter_definite_absent
presence_filter_maybe_present
work_read_ahead_refills
work_read_ahead_commands
work_read_ahead_hits
```

- [ ] **Step 1: Extend worker, atomic, and serialized snapshots**

  Follow the existing `WorkerMetrics -> Atomic*Stats -> progress JSON` aggregation path.
  Counters use relaxed atomics and checked arithmetic where the surrounding code already
  requires it.

- [ ] **Step 2: Add derived ratios to progress output**

  Report dead-cache hit rate, presence-filter negative rate, prepared-choice reuse,
  unique keys per MultiGet, keys filtered before DB, unplanned candidate-read rate, and
  effective commands per committed batch.

- [ ] **Step 3: Test aggregation and serialization**

  Seed nonzero values in a `WorkerMetrics`, record them, snapshot them, and assert every
  field. Ensure division-by-zero produces `null`, matching current telemetry style.

**Expected performance:** Neutral to -1%; this task is instrumentation.

**Acceptance:** Existing metrics retain their meanings. In particular, speculative
candidate planning does not increment semantic `candidates.examined`; that counter is
incremented only when an actual `choose_next` consumes a prepared placement.

**Rollback:** Remove only the new counters; no behavior or schema changes.

---

### Task 2: Add a lossless 47-bit in-memory carrier identity

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

**Interface:**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactCarrierKey(u64);

impl CompactCarrierKey {
    fn from_state_key(state: StateKey) -> Result<Self>;
    const fn stored(self) -> u64; // high bit marks an occupied cache slot
}
```

**Encoding:**

```text
bits  0..40: ten four-bit columns, column 0 first
bits 40..47: nonempty seven-bit bag mask
bit      63: occupied marker when stored in a cache slot
```

The database continues using the existing 31-byte `StateKey`. This compact key is only
an exact in-memory identity for states satisfying `MAX_CARRIER_HEIGHT == 4`.

- [ ] **Step 1: Write boundary tests**

  Cover empty board, every individual carrier cell, all ten full four-cell columns,
  mixed columns, every valid nonempty bag mask, and the maximum packed value.

- [ ] **Step 2: Reject non-carrier and malformed identities**

  Assert that any column bit above row 3 and any invalid bag byte is rejected. Do not
  truncate invalid input.

- [ ] **Step 3: Add deterministic property coverage**

  With fixed seeds, generate many admissible boards and bag masks. Assert equal state
  keys produce equal compact keys and unequal board/bag pairs never collide.

- [ ] **Step 4: Add compile-time size assertions**

  Assert the compact key is eight bytes and all valid payload bits are below bit 47.

**Expected performance:** Negligible alone. It makes a high-capacity exact cache viable:
eight bytes per slot instead of storing a 31-byte key plus hash-table overhead.

**Acceptance:** Lossless identity is demonstrated for the entire carrier representation;
no database bytes change.

**Rollback:** Delete the type and tests.

---

### Task 3: Add a shared exact dead-state hot cache

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

**Interface sketch:**

```rust
struct DeadHotCache {
    slots: Box<[AtomicU64]>,
    mask: usize,
}

impl DeadHotCache {
    fn new(bytes: usize) -> Result<Self>;
    fn contains(&self, key: CompactCarrierKey) -> bool;
    fn insert(&self, key: CompactCarrierKey) -> bool;
}
```

Use a source constant initially sized at 512 MiB. Capacity must be a power of two and is
charged against `--ram-mib`. Use a stable integer mixer for the slot index. Store the
complete compact key with bit 63 set; compare the complete stored value on lookup.
Relaxed atomic load/store is sufficient because races only replace cache entries.

- [ ] **Step 1: Test empty, hit, miss, and replacement behavior**

  Force two keys into the same slot. After replacement, the old key must miss and the
  new key must hit. No unrelated key may hit.

- [ ] **Step 2: Insert only durable exact-death observations**

  Insert immediately whenever an exact DB record verifies a previously committed dead
  node. When a node transitions to dead in the current write batch, append its compact
  key to a bounded `pending_dead_cache_inserts` scratch list and publish those entries
  only after `db.write_opt` succeeds. If commit fails, publish none of them. The cache
  starts empty after restart and warms opportunistically; recovery need not populate it.

- [ ] **Step 3: Consult before the probabilistic dead Bloom**

  An exact hit returns `CandidateState::Dead` without RocksDB. A miss falls through to
  the existing dead Bloom and exact verification path.

- [ ] **Step 4: Add cache telemetry**

  Count checks, hits, inserts, replacements, allocated bytes, and avoided candidate DB
  reads.

- [ ] **Step 5: Add concurrent stress coverage**

  Fixed threads repeatedly insert colliding and non-colliding keys. Verify that every
  observed hit is for the queried complete key. Missing due to races is allowed.

**Estimated improvement:**

- Addressable historical work: at least 4.520B repeated dead verifications.
- Expected cache hit rate: 20-60%, to be measured.
- Expected end-to-end gain by itself: 1.1-1.6x.
- High case with strong temporal locality: about 2x.

**Acceptance:** Zero false hits in tests; cache memory appears in startup telemetry and
RAM accounting; `dead_hot_cache_hits` equals the number of exact DB reads avoided.

**Rollback:** The cache is ephemeral. Removing it leaves the database fully compatible.

---

### Task 4: Add a global node-presence Bloom filter

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

Refactor the current dead Bloom implementation into a reusable fixed-size atomic Bloom
or add a separate equivalent type. The node-presence filter contains every node ever
created, whether live or dead.

Start with a 1 GiB source constant. At approximately 3.057B nodes and four hashes, its
predicted false-positive probability is about 33%; it can still reject roughly two-thirds
of truly absent states. A 2 GiB filter predicts about 6.7% false positives but must not be
selected until RAM/cache tradeoffs are measured.

- [ ] **Step 1: Build presence during the existing recovery scan**

  The recovery loop already visits every node to reconstruct the dead Bloom and counts.
  Insert every valid node key into the presence filter in that same pass; do not add a
  second RocksDB scan.

- [ ] **Step 2: Insert before publishing every new node**

  Centralize the absent-to-created transition so it cannot omit the presence insertion.
  Inserting before DB commit is safe: a failed commit creates a false positive, which
  only causes an unnecessary exact DB lookup. Inserting after another thread can observe
  the node would be unsafe because it could create a false negative.

- [ ] **Step 3: Use negatives to prove absence**

  For candidate classification, a presence-filter negative produces
  `DefinitelyNew` without `key_may_exist`. For exact node loads that legally allow
  absence, record an overlay `None` without a DB Get. A positive must fall through to an
  exact read or conservative existing behavior.

- [ ] **Step 4: Test the no-false-negative lifecycle**

  Cover recovery, new-node creation, precommit insertion, simulated write failure,
  restart rebuild, and concurrent reader/creator timing. Tests may observe false
  positives but must never observe a false negative for a published node.

- [ ] **Step 5: Account for memory before constructing RocksDB caches**

  Reserve the dead Bloom, presence Bloom, dead hot cache, native overhead, and per-writer
  scratch before dividing remaining `--ram-mib` between the block cache and write buffer.
  Reject configurations that cannot provide the fixed reserves.

**Estimated improvement:**

- Historical definitely-absent candidate probes: approximately 9.0B.
- 1 GiB expected avoidance at current scale: roughly 6.0B probes.
- 2 GiB expected avoidance at current scale: roughly 8.4B probes.
- End-to-end gain by itself: 1.1-1.5x, because RocksDB's own Bloom makes some misses
  inexpensive even though the API/LSM traversal is not free.

**Acceptance:** No false negatives; no extra recovery scan; presence negatives and DB
misses reconcile within the new telemetry.

**Rollback:** The filter is ephemeral and has no metadata. Removing it is restart-safe.

---

### Task 5: Prepare deterministic choices once per batch

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

**New bounded scratch structures:**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ChoiceKey {
    pred: StateKey,
    piece: TetrisPiece,
}

enum PreparedPlacementState {
    Lost,
    RejectedHeight,
    RejectedHoles,
    Admissible {
        succ: StateKey,
        score: f32,
        orientation: TetrisPieceOrientation,
    },
}

struct PreparedChoice {
    key: ChoiceKey,
    placements: Range<usize>,
}
```

Allocate reusable vectors once in `WorkerScratch`:

- At most `WORK_BATCH_COMMANDS * TetrisPiece::NUM_PIECES` choice requests.
- At most `MAX_CANDIDATES_PER_BATCH` prepared placements.
- At most `MAX_CANDIDATES_PER_BATCH + WORK_BATCH_COMMANDS` node-prefetch keys.

Do not place the maximum candidate array directly on a writer thread's stack.

- [ ] **Step 1: Identify potential choice requests from commands**

  - `Expand { state }`: every piece in `state.bag()` may need a choice.
  - rejected `AttachResult`: `(pred, piece)` may reroute.
  - `ChildDead`: `(pred, piece)` may reroute.
  - Other commands cannot call `choose_next`.

  These are safe supersets. Stale commands may leave a prepared choice unused.

- [ ] **Step 2: Sort/deduplicate `ChoiceKey`s**

  If multiple commands may request the same predecessor and piece, generate its Tetris
  transitions once and let all consumers share the prepared range.

- [ ] **Step 3: Generate every placement outcome once**

  Apply the piece, loss check, carrier height check, hole check, successor encoding, and
  score during preparation. Do not classify existence/death yet.

- [ ] **Step 4: Make `choose_next` consume a prepared range**

  Preserve the current `candidate_order`, orientation tie-break, and existing-state
  preference. Increment semantic candidate counters while consuming the range, not when
  preparing speculative work.

- [ ] **Step 5: Keep a scalar reference under tests**

  Compare scalar and prepared choice results over every piece, fixed randomized carrier
  boards, missing/live/dead successors, duplicate successors, and score ties.

**Estimated improvement:** Neutral until Task 6; possibly a small regression because
some stale commands prepare unused candidates. It is separated so correctness can be
established before changing the DB read path.

**Acceptance:** Single-writer scalar and prepared traces match when supplied the same
candidate classifications. All existing candidate counters retain semantic parity.

**Rollback:** Restore scalar `choose_next`; no persistent changes.

---

### Task 6: Perform one whole-batch candidate-aware node MultiGet

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

Replace `prefetch_command_nodes` with a broader `prepare_and_prefetch_batch`:

```text
claimed commands
  -> primary node keys
  -> unique potential ChoiceKeys
  -> prepared admissible successors
  -> exact dead-cache filtering
  -> presence-filter definite-absence filtering
  -> one combined StateKey list
  -> sort_unstable + dedup
  -> RocksDB batched_multi_get_cf_opt
  -> node overlay
  -> serial apply
```

- [ ] **Step 1: Build one combined node-read plan**

  Include primary command keys and candidate keys not resolved by exact in-memory facts.
  Keep `StateKey` values directly; do not allocate encoded byte-vector copies.

- [ ] **Step 2: Sort and deduplicate once**

  Continue passing `sorted_input = true` to RocksDB. Sorting remains necessary because
  RocksDB is an LSM, not a B-tree, and otherwise sorts its own temporary pointer list.

- [ ] **Step 3: Populate baseline overlay entries**

  Present values decode into `NodeRecord`; missing values become `NodeOverlayEntry` with
  `record: None`. Count every key result but only one logical prefetch call.

- [ ] **Step 4: Apply commands serially**

  The existing overlay is the coherence mechanism:

  - Prefetched missing, then created earlier in the batch -> later reads see created.
  - Prefetched live, then killed earlier in the batch -> later reads see dead.
  - Prefetched existing, then modified earlier -> later reads see modified.

  Do not cache a detached copy of candidate classification outside this overlay unless it
  is an exact monotonic dead-cache hit or a proven presence-filter negative.

- [ ] **Step 5: Eliminate normal scalar candidate reads**

  `classify_candidate` should resolve through prepared facts and overlay entries.
  Retain a correctness fallback initially, but count it as
  `candidate_unplanned_db_reads`. The acceptance target is zero in normal production
  batches.

- [ ] **Step 6: Use exact reuse classification**

  Missing exact result -> `DefinitelyNew`; present live result -> `Existing`; dead exact
  result -> reject. This removes false-positive `MayExist` preferences while preserving
  the requested policy of preferring actually existing successors. Do not persist a
  policy field or add a policy flag.

- [ ] **Step 7: Add focused correctness cases**

  Cover empty plan, all command kinds, all pieces in an Expand, conditional reroute,
  stale commands, duplicate candidate keys, DB error propagation, wrong result count,
  overlay capacity, candidate capacity, and stop-limit boundaries.

**Estimated improvement:**

- Historical scalar candidate probes targeted: 18.695B, or 267 per batch.
- RocksDB may internally process MultiGet keys in smaller chunks, but the Rust/C API
  boundary, key preparation, and table access are shared.
- Expected incremental end-to-end gain after command-node prefetch: 1.5-3x.
- High I/O-latency case: up to about 4x.

**Acceptance:**

- `candidate_unplanned_db_reads == 0` after warm-up.
- At least 95% fewer Rust-level point-read calls per 1M commands versus prod4.
- No new RocksDB background errors, stalls, or correctness-test differences.
- Same-checkpoint median commands/sec improves by at least 1.5x over Task 0 candidate
  baseline; otherwise profile before proceeding.

**Rollback:** Restore command-only prefetch and scalar classification; database remains
compatible.

---

### Task 7: Benchmark and tune the memory split

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`
- Create locally: benchmark logs under `artifacts/benchmarks/` only

- [ ] **Step 1: Measure 512 MiB dead cache plus 1 GiB presence Bloom**

  Report application-filter hit rates, block-cache hit rate, table-reader memory, RSS,
  page pressure, candidate DB keys, and commands/sec.

- [ ] **Step 2: Compare presence-filter sizes by source commit**

  Compare 1 GiB and 2 GiB from identical checkpoint clones. Do not add a runtime policy
  flag. The winning source constant must improve end-to-end throughput, not merely Bloom
  false-positive rate.

- [ ] **Step 3: Compare dead-cache sizes only if its hit rate is useful**

  If the 512 MiB cache hits fewer than 10% of checks, do not enlarge it. If it hits more
  than 30%, compare 512 MiB and 1 GiB while keeping total `--ram-mib` fixed.

- [ ] **Step 4: Preserve enough RocksDB cache**

  Reject any configuration where application structures improve avoided-read counters
  but reduce commands/sec because the RocksDB block cache becomes too small.

**Expected improvement:** This task chooses within the Task 3/4 range; it does not add an
independent multiplier.

**Acceptance:** Select the median-throughput winner with peak RSS safely below physical
RAM and no sustained memory pressure. Record the losing ablations.

---

### Task 8: Add bounded work-queue read-ahead and improve batch occupancy

**Files:**

- Modify: `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

The current claim path constructs an iterator and seeks into the work column family for
each batch. Introduce bounded per-owner read-ahead buffers while RocksDB remains the
durable source of truth.

**Interface sketch:**

```rust
struct WorkLaneBuffer {
    priority: u8,
    producer: u8,
    next_sequence: u64,
    commands: VecDeque<ClaimedCommand>,
}

struct WorkReadAhead {
    lanes: Box<[WorkLaneBuffer]>,
}
```

- [ ] **Step 1: Refill lanes in large sequential ranges**

  Use current persistent heads as lower bounds. A buffer is only a mirror; consuming it
  does not advance durable state until the normal write batch commits.

- [ ] **Step 2: Recreate iterators when a lane is exhausted**

  Do not retain a RocksDB iterator indefinitely because it would not reliably expose
  subsequently committed commands. Refill in bounded chunks and reopen from the current
  sequence.

- [ ] **Step 3: Preserve failure semantics**

  If processing or commit fails, discard uncommitted in-memory consumption. Restart
  reconstructs work entirely from RocksDB and persistent heads.

- [ ] **Step 4: Preserve strict priority initially**

  Always probe/refill higher-priority death work before serving edge or expansion work.
  Measure claim time and batch occupancy without scheduling changes first.

- [ ] **Step 5: Test the optional fuller-batch scheduling experiment separately**

  Only if average occupancy remains below 512, test filling the remainder with the next
  priority after the current priority is exhausted at the claim snapshot. This changes
  scheduling, so compare closure reproduction, death cascade, outstanding work, and
  frontier trend from identical checkpoints. Adopt it directly in source only if the
  research metrics do not regress materially; do not leave a production policy flag.

- [ ] **Step 6: Test resume and producer interleavings**

  Cover sparse lanes, newly appended commands, sequence gaps, coarse reclaimed ranges,
  wraparound producer start, priority changes, commit failure, and graceful shutdown.

**Estimated improvement:**

- Read-ahead alone: 30-70% reduction in claim time, usually 1.03-1.08x end-to-end before
  other optimizations.
- Raising occupancy from about 261 to 512: potentially 1.2-1.5x after processing becomes
  fast.
- Raising occupancy toward 768-1024: potentially 1.5-2x after processing becomes fast.

**Acceptance:**

- Claim plus commit <= 4.5 us/command in the target workload, or total throughput already
  meets 10x without it.
- Median batch occupancy >= 512, with >= 768 as the target when backlog permits.
- No lost, duplicated, or reordered-within-priority commands.
- Closure-quality metrics remain within the predeclared tolerance in Task 10.

**Rollback:** Discard buffers and resume iterator-per-claim. Persistent data is unchanged.

---

### Task 9: Add a deterministic transition cache only if profiling requires it

**Files:**

- Modify conditionally:
  `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`

After Tasks 3-8, profile where `process_micros` remains. If board placement, carrier
checks, or `height_mse_board_score` account for less than 20% of processing time, skip
this task.

The cache key is `(CompactCarrierKey, piece)`. Its immutable value contains only
deterministic placement results: rejection reason or `(successor, orientation, score)`.
It must not cache live/dead/existing classification.

- [ ] **Step 1: Instrument repeated `ChoiceKey`s across batches**

  Estimate hit rate before reserving memory. Do not infer locality solely from reroute
  count.

- [ ] **Step 2: Use a writer-local bounded direct-mapped or set-associative cache**

  Preallocate once. Complete key comparison is mandatory. Replacement creates misses
  only.

- [ ] **Step 3: Reclassify every cached successor at use time**

  Board transitions and scores are immutable; node status is not. Route status through
  the exact dead cache, presence filter, and current batch overlay.

- [ ] **Step 4: Compare against no-cache source commit**

  The cache must repay its memory and lookup cost with at least a 10% end-to-end gain.

**Estimated improvement:** 1.0-1.7x, entirely dependent on measured temporal locality.

**Acceptance:** Keep only if hit rate is at least 20%, commands/sec improves at least
10%, and application memory does not displace more valuable RocksDB blocks.

**Rollback:** Remove the ephemeral cache.

---

### Task 10: Run the full correctness, performance, and solver-quality gate

**Files:**

- Create locally: `artifacts/benchmarks/<candidate-id>/...`
- Do not commit generated benchmark output

- [ ] **Step 1: Run formatting, focused release tests, strict Clippy, and workspace tests**

  Record every command and full output.

- [ ] **Step 2: Run a deterministic single-writer equivalence test**

  From identical small databases, process a fixed command budget with scalar reference
  logic and batch logic. Compare node records, reverse edges, outstanding work commands,
  persistent heads, node/death counts, and seed status.

- [ ] **Step 3: Run identical-checkpoint eight-writer performance trials**

  Use at least three baseline and three candidate runs, alternating order if practical.
  Report warm-up policy, N, median, mean, standard deviation, p25/p75/p95, hardware,
  `rustc -Vv`, source commit, dirty status, binary hash, and full CLI.

- [ ] **Step 4: Report the complete throughput decomposition**

  Include commands/sec, candidates/sec, nodes/sec, process/claim/commit us per command,
  commands per batch, scalar and batched DB calls, keys per MultiGet, application-cache
  hit rates, RocksDB cache hit rate, bytes read/written, compaction bytes, stalls,
  background errors, RSS, and CPU utilization.

- [ ] **Step 5: Run a solver-quality ablation**

  Compare at equal processed-command budgets, not only equal wall time:

  - expansion reproduction,
  - frontier delta,
  - novelty rate,
  - realized successor reuse,
  - death cascade factor,
  - unresolved states and forced slots,
  - nodes created and dead.

- [ ] **Step 6: Apply the acceptance decision**

  The primary target is >= 10x median commands/sec. A result below 10x is reported
  honestly and retained only if it is materially faster, correct, and does not worsen
  solver progress enough to erase its wall-time advantage.

**Performance acceptance:**

- Candidate median commands/sec >= 10x baseline median.
- Candidate processing <= 8.0 us/command.
- Candidate claim plus commit <= 4.5 us/command.
- No RocksDB background errors or sustained write stalls.
- Peak memory stays within the declared run budget and machine headroom.

**Correctness acceptance:**

- All tests pass.
- No missing or duplicate durable work in equivalence checks.
- No probabilistic false-death decisions.
- Restart produces the same counts and valid metadata.
- Every persisted active/pending edge and death transition satisfies existing hardening
  assertions.

**Solver-quality acceptance:**

- No statistically meaningful degradation greater than 5% in expansion reproduction or
  resolved-state progress at equal command budgets unless the research director
  explicitly accepts the tradeoff.
- Existing-successor preference uses exact existence and remains at least as effective as
  the old probabilistic hint according to realized successor reuse.

---

## Contingency: Fresh-Database Storage Redesign

Do not start this section unless Tasks 0-10 are correct and the measured result remains
below the target. These changes are not resumable from prod4 without migration.

### Contingency A: Six-byte persistent carrier keys

All persisted states satisfy the height-four carrier, so a new schema can encode a state
in six bytes instead of 31:

```text
5 bytes: ten four-bit columns
1 byte: seven-bit bag mask
```

Consequences:

- Node keys: 31 -> 6 bytes.
- Reverse-edge keys: 64 -> 14 bytes.
- Work command payloads containing states shrink by roughly 4-5x.
- Hashing, comparison, sorting, index keys, cache density, compaction, and write bytes all
  improve.

Expected end-to-end gain: approximately 1.5-3x by itself, possibly 4-5x in
write/compaction-heavy phases. RocksDB prefix compression means raw key-width reduction
does not translate directly into a 5x database reduction.

This requires a schema-version change and a fresh database. The old DB must be rejected
clearly; no silent reinterpretation is allowed.

### Contingency B: Partition-native state and work storage

If RocksDB remains dominant after compact keys, the true >10x architecture is:

- owner-sharded state tables keyed by compact carrier id,
- append-oriented, sequential work-lane logs plus durable heads,
- bounded in-memory read windows,
- append-oriented reverse-edge segments with compaction specialized for death draining,
- group commit across owner batches,
- checkpointed recovery metadata.

This removes general-purpose LSM costs but is a separate storage-engine project with a
substantially larger proof and recovery burden. Estimated potential is 8-20x over the
old RocksDB implementation; confidence is low until a representative prototype measures
lookup, append, recovery, and compaction behavior.

---

## Recommended Execution Order

```text
Task 0  reproducible baseline
  -> Task 1 metrics
  -> Task 2 compact in-memory identity
  -> Task 3 exact dead hot cache
  -> Task 4 node-presence Bloom
  -> Task 5 prepared deterministic choices
  -> Task 6 whole-batch candidate MultiGet
  -> Task 7 memory ablation
  -> Task 8 work read-ahead / batch occupancy
  -> Task 9 transition cache only if profiling says so
  -> Task 10 final gate
  -> contingency only if measured throughput remains insufficient
```

The most important checkpoint is after Task 6. At that point both primary and candidate
node reads are batched, the two largest avoidable key classes can be rejected in memory,
and the same database remains resumable. If that checkpoint does not provide a large
gain, stop and profile rather than adding speculative caches or increasing constants.

## Rollback Summary

Tasks 1-9 add only metrics, ephemeral memory structures, scratch planning, and a work
read-ahead mirror. They do not change persisted bytes. Any of them can be rolled back by
building an earlier source checkpoint and reopening the same gracefully paused database.

The contingency changes persistent encoding and require a new database. They must live
behind a schema bump and a separate, explicit research decision—not an automatic upgrade
or migration path.
