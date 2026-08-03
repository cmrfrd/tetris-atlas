# Partitioned Greedy Closed-Carrier Atlas (RocksDB, fixed-RAM)

**Date:** 2026-07-15
**Revised:** 2026-07-25
**Status:** Implementation in progress
**Target file:** `crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`
**Reference:** `crates/tetris-playground/src/bin/atlas/tetris_atlas_inmemory.rs`

## 1. Goal

Replace the current RocksDB beam-search atlas with a disk-backed adversarial
AND-OR survival engine. The engine searches for a closed policy inside a finite
admissibility carrier:

- a state is `(board, remaining_bag)`;
- the bag adversarially chooses any remaining piece;
- the player chooses a legal placement for that piece;
- every selected successor must remain in the carrier;
- death propagates backward monotonically;
- a surviving state has one committed placement for every forced piece.

The implementation must:

1. Preserve the greatest-fixpoint survival semantics of the in-memory engine.
2. Keep application memory bounded independently of graph size.
3. Use partitioned mutation ownership so RocksDB writes can proceed concurrently.
4. Never rely on a stale read to install an edge to a dead successor.
5. Keep the persistent schema compact enough for large state counts.
6. Remain deterministic across worker counts.

The target artifact is the surviving `node` table. Starting from the empty
board and full bag, that table is a constructive infinite-survival policy if
the seed survives and the final closure validator passes.

## 2. Semantics

### 2.1 AND-OR survival

For state `S = (board, bag)`:

- `S` is an AND node. It survives iff every piece `p` in `bag` has a surviving
  player choice.
- `(S, p)` is an OR node. It survives iff at least one legal placement of `p`
  reaches a surviving state.
- A successor that tops out or leaves the admissibility carrier is not an
  option.
- A state dies when any forced piece exhausts every option.
- Death is monotone. A dead state never becomes live again.

Cyclic dependencies with no path to a dead sink remain live. This is the
greatest-fixpoint interpretation required for infinite survival.

### 2.2 Lazy greedy choice

Each `(state, piece)` keeps at most one active successor.

Candidates are considered in this deterministic order:

1. Higher `height_mse_board_score` is better.
2. Lower `TetrisPieceOrientation::index()` breaks ties.

The implementation uses a linear best-candidate scan. It does not sort or
persist candidate lists. When the selected successor dies, the parent
recomputes the candidates and selects the best successor not currently known
dead.

`height_mse_board_score` returns `f32`. Comparisons use `f32::total_cmp`.
Candidate boards are finite and non-topout, so the scorer must not produce
NaN. A NaN is treated as a fatal invariant violation, not silently ordered.

Greedy order chooses the policy, not the fixpoint: if the preferred successor
dies, the engine tries the next surviving option until the OR node either
finds one or exhausts all options.

### 2.3 Carrier

The only semantic tuning point is:

```rust
#[inline(always)]
const fn is_admissible(board: TetrisBoard, _bag: TetrisPieceBagState) -> bool {
    const MAX_HEIGHT: u32 = 8;
    const MAX_HOLES: u32 = 1;

    board.height() <= MAX_HEIGHT && board.total_holes() <= MAX_HOLES
}
```

The carrier must contain the seed. Changing worker count, batch size, cache
size, or compaction settings must not change the result.

## 3. Architecture

### 3.1 One database, partition-owned mutation

The first implementation uses one physical RocksDB database shared by several
writer threads. State ownership is partitioned:

- hash each `StateKey` into one of 256 virtual shards;
- map every virtual shard to exactly one writer for the lifetime of the run;
- only that writer may mutate the state's `node` record;
- only the successor's owner may mutate reverse edges under that successor's
  prefix;
- any writer may append an immutable command to another writer's work queue;
- only the target writer may claim or delete commands from its queue prefix.

The virtual-shard map is fixed at database creation and stored in metadata.
There is no live shard migration in the initial implementation.

Use a stable 64-bit hash over the packed state bytes followed by an avalanche
finalizer. Do not use the raw low byte of FNV-1a as the shard directly. The
owner map balances the 256 virtual shards across `NUM_WRITERS`.

Initial runtime default:

```text
NUM_WRITERS = min(4, available_parallelism)
```

`--writers` may override it in `1..=256`. Performance runs compare 1, 2, 4,
and 8 writers before choosing a machine-specific default.

### 3.2 Why commands are required

A predecessor owner cannot directly install a reverse edge under a successor
owned by another writer. Doing so creates this unsound interleaving:

1. The predecessor reads the successor as live.
2. The successor owner marks it dead and drains its reverse-edge prefix.
3. The predecessor commits a new edge after the drain.
4. No later death task exists to invalidate the predecessor.

The engine therefore uses an attachment handshake:

1. The predecessor records a pending orientation.
2. It sends `AttachEdge` to the successor owner.
3. The successor owner checks the authoritative successor record.
4. The successor owner either records the edge and accepts, or rejects because
   the successor is dead.
5. The predecessor activates the orientation only after acceptance.

Successor death and reverse-edge attachment are serialized by the same owner.
Once the owner marks a successor dead, later attachments are rejected and its
reverse-edge prefix is permanently closed to new inserts.

### 3.3 Compute parallelism

Writer ownership applies only to persistent mutations. Candidate generation,
piece placement, scoring, key construction, and read-set preparation may use a
separate compute pool.

Each writer remains the coordinator for its bounded command batch:

1. claim commands;
2. prepare bounded compute jobs;
3. run candidate computation in parallel;
4. issue batched reads;
5. apply state-machine transitions in command order;
6. commit one bounded `WriteBatch`.

Parallel compute is optional. First benchmark the writer-coordinated
implementation before adding dispatch overhead.

## 4. Persistent Schema

The database uses the default column family for metadata and three explicit
column families:

| CF | Purpose |
|---|---|
| `node` | Canonical state records and final atlas |
| `redge` | Successor-owned reverse edges |
| `work` | Partitioned durable command queue |

No merge operator, transactions, lookup CF, expansion CF, or death CF is
needed.

### 4.1 Compact state key

Canonical boards have 10 columns and 20 playable rows. Only the low 20 bits of
each `u32` column are semantic for a non-topout board. Encode each column in
three little-endian bytes:

| Field | Bytes |
|---|---:|
| 10 packed columns | 30 |
| remaining bag | 1 |
| **Total** | **31** |

```rust
const PACKED_COLUMN_BYTES: usize = 3;
const STATE_KEY_LEN: usize = StandardTetris::COLS * PACKED_COLUMN_BYTES + 1;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateKey([u8; STATE_KEY_LEN]);
```

Encoding rules:

- columns are encoded in column order;
- each column uses its low, middle, then high byte;
- bits above `StandardTetris::ROWS` must be zero;
- the bag byte must be nonzero and contain no bits outside
  `TetrisPieceBagState::FULL_MASK`, because a drained bag is refilled before
  state construction;
- invalid inputs return an error and stop the run;
- decoding rebuilds `[u32; 10]` and uses `TetrisBoard::from_limbs`.

The byte order exists for exact identity and prefix grouping. It does not claim
to match `TetrisBoard`'s numeric `Ord`.

### 4.2 Node record

`node`:

| | Layout | Size |
|---|---|---:|
| Key | `StateKey` | 31 B |
| Value | seven move slots | 7 B |

One slot is indexed by `TetrisPiece::index()`:

| Byte | Meaning |
|---|---|
| `0x00..=0x27` | Active orientation index |
| `0x40..=0x67` | Pending orientation index (`byte - 0x40`) |
| `0xFE` | Unexpanded |
| `0xFF` | Exhausted |

All other bytes are invalid.

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
struct NodeRecord([u8; 7]);

enum MoveSlot {
    Active(u8),
    Pending(u8),
    Unexpanded,
    Exhausted,
}
```

`NodeRecord::INIT` is seven `Unexpanded` bytes.

The bag determines which slots matter:

- `dead(record, bag)`: any forced slot is `Exhausted`;
- `resolved(record, bag)`: every forced slot is `Active`;
- `has_pending(record, bag)`: any forced slot is `Pending`;
- non-forced slots remain `Unexpanded`.

An active or pending orientation must be valid for the slot's piece, not merely
inside the global `0..40` orientation range. Contextual node reads validate
this before using the orientation.

There is no separate dead byte and no `NOT_FORCED` sentinel. A state can only
become dead because one of its forced OR nodes exhausts its candidates, so the
exhausted slot is the state-level death witness.

Only the state's owner writes its `NodeRecord`. Writers use full seven-byte
`Put`s, not field merge operands.

### 4.3 Reverse edges

`redge`:

| | Layout | Size |
|---|---|---:|
| Key | `succ(31) || pred(31) || piece(1) || orientation(1)` | 64 B |
| Value | empty | 0 B |

The successor leads the key so one prefix scan finds every predecessor that
currently or formerly selected it.

The successor's owner is the sole writer for its prefix:

- `AttachEdge` inserts an edge only while the successor is not dead;
- `DetachEdge` removes an exact stale edge;
- `DrainDead` removes a dead successor's prefix in bounded chunks;
- after successor death, `AttachEdge` rejects and cannot reopen the prefix.

Including orientation makes stale notifications self-identifying. A parent
acts on a notification only if its current pending or active orientation still
derives the same successor.

### 4.4 Work queue

`work`:

| | Layout | Size |
|---|---|---:|
| Key | `owner(1) || priority(1) || producer(1) || sequence_be(8)` | 11 B |
| Value | encoded `Command` | bounded by command schema |

Each writer is a producer and owns one local monotonically increasing sequence
counter. A work key is unique because it includes both producer and sequence.
The writer allocates sequence numbers in command-emission order and commits its
batches serially, so commands from one producer become visible in sequence
order. Unique keys are required for exact task accounting; commands must never
collapse because two semantic payloads happen to match.
`u64::MAX` is reserved and never issued, leaving `last_sequence + 1` available
as the exclusive upper bound for a consumed lane range.

The producer lane is necessary. A single global atomic sequence would not be
safe for an advancing queue cursor: producer A could allocate sequence 100 and
stall before commit while producer B allocates and commits 101. Consuming 101
must not hide A's later publication of 100. Per-producer serial commit removes
that publication inversion.

Priorities:

| Priority | Commands |
|---:|---|
| 0 | `DrainDead`, `ChildDead`, rejected `AttachResult` |
| 1 | `AttachEdge`, `DetachEdge`, accepted `AttachResult` |
| 2 | `Expand` |

`AttachEdge` and `DetachEdge` deliberately share a priority. A detach emitted
after an attach receives a larger sequence and therefore cannot overtake it in
their shared producer lane.

Each owner processes the lowest available priority. Within a priority it starts
at a rotating producer, scans the lexicographically contiguous producer range
to the last producer, then wraps once to producer zero if capacity remains.
This needs at most two RocksDB seeks per priority rather than one seek per
producer. There is no per-producer quota: the bounded command count limits the
scan, and rotating the start producer prevents a permanently busy lane from
starving the others. Claimed commands are sorted by full work key before
execution, so sequence order within every producer lane is preserved. New
commands from that producer always append after consumed sequence numbers, so
later inserts are never hidden.

### 4.5 Commands

```rust
enum Command {
    Expand {
        state: StateKey,
    },
    AttachEdge {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    AttachResult {
        accepted: bool,
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    ChildDead {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    DetachEdge {
        succ: StateKey,
        pred: StateKey,
        piece: TetrisPiece,
        orientation: TetrisPieceOrientation,
    },
    DrainDead {
        succ: StateKey,
    },
}
```

The wire value is one tag byte followed by the listed fields in declaration
order. Its largest form is `AttachResult`: tag, accepted byte, two 31-byte
states, piece, and orientation, for 66 bytes. Encoders write fixed stack arrays;
decoders copy into the explicit semantic types after validation.

Every decoder validates:

- exact payload length;
- valid piece and orientation indices, including that the orientation belongs
  to the command's piece;
- work-key priority matches the command tag and accepted/rejected result;
- target owner matches the command's authoritative state (`succ` for
  attach/detach/drain, `pred` for results and child-death, `state` for expand);
- producer matches the authoritative sender: `owner(state)` for expand,
  `owner(pred)` for attach/detach, and `owner(succ)` for results,
  child-death, and drain;
- all `StateKey` invariants.

Malformed data is a fatal database error. It is never ignored and no callback
runs across an FFI boundary.

### 4.6 Metadata

The default CF stores:

- schema version;
- build status: `building`, `paused`, `complete`, `seed-dead`, or `aborted`;
- carrier parameters or a carrier-version identifier;
- scorer identifier;
- writer count and the 256-entry virtual-shard owner map;
- the next command sequence for each producer;
- source commit hash;
- completion summary and validation result.

`create` initializes an empty directory or resumes a cleanly `paused` database.
It also accepts legacy `aborted` databases created before the paused status
existed. `explore` only inspects or replays a `complete` database. Both reject
`building` because an unclean process loss is not recoverable with the WAL
disabled.

Runtime telemetry remains log-only and is reconstructed where needed; it is
not stored in the database.

## 5. State Machine

### 5.1 Candidate enumeration

For `(state, piece)`:

1. derive the next bag by removing `piece` and refilling when empty;
2. iterate `TetrisPiecePlacement::all_from_piece(piece)`;
3. apply each placement to a local board copy;
4. discard topouts;
5. discard successors failing `is_admissible`;
6. construct the packed successor key;
7. compute `height_mse_board_score`;
8. retain the best candidate whose successor is absent or not dead.

The first read of remote successor status is only an optimization. It may be
stale. `AttachEdge` performs the authoritative check at the successor owner.

No candidate list is persisted. Within a command batch, computed candidates
may be retained in bounded scratch and reused for read-set construction and
selection, avoiding duplicate placement and scoring work.

### 5.2 Choosing a move

`choose_next(pred, piece)` returns:

- `Pending(orientation)` plus an `AttachEdge` command for the best available
  candidate; or
- `Exhausted` when every candidate is inadmissible, topout, or known dead.

The predecessor owner writes the pending slot and publishes the attachment
command in the same `WriteBatch`.

Pending is load-bearing. A move is not part of the atlas until the successor
owner has recorded the reverse edge and returned acceptance.

### 5.3 Expand

Target owner: `owner(state)`.

1. Load the authoritative node record.
2. Ignore the command if the state is dead or every forced slot is no longer
   `Unexpanded`.
3. For each forced unexpanded piece, run `choose_next`.
4. If a piece exhausts:
   - stop creating further attachments for this state;
   - call the common state-death transition in section 5.7, which writes
     `Exhausted`, enqueues exactly one `DrainDead(state)`, and detaches any
     pending or active choices already created for other pieces.
5. Otherwise write each chosen orientation as `Pending` and enqueue its
   `AttachEdge`.

An absent node at `Expand` is an invariant violation. Interning always writes
the node before publishing its expansion command in the same atomic batch.

### 5.4 AttachEdge

Target owner: `owner(succ)`.

1. Load `succ` through the batch overlay.
2. If absent:
   - insert `NodeRecord::INIT`;
   - enqueue `Expand(succ)`.
3. If `succ` is dead, enqueue a rejected `AttachResult`.
4. Otherwise:
   - insert `redge(succ, pred, piece, orientation)`;
   - enqueue an accepted `AttachResult`.

The node insert, reverse-edge insert, result command, expansion command, and
consumed-command delete are one atomic `WriteBatch`.

Because only this owner can mark `succ` dead or mutate its redge prefix, the
edge is either installed before death and included in the later drain, or
rejected after death. There is no third interleaving.

### 5.5 AttachResult

Target owner: `owner(pred)`.

Ignore the result when:

- `pred` is already dead;
- the slot is no longer `Pending(orientation)`;
- applying `(piece, orientation)` to `pred` does not derive the command's
  `succ`.

For a matching accepted result, change the slot from pending to active.

For a matching rejected result, run `choose_next` again. Either install a new
pending orientation and enqueue its attachment, or exhaust the slot and
call the common state-death transition in section 5.7.

### 5.6 ChildDead

Target owner: `owner(pred)`.

Act only when:

- `pred` is not already dead;
- its slot is `Pending(orientation)` or `Active(orientation)`;
- that placement still derives the command's dead `succ`.

Then run `choose_next` exactly as for a rejected attachment. If no candidate
remains, call the common state-death transition in section 5.7.

This guard makes command ordering safe:

- accepted result before child death: active becomes a new choice;
- child death before accepted result: pending becomes a new choice and the
  later acceptance is stale;
- duplicate child-death commands: only the first matching generation acts.

Orientation is sufficient as the generation because a dead successor is never
eligible again. The same `(pred, piece, orientation)` cannot become active a
second time after its successor dies.

### 5.7 State death

When a forced slot first becomes `Exhausted`, the owner:

1. writes the exhausted node record;
2. enqueues one `DrainDead(state)`;
3. enqueues `DetachEdge` for every other forced slot that is pending or active.

Detach commands are cleanup, not part of the death proof. A stale reverse edge
can only cause a guarded no-op notification. Detaching prevents dead parents
from accumulating indefinitely under successors that remain live.

The live-to-dead test uses the batch overlay, so two commands touching the same
state in one batch cannot enqueue two authoritative death transitions.

### 5.8 DetachEdge

Target owner: `owner(succ)`.

Delete the exact reverse-edge key. Deleting an absent edge is a no-op.

Attach and detach from the same parent owner are emitted in commit order, have
the same priority, and use increasing sequence numbers. The detach therefore
cannot be processed before its earlier attach.

If the successor is already dead, either the drain or detach removes the key.
Both outcomes are safe.

### 5.9 DrainDead

Target owner: `owner(succ)`.

`succ` must already be dead. Scan from its redge prefix until reaching either
`MAX_REDGES_PER_DRAIN` or the projected `MAX_WRITE_BATCH_BYTES` limit:

1. emit one `ChildDead` command per scanned edge;
2. range-delete exactly the contiguous scanned portion;
3. if more keys remain, enqueue another `DrainDead(succ)`;
4. consume the current drain command.

The prefix is closed because this owner rejects every later attachment to the
dead successor. Chunked range deletion is therefore safe.

The scan never materializes the whole predecessor set. Memory and
`WriteBatch` size remain bounded even for a successor with extreme in-degree.

## 6. Batching and Concurrency

### 6.1 Claim ownership

Each writer owns one `owner` byte in the work key. Only that writer scans and
deletes its producer-lane prefixes.

For each loop iteration:

1. scan priority 0 from the rotating start producer through the last producer,
   wrapping once if batch capacity remains;
2. if all are empty, inspect priority 1;
3. if all are empty, inspect priority 2;
4. stop at the command limit, then sort claimed commands by full work key;
5. during processing, stop before any candidate, edge, or byte limit would be
   exceeded;
6. leave claimed keys in RocksDB until the processing batch commits.

One owner means claims require no transaction, claim marker, or lock. Sequence
keys are append-only within `(owner, priority, producer)`, so new work cannot
land behind consumed work. The rotating start producer advances after each
batch, preventing one busy producer from starving the others while avoiding
per-lane seek amplification.

Output size is not always known at claim time. If processing the next copied
command would exceed a candidate, edge, or write-byte cap, stop before that
command. Delete and advance lane cursors only through the processed input
prefix. Copied but unprocessed commands remain in RocksDB and are claimed
again. Every individual command is configured to fit in an otherwise-empty
batch.

### 6.2 Batch overlay

Commands in one owner batch may touch the same owned state. A bounded
`BatchOverlay` is the authoritative view for the batch:

- initial records come from one batched MultiGet;
- commands are applied in work-key order;
- later commands see earlier changes through the overlay;
- each dirty node receives at most one final seven-byte `Put`.

The overlay is bounded by the command and candidate caps. It is cleared and
reused after every commit.

Reverse-edge operations retain command order. In particular, an attach
followed by a detach for the same key must become `Put` then `Delete` in the
`WriteBatch`.

### 6.3 Remote reads

A writer may read any node to build a candidate set, but only the owner may
write it. A remote read can return an older live record while the owner is
committing death. This affects only speculative candidate selection:

- the parent records the choice as pending;
- `AttachEdge` revalidates at the successor owner;
- a dead successor rejects the attachment;
- the move never becomes active without acceptance.

Correctness therefore does not require cross-owner read locks or snapshots
spanning multiple command batches.

### 6.4 Exact task accounting

The work queue uses unique `(producer, sequence)` physical identities, so an
exact `in_flight: AtomicU64` counter is valid. Sequence allocation uses
`checked_add` and rejects `u64::MAX`; exhaustion is fatal rather than wrapping
into an existing key.

`in_flight` is the only shared atomic used to prove task quiescence and uses
`SeqCst`. Stop flags, wakeup hints, and monitoring counters do not participate
in that proof. Producer sequence counters are writer-local `u64`s. The
quiescence atomic is not expected to be a throughput bottleneck because each
batch performs O(1) accounting operations, not one operation per candidate or
reverse edge.

For a batch consuming `n` processed commands and producing `m` commands:

1. allocate all output sequence numbers;
2. increment `in_flight` by `m`;
3. commit one atomic `WriteBatch` containing:
   - node and reverse-edge mutations;
   - deletion of the `n` input work keys;
   - insertion of the `m` output work keys;
4. after successful commit, decrement `in_flight` by `n`.

On commit failure, set `stop` and fail the run. A later clean resume rebuilds
exact queue and node gauges by scanning the durable `work` and `node` CFs;
historical rates and cumulative operational counters remain log-only.

The ordering prevents a false zero:

- an input command remains counted while queued or in progress;
- an output is counted before it becomes visible;
- an input is removed from the count only after its deletion commits.

`in_flight == 0` is therefore exact quiescence and implies that every work
prefix is empty. The converse can be briefly false after a consuming commit and
before its input decrement, which is conservative. No task epoch,
full-queue scan, or drift-prone monitoring counter participates in
termination.

### 6.5 Wakeups

After a successful commit, notify every owner receiving output commands.
Notifications are throughput hints only. A lost notification may delay work
until the bounded park timeout, but cannot lose a persisted command or produce
false quiescence.

### 6.6 Finite termination

For a finite carrier:

- each state is interned once by its owner;
- each forced slot has finitely many legal orientations;
- an orientation is abandoned only after its successor is dead;
- dead successors are monotone and never become eligible again;
- a slot therefore advances through finitely many choices before activating a
  surviving choice or exhausting;
- each reverse edge is attached once for a particular pending choice and is
  eventually detached or drained;
- chunked drains strictly reduce a closed reverse-edge prefix;
- stale or duplicate commands are consumed as guarded no-ops and cannot create
  a new state-machine generation.

Therefore command production is finite and exact task accounting eventually
reaches zero unless the run is stopped by error, Ctrl-C, or seed death.

## 7. Fixed-RAM Contract

The graph, reverse edges, and work queue are disk-resident. Application memory
is bounded by runtime configuration:

```text
WORK_BATCH_COMMANDS
MAX_CANDIDATES_PER_BATCH
MAX_REDGES_PER_DRAIN
MAX_WRITE_BATCH_BYTES
NUM_WRITERS
```

Initial throughput values:

```text
WORK_BATCH_COMMANDS       = 256
MAX_CANDIDATES_PER_BATCH = 256 * TetrisPiecePlacement::NUM_PLACEMENTS
MAX_REDGES_PER_DRAIN     = 4_096
MAX_WRITE_BATCH_BYTES    = 8 MiB
```

These values are starting points, not performance claims.

Each writer owns reusable bounded scratch for:

- claimed work keys and decoded commands;
- node keys and records;
- candidate records and deduplicated read keys;
- reverse-edge scan entries;
- batch overlay entries;
- encoded output commands;
- the current `WriteBatch`.

Before adding an item, code checks both the item cap and byte cap. When either
would be exceeded, the current batch ends without consuming the next input
command. No `Vec` is allowed to grow from graph in-degree or graph size.

### 7.1 RocksDB memory budget

Create mode accepts `--ram-mib`. Startup constructs:

- one shared block cache for all CFs;
- one `WriteBufferManager` limiting aggregate memtable memory;
- a bounded number of background flush/compaction jobs;
- per-writer scratch budgets checked against the requested total.

The program rejects a writer/batch configuration whose worst-case application
scratch exceeds its assigned budget.

RocksDB and its allocator have small native overhead outside the cache and
write-buffer budgets, so `--ram-mib` is an engineering budget rather than an
OS-enforced hard RSS limit. Every run reports peak RSS. Use an external process
limit when a strict operating-system cap is required.

## 8. RocksDB Access Strategy

### 8.1 Node CF

Access pattern: random point reads and owner-exclusive full-record puts.

Use:

- block-based table;
- whole-key bloom filter;
- shared block cache;
- batched MultiGet for candidate and command read sets;
- pinned get for isolated checks;
- leveled compaction unless measurements justify another style.

The node value is seven bytes and the packed key is 31 bytes.

### 8.2 Reverse-edge CF

Access pattern: inserts/deletes of exact 64-byte keys and successor-prefix
scans.

Use:

- a 31-byte fixed prefix extractor;
- prefix bloom filtering;
- raw iterators with `prefix_same_as_start`;
- bounded chunk range deletion after death;
- compaction settings chosen from measured range-tombstone and write
  amplification.

### 8.3 Work CF

Access pattern: append by producer sequence, ordered owner/priority/producer
scans, and deletion after one consumption.

Use:

- raw iterators with `fill_cache(false)`;
- small readahead;
- one range deletion for each contiguous consumed producer-lane run;
- universal compaction initially because the CF is churn-heavy;
- no bloom filter unless a later point-read use case appears.

Work range deletion is safe only inside one
`(owner, priority, producer)` lane and only through the last claimed sequence.
That producer commits serially and never publishes a lower sequence later.
The claimed run must include every existing key in the lane from its first
through last key; if a byte limit stops the scan, the last included key becomes
the exclusive deletion boundary's predecessor.
Never range-delete across a producer or priority boundary. Reverse-edge range
deletion separately follows the closed-prefix rule described above.

### 8.4 Stock crate first

Implement against the workspace's stock `rocksdb` 0.22 API.

Do not add:

- a custom rust-rocksdb fork;
- workspace-wide Clippy bans on RocksDB methods;
- allocation-count tests tied to crate internals;
- unsafe semantic overlays onto DB bytes.

Use small explicit codecs and preallocated scratch. Optimize wrapper
allocations only after profiles show they materially affect end-to-end
throughput.

### 8.5 Durability and clean resume

WAL and sync writes remain disabled for throughput. This provides a clean
pause/resume boundary, not crash recovery:

- `create` initializes an empty directory and writes `status = building`, or
  resumes an existing cleanly paused database;
- every committed batch atomically replaces its input work with its output
  work and persists that producer's next sequence;
- Ctrl-C sets a pause request and stops writers from claiming another batch;
- each writer finishes its current bounded batch before joining;
- after all writers join, flush every CF and write and flush
  `status = paused`;
- resumed `create` reopens `paused`, scans `work` and `node` to reconstruct exact
  runtime gauges, restores producer sequences, writes and flushes
  `status = building`, and starts the same worker loop;
- legacy `aborted` databases may use the maximum live work-key sequence as a
  one-time producer-sequence fallback before the recovered cursor is
  persisted.

The owner map and writer count are immutable across resume. Runtime telemetry
is not restored because it is not part of the proof state; resumed logs expose
carried-in queue counts separately from commands produced in the new process.

SIGKILL, power loss, or process failure can leave `status = building`. Such a
database is rejected because WAL-disabled writes may not all be durable. An
operator must use a prior cleanly paused database or recreate the atlas.

On successful convergence:

1. run the closure validator;
2. flush all CFs;
3. write and flush `status = complete` plus the summary metadata.

If the seed dies, write `status = seed-dead` after stopping workers. This is a
sound negative result, not an explorable atlas. Other errors write
`status = aborted`; accepting that status in `create` preserves compatibility
with older binaries whose clean Ctrl-C path used `aborted`.

## 9. Worker Loop

High-level pseudocode:

```text
seed:
    in_flight = 1
    Put node[EMPTY_BOARD, FULL_BAG] = INIT
    enqueue Expand(seed)

writer(owner):
    while not stopped:
        batch = claim_highest_priority(owner)

        if batch is empty:
            if seed is dead:
                stop with seed-dead
            else if in_flight == 0:
                stop with converged
            else:
                park briefly
            continue

        reads = prepare_bounded_reads(batch)
        records = batched_read(reads)
        overlay = records

        for command in work-key order:
            apply command to overlay
            append bounded DB mutations and output commands

        in_flight += output_count
        commit:
            node/redge mutations
            delete input work keys
            put output work keys
        in_flight -= input_count
        notify target owners
```

The logger and termination coordinator do not mutate graph state.

## 10. Closure and Output

At exact quiescence, every non-dead reachable node must have:

- no pending forced slots;
- one active orientation for every piece in its bag;
- a legal, non-topout, admissible successor for each active orientation;
- a present, non-dead successor node;
- the matching reverse edge.

The final validator scans every non-dead node and independently recomputes
these facts. It does not trust work counters or reverse-edge scans as proof.

Explore mode:

1. encode `(board, bag)` as `StateKey`;
2. read its seven-byte `NodeRecord`;
3. require the state to be non-dead;
4. decode the requested forced piece's slot as `Active(orientation)`;
5. return `TetrisPieceOrientation::from_index(orientation)`.

No candidate recomputation is needed for atlas lookup.

The surviving policy graph need not be one simple cycle. Because the carrier
is finite and every adversarial transition remains in the surviving graph,
every infinite play eventually remains in a recurrent component. If the empty
seed survives, the artifact is an M3-style reachable closed-policy witness.

## 11. Performance Plan

Correctness lands before performance claims. Then benchmark:

1. `--writers 1,2,4,8`;
2. command batches `64,256,512`;
3. candidate scratch reuse versus recomputation;
4. packed 31-byte keys versus the old 41-byte encoding;
5. compute-pool off versus bounded parallel candidate generation;
6. relevant RocksDB cache, memtable, and compaction settings.

Collect:

- states interned and expanded;
- states dead and surviving;
- reverse edges inserted, detached, and drained;
- commands by type;
- attach acceptance and rejection rates;
- stale-command no-op counts;
- commands pending and processed per owner;
- per-owner busy/idle time and load skew;
- MultiGet keys, hit rate, and latency;
- batch commands and bytes;
- node overlay hits, database gets, found keys, and missing keys;
- candidate-filter node overlay hits and database gets;
- block-cache data, index, and filter hits and misses;
- Bloom useful, full-positive, and true-positive counts;
- iterator seeks, next operations, and bytes read;
- memtable and compaction bytes;
- write amplification and `STALL_MICROS`;
- DB bytes per node and per surviving node;
- peak RSS;
- wall time, states/sec, commands/sec, and placements/sec.

Runtime telemetry is emitted as structured JSON lines on stdout. Fast
application counters are logged every three seconds and slower RSS/RocksDB
gauges every thirty seconds. A terminal JSON record is emitted for complete,
paused, seed-dead, aborted, and error exits. Telemetry is not stored in
RocksDB; the database remains a proof artifact containing only operational
schema/status/producer-cursor metadata and the exact closure summary for
completed builds.

Closure progress must not be presented as a percentage of the unknown final
reachable state set. Each progress record instead reports:

- interval `Expand` production and processing, reproduction ratio, net frontier
  growth, and a drain ETA only while the frontier is shrinking;
- interval new-node and existing-node hits plus the resulting novelty rate;
- committed nondead-state gauges: locally resolved, pending, and unexpanded;
- committed forced-slot gauges: active, pending, and unexpanded;
- interval deaths, reroutes, death-cascade fanout, and resolution velocity;
- exact known proof deficits: pending work, unresolved discovered states,
  pending attachment handshakes, and undrained death work.

Resolution gauges are derived from the before/after `NodeRecord` values already
present in the batch overlay. Writers accumulate signed deltas locally and
publish them to per-owner atomics only after a successful database commit. This
adds no database reads and no per-candidate atomics.

Checking every active successor and reverse edge requires a full graph scan.
Those two proof-deficit fields remain `null` in periodic logs; terminal
validation reports exact closure separately. Periodic telemetry must not
silently run this expensive scan.

Use a fresh database and identical carrier for each configuration. Report
hardware, OS, `rustc -Vv`, commit hash, command line, and at least five repeated
runs for performance comparisons.

Stop increasing writer count when:

- aggregate throughput no longer improves materially;
- owner load skew dominates;
- RocksDB write stalls increase;
- compaction or device bandwidth is saturated;
- peak RSS violates the configured budget.

### 11.1 Physical database sharding

Separate RocksDB instances are not part of the first implementation. They
multiply caches, memtables, background jobs, and operational complexity, and
remove cross-partition atomic batches.

Consider one DB per owner only if profiling shows that the shared database's
global write or compaction behavior is the limiting resource after tuning.
The command protocol is deliberately ownership-based so physical sharding can
be added later, but doing so requires an explicit outbox/inbox commit protocol
and a new design review.

## 12. Implementation Scope

Leave the existing `tetris_atlas_rocksdb.rs` implementation unchanged. The
new `tetris_atlas_partitioned.rs` binary does not carry forward:

- beam search and its constants/imports;
- `(board, piece)` lookup records;
- hashed lossy frontier keys;
- optimistic transactions;
- merge operators;
- cursor lists and stable-sort persistence;
- separate expansion/death queues;
- shard claim races and epoch-based quiescence;
- unbounded reverse-edge accumulation in worker scratch.

Add:

- packed 31-byte state codec;
- seven-byte `NodeRecord` and `MoveSlot`;
- virtual-shard owner map;
- `node`, `redge`, and `work` CFs;
- explicit command codecs;
- exact task sequence and in-flight accounting;
- owner-specific worker loops;
- bounded batch overlay and scratch;
- pending/active attachment handshake;
- bounded death-prefix draining;
- final independent closure validation;
- metadata and build-status checks;
- fixed-RAM configuration and metrics.

No new dependency is required by the design.

## 13. Testing

### 13.1 Unit tests

- State codec round-trips boards and bags, rejects high board bits, empty bags,
  wrong lengths, and malformed piece/orientation bytes.
- Packed state identity distinguishes every generated tiny-carrier state.
- `MoveSlot` round-trips active and pending orientations and rejects reserved
  bytes.
- `NodeRecord` correctly derives dead, resolved, and pending status from the
  bag.
- Owner hashing is deterministic; every virtual shard has one owner; shard
  counts differ by at most one.
- Candidate comparison is deterministic, score-descending, and
  orientation-ascending on ties.
- Work keys order by owner, priority, producer, then sequence.
- A delayed producer cannot publish behind another producer's cursor because
  cursors are lane-local.
- Command codecs reject wrong lengths, wrong target owner, and invalid fields.
- Batch overlay applies multiple updates to one state in command order and
  emits one final node put.
- Exact task accounting never reaches zero while an input is processing or an
  output is unpublished.

### 13.2 State-machine tests

- Attach before death: edge is installed, death drains it, parent reroutes.
- Death before attach: attach is rejected and no edge is installed.
- Child death before accepted result: pending choice advances and late
  acceptance is ignored.
- Accepted result before child death: choice activates and then advances.
- Duplicate and stale results are no-ops.
- Parent death emits detaches for active and pending edges.
- Attach followed by detach preserves sequence order and leaves no edge.
- A self-loop and a multi-state cycle survive.
- A chain to an exhausted state dies backward.
- One surviving alternative prevents OR-node death.
- Any exhausted forced piece kills the AND node.

### 13.3 Bounded-memory tests

- A synthetic successor with in-degree greater than
  `MAX_REDGES_PER_DRAIN` is processed across several drain commands.
- No command batch exceeds command, candidate, edge, or byte limits.
- Scratch capacities remain constant after warm-up.
- A rejected RAM/writer configuration fails before opening the database.

### 13.4 Integration tests

- A tiny carrier converges and passes the independent closure validator.
- A known-empty tiny carrier reports `seed-dead`.
- A fully enumerated tiny-carrier reference solver agrees on the seed verdict
  and surviving policy reachability.
- Runs with 1, 2, and 4 writers produce the same seed verdict and identical
  active orientations for every surviving state reachable under the policy.
- Every stored reverse edge has valid schema and ownership.
- `in_flight == 0` implies every work-owner prefix is empty; a deliberately
  paused post-commit/pre-decrement worker demonstrates that the conservative
  reverse implication need not hold transiently.
- A clean Ctrl-C finishes current batches, flushes every CF, and sets
  `status = paused`.
- Reopening a paused database reconstructs exact queued-work, node, closure,
  and producer-sequence state before processing another batch.
- rerunning `create` resumes paused databases and rejects databases left in
  `building` by an unclean stop.
- `explore` only inspects complete databases.
- Successful builds flush and set `status = complete`.

Do not compare the raw set of interned nodes against the eager in-memory
engine. Lazy exploration intentionally avoids unchosen alternatives. Compare
the seed verdict, independently verified closure, and the policy-reachable
surviving graph.

## 14. Open Risks

### 14.1 Shared RocksDB bottleneck

Several owner threads still share one RocksDB write path and global compaction
resources. Multiple writers improve throughput only while CPU, memtable
insertion, and storage bandwidth have capacity. The scaling benchmark decides
the useful writer count.

### 14.2 Command amplification

Cross-owner edges require attach/result commands, and death requires
child-death commands. Track commands per expanded state and attach rejection
rate. Batching and candidate quality are the primary controls; do not bypass
the handshake for speed.

### 14.3 Load imbalance

Hashing should distribute states, but work cost can still vary by shard.
Measure per-owner queue depth and busy time. The 256 virtual shards allow a
better static owner map on a future fresh run. Live migration is out of scope.

### 14.4 Stale-edge cleanup

Detach commands remove edges from parents that die for another piece. A stale
edge is correctness-safe because notifications are guarded, but excessive
stale edges waste disk and death bandwidth. Track detach hits and stale
notifications before adding more cleanup machinery.

### 14.5 Priority starvation

Death and attachment work outrank expansion. This is intentional because it
prevents expansion of states already becoming invalid. The graph is finite,
and higher-priority commands are generated only by finite lower-level work, so
expansion eventually resumes. Track time spent by priority to detect harmful
command amplification.

## 15. Rollback

The implementation remains confined to
`crates/tetris-playground/src/bin/atlas/tetris_atlas_partitioned.rs`, its
explicit Cargo binary entry, and this design document. It uses a new schema
version and a fresh database directory.

Rollback:

1. remove the partitioned RocksDB binary and its Cargo entry;
2. revert this document;
3. delete databases created with the partitioned schema.

The in-memory engine and other atlas binaries remain unchanged.
