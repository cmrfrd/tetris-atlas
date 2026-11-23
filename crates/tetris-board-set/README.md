# tetris-board-set

## Archived research snapshot — November 23, 2026

Active development of this experiment is closed. The source, tests, benchmark
fixtures and measurement tools are preserved on
[`experiment/board-set-archive-20261123`](https://github.com/cmrfrd/tetris-atlas/tree/experiment/board-set-archive-20261123).
It is intentionally unmerged and retains its original base,
`e33a40a06c2d4727432caf90b8d8392d9a55b4c9`; this archival step does not rebase it
onto current main or claim integration readiness. The retained worktree need not
be deleted to keep this snapshot. This branch contains BoardSet work, not the
separate beam-search experiment.

### Outcome and retained decisions

The original investigation considered per-column/prefix tries, alternate ordered
indexes and faster hash-based membership. The selected implementation is a compact
hash directory over independently compressed whole-board records, rather than a
prefix-sharing trie. Complete board comparison prevents false positives from hash
collisions or combinations of columns that were never inserted together.

Memory savings are the primary objective. The retained design uses adaptive byte
and nibble encoding, compact fingerprint/offset slots, append-only records,
one initialized arena tail, bounded column unrolling, and small shared decode
tables. `insert` already provides first-visit processing, so there is no separate
entry API. Sorted iteration computes temporary ordering only for export, avoiding
a permanent ordering index. No `BTreeMap`, custom allocator, mapped-memory backend,
individual deletion API, or backend selector is part of the production crate.

The final cleanup split public membership, arena encoding and hash-directory
mechanics into three files, kept unit tests beside each implementation, expanded
independent encoding/membership oracles, documented the formats, and named packed
iterator-offset constants. Hot encoder stages remain together: extracting helpers,
even with forced inlining, regressed four-column lookup by about 8.3% in historical
alternating measurements and was rejected. Directory-growth/probe alternatives and
the mapped arena prototype likewise were not promoted. Detailed retained and
rejected decisions are recorded in [PERFORMANCE.md](PERFORMANCE.md).

### Historical results and limits

These are archived measurements, not new timing results from the archival push.
They measured requested retained heap on fixed fixtures; allocator bookkeeping,
physical RSS, input buffers and temporary export scratch are excluded.

| Fixture | Distinct boards | BoardSet MiB | Standard HashSet MiB |
|---|---:|---:|---:|
| shallow1m | 1,000,000 | 10.305 | 82.000 |
| deep1m | 981,666 | 22.534 | 82.000 |
| fresh-low1m | 989,846 | 15.815 | 82.000 |
| fresh-high1m | 926,797 | 22.419 | 82.000 |
| random1m | 1,000,000 | 43.826 | 82.000 |

The observed retained-heap saving was approximately 47–87%, depending on the board
distribution. Compression costs CPU: versus the preceding codec, reserved
insertions were 22–52% slower and successful lookups about 1–38% slower in that
historical comparison. These percentages are **not** comparisons against HashSet.
Later portable arena initialization improved insertion 5.2–13.1% with reservation
and 2.3–6.8% during growth, with workload/noise qualifications in PERFORMANCE.md.
These improvements must not be added together as a cumulative speedup.

The set object was 80 bytes; million-board sorted exports needed roughly
2.8–3.8 MiB additional iterator scratch. There is no universal compression ratio,
HashSet speed advantage claim, sub-100-nanosecond guarantee, solver-throughput
claim, or all-sequences survival proof. Earlier million-board research fixtures
could be shallow, and labels alone do not establish late-game coverage.

### What is preserved and how to resume

The branch includes the complete crate (including `src/arena.rs`), workspace
registration/lockfile, the bounded unroll support in `tetris-utils`, and:

- [Operation benchmarks](../tetris-benches/src/benches/board_set.rs),
  [width benchmarks](../tetris-benches/src/benches/board_set_widths.rs), and
  [fixed-seed fixtures](../tetris-benches/src/benches/board_set_fixtures.rs).
- [Allocation measurement](../tetris-benches/examples/board_set_memory.rs) and
  [bounded timing/fixture tooling](../tetris-benches/examples/board_set_perf.rs),
  with their allocation-counter support module.
- In-file encoding/directory/set tests, plus engine and differential regression
  coverage in [tests/membership.rs](tests/membership.rs).

Generated binaries, raw benchmark outputs and large replay inputs remain local in
ignored `artifacts/`; they are not part of this GitHub source archive. Existing
local evidence directories are `allocator-quiet-20261109`,
`portable-promotion-20261109`, `human-review-20261110`, `cleanup-20261111`,
`rustdoc-audit-20261113` and `offset-consts-20261113`. The older consolidation
recovery bundle is under the main worktree's
`artifacts/board-set-consolidation-20261109/`. These paths are historical local
locations, not downloadable links or prerequisites for reading the design summary.
Build outputs removed by `cargo clean` can be rebuilt; source and recorded results
outside `target/` are separate. This archival step does not delete local evidence.

A fresh checkout can generate a bounded synthetic fixture without those archives:

```sh
mkdir -p artifacts/board-set
cargo build --release -p tetris-benches --example board_set_perf --example board_set_memory
target/release/examples/board_set_perf prepare random 100000 artifacts/board-set/random.bsmic
target/release/examples/board_set_perf run artifacts/board-set/random.bsmic trie reserved 3
target/release/examples/board_set_perf run artifacts/board-set/random.bsmic std reserved 3
target/release/examples/board_set_perf run artifacts/board-set/random.bsmic fast reserved 3
target/release/examples/board_set_memory artifacts/board-set/random.bsmic trie reserved
target/release/examples/board_set_memory artifacts/board-set/random.bsmic std reserved
```

Use a new fixture path for another preparation run; preparation rejects overwriting
an existing file. The historical backend name `trie` selects this BoardSet.
`shared` is another self-contained source, and `growing` tests unreserved builds.
Time the built executables separately from compilation; these commands reproduce
the measurement workflow, not the historical memory table's exact datasets.

To resume, check out this branch, reconcile its workspace manifests and shared
utility macro with current main, then run the validation commands below. Before
considering a merge, refresh named benchmarks against standard and fast HashSet
controls using identical bounded fixtures, and measure retained heap, build peak
and iteration scratch separately. Check growth boundaries and representative
late-game boards. Historical validation and timing do not establish readiness on
a newer engine/toolchain or main revision.

### Archival validation

The November 23 archival checks passed on the retained base:

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` and `git diff --check` | Passed |
| `cargo test --locked -p tetris-board-set -p tetris-utils` | Passed; BoardSet has 34 unit tests, 4 integration tests and 4 doctests; stress sweep excluded here |
| Strict Clippy for `tetris-board-set`, `tetris-utils`, `tetris-benches`, all targets | Passed with `-D warnings` |
| Private crate rustdocs | Passed with `RUSTDOCFLAGS=-D warnings` |
| Ignored release membership stress sweep | Passed |
| `board_set` Criterion smoke filter | All 65 registered cases passed |
| Standalone benchmark example tests | Both fixture/oracle tests passed |

No statistical benchmarks, complete workspace test suite, optional backend/GPU
checks or integration against newer main were rerun for this archive. Historical
full-workspace results remain distinguished in PERFORMANCE.md. The implementation,
tests, benchmark tooling and manifests are byte-identical to the pre-archive
working snapshot; only README/PERFORMANCE documentation was added or clarified.
Local logs are in `artifacts/board-set-archive-20261123/` and are not committed.

An exact, memory-focused set for column-major Tetris boards. The crate exposes one
collection, `BoardSet<const N: usize = 10>`, and uses the engine's current `u32`
column backing type. Every backing bit is significant, including overflow and
terminal values. Bag state is outside the key. Supported widths are `0..=256`.

```rust
use tetris_board_set::BoardSet;

let mut seen = BoardSet::with_capacity(1024);
assert!(seen.insert([0; 10]));
assert!(!seen.insert([0; 10]));
assert!(seen.contains(&[0; 10]));
assert_eq!(seen.iter().collect::<Vec<_>>(), vec![[0; 10]]);
seen.clear();
assert!(seen.is_empty());
```

The public API is `new`, `with_capacity`, `insert`, `contains`, `len`, `is_empty`,
`iter`, `clear`, and `Default`. Use `if seen.insert(board) { ... }` for first-visit
processing: it checks membership and inserts with one probe unless growth changes
the directory mapping. `new()`, `with_capacity()`, `len()` and `is_empty()` are
const-compatible. Both constructors allocate nothing: `with_capacity()` records
an initial directory hint. Allocations grow as boards arrive, and `clear()`
retains them for reuse.

## Storage and lookup

A compact hash directory holds fingerprints and arena offsets. Each unique whole
board occupies one immutable encoded record in a contiguous append-only arena.
Fingerprint matches are verified against the complete decoded board, preserving
exact membership under collisions and mixed-column queries. The common directory
slot occupies four bytes: five fingerprint bits and a 27-bit byte offset. Larger
offsets automatically promote the directory representation.

Column payloads omit leading zero bytes, or use four-bit units when that makes
the entire record smaller. Adaptive metadata describes column widths; uniform
widths and patterns with one or two exceptions use shorter headers. The encoder
retains byte payloads on ties and preserves every input bit. Records are
self-contained and do not depend on other boards sharing a prefix.

Bounded, safe word reads and overlapping writes use one initialized tail region
for the arena, rather than padding every record. The shared unroll macro
specializes column operations for `N`. Constant decode tables total a nominal
1.75 KiB per process; no lookup table is allocated per set. No unsafe code or
architecture-specific instructions are used by this crate.

Arena growth keeps 1/64 headroom and initializes the full new byte capacity once.
Appends and `clear()` reuse those initialized bytes, avoiding per-record zeroing.
Unused capacity does not participate in membership or iteration, though early
initialization can touch reserved pages sooner.

Reserved directories target 84% occupancy. Growing directories use 50% steps up
to the next power of two and rebuild before exceeding 87.5% occupancy. Rehashing
walks records in append order without a second directory or a visited bitmap.

## Iteration

`iter()` reconstructs owned boards in lexicographic order. It scans directory
slots and sorts temporary compact references, requiring O(slots + len log len)
comparisons and 3, 4, or `sizeof(usize)` scratch bytes per member, depending on the
arena extent. Each iterator owns and releases its scratch. This favors occasional
export/inspection over hot-loop iteration and avoids a retained ordering index.

## Reading the implementation

Start with [`src/lib.rs`](src/lib.rs): it contains the public `BoardSet` API,
exact probing, directory rebuilds, and temporary sorted iteration. Two private
modules carry the storage details:

| File | Responsibility | Tests beside the implementation |
|---|---|---|
| [`src/arena.rs`](src/arena.rs) | Record formats, encoding, decoding, equality, numeric ordering | Independent reference encoding, every width boundary, poisoned spare bytes |
| [`src/lookup.rs`](src/lookup.rs) | Fingerprints, slots, capacity planning, offset promotion | Probe wraparound, growth limits, all tags through promotion |
| [`src/lib.rs`](src/lib.rs) | Set operations and iteration | Membership oracles, mixed-column misses, rebuilds, clear/refill, iterator progress |

A successful insert follows `hash → locate → prepare_lookup → append → publish`.
A duplicate stops at `locate`. On a miss, directory growth can invalidate the
vacant slot, so insertion probes again only after a rebuild. A fingerprint is
always a filter; decoded whole-board equality decides membership.

`Arena::append` keeps the encoding stages together: select byte widths, try
shorter sparse headers, consider half-byte payloads, pack metadata, then reserve
initialized storage and write payloads. Replacements must strictly shorten the
record, preserving deterministic ties. Extracting these blocks into helpers was
measured and rejected because narrow-board lookups regressed; see PERFORMANCE.md.

The codec has six named header selectors plus literal three-bit widths. The first
literal width shares the selector bits and must be at most four; later half-byte
widths can reach seven. Binary and two-bit selectors store per-column widths;
the nonzero selector biases widths by one. Uniform and exception selectors describe repeated byte widths compactly.
The reference encoder in `arena::tests` intentionally keeps its own numeric
encoding instead of calling production format helpers.

Keep these invariants in view when editing:

- `arena.used` counts logical record bytes, while the full byte allocation stays
  initialized. Spare capacity is not a key and is not walked during rehashing.
- Every directory entry names one immutable record; there are no individual
  deletions. `clear()` resets membership and the logical end together.
- A zero-column board is a valid singleton with a zero-byte record. Rebuilds
  count members rather than relying only on byte offsets.
- Iteration owns temporary offsets, not a persistent ordering index. Its scratch
  allocation is separate from retained set memory.

## Validation and measurements

Unit tests live beside the implementation; integration tests exercise the engine
and differential `HashSet` oracles. Coverage includes all backing bits, independent
canonical encoding, nibble/byte boundaries, sparse headers through column 255,
poisoned tails, collision chains, growth, sorted iteration, and clear/reuse.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p tetris-board-set --release --test membership -- --ignored
cargo bench -p tetris-benches --bench benches -- board_set_operations --test
cargo bench -p tetris-benches --bench benches -- board_set_widths --test
```

Benchmarks use the workspace's ordinary module registry. Operation names remain
`board_set_operations/...` and `board_set_widths/...`; the historical `trie` label
identifies this collection. Standard and fast hash sets provide timing controls.
Allocation counting runs in the separate `board_set_memory` example, outside
statistical timing. `board_set_perf` validates replayable `BSMIC001` fixtures and
reports bounded process-level operation statistics.

[PERFORMANCE.md](PERFORMANCE.md) records the memory/CPU tradeoff, measurement limits,
and the remaining performance gate before merging. This is a review candidate,
not an Atlas policy or a proof of survival.
