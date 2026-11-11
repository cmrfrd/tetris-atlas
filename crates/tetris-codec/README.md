# Tetris board codec

`tetris-codec` maps bounded canonical Tetris boards to dense integer IDs and back, without runtime
allocation. `BoardCodec<C, ROWS, MAX_HOLES>` takes its width and piece set from `TetrisGameConfig`.
`ROWS` caps **every post-clear board in the history**; `MAX_HOLES` limits empty cells below occupied
cells in the target board. Earlier hole excursions are allowed. Transient placement cells may
exceed the height cap before clearing rows.

Except for the full-bag one-row specialization, encoded boards satisfy necessary conditions;
they are **not all proved reachable**. Encoding a board does not certify an infinite-survival policy.

## Usage and sizes

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features, long_running_const_eval)]
use tetris_codec::BoardCodec;
use tetris_game::{StandardTetris, TetrisBoard};

type Codec = BoardCodec<StandardTetris, 4, 2>;
const CODEC: Codec = Codec::new();
let board = TetrisBoard::<10, 20>::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
let id: u32 = CODEC.id(&board)?;
assert_eq!(CODEC.board_from_id(id)?, board);
assert_eq!(Codec::BOARD_COUNT, 392_431_346);
# Ok::<(), tetris_codec::Error>(())
```

| Columns | Height cap | Hole cap | States | Packed bits | Native ID | Node payload |
|---:|---:|---:|---:|---:|---|---:|
| 10 | 1 | Any | 8 | 3 | `u8` | 0 B |
| 10 | 4 | 1 | 58,149,719 | 26 | `u32` | 0 B |
| 10 | 5 | 1 | 459,948,915 | 29 | `u32` | 0 B |
| 10 | 6 | 1 | 2,551,894,523 | 32 | `u32` | 0 B |
| 10 | 4 | 2 | 392,431,346 | 29 | `u32` | 106,088 B |
| 10 | 2 | 4 | 320,974 | 19 | `u32` | 13,200 B |
| 10 | 20 | 0 | 3,219,940,489,101 | 42 | `u64` | 21,504 B |
| 6 | 3 | 2 | 26,365 | 15 | `u16` | 7,500 B |

These counts use the full seven-piece bag. Node payload excludes arithmetic constants, metadata
headers and temporary compiler workspaces. Codec handles are zero-sized.

- `id` / `board_from_id`: native column-major engine boards, with automatically selected
  `u8`, `u16`, `u32` or `u64` IDs. `BoardId<C, R, H>` and `board_id_type!(C, R, H)` name the ID type.
- `id_rows` / `rows_from_id`: bottom-first `[u16; ROWS]` row masks.
- `_u64` methods: explicit `u64` input/output; decoding still checks the range.
- `BOARD_COUNT`, `ID_BITS`, `ID_BYTES`: dense range length, packed width and native storage width.
- `table_payload_bytes()`: retained decision-node and level bytes.
- `required_bytes(bits)`: dense per-state payload size, without allocating or doing I/O.
  At eight bits per state, the file size equals `BOARD_COUNT` bytes.

Invalid boards, high row-mask bits and out-of-range IDs return errors. Mirrors remain distinct.
Numeric IDs are implementation details: changing rules or ordering requires fresh datasets or
reindexing from raw boards. There is no compatibility decoder or version switch.

## Model and implementation

The engine in `tetris-game` owns hard drops, line clearing, bag behavior and piece geometry.
The codec reads `TetrisPiece::shape_cols`, normalizes the masks and checks all placements and
rotations against the engine. Tests and fixtures live in their owning modules; ID tests reuse
the configurable game in `board.rs::tests`.

| File | Responsibility |
|---|---|
| [src/board.rs](src/board.rs) | Public API, configuration, storage, traversal and engine-based admission tests |
| [src/arithmetic.rs](src/arithmetic.rs) | Arithmetic counts, rank/select and one-row specialization |
| [src/decision.rs](src/decision.rs) | Const symbolic rules, graph reduction and packed-record construction |
| [src/id.rs](src/id.rs) | Compile-time native ID type selection and its macro |
| [src/lib.rs](src/lib.rs) | Exports and shared test fixture |

Construction runs on demand during const evaluation. There is no build script, configuration
registry, generated asset, runtime initialization or excluded-board catalog. Generic consumers
use `GameDimensions<C, R, H>: CompiledConfig` or `IdConfig`, plus `[(); C::COLS]:`.
The internal geometry descriptor and full-bag flag are derived from `C`, rather than independently
supplied public parameters.

Supported dimensions are 1–12 columns and history caps 1–28, subject to symbolic workspace/node
budgets and `u64` counts. The cap must fit `C::ROWS`, which must also be at most 28 to preserve
engine spawn headroom. `try_new()` handles constructor configuration errors; excessive symbolic
work or counts can fail const evaluation. Large hole budgets can be expensive or exceed budgets.
A fresh default consumer historically took about 2–3 minutes to compile; separate binaries may
each evaluate it. The compiler's long-running const-evaluation notice identifies a progress
threshold, not a measured runtime hotspot. Use the pinned nightly.

### Shared necessary conditions

The admitted language rejects full rows, empty internal rows, excessive target holes and occupied
cell counts incompatible with `gcd(4, C::COLS)`. Each move adds four cells and each clear removes
`C::COLS`, proving the residue invariant. Connected hard drops cannot cross an entirely empty
horizontal barrier; row clearing preserves the absence of internal empty rows.

For height caps **at least two** and widths **at least four**, a target reaching the history cap
can have at most one unsupported occupied top-row run, of length at most three. Inductively, a zero-clear
move cannot introduce another unsupported run: multi-row tetrominoes have support beneath their
top row, and a horizontal I must land on support. After clears below the cap, only one contiguous
row of the new piece can remain at the cap. At width four, an above-cap horizontal-I clear leaves
the previous board unchanged. This argument fails at width three; a legal history reaches
columns `[2,1,2]`, so the codec omits this condition below four columns.

At widths **greater than four**, a cap-height board with separated top-row runs requires a legal
**zero-clear predecessor**:

1. Four new cells cannot fill an entirely empty row above the previous height cap.
2. After any clear, surviving old cells move below the final cap-height top row. Its cells must
   therefore come from one contiguous row of the last tetromino.
3. Separated runs rule out clearing moves. Remove every translated tetromino subset, reject
   predecessors with internal empty rows, then require an exact zero-clear hard-drop replay.
4. If none succeeds, no last move exists. The board is impossible, even allowing earlier hole
   excursions and arbitrary piece order.

For example, columns `[15,0,9,0,0,0,0,0,0,0]` have separated cap-height cells. Removing their only
possible last piece, the vertical I, leaves an internal empty row. This family has 216 translated
and reflected examples at ten columns. A complete historical audit removed **1,254,120** boards
from the original default carrier: **393,685,466 → 392,431,346**, checking every admitted native
round trip and every rejection against independent engine replay. Six-column `3/2` similarly
shrinks **28,492 → 26,365**.

A predecessor alone does not prove reachability: at width four, rows `[5,5]` can receive an I that
clears itself and returns the same unreachable board. The graph fallback conservatively allows
all seven pieces, including for restricted piece sets; it does not establish bag-sensitive
seed reachability. Symbolic translations can also be a conservative superset of engine actions.

### Exact one-row arithmetic

For a full seven-piece bag, widths **8–12**, and a whole-history height cap of **one**, only empty
and the first horizontal-I placements are reachable: `C::COLS - 2` boards. Only an I fits the
first move; its four cells cannot clear such a wide row. It cannot be drawn again before the
other six pieces, and none can fit or clear a row alongside it within the cap.

Empty has ID zero; four consecutive occupied bits have ID `1 + trailing_zeros(row)`. Decoding
shifts `0b1111` back into place. The hole budget is irrelevant. There are no retained nodes.
This proof does not apply to an I-only game or a low target reached after a higher history;
regressions cover both. Independent full board/bag closure checks all five widths, every
16-bit row mask at ten columns and all native one-row boards. An unrestricted-piece-order
control reaches 41 boards instead of eight. These finite states are not a winning policy.

### Ten-column one-hole arithmetic

Caps **4–6**, with `MAX_HOLES = 1`, use column heights and optional hole metadata instead of a
graph. Solid-height words on `[a,b]`, with `N=b-a+1` symbols, even-minus-odd imbalance `D`,
length `n` and occupancy parity `p`, have count:

```text
(N^n + (-1)^p D^n) / 2
```

Powers come from a shared 352-byte const table; `D` is only -1, 0 or 1. Inclusion-exclusion
removes all-positive heights (full rows). For a hole at row `y` in height `h`, other columns
must satisfy minimum/maximum bounds to prevent full or internal empty rows. The hole column
contributes `h-1` cells to parity. Empty intervals retain a cheap early return.

A stronger impossible family is also removed: the hole column reaches `R`, with
`R-4 <= y <= R-2`; every other column is solid with height at most `y` or exactly `R`; at least
one other column reaches `R`; and all cap-height columns are nonadjacent. Separated top runs
force zero clears. The hole's upper fragment is too short and isolated to remove as a piece.
Removing other cap columns must lower them to at most `y`; removing the last would leave an
internal empty row. Moves among low columns preserve the family. Thus every valid predecessor
stays in the family while losing four cells; empty is outside it, proving unreachability.

Weighted lower/cap symbols obey `F(0)=1`, `F(1)=A+U`,
`F(n)=A*F(n-1)+A*U*F(n-2)`. Left/right intervals factor around the fixed hole column.
Subtract all-lower and zero-free words; signed weights select parity. Differentiating by `U`
counts distinguished cap positions. This gives suffix counts without board or excluded-ID tables.

IDs order zero-hole boards first, then hole-column height, hole row, hole column and remaining
column heights. Ranking indexes metadata directly; selection binary-partitions cumulative
offsets, then uses suffix counts. Eligible suffixes use direct radix/parity arithmetic.
Shared arrays hold 616 B of signed recurrence counts, 1,252 B of offsets and 112 B of column
descriptions: **1,980 B**, plus the 352 B power table. Previous graph counts at caps 4/5/6 were
58,527,051 / 463,654,889 / 2,573,076,285; current counts appear above. Four-/five-hole arithmetic
ranking remains unresolved; those configurations retain graph rules.

### Decision graph fallback

Other configurations use a reduced binary decision diagram, ordered by column then top-to-bottom
cell. Equivalent remaining languages share nodes. Each node counts valid zero-branch completions;
invalid branches have zero and receive no IDs. Encoding adds that count for a one branch;
decoding compares/subtracts the same count. Consequently `0..BOARD_COUNT` is dense and reversible.
The empty board has ID zero; graph order is column lexicographic.

Children and counts use narrow fields when lossless. Even-width residue constraints prevent
skipped live variables, removing level bookkeeping; odd widths retain levels. The default graph
has 13,261 eight-byte records (**106,088 B**), compared with 2,940,680 B for the historical
counting-plus-exclusion-index implementation. Runtime queries use scalar state and board arrays.

## Verification and benchmarks

```sh
cargo fmt --all -- --check
cargo test -p tetris-codec
cargo test -p tetris-game
cargo test --release -p tetris-codec false_exclusions -- --ignored --nocapture --test-threads=1
cargo clippy -p tetris-codec -p tetris-game --all-targets -- -D warnings
cargo run --release -p tetris-codec --example board_roundtrip
cargo bench -p tetris-benches --bench benches -- board_codec/ --save-baseline codec-before
# After editing, keep host, toolchain, features, fixtures and settings identical:
cargo bench -p tetris-benches --bench benches -- board_codec/ --baseline codec-before
cargo asm -p tetris-codec --lib --features never-inline arithmetic::rank_one_row
```

Unit tests cover independent counts, rank/select boundaries, invalid inputs, mirrors, packed
storage, shape normalization and small exhaustive languages. Three ignored correctness audits
check complete small forward closures and fixed-seed larger legal histories, allowing parents
outside the target hole budget. Historical arithmetic differential checks covered all
26,252,300 closed-family members and three million sampled IDs. These checks do not prove
reachability of every retained large-carrier board. Run the full workspace and relevant backend
checks from the repository's `AGENTS.md` before merging.

The permanent suite has **32 cases**: four operations (native/row encode/decode) over 256 fixed
raw-board witnesses for eight configurations: ten-column `1/0`, `4/1`, `5/1`, `6/1`, `4/2`, `2/4`,
`20/0`, plus six-column `3/2`. Fixtures are independent of codec ID ordering, checked before
timing, and print hashes. Setup/allocation is excluded. There are no research/large-codec flags.
The historical shared `4/2` fixture remains fixed. Renamed or changed fixtures need fresh baselines.
`never-inline` is for assembly inspection, not normal measurements.

Historical measurements below are separate experiments on native ARM64 M1 Pro with the pinned
nightly. They are not fresh measurements of this checkout, and their percentages must not be
added together. Exact dirty sources, executables, commands, seeds, host/power settings, samples
and limitations are in the local gitignored artifact directories listed below.

| Experiment | Measured result | Scope |
|---|---|---|
| Packed graph vs excluded-ID index | Native encode 929 → 66.7 ns; decode 1,102 → 98.0 ns | Fixed shared default fixtures; N=2 trials/variant, 40 Criterion samples/trial; background-load caveats |
| Same graph engine replay | 1.2886 → 0.1823 s median, 7.07× throughput | N=4/variant; identical 2,735,744 placements, 532,608 round trips and output checksum |
| One-row specialization | Native encode/decode −86.65% / −81.48%; row −95.04% / −96.60% | Eight reachable boards repeated to 256; two trials/variant, 30 samples/trial |
| Power-table/parity leaf | Encode −29–34%; decode −35–41% across caps 4–6 | Public native/row cases; two trials, 40 samples/trial; unchanged controls −3.93% to +0.79% |
| Integrated arithmetic caches | Encode −12–21%; decode −0.2–5.1% across caps 4–6 | Full 32-case ABBA, two trials/variant, 60 samples/trial; some decode intervals include zero |
| Sampler parallelism | Serial 4.463 s → four-worker 1.896 s median | 20 batches including persistence; N=4/5; one-worker cost did not improve; identical logical outputs |

The integrated-cache experiment reports no Criterion regressions across 32 cases; old-executable
control drift was −2.44% to +1.39%. Shared optimized `select` shrank from 3,534 to 1,778 static
instruction sites, while rank's branch sites increased: performance is measured, not inferred
from branchlessness. The sampler comparison used about 196 MiB RSS and more CPU resources.
These are fixture/bounded-workload results, not general gameplay, survival or move-p95 claims.

Local evidence under `artifacts/output/`:

- `codec-reachability-analysis/`, `codec-playground-tool/`: original discovery, witnesses,
  replay and sampler scaling; old carrier counts and IDs are historical.
- `codec-rule-bdd/`, `codec-validation-20261101/`: complete default sweep, graph benchmarks,
  domain corrections and validation. The stronger 380,842,218-state prototype remains research.
- `codec-10-column-sweep-20261103/`: arithmetic derivation, exhaustive family checks and old
  packed-kernel tradeoffs (caps four/five initially slower; cap six faster).
- `codec-reachability-refine-20261109/`, `codec-leaf-perf-20261109T220516Z/`: one-row and leaf trials.
- `codec-combinations-20261110T014016Z/`, `codec-integrated-20261110T033902Z/`: ablations,
  combined measurements, assembly and workspace checks for their recorded source snapshots.
- `codec-shared-geometry-20261111T182724Z/`: all 28 const shapes, focused validation, 32 fixture
  checks and byte-identical 309,140-byte read-only constant sections after sharing engine geometry.

## Reachability sampler

The [playground sampler](../tetris-playground/src/bin/examples/codec_reachability/main.rs) collects
replayable evidence using the canonical engine; it does not automatically change codec membership.
Build once, then run from the repository root:

```sh
cargo build --release -p tetris-playground --bin tetris_codec_reachability
# Bounded discovery run; --controls adds 216 known impossible examples:
target/release/tetris_codec_reachability run \
  --output artifacts/data/codec-study --batches 100 --threads 4 --controls
# Without --batches, run until Ctrl-C; finish the batch and checkpoint:
target/release/tetris_codec_reachability run \
  --output artifacts/data/codec-continuous
target/release/tetris_codec_reachability run \
  --output artifacts/data/codec-continuous --resume
target/release/tetris_codec_reachability verify --output artifacts/data/codec-continuous
```

Stop the writer before verification. Defaults: seed `0xb04dc0de`, 10,000 uniform parents with all
placements, 1,000 episodes per play mode, 10,000 candidates per batch, horizon 128, up to four
saved random paths per mode, 10,000 watched targets, 60-second checkpoints and up to four workers.
`--batches` counts additional batches per invocation; use `run --help` for other options.
Changing `Game`, `ROWS` or `MAX_HOLES` in `state.rs` selects another codec. The proof checker
supports widths 5–10 with the seven standard pieces. Default `4/2` uses **187.1 MiB** for four
shared exact bitmaps, plus journal/path storage; `--max-bitmap-mib` defaults to 512.

| Observation | Meaning |
|---|---|
| Incoming hit | A sampled parent has an edge; its reachability and bag history are unknown. |
| Strict witness | Legal seven-bag play from empty within height/hole caps and current codec. |
| Relaxed witness | Legal play within the height cap, with possible earlier hole excursions. |
| Proved impossible | Exhaustive last-move checking excludes the board under the stated model. |
| No observation | Unknown; sampling absence cannot justify exclusion. |

The default codec already excludes the sampler's last-move family. Further compression needs a
new sound rule. The original million-parent study witnessed 639,472 boards but left most targets
unresolved: 158 of 165 witnessed uniform targets had **no incoming hit**. Coverage of the true
reachable set remains unknown. Disconnected cycles and arbitrary incoming edges are insufficient.

Each dataset has `manifest.json` (model, workload, source/Git provenance), `findings.jsonl` (raw
boards and replay actions/bags), `candidates.csv` (watchlist flags), `checkpoint.bin` (checksummed
RNG/counters/bitmaps and durable journal offset) and `summary.json` (totals/invocation cost).
Parallel workers share bitmaps; streams and reductions preserve deterministic results.

Resume requires matching source fingerprints, physical model and workload. Thread count,
batch/time limits and checkpoint interval may change; repeat the remaining options. Keep old
binaries for old datasets. Codec ID/order changes require a fresh dataset; `verify` can replay
old raw findings if the engine/model still match, and strict paths also check current admission.
After a reviewed engine refactor, `verify --recheck-engine` explicitly replays against the new
engine; it never relaxes resume guards. Source-only edits to fingerprinted files also invalidate
resume. After a crash, the extra journal tail is preserved as `uncheckpointed-*.jsonl.part`, then
rolled back and deterministically resampled. Corrupt/shortened data errors instead of resetting.
Fingerprints/checksums detect accidental mismatch; they are not authentication.
