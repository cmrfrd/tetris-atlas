# Tetris board codec

`tetris-codec` maps admissible `TetrisBoard<C>` values to dense integer IDs and back. The game
configuration supplies `C::COLS`; `ROWS` bounds every post-clear board in the considered history,
and `MAX_HOLES` bounds empty cells below occupied cells in the encoded board.

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features, long_running_const_eval)]
use tetris_codec::BoardCodec;
use tetris_game::{StandardTetris, TetrisBoard};

type Codec = BoardCodec<StandardTetris, 4, 2>;
const CODEC: Codec = Codec::new();
const COUNT: u64 = Codec::BOARD_COUNT; // 392_431_346
let board = TetrisBoard::<StandardTetris>::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
let id: u64 = CODEC.id_u64(&board)?;
assert_eq!(CODEC.board_from_u64(id)?, board);
# Ok::<(), tetris_codec::Error>(())
```

`id()` and `id_rows()` instead return the smallest supported integer type for these parameters:
`u16`, `u32`, or `u64`. `BoardId<C, ROWS, MAX_HOLES>` and `board_id_type!(C, ROWS, MAX_HOLES)` name
that type. `ID_BITS` gives packed width; `ID_BYTES` gives native storage width. Defaults need
29 packed bits and a `u32`. `rows_from_id`/`id_rows` use bottom-first `[u16; ROWS]` row masks;
native boards remain the engine's column-major arrays. `_u64` entry points accept/return `u64`
and range-check decoding. Inputs outside the row masks, caps, or admitted language are rejected.

## Intrinsic counting and ranking

There is **no excluded-board catalog or excluded-ID table**. The compiler constructs a reduced
binary decision diagram from the rules, using columns left to right and cells top to bottom.
Every node stores the number of valid completions on its zero branch. Invalid branches have
zero completions and receive no IDs. Equivalent remaining languages share nodes.

Encoding adds the zero-branch count when taking a one branch. Decoding compares the ID with
that count and follows the corresponding branch. They use the same compiled structure, so
`0..BOARD_COUNT` is dense and every ID decodes to exactly one admitted board. Exact mirrors
remain distinct; ID order is column lexicographic, with the empty board at zero.

The admitted language enforces:

- No uncleared full rows or empty internal rows.
- Cell-count residue modulo `gcd(4, C::COLS)`.
- The requested hole budget, clamped to its physical maximum.
- The capped-history top-support condition at widths >=4. Three-column clears above the cap
  can leave multiple unsupported runs, so the condition must not be applied there.
- The proved last-move condition: at the cap, separated top runs require a legal zero-clear
  predecessor. If none exists, the board cannot be encoded. This proof applies at widths >4
  and caps <=28; [EXCLUSIONS.md](EXCLUSIONS.md) gives the argument and exact scope.

These are necessary conditions, not the exact seed-reachable set or an infinite-survival
certificate. The diagram currently uses all seven standard tetrominoes; types with equal width
and caps share it even if their piece sets differ. Symbolic translations span the requested width;
an engine action inventory with fewer placements makes this a conservative superset. Bag
obligations are not encoded.

## Construction, storage, and limits

Construction runs automatically during const evaluation, on demand for the parameters used.
There is no `build.rs`, registration list, asset generation, runtime initialization, allocation,
or lazy cache. The engine exposes its existing piece masks through a const accessor; the codec
uses those masks directly. New rules change the language/counts without generating exclusion files.

Runtime records use narrow children and completion counts when lossless. Even-width residue
constraints make every cell significant, allowing traversal without per-node level bookkeeping;
odd widths retain levels for skipped/free variables. The default diagram uses **106,088 bytes**
of retained record payload. The codec itself remains one pointer. `table_payload_bytes()` excludes
small metadata headers and temporary compiler workspaces. `required_bytes(bits)` reports a dense
per-board payload size and performs no I/O.

The codec supports 1–12 columns and history caps of 1–28 rows, subject to bounded symbolic
node/workspace and `u64` count budgets. The cap must fit the game's playable rows, which must be
below 32; `try_new()` handles constructor configuration errors. Higher history caps are unsupported:
the engine's top-aligned spawn masks can overlap existing cells there, invalidating even the
cell-count residue rule. A legal 31-row seven-bag counterexample is retained in the unit tests.
Very large diagrams fail const evaluation with a budget error. Fresh compilation of a
consuming target with a nontrivial configuration is expensive (about two minutes for the default
in this experiment); separate binaries may each evaluate it. Normal queries do none of that work.
Use the pinned nightly and allow the compiler's long-running const-evaluation progress lint in
consumers.

Generic consumers use `GameDimensions<C, R, H>: CompiledConfig` or `IdConfig`, plus
`[(); C::COLS]:`. Automatic integer selection stays entirely within this crate.

IDs are implementation details. Rule/order changes require fresh numeric-ID datasets or explicit
reindexing from raw boards. The sampler refuses incompatible resume and can replay saved raw facts.
There is no historical-format decoder or codec version switch.

## Checks and measurements

```sh
cargo test -p tetris-codec
cargo test -p tetris-game
cargo run --release -p tetris-codec --example board_roundtrip
cargo clippy -p tetris-codec --all-targets -- -D warnings
cargo bench -p tetris-benches --bench benches --no-run
# Run the executable Cargo prints; compare fixed raw-board fixtures:
target/release/deps/benches-<hash> --bench board_codec/standard_4_2_reachable_shared
```

The six-hole benchmark retains its name and is opt-in with `large-codec-benchmarks`, since
symbolic construction for that larger language is costly. The codec itself needs no such feature.
See [PERFORMANCE.md](PERFORMANCE.md) for before/after evidence, compilation cost, and limitations.
The older reachability investigation remains in [REACHABILITY.md](REACHABILITY.md).
