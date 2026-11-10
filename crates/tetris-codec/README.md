# Tetris board codec

`tetris-codec` maps admissible `TetrisBoard<{ C::COLS }, { C::ROWS }>` values to dense integer IDs and back. The game
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
let board = TetrisBoard::<10, 20>::from_limbs([3, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
let id: u64 = CODEC.id_u64(&board)?;
assert_eq!(CODEC.board_from_u64(id)?, board);
# Ok::<(), tetris_codec::Error>(())
```

`id()` and `id_rows()` instead return the smallest supported integer type for these parameters:
`u8`, `u16`, `u32`, or `u64`. `BoardId<C, ROWS, MAX_HOLES>` and `board_id_type!(C, ROWS, MAX_HOLES)` name
that type. `ID_BITS` gives packed width; `ID_BYTES` gives native storage width. Defaults need
29 packed bits and a `u32`. `rows_from_id`/`id_rows` use bottom-first `[u16; ROWS]` row masks;
native boards remain the engine's column-major arrays. `_u64` entry points accept/return `u64`
and range-check decoding. Inputs outside the row masks, caps, or admitted language are rejected.

For ten columns, a one-row cap and the full seven-piece bag, the codec is **exactly seed-reachable**:
only empty and the seven first-horizontal-I boards are encoded. This is eight states, three packed
bits, a native `u8`, and zero lookup-node payload (previously 134 states/eight bits/`u16`/426 bytes
of node payload). A second I cannot occur until the other six pieces have played; none can stay
inside that history cap. [The proof and exhaustive engine check](REACHABLE.md) cover the full
one-row board universe. This rule is not applied to restricted piece sets, where two consecutive
I pieces can be legal. `GameDimensions` derives its full-bag policy from `C::PIECE_SET` automatically.

## Intrinsic counting and ranking

Ten-column codecs with height caps 4–6 and at most one hole (`MAX_HOLES = 1`) use
[algebraic counting and ranking](ARITHMETIC.md), with no retained lookup nodes. Their counts
are 58,149,719 / 459,948,915 / 2,551,894,523 (26 / 29 / 32 packed bits, all `u32`).
They also exclude a proved, recursively closed impossible family. IDs order zero-hole boards
first, then hole metadata and column heights. Older numeric datasets require reindexing;
there is no compatibility decoder.

Other configurations use the decision graph described below. There is **no excluded-board catalog or excluded-ID table**. The compiler constructs a reduced
binary decision diagram from the rules, using columns left to right and cells top to bottom.
Every node stores the number of valid completions on its zero branch. Invalid branches have
zero completions and receive no IDs. Equivalent remaining languages share nodes.

Encoding adds the zero-branch count when taking a one branch. Decoding compares the ID with
that count and follows the corresponding branch. They use the same compiled structure, so
`0..BOARD_COUNT` is dense and every ID decodes to exactly one admitted board. Exact mirrors
remain distinct; graph ID order is column lexicographic, with the empty board at zero.

The admitted language enforces:

- No uncleared full rows or empty internal rows.
- Cell-count residue modulo `gcd(4, C::COLS)`.
- The requested hole budget, clamped to its physical maximum.
- The capped-history top-support condition at widths >=4. Three-column clears above the cap
  can leave multiple unsupported runs, so the condition must not be applied there.
- The proved last-move condition: at the cap, separated top runs require a legal zero-clear
  predecessor. If none exists, the board cannot be encoded. This proof applies at widths >4
  and caps <=28; [EXCLUSIONS.md](EXCLUSIONS.md) gives the argument and exact scope.

Outside the exact one-row specialization, these are necessary conditions, not the exact
seed-reachable set. No specialization is an infinite-survival certificate. The graph fallback uses all seven standard tetrominoes conservatively, even for restricted
piece sets. The configured mask selects validated game-specific specializations. Symbolic translations span the requested width;
an engine action inventory with fewer placements makes this a conservative superset. Bag ordering is enforced by the proved one-row specialization; other carriers do not yet
encode the complete bag-sensitive reachable set.

## Construction, storage, and limits

Construction runs automatically during const evaluation, on demand for the parameters used.
There is no `build.rs`, registration list, asset generation, runtime initialization, allocation,
or lazy cache. The codec owns the normalized symbolic footprints it needs for const construction.
Tests compare every canonical placement and symmetric rotation with the public engine placement API;
the engine crate has no codec-specific changes. New rules change the language/counts without generating exclusion files.

Runtime records use narrow children and completion counts when lossless. Even-width residue
constraints make every cell significant, allowing traversal without per-node level bookkeeping;
odd widths retain levels for skipped/free variables. The default diagram uses **106,088 bytes**
of retained record payload. Codec values are zero-sized; their type supplies the shared tables
through an associated constant. `table_payload_bytes()` reports node and level payload, excluding
small metadata headers, arithmetic constant arrays and temporary compiler workspaces. The arithmetic
path shares 1,980 bytes of recurrence counts, metadata offsets and column descriptions, plus its
existing 352-byte power table. `required_bytes(bits)` reports a dense per-board payload size and
performs no I/O.

The codec supports 1–12 columns and history caps of 1–28 rows, subject to bounded symbolic
node/workspace and `u64` count budgets. The cap must fit the game's playable rows, which must be
at most 28; `try_new()` handles constructor configuration errors. The current engine rejects
gameplay above 28 rows before placement, reserving space for spawning pieces. The historical
31-row spawn-overlap board remains an invalid-input regression fixture.
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

The permanent suite runs the same eight configurations without codec benchmark feature flags:
ten columns at height/hole caps `1/0`, `4/1`, `5/1`, `6/1`, `4/2`, `2/4`, and `20/0`, plus six
columns at `3/2`. Every case times native/row encoding and decoding with 256 fixed raw-board
witnesses. Inputs are independent of codec admission and ID numbering. The shared historical
`4/2` fixture remains unchanged; renamed cases and the new fixture definitions use fresh baselines.
The bounded `2/4` graph covers a larger hole budget without the excessive cold compilation cost
of a ten-column `4/4` graph. There are no external benchmark table files or variant selectors.

```sh
cargo bench -p tetris-benches --bench benches -- board_codec/ --save-baseline codec-before
# After a change, with identical fixtures, toolchain, features and host settings:
cargo bench -p tetris-benches --bench benches -- board_codec/ --baseline codec-before
# Inspect emitted kernels with cargo-show-asm; never-inline is for inspection only:
cargo asm -p tetris-codec --lib --features never-inline single_row::rank
cargo asm -p tetris-codec --lib --features never-inline arithmetic::rank
```

The public type is `BoardCodec<C, HEIGHT, MAX_HOLES>`; it does not duplicate `C::COLS` or the piece
set as independently supplied parameters. A numeric geometry descriptor and derived `FULL_BAG`
boolean let the compiler select rule storage and the native ID type. Bag values use
`TetrisPieceBagState`; that engine type cannot itself be a const parameter without a
`ConstParamTy` implementation, which would require an engine change. Test algorithms live beside
their owning module's tests. Only the small configurable `Game` fixture is shared.

See [PERFORMANCE.md](PERFORMANCE.md) for before/after evidence, compilation cost, and limitations.
The older reachability investigation remains in [REACHABILITY.md](REACHABILITY.md).
