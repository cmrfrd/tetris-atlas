# `tetris_success_set_solver`

Searches for a closed winning set in the canonical no-preview, no-hold 7-bag game, subject to
configured board-admissibility bounds. The current CLI runs the proof-graph implementation in
`proof.rs`; the older recovery-to-empty retrograde implementation is retained under `cfg(test)`.

## State and obligations

A bag node is an exact `(board, remaining_bag)` state. It must satisfy every remaining-piece
obligation; a piece node needs at least one non-losing placement whose successor stays in the
winning set. Removing the last bag piece refills the membership mask with all seven pieces.

The solver uses `TetrisBoard<10, 20>`, `TetrisPieceBagState`, and
`TetrisPiecePlacement<10>` directly. Actions come from `all_from_piece(piece)` and transitions
use `apply_piece_placement`. It does not sample or shuffle a `TetrisGame` stream during graph
construction.

The default root is the empty board with a full bag. `--root-board-rows` and `--root-bag-mask`
can select another root; a result for that root requires a separate reachability argument before
it establishes survival from a new canonical game.

## Solve modes

`--solve-mode` accepts `full`, `core-first` (the CLI default), `template-kernel`, and `optimistic`.
The latter modes change how candidate cores, templates, or repairs are explored. A partial
candidate or a budget-limited probe is not certified merely because its unresolved frontier
has no observed death.

The root conclusion is `yes`, `no`, or `unresolved`. A completed negative result applies to the
configured admissible carrier, not automatically to the unrestricted canonical game.
`unresolved` must remain distinct from either conclusion.

## Verification and export

For resolved conclusions, the CLI runs both `verify_proof` and `verify_replay` and rejects
verification failures. The graph checker validates witnesses and resolutions; the replay checker
recomputes concrete placements and bag transitions with the engine. Unknown results use no
completed-certificate claim.

`--export-path` selects a summary JSON path; a proof-graph artifact is written beside it. Preserve
the configuration, root, source revision, verification output, and artifacts together. A positive
result at the empty/full-bag seed must include verified closure over every legal next piece.

## Modules

| File | Responsibility |
|---|---|
| `main.rs` | CLI, result reporting, verification orchestration, exports |
| `config.rs` | Board admissibility and root-state configuration |
| `admissibility.rs` | Concrete admissibility checks |
| `state.rs` | State keys, bag transitions, packed placement helpers |
| `proof.rs` | Current proof-graph search, solve modes, witness and resolution checks |
| `verifier.rs` | Concrete engine replay verification |
| `graph.rs`, `universe.rs`, `retrograde.rs`, `policy.rs` | Test-only recovery solver and reference infrastructure |

## Usage

```sh
cargo run -p tetris-playground --bin tetris_success_set_solver -- --help

mkdir -p artifacts/output/success-set
cargo run --release -p tetris-playground --bin tetris_success_set_solver -- \
  --max-height 2 --max-holes 0 --solve-mode full \
  --export-path artifacts/output/success-set/summary.json

cargo test -p tetris-playground --bin tetris_success_set_solver
```

These commands illustrate the interface; they are not recorded proof results.
