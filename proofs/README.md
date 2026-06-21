# Tetris Proofs

A Lean 4 formalization of canonical Tetris, building toward a formal proof that
the game is survivable for infinite horizons. See the design at
`docs/superpowers/specs/2026-05-29-tetris-proofs-infrastructure-design.md`.

The canonical rules are mirrored from the Rust reference engine in
`crates/tetris-game` (10×20 board, 7-bag randomizer, hard-drop-only).

## Building

Uses Lean (pinned in `lean-toolchain`) and mathlib via lake.

```sh
cd proofs
lake exe cache get   # download prebuilt mathlib (first time only)
lake build
```

## Module layout

| Module | Contents |
|---|---|
| `Proofs.Config` | `GameConfig` (board dimensions); `standard` = 10×20 |
| `Proofs.Piece` | `Piece`, `Rotation`, `shape` and bottom-up `shapeUp` for all 7 tetrominoes |
| `Proofs.Board` | `Board` as a `Finset` of filled cells; `count`, `colHeight`, `isLost`, `isFull`, `clearLines` (gravity), `WF` |
| `Proofs.Placement` | hard-drop placement: `dropOffset`, `dropped`, `place`, `applyStep` |
| `Proofs.Bag` | the 7-bag randomizer: `full`, `canDraw`, `draw` |
| `Proofs.Game` | `GameState`, `step`, `lost`, in-bounds `Valid`, inductive `Reachable` |
| `Proofs.Theorems.PieceGeometry` | every piece/rotation has exactly 4 cells |
| `Proofs.Theorems.BoardCount` | empty/set/clear/full-row cell-count facts |
| `Proofs.Theorems.StepInvariants` | loss basics, well-formedness, line-clear counts, reachable-parity |
| `Proofs.Theorems.Gameplay` | line-clearing correctness and loss characterization |
| `Proofs.Theorems.GameplayExtra` | line-clear algebra, hard-drop maximality, 7-bag & reachability invariants, piece geometry |
| `Proofs.Experiments.FiveBagReset` | five-bag reset arithmetic, backward winning layers, online solver extraction, and finite-search solvability reduction |

## Key theorems

- `Placement.count_place` — hard-dropping a piece adds exactly 4 filled cells.
- `Board.clearLines_count_add` — a line clear removes a multiple of `cols` cells.
- `Board.clearLines_no_full` — after clearing, no row is full (every complete
  line is actually removed).
- `Board.clearLines_count_le`, `clearLines_id_of_no_full`, `clearLines_empty` —
  clearing never adds cells, is a no-op when nothing is full, fixes the empty board.
- `Board.isLost_iff` — a board is lost iff some column's height exceeds `rows`;
  `not_isLost_of_forall_lt`, `not_isLost_empty`, `isLost_mono` — loss basics.
- `reachable_even_count` / `reachable_even_count_standard` — **every reachable
  board (on the standard 10-wide board) has an even number of filled cells.**

## More gameplay theorems (`GameplayExtra`)

`clearLines_idem` (clearing is idempotent), `linesCleared_le`,
`linesCleared_place_le_four` (a hard drop clears ≤ 4 lines), `colHeight_mono`,
`subset_place`, `dropOffset_empty`, `dropped_resting` (hard-drop maximality),
`applyStep_count`, `isLost_or_bounded`, the 7-bag invariants (`canDraw_full`,
`draw_card`, `not_canDraw_after_draw`), the reachability invariants
(`reachable_bag_nonempty`, `reachable_bag_subset`, `reachable_no_full`,
`reachable_count_mod`), and piece geometry (`shape_lt_four`, `shapeUp_cols`,
`shape_O_rotation_invariant`). All proven.

## Representation

Boards and pieces use a clean mathematical model (`Finset` of `(col, row)`
coordinates over `ℕ`) rather than the Rust bit-column layout, to keep
counting/geometry proofs natural. A bit-column representation with a proven
equivalence can be added later if `decide`-style computation is needed.

## Next steps (not yet implemented)

- Survivability: define a policy and prove non-losing play over infinite
  horizons (the project's ultimate goal).
- Five-bag reset experiment: construct or machine-check a `Certificate` proving
  that one online solver survives every legal 35-piece branch and clears
  exactly 14 lines.
- Connect the Lean model to the Rust engine (extraction or a proven bridge).
