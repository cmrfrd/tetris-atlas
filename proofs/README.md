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
lake exe cache get            # download prebuilt mathlib (first time only)
lake build                    # the green standard library (base-axiom-clean)
lake build ProofsExperiments  # the research routes (may use native_decide)
```

The build is split into two lake libraries: the default `Proofs` target is the
curated standard library (no `sorry`, no `native_decide`, only the base axioms
`propext`/`Classical.choice`/`Quot.sound`); `ProofsExperiments` holds the active and
floored research routes. See `LIBRARY.md` for the full module map and curated theorem
spine, and `ROADMAP.md` for the proof strategy.

## Module layout

The library is organized bottom-up by dependency layer (`LIBRARY.md` has the full
module map and the curated theorem spine):

| Layer (`Proofs/…`) | Contents |
|---|---|
| `Model/` | the game model: `Config`, `Piece`, `Board`, `Placement`, `Bag`, `Game` |
| `Combinatorics/` | piece geometry, board/column cell-counting, 7-bag renewal (`BagBurst`) |
| `Invariants/` | reachability/WF/line-clear invariants (`StepInvariants`, `Gameplay`), `Holes`, hole-debt (`HoleDebt`), surface fiber (`SurfaceFiber`), WQO monotonicity (`Wqo`), finite state space (`StateSpace`) |
| `Survival/` | `Policy`, `trace`, `SurvivesForever`, `safe_invariant`, `ClosedCycle` |
| `Safety/` | the safe-set GFP, `Atlas`, the `safe_extract` solvability reduction, computable safe-set iteration (`SafeIterateFinite`), pigeonhole obstructions (`Safety`) |
| `Experiments/` | active + floored research routes (the separate `ProofsExperiments` lake target; may use `native_decide`) |

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

## Status & the open question

The survivability **framework is built and base-axiom-clean**. The goal
`TetrisSolvable` is reduced — with proof — to a single membership claim,
`GameState.init ∈ safe GameConfig.standard`, characterized equivalently as a greatest
fixed point, a reachable closed cycle, and a nonempty closed `Atlas` (`safe_extract`
plus the M2/M3/M4 equivalences in `Safety/`). The survival vocabulary (`Policy`,
`SurvivesForever`, `ClosedCycle`, `safe_invariant`) and the computable safe-set
iteration (`safeIterFinite`, `decideSafeFromUniverse`) are all in place.

What remains **open** is that one membership claim — the *witness*, not the framework.
See `ROADMAP.md` for the attack plan and `LIBRARY.md` §5 for the single open crux
(the I-drain regulator geometry).

Still genuinely not done:
- A concrete `AdversarialClosedCycle` / closed-`Atlas` witness (even on a degenerate config).
- Connecting the Lean model to the Rust engine (extraction or a proven bridge).
