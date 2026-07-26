# Tetris Proofs — Infrastructure Design

Date: 2026-05-29
Status: approved (design), scaffolding in progress

## Goal

Establish a `proofs/` top-level directory containing a **Lean 4** formalization of
canonical Tetris. The long-term mission is to formally prove that Tetris is
survivable for infinite horizons (mirroring the Atlas mission in `AGENTS.md`),
built up from a ladder of base-level theorems.

This document covers the **initial infrastructure** only: the Lean package
skeleton, the formal game model, and a first set of basic theorems. Where the
proof effort goes next is deliberately out of scope and will be decided later.

## High-level decisions (from the user)

1. Formalize the entire system in **Lean 4**, as a proper Lean (lake) package.
2. Provide a **formal representation of the game**: standard 10×20 Tetris, the
   standard 7-piece bag randomizer, **hard drop only**. The Rust engine in
   `crates/tetris-game/src/tetris.rs` is the reference for the rules, not a
   spec to reimplement bit-for-bit.
3. Prove a set of **basic theorems** about the game/system first (e.g. every
   reachable board has an even number of filled cells), then revisit direction.

## Game model (canonical rules, from `tetris-game`)

- **Board**: `COLS = 10`, `ROWS = 20`. Each cell is filled or empty.
- **Loss**: a board is lost when any filled cell sits above row `ROWS-1`.
- **Pieces**: 7 tetrominoes (O, I, S, Z, T, L, J), each exactly 4 cells.
  Distinct rotations: O=1, I/S/Z=2, T/L/J=4.
- **Action (hard drop only)**: a placement is `(piece, rotation, column)`. The
  piece falls straight down until it rests on the stack/floor, then merges.
- **Line clear**: any fully-filled row is removed; rows above shift down.
- **Bag**: standard 7-bag — draw without replacement from a bag of all 7 pieces;
  refill when empty.

## Representation choice

**Clean mathematical model** (chosen over mirroring the Rust bit-columns):

- Board: `Fin COLS → Fin ROWS → Bool` (or a `Finset` of occupied coordinates).
- Piece: a finite set of 4 offset cells per rotation.

Rationale: counting / parity / geometry theorems are natural with mathlib's
`Finset`/`Fintype` machinery. The Rust `[u32; 10]` bit-column representation is
fast but awkward to reason about; if `decide`-style computation is ever needed,
a bit-column representation can be added later as a *separate, proven-equivalent*
model. Proof-first project ⇒ math model first.

## Package layout

A standalone lake package under `proofs/`, depending on **mathlib**:

```
proofs/
├── lean-toolchain          # pins Lean version (matched to mathlib)
├── lakefile.toml           # package + mathlib dependency
├── TetrisProofs.lean       # root: imports all modules
└── TetrisProofs/
    ├── Config.lean         # GameConfig (ROWS, COLS); Standard = 20×10
    ├── Piece.lean          # Piece, Rotation, 4-cell offset shapes for all 7
    ├── Board.lean          # Board rep; count, set/clear, full-row
    ├── Placement.lean      # placement = (piece,rot,col); hard-drop landing + merge
    ├── Bag.lean            # 7-bag state and draw/refill
    ├── Game.lean           # GameState, step, is_lost, reachability
    └── Theorems/
        ├── PieceGeometry.lean
        ├── BoardCount.lean
        └── StepInvariants.lean
```

## Starter theorems (graded easy → meaty)

1. **Piece geometry**: every piece, in every rotation, occupies exactly 4 cells;
   the rotation count is 1/2/4 as specified.
2. **Board basics**: empty board has 0 filled cells; set/clear changes the count
   by ±1; a full row has exactly `COLS` filled cells.
3. **Step invariants**: a placement adds exactly 4 cells before clears; a line
   clear removes a multiple of `COLS` cells; therefore **every reachable board's
   filled-cell count is even** (since `gcd(4, COLS=10) = 2`).
4. **Loss basics**: the empty board is not lost; `is_lost` is monotone in height.

## Scope of the first scaffold

"Basic infrastructure" deliverable:

- Lake package that builds with mathlib (via `lake exe cache get`).
- `Config`, `Piece`, `Board` core definitions.
- A first batch of proved theorems from groups (1) and (2) above to prove the
  toolchain + model work end to end.

`Placement` / `Bag` / `Game` step semantics and the reachable-parity theorem
(group 3) and loss basics (group 4) follow once the foundation builds clean.

## Out of scope (for now)

- Any infinite-play / survivability theorem.
- Bit-column representation and Rust-equivalence bridge.
- Connecting Lean proofs to the Rust engine via codegen/extraction.
