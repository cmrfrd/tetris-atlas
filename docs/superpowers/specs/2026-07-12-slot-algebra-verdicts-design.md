# Slot algebra + flush-zone verdicts — the redirect answer and the zone-game design space

**Date**: 2026-07-12 (second spec of the day; follows the plinth foundation)
**Status**: approved (brainstorming session)
**Targets**: green `Proofs` (SlotAlgebra, base-axiom-clean) + `ProofsExperiments`
(FlushZoneGame, `native_decide` allowed per the accepted ZoneGame precedent).

## Findings driving this design

**F1 — redirects onto step-zones are impossible (provable).** From the
`Piece.shape` table: the only 2-wide rotations whose bottom profile is
`(1,0)` or `(0,1)` — the shapes that seat flush on a standing ±1 step —
are vertical S, vertical Z, and vertical T (rot 1/3). O has only `(0,0)`;
L/J 2-wide rotations have `(0,0)`, `(2,0)`, `(0,2)`; flats need `(0,0,0)`;
I needs `(0)`/`(0,0,0,0)`. The 2026-07-12 plinth spec's D4 rate fix
("3 redirected O/L/J + 1 band-I per step-zone") is dead on arrival.

**F2 — the isolated 3-column {O,L,J} zone appears not to close** flat →
flat+4 under any order: T-parity permits a 3×4 tiling but gravity refuses
(hand analysis of the order tree; the plausible "tiling" is secretly
O+L+L). Discovered en route: **L rot 3 consumes a `(2,0)` step and leaves
a flat pair; J rot 1 mirrors** — L/J are ±2-step machines.

**F3 — the schedule cannot be zone-decomposed as specced**: step-zones
cannot receive redirects (F1) and the OLJ zone cannot close alone (F2),
matching the ZoneView rate-coupling verdict from the geometry side. Any
inhabitant is an entangled cyclic pattern where L/J manufacture/consume ±2
steps, S/Z/T cycle ±1 steps, O rides flat pairs. Which small entangled
windows close is a finite AND-OR question at ZoneGame scale.

## Decisions

1. **Verdict method**: ZoneGame-style — a computable flush-only zone game
   in `ProofsExperiments` with `native_decide` verdict instances at zone
   scale (≤ 6 columns). This matches the accepted precedent (ZoneGame's
   in-kernel 10-bag family verdicts, 3 commits); the rejected BandGame line
   (band-scale deciders) is not crossed. Found cycles get hand-formalized
   green later; dead verdicts are findings.
2. F1 lands as green `decide` theorems over shape data (established
   pattern).
3. No change to `PlinthCert` — it is schedule-agnostic.

## Part A — `Proofs/Invariants/SlotAlgebra.lean` (new, green)

Definitions (computable, over `Piece.shapeUp`):

```lean
def colBot (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  (((p.shapeUp r).filter (fun c => c.1 = i)).image (fun c => c.2)).min.getD 0

def colTop (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  (((p.shapeUp r).filter (fun c => c.1 = i)).image (fun c => c.2)).max.getD 0
```

Theorems (all `by decide` over the 7-piece × 4-rotation table; state with
`∀ p r`, hypotheses as decidable conjuncts):

- `sStep_exclusive : ∀ p r, (∀ cell ∈ p.shapeUp r, cell.1 < 2) →
     colBot p r 0 = colBot p r 1 + 1 → p = Piece.S ∨ p = Piece.T`
- `zStep_exclusive` (bottoms `(0,1)` → Z or T)
- `flatPair_receivers` (2-wide bottoms `(0,0)` → O, L, or J)
- `twoStep_receivers_left` (bottoms `(2,0)` → L) and
  `twoStep_receivers_right` (bottoms `(0,2)` → J)
- Tops-value lemmas for the named 2-wide rotations (S/Z/T verticals, O,
  L rot 1/3, J rot 1/3, I vertical) and the flats — in particular the
  flattening facts: L rot 3 on `(2,0)` leaves equal tops; J rot 1 mirrors.

(If a `decide` is slow or a `Rotation`-universe quantifier resists, split
into per-rotation-value lemmas `r.val = k` — the shape table is matched on
`r.val`.)

## Part B — `Proofs/Experiments/FlushZoneGame.lean` (new, Experiments)

Mirrors `ZoneGame.lean`'s style: everything computable, verdicts by
`native_decide`.

Core:
- `FShape := List (ℕ × ℕ)` — a rotation as per-column `(bot, top)` pairs
  (width = length). A concrete table `shapesOf : FPiece → List FShape` for
  the 7 pieces (transcribed from the shape table; consistency with Part A
  is by construction, documented in the header).
- State `ZS := List ℕ` (column heights, relative); `norm : ZS → ZS`
  subtracts the min (the base-shift quotient justified by the proven
  transport theory).
- `fits (h : ZS) (c : ℕ) (s : FShape) : Bool` — flush seat: the surface
  segment at `c` matches `s`'s bottoms up to a common offset, in-bounds.
- `apply (h : ZS) (c : ℕ) (s : FShape) : ZS` — replace segment heights by
  `off + top + 1`.
- The AND-OR bag game:
  `answersBag (spread : ℕ) : ZS → List FPiece → Bool` — for every
  permutation-prefix (adversarial next piece from the remaining multiset),
  some `(rotation, column)` flush response keeping `norm`-spread ≤ cap,
  recursing until the bag is empty. Memoization-free direct recursion is
  fine at these sizes (≤ 6 columns, ≤ 3-4 pieces per bag; ZoneGame handled
  larger).
- `aliveBags (spread bags : ℕ) (bagPieces : List FPiece) (h : ZS) : Bool` —
  iterate `answersBag` over end-of-bag state SETS (dedup by `norm`) for
  `bags` rounds; alive iff no round dies.

Verdict instances (each `theorem name : aliveBags … = false/true := by
native_decide`, generous caps; wall-times recorded in PROGRESS):

| instance | cols | per-bag pieces | expectation |
|---|---|---|---|
| `olj3` | 3 | O,L,J | dead (confirms F2) |
| `olj4` | 4 | O,L,J | probably dead (hand: L-first branches die) |
| `olj5` | 5 | O,L,J | open |
| `szt4` | 4 | S,Z,T | probably dead (T-flip strands Z) |
| `szt6` | 6 | S,Z,T | open (3 step-zones) |
| `oljt5` | 5 | O,L,J,T | open — first entangled candidate |
| `oljsz6` | 6 | O,L,J,S,Z | open — second entangled candidate |

The instance list may be adjusted during execution as verdicts land (e.g.,
if `olj5` is alive, decide `olj4` variants with different spread caps to
find the boundary); every decided instance gets committed with its verdict,
alive or dead. No instance above 6 columns (the zone-scale line).

## Part C — docs

- PROGRESS.md: F1/F2/F3 recorded; the verdict table with results and
  wall-times; the schedule-design implication (entangled cycles or a new
  impossibility composition).
- LIBRARY.md: SlotAlgebra row (green spine, layer 1–2); FlushZoneGame noted
  in the Experiments keep-active list.
- The next-session pointer: hand-formalize whatever cycle the verdicts
  found (green, via the mechanisms + SlotAlgebra tables), or compose the
  dead verdicts with the 9-column budget into an impossibility theorem.

## Acceptance

- `lake build` green; SlotAlgebra theorems base-axiom-clean (axiom gate);
  hygiene gate clean (SlotAlgebra has no `native_decide`).
- `lake build ProofsExperiments` green with all verdict instances checked.
- Every verdict committed with its actual result — expectations in the
  table above are hypotheses, not requirements.

## Process and hygiene

As established: foreground builds; SlotAlgebra imported from `Proofs.lean`,
FlushZoneGame from `ProofsExperiments.lean`; one commit per logical unit
staging only `proofs/`; this spec not committed.

## Risks

1. `native_decide` wall-time on `oljsz6`-scale instances (6 cols × 5
   pieces × orders). Mitigation: dedup end-of-bag state sets by `norm`,
   cap spread tightly first, widen only if dead; ZoneGame's 10-bag
   families are the feasibility precedent.
2. `decide` on `∀ r : Rotation` quantifiers in Part A — fallback to
   per-`r.val` case lemmas.
3. The `FShape` table transcription must match `Piece.shape` — guard with
   a consistency `decide` lemma per piece (`shapesOf`-vs-`shapeUp` checks)
   inside the Experiments file.
