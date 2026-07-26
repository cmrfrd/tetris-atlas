# WQO / Finite-Basis Route — Findings (HONEST FLOOR)

**Status: HONEST FLOOR.** The well-quasi-order / Dickson finite-basis route produced
real, reusable, sorry-free Lean scaffolding and passed its basis-size make-or-break, but
hits a genuine obstruction: **no domination order is simultaneously monotone under the
Tetris dynamics and fine enough to define a closed carrier.** A bounded negative result.

**Date:** 2026-06-24
**Lean file:** `proofs/Proofs/Experiments/WqoCarrier.lean` (builds green, sorry-free)

---

## The idea

Column-height vectors live in `ℕ¹⁰`; the product order is a well-quasi-order (Dickson).
If the safe set is downward-closed under height-domination `b ≼ β := ∀ j, colHeight b j ≤
colHeight β j`, it has a FINITE BASIS (finitely many maximal surfaces) — a finite
certificate for the huge (>5×10⁵) survival carrier. Carrier `C T b := ∃ β ∈ basis T,
b ≼ β`, fed to the proven `tetrisSolvableValid_of_bag_indexed_invariant`.

## What was PROVEN (sorry-free, axioms [propext, Classical.choice, Quot.sound]) — reusable

- `place_domLE_mono` : `domLE b β → domLE (pl.place b) (pl.place β)` — the no-clear drop
  preserves height-domination (the keystone). Via `dropOffset_mono`, `colHeight_union`,
  `colHeight_dropped_mono`.
- `clearLines_domLE` : `domLE (clearLines b) b` — line clears only lower column heights.
- `tetrisSolvableValid_of_wqo` : `TetrisSolvableValid` follows from a height-bounded,
  init-dominating `basis : Bag → Finset Board` whose every element, for every drawable
  piece, has a valid placement whose no-clear `place` is dominated by another basis
  element. (The clear case handled by `applyStep g pl = clearLines(place g pl) ≼ place g
  pl ≼ place β pl ≼ β'`.)

## Make-or-break #1 (S1): basis size — PASSED

`tetris_carrier_probe basis` (sampled, depth-2, random orders): over 200k bags / 130k
DISTINCT height-vectors, the maximal antichain under domination stays **bounded at
10–28** (oscillates; drops when a tall surface dominates many lower ones). The carrier
basis is ~tens of surfaces — trivially `native_decide`-able. The deep player keeps
surfaces low+tight, so they collapse to a few maximal ones.

## Make-or-break #2 (S4): is the carrier closed? — FAILED (the floor)

The `place`-based `hclosure` (needed so the lift covers non-clearing dominated boards)
is **non-draining**: `place` never removes cells, so iterating it escalates height to 20
— no finite basis satisfies it. Switching to a draining (`applyStep`) closure breaks the
lift (it no longer dominates non-clearing `g ≼ β`). Root cause: **pure height-domination
is too coarse** — `{b : heights(b) ≤ β}` includes holey, well-less, NON-DRAINABLE boards
that escape (every placement raises them past β, they can't clear). Same lesson as
SurfaceInvariant iter332: a declarative carrier admits adversary-junk the strategy never
reaches.

**The refinement also fails.** Adding a hole bound to the order would exclude the
non-drainable boards, but `tetris_carrier_probe hmono` (16M no-clear checks on
height+hole-dominated pairs): **HEIGHT-order 0 violations, HOLE-order 364,374 (2.28%)
violations.** Placing a piece creates holes according to the local surface, not
monotonically — so the refined (height+hole) order has **no clean keystone**.

## Conclusion

The WQO route is caught in an irreducible bind:
- The order must be **coarse** (height-only) to be **monotone** (keystone holds) — but
  then the carrier is too coarse to be closed (non-drainable boards escape).
- The order must be **fine** (height+holes/well) to define a **closed** carrier — but
  then it is **not monotone** (the keystone fails).

There is no domination order that is both. This is the same fundamental obstruction every
route in this project hits: **the survival carrier is a genuinely complex object** (the
deep player's reachable set, >5×10⁵ surfaces, no simple closed-form), so it is neither
enumerable (too big) nor cleanly characterizable by an order/potential/abstraction. The
WQO route was the most principled attempt yet — purpose-built math for "huge but
structured" sets — and its failure mode is sharply diagnostic: the structure is real
(small antichain) but the dynamics don't respect any sufficiently-fine order.

**Reusable residue:** `place_domLE_mono`, `clearLines_domLE`, and the reduction structure
are correct sorry-free theorems; any future order-based attempt reuses them. The
`monotone`/`hmono`/`basis` probe modes are reusable diagnostics.
