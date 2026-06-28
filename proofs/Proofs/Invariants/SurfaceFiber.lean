import Mathlib
import Proofs.Model.Board
import Proofs.Model.Placement
import Proofs.Invariants.Wqo
import Proofs.Invariants.HoleyCarrier

/-!
# Surface fibers — the placement dynamics are HOLE-INDEPENDENT (a positive foundation)

## Background

`HoleyCarrier.lean` proved a sharp negative: the hole-aware order
`safeLE b β := domLE b β ∧ holes b ⊆ holes β` is **not a congruence** for the Tetris
step (both `place_holes_mono_false` and `clearLines_holes_le_false`). So the WQO route's
"dominated by a small basis" carrier cannot carry holes. The natural rescue — *fiber the
state space and only compare within a fiber* — raises the question: **fiber by what?**

A tempting choice is to fiber by the **hole-set** (compare only boards with identical
buried empties). This file shows that is the **wrong** fiber: the `HoleyCarrier`
counterexample `b = ∅`, `β = {(2,0)}` already has *equal* hole-sets (both `∅`), yet the
S-drop creates a hole on the emptier board and not on `β`. The cell `(2,0)` is a *surface*
cell (it raises `colHeight`), not a hole — so it lies in the same hole-fiber but a
different **surface**. See `place_holes_mono_within_hole_fiber_false`.

## What this file proves (the right fiber: the surface)

The correct fiber is the **surface profile** — the column-height vector `colHeight`. The
key structural fact, proven here, is that **where a piece lands depends only on
the surface, never on the buried holes**:

* `dropOffset_eq_of_colHeight_eq` — equal surfaces ⇒ identical hard-drop offset.
* `dropped_eq_of_colHeight_eq`   — equal surfaces ⇒ the piece occupies identical cells.
* `colHeight_place_eq_of_colHeight_eq` — equal surfaces ⇒ the *post-placement surface is
  identical*. **The surface evolves as a deterministic automaton, independent of holes.**
* `holes_place_subset_of_colHeight_eq` — within a surface fiber, a *fuller* board has
  *fewer* holes after placement (`b ⊆ β ⇒ holes(place β) ⊆ holes(place b)`). Holes are
  monotone in the right direction once the surface is pinned.

## The representation this justifies

These lemmas are the structural backbone of the **surface-automaton × hole-debt** (WSTS)
representation: decompose a board into (i) its surface — a finite-state, hole-independent
automaton that drives all piece interaction — and (ii) a downstream **hole debt** that the
surface dynamics never read and that only clears can discharge. Because placement is
hole-blind (`dropOffset_eq_of_colHeight_eq`), the surface controller is well-defined on
its own; holes ride along as a separate counter. That cleanly quarantines the proven
non-congruence: the cross-fiber, hole-changing edges (`place_holes_mono_false`,
`clearLines_holes_le_false`) become labeled transitions of the debt counter, not failed
monotonicity obligations.
-/

namespace Tetris.SurfaceFiber

open Tetris Tetris.WqoCarrier Tetris.HoleyCarrier

/-- Membership in `holes`, unfolded. -/
theorem mem_holes_iff {cfg : GameConfig} {b : Board} {p : Coord} :
    p ∈ holes cfg b ↔
      p ∈ Finset.range cfg.cols ×ˢ Finset.range cfg.rows ∧
        p ∉ b ∧ p.2 < b.colHeight p.1 := by
  unfold holes; rw [Finset.mem_filter]

/-! ## The surface determines the landing (hole-independence of placement) -/

/-- **The hard-drop offset depends only on the surface.** `dropOffset` is a `Finset.sup`
of `colHeight`-differences, so equal column heights give an equal offset — buried holes
below the surface are invisible to the drop. -/
theorem dropOffset_eq_of_colHeight_eq {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) :
    pl.dropOffset b = pl.dropOffset β := by
  unfold Placement.dropOffset
  apply Finset.sup_congr rfl
  intro cell _
  rw [h]

/-- **The dropped piece occupies identical cells on equal surfaces.** Immediate from
`dropOffset_eq_of_colHeight_eq`, since `dropped = cellsAt (dropOffset …)`. -/
theorem dropped_eq_of_colHeight_eq {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) :
    pl.dropped b = pl.dropped β := by
  unfold Placement.dropped
  rw [dropOffset_eq_of_colHeight_eq pl h]

/-- **Placement preserves subset within a surface fiber.** Equal surfaces drop the piece
to the same cells, so the only difference between `place b` and `place β` is the original
boards — and `b ⊆ β` survives the shared union. -/
theorem place_subset_of_colHeight_eq_of_subset {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) (hsub : b ⊆ β) :
    pl.place b ⊆ pl.place β := by
  rw [Placement.place_eq_union_dropped b pl, Placement.place_eq_union_dropped β pl,
    dropped_eq_of_colHeight_eq pl h]
  exact Finset.union_subset_union hsub (Finset.Subset.refl _)

/-- **THE surface-automaton fact: the post-placement surface depends only on the
pre-placement surface.** `colHeight (place b)` is the max of the board's and the dropped
piece's heights (`colHeight_union`); both summands are surface-determined, so the whole
post-surface is too — *holes never influence the surface evolution.* -/
theorem colHeight_place_eq_of_colHeight_eq {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) (j : ℕ) :
    (pl.place b).colHeight j = (pl.place β).colHeight j := by
  rw [Placement.place_eq_union_dropped b pl, Placement.place_eq_union_dropped β pl,
    colHeight_union, colHeight_union, dropped_eq_of_colHeight_eq pl h, h j]

/-- **Within a surface fiber, holes are monotone (fuller ⇒ fewer holes).** If `b ⊆ β`
share a surface, then after the same placement `β` has no holes that `b` lacks: the
post-surfaces coincide (`colHeight_place_eq_of_colHeight_eq`) and every empty of `place β`
is an empty of `place b` (`place_subset_…`). This is the *correct* hole-monotonicity —
note the direction is opposite to the (false) `safeLE` one, because here `β` is the
*fuller* board, not merely the taller one. -/
theorem holes_place_subset_of_colHeight_eq {cfg : GameConfig} {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) (hsub : b ⊆ β) :
    holes cfg (pl.place β) ⊆ holes cfg (pl.place b) := by
  intro p hp
  rw [mem_holes_iff] at hp ⊢
  obtain ⟨hgrid, hnotin, hlt⟩ := hp
  refine ⟨hgrid, ?_, ?_⟩
  · exact fun hc => hnotin (place_subset_of_colHeight_eq_of_subset pl h hsub hc)
  · rwa [colHeight_place_eq_of_colHeight_eq pl h p.1]

/-! ## The wrong fiber: hole-set equality is NOT enough

`place_domLE_mono` already gives height-monotonicity *unconditionally*, so the literal
"is placement height-monotone within a hole-fiber?" is trivially yes and not the binding
constraint. The binding constraint is hole-control, and fibering by hole-set does **not**
supply it — the same surface cell that prevents a hole lives in the same hole-fiber. -/

/-- **Fibering by the hole-set is insufficient.** Reusing the `HoleyCarrier` witness:
`b = ∅` and `β = {(2,0)}` have *equal* hole-sets (both `∅`) and `domLE b β`, yet the
S-drop puts a hole at `(2,0)` on `place b` that is absent from `place β`. So even with
identical holes and height-domination, holes are not `⊆`-controlled — only **surface**
equality (above) controls them. -/
theorem place_holes_mono_within_hole_fiber_false :
    ¬ ∀ (cfg : GameConfig) (b β : Board) (pl : Placement),
        holes cfg b = holes cfg β → domLE b β →
        holes cfg (pl.place b) ⊆ holes cfg (pl.place β) := by
  intro H
  have hfiber : holes GameConfig.standard (∅ : Board)
      = holes GameConfig.standard {((2 : ℕ), (0 : ℕ))} := by decide
  have hdom : domLE (∅ : Board) {((2 : ℕ), (0 : ℕ))} := by
    intro j; rw [Board.colHeight_empty]; exact Nat.zero_le _
  have hsub := H GameConfig.standard (∅ : Board) {((2 : ℕ), (0 : ℕ))}
    ⟨Piece.S, 0, 0⟩ hfiber hdom
  have hmem : ((2 : ℕ), (0 : ℕ)) ∈
      holes GameConfig.standard (Placement.place (∅ : Board) ⟨Piece.S, 0, 0⟩) := by decide
  have hnot : ((2 : ℕ), (0 : ℕ)) ∉
      holes GameConfig.standard
        (Placement.place {((2 : ℕ), (0 : ℕ))} ⟨Piece.S, 0, 0⟩) := by decide
  exact hnot (hsub hmem)


end Tetris.SurfaceFiber
