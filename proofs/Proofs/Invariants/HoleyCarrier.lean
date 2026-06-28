import Mathlib
import Proofs.Model.Board
import Proofs.Model.Placement
import Proofs.Invariants.Wqo

/-!
# Hole-aware order, and why the dominated-basis route is NON-CONGRUENT for holes

`Invariants.Wqo` carries the finite-basis route on the **height-only** domination order
`domLE`. This module asks whether the same machinery survives a **hole-aware** order
`safeLE b β := domLE b β ∧ holes b ⊆ holes β` (β the worst case: tallest *and* holiest).
The answer is a clean **negative result**, proven here as concrete refutations:

* `place_holes_mono_false`: the no-clear drop is *not* hole-monotone — the *emptier* board
  can end up holier (S-overhang).
* `clearLines_holes_le_false`: line clears *create* holes.

So holes make the step **non-congruent** w.r.t. `safeLE`: a faithful hole carrier must be an
*explicit* closed set (atlas-style), not "everything dominated by a small basis". `HoleyBoard`
(skyline + budgeted transient holes) bounds the size of that explicit set.

These are pure board facts (kernel `decide` witnesses, base-axiom clean), reused by
`SurfaceFiber` and `HoleDebt`. The `TetrisSolvableValid` reduction
(`tetrisSolvableValid_of_holey_wqo_basis`) lives in `Proofs.Experiments.HoleyCarrier`.
-/

namespace Tetris.HoleyCarrier

open Tetris Tetris.WqoCarrier

/-! ## The hole-aware state and order -/

/-- **Buried empties** ("holes"): in-field cells that are *not* filled yet sit strictly
below their column's stack height — i.e. covered by something above. This is the feature
the height-only order throws away. -/
def holes (cfg : GameConfig) (b : Board) : Finset Coord :=
  (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).filter
    (fun p => p ∉ b ∧ p.2 < b.colHeight p.1)

/-- **Hole-aware domination.** `b` is "no harder to survive than" `β` when it is no
taller in every column *and* has no holes `β` lacks. The first conjunct is exactly
`WqoCarrier.domLE`. -/
def safeLE (cfg : GameConfig) (b β : Board) : Prop :=
  domLE b β ∧ holes cfg b ⊆ holes cfg β

/-- The height half of `safeLE` is `domLE` (definitionally the first projection). -/
theorem safeLE_domLE {cfg : GameConfig} {b β : Board} (h : safeLE cfg b β) : domLE b β :=
  h.1

@[refl] theorem safeLE_refl (cfg : GameConfig) (b : Board) : safeLE cfg b b :=
  ⟨fun _ => le_refl _, Finset.Subset.refl _⟩

/-- `safeLE` is transitive. -/
theorem safeLE_trans {cfg : GameConfig} {a b c : Board}
    (h1 : safeLE cfg a b) (h2 : safeLE cfg b c) : safeLE cfg a c :=
  ⟨domLE_trans h1.1 h2.1, fun _ hx => h2.2 (h1.2 hx)⟩

/-! ## What fails: the step is NOT `safeLE`-monotone (concrete refutations)

The WQO route's power comes from lifting a *basis-only* closure check to every dominated
state, which requires the step to be monotone in the order. For `safeLE` it is not —
neither the no-clear drop (Obligation 1) nor line clears (Obligation 2). Both are refuted
by concrete decidable witnesses, so these are theorems, not open problems. -/

/-- **Obligation 1 is FALSE — the no-clear drop is not hole-monotone.** Witness:
`b = ∅ ≼ β = {(2,0)}` under `safeLE`, but dropping the S-piece (`rot 0, col 0`) makes the
*emptier* board `b` holier: `(2,0)` is a hole of `place b` yet filled in `place β`. -/
theorem place_holes_mono_false :
    ¬ ∀ (cfg : GameConfig) (b β : Board) (pl : Placement), safeLE cfg b β →
        holes cfg (pl.place b) ⊆ holes cfg (pl.place β) := by
  intro H
  have hsafe : safeLE GameConfig.standard (∅ : Board) {((2 : ℕ), (0 : ℕ))} := by
    refine ⟨fun j => ?_, ?_⟩
    · rw [Board.colHeight_empty]; exact Nat.zero_le _
    · decide
  have hsub := H GameConfig.standard (∅ : Board) {((2 : ℕ), (0 : ℕ))}
    ⟨Piece.S, 0, 0⟩ hsafe
  have hmem : ((2 : ℕ), (0 : ℕ)) ∈
      holes GameConfig.standard (Placement.place (∅ : Board) ⟨Piece.S, 0, 0⟩) := by decide
  have hnot : ((2 : ℕ), (0 : ℕ)) ∉
      holes GameConfig.standard
        (Placement.place {((2 : ℕ), (0 : ℕ))} ⟨Piece.S, 0, 0⟩) := by decide
  exact hnot (hsub hmem)

/-- **Obligation 2 is FALSE — line clears create holes.** Witness board: row `2` filled
across all 10 columns, plus column `0` cells at rows `0` and `5`. Clearing the full row
2 drops column 0 to `{0,4}`, and `(0,2)` — filled before (it was *in* the cleared row) —
becomes a fresh buried empty. -/
theorem clearLines_holes_le_false :
    ¬ ∀ (cfg : GameConfig) (b : Board),
        holes cfg (Board.clearLines cfg b) ⊆ holes cfg b := by
  intro H
  -- the witness board: full row 2 ∪ {(0,0),(0,5)}
  let row2 : Board := (Finset.range 10).image (fun c => (c, 2))
  let b : Board := insert ((0 : ℕ), (0 : ℕ)) (insert ((0 : ℕ), (5 : ℕ)) row2)
  have hsub := H GameConfig.standard b
  have hmem : ((0 : ℕ), (2 : ℕ)) ∈
      holes GameConfig.standard (Board.clearLines GameConfig.standard b) := by decide
  have hnot : ((0 : ℕ), (2 : ℕ)) ∉ holes GameConfig.standard b := by decide
  exact hnot (hsub hmem)

/-! ## The sound carrier shape: skyline + bounded transient holes

Because the step is non-congruent for `safeLE`, a faithful hole carrier cannot be
"everything dominated by a small basis"; it must be an explicit closed set whose closure
is checked per concrete state (atlas-style, cf. the Lean atlas builder). `HoleyBoard`
factors a board into the clean-WQO skyline and the WQO-breaking holes, then *bounds the
holes by a budget* `K` to keep that explicit set small. `K` is small in practice: a
surviving loop's steady state is hole-free, with only transient holes (e.g. the
empty-board bootstrap's single S-first hole, cleared within bag 1). -/

/-- A board factored as a skyline plus an explicit, budgeted set of buried empties.
Columns/rows are plain `ℕ` (matching `Board`), with the configured dimensions entering
only through predicates — keeping the factorisation `Fin`-bookkeeping free. -/
structure HoleyBoard (cfg : GameConfig) (K : ℕ) where
  /-- The surface profile (the clean-WQO part), indexed by column. -/
  height : ℕ → ℕ
  /-- The explicit holes (the part that breaks clean WQOs). -/
  buried : Finset Coord
  /-- Every recorded hole really is buried under its column's surface. -/
  covered : ∀ p ∈ buried, p.2 < height p.1
  /-- The transient-hole budget: at most `K` live holes. -/
  budget : buried.card ≤ K

/-- The hole-aware domination on factored boards: shorter skyline and contained holes —
the `HoleyBoard` mirror of `safeLE`. (Same non-congruence applies; this type is for the
*enumerated* carrier, where closure is checked, not lifted.) -/
def HoleyBoard.le {cfg : GameConfig} {K : ℕ} (g β : HoleyBoard cfg K) : Prop :=
  (∀ j, g.height j ≤ β.height j) ∧ g.buried ⊆ β.buried

#print axioms place_holes_mono_false
#print axioms clearLines_holes_le_false

end Tetris.HoleyCarrier
