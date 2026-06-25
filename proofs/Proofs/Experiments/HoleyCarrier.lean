import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.SafeSet
import Proofs.Experiments.WqoCarrier

/-!
# Hole-aware carrier — and why the dominated-basis route is NON-CONGRUENT for holes

`WqoCarrier.lean` carries the finite-basis route on the **height-only** domination
order `domLE b β := ∀ j, colHeight b j ≤ colHeight β j`. That route works because of two
monotonicity facts: the no-clear drop is `domLE`-monotone (`place_domLE_mono`) and line
clears only *lower* heights (`clearLines_domLE`), so the no-clear placement at the basis
is the worst successor and everything dominated by the basis inherits closure.

This file asks whether the same dominated-basis machinery survives if we make the order
**hole-aware**: `safeLE b β := domLE b β ∧ holes b ⊆ holes β` (β is the worst case:
tallest *and* holiest). The answer is a clean **negative result**:

* **What survives — `tetrisSolvableValid_of_holey_wqo_basis` (sorry-free).** If the
  hole-aware dominated carrier is *already* closed under steps, Tetris is solvable. This
  needs only the *height half* of `safeLE` (to discharge loss) and is proven outright.

* **What fails — the LIFT of a finite basis-closure to the dominated carrier.** That lift
  needs the step to be `safeLE`-monotone. Both halves are **FALSE**, proven here as
  concrete refutations:
  - `place_holes_mono_false`: the no-clear drop is *not* hole-monotone. Drop the S-piece
    flat. On `∅` it lands `{(0,0),(1,0),(1,1),(2,1)}`, leaving `(2,0)` a hole. On the
    "safer" `β = {(2,0)}` (which `safeLE`-dominates `∅`) the *same* drop lands identically
    but `(2,0)` is now filled — so the **emptier board ends up holier**. Fewer cells is
    not safer once holes are in play.
  - `clearLines_holes_le_false`: clears *create* holes. With column `0 = {0,2,5}` and row
    `2` board-wide-full, clearing drops the column to `{0,4}`; `(0,2)` was filled (it was
    in the full row) but is a *new* buried empty afterward.

The upshot: holes make the step **non-congruent** with respect to `safeLE`, so the
"dominated by a small basis" carrier — the heart of the WQO route — does not transfer.
A faithful hole carrier must be an *explicit* closed set (atlas-style), with closure
checked per concrete state, not lifted through domination. `HoleyBoard` (skyline +
budgeted transient holes) bounds the size of that explicit set; it is the right shape,
but it is enumerated, not dominated.
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

/-! ## What survives: the sorry-free dominated-carrier reduction

This is the hole-aware mirror of `WqoCarrier`'s *basis-level* reduction. It does NOT use
any hole monotonicity — it takes the per-state closure (`hstep`) as a hypothesis and only
uses the height half of `safeLE` to discharge the loss obligation. Hence sorry-free. -/

/-- The dominated-by-finite-basis carrier, hole-aware. -/
def Carrier (basis : Bag → Finset Board) : Set GameState :=
  {g | ∃ β ∈ basis g.bag, safeLE GameConfig.standard g.board β}

/-- **The hole-aware finite-basis reduction (sorry-free).** `hheight`/`hinit` are
discharged here from the height half of `safeLE`; the per-piece closure on the whole
carrier (`hstep`) is the remaining obligation — and, per the refutations below, it is
*not* obtainable by lifting a basis-only check through domination. -/
theorem tetrisSolvableValid_of_holey_wqo_basis
    (basis : Bag → Finset Board)
    (hbheight : ∀ (T : Bag) (β : Board), β ∈ basis T →
      ∀ j, β.colHeight j ≤ GameConfig.standard.rows)
    (hinit : ∃ β ∈ basis Bag.full, safeLE GameConfig.standard Board.empty β)
    (hstep : ∀ g ∈ Carrier basis, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ Carrier basis) :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant (Carrier basis) ?_ ?_ hstep
  · -- hinit : GameState.init ∈ Carrier basis
    obtain ⟨β, hβ, hdom⟩ := hinit
    exact ⟨β, by simpa using hβ, by simpa using hdom⟩
  · -- hheight : carrier states are height-bounded (height half of safeLE + basis bound)
    rintro g ⟨β, hβ, hdom⟩ j
    exact le_trans (hdom.1 j) (hbheight g.bag β hβ j)

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

#print axioms tetrisSolvableValid_of_holey_wqo_basis
#print axioms place_holes_mono_false
#print axioms clearLines_holes_le_false

end Tetris.HoleyCarrier
