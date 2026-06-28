import Mathlib
import Proofs.Safety.SafeSet
import Proofs.Invariants.Wqo
import Proofs.Invariants.HoleyCarrier

/-!
# Hole-aware finite-basis reduction (research half)

The hole-aware order `safeLE`, its refutations (`place_holes_mono_false`,
`clearLines_holes_le_false`), and `HoleyBoard` now live in the green
`Proofs.Invariants.HoleyCarrier`. This file is the route-specific reduction to
`TetrisSolvableValid`, which uses the safe-set machinery; it lives in the
`ProofsExperiments` target.

What survives the non-congruence: `tetrisSolvableValid_of_holey_wqo_basis` (sorry-free).
If the hole-aware dominated carrier is *already* closed under steps, Tetris is solvable —
this needs only the height half of `safeLE` to discharge loss. The *lift* of a basis-only
closure to the dominated carrier does NOT survive (refuted in the green module).
-/

namespace Tetris.HoleyCarrier

open Tetris Tetris.WqoCarrier

/-- The dominated-by-finite-basis carrier, hole-aware. -/
def Carrier (basis : Bag → Finset Board) : Set GameState :=
  {g | ∃ β ∈ basis g.bag, safeLE GameConfig.standard g.board β}

/-- **The hole-aware finite-basis reduction (sorry-free).** `hheight`/`hinit` are
discharged here from the height half of `safeLE`; the per-piece closure on the whole
carrier (`hstep`) is the remaining obligation — and, per the refutations in
`Invariants.HoleyCarrier`, it is *not* obtainable by lifting a basis-only check through
domination. -/
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

#print axioms tetrisSolvableValid_of_holey_wqo_basis

end Tetris.HoleyCarrier
