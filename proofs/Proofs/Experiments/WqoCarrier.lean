import Mathlib
import Proofs.SafeSet
import Proofs.Invariants.Wqo

/-!
# WQO / finite-basis route to `TetrisSolvableValid` — the reduction

The monotonicity primitives this route rests on (`domLE`, `place_domLE_mono`,
`clearLines_domLE`, `domLE_trans`) now live in the green `Proofs.Invariants.Wqo` module.
This file is the research half: the dominated-by-finite-basis carrier and the reduction to
`TetrisSolvableValid`, which uses the safe-set machinery (`adversarialStep`,
`tetrisSolvableValid_of_height_bounded_invariant`). It is route-specific and lives in the
`ProofsExperiments` target, not the standard library.
-/

namespace Tetris.WqoCarrier

open Tetris

/-! ## S2 — the finite-basis carrier wired to the proven reduction

The carrier `Carrier basis := { g | g.board is dominated by some β ∈ basis g.bag }`.
Fed to the proven `tetrisSolvableValid_of_height_bounded_invariant`. The `hheight`
obligation is FREE (domination + a height-bounded basis), `hinit` is the empty board
under a basis element, and the per-piece `hstep` is the one remaining hard obligation —
it lifts a FINITE closure check at the basis (S4, `native_decide`) to the whole dominated
carrier via `place_domLE_mono`, modulo line clears (S3, bag-boundary handling). -/

/-- The dominated-by-finite-basis carrier as a set of game states. -/
def Carrier (basis : Bag → Finset Board) : Set GameState :=
  {g | ∃ β ∈ basis g.bag, domLE g.board β}

/-- **The WQO finite-basis reduction.** `hheight`/`hinit` are discharged here; the per-piece
closure on the whole carrier (`hstep`) is the remaining obligation, to be supplied by lifting
a finite basis-closure check through `place_domLE_mono` (clears handled at bag boundaries). -/
theorem tetrisSolvableValid_of_wqo_basis
    (basis : Bag → Finset Board)
    (hbheight : ∀ (T : Bag) (β : Board), β ∈ basis T →
      ∀ j, β.colHeight j ≤ GameConfig.standard.rows)
    (hinit : ∃ β ∈ basis Bag.full, domLE Board.empty β)
    (hstep : ∀ g ∈ Carrier basis, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ Carrier basis) :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant (Carrier basis) ?_ ?_ hstep
  · -- hinit : GameState.init ∈ Carrier basis
    obtain ⟨β, hβ, hdom⟩ := hinit
    exact ⟨β, by simpa using hβ, by simpa using hdom⟩
  · -- hheight : every carrier state is height-bounded (domination + basis height bound)
    rintro g ⟨β, hβ, hdom⟩ j
    exact le_trans (hdom j) (hbheight g.bag β hβ j)

/-! ## S3 — lift a FINITE basis-closure check to the whole dominated carrier

The closure obligation is checked only at the finite basis, using the no-clear `place`
(the *highest* possible successor). For any dominated `g ≼ β`, the same placement gives
`applyStep g pl = clearLines(place g pl) ≼ place g pl ≼ place β pl ≼ β'` — chaining
`clearLines_domLE`, `place_domLE_mono`, and the basis check. Validity is board-independent,
so it transfers from `β` to `g`. -/
theorem hstep_of_basis_closure
    (basis : Bag → Finset Board)
    (hclosure : ∀ (T : Bag) (β : Board), β ∈ basis T → ∀ p, p ∈ T →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        ∃ β' ∈ basis (T.draw p),
          domLE (Placement.place β { pl with piece := p }) β') :
    ∀ g ∈ Carrier basis, ∀ p, p ∈ g.bag →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        adversarialStep GameConfig.standard g p pl ∈ Carrier basis := by
  rintro g ⟨β, hβ, hgβ⟩ p hp
  obtain ⟨pl, hpiece, hvalid, β', hβ', hpβ'⟩ := hclosure g.bag β hβ p hp
  refine ⟨pl, hpiece, hvalid, β', ?_, ?_⟩
  · simpa [adversarialStep] using hβ'
  · show domLE (adversarialStep GameConfig.standard g p pl).board β'
    have step1 :
        domLE (Placement.applyStep GameConfig.standard g.board { pl with piece := p })
          (Placement.place g.board { pl with piece := p }) := by
      rw [Placement.applyStep]; exact clearLines_domLE _ _
    have step2 :
        domLE (Placement.place g.board { pl with piece := p })
          (Placement.place β { pl with piece := p }) :=
      place_domLE_mono _ hgβ
    have hchain :
        domLE (Placement.applyStep GameConfig.standard g.board { pl with piece := p }) β' :=
      domLE_trans (domLE_trans step1 step2) hpβ'
    simpa [adversarialStep] using hchain

/-- **S2+S3 composed: `TetrisSolvableValid` from a FINITE basis-closure check.** The whole
goal now reduces to: a finite, height-bounded, init-dominating `basis` whose every element,
for every drawable piece, has a valid placement whose *no-clear* drop is dominated by another
basis element. That hypothesis is a finite `native_decide` obligation (S4) on a concrete basis
(the ~tens-element antichain measured in S1). -/
theorem tetrisSolvableValid_of_wqo
    (basis : Bag → Finset Board)
    (hbheight : ∀ (T : Bag) (β : Board), β ∈ basis T →
      ∀ j, β.colHeight j ≤ GameConfig.standard.rows)
    (hinit : ∃ β ∈ basis Bag.full, domLE Board.empty β)
    (hclosure : ∀ (T : Bag) (β : Board), β ∈ basis T → ∀ p, p ∈ T →
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        ∃ β' ∈ basis (T.draw p),
          domLE (Placement.place β { pl with piece := p }) β') :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_wqo_basis basis hbheight hinit
    (hstep_of_basis_closure basis hclosure)

end Tetris.WqoCarrier

#print axioms Tetris.WqoCarrier.tetrisSolvableValid_of_wqo
