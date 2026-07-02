import Proofs.Safety.SafeSet
import Proofs.Invariants.Skyline

/-!
# Solvability from a skyline (surface-profile) invariant

The green bridge between the surface world and the solvability reduction:
a **bag-indexed profile invariant** — a predicate `P : Bag → (ℕ → ℕ) → Prop`
on hole-free surfaces, closed under one flush response per drawn piece — proves
`TetrisSolvableValid` outright.

This is the exact receptacle for a search-produced carrier automaton over
skyline profiles: `hstep`'s obligations (`applyStep … = skyline h'`) are
discharged transition-by-transition with `place_flush_skyline` +
`applyStep_flush_skyline` (`Proofs/Invariants/Skyline.lean`), and the height
obligation is read off the carrier's spread bound. Compared to the experiments
route (`tetrisSolvableValid_of_strategy_bagInvariant`), this version lives
entirely in the green library and asks only for an *existence*-style step
(any placement, not a fixed strategy function).
-/

namespace Tetris

open Board

/-- **Solvability from a closed skyline invariant.** If `P` is a bag-indexed
predicate on height profiles with: the zero profile at the full bag (`hinit`),
an in-field height bound (`hheight`), and closure — for every pending piece a
valid placement whose full move maps the skyline to a skyline still in `P`
(`hstep`) — then Tetris is solvable. -/
theorem tetrisSolvableValid_of_skyline_invariant
    (P : Bag → (ℕ → ℕ) → Prop)
    (hinit : P Bag.full (fun _ => 0))
    (hheight : ∀ T h, P T h → ∀ j < GameConfig.standard.cols,
      h j ≤ GameConfig.standard.rows)
    (hstep : ∀ T h p, P T h → p ∈ T →
      ∃ (pl : Placement) (h' : ℕ → ℕ), pl.piece = p ∧
        pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (skyline GameConfig.standard h) pl = skyline GameConfig.standard h' ∧
        P (T.draw p) h') :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant
    { g | ∃ h : ℕ → ℕ, g.board = skyline GameConfig.standard h ∧ P g.bag h }
    ⟨(fun _ => 0), init_board_eq_skyline GameConfig.standard, by
      simpa using hinit⟩ ?_ ?_
  · rintro g ⟨h, hb, hP⟩ j
    rw [hb]
    by_cases hj : j < GameConfig.standard.cols
    · rw [colHeight_skyline hj]
      exact hheight g.bag h hP j hj
    · rw [colHeight_skyline_eq_zero (by omega)]
      omega
  · rintro g ⟨h, hb, hP⟩ p hp
    obtain ⟨pl, h', hpiece, hvalid, happly, hP'⟩ := hstep g.bag h p hP hp
    refine ⟨pl, hpiece, hvalid, h', ?_, hP'⟩
    show Placement.applyStep GameConfig.standard g.board { pl with piece := p }
        = skyline GameConfig.standard h'
    have hpl : ({ pl with piece := p } : Placement) = pl := by
      cases pl
      simp_all
    rw [hpl, hb, happly]

end Tetris
