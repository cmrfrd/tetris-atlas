import Proofs.Safety.SafeSet
import Proofs.Invariants.Skyline
import Proofs.Invariants.HoledSkyline

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

/-- **Solvability from a closed debt-1 invariant.** The debt-carrying
generalisation of `tetrisSolvableValid_of_skyline_invariant`: states are
profiles with an optional strictly covered hole (`Board.debtBoard`), and the
invariant `P` is closed under one full move per pending piece. The flat-
stratum exclusion (`no_holefree_S_on_flat`) shows hole-free invariants cannot
contain `init`, so THIS is the reduction any certificate from the empty board
must feed. Each `hstep` obligation is discharged by `place_holedSkyline` /
`place_flush_skyline` plus exactly one of the three clearing laws
(`clearLines_holedSkyline_of_le` / `_of_lt` / `_exposed` or
`clearLines_skyline`). -/
theorem tetrisSolvableValid_of_debt_invariant
    (P : Bag → (ℕ → ℕ) → Option Coord → Prop)
    (hinit : P Bag.full (fun _ => 0) none)
    (hcover : ∀ T h x, P T h (some x) → x.1 < GameConfig.standard.cols ∧
      x.2 + 1 < h x.1)
    (hheight : ∀ T h ho, P T h ho → ∀ j < GameConfig.standard.cols,
      h j ≤ GameConfig.standard.rows)
    (hstep : ∀ T h ho p, P T h ho → p ∈ T →
      ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord), pl.piece = p ∧
        pl.Valid GameConfig.standard ∧
        Placement.applyStep GameConfig.standard
          (Board.debtBoard GameConfig.standard h ho) pl
          = Board.debtBoard GameConfig.standard h' ho' ∧
        P (T.draw p) h' ho') :
    TetrisSolvableValid := by
  refine tetrisSolvableValid_of_height_bounded_invariant
    { g | ∃ (h : ℕ → ℕ) (ho : Option Coord),
        g.board = Board.debtBoard GameConfig.standard h ho ∧ P g.bag h ho }
    ⟨(fun _ => 0), none, init_board_eq_skyline GameConfig.standard, by
      simpa using hinit⟩ ?_ ?_
  · rintro g ⟨h, ho, hb, hP⟩ j
    rw [hb, colHeight_debtBoard (fun x hx => (hcover g.bag h x (hx ▸ hP)).2)]
    by_cases hj : j < GameConfig.standard.cols
    · rw [if_pos hj]
      exact hheight g.bag h ho hP j hj
    · rw [if_neg hj]
      omega
  · rintro g ⟨h, ho, hb, hP⟩ p hp
    obtain ⟨pl, h', ho', hpiece, hvalid, happly, hP'⟩ := hstep g.bag h ho p hP hp
    refine ⟨pl, hpiece, hvalid, h', ho', ?_, hP'⟩
    show Placement.applyStep GameConfig.standard g.board { pl with piece := p }
        = Board.debtBoard GameConfig.standard h' ho'
    have hpl : ({ pl with piece := p } : Placement) = pl := by
      cases pl
      simp_all
    rw [hpl, hb, happly]

/-! ## The certificate schema: what remains, written down

The squeeze converges here. Necessity says any certificate is a bag-indexed
family of debt-≤1 boards excluding the flat stratum and presenting, whenever
the corresponding piece is pending, an S-window, a Z-window, and an O-window
(I never starves). Sufficiency says such a family that is CLOSED under one
full move per pending piece proves `TetrisSolvableValid`. `DebtCertificate`
packages exactly that closure data — the single object left to construct.
Solving Tetris is now literally: inhabit this structure. -/

/-- **The certificate.** A bag-indexed debt-1 family, containing the initial
state, height-bounded, and closed under one full move per pending piece.
Inhabiting this structure — the one remaining open obligation of the whole
development — proves Tetris solvable (`tetrisSolvableValid_of_debtCertificate`). -/
structure DebtCertificate where
  /-- The invariant: pending bag, surface profile, optional buried cell. -/
  P : Bag → (ℕ → ℕ) → Option Coord → Prop
  /-- The empty board at a fresh bag is in the family. -/
  init : P Bag.full (fun _ => 0) none
  /-- Holes are in-field and strictly covered. -/
  cover : ∀ T h x, P T h (some x) → x.1 < GameConfig.standard.cols ∧
    x.2 + 1 < h x.1
  /-- The family respects the ceiling. -/
  height : ∀ T h ho, P T h ho → ∀ j < GameConfig.standard.cols,
    h j ≤ GameConfig.standard.rows
  /-- Closure: every pending piece has a full-move response inside the family. -/
  step : ∀ T h ho p, P T h ho → p ∈ T →
    ∃ (pl : Placement) (h' : ℕ → ℕ) (ho' : Option Coord), pl.piece = p ∧
      pl.Valid GameConfig.standard ∧
      Placement.applyStep GameConfig.standard
        (Board.debtBoard GameConfig.standard h ho) pl
        = Board.debtBoard GameConfig.standard h' ho' ∧
      P (T.draw p) h' ho'

/-- Inhabiting the certificate schema proves Tetris solvable. -/
theorem tetrisSolvableValid_of_debtCertificate (C : DebtCertificate) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_debt_invariant C.P C.init C.cover C.height C.step

end Tetris
