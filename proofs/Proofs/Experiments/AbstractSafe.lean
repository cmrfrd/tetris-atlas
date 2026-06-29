import Proofs.Safety.SafeSet
import Proofs.Safety.Adversarial

/-!
# AbstractSafe — route (B): `init ∈ safe` via a SOUND FINITE ABSTRACTION

Sorry-free reduction. The idea of route B: instead of a concrete carrier over
`GameState` (2^200 boards), pick a SMALL abstract state space `AbsState` (a `Fintype`)
and an abstract winning set `A` whose `α init ∈ A` is `decide`-able by a finite abstract
fixpoint. The payoff: the abstract fixpoint discharges the *all-orders / interleaving*
closure for free (it is just a finite GFP). Only the *per-(class, piece) placement
realization* needs concrete Tetris content — a finite, uniform obligation, NOT a
5040-order argument.

This file states that reduction honestly, with no `sorry`:

* `tetrisSolvableValid_of_abstraction` — the general principle: any abstraction
  `α : GameState → AbsState` with a SOUND winning set `A` (non-loss + per-piece valid
  realization landing back in `A`) containing `α init` proves `TetrisSolvableValid`.
  A thin specialization of the green `tetrisSolvableValid_of_invariant` to the pullback
  invariant `{g | α g ∈ A}`.
* `not_lost_of_absLost_false` — **survival is free**: under the lost-bit-sound classifier
  the non-loss obligation is mechanical.
* `tetrisSolvableValid_of_realization` — route B specialized: with survival auto-discharged,
  solvability reduces to exactly two explicit obligations — the per-(class, piece)
  `realization` (the crux #66/#72) and `α init ∈ A`.

The abstract state below records the PROPOSED quotient: a surface shape class, the floor
height mod 4 (the once-per-bag vertical-`I` drain decrements it by exactly 4), and the
reserved drain-well column. A faithful surface classifier instantiating `realization` is
the open work; the lost-bit classifier here suffices to prove survival is free.
-/

namespace Tetris.AbstractSafe

open Tetris

/-! ## The general sound-abstraction reduction -/

/-- **Route B, as a sorry-free reduction.** An abstraction `α : GameState → AbsState`
with a SOUND abstract winning set `A` — every concretization of an `A`-state is non-lost
and, for each drawable piece, has a valid placement landing back in `A` — that contains
`α init` proves `TetrisSolvableValid`. Pullback `{g | α g ∈ A}` is a closed invariant, so
this is `tetrisSolvableValid_of_invariant` transported through `α`. -/
theorem tetrisSolvableValid_of_abstraction
    {AbsState : Type*} (α : GameState → AbsState) (A : Set AbsState)
    (hsound : ∀ g, α g ∈ A →
      ¬ g.lost GameConfig.standard ∧
      ∀ p ∈ g.bag, ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        α (adversarialStep GameConfig.standard g p pl) ∈ A)
    (hinit : α GameState.init ∈ A) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_invariant {g | α g ∈ A} hinit
    (fun g hg => (hsound g hg).1)
    (fun g hg p hp => (hsound g hg).2 p hp)

/-! ## The proposed abstraction: surface class × floor mod 4 × drain well -/

/-- Number of distinct surface-shape classes the strategy maintains (placeholder knob —
a faithful classifier instantiates this with the enumerated reset/steady-state surfaces).
Kept as a `Fin` so `AbsState` is a `Fintype` by construction. -/
abbrev surfaceClassCount : ℕ := 64
abbrev SurfaceClass := Fin surfaceClassCount

/-- The quotient of `Board` the strategy distinguishes: a surface class, the floor height
mod 4 (the quantity the once-per-bag vertical-`I` drain decrements by exactly 4), and the
reserved drain-well column. -/
structure BoardAbs where
  surface : SurfaceClass
  floorMod4 : Fin 4
  well : Fin 10
deriving DecidableEq, Fintype

/-- An abstract game state: an abstract board with the EXACT remaining bag (`≤ 128`). -/
structure AbsState where
  board : BoardAbs
  bag : Bag
deriving DecidableEq, Fintype

/-- Lost-bit surface classifier: surface `1` ⇔ the board is lost, surface `0` ⇔ live.
This is the minimal information making the abstraction SOUND for the survival conjunct.
The full classifier (surface shape, floor, well) instantiating `realization` is open. -/
def boardAbs (b : Board) : BoardAbs :=
  ⟨if Board.isLost GameConfig.standard b then 1 else 0, 0, 0⟩

/-- The abstraction map: board quotiented, bag exact. -/
def α (g : GameState) : AbsState := ⟨boardAbs g.board, g.bag⟩

/-- Abstract loss: surface bucket is the lost-bit `1`. -/
def absLost (a : AbsState) : Bool := decide (a.board.surface = 1)

/-! ## Survival is free under the lost-bit classifier -/

/-- **Survival is mechanical.** If `α g` is not abstractly lost then `g` is not lost — the
lost-bit classifier is sound, so the non-loss obligation of `tetrisSolvableValid_of_abstraction`
is discharged for free by any winning set that avoids `absLost`. -/
theorem not_lost_of_absLost_false {g : GameState} (h : absLost (α g) = false) :
    ¬ g.lost GameConfig.standard := by
  intro hlost
  have htrue : absLost (α g) = true := by
    unfold absLost α boardAbs
    simp only [decide_eq_true_eq]
    exact if_pos (show Board.isLost GameConfig.standard g.board from hlost)
  rw [h] at htrue
  exact Bool.noConfusion htrue

/-! ## Route B, specialized: solvability ⟸ {realization, init ∈ A} -/

/-- **Route B reduction.** With survival discharged by `not_lost_of_absLost_false`, the
solvability goal reduces to exactly the two open obligations: `realization` (every drawable
piece has a *valid concrete placement* landing back in `A` — the per-(class, piece) geometry,
crux #66/#72) and `hinit` (`α init` is in the abstract winning set, a finite `decide`). The
`hAlost` premise (`A` avoids lost states) is what the decidable abstract fixpoint guarantees
by construction (its operator filters by `¬ absLost`). -/
theorem tetrisSolvableValid_of_realization
    (A : Set AbsState)
    (hAlost : ∀ a ∈ A, absLost a = false)
    (realization : ∀ g, α g ∈ A → ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
        α (adversarialStep GameConfig.standard g p pl) ∈ A)
    (hinit : α GameState.init ∈ A) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_abstraction α A
    (fun g hg => ⟨not_lost_of_absLost_false (hAlost (α g) hg), realization g hg⟩)
    hinit

end Tetris.AbstractSafe
