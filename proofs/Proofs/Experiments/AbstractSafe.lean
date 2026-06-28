/-
# AbstractSafe — route (B): prove `init ∈ safe` via a SOUND FINITE ABSTRACTION

SCOPING DRAFT (unbuilt scaffold — proofs are `sorry`, statements are the deliverable).

Goal: prove `GameState.init ∈ Tetris.safe GameConfig.standard` — which, via the
already-proven bridge (`SafeIterateFinite.tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle`
∘ `init_safe_iff_exists_init_adversarialClosedCycle`), is `TetrisSolvableValid` —
WITHOUT exhibiting a concrete carrier set over `GameState` (2^200 boards).

Strategy: pick a SMALL abstract state space `AbsState` (a `Fintype`), define an
abstract safety game on it whose winning region `absSafe` is computed by a finite
fixpoint iteration (so `α init ∈ absSafe` is `decide`-able), and prove ONE
simulation lemma lifting abstract-safe membership to the concrete `gfp`.

The payoff vs. the hand-built skyline carrier: the abstract fixpoint `decide`
discharges the *all-orders / interleaving* closure for free (it is just the finite
abstract GFP). Only the *per-(class, piece) placement realization* needs a concrete
proof — a finite, uniform obligation, NOT a 5040-order argument. The whole hard
content of the theorem (the part Burgiel guarantees is irreducible) lives in the
single `abs_simulation` keystone below.
-/
import Proofs.Safety.SafeSet
import Proofs.Safety.Adversarial
import Proofs.Safety.SafeIterateFinite

set_option maxRecDepth 4000

namespace Tetris.AbstractSafe

open Tetris

/-! ## 1. The abstract state space (must be small enough to `decide`) -/

/-- Number of distinct *surface shape classes* the strategy ever maintains.
PLACEHOLDER: to be instantiated with the actual enumerated reset/steady-state
surfaces (flat-top, double-SZ-valley, spread-band, …). Kept as a `Fin` so the
cardinality is explicit and the type is a `Fintype` by construction. -/
abbrev surfaceClassCount : ℕ := 64
abbrev SurfaceClass := Fin surfaceClassCount

/-- Abstraction of a `Board`: its surface class, the floor height mod 4 (the
quantity the once-per-bag vertical-`I` drain decrements by exactly 4), and which
column holds the reserved drain well. This is the *quotient* of `Board` the
strategy actually distinguishes. -/
structure BoardAbs where
  surface : SurfaceClass
  floorMod4 : Fin 4
  well : Fin 10
deriving DecidableEq, Fintype

/-- An abstract game state: an abstract board together with the EXACT remaining
bag (`Bag = Finset Piece`, ≤ 128 values). The bag is kept exact because the
adversary's piece choice quantifies over it and it is already small. -/
structure AbsState where
  board : BoardAbs
  bag   : Bag
deriving DecidableEq, Fintype

-- CARDINALITY: |BoardAbs| = surfaceClassCount * 4 * 10 = 64 * 40 = 2560.
--              |AbsState| = |BoardAbs| * |Finset Piece| = 2560 * 128 = 327_680.
-- This is well within `native_decide` range for a finite fixpoint iteration, and
-- plausibly within kernel `decide`. Raising surfaceClassCount to a few hundred
-- stays under ~10^6. THIS is the knob that trades decide-feasibility against how
-- expressive the abstraction can be (i.e. how provable `abs_simulation` is).

/-! ## 2. The abstraction map `α : GameState → AbsState` -/

/-- Classify a concrete board into its abstract surface/floor/well.
SMALLEST-INSTANCE version: the surface bucket tracks only the single bit "is this
board lost?" (`1` = lost, `0` = live). This is the minimal information needed to
make the abstraction SOUND for the survival (`¬ lost`) conjunct of `abs_simulation`.
The full classifier (surface shape, floor, well) needed for the *realization*
conjunct is the open crux and is not yet encoded. -/
def boardAbs (b : Board) : BoardAbs :=
  ⟨if Board.isLost GameConfig.standard b then 1 else 0, 0, 0⟩

/-- The abstraction map. Board is quotiented; bag is exact. -/
def α (g : GameState) : AbsState := ⟨boardAbs g.board, g.bag⟩

/-! ## 3. The abstract safety game (decidable winning region) -/

/-- An abstract state is "lost" iff its surface bucket is the lost-bit (`1`).
Matches `boardAbs`: `absLost (α g) = true ↔ g` is lost. -/
def absLost (a : AbsState) : Bool := decide (a.board.surface = 1)

/-- Over-approximation-free abstract successors: the set of abstract boards
reachable from `b` by SOME valid placement of `p` that the strategy would choose.
SPEC (this is what `abs_simulation` must witness): for every concrete `g` with
`boardAbs g.board = b` and every `b' ∈ absMoves b p`, there is a *valid* placement
`pl` with `pl.piece = p` and `boardAbs (adversarialStep _ g p pl).board = b'`. -/
def absMoves (b : BoardAbs) (p : Piece) : Finset BoardAbs := sorry

/-- One step of the abstract safety operator on a `Finset` of abstract states:
keep `a` iff it is not lost and, for every drawable piece, some abstract move lands
back in the current set. Fully decidable. -/
def absSafeOp (S : Finset AbsState) : Finset AbsState :=
  S.filter fun a =>
    ¬ absLost a ∧ ∀ p ∈ a.bag, ∃ b' ∈ absMoves a.board p,
      (⟨b', a.bag.draw p⟩ : AbsState) ∈ S

/-- Finite greatest-fixpoint iteration from `univ`; converges in ≤ card steps
(monotone decreasing on a finite lattice), mirroring `safeIterFinite`. -/
def absIter : ℕ → Finset AbsState
  | 0     => Finset.univ
  | n + 1 => absSafeOp (absIter n)

/-- The abstract winning region. Defined with `irreducible_def` so the constant is
sealed to BOTH elaboration and the kernel (a plain `def`/`@[irreducible]` is not —
the kernel still eagerly evaluates the 327k-fold filter and hits deep recursion when
the soundness/lift proofs typecheck). The generated `absSafe_def : absSafe = …`
unfolds it for the eventual `init_absSafe` (`rw [absSafe_def]; native_decide`). -/
irreducible_def absSafe : Finset AbsState := absSafeOp (absIter 327680)

/-! ## 4. THE keystone: abstract-safe ⟹ one concrete safe move (simulation)

This is the ONLY lemma carrying real content. It says the abstraction is SOUND:
if `α g` is in the abstract winning region then `g` is not lost and every piece the
adversary can draw has a *concrete valid placement* whose result is again
abstract-safe. The `absSafe` fixpoint guarantees the abstract successor stays
winning; the work here is realizing the abstract move by an actual `Placement` on
the actual board — the per-(class, piece) geometry, proven once per class, NOT per
adversary order. -/
/-- Membership in `absSafe` implies the abstract state is not lost — the survival
face of the fixpoint. Every iterate is a `filter` by `¬ absLost ∧ …`, so any member
of `absSafe = absIter 327680 = absSafeOp (absIter 327679)` satisfies `¬ absLost`. -/
theorem absSafe_not_absLost {a : AbsState} (h : a ∈ absSafe) : absLost a = false := by
  rw [absSafe_def] at h
  simp only [absSafeOp, Finset.mem_filter] at h
  simpa using h.2.1

theorem abs_simulation (g : GameState) (h : α g ∈ absSafe) :
    ¬ g.lost GameConfig.standard ∧
      ∀ p, p ∈ g.bag →
        ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
          α (adversarialStep GameConfig.standard g p pl) ∈ absSafe := by
  refine ⟨?_, ?_⟩
  · -- SURVIVAL half: PROVEN end-to-end. `absSafe` membership ⟹ `¬ absLost` ⟹ the
    -- surface bucket is `0` ⟹ (by `boardAbs`) the board is not lost.
    intro hlost
    have hfalse : absLost (α g) = false := absSafe_not_absLost h
    have htrue : absLost (α g) = true := by
      unfold absLost α boardAbs
      simp only [decide_eq_true_eq]
      rw [if_pos (show Board.isLost GameConfig.standard g.board from hlost)]
    rw [hfalse] at htrue
    exact Bool.noConfusion htrue
  · -- REALIZATION half: the crux (#66/#72). Needs the real `absMoves`/`boardAbs`
    -- (surface shape, not just the lost-bit) and the per-(class, piece) placement
    -- geometry. NOT closable under the lost-bit-only abstraction.
    sorry

/-! ## 5. Lift to the concrete `safe` GFP — MECHANICAL (no Tetris content)

Pure `gfp` coinduction: `{g | α g ∈ absSafe}` is a `safeOp`-post-fixpoint by
`abs_simulation`, hence contained in `safe = gfp safeOp` via `OrderHom.le_gfp`. -/
theorem concrete_safe_of_absSafe (g : GameState) (h : α g ∈ absSafe) :
    g ∈ safe GameConfig.standard := by
  have hpost :
      {g : GameState | α g ∈ absSafe} ⊆
        safeOp GameConfig.standard {g : GameState | α g ∈ absSafe} := by
    intro g' hg'
    obtain ⟨hnl, hstep⟩ := abs_simulation g' hg'
    exact ⟨hnl, hstep⟩
  exact safe_greatest _ hpost h

/-! ## 6. Close the goal: `init` abstract-safe ⟹ `TetrisSolvableValid`

`init_absSafe` is a FINITE `decide` (does `α init` survive the abstract fixpoint?),
contingent on the classifier `boardAbs`/`absMoves`/`absLost` being concrete. The
final wiring composes `concrete_safe_of_absSafe` with the existing M2/M3 bridge. -/
theorem init_absSafe : α GameState.init ∈ absSafe := by
  -- once boardAbs/absLost/absMoves are concrete: `by decide` or `by native_decide`
  sorry

theorem tetrisSolvableValid_of_abstraction : TetrisSolvableValid := by
  have hinit : GameState.init ∈ safe GameConfig.standard :=
    concrete_safe_of_absSafe GameState.init init_absSafe
  -- bridge `tetrisSolvableValidFor_iff_init_safe` (SafeSet.lean): `init ∈ safe ↔
  -- TetrisSolvableValidFor`, with `4 ≤ standard.cols` discharged by `decide`.
  rw [tetrisSolvableValid_iff_for_standard]
  exact (tetrisSolvableValidFor_iff_init_safe (by decide)).mpr hinit

end Tetris.AbstractSafe

#print axioms Tetris.AbstractSafe.tetrisSolvableValid_of_abstraction
