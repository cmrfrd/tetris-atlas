import Mathlib
import Proofs.Safety.SafeIterate

/-!
# Witness-free existence: what the abstract machinery can and cannot deliver

Can "a solver exists" be proved *purely abstractly* — no enumeration, no
constructed witness? This file assembles the complete witness-free toolkit and
proves the one piece that was missing, so that the exact boundary of the
abstract method becomes a theorem rather than a feeling.

## What comes for free

* **Tarski**: the safe set exists as the greatest fixed point — no construction
  needed (`safe`, already in the library). What Tarski does *not* give is that
  `init` belongs to it.
* **Compactness (NEW here)**: `iInter_safeIterate_subset_safe`, hence
  **`safe_eq_iInter_safeIterate`** — the safe set is *exactly* the intersection
  of the finite-horizon iterates, unconditionally. The proof is König's lemma
  in miniature: at each state only finitely many placements exist, so a
  placement that works at every finite depth can be extracted by pigeonhole
  (`Finite.exists_infinite_fiber`), with no finiteness hypothesis on the state
  space and no convergence assumption.
* **Consequently** (`solvable_iff_forall_horizon`): a solver exists **iff the
  player wins every finite-horizon game**. Infinite survival carries no extra
  content beyond all of its finite approximations.
* **Classical dichotomy** (`solvable_or_finite_refutation`): either Tetris is
  solvable, or some finite depth `N` already refutes it — an `N`-move
  adversarial kill certificate. Tertium non datur; both disjuncts are
  first-order concrete.

## The irreducible core

`solvable_iff_exists_invariant`: **a solver exists iff a closed invariant
exists** — a set of states containing `init`, avoiding loss, and closed under
"some safe reply to every announced piece". The forward direction instantiates
the invariant as `safe` itself; the backward direction is coinduction
(`tetrisSolvableValid_of_invariant`).

This iff is the honest answer to "can we avoid the witness": **no — the witness
is equivalent to the theorem.** Any proof of solvability *is* (up to the
equivalence) the exhibition of a closed invariant; conversely the invariant
need not be a strategy, a lookup table, or an enumeration — any finitely
*described* predicate whose closure is provable symbolically will do. The
abstract machinery reduces "build the Atlas" to "describe one closed shape
family and verify finitely many closure obligations"; it cannot reduce it
further, because by this iff there is no further.

Combined with the counting barrier (`Safety/CountingBarrier`): the invariant's
description must carry geometric content — config-generic bookkeeping holds on
the unsolvable 10×1 board too — so the residue of the whole problem is exactly
the carrier program: one describable, geometry-aware, closed family of boards.
-/

namespace Tetris

/-- **Horizon compactness.** A state surviving every finite horizon is safe
outright: the finite-horizon iterates converge to the greatest fixed point with
no assumptions. König's argument: the intersection of the iterates is itself
closed under `safeOp`, because at each state and piece only finitely many
placements exist, so some single placement must succeed at cofinally many
depths — and then, by antitonicity, at all of them. -/
theorem iInter_safeIterate_subset_safe (cfg : GameConfig) :
    (⋂ n, safeIterate cfg n) ⊆ safe cfg := by
  apply safe_greatest
  intro g hg
  have hmem : ∀ n, g ∈ safeIterate cfg n := Set.mem_iInter.mp hg
  have hsucc : ∀ n, g ∈ safeOp cfg (safeIterate cfg n) := fun n => by
    have h := hmem (n + 1)
    rwa [safeIterate_succ] at h
  refine ⟨(hsucc 0).1, ?_⟩
  intro p hp
  -- choose, for every depth, a placement that survives that depth
  choose pl hpiece hvalid hstep using fun n => (hsucc n).2 p hp
  -- all chosen placements live in the finite move set
  have hmemA : ∀ n, pl n ∈ Placement.allValidFor cfg p := fun n =>
    (Placement.mem_allValidFor cfg p (pl n)).mpr ⟨hpiece n, hvalid n⟩
  -- pigeonhole: one placement is chosen at cofinally many depths
  obtain ⟨q, hq⟩ := Finite.exists_infinite_fiber
    (fun n => (⟨pl n, hmemA n⟩ : {x // x ∈ Placement.allValidFor cfg p}))
  rw [Set.infinite_coe_iff] at hq
  have hcofinal : ∀ m : ℕ, ∃ n, m ≤ n ∧ pl n = (q : Placement) := by
    intro m
    obtain ⟨n, hn, hmn⟩ := hq.exists_gt m
    have : pl n = (q : Placement) := congrArg Subtype.val hn
    exact ⟨n, le_of_lt hmn, this⟩
  -- that placement survives every depth, hence the intersection
  obtain ⟨n₀, -, hEq₀⟩ := hcofinal 0
  refine ⟨(q : Placement), by rw [← hEq₀]; exact hpiece n₀,
    by rw [← hEq₀]; exact hvalid n₀, Set.mem_iInter.mpr ?_⟩
  intro n
  obtain ⟨n', hn', hEq⟩ := hcofinal n
  have h := hstep n'
  rw [hEq] at h
  exact safeIterate_antitone cfg hn' h

/-- **The safe set is exactly the limit of its finite-horizon approximations.**
Unconditional: no finite-universe hypothesis, no convergence stage. -/
theorem safe_eq_iInter_safeIterate (cfg : GameConfig) :
    safe cfg = ⋂ n, safeIterate cfg n :=
  Set.Subset.antisymm (Set.subset_iInter (safe_subset_safeIterate cfg))
    (iInter_safeIterate_subset_safe cfg)

/-- A state is safe iff it survives every finite horizon. -/
theorem mem_safe_iff_forall_safeIterate (cfg : GameConfig) (g : GameState) :
    g ∈ safe cfg ↔ ∀ n, g ∈ safeIterate cfg n := by
  rw [safe_eq_iInter_safeIterate]
  exact Set.mem_iInter

/-- **A solver exists iff the player wins every finite-horizon game.** The
infinite-play statement is the conjunction of its finite approximations —
nothing more. -/
theorem solvable_iff_forall_horizon :
    TetrisSolvableValid ↔
      ∀ n, GameState.init ∈ safeIterate GameConfig.standard n := by
  rw [tetrisSolvableValid_iff_init_safe, mem_safe_iff_forall_safeIterate]

/-- **The classical dichotomy.** Either Tetris is solvable, or some finite
depth already refutes it — a concrete `N`-move adversarial kill certificate.
There is no third possibility, and both sides are finitary statements. -/
theorem solvable_or_finite_refutation :
    TetrisSolvableValid ∨
      ∃ n, GameState.init ∉ safeIterate GameConfig.standard n := by
  rcases Classical.em
      (∀ n, GameState.init ∈ safeIterate GameConfig.standard n) with h | h
  · exact Or.inl (solvable_iff_forall_horizon.mpr h)
  · push Not at h
    exact Or.inr h

/-- **The irreducible core: solvability ⟺ a closed invariant exists.** The
witness cannot be eliminated — it is equivalent to the theorem — but it can be
*compressed*: the invariant is a set of states, not a strategy, a table, or an
enumeration, and any finitely described predicate with a symbolic closure proof
qualifies. This is the exact residue of the existence problem. -/
theorem solvable_iff_exists_invariant :
    TetrisSolvableValid ↔
      ∃ S : Set GameState,
        GameState.init ∈ S ∧
        (∀ g ∈ S, ¬ g.lost GameConfig.standard) ∧
        ∀ g ∈ S, ∀ p, p ∈ g.bag →
          ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard ∧
            adversarialStep GameConfig.standard g p pl ∈ S := by
  constructor
  · intro h
    exact ⟨safe GameConfig.standard, tetrisSolvableValid_iff_init_safe.mp h,
      fun g hg => safe_not_lost hg, fun g hg => safe_forall_step hg⟩
  · rintro ⟨S, hinit, hnl, hstep⟩
    exact tetrisSolvableValid_of_invariant S hinit hnl hstep

end Tetris
