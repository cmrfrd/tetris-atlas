import Proofs.Safety.SolverFunction

/-!
# Constructing, computing, and discovering the ideal solver function

The previous experiments asked *what* the ideal solver `(board, bag, piece) → placement` is and
*what properties* it has. This file asks the constructive questions: **how do we build it, how do we
compute it, what does it guarantee, and how do we discover whether it exists at all?**

The engine is the retrograde operator `F_finite`: one *round* keeps a state only if it is not lost
and, for every drawable piece, some valid placement lands back in the current candidate set —
otherwise the adversary can force an escape, so the state is removed. Iterating `F_finite` from a
finite universe is exactly **value iteration / retrograde death-propagation**: it shrinks
monotonically to the greatest fixed point, which *is* the winning region `safe`. From a fixed point
containing `init` the solver is extracted by choosing, at each state, a successor that stays in the
set. Existence of the function is then decided by a terminating computation: run the rounds to
convergence and check whether `init` survived.

Abstractly: the solver is a memoryless policy rendering `safe` invariant; `safe` is the greatest
fixed point of a monotone operator on a finite lattice (Knaster–Tarski); the construction is the
descending Kleene iteration to that fixed point; and existence is the decidable membership of
`init`.
-/

namespace Tetris

variable {cfg : GameConfig}

/-! ## Part 1 — One round of construction: the retrograde operator -/

/-- **One construction round only removes states.** Applying `F_finite` to a candidate set yields a
subset: a round prunes now-doomed states and never invents new ones. -/
theorem round_removes_only (S : Finset GameState) : F_finite cfg S ⊆ S :=
  F_finite_subset cfg S

/-- **Survival criterion for a round.** A state survives iff it is not lost and, for every drawable
piece, some valid placement lands back in the candidate set. -/
theorem round_survives_iff (S : Finset GameState) (g : GameState) :
    g ∈ F_finite cfg S ↔ g ∈ S ∧ ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ S :=
  mem_F_finite_iff cfg S g

/-- **A round drops every lost state.** -/
theorem round_drops_lost {S : Finset GameState} {g : GameState} (h : g.lost cfg) :
    g ∉ F_finite cfg S :=
  fun hm => not_lost_of_mem_F_finite hm h

/-- **A round drops any trapped state** — one with a drawable piece for which no valid placement
stays in the candidate set (the adversary plays that piece to force an escape). -/
theorem round_drops_trapped {S : Finset GameState} {g : GameState} {p : Piece}
    (hp : p ∈ g.bag)
    (htrap : ∀ pl ∈ Placement.allValidFor cfg p, adversarialStep cfg g p pl ∉ S) :
    g ∉ F_finite cfg S := by
  intro hm
  obtain ⟨pl, hpl, hstep⟩ := step_exists_of_mem_F_finite hm p hp
  exact htrap pl hpl hstep

/-- **A round is monotone**: more candidates keep at least as many survivors. -/
theorem round_monotone {S T : Finset GameState} (h : S ⊆ T) :
    F_finite cfg S ⊆ F_finite cfg T :=
  F_finite_mono cfg h

end Tetris
