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

/-! ## Part 2 — Fixed points: self-sustaining candidate regions -/

/-- **A region is a fixed point iff it is self-sustaining.** `F_finite S = S` exactly when every
state in `S` is alive and has, for each drawable piece, a valid response landing back in `S`. Such a
region is a complete certificate: the construction can never prune it. -/
theorem round_self_sustaining_iff (S : Finset GameState) :
    F_finite cfg S = S ↔ ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ S := by
  rw [F_finite_eq_self_iff]
  refine ⟨fun h g hg => ((mem_F_finite_iff cfg S g).mp (h g hg)).2, fun h g hg => ?_⟩
  exact mem_F_finite_of hg (h g hg).1 (h g hg).2

/-- **Self-sustaining ⇒ fixed point (constructor).** A region all of whose states are alive with an
in-region response to every piece is a fixed point — a region the construction preserves intact. -/
theorem closed_region_is_fixed {S : Finset GameState}
    (h : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ S) :
    F_finite cfg S = S :=
  F_finite_eq_self_of h

/-- **Any closed cycle is a fixed point.** Exhibiting an adversarial closed cycle is exhibiting a
self-sustaining region — the construction confirms it (prunes nothing). -/
theorem cycle_is_fixed (C : AdversarialClosedCycle cfg) :
    F_finite cfg C.states = C.states :=
  C.F_finite_states_eq_states

/-- **The empty region is the trivial fixed point** — the certificate of "no winning region". -/
theorem empty_is_fixed : F_finite cfg (∅ : Finset GameState) = ∅ :=
  F_finite_empty cfg

/-! ## Part 3 — The construction as a descending iteration -/

/-- The construction starts from the full candidate universe. -/
theorem construct_start (S₀ : Finset GameState) : safeIterFinite cfg S₀ 0 = S₀ :=
  safeIterFinite_zero cfg S₀

/-- Each construction step applies one retrograde round. -/
theorem construct_round (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) = F_finite cfg (safeIterFinite cfg S₀ n) :=
  safeIterFinite_succ cfg S₀ n

/-- Each round prunes: the iterates form a descending chain. -/
theorem construct_descending (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) ⊆ safeIterFinite cfg S₀ n :=
  safeIterFinite_succ_subset cfg S₀ n

/-- The construction is antitone: later rounds are contained in earlier ones. -/
theorem construct_antitone (S₀ : Finset GameState) {m n : ℕ} (h : m ≤ n) :
    safeIterFinite cfg S₀ n ⊆ safeIterFinite cfg S₀ m :=
  safeIterFinite_antitone cfg S₀ h

/-- **The construction terminates** within `|S₀|` rounds at a fixed point. -/
theorem construct_terminates (S₀ : Finset GameState) :
    ∃ N, N ≤ S₀.card ∧ safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N :=
  safeIterFinite_converges cfg S₀

/-- **Strict-or-done**: each round either strictly shrinks the region or is the fixed point. -/
theorem construct_strict_or_done (S₀ : Finset GameState) (n : ℕ) :
    (safeIterFinite cfg S₀ (n + 1)).card < (safeIterFinite cfg S₀ n).card ∨
    safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n :=
  safeIterFinite_strict_or_stable cfg S₀ n

/-- **Stability persists**: once a round is a no-op, every later round is too. -/
theorem construct_stable (S₀ : Finset GameState) {n : ℕ}
    (h : safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n) (k : ℕ) :
    safeIterFinite cfg S₀ (n + k) = safeIterFinite cfg S₀ n :=
  safeIterFinite_stable cfg S₀ h k

/-! ## Part 4 — Correctness of the computed region -/

/-- **Soundness.** The construction's fixed point is contained in the true winning region `safe`:
every surviving state really is survivable. The construction never over-claims. -/
theorem computed_sound (S₀ : Finset GameState) (N : ℕ)
    (h : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (↑(safeIterFinite cfg S₀ N) : Set GameState) ⊆ safe cfg :=
  safeIterFinite_subset_safe cfg S₀ N h

/-- **Completeness.** If the starting universe covers `safe`, every round still contains all of
`safe`: the construction never drops a survivable state. -/
theorem computed_complete (S₀ : Finset GameState) (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ) :
    safe cfg ⊆ ↑(safeIterFinite cfg S₀ n) :=
  safe_subset_safeIterFinite cfg S₀ hS₀ n

/-- **Exactness.** With a covering universe, the fixed point *equals* `safe` pointwise: membership
in the computed region is membership in the winning region — sound and complete together. -/
theorem computed_exact (S₀ : Finset GameState) (N : ℕ) (hS₀ : safe cfg ⊆ ↑S₀)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    g ∈ safe cfg ↔ g ∈ (↑(safeIterFinite cfg S₀ N) : Set GameState) :=
  safe_iff_mem_fixedPoint cfg S₀ N hS₀ hfix g

/-! ## Part 5 — Deciding existence of the function -/

/-- **Existence is decided by the construction.** A solving function exists iff, at the
construction's fixed point, `init` is among the survivors — a finite-set membership check. -/
theorem existence_decided (S₀ : Finset GameState) (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ)
      ↔ GameState.init ∈ safeIterFinite cfg S₀ N :=
  solver_exists_iff_init_in_fixedpoint hcols hS₀ N hfix

/-- **Refutation by pruning.** If `init` is ever pruned at any round, no solving function exists —
one snapshot of the construction with `init` gone is a complete impossibility proof. -/
theorem no_solver_of_init_pruned (S₀ : Finset GameState) (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ) (h : GameState.init ∉ safeIterFinite cfg S₀ n) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  no_solver_of_init_not_mem_safeIterFinite hcols hS₀ n h

/-- **The construction makes safety decidable.** From a covering universe and a fixed-point witness,
membership of any state in `safe` reduces to a finite lookup. -/
def construct_decide_safe (S₀ : Finset GameState) (N : ℕ) (hS₀ : safe cfg ⊆ ↑S₀)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    Decidable (g ∈ safe cfg) :=
  decideSafeFromUniverse cfg S₀ N hS₀ hfix g

/-! ## Part 6 — Extracting the solver from a fixed point -/

/-- **Any fixed point is a verified winning region.** A self-sustaining region (`F_finite S = S`),
whether reached by the construction or exhibited directly, is contained in `safe`: every one of its
states is genuinely survivable. This is the coinductive soundness of a certificate. -/
theorem fixed_point_subset_safe {S : Finset GameState} (hfix : F_finite cfg S = S) :
    (↑S : Set GameState) ⊆ safe cfg := by
  apply safe_greatest
  intro g hg
  rw [Finset.mem_coe] at hg
  have hg' : g ∈ F_finite cfg S := by rw [hfix]; exact hg
  rw [F_finite, Finset.mem_filter] at hg'
  obtain ⟨_, hnlost, hmoves⟩ := hg'
  refine ⟨hnlost, fun p hp => ?_⟩
  obtain ⟨pl, hpl_mem, hstep⟩ := hmoves p hp
  rw [Placement.mem_allValidFor] at hpl_mem
  exact ⟨pl, hpl_mem.1, hpl_mem.2, Finset.mem_coe.mpr hstep⟩

/-- **From a fixed point to the function.** Exhibit any self-sustaining region containing `init`;
then the canonical `safeSolver` is a valid solving program. This is the construction recipe: build a
closed region from the empty board, and the function is extracted. -/
theorem solver_from_fixed_point {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    SolvesTetrisValid cfg (safeSolver cfg) :=
  init_safe_implies_solvesTetrisValid hcols
    (fixed_point_subset_safe hfix (Finset.mem_coe.mpr hinit))

/-! ## Part 7 — The construction computes the greatest fixed point -/

/-- **Every fixed point sits inside every round.** Under a covering universe, any self-sustaining
region `T` is contained in each iterate — the construction never drops a certificate. So the limit
is the *greatest* fixed point: the union of all winning regions, the maximal survivable set. -/
theorem fixed_point_subset_iter {T S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ)
    (hT : F_finite cfg T = T) :
    (↑T : Set GameState) ⊆ ↑(safeIterFinite cfg S₀ n) :=
  fun _ hg => computed_complete S₀ hS₀ n (fixed_point_subset_safe hT hg)

/-- **`safe` is a fixed point of the abstract one-step operator.** The winning region equals its own
one-step safe-preimage — the semantic counterpart of `F_finite S = S`. -/
theorem safe_is_fixed_point : safeOp cfg (safe cfg) = safe cfg :=
  safe_eq cfg

end Tetris
