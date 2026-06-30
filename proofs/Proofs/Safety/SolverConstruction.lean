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

/-! ## Part 8 — Running the construction on the canonical universe -/

/-- Run from `inFieldStates`, every round stays inside `inFieldStates`. -/
theorem construct_inField_subset (n : ℕ) :
    safeIterFinite cfg (inFieldStates cfg) n ⊆ inFieldStates cfg :=
  safeIterFinite_inFieldStates_subset cfg n

/-- Run from `inFieldStates`, every round has card at most `|inFieldStates|`. -/
theorem construct_inField_card_le (n : ℕ) :
    (safeIterFinite cfg (inFieldStates cfg) n).card ≤ (inFieldStates cfg).card :=
  safeIterFinite_inFieldStates_card_le cfg n

/-! ## Part 9 — The construction produces the proof artifact -/

/-- **The construction yields a closed Atlas.** From a self-sustaining region containing `init`,
there is a concrete closed Atlas covering `init` — the M4 proof artifact
that discharges solvability. -/
theorem construct_yields_atlas {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    ∃ (A : Atlas cfg) (S' : Finset GameState),
      A.IsClosedOn cfg S' ∧ GameState.init ∈ S' :=
  solver_exists_yields_closed_atlas hcols
    ⟨safeSolver cfg, solver_from_fixed_point hcols hfix hinit⟩

/-- **The construction yields a finite closed cycle.** Equivalently, a WF closed cycle through
`init` — the M2/M3 artifact. -/
theorem construct_yields_cycle {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states :=
  (init_safe_iff_exists_init_adversarialClosedCycleWF cfg).mp
    (fixed_point_subset_safe hfix (Finset.mem_coe.mpr hinit))

/-! ## Part 10 — Death propagation: what each round removes -/

/-- **A round removes exactly the lost-or-trapped states.** A candidate `g ∈ S` is pruned iff it is
already lost, or some drawable piece has no valid placement staying in `S` (the adversary's escape).
This is the backward death-propagation: death spreads from lost states and from traps. -/
theorem round_removes_iff {S : Finset GameState} {g : GameState} (hg : g ∈ S) :
    g ∉ F_finite cfg S ↔ g.lost cfg ∨ ∃ p ∈ g.bag,
      ∀ pl ∈ Placement.allValidFor cfg p, adversarialStep cfg g p pl ∉ S := by
  rw [mem_F_finite_iff]
  constructor
  · intro h
    by_cases hlost : g.lost cfg
    · exact Or.inl hlost
    · refine Or.inr ?_
      by_contra hcon
      refine h ⟨hg, hlost, fun p hp => ?_⟩
      by_contra hp2
      exact hcon ⟨p, hp, fun pl hpl hstep => hp2 ⟨pl, hpl, hstep⟩⟩
  · rintro (hlost | ⟨p, hp, htrap⟩) ⟨_, hnlost, hmoves⟩
    · exact hnlost hlost
    · obtain ⟨pl, hpl, hstep⟩ := hmoves p hp
      exact htrap pl hpl hstep

/-! ## Part 11 — The complete construction pipeline -/

/-- **The construction pipeline, in one statement.** From any universe covering `safe`, there is a
round count `N ≤ |S₀|` at which the construction is at a fixed point that is (1) *sound* (`⊆ safe`),
(2) *exact* (equals `safe` pointwise), (3) *decisive* (a solver exists iff `init` survived), and (4)
*constructive* (if `init` survived, the canonical `safeSolver` is a valid solving program).
Terminate, verify, decide, extract — the recipe for building and discovering the ideal function. -/
theorem construction_pipeline (S₀ : Finset GameState) (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) :
    ∃ N, N ≤ S₀.card ∧
      (↑(safeIterFinite cfg S₀ N) : Set GameState) ⊆ safe cfg ∧
      (∀ g, g ∈ safe cfg ↔ g ∈ (↑(safeIterFinite cfg S₀ N) : Set GameState)) ∧
      ((∃ σ : Solver cfg, SolvesTetrisValid cfg σ)
        ↔ GameState.init ∈ safeIterFinite cfg S₀ N) ∧
      (GameState.init ∈ safeIterFinite cfg S₀ N → SolvesTetrisValid cfg (safeSolver cfg)) := by
  obtain ⟨N, hN, hfix⟩ := safeIterFinite_converges cfg S₀
  exact ⟨N, hN, computed_sound S₀ N hfix, fun g => computed_exact S₀ N hS₀ hfix g,
    existence_decided S₀ hcols hS₀ N hfix,
    fun hinit => init_safe_implies_solvesTetrisValid hcols
      (computed_sound S₀ N hfix (Finset.mem_coe.mpr hinit))⟩

/-! ## Part 12 — Existence in pure construction terms -/

/-- **Existence ⟺ a finite self-sustaining region through `init`.** The ideal function exists if and
only if there is a finite set `S`, fixed by the construction (`F_finite S = S`), containing `init`.
So "discover the function" means "find one finite self-sustaining region from the empty board." -/
theorem exists_solver_iff_exists_fixed_point_through_init (hcols : 4 ≤ cfg.cols) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔
      ∃ S : Finset GameState, F_finite cfg S = S ∧ GameState.init ∈ S := by
  refine ⟨fun hex => ?_, fun ⟨_, hfix, hinit⟩ => ⟨safeSolver cfg,
    solver_from_fixed_point hcols hfix hinit⟩⟩
  obtain ⟨C, hinit⟩ := (init_safe_iff_exists_init_adversarialClosedCycleWF cfg).mp
    ((solver_exists_iff_init_safe hcols).mp hex)
  exact ⟨C.toAdversarialClosedCycle.states,
    cycle_is_fixed C.toAdversarialClosedCycle, hinit⟩

/-! ## Part 13 — Monotonicity and universe-independence -/

/-- **The construction is monotone in its universe.** A larger starting set yields a
pointwise-larger construction at every round. -/
theorem construct_mono_universe {S₀ S₀' : Finset GameState} (h : S₀ ⊆ S₀') (n : ℕ) :
    safeIterFinite cfg S₀ n ⊆ safeIterFinite cfg S₀' n := by
  induction n with
  | zero => simpa using h
  | succ k ih =>
      rw [safeIterFinite_succ, safeIterFinite_succ]
      exact F_finite_mono cfg ih

/-- **Universe-independence of the limit.** At their fixed points, two universes both covering
`safe` compute the same winning region: membership agrees with `safe` for either. -/
theorem limit_universe_independent {S₀ S₀' : Finset GameState} {N N' : ℕ}
    (hS₀ : safe cfg ⊆ ↑S₀) (hS₀' : safe cfg ⊆ ↑S₀')
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (hfix' : safeIterFinite cfg S₀' (N' + 1) = safeIterFinite cfg S₀' N') (g : GameState) :
    g ∈ safeIterFinite cfg S₀ N ↔ g ∈ safeIterFinite cfg S₀' N' := by
  rw [← Finset.mem_coe, ← computed_exact S₀ N hS₀ hfix g,
      computed_exact S₀' N' hS₀' hfix' g, Finset.mem_coe]

/-! ## Part 14 — Complexity of the construction -/

/-- **Termination with a stays-put guarantee.** There is a round `N ≤ |S₀|` after which the
construction is constant: every later round equals round `N`. -/
theorem construct_rounds_bounded (S₀ : Finset GameState) :
    ∃ N, N ≤ S₀.card ∧ ∀ k, N ≤ k → safeIterFinite cfg S₀ k = safeIterFinite cfg S₀ N := by
  obtain ⟨N, hN, hfix⟩ := safeIterFinite_converges cfg S₀
  refine ⟨N, hN, fun k hk => ?_⟩
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hk
  exact safeIterFinite_stable cfg S₀ hfix j

/-- **Each non-final round strictly shrinks the region** — so the construction makes monotone
progress and cannot stall before the fixed point. -/
theorem construct_progress (S₀ : Finset GameState) (n : ℕ)
    (h : safeIterFinite cfg S₀ (n + 1) ≠ safeIterFinite cfg S₀ n) :
    (safeIterFinite cfg S₀ (n + 1)).card < (safeIterFinite cfg S₀ n).card :=
  (construct_strict_or_done S₀ n).resolve_right h

/-! ## Part 15 — The construction on real Tetris -/

/-- **The construction halts on canonical Tetris within `2^207` rounds.** Run from the full in-field
universe (`2^207` states), the retrograde iteration reaches its fixed point in at most `2^207`
rounds — provably terminating on the real game. -/
theorem construct_standard_terminates :
    ∃ N, N ≤ 2 ^ 207 ∧
      safeIterFinite GameConfig.standard (inFieldStates GameConfig.standard) (N + 1)
        = safeIterFinite GameConfig.standard (inFieldStates GameConfig.standard) N := by
  obtain ⟨N, hN, hfix⟩ := construct_terminates (inFieldStates GameConfig.standard)
  rw [standard_inFieldStates_card_eq_two_pow_207] at hN
  exact ⟨N, hN, hfix⟩

/-- **The in-field construction is sound.** Its fixed point is contained in `safe`. -/
theorem construct_inField_sound (N : ℕ)
    (hfix : safeIterFinite cfg (inFieldStates cfg) (N + 1)
      = safeIterFinite cfg (inFieldStates cfg) N) :
    (↑(safeIterFinite cfg (inFieldStates cfg) N) : Set GameState) ⊆ safe cfg :=
  computed_sound (inFieldStates cfg) N hfix

/-! ## Part 16 — Abstract proof method: coinduction -/

/-- **Coinduction principle.** To certify a region as winning it suffices to show it is *one-step
closed*: contained in its own one-step safe-preimage. This is how a winning region is *proved*
without unrolling the infinite horizon — the abstract counterpart of exhibiting a fixed point. -/
theorem coinduction_principle (S : Set GameState) (hS : S ⊆ safeOp cfg S) :
    S ⊆ safe cfg :=
  safe_greatest S hS

/-- **A closed cycle is contained in `safe`.** Its finite state set, being a fixed point, certifies
each of its members as survivable. -/
theorem cycle_subset_safe (C : AdversarialClosedCycle cfg) :
    (↑C.states : Set GameState) ⊆ safe cfg :=
  fixed_point_subset_safe C.F_finite_states_eq_states

/-! ## Part 17 — Properties of the converged region -/

/-- **The limit is a genuine fixed point.** At convergence the computed region is fixed by one
round: applying `F_finite` leaves it unchanged. -/
theorem limit_is_fixed (S₀ : Finset GameState) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    F_finite cfg (safeIterFinite cfg S₀ N) = safeIterFinite cfg S₀ N := by
  rw [← safeIterFinite_succ]; exact hfix

/-- **The limit is self-sustaining.** Every survivor is alive and, for each drawable piece, has a
valid placement landing back in the region — the computed table is complete. -/
theorem limit_self_sustaining (S₀ : Finset GameState) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    {g : GameState} (hg : g ∈ safeIterFinite cfg S₀ N) :
    ¬ g.lost cfg ∧ ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ safeIterFinite cfg S₀ N :=
  (round_self_sustaining_iff _).mp (limit_is_fixed S₀ N hfix) g hg

/-! ## Part 18 — Value iteration: death propagates one layer per round -/

/-- **Newly-doomed characterization.** A state present at round `n` is removed at round `n+1` iff it
is lost or *trapped* relative to the round-`n` region (some piece has no in-region response). Each
round propagates death one layer outward from the lost states — value iteration. -/
theorem newly_doomed_iff (S₀ : Finset GameState) (n : ℕ) {g : GameState}
    (hg : g ∈ safeIterFinite cfg S₀ n) :
    g ∉ safeIterFinite cfg S₀ (n + 1) ↔ g.lost cfg ∨ ∃ p ∈ g.bag,
      ∀ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∉ safeIterFinite cfg S₀ n := by
  rw [safeIterFinite_succ]; exact round_removes_iff hg

/-! ## Part 19 — The explicit solver builder -/

/-- **The solver built from a fixed point.** At each state in the region `S` it picks a valid
placement (for the drawn piece) that lands back in `S` — a witness guaranteed by self-sustainment.
Outside the region it returns the trivial default. This is the explicit construction of the function
from a self-sustaining region. -/
noncomputable def buildSolver {S : Finset GameState} (hfix : F_finite cfg S = S) :
    Solver cfg :=
  fun g p =>
    if h : g ∈ S ∧ p ∈ g.bag then
      Classical.choose (((round_self_sustaining_iff S).mp hfix g h.1).2 p h.2)
    else ⟨p, 0, 0⟩

/-- **Builder spec.** On a region state with a drawable piece, the built solver's choice is a valid
placement that keeps the state in the region. -/
theorem buildSolver_spec {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    buildSolver hfix g p ∈ Placement.allValidFor cfg p ∧
    adversarialStep cfg g p (buildSolver hfix g p) ∈ S := by
  unfold buildSolver
  rw [dif_pos ⟨hg, hp⟩]
  exact Classical.choose_spec (((round_self_sustaining_iff S).mp hfix g hg).2 p hp)

/-- The built solver announces the input piece (on region states). -/
theorem buildSolver_piece {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).piece = p :=
  ((Placement.mem_allValidFor cfg p _).mp (buildSolver_spec hfix hg hp).1).1

/-- The built solver's choice is valid (on region states). -/
theorem buildSolver_valid_at {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).Valid cfg :=
  ((Placement.mem_allValidFor cfg p _).mp (buildSolver_spec hfix hg hp).1).2

/-- The built solver's move keeps the state in the region. -/
theorem buildSolver_step_mem {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (buildSolver hfix g p) ∈ S :=
  (buildSolver_spec hfix hg hp).2

/-- **The built solver's orbit stays in the region.** From `init ∈ S`, every state of the play under
the built solver remains in `S` — the constructed function is confined to its finite region, a
stronger invariant than mere safety. -/
theorem buildSolver_trace_mem {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ S := by
  induction n with
  | zero => simpa using hinit
  | succ k ih =>
      rw [adversarialTrace_succ]
      have hbag : s k ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init k).bag := by
        rw [adversarialTrace_bag]; exact hl k
      exact buildSolver_step_mem hfix ih hbag

/-- **The built solver never tops out.** Every state of its orbit is in the region, and region
states are alive — so it survives forever. -/
theorem buildSolver_survives {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).lost cfg :=
  ((round_self_sustaining_iff S).mp hfix _ (buildSolver_trace_mem hfix hinit hl n)).1

/-- **The built solver is a valid solver.** On region states it plays valid moves; off the region it
plays the in-bounds default — valid everywhere when `cols ≥ 4`. -/
theorem buildSolver_validSolver {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) : ValidSolver cfg (buildSolver hfix) := by
  intro g p hp
  by_cases h : g ∈ S
  · exact ⟨buildSolver_piece hfix h hp, buildSolver_valid_at hfix h hp⟩
  · have hbs : buildSolver hfix g p = ⟨p, 0, 0⟩ := by
      unfold buildSolver; rw [dif_neg (fun hc => h hc.1)]
    rw [hbs]
    refine ⟨rfl, fun cell hcell => ?_⟩
    have hc4 := Piece.shapeUp_col_lt_four p 0 cell hcell
    change 0 + cell.1 < cfg.cols
    omega

/-- **The built solver is a complete valid solving program** (from a fixed point containing
`init`). -/
theorem buildSolver_solvesTetrisValid {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    SolvesTetrisValid cfg (buildSolver hfix) :=
  ⟨buildSolver_validSolver hcols hfix, fun _ hl n => buildSolver_survives hfix hinit hl n⟩

/-- **Construction yields a region-confined solver.** From a self-sustaining region containing
`init`, the built solver survives forever *and* its whole play stays in the finite region `S` — the
function realized as a finite-state controller. -/
theorem construct_confined_solver {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ ∧
      ∀ s, LegalSequence s → ∀ n, adversarialTrace cfg σ s GameState.init n ∈ S :=
  ⟨buildSolver hfix, buildSolver_solvesTetrisValid hcols hfix hinit,
   fun _ hl n => buildSolver_trace_mem hfix hinit hl n⟩

/-- **The built solver's reachable set is contained in the region.** -/
theorem buildSolver_reachable_mem {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) {g : GameState}
    (hr : solverReachable (buildSolver hfix) g) : g ∈ S := by
  induction hr with
  | init => exact hinit
  | step p _ hp ih => exact buildSolver_step_mem hfix ih hp

/-! ## Part 20 — The construction yields a closed atlas explicitly -/

/-- **The built solver's atlas is closed on the region.** The table `(buildSolver hfix).toAtlas`
satisfies all four closure obligations on `S` — a concrete M4 artifact, built from the fixed
point. -/
theorem buildSolver_atlas_closed {S : Finset GameState} (hfix : F_finite cfg S = S) :
    (buildSolver hfix).toAtlas.IsClosedOn cfg S :=
  { not_lost := fun g hg => ((round_self_sustaining_iff S).mp hfix g hg).1
    total := fun _ _ _ _ => rfl
    valid := fun g hg p hp pl hpl => by
      simp only [Solver.toAtlas, Option.some.injEq] at hpl
      subst hpl
      exact ⟨buildSolver_piece hfix hg hp, buildSolver_valid_at hfix hg hp⟩
    closed := fun g hg p hp pl hpl => by
      simp only [Solver.toAtlas, Option.some.injEq] at hpl
      subst hpl
      exact buildSolver_step_mem hfix hg hp }

/-- **The constructed atlas discharges solvability.** A closed atlas containing `init` yields a
solving program — the construction closes the loop from fixed point to proof. -/
theorem buildSolver_atlas_solves {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) : TetrisSolvableFor cfg :=
  (buildSolver_atlas_closed hfix).tetrisSolvableFor_of_init_mem hinit

/-- **The built solver's footprint is bounded by `|S|`.** Its reachable set sits inside `S`, so it
visits at most `|S|` distinct states. -/
theorem buildSolver_footprint_card_le {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) :
    ∀ g, solverReachable (buildSolver hfix) g → g ∈ S :=
  fun _ hr => buildSolver_reachable_mem hfix hinit hr

/-! ## Part 21 — Unconditional discovery of existence -/

/-- **Unconditional positive test.** Run the construction for `|S₀|` rounds from *any* universe; if
`init` still survives, a solving program exists — no coverage hypothesis needed. This is the
discovery test: pick a universe, iterate `|S₀|` times, check `init`. -/
theorem solver_exists_of_init_survives (hcols : 4 ≤ cfg.cols) (S₀ : Finset GameState)
    (h : GameState.init ∈ safeIterFinite cfg S₀ S₀.card) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  tetrisSolvableValidFor_of_init_mem_safeIterFinite_at_S₀_card hcols h

/-- **Soundness at the round bound.** Any state surviving to round `|S₀|` is genuinely in `safe`. -/
theorem survives_card_rounds_safe (S₀ : Finset GameState) {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ S₀.card) : g ∈ safe cfg :=
  mem_safe_of_mem_safeIterFinite_at_S₀_card h

/-- **Standard discovery test.** On canonical 10×20 Tetris, if `init` survives `|S₀|` rounds from
any universe, the game is solvable. -/
theorem standard_solver_exists_of_init_survives (S₀ : Finset GameState)
    (h : GameState.init ∈ safeIterFinite GameConfig.standard S₀ S₀.card) :
    TetrisSolvableValid :=
  tetrisSolvableValid_of_init_mem_safeIterFinite_at_S₀_card_standard h

/-- **Tiny discovery test.** Same, on the 4×4 toy board — finitely checkable. -/
theorem tiny_solver_exists_of_init_survives (S₀ : Finset GameState)
    (h : GameState.init ∈ safeIterFinite GameConfig.tiny S₀ S₀.card) :
    TetrisSolvableValidFor GameConfig.tiny :=
  tetrisSolvableValidFor_tiny_of_init_mem_safeIterFinite_at_S₀_card h

/-! ## Part 22 — Detecting impossibility and the rank structure -/

/-- **Empty region ⇒ no solver.** If the construction ever collapses to `∅` from a covering
universe, no solving program exists — the construction certifies impossibility. -/
theorem no_solver_of_construct_empty {S₀ : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) {n : ℕ} (hempty : safeIterFinite cfg S₀ n = ∅) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  no_solver_of_safeIterFinite_empty hcols hS₀ hempty

/-- **Survival is monotone in rounds.** A state surviving round `n+1` survived round `n` — survival
times are downward closed; each state has a well-defined removal round (its rank). -/
theorem survives_succ_survives (S₀ : Finset GameState) (n : ℕ) {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ (n + 1)) : g ∈ safeIterFinite cfg S₀ n :=
  safeIterFinite_succ_subset cfg S₀ n h

/-- **Safe states are permanent survivors.** Under a covering universe, a survivable state is in
every round — it is never pruned. -/
theorem safe_survives_all_rounds {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ)
    {g : GameState} (hg : g ∈ safe cfg) : g ∈ safeIterFinite cfg S₀ n :=
  Finset.mem_coe.mp (computed_complete S₀ hS₀ n hg)

/-- **Converged by round `|S₀|`.** Under coverage, membership at round `|S₀|` is exactly
`safe`-membership — the construction has definitely settled by then. -/
theorem mem_round_card_iff_safe {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (g : GameState) :
    g ∈ safeIterFinite cfg S₀ S₀.card ↔ g ∈ safe cfg :=
  ⟨mem_safe_of_mem_safeIterFinite_at_S₀_card,
   fun hg => safe_survives_all_rounds hS₀ S₀.card hg⟩

/-! ## Part 23 — Existence is computably decidable -/

/-- **Existence is decidable.** From a covering universe and a fixed-point witness, whether the
ideal function exists is a computable yes/no — `init`'s membership in the converged region. -/
def construct_decide_existence (S₀ : Finset GameState) (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    Decidable (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :=
  decidable_of_iff (GameState.init ∈ safeIterFinite cfg S₀ N)
    (existence_decided S₀ hcols hS₀ N hfix).symm

/-! ## Part 24 — The lattice of fixed points -/

/-- **Every fixed point is a subset of the limit.** Under coverage, any self-sustaining region is
contained in each iterate — the limit is the greatest fixed point as a `Finset`. -/
theorem fixed_subset_limit {S₀ T : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hT : F_finite cfg T = T) :
    T ⊆ safeIterFinite cfg S₀ N :=
  fun _ hg => Finset.mem_coe.mp (fixed_point_subset_iter hS₀ N hT (Finset.mem_coe.mpr hg))

/-- **`∅` is the bottom fixed point.** The construction's fixed points include the empty region (no
winning states) and are bounded above by the limit — they form a bounded family. -/
theorem empty_fixed_subset (S : Finset GameState) : (∅ : Finset GameState) ⊆ S :=
  Finset.empty_subset S

/-! ## Part 25 — The doomed states -/

/-- **Death propagation computes the complement of `safe`.** Under coverage, a state is pruned by
the construction iff it is genuinely unsurvivable — the removed states are exactly the doomed. -/
theorem doomed_iff_not_safe {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    g ∉ safeIterFinite cfg S₀ N ↔ g ∉ safe cfg := by
  rw [← Finset.mem_coe, ← computed_exact S₀ N hS₀ hfix g]

/-! ## Part 26 — The canonical universe instantiation -/

/-- **Canonical existence test.** Run the construction on `inFieldStates` for `|inFieldStates|`
rounds; if `init` survives, the ideal function exists — the actual atlas-builder's verdict. -/
theorem inField_solver_exists_of_init_survives (hcols : 4 ≤ cfg.cols)
    (h : GameState.init
      ∈ safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  solver_exists_of_init_survives hcols (inFieldStates cfg) h

/-- **Canonical soundness at the round bound.** A state surviving `|inFieldStates|` rounds from the
canonical universe is genuinely survivable. -/
theorem inField_survives_card_safe {g : GameState}
    (h : g ∈ safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card) :
    g ∈ safe cfg :=
  survives_card_rounds_safe (inFieldStates cfg) h

/-! ## Part 27 — The construction grand synthesis -/

/-- **Everything from one fixed point.** A self-sustaining region containing `init` yields, all at
once: a valid solving program, the guarantee that its play stays confined to the region, the closed
atlas it induces, and the solvability proof. The complete deliverable of the construction. -/
theorem construction_grand_synthesis {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    SolvesTetrisValid cfg (buildSolver hfix) ∧
    (∀ s, LegalSequence s → ∀ n,
      adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ S) ∧
    (buildSolver hfix).toAtlas.IsClosedOn cfg S ∧
    TetrisSolvableFor cfg :=
  ⟨buildSolver_solvesTetrisValid hcols hfix hinit,
   fun _ hl n => buildSolver_trace_mem hfix hinit hl n,
   buildSolver_atlas_closed hfix,
   buildSolver_atlas_solves hfix hinit⟩

/-! ## Part 28 — Algebra of the operator -/

/-- `F_finite` is idempotent at a fixed point. -/
theorem F_idempotent_at_fixed {S : Finset GameState} (hfix : F_finite cfg S = S) :
    F_finite cfg (F_finite cfg S) = F_finite cfg S := by conv_lhs => rw [hfix]

/-- One round never grows the candidate set's cardinality. -/
theorem round_card_le (S : Finset GameState) : (F_finite cfg S).card ≤ S.card :=
  F_finite_card_le cfg S

/-- The limit is its own `F_finite` image (re-stated as a fixed point of the round). -/
theorem limit_round_fixed (S₀ : Finset GameState) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    F_finite cfg (F_finite cfg (safeIterFinite cfg S₀ N))
      = F_finite cfg (safeIterFinite cfg S₀ N) :=
  F_idempotent_at_fixed (limit_is_fixed S₀ N hfix)

/-! ## Part 29 — Tiny-board instantiation -/

/-- The construction halts on the 4×4 board within `|inFieldStates|` rounds. -/
theorem construct_tiny_terminates :
    ∃ N, N ≤ (inFieldStates GameConfig.tiny).card ∧
      safeIterFinite GameConfig.tiny (inFieldStates GameConfig.tiny) (N + 1)
        = safeIterFinite GameConfig.tiny (inFieldStates GameConfig.tiny) N :=
  construct_terminates (inFieldStates GameConfig.tiny)

/-- The tiny in-field construction is sound. -/
theorem construct_tiny_sound (N : ℕ)
    (hfix : safeIterFinite GameConfig.tiny (inFieldStates GameConfig.tiny) (N + 1)
      = safeIterFinite GameConfig.tiny (inFieldStates GameConfig.tiny) N) :
    (↑(safeIterFinite GameConfig.tiny (inFieldStates GameConfig.tiny) N) : Set GameState)
      ⊆ safe GameConfig.tiny :=
  construct_inField_sound N hfix

/-! ## Part 30 — The constructed function inherits the function-properties -/

/-- The constructed function's output lies in the finite per-piece menu. -/
theorem buildSolver_output_in_menu {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    buildSolver hfix g p ∈ Placement.allValidFor cfg p :=
  solver_output_in_action_set (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output encodes to a single integer `< 4·cols`. -/
theorem buildSolver_output_code_lt {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    4 * (buildSolver hfix g p).col + ((buildSolver hfix g p).rot : ℕ) < 4 * cfg.cols :=
  solver_output_code_lt (buildSolver_validSolver hcols hfix) hp

/-- The constructed function has finite range. -/
theorem buildSolver_range_finite {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) :
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ buildSolver hfix g p = pl}.Finite :=
  solver_range_finite (buildSolver_validSolver hcols hfix)

/-- The constructed function reads exactly `(board, bag, piece)`. -/
theorem buildSolver_reads_board_bag {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g₁ g₂ : GameState) (p : Piece) (hb : g₁.board = g₂.board) (hbag : g₁.bag = g₂.bag) :
    buildSolver hfix g₁ p = buildSolver hfix g₂ p :=
  solver_reads_board_bag g₁ g₂ p hb hbag

/-- The constructed function's orbit keeps the max height within the ceiling. -/
theorem buildSolver_trace_maxHeight {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board
      ≤ cfg.rows :=
  solver_maintains_maxHeight (buildSolver_solvesTetrisValid hcols hfix hinit).2 hl n

/-- The constructed function's orbit keeps the cell count within capacity. -/
theorem buildSolver_trace_count {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board.count
      ≤ cfg.cols * cfg.rows :=
  solver_count_le_capacity (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's play stays in `inFieldStates`. -/
theorem buildSolver_trace_in_field {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ inFieldStates cfg :=
  solver_trace_mem_inFieldStates (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's play is eventually periodic. -/
theorem buildSolver_eventually_repeats {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≠ j ∧
      adversarialTrace cfg (buildSolver hfix) s GameState.init i
        = adversarialTrace cfg (buildSolver hfix) s GameState.init j :=
  solver_play_eventually_repeats (buildSolver_solvesTetrisValid hcols hfix hinit) hl

/-- The constructed function is forced to clear inside every capacity-sized window. -/
theorem buildSolver_clears_within {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) {M : ℕ} (hM : cfg.cols * cfg.rows < 4 * M) :
    ∃ k < M, (adversarialTrace cfg (buildSolver hfix) s GameState.init (k + 1)).board.count
           ≠ (adversarialTrace cfg (buildSolver hfix) s GameState.init k).board.count + 4 :=
  solver_clears_within (buildSolver_solvesTetrisValid hcols hfix hinit) hl hM

/-- The constructed function's cell count is 4-Lipschitz. -/
theorem buildSolver_count_lipschitz {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)).board.count
      ≤ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board.count + 4 :=
  solver_trace_count_le_succ (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function holds the surface-area energy within capacity. -/
theorem buildSolver_trace_surfaceArea {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.surfaceArea cfg
      (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board ≤ cfg.cols * cfg.rows :=
  solver_surfaceArea_le_capacity (buildSolver_solvesTetrisValid hcols hfix hinit).2 hl n

/-- The constructed function holds the hole-debt within capacity. -/
theorem buildSolver_trace_debt {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.debt cfg
      (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board ≤ cfg.cols * cfg.rows :=
  solver_debt_le_capacity (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function keeps every playable column within the ceiling. -/
theorem buildSolver_columns_le {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) {j : ℕ} (hj : j < cfg.cols) :
    Board.colHeight (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board j
      ≤ cfg.rows :=
  solver_columns_le_rows (buildSolver_solvesTetrisValid hcols hfix hinit).2 hl n hj

/-- The constructed function never places a cell in the death zone. -/
theorem buildSolver_no_death_cell {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) {c : Coord}
    (hc : c ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board) :
    c.2 < cfg.rows :=
  solver_no_cell_in_death_zone (buildSolver_solvesTetrisValid hcols hfix hinit).2 hl n hc

/-- The constructed function's boards are well-formed. -/
theorem buildSolver_trace_wf {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board :=
  solver_board_wf (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's states are reachable from the empty board. -/
theorem buildSolver_trace_reachable {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n) :=
  solver_states_reachable_from_empty (buildSolver_solvesTetrisValid hcols hfix hinit)
    (adversarialTrace_solverReachable (buildSolver hfix) hl n)

/-- For the constructed function, survival is exactly `maxHeight ≤ rows`. -/
theorem buildSolver_not_lost_iff {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).lost cfg ↔
      Board.maxHeight cfg
        (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board ≤ cfg.rows :=
  solver_not_lost_iff_maxHeight (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- No legal sequence ever beats the constructed function. -/
theorem buildSolver_no_killing_sequence {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    ¬ ∃ (s : ℕ → Piece) (n : ℕ),
        LegalSequence s ∧
        (adversarialTrace cfg (buildSolver hfix) s GameState.init n).lost cfg :=
  solver_no_killing_sequence (buildSolver_solvesTetrisValid hcols hfix hinit).2

/-- The constructed function keeps an even cell count (even-width boards). -/
theorem buildSolver_even_count {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) (hev : Even cfg.cols)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Even (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board.count :=
  solver_even_count hev (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's max height is 4-Lipschitz. -/
theorem buildSolver_maxHeight_lipschitz {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)).board
      ≤ Board.maxHeight cfg
          (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board + 4 :=
  solver_trace_maxHeight_le_succ (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's three pillars: in `safe`, `maxHeight ≤ rows`, `count ≤ cols·rows`. -/
theorem buildSolver_pillars {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ safe cfg ∧
    Board.maxHeight cfg
      (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board ≤ cfg.rows ∧
    (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board.count
      ≤ cfg.cols * cfg.rows :=
  solving_program_pillars (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's output reduces to `⟨p, rot, col⟩`. -/
theorem buildSolver_output_eq_mk {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    buildSolver hfix g p = ⟨p, (buildSolver hfix g p).rot, (buildSolver hfix g p).col⟩ :=
  solver_output_eq_mk (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output is a point in the `Rotation × range cols` grid. -/
theorem buildSolver_output_in_grid {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    ((buildSolver hfix g p).rot, (buildSolver hfix g p).col)
      ∈ (Finset.univ : Finset Rotation) ×ˢ Finset.range cfg.cols :=
  solver_output_in_grid (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's move places exactly 4 cells. -/
theorem buildSolver_places_four {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) :
    ((buildSolver hfix g p).place b).count = b.count + 4 :=
  solver_output_places_four b g p

/-- The constructed function's move preserves well-formedness. -/
theorem buildSolver_move_wf {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    Board.WF cfg ((buildSolver hfix g p).place b) :=
  solver_move_preserves_wf (buildSolver_validSolver hcols hfix) hp hWF

/-- The constructed function is causal: identical past input gives identical play. -/
theorem buildSolver_causal {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s s' : ℕ → Piece) (g0 : GameState) (n : ℕ) (h : ∀ i < n, s i = s' i)
    (k : ℕ) (hk : k ≤ n) :
    adversarialTrace cfg (buildSolver hfix) s g0 k
      = adversarialTrace cfg (buildSolver hfix) s' g0 k :=
  solver_is_causal s s' g0 n h k hk

/-- The constructed function's dynamics are Markov: the next state depends on the current state. -/
theorem buildSolver_markov {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)
      = adversarialStep cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)
          (buildSolver hfix (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)) :=
  solver_markov_step s n

/-- The constructed function is a section of the piece-projection. -/
theorem buildSolver_section {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (g : GameState) {p : Piece} (hp : p ∈ g.bag) :
    (Placement.piece ∘ buildSolver hfix g) p = p :=
  solver_section_of_piece (buildSolver_validSolver hcols hfix) g hp

/-- The constructed function as the uncurried `(GameState × Piece) → Placement` map. -/
theorem buildSolver_uncurry {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    Function.uncurry (buildSolver hfix) (g, p) = buildSolver hfix g p :=
  solver_uncurry_apply g p

/-- The constructed function's atlas is total. -/
theorem buildSolver_toAtlas_isSome {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    ((buildSolver hfix).toAtlas g p).isSome = true :=
  solver_toAtlas_isSome g p

/-- The constructed function round-trips through its atlas. -/
theorem buildSolver_toAtlas_toSolver {S : Finset GameState} (hfix : F_finite cfg S = S) :
    (buildSolver hfix).toAtlas.toSolver = buildSolver hfix :=
  solver_toAtlas_toSolver

/-- The constructed function's board orbit applies its outputs to the running board. -/
theorem buildSolver_trace_board_succ {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)).board
      = (buildSolver hfix (adversarialTrace cfg (buildSolver hfix) s GameState.init n)
          (s n)).applyStep cfg
          (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board :=
  solver_trace_board_succ (buildSolver_validSolver hcols hfix) hl n

/-- The constructed function's play depends only on past input (no lookahead). -/
theorem buildSolver_no_lookahead {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s s' : ℕ → Piece) (n : ℕ) (h : ∀ i < n, s i = s' i) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n
      = adversarialTrace cfg (buildSolver hfix) s' GameState.init n :=
  solver_is_causal s s' GameState.init n h n (le_refl n)

/-- The constructed function's opening move is non-losing. -/
theorem buildSolver_opening_safe {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hrows : 4 ≤ cfg.rows) {p : Piece}
    (hp : p ∈ GameState.init.bag) :
    ¬ Board.isLost cfg
        ((buildSolver hfix GameState.init p).applyStep cfg GameState.init.board) :=
  solver_opening_move_safe (buildSolver_validSolver hcols hfix) hrows hp

/-- The constructed function's per-piece image has at most `4·cols` distinct outputs. -/
theorem buildSolver_image_card {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {p : Piece} (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => buildSolver hfix g p)).card ≤ cfg.cols * 4 :=
  solver_image_per_piece_card_le (buildSolver_validSolver hcols hfix) T hT

/-- The constructed function is never stuck: every reachable bag piece has a valid placement. -/
theorem buildSolver_never_stuck {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) {p : Piece}
    (hp : p ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  solver_never_stuck (buildSolver_solvesTetrisValid hcols hfix hinit) hl n hp

/-- Every state reachable under the constructed function is safe: no dead ends. -/
theorem buildSolver_no_dead_ends {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {g : GameState}
    (hr : solverReachable (buildSolver hfix) g) : g ∈ safe cfg :=
  solver_no_dead_ends (buildSolver_solvesTetrisValid hcols hfix hinit) hr

/-- Along any legal play, the constructed function's chosen move is on the menu. -/
theorem buildSolver_play_outputs_in_menu {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    buildSolver hfix (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)
      ∈ Placement.allValidFor cfg (s n) :=
  solver_play_outputs_in_menu (buildSolver_validSolver hcols hfix) s hl n

/-- For any piece the current bag can draw, the constructed function returns a valid placement. -/
theorem buildSolver_output_valid {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).Valid cfg :=
  solver_output_valid (buildSolver_validSolver hcols hfix) hp

/-- The constructed function answers the piece it was asked about. -/
theorem buildSolver_announces_piece {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).piece = p :=
  solver_output_announces_piece (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output lives in the global action set (union over all pieces). -/
theorem buildSolver_output_in_total_action_set {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    buildSolver hfix g p ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  solver_output_in_total_action_set (buildSolver_validSolver hcols hfix) hp

/-- Full compressed portrait of one constructed-function output: piece, col, rot, code, menu. -/
theorem buildSolver_output_compressed {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).piece = p ∧
    (buildSolver hfix g p).col < cfg.cols ∧
    ((buildSolver hfix g p).rot : ℕ) < 4 ∧
    4 * (buildSolver hfix g p).col + ((buildSolver hfix g p).rot : ℕ) < 4 * cfg.cols ∧
    buildSolver hfix g p ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  solver_output_compressed (buildSolver_validSolver hcols hfix) hp

/-- On a piece-`p` slice larger than the menu, the constructed function collides (pigeonhole). -/
theorem buildSolver_per_piece_noninjective {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {p : Piece} (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) (hcard : (Placement.allValidFor cfg p).card < T.card) :
    ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧ buildSolver hfix g₁ p = buildSolver hfix g₂ p :=
  solver_per_piece_noninjective (buildSolver_validSolver hcols hfix) T hT hcard

/-- The placements the constructed function realizes along any legal play form a finite set. -/
theorem buildSolver_realized_outputs_finite {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (s : ℕ → Piece) (hl : LegalSequence s) :
    (Set.range fun n =>
      buildSolver hfix
        (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)).Finite :=
  solver_realized_outputs_finite (buildSolver_validSolver hcols hfix) s hl

/-- A no-clear run of the constructed function is capacity-bounded: `4·n ≤ cols·rows`. -/
theorem buildSolver_no_clear_window_bounded {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ)
    (hno : ∀ k < n,
        (adversarialTrace cfg (buildSolver hfix) s GameState.init (k + 1)).board.count
          = (adversarialTrace cfg (buildSolver hfix) s GameState.init k).board.count + 4) :
    4 * n ≤ cfg.cols * cfg.rows :=
  solver_no_clear_window_bounded (buildSolver_solvesTetrisValid hcols hfix hinit) hl n hno

/-- The constructed function's response table at a state has at most `bag.card` distinct entries. -/
theorem buildSolver_response_table_card_le {S : Finset GameState}
    (hfix : F_finite cfg S = S) (g : GameState) :
    (g.bag.image (fun p => buildSolver hfix g p)).card ≤ g.bag.card :=
  solver_response_table_card_le (σ := buildSolver hfix) g

/-- If the constructed function's trace and the piece stream both repeat, the orbit is periodic. -/
theorem buildSolver_periodic_play {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (g0 : GameState) {b d : ℕ}
    (htrace : adversarialTrace cfg (buildSolver hfix) s g0 b
        = adversarialTrace cfg (buildSolver hfix) s g0 (b + d))
    (hs : ∀ k, s (b + k) = s (b + d + k)) (k : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s g0 (b + k)
      = adversarialTrace cfg (buildSolver hfix) s g0 (b + d + k) :=
  solver_periodic_play (σ := buildSolver hfix) s g0 htrace hs k

/-- Each piece presented to the constructed function along legal play is in the current bag. -/
theorem buildSolver_queried_in_bag {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    s n ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).bag :=
  solver_queried_in_bag (σ := buildSolver hfix) s hl n

/-- Every placement the constructed function emits drops exactly four cells. -/
theorem buildSolver_output_dropped_card {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) :
    ((buildSolver hfix g p).dropped b).card = 4 :=
  solver_output_dropped_card (σ := buildSolver hfix) b g p

/-- Output portrait: the constructed function's move announces, validates, fits, and is on menu. -/
theorem buildSolver_output_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).piece = p ∧
    (buildSolver hfix g p).Valid cfg ∧
    (buildSolver hfix g p).col < cfg.cols ∧
    buildSolver hfix g p ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  solver_output_portrait (buildSolver_validSolver hcols hfix) hp

/-- Response portrait: per state the constructed function is an injective menu section. -/
theorem buildSolver_response_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (g : GameState) :
    (∀ p ∈ g.bag, buildSolver hfix g p ∈ Placement.allValidFor cfg p) ∧
    (g.bag.image (fun p => buildSolver hfix g p)).card = g.bag.card ∧
    Set.InjOn (fun p => buildSolver hfix g p) g.bag :=
  solver_response_portrait (buildSolver_validSolver hcols hfix) g

/-- The constructed function maps distinct bag pieces to distinct placements. -/
theorem buildSolver_outputs_differ_by_piece {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p p' : Piece}
    (hp : p ∈ g.bag) (hp' : p' ∈ g.bag) (hne : p ≠ p') :
    buildSolver hfix g p ≠ buildSolver hfix g p' :=
  solver_outputs_differ_by_piece (buildSolver_validSolver hcols hfix) hp hp' hne

/-- Two constructed-function outputs for the same piece coincide once rot and col agree. -/
theorem buildSolver_eq_of_rotcol {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g₁ g₂ : GameState} {p : Piece}
    (hp₁ : p ∈ g₁.bag) (hp₂ : p ∈ g₂.bag)
    (hr : (buildSolver hfix g₁ p).rot = (buildSolver hfix g₂ p).rot)
    (hc : (buildSolver hfix g₁ p).col = (buildSolver hfix g₂ p).col) :
    buildSolver hfix g₁ p = buildSolver hfix g₂ p :=
  solver_eq_of_rotcol (buildSolver_validSolver hcols hfix) hp₁ hp₂ hr hc

/-- When the constructed function differs for one piece, it differs in rotation or column. -/
theorem buildSolver_diff_in_rotcol {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g₁ g₂ : GameState} {p : Piece}
    (hp₁ : p ∈ g₁.bag) (hp₂ : p ∈ g₂.bag) (hne : buildSolver hfix g₁ p ≠ buildSolver hfix g₂ p) :
    (buildSolver hfix g₁ p).rot ≠ (buildSolver hfix g₂ p).rot ∨
      (buildSolver hfix g₁ p).col ≠ (buildSolver hfix g₂ p).col :=
  solver_diff_in_rotcol (buildSolver_validSolver hcols hfix) hp₁ hp₂ hne

/-- The constructed function's atlas wrapper returns `some` of its placement everywhere. -/
theorem buildSolver_toAtlas_apply {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    (buildSolver hfix).toAtlas g p = some (buildSolver hfix g p) :=
  solver_toAtlas_apply (σ := buildSolver hfix) g p

/-- Trace composition (semigroup law) for the constructed function's orbit. -/
theorem buildSolver_trace_compose {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (g0 : GameState) (n m : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s g0 (n + m) =
      adversarialTrace cfg (buildSolver hfix) (fun k => s (n + k))
        (adversarialTrace cfg (buildSolver hfix) s g0 n) m :=
  solver_trace_compose (σ := buildSolver hfix) s g0 n m

/-- Under a constant piece stream the constructed function's trace is an iterated step map. -/
theorem buildSolver_trace_const_eq_iterate {S : Finset GameState} (hfix : F_finite cfg S = S)
    (p : Piece) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) (fun _ => p) GameState.init n
      = (solverStep cfg (buildSolver hfix) p)^[n] GameState.init :=
  solver_trace_const_eq_iterate (σ := buildSolver hfix) p n

/-- One-step unfolding of the constructed function's trace. -/
theorem buildSolver_trace_eq_solverStep {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)
      = solverStep cfg (buildSolver hfix) (s n)
          (adversarialTrace cfg (buildSolver hfix) s GameState.init n) :=
  solver_trace_eq_solverStep (σ := buildSolver hfix) s n

/-- Every trace state visited by the constructed function under legal play is safe. -/
theorem buildSolver_trace_mem_safe {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ safe cfg :=
  solver_trace_mem_safe (buildSolver_solvesTetrisValid hcols hfix hinit) hl n

/-- The constructed function's trace starts at the initial state. -/
theorem buildSolver_trace_zero {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init 0 = GameState.init :=
  solver_trace_zero (σ := buildSolver hfix) s

/-- Every state reachable by the constructed function is both safe and reachable. -/
theorem buildSolver_operates_in_safe_and_reachable {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {g : GameState}
    (hr : solverReachable (buildSolver hfix) g) :
    g ∈ safe cfg ∧ Reachable cfg g :=
  solver_operates_in_safe_and_reachable (buildSolver_solvesTetrisValid hcols hfix hinit) hr

/-- The constructed function's reachable set is step-closed. -/
theorem buildSolver_reachable_step_closed {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hr : solverReachable (buildSolver hfix) g) {p : Piece} (hp : p ∈ g.bag) :
    solverReachable (buildSolver hfix) (adversarialStep cfg g p (buildSolver hfix g p)) :=
  solver_reachable_step_closed (σ := buildSolver hfix) hr hp

/-- The constructed function's play repeats a state within `|inFieldStates|` steps. -/
theorem buildSolver_repeats_within_inFieldStates {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ (inFieldStates cfg).card ∧
      adversarialTrace cfg (buildSolver hfix) s GameState.init i
        = adversarialTrace cfg (buildSolver hfix) s GameState.init j :=
  solver_repeats_within_inFieldStates (buildSolver_solvesTetrisValid hcols hfix hinit) hl

/-- The constructed function's per-state response table has exactly `bag.card` entries. -/
theorem buildSolver_response_table_card_eq {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (g : GameState) :
    (g.bag.image (fun p => buildSolver hfix g p)).card = g.bag.card :=
  solver_response_table_card_eq (buildSolver_validSolver hcols hfix) g

/-- Every state reachable by the constructed function lies in the finite in-field universe. -/
theorem buildSolver_active_domain_finite {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {g : GameState}
    (hr : solverReachable (buildSolver hfix) g) :
    g ∈ inFieldStates cfg :=
  solver_active_domain_finite (buildSolver_solvesTetrisValid hcols hfix hinit) hr

/-- Atlas portrait of the constructed function: total, some-valued, round-trips, injective. -/
theorem buildSolver_atlas_portrait {S : Finset GameState} (hfix : F_finite cfg S = S) :
    (∀ g p, (buildSolver hfix).toAtlas g p = some (buildSolver hfix g p)) ∧
    (∀ g p, ((buildSolver hfix).toAtlas g p).isSome = true) ∧
    ((buildSolver hfix).toAtlas.toSolver = buildSolver hfix) ∧
    (∀ (σ₁ σ₂ : Solver cfg), σ₁.toAtlas = σ₂.toAtlas → σ₁ = σ₂) :=
  solver_atlas_portrait (σ := buildSolver hfix)

/-- Near capacity, any surviving constructed-function move must clear a line. -/
theorem buildSolver_move_must_clear {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) (hnear : cfg.cols * cfg.rows < b.count + 4)
    (hsurv : ¬ Board.isLost cfg ((buildSolver hfix g p).applyStep cfg b)) :
    0 < Board.linesCleared cfg ((buildSolver hfix g p).place b) :=
  solver_move_must_clear (buildSolver_validSolver hcols hfix) hp hWF hnear hsurv

/-- A constructed-function move adds at most four to the cell count. -/
theorem buildSolver_move_count_le {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    ((buildSolver hfix g p).applyStep cfg b).count ≤ b.count + 4 :=
  solver_move_count_le (buildSolver_validSolver hcols hfix) hp hWF

/-- A constructed-function move raises max height by at most four. -/
theorem buildSolver_move_maxHeight_le {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) (b : Board) :
    Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b) ≤ Board.maxHeight cfg b + 4 :=
  solver_move_maxHeight_le (buildSolver_validSolver hcols hfix) hp b

/-- A constructed-function move recovers at most `cols·4` cells against the +4 added. -/
theorem buildSolver_move_recovery_bounded {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    b.count + 4 ≤ ((buildSolver hfix g p).applyStep cfg b).count + cfg.cols * 4 :=
  solver_move_recovery_bounded (buildSolver_validSolver hcols hfix) hp hWF hnf

/-- Clearing in a constructed-function move never increases hole-debt. -/
theorem buildSolver_clear_reduces_debt {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    HoleDebt.debt cfg ((buildSolver hfix g p).applyStep cfg b)
      ≤ HoleDebt.debt cfg ((buildSolver hfix g p).place b) :=
  solver_clear_reduces_debt (buildSolver_validSolver hcols hfix) hp hWF

/-- On a low stack the constructed function's move is non-losing. -/
theorem buildSolver_lowstack_move_safe {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hrows : 4 ≤ cfg.rows) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b)
    (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4) :
    ¬ Board.isLost cfg ((buildSolver hfix g p).applyStep cfg b) :=
  solver_lowstack_move_safe (buildSolver_validSolver hcols hfix) hrows hp hWF hlow

/-- The construction terminates in `≤ |S₀|` rounds and decides solver existence by an init test. -/
theorem construct_decision_procedure {S₀ : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) :
    ∃ N, N ≤ S₀.card ∧
      (↑(safeIterFinite cfg S₀ N) : Set GameState) ⊆ safe cfg ∧
      ((∃ σ : Solver cfg, SolvesTetrisValid cfg σ)
        ↔ GameState.init ∈ safeIterFinite cfg S₀ N) :=
  solver_decision_procedure hcols hS₀

/-- Existence dichotomy: either the canonical solver solves, or no solver does. -/
theorem construct_existence_dichotomy (hcols : 4 ≤ cfg.cols) :
    SolvesTetrisValid cfg (safeSolver cfg) ∨ (∀ σ : Solver cfg, ¬ SolvesTetrisValid cfg σ) :=
  solver_dichotomy hcols

/-- Three faces of solvability: safe membership, an init-cycle, and a closed atlas. -/
theorem construct_three_faces :
    (TetrisSolvableValid ↔ GameState.init ∈ safe GameConfig.standard) ∧
    (TetrisSolvableValid ↔ ∃ C : AdversarialClosedCycleWF GameConfig.standard,
        GameState.init ∈ C.toAdversarialClosedCycle.states) ∧
    (TetrisSolvableValid → ∃ (A : Atlas GameConfig.standard) (S : Finset GameState),
        A.IsClosedOn GameConfig.standard S ∧ GameState.init ∈ S) :=
  solver_three_faces_of_solvability

/-- Solvability is equivalent to the existence of a well-founded init-containing closed cycle. -/
theorem construct_solvable_iff_init_cycle :
    TetrisSolvableValid ↔
      ∃ C : AdversarialClosedCycleWF GameConfig.standard,
        GameState.init ∈ C.toAdversarialClosedCycle.states :=
  solver_solvable_iff_init_cycle

/-- Master equivalences: solvability ⟺ init safe ⟺ an init-containing closed cycle. -/
theorem construct_master_equivalences :
    (TetrisSolvableValid ↔ GameState.init ∈ safe GameConfig.standard) ∧
    (TetrisSolvableValid ↔
      ∃ C : AdversarialClosedCycleWF GameConfig.standard,
        GameState.init ∈ C.toAdversarialClosedCycle.states) :=
  solver_master_equivalences

/-- Dynamical substrate: any placement clears at most four lines from a non-full board. -/
theorem construct_clears_at_most_four {b : Board} (pl : Placement)
    (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    Board.linesCleared cfg (pl.place b) ≤ 4 :=
  solver_clears_at_most_four pl hnf

/-- Loss dichotomy: a board is lost or every column is within the field. -/
theorem construct_loss_dichotomy (b : Board) :
    Board.isLost cfg b ∨ ∀ j, Board.colHeight b j ≤ cfg.rows :=
  solver_loss_dichotomy b

/-- A board is non-losing exactly when all its cells sit below the top row. -/
theorem construct_not_lost_iff_cells_below (b : Board) :
    ¬ Board.isLost cfg b ↔ ∀ p ∈ b, p.2 < cfg.rows :=
  solver_not_lost_iff_cells_below_rows b

/-- Placement never lowers max height: the climb is one-directional. -/
theorem construct_placement_raises_maxHeight (b : Board) (pl : Placement) :
    Board.maxHeight cfg b ≤ Board.maxHeight cfg (pl.place b) :=
  solver_placement_raises_maxHeight b pl

/-- Placement is skyline-monotone: it preserves the dominance order on boards. -/
theorem construct_placement_skyline_monotone {b β : Board} (pl : Placement)
    (h : WqoCarrier.domLE b β) :
    WqoCarrier.domLE (pl.place b) (pl.place β) :=
  solver_placement_skyline_monotone pl h

/-- Clearing lowers the skyline: the cleared board is dominated by the original. -/
theorem construct_clearing_lowers_skyline (b : Board) :
    WqoCarrier.domLE (Board.clearLines cfg b) b :=
  solver_clearing_lowers_skyline b

/-- Surface evolution under placement is hole-independent: equal skylines stay equal. -/
theorem construct_surface_evolution_hole_independent {b β : Board} (pl : Placement)
    (h : ∀ j, b.colHeight j = β.colHeight j) (j : ℕ) :
    (pl.place b).colHeight j = (pl.place β).colHeight j :=
  solver_surface_evolution_hole_independent pl h j

/-- Energy split: hole-debt plus cell count equals surface area on a well-formed board. -/
theorem construct_energy_split {b : Board} (hwf : Board.WF cfg b) :
    HoleDebt.debt cfg b + b.count = HoleDebt.surfaceArea cfg b :=
  solver_energy_split hwf

/-- Energy brackets max height: maxHeight ≤ surfaceArea ≤ cols·maxHeight. -/
theorem construct_energy_brackets_maxHeight (b : Board) :
    Board.maxHeight cfg b ≤ HoleDebt.surfaceArea cfg b ∧
    HoleDebt.surfaceArea cfg b ≤ cfg.cols * Board.maxHeight cfg b :=
  solver_energy_brackets_maxHeight b

/-- The constructed function's placement never removes holes. -/
theorem buildSolver_place_never_removes_holes {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) (b : Board) :
    (HoleyCarrier.holes cfg b).card
      ≤ (HoleyCarrier.holes cfg ((buildSolver hfix g p).place b)).card :=
  solver_place_never_removes_holes (σ := buildSolver hfix) g p b

/-- The constructed function's skyline effect factors through the surface alone. -/
theorem buildSolver_skyline_factors_through_surface {S : Finset GameState}
    (hfix : F_finite cfg S = S) {g₁ g₂ : GameState} {p : Piece}
    (hsurf : ∀ j, g₁.board.colHeight j = g₂.board.colHeight j)
    (hpl : buildSolver hfix g₁ p = buildSolver hfix g₂ p) (j : ℕ) :
    ((buildSolver hfix g₁ p).place g₁.board).colHeight j
      = ((buildSolver hfix g₂ p).place g₂.board).colHeight j :=
  solver_skyline_effect_factors_through_surface (σ := buildSolver hfix) hsurf hpl j

/-- Bag discipline: a piece just drawn from a non-emptied bag cannot be drawn again. -/
theorem construct_no_repeat_within_bag (bag : Bag) (p : Piece) (hp : p ∈ bag)
    (hne : bag.draw p ≠ Bag.full) :
    ¬ (bag.draw p).canDraw p :=
  solver_no_repeat_within_bag bag p hp hne

/-- Dynamical portrait of the constructed function: orbit is iterated step; bag/board law. -/
theorem buildSolver_dynamical_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (p : Piece) :
    (∀ n, adversarialTrace cfg (buildSolver hfix) (fun _ => p) GameState.init n
        = (solverStep cfg (buildSolver hfix) p)^[n] GameState.init) ∧
    (∀ g, (solverStep cfg (buildSolver hfix) p g).bag = g.bag.draw p) ∧
    (∀ g, p ∈ g.bag →
      (solverStep cfg (buildSolver hfix) p g).board
        = (buildSolver hfix g p).applyStep cfg g.board) :=
  solver_dynamical_portrait (buildSolver_validSolver hcols hfix) p

/-- Move-effect portrait: count +≤4, height +≤4, raw place +4, and WF preserved. -/
theorem buildSolver_move_effect_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    (((buildSolver hfix g p).applyStep cfg b).count ≤ b.count + 4) ∧
    (Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b) ≤ Board.maxHeight cfg b + 4) ∧
    (((buildSolver hfix g p).place b).count = b.count + 4) ∧
    Board.WF cfg ((buildSolver hfix g p).applyStep cfg b) :=
  solver_move_effect_portrait (buildSolver_validSolver hcols hfix) hp hWF

/-- Move-geometry portrait: the constructed function's drop is a 4-cell 4×4-confined footprint. -/
theorem buildSolver_move_geometry_portrait {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) (b : Board) :
    (b ⊆ (buildSolver hfix g p).place b) ∧
    (((buildSolver hfix g p).dropped b).card = 4) ∧
    (∀ c ∈ (buildSolver hfix g p).dropped b,
      (buildSolver hfix g p).col ≤ c.1 ∧ c.1 < (buildSolver hfix g p).col + 4) ∧
    (∀ c ∈ (buildSolver hfix g p).dropped b,
      (buildSolver hfix g p).dropOffset b ≤ c.2 ∧ c.2 < (buildSolver hfix g p).dropOffset b + 4) :=
  solver_move_geometry_portrait (σ := buildSolver hfix) g p b

/-- Output-finiteness portrait: per-piece menu ≤ cols·4, total menu ≤ |Piece|·cols·4. -/
theorem construct_output_finiteness_portrait (p : Piece) :
    (Placement.allValidFor cfg p).card ≤ cfg.cols * 4 ∧
    ((Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg)).card
      ≤ Fintype.card Piece * (cfg.cols * 4) :=
  solver_output_finiteness_portrait p

/-- Footprint portrait: reachable states stay in-field and realized placements are finite. -/
theorem buildSolver_footprint_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    (∀ g, solverReachable (buildSolver hfix) g → g ∈ inFieldStates cfg) ∧
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ buildSolver hfix g p = pl}.Finite :=
  solver_footprint_portrait (buildSolver_solvesTetrisValid hcols hfix hinit)

/-- Two-number portrait: each output is pinned by its (rot, col) pair in a finite grid. -/
theorem buildSolver_two_number_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p = ⟨p, (buildSolver hfix g p).rot, (buildSolver hfix g p).col⟩) ∧
    (∀ g₂, p ∈ g₂.bag →
      (buildSolver hfix g p).rot = (buildSolver hfix g₂ p).rot →
        (buildSolver hfix g p).col = (buildSolver hfix g₂ p).col →
          buildSolver hfix g p = buildSolver hfix g₂ p) ∧
    (((buildSolver hfix g p).rot, (buildSolver hfix g p).col)
      ∈ (Finset.univ : Finset Rotation) ×ˢ Finset.range cfg.cols) :=
  solver_two_number_portrait (buildSolver_validSolver hcols hfix) hp

/-- Collapse portrait: a piece-slice lands in the menu, ≤ cols·4, and collides past it. -/
theorem buildSolver_collapse_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (p : Piece) (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => buildSolver hfix g p) ⊆ Placement.allValidFor cfg p) ∧
    ((T.image (fun g => buildSolver hfix g p)).card ≤ cfg.cols * 4) ∧
    ((Placement.allValidFor cfg p).card < T.card →
      ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧ buildSolver hfix g₁ p = buildSolver hfix g₂ p) :=
  solver_collapse_portrait (buildSolver_validSolver hcols hfix) p T hT

/-- Under-determination portrait: closedness on safe + init∈safe suffices for ANY solver. -/
theorem construct_under_determination_portrait :
    (∀ (σ' : Solver cfg),
      (∀ g ∈ safe cfg, ∀ p ∈ g.bag, adversarialStep cfg g p (σ' g p) ∈ safe cfg) →
        GameState.init ∈ safe cfg → SolvesTetris cfg σ') ∧
    (∀ g, g ∈ safe cfg → ∀ p ∈ g.bag,
      ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg) :=
  solver_under_determination_portrait

/-- Input portrait: along legal play the constructed function is fed nonempty in-bag pieces. -/
theorem buildSolver_input_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (s : ℕ → Piece) (hl : LegalSequence s) :
    (∀ n, (adversarialTrace cfg (buildSolver hfix) s GameState.init n).bag.Nonempty) ∧
    (∀ n, s n ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).bag) ∧
    (∀ n, buildSolver hfix (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)
        ∈ Placement.allValidFor cfg (s n)) :=
  solver_input_portrait (buildSolver_validSolver hcols hfix) s hl

/-- The target region of the construction is exactly the safe-operator's fixed point. -/
theorem construct_region_fixed_point : safeOp cfg (safe cfg) = safe cfg :=
  solver_region_fixed_point

/-- The region is the greatest fixed point: any safe-closed set is contained in it. -/
theorem construct_region_greatest_fixed_point (T : Set GameState) (hT : safeOp cfg T = T) :
    T ⊆ safe cfg :=
  solver_region_greatest_fixed_point T hT

/-- The region is computable: finite descending iteration stabilizes within `|inFieldStates|`. -/
theorem construct_region_computable :
    ∃ N, N ≤ (inFieldStates cfg).card ∧
      safeIterFinite cfg (inFieldStates cfg) (N + 1) = safeIterFinite cfg (inFieldStates cfg) N :=
  solver_region_computable

/-- Local certificate: any self-sustaining region of states is contained in the safe region. -/
theorem construct_region_local_certificate (S : Set GameState)
    (hS : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p, p ∈ g.bag → ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    S ⊆ safe cfg :=
  solver_region_local_certificate S hS

/-- Computed-region soundness: at a fixed point the computed finite set lies inside safe. -/
theorem construct_computed_region_sound (N : ℕ)
    (hfix : safeIterFinite cfg (inFieldStates cfg) (N + 1)
      = safeIterFinite cfg (inFieldStates cfg) N) :
    (↑(safeIterFinite cfg (inFieldStates cfg) N) : Set GameState) ⊆ safe cfg :=
  solver_computed_region_sound N hfix

/-- Computed-region completeness: from a universe covering safe, the iterate equals safe. -/
theorem construct_computed_region_complete {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀)
    (N : ℕ) (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    g ∈ safe cfg ↔ g ∈ (↑(safeIterFinite cfg S₀ N) : Set GameState) :=
  solver_computed_region_complete hS₀ N hfix g

/-- Existence is decidable: a single init-membership test in a computed iterate settles it. -/
theorem construct_existence_decidable {S₀ : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hS₀ : safe cfg ⊆ ↑S₀) :
    ∃ N : ℕ, (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔
      GameState.init ∈ safeIterFinite cfg S₀ N :=
  solver_existence_decidable hcols hS₀

/-- A local survival certificate containing init already yields a full solver. -/
theorem construct_exists_of_local_certificate (hcols : 4 ≤ cfg.cols) (S : Set GameState)
    (hinit : GameState.init ∈ S)
    (hS : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p, p ∈ g.bag → ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ S) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  solver_exists_of_local_certificate hcols S hinit hS

/-- Death propagation is a descending chain: later iterates are subsets of earlier ones. -/
theorem construct_death_propagation_monotone {n m : ℕ} (hnm : n ≤ m) :
    safeIterFinite cfg (inFieldStates cfg) m ⊆ safeIterFinite cfg (inFieldStates cfg) n :=
  solver_death_propagation_monotone hnm

/-- Existence of a solver is equivalent to an init-containing well-founded closed cycle. -/
theorem construct_exists_iff_init_cycle (hcols : 4 ≤ cfg.cols) :
    (∃ σ : Solver cfg, SolvesTetrisValid cfg σ) ↔
      ∃ C : AdversarialClosedCycleWF cfg,
        GameState.init ∈ C.toAdversarialClosedCycle.states :=
  solver_exists_iff_init_cycle hcols

/-- On the standard config, solver existence coincides with the headline solvability predicate. -/
theorem construct_exists_iff_tetrisSolvableValid :
    (∃ σ : Solver GameConfig.standard, SolvesTetrisValid GameConfig.standard σ)
      ↔ TetrisSolvableValid :=
  solver_exists_iff_tetrisSolvableValid

/-- At the empty board the construction handles all seven pieces with a safe successor. -/
theorem construct_handles_all_seven_at_init (h : GameState.init ∈ safe cfg) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg GameState.init p pl ∈ safe cfg :=
  solver_handles_all_seven_at_init h p

/-- A clearing move of the constructed function adds strictly fewer than four cells. -/
theorem buildSolver_move_count_clear {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) (hpos : 0 < cfg.cols)
    (hclear : 0 < Board.linesCleared cfg ((buildSolver hfix g p).place b)) :
    ((buildSolver hfix g p).applyStep cfg b).count < b.count + 4 :=
  solver_move_count_clear (buildSolver_validSolver hcols hfix) hp hWF hpos hclear

/-- A non-clearing move of the constructed function adds exactly four cells. -/
theorem buildSolver_move_count_no_clear {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b)
    (h0 : Board.linesCleared cfg ((buildSolver hfix g p).place b) = 0) :
    ((buildSolver hfix g p).applyStep cfg b).count = b.count + 4 :=
  solver_move_count_no_clear (buildSolver_validSolver hcols hfix) hp hWF h0

/-- A constructed-function move clears at most four lines from a non-full board. -/
theorem buildSolver_move_clears_le_four {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) {b : Board} (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    Board.linesCleared cfg ((buildSolver hfix g p).place b) ≤ 4 :=
  solver_move_clears_le_four (σ := buildSolver hfix) g p hnf

/-- A constructed-function move keeps the cell count within capacity when there is room. -/
theorem buildSolver_move_count_le_capacity {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) (hbelow : b.count + 4 ≤ cfg.cols * cfg.rows) :
    ((buildSolver hfix g p).applyStep cfg b).count ≤ cfg.cols * cfg.rows :=
  solver_move_count_le_capacity (buildSolver_validSolver hcols hfix) hp hWF hbelow

/-- The constructed function's raw placement is skyline-monotone in the dominance order. -/
theorem buildSolver_move_skyline_monotone {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) {b β : Board} (h : WqoCarrier.domLE b β) :
    WqoCarrier.domLE ((buildSolver hfix g p).place b) ((buildSolver hfix g p).place β) :=
  solver_move_skyline_monotone (σ := buildSolver hfix) g p h

/-- The constructed function's placement skyline is hole-independent. -/
theorem buildSolver_move_skyline_hole_independent {S : Finset GameState}
    (hfix : F_finite cfg S = S) (g : GameState) (p : Piece) {b β : Board}
    (h : ∀ j, b.colHeight j = β.colHeight j) (j : ℕ) :
    ((buildSolver hfix g p).place b).colHeight j = ((buildSolver hfix g p).place β).colHeight j :=
  solver_move_skyline_hole_independent (σ := buildSolver hfix) g p h j

/-- A constructed-function placement only adds cells: the board is a subset of its image. -/
theorem buildSolver_move_superset {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) :
    b ⊆ (buildSolver hfix g p).place b :=
  solver_move_superset (σ := buildSolver hfix) b g p

/-- The constructed function draws the queried piece from the bag at each step. -/
theorem buildSolver_next_bag {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    (solverStep cfg (buildSolver hfix) p g).bag = g.bag.draw p :=
  solver_next_bag (σ := buildSolver hfix) g p

/-- The constructed function's next board is its placement applied to the current board. -/
theorem buildSolver_next_board {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (solverStep cfg (buildSolver hfix) p g).board
      = (buildSolver hfix g p).applyStep cfg g.board :=
  solver_next_board (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output column is in range. -/
theorem buildSolver_col_lt_cols {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).col < cfg.cols :=
  solver_col_lt_cols (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output rotation is one of four. -/
theorem buildSolver_rot_lt_four {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    ((buildSolver hfix g p).rot : ℕ) < 4 :=
  solver_rot_lt_four (σ := buildSolver hfix) g p

/-- The cells the constructed function drops are disjoint from the existing board. -/
theorem buildSolver_dropped_disjoint {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) :
    Disjoint b ((buildSolver hfix g p).dropped b) :=
  solver_dropped_disjoint (σ := buildSolver hfix) b g p

/-- On the empty board the constructed function drops to the floor (offset zero). -/
theorem buildSolver_empty_drop_zero {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    (buildSolver hfix g p).dropOffset Board.empty = 0 :=
  solver_empty_drop_zero (σ := buildSolver hfix) g p

/-- The constructed function's move preserves board well-formedness. -/
theorem buildSolver_applyStep_wf {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    Board.WF cfg ((buildSolver hfix g p).applyStep cfg b) :=
  solver_applyStep_wf (buildSolver_validSolver hcols hfix) hp hWF

/-- Safe states have a safe response to each bag piece — the construction's per-state guarantee. -/
theorem construct_safe_response_each_piece {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg :=
  solver_safe_response_each_piece hg hp

/-- Loss is monotone: adding cells to a lost board keeps it lost. -/
theorem construct_loss_monotone {b b' : Board} (h : b ⊆ b') (hlost : Board.isLost cfg b) :
    Board.isLost cfg b' :=
  solver_loss_monotone h hlost

/-- Bag renewal: each draw either drains the bag by one or refills a full bag. -/
theorem construct_bag_renewal (bag : Bag) (p : Piece) (hp : p ∈ bag) :
    (bag.draw p).card = bag.card - 1 ∨ bag.draw p = Bag.full :=
  solver_bag_renewal bag p hp

/-- The bag evolves identically under any two solvers: it is policy-independent. -/
theorem construct_bag_evolution_independent (σ₁ σ₂ : Solver cfg) (g : GameState) (p : Piece) :
    (solverStep cfg σ₁ p g).bag = (solverStep cfg σ₂ p g).bag :=
  solver_bag_evolution_independent σ₁ σ₂ g p

/-- Clearing never increases hole-debt on a well-formed board. -/
theorem construct_clearing_reduces_debt {b : Board} (hwf : Board.WF cfg b) :
    HoleDebt.debt cfg (Board.clearLines cfg b) ≤ HoleDebt.debt cfg b :=
  solver_clearing_reduces_debt hwf

/-- Clearing is idempotent: re-clearing a cleared board changes nothing. -/
theorem construct_clearing_idempotent (b : Board) (hcol : 0 < cfg.cols) :
    Board.clearLines cfg (Board.clearLines cfg b) = Board.clearLines cfg b :=
  solver_clearing_idempotent b hcol

/-- The constructed function's raw placement raises max height by at most four. -/
theorem buildSolver_place_maxHeight_le {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) (b : Board) :
    Board.maxHeight cfg ((buildSolver hfix g p).place b) ≤ Board.maxHeight cfg b + 4 :=
  solver_place_maxHeight_le (buildSolver_validSolver hcols hfix) hp b

/-- Any placement only raises per-column heights. -/
theorem construct_placement_raises_columns (b : Board) (pl : Placement) (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight (pl.place b) j :=
  solver_placement_raises_columns b pl j

/-- The constructed function's dropped cells span at most four columns. -/
theorem buildSolver_move_cols_bounded {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) {c : Coord}
    (hc : c ∈ (buildSolver hfix g p).dropped b) :
    (buildSolver hfix g p).col ≤ c.1 ∧ c.1 < (buildSolver hfix g p).col + 4 :=
  solver_move_cols_bounded (σ := buildSolver hfix) b g p hc

/-- The constructed function's dropped cells span at most four rows. -/
theorem buildSolver_move_rows_bounded {S : Finset GameState} (hfix : F_finite cfg S = S)
    (b : Board) (g : GameState) (p : Piece) {c : Coord}
    (hc : c ∈ (buildSolver hfix g p).dropped b) :
    (buildSolver hfix g p).dropOffset b ≤ c.2 ∧ c.2 < (buildSolver hfix g p).dropOffset b + 4 :=
  solver_move_rows_bounded (σ := buildSolver hfix) b g p hc

/-- The action space is never empty: each piece has at least one valid placement. -/
theorem construct_action_space_nonempty (hcols : 4 ≤ cfg.cols) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  solver_action_space_nonempty hcols p

/-- Along legal play the constructed function always faces a nonempty bag. -/
theorem buildSolver_always_has_a_piece {S : Finset GameState} (hfix : F_finite cfg S = S)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg (buildSolver hfix) s GameState.init n).bag.Nonempty :=
  solver_always_has_a_piece (buildSolver hfix) hl n

/-- Every cell the constructed function ever lays sits within the playfield rectangle. -/
theorem buildSolver_board_in_field {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) {c : Coord}
    (hc : c ∈ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board) :
    c.1 < cfg.cols ∧ c.2 < cfg.rows :=
  solver_board_in_field (buildSolver_solvesTetrisValid hcols hfix hinit) hl n hc

/-- Per-bag resource ledger: each 7-bag order contains exactly one I and two S/Z. -/
theorem construct_per_bag_resource {l : List Piece} (h : BagBurst.IsBagOrder l) :
    l.countP BagBurst.isI = 1 ∧ l.countP BagBurst.isSZ = 2 :=
  solver_per_bag_resource h

/-- The constructed cycle realizes at most `2^(cols·rows)` distinct boards. -/
theorem construct_distinct_boards_le (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      (C.toAdversarialClosedCycle.states.image GameState.board).card
        ≤ 2 ^ (cfg.cols * cfg.rows) :=
  solver_distinct_boards_le hcols hex

/-- The constructed cycle realizes at most 128 distinct bag states. -/
theorem construct_distinct_bags_le (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      (C.toAdversarialClosedCycle.states.image GameState.bag).card ≤ 128 :=
  solver_distinct_bags_le hcols hex

/-- The standard universe the construction searches has exactly `2^207` states. -/
theorem construct_universe_size_standard :
    (inFieldStates GameConfig.standard).card = 2 ^ 207 :=
  solver_universe_size_standard

/-- The constructed function's out-degree at a state is at most the bag size. -/
theorem buildSolver_outdegree_le {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) :
    (g.bag.image (fun p => solverStep cfg (buildSolver hfix) p g)).card ≤ g.bag.card :=
  solver_outdegree_le (σ := buildSolver hfix) g

/-- The opening can never lose: any valid first placement keeps the empty board alive. -/
theorem construct_opening_cannot_lose (hrows : 4 ≤ cfg.rows) (pl : Placement) (hv : pl.Valid cfg) :
    ¬ Board.isLost cfg (Placement.applyStep cfg GameState.init.board pl) :=
  solver_opening_cannot_lose hrows pl hv

/-- A hole is a double obstruction: an unfilled cell with an occupied cell strictly above it. -/
theorem construct_hole_obstruction {b : Board} {p : Coord}
    (hp : p ∈ HoleyCarrier.holes cfg b) :
    ¬ Board.isFull cfg b p.2 ∧ ∃ r, p.2 < r ∧ (p.1, r) ∈ b :=
  solver_hole_obstruction hp

/-- Benign pieces leave no holes: O and I placed flat on empty create a clean surface. -/
theorem construct_benign_pieces_no_holes :
    HoleyCarrier.holes GameConfig.standard (Placement.place ∅ ⟨Piece.O, 0, 0⟩) = ∅ ∧
    HoleyCarrier.holes GameConfig.standard (Placement.place ∅ ⟨Piece.I, 0, 0⟩) = ∅ :=
  solver_benign_pieces_no_holes

/-- Both roughness pieces inject holes: S buries a cell and Z creates a nonempty hole set. -/
theorem construct_both_roughness_inject_holes :
    ((2 : ℕ), (0 : ℕ)) ∈ HoleyCarrier.holes GameConfig.standard
        (Placement.place ∅ ⟨Piece.S, 0, 0⟩)
    ∧ 0 < (HoleyCarrier.holes GameConfig.standard
        (Placement.place ∅ ⟨Piece.Z, 0, 0⟩)).card :=
  solver_both_roughness_inject_holes

/-- On the empty board the constructed function's dropped cells sit in the bottom four rows. -/
theorem buildSolver_empty_cells_low {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) {c : Coord}
    (hc : c ∈ (buildSolver hfix g p).dropped Board.empty) : c.2 < 4 :=
  solver_empty_cells_low (σ := buildSolver hfix) g p hc

/-- Drain budget suffices: the I-piece supply exceeds the clearing requirement over any bags. -/
theorem construct_drain_budget_suffices {bags : List (List Piece)}
    (h : ∀ b ∈ bags, BagBurst.IsBagOrder b) :
    14 * bags.length ≤ 20 * bags.flatten.countP BagBurst.isI :=
  solver_drain_budget_suffices h

/-- Roughness scales with drains: each bag order has twice as many S/Z as I pieces. -/
theorem construct_roughness_two_per_drain {l : List Piece} (h : BagBurst.IsBagOrder l) :
    l.countP BagBurst.isSZ = 2 * l.countP BagBurst.isI :=
  solver_roughness_two_per_drain h

/-- Any placement never removes holes — the hole count is monotone under placing. -/
theorem construct_placement_never_removes_holes (b : Board) (pl : Placement) :
    (HoleyCarrier.holes cfg b).card ≤ (HoleyCarrier.holes cfg (pl.place b)).card :=
  solver_placement_never_removes_holes b pl

/-- The constructed function's output column lands in the column range finset. -/
theorem buildSolver_col_mem_range {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).col ∈ Finset.range cfg.cols :=
  solver_col_mem_range (buildSolver_validSolver hcols hfix) hp

/-- The constructed function's output rotation lands in the rotation universe. -/
theorem buildSolver_rot_mem_univ {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    (buildSolver hfix g p).rot ∈ (Finset.univ : Finset Rotation) :=
  solver_rot_mem_univ (σ := buildSolver hfix) g p

/-- Compressibility grand portrait: coded outputs, finite image, board+bag determined, total. -/
theorem buildSolver_compressibility_grand_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) :
    (∀ g p, p ∈ g.bag →
        4 * (buildSolver hfix g p).col + ((buildSolver hfix g p).rot : ℕ) < 4 * cfg.cols) ∧
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ buildSolver hfix g p = pl}.Finite ∧
    (∀ g₁ g₂ p, g₁.board = g₂.board → g₁.bag = g₂.bag →
        buildSolver hfix g₁ p = buildSolver hfix g₂ p) ∧
    (∀ g p, (buildSolver hfix).toAtlas g p = some (buildSolver hfix g p)) :=
  solver_compressibility_grand_portrait (buildSolver_validSolver hcols hfix)

/-- Dynamical grand portrait: step law, orbit-as-iterate, bag/board laws, nontrivial motion. -/
theorem buildSolver_dynamical_grand_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (p : Piece) :
    (∀ g, solverStep cfg (buildSolver hfix) p g
        = adversarialStep cfg g p (buildSolver hfix g p)) ∧
    (∀ n, adversarialTrace cfg (buildSolver hfix) (fun _ => p) GameState.init n
        = (solverStep cfg (buildSolver hfix) p)^[n] GameState.init) ∧
    (∀ g, (solverStep cfg (buildSolver hfix) p g).bag = g.bag.draw p) ∧
    (∀ g, p ∈ g.bag → (solverStep cfg (buildSolver hfix) p g).board
        = (buildSolver hfix g p).applyStep cfg g.board) ∧
    (∀ g, g.bag.draw p ≠ g.bag → solverStep cfg (buildSolver hfix) p g ≠ g) :=
  solver_dynamical_grand_portrait (buildSolver_validSolver hcols hfix) p

/-- The constructed function reads only board and bag (eta over the state record). -/
theorem buildSolver_eta {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) :
    buildSolver hfix g p = buildSolver hfix ⟨g.board, g.bag⟩ p :=
  solver_eta (σ := buildSolver hfix) g p

/-- Representability: the constructed function has finite image and a total atlas. -/
theorem buildSolver_representable {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) :
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ buildSolver hfix g p = pl}.Finite ∧
    (∀ g p, (buildSolver hfix).toAtlas g p = some (buildSolver hfix g p)) :=
  solver_representable (buildSolver_validSolver hcols hfix)

/-- Closed atlases compose: their union is closed on the union of regions. -/
theorem construct_atlas_composes {A B : Atlas cfg} {S₁ S₂ : Finset GameState}
    (hA : A.IsClosedOn cfg S₁) (hB : B.IsClosedOn cfg S₂) :
    (A.unionOn B S₁ S₂).IsClosedOn cfg (S₁ ∪ S₂) :=
  solver_atlas_composes hA hB

/-- Standard total response table over states×pieces has at most 280 distinct entries. -/
theorem construct_image_card_le_standard (σ : Solver GameConfig.standard)
    (hv : ValidSolver GameConfig.standard σ) (T : Finset (GameState × Piece))
    (hT : ∀ gp ∈ T, gp.2 ∈ gp.1.bag) :
    (T.image (fun gp => σ gp.1 gp.2)).card ≤ 280 :=
  solver_image_card_le_standard σ hv T hT

/-- Standard per-piece response slice has at most 40 distinct entries. -/
theorem construct_image_per_piece_standard_le (σ : Solver GameConfig.standard)
    (hv : ValidSolver GameConfig.standard σ) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => σ g p)).card ≤ 40 :=
  solver_image_per_piece_standard_le σ hv T hT

/-- The constructed function's piece-slice image is contained in that piece's menu. -/
theorem buildSolver_image_per_piece_subset {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {p : Piece} (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) :
    T.image (fun g => buildSolver hfix g p) ⊆ Placement.allValidFor cfg p :=
  solver_image_per_piece_subset (buildSolver_validSolver hcols hfix) T hT

/-- The construction's start state: full bag, zero cells, empty board. -/
theorem construct_initial_state :
    GameState.init.bag = Bag.full ∧
    GameState.init.board.count = 0 ∧
    GameState.init.board = (∅ : Board) :=
  solver_initial_state

/-- The full state×piece query table over the standard universe has size `2^207 · 7`. -/
theorem construct_table_domain_card :
    ((inFieldStates cfg) ×ˢ (Finset.univ : Finset Piece)).card
      = (inFieldStates cfg).card * Fintype.card Piece :=
  solver_table_domain_card

/-- Energy split holds after a constructed-function move: debt + count = surface area. -/
theorem buildSolver_move_energy_split {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    HoleDebt.debt cfg ((buildSolver hfix g p).applyStep cfg b)
        + ((buildSolver hfix g p).applyStep cfg b).count
      = HoleDebt.surfaceArea cfg ((buildSolver hfix g p).applyStep cfg b) :=
  solver_move_energy_split (buildSolver_validSolver hcols hfix) hp hWF

/-- Energy brackets max height after a constructed-function move. -/
theorem buildSolver_move_energy_brackets {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) (b : Board) :
    Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b)
        ≤ HoleDebt.surfaceArea cfg ((buildSolver hfix g p).applyStep cfg b) ∧
    HoleDebt.surfaceArea cfg ((buildSolver hfix g p).applyStep cfg b)
        ≤ cfg.cols * Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b) :=
  solver_move_energy_brackets (σ := buildSolver hfix) g p b

/-- A constructed-function placement only raises per-column heights. -/
theorem buildSolver_place_raises_columns {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) (b : Board) (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight ((buildSolver hfix g p).place b) j :=
  solver_place_raises_columns (σ := buildSolver hfix) g p b j

/-- At each state the constructed function injects pieces into placements. -/
theorem buildSolver_slice_injOn_bag {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (g : GameState) :
    Set.InjOn (fun p => buildSolver hfix g p) g.bag :=
  solver_slice_injOn_bag (buildSolver_validSolver hcols hfix) g

/-- For each piece the constructed function maps in-bag states into that piece's menu. -/
theorem buildSolver_slice_mapsTo {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (p : Piece) :
    Set.MapsTo (fun g => buildSolver hfix g p) {g | p ∈ g.bag}
      ↑(Placement.allValidFor cfg p) :=
  solver_slice_mapsTo (buildSolver_validSolver hcols hfix) p

/-- Solver tables are bounded: the closed cycle has card in `[28, |inFieldStates|]`. -/
theorem construct_table_size_bounded (hcols : 4 ≤ cfg.cols)
    (hex : ∃ σ : Solver cfg, SolvesTetrisValid cfg σ) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  solver_table_size_bounded hcols hex

/-- The atlas wrapper is faithful: equal atlases come from equal solvers. -/
theorem construct_toAtlas_inj {σ₁ σ₂ : Solver cfg} (h : σ₁.toAtlas = σ₂.toAtlas) : σ₁ = σ₂ :=
  solver_toAtlas_inj h

/-- A non-lost state is unsafe exactly when some bag piece kills every response (killer piece). -/
theorem construct_unsafe_iff_killer_piece {g : GameState} (hnl : ¬ g.lost cfg) :
    g ∉ safe cfg ↔ ∃ p ∈ g.bag, ∀ pl : Placement,
      pl.piece = p → pl.Valid cfg → adversarialStep cfg g p pl ∉ safe cfg :=
  solver_unsafe_iff_killer_piece hnl

/-- Hole-free energy is fully clearable: zero debt means surface area equals the cell count. -/
theorem construct_hole_free_energy_all_clearable {b : Board} (hwf : Board.WF cfg b)
    (h0 : HoleDebt.debt cfg b = 0) :
    HoleDebt.surfaceArea cfg b = b.count :=
  solver_hole_free_energy_all_clearable hwf h0

/-- Low-stack comfort zone: from a sufficiently low board any piece has a non-losing placement. -/
theorem construct_low_stack_comfort_zone (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {b : Board} (hWF : Board.WF cfg b) (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4)
    (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      ¬ Board.isLost cfg (Placement.applyStep cfg b pl) :=
  solver_low_stack_comfort_zone hcols hrows hWF hlow p

/-- The safe region is not captured by a dominated hole-basis: both monotonicities fail. -/
theorem construct_region_not_dominated_basis :
    (¬ ∀ (cfg : GameConfig) (b β : Board) (pl : Placement), HoleyCarrier.safeLE cfg b β →
        HoleyCarrier.holes cfg (pl.place b) ⊆ HoleyCarrier.holes cfg (pl.place β))
    ∧ (¬ ∀ (cfg : GameConfig) (b : Board),
        HoleyCarrier.holes cfg (Board.clearLines cfg b) ⊆ HoleyCarrier.holes cfg b) :=
  solver_region_not_dominated_basis

/-- The canonical safe solver's move always lies in the finite action menu. -/
theorem construct_safeSolver_move_in_menu (hcols : 4 ≤ cfg.cols) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    safeSolver cfg g p ∈ Placement.allValidFor cfg p :=
  solver_move_in_finite_action_set hcols hp

/-- Play depends only on the solver's behaviour on reachable states. -/
theorem construct_trace_eq_of_agree_on_reachable (s : ℕ → Piece) {σ₁ σ₂ : Solver cfg}
    (hl : LegalSequence s)
    (hagree : ∀ g, solverReachable σ₁ g → ∀ p ∈ g.bag, σ₁ g p = σ₂ g p) (n : ℕ) :
    adversarialTrace cfg σ₁ s GameState.init n
      = adversarialTrace cfg σ₂ s GameState.init n :=
  solver_trace_eq_of_agree_on_reachable s hl hagree n

/-- Play is determined by the solver's choices on the states it actually visits. -/
theorem construct_trace_determined_by_visited (s : ℕ → Piece) {σ₁ σ₂ : Solver cfg} :
    ∀ n, (∀ k < n, σ₁ (adversarialTrace cfg σ₁ s GameState.init k) (s k)
                 = σ₂ (adversarialTrace cfg σ₁ s GameState.init k) (s k)) →
      adversarialTrace cfg σ₁ s GameState.init n
        = adversarialTrace cfg σ₂ s GameState.init n :=
  solver_trace_determined_by_visited s

/-- Locality portrait: equal choices give equal steps, and reachable-agreement gives equal play. -/
theorem construct_locality_portrait {σ₁ σ₂ : Solver cfg} (s : ℕ → Piece) (hl : LegalSequence s) :
    (∀ g p, σ₁ g p = σ₂ g p → solverStep cfg σ₁ p g = solverStep cfg σ₂ p g) ∧
    ((∀ g, solverReachable σ₁ g → ∀ p ∈ g.bag, σ₁ g p = σ₂ g p) →
      ∀ n, adversarialTrace cfg σ₁ s GameState.init n
        = adversarialTrace cfg σ₂ s GameState.init n) :=
  solver_locality_portrait s hl

/-! ### The canonical construction from the computed region

The descending iteration `safeIterFinite` over the whole in-field universe stabilizes at
index `|inFieldStates|`. We name that stabilized set the *converged region* and develop the
canonical construction over it: it is an `F_finite` fixed point, it equals the abstract safe
region, and the explicit `buildSolver` over it is a global solver whenever init survives. -/

/-- The canonical computed region: the in-field iterate at index `|inFieldStates|`. -/
noncomputable def convergedSet (cfg : GameConfig) : Finset GameState :=
  safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card

/-- The converged region is an `F_finite` fixed point — the construction has halted. -/
theorem convergedSet_fixed : F_finite cfg (convergedSet cfg) = convergedSet cfg :=
  safeIterFinite_inFieldStates_F_finite_fixed_at_card cfg

/-- Soundness of the construction: every converged state is genuinely safe. -/
theorem convergedSet_subset_safe : ↑(convergedSet cfg) ⊆ safe cfg :=
  fun _ hg => mem_safe_of_mem_safeIterFinite_at_S₀_card (Finset.mem_coe.mp hg)

/-- Every converged state is non-losing. -/
theorem convergedSet_not_lost {g : GameState} (hg : g ∈ convergedSet cfg) : ¬ g.lost cfg :=
  safe_not_lost (convergedSet_subset_safe (Finset.mem_coe.mpr hg))

/-- Positive test: if init survives into the converged region, Tetris is solvable. -/
theorem init_mem_convergedSet_solvable (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) : TetrisSolvableValidFor cfg :=
  tetrisSolvableValidFor_of_init_mem_safeIterFinite_at_S₀_card hcols hinit

/-- The canonical solver, built explicitly over the converged region. -/
noncomputable def convergedSolver : Solver cfg :=
  buildSolver convergedSet_fixed

/-- The canonical converged solver is a valid solver. -/
theorem convergedSolver_validSolver (hcols : 4 ≤ cfg.cols) :
    ValidSolver cfg convergedSolver :=
  buildSolver_validSolver hcols convergedSet_fixed

/-- The converged solver solves Tetris whenever init survives into the converged region. -/
theorem convergedSolver_solves (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    SolvesTetrisValid cfg convergedSolver :=
  buildSolver_solvesTetrisValid hcols convergedSet_fixed hinit

/-- Under the converged solver no legal play ever tops out (given init survives). -/
theorem convergedSolver_survives (hinit : GameState.init ∈ convergedSet cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg :=
  buildSolver_survives convergedSet_fixed hinit hl n

/-- The converged solver never leaves the converged region: it is region-confined. -/
theorem convergedSolver_confined (hinit : GameState.init ∈ convergedSet cfg) {g : GameState}
    (hr : solverReachable (convergedSolver (cfg := cfg)) g) : g ∈ convergedSet cfg :=
  buildSolver_reachable_mem convergedSet_fixed hinit hr

/-- The converged solver's atlas is closed on the converged region — the M4 proof artifact. -/
theorem convergedSolver_atlas_closed (cfg : GameConfig) :
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) :=
  buildSolver_atlas_closed convergedSet_fixed

/-- The converged region is no larger than the in-field universe it was carved from. -/
theorem convergedSet_card_le (cfg : GameConfig) :
    (convergedSet cfg).card ≤ (inFieldStates cfg).card :=
  safeIterFinite_inFieldStates_card_le cfg (inFieldStates cfg).card

/-- The converged region is contained in the in-field universe. -/
theorem convergedSet_subset_inFieldStates (cfg : GameConfig) :
    convergedSet cfg ⊆ inFieldStates cfg :=
  safeIterFinite_inFieldStates_subset cfg (inFieldStates cfg).card

/-- On the standard board the converged region has at most `2^207` states. -/
theorem convergedSet_standard_card_le :
    (convergedSet GameConfig.standard).card ≤ 2 ^ 207 := by
  have h := convergedSet_card_le GameConfig.standard
  rwa [solver_universe_size_standard] at h

/-- The converged solver returns a valid placement for any bag piece. -/
theorem convergedSolver_output_valid (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) : (convergedSolver (cfg := cfg) g p).Valid cfg :=
  buildSolver_output_valid hcols convergedSet_fixed hp

/-- The converged solver answers the queried piece. -/
theorem convergedSolver_announces_piece (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) : (convergedSolver (cfg := cfg) g p).piece = p :=
  buildSolver_announces_piece hcols convergedSet_fixed hp

/-- Every state reachable by the converged solver is safe. -/
theorem convergedSolver_no_dead_ends (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {g : GameState}
    (hr : solverReachable (convergedSolver (cfg := cfg)) g) : g ∈ safe cfg :=
  buildSolver_no_dead_ends hcols convergedSet_fixed hinit hr

/-- The converged solver's play eventually revisits a state. -/
theorem convergedSolver_eventually_repeats (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≠ j ∧
      adversarialTrace cfg convergedSolver s GameState.init i
        = adversarialTrace cfg convergedSolver s GameState.init j :=
  buildSolver_eventually_repeats hcols convergedSet_fixed hinit hl

/-- No legal piece sequence ever kills the converged solver. -/
theorem convergedSolver_no_killing_sequence (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    ¬ ∃ (s : ℕ → Piece) (n : ℕ),
        LegalSequence s ∧
        (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg :=
  buildSolver_no_killing_sequence hcols convergedSet_fixed hinit

/-- Every state visited by the converged solver is safe. -/
theorem convergedSolver_trace_mem_safe (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init n ∈ safe cfg :=
  buildSolver_trace_mem_safe hcols convergedSet_fixed hinit hl n

/-- Every state visited by the converged solver lies in the in-field universe. -/
theorem convergedSolver_trace_in_field (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init n ∈ inFieldStates cfg :=
  buildSolver_trace_in_field hcols convergedSet_fixed hinit hl n

/-- The converged solver clears a line within any capacity-sized window. -/
theorem convergedSolver_clears_within (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s)
    {M : ℕ} (hM : cfg.cols * cfg.rows < 4 * M) :
    ∃ k < M, (adversarialTrace cfg convergedSolver s GameState.init (k + 1)).board.count
           ≠ (adversarialTrace cfg convergedSolver s GameState.init k).board.count + 4 :=
  buildSolver_clears_within hcols convergedSet_fixed hinit hl hM

/-- Every column stays within the field along converged-solver play. -/
theorem convergedSolver_columns_le (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) {j : ℕ} (hj : j < cfg.cols) :
    Board.colHeight (adversarialTrace cfg convergedSolver s GameState.init n).board j ≤ cfg.rows :=
  buildSolver_columns_le hcols convergedSet_fixed hinit hl n hj

/-- The retrograde operator is idempotent on the converged region (it has fully halted). -/
theorem convergedSet_F_finite_idempotent :
    F_finite cfg (F_finite cfg (convergedSet cfg)) = convergedSet cfg := by
  rw [convergedSet_fixed, convergedSet_fixed]

/-- Per state the converged solver is an injective section of the menu. -/
theorem convergedSolver_response_portrait (hcols : 4 ≤ cfg.cols) (g : GameState) :
    (∀ p ∈ g.bag, convergedSolver (cfg := cfg) g p ∈ Placement.allValidFor cfg p) ∧
    (g.bag.image (fun p => convergedSolver (cfg := cfg) g p)).card = g.bag.card ∧
    Set.InjOn (fun p => convergedSolver (cfg := cfg) g p) g.bag :=
  buildSolver_response_portrait hcols convergedSet_fixed g

/-- Grand portrait of the canonical converged solver: valid, solves, closed atlas, confined. -/
theorem convergedSolver_grand_portrait (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    ValidSolver cfg convergedSolver ∧
    SolvesTetrisValid cfg convergedSolver ∧
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) ∧
    (∀ g, solverReachable (convergedSolver (cfg := cfg)) g → g ∈ convergedSet cfg) :=
  ⟨convergedSolver_validSolver hcols,
   convergedSolver_solves hcols hinit,
   convergedSolver_atlas_closed cfg,
   fun _ hr => convergedSolver_confined hinit hr⟩

/-- Grand safety portrait of the converged solver: in-region, safe, alive, in-field. -/
theorem convergedSolver_safety_portrait (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init n ∈ convergedSet cfg ∧
    adversarialTrace cfg convergedSolver s GameState.init n ∈ safe cfg ∧
    ¬ (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg ∧
    adversarialTrace cfg convergedSolver s GameState.init n ∈ inFieldStates cfg :=
  ⟨buildSolver_trace_mem convergedSet_fixed hinit hl n,
   convergedSolver_trace_mem_safe hcols hinit hl n,
   convergedSolver_survives hinit hl n,
   convergedSolver_trace_in_field hcols hinit hl n⟩

/-- Output grand portrait of the construction: piece-faithful, valid, in range, on menu. -/
theorem buildSolver_output_grand_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag) :
    (buildSolver hfix g p).piece = p ∧
    (buildSolver hfix g p).Valid cfg ∧
    (buildSolver hfix g p).col < cfg.cols ∧
    ((buildSolver hfix g p).rot : ℕ) < 4 ∧
    buildSolver hfix g p ∈ Placement.allValidFor cfg p :=
  ⟨buildSolver_announces_piece hcols hfix hp,
   buildSolver_output_valid hcols hfix hp,
   buildSolver_col_lt_cols hcols hfix hp,
   buildSolver_rot_lt_four hfix g p,
   (buildSolver_response_portrait hcols hfix g).1 p hp⟩

/-- Every state on a converged-solver play stays inside the converged region. -/
theorem convergedSolver_trace_mem (hinit : GameState.init ∈ convergedSet cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init n ∈ convergedSet cfg :=
  buildSolver_trace_mem convergedSet_fixed hinit hl n

/-! ### Synthesis capstones for the constructed function -/

/-- Grand solver portrait: the construction is valid, solves, closes an atlas, and is confined. -/
theorem buildSolver_grand_solver_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) :
    ValidSolver cfg (buildSolver hfix) ∧
    SolvesTetrisValid cfg (buildSolver hfix) ∧
    (buildSolver hfix).toAtlas.IsClosedOn cfg S ∧
    (∀ g, solverReachable (buildSolver hfix) g → g ∈ S) :=
  ⟨buildSolver_validSolver hcols hfix,
   buildSolver_solvesTetrisValid hcols hfix hinit,
   buildSolver_atlas_closed hfix,
   fun _ hr => buildSolver_reachable_mem hfix hinit hr⟩

/-- Grand safety portrait: along legal play the trace stays in-region, safe, alive, and in-field. -/
theorem buildSolver_grand_safety_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ S ∧
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ safe cfg ∧
    ¬ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).lost cfg ∧
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ inFieldStates cfg :=
  ⟨buildSolver_trace_mem hfix hinit hl n,
   buildSolver_trace_mem_safe hcols hfix hinit hl n,
   buildSolver_survives hfix hinit hl n,
   buildSolver_trace_in_field hcols hfix hinit hl n⟩

/-- Grand bounds portrait: count and height are 4-Lipschitz and every column stays in field. -/
theorem buildSolver_grand_bounds_portrait {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)).board.count
        ≤ (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board.count + 4 ∧
    Board.maxHeight cfg
          (adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)).board
        ≤ Board.maxHeight cfg
            (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board + 4 ∧
    (∀ j, j < cfg.cols →
      Board.colHeight (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board j
        ≤ cfg.rows) :=
  ⟨buildSolver_count_lipschitz hcols hfix hinit hl n,
   buildSolver_maxHeight_lipschitz hcols hfix hinit hl n,
   fun _ hj => buildSolver_columns_le hcols hfix hinit hl n hj⟩

/-- Step-law portrait: the orbit advances by one adversarial step, equally one solver step. -/
theorem buildSolver_step_law_portrait {S : Finset GameState} (hfix : F_finite cfg S = S)
    (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)
        = adversarialStep cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)
            (buildSolver hfix (adversarialTrace cfg (buildSolver hfix) s GameState.init n) (s n)) ∧
    adversarialTrace cfg (buildSolver hfix) s GameState.init (n + 1)
        = solverStep cfg (buildSolver hfix) (s n)
            (adversarialTrace cfg (buildSolver hfix) s GameState.init n) :=
  ⟨buildSolver_markov hfix s n, buildSolver_trace_eq_solverStep hfix s n⟩

/-- Output portrait of the converged solver: piece-faithful, valid, in column range. -/
theorem convergedSolver_output_portrait (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) :
    (convergedSolver (cfg := cfg) g p).piece = p ∧
    (convergedSolver (cfg := cfg) g p).Valid cfg ∧
    (convergedSolver (cfg := cfg) g p).col < cfg.cols :=
  ⟨buildSolver_announces_piece hcols convergedSet_fixed hp,
   buildSolver_output_valid hcols convergedSet_fixed hp,
   buildSolver_col_lt_cols hcols convergedSet_fixed hp⟩

/-- The converged solver is never stuck: every reachable bag piece has a valid placement. -/
theorem convergedSolver_never_stuck (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s)
    (n : ℕ) {p : Piece}
    (hp : p ∈ (adversarialTrace cfg convergedSolver s GameState.init n).bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  buildSolver_never_stuck hcols convergedSet_fixed hinit hl n hp

/-- The converged solver's atlas is total: it returns `some` placement everywhere. -/
theorem convergedSolver_toAtlas_total (g : GameState) (p : Piece) :
    (convergedSolver (cfg := cfg)).toAtlas g p = some (convergedSolver (cfg := cfg) g p) :=
  buildSolver_toAtlas_apply convergedSet_fixed g p

/-- The converged solver's reachable set is step-closed. -/
theorem convergedSolver_reachable_step_closed {g : GameState}
    (hr : solverReachable (convergedSolver (cfg := cfg)) g) {p : Piece} (hp : p ∈ g.bag) :
    solverReachable (convergedSolver (cfg := cfg))
      (adversarialStep cfg g p (convergedSolver (cfg := cfg) g p)) :=
  buildSolver_reachable_step_closed convergedSet_fixed hr hp

/-- Move bundle: a constructed move adds ≤4 cells, raises height ≤4, and never grows debt. -/
theorem buildSolver_move_bundle {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    ((buildSolver hfix g p).applyStep cfg b).count ≤ b.count + 4 ∧
    Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b) ≤ Board.maxHeight cfg b + 4 ∧
    HoleDebt.debt cfg ((buildSolver hfix g p).applyStep cfg b)
      ≤ HoleDebt.debt cfg ((buildSolver hfix g p).place b) :=
  ⟨buildSolver_move_count_le hcols hfix hp hWF,
   buildSolver_move_maxHeight_le hcols hfix hp b,
   buildSolver_clear_reduces_debt hcols hfix hp hWF⟩

/-- Geometry bundle: the drop is 4 cells, disjoint from the board, within a 4×4 footprint. -/
theorem buildSolver_geometry_bundle {S : Finset GameState} (hfix : F_finite cfg S = S)
    (g : GameState) (p : Piece) (b : Board) :
    ((buildSolver hfix g p).dropped b).card = 4 ∧
    Disjoint b ((buildSolver hfix g p).dropped b) ∧
    b ⊆ (buildSolver hfix g p).place b :=
  ⟨buildSolver_output_dropped_card hfix b g p,
   buildSolver_dropped_disjoint hfix b g p,
   buildSolver_move_superset hfix b g p⟩

/-- Energy bundle: after a constructed move the split identity and surface brackets hold. -/
theorem buildSolver_energy_bundle {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {g : GameState} {p : Piece} (hp : p ∈ g.bag)
    {b : Board} (hWF : Board.WF cfg b) :
    HoleDebt.debt cfg ((buildSolver hfix g p).applyStep cfg b)
        + ((buildSolver hfix g p).applyStep cfg b).count
      = HoleDebt.surfaceArea cfg ((buildSolver hfix g p).applyStep cfg b) ∧
    Board.maxHeight cfg ((buildSolver hfix g p).applyStep cfg b)
        ≤ HoleDebt.surfaceArea cfg ((buildSolver hfix g p).applyStep cfg b) :=
  ⟨buildSolver_move_energy_split hcols hfix hp hWF,
   (buildSolver_move_energy_brackets hfix g p b).1⟩

/-- Compression bundle: a piece-slice lands in the menu and is bounded by `4·cols`. -/
theorem buildSolver_compression_bundle {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) {p : Piece} (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) :
    T.image (fun g => buildSolver hfix g p) ⊆ Placement.allValidFor cfg p ∧
    (T.image (fun g => buildSolver hfix g p)).card ≤ cfg.cols * 4 :=
  ⟨buildSolver_image_per_piece_subset hcols hfix T hT,
   buildSolver_image_card hcols hfix T hT⟩

/-- The converged solver's cell count is 4-Lipschitz along play. -/
theorem convergedSolver_count_lipschitz (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg convergedSolver s GameState.init (n + 1)).board.count
      ≤ (adversarialTrace cfg convergedSolver s GameState.init n).board.count + 4 :=
  buildSolver_count_lipschitz hcols convergedSet_fixed hinit hl n

/-- The converged solver's max height is 4-Lipschitz along play. -/
theorem convergedSolver_maxHeight_lipschitz (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg (adversarialTrace cfg convergedSolver s GameState.init (n + 1)).board
      ≤ Board.maxHeight cfg (adversarialTrace cfg convergedSolver s GameState.init n).board + 4 :=
  buildSolver_maxHeight_lipschitz hcols convergedSet_fixed hinit hl n

/-- The converged solver keeps the cell count even (on an even-width board). -/
theorem convergedSolver_even_count (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) (hev : Even cfg.cols)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Even (adversarialTrace cfg convergedSolver s GameState.init n).board.count :=
  buildSolver_even_count hcols convergedSet_fixed hinit hev hl n

/-- Every state reachable by the converged solver is both safe and reachable. -/
theorem convergedSolver_operates_in_safe_and_reachable (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {g : GameState}
    (hr : solverReachable (convergedSolver (cfg := cfg)) g) :
    g ∈ safe cfg ∧ Reachable cfg g :=
  buildSolver_operates_in_safe_and_reachable hcols convergedSet_fixed hinit hr

/-- The converged solver revisits a state within `|inFieldStates|` steps. -/
theorem convergedSolver_repeats_within_inFieldStates (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ (inFieldStates cfg).card ∧
      adversarialTrace cfg convergedSolver s GameState.init i
        = adversarialTrace cfg convergedSolver s GameState.init j :=
  buildSolver_repeats_within_inFieldStates hcols convergedSet_fixed hinit hl

/-- A no-clear run of the converged solver is capacity-bounded. -/
theorem convergedSolver_no_clear_window_bounded (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ)
    (hno : ∀ k < n,
        (adversarialTrace cfg convergedSolver s GameState.init (k + 1)).board.count
          = (adversarialTrace cfg convergedSolver s GameState.init k).board.count + 4) :
    4 * n ≤ cfg.cols * cfg.rows :=
  buildSolver_no_clear_window_bounded hcols convergedSet_fixed hinit hl n hno

/-- The converged solver's per-state response table has exactly `bag.card` entries. -/
theorem convergedSolver_response_table_card_eq (hcols : 4 ≤ cfg.cols) (g : GameState) :
    (g.bag.image (fun p => convergedSolver (cfg := cfg) g p)).card = g.bag.card :=
  buildSolver_response_table_card_eq hcols convergedSet_fixed g

/-- The converged solver's output lies in the global action set. -/
theorem convergedSolver_output_in_total_action_set (hcols : 4 ≤ cfg.cols) {g : GameState}
    {p : Piece} (hp : p ∈ g.bag) :
    convergedSolver (cfg := cfg) g p
      ∈ (Finset.univ : Finset Piece).biUnion (Placement.allValidFor cfg) :=
  buildSolver_output_in_total_action_set hcols convergedSet_fixed hp

/-- The converged solver collides per-piece on a slice larger than the menu (pigeonhole). -/
theorem convergedSolver_per_piece_noninjective (hcols : 4 ≤ cfg.cols) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag)
    (hcard : (Placement.allValidFor cfg p).card < T.card) :
    ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧
      convergedSolver (cfg := cfg) g₁ p = convergedSolver (cfg := cfg) g₂ p :=
  buildSolver_per_piece_noninjective hcols convergedSet_fixed T hT hcard

/-- The converged solver's cell count never exceeds capacity. -/
theorem convergedSolver_trace_count (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg convergedSolver s GameState.init n).board.count ≤ cfg.cols * cfg.rows :=
  buildSolver_trace_count hcols convergedSet_fixed hinit hl n

/-- The converged solver's max height never exceeds the board height. -/
theorem convergedSolver_trace_maxHeight (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.rows :=
  buildSolver_trace_maxHeight hcols convergedSet_fixed hinit hl n

/-- The converged solver's hole-debt never exceeds capacity. -/
theorem convergedSolver_trace_debt (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.debt cfg
      (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.cols * cfg.rows :=
  buildSolver_trace_debt hcols convergedSet_fixed hinit hl n

/-- No converged-solver cell ever lands in the death zone (row ≥ rows). -/
theorem convergedSolver_no_death_cell (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ)
    {c : Coord} (hc : c ∈ (adversarialTrace cfg convergedSolver s GameState.init n).board) :
    c.2 < cfg.rows :=
  buildSolver_no_death_cell hcols convergedSet_fixed hinit hl n hc

/-- Three pillars of the converged solver's play: safe, bounded height, bounded count. -/
theorem convergedSolver_pillars (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init n ∈ safe cfg ∧
    Board.maxHeight cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.rows ∧
    (adversarialTrace cfg convergedSolver s GameState.init n).board.count
      ≤ cfg.cols * cfg.rows :=
  buildSolver_pillars hcols convergedSet_fixed hinit hl n

/-- The converged solver is non-losing exactly when its max height is within the board. -/
theorem convergedSolver_not_lost_iff (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ¬ (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg ↔
      Board.maxHeight cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.rows :=
  buildSolver_not_lost_iff hcols convergedSet_fixed hinit hl n

/-- Every board on a converged-solver play is well-formed. -/
theorem convergedSolver_trace_wf (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg convergedSolver s GameState.init n).board :=
  buildSolver_trace_wf hcols convergedSet_fixed hinit hl n

/-- **Grand characterization of the canonical construction.** If init survives the terminating
descending iteration, the explicit converged solver solves Tetris, closes an atlas over the
converged region, never leaves that region, and that region fits inside the in-field universe. -/
theorem convergedSolver_grand_characterization (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    SolvesTetrisValid cfg convergedSolver ∧
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) ∧
    (∀ g, solverReachable (convergedSolver (cfg := cfg)) g → g ∈ convergedSet cfg) ∧
    (convergedSet cfg).card ≤ (inFieldStates cfg).card :=
  ⟨convergedSolver_solves hcols hinit,
   convergedSolver_atlas_closed cfg,
   fun _ hr => convergedSolver_confined hinit hr,
   convergedSet_card_le cfg⟩

/-- The converged solver's orbit advances by one Markov step. -/
theorem convergedSolver_markov (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init (n + 1)
      = adversarialStep cfg (adversarialTrace cfg convergedSolver s GameState.init n) (s n)
          (convergedSolver (cfg := cfg)
            (adversarialTrace cfg convergedSolver s GameState.init n) (s n)) :=
  buildSolver_markov convergedSet_fixed s n

/-- The converged solver is causal: its play depends only on past pieces. -/
theorem convergedSolver_no_lookahead (s s' : ℕ → Piece) (n : ℕ) (h : ∀ i < n, s i = s' i) :
    adversarialTrace cfg convergedSolver s GameState.init n
      = adversarialTrace cfg convergedSolver s' GameState.init n :=
  buildSolver_no_lookahead convergedSet_fixed s s' n h

/-- Dynamical portrait of the converged solver: orbit is iterated step; bag/board laws. -/
theorem convergedSolver_dynamical_portrait (hcols : 4 ≤ cfg.cols) (p : Piece) :
    (∀ n, adversarialTrace cfg convergedSolver (fun _ => p) GameState.init n
        = (solverStep cfg convergedSolver p)^[n] GameState.init) ∧
    (∀ g, (solverStep cfg convergedSolver p g).bag = g.bag.draw p) ∧
    (∀ g, p ∈ g.bag →
      (solverStep cfg convergedSolver p g).board
        = (convergedSolver (cfg := cfg) g p).applyStep cfg g.board) :=
  buildSolver_dynamical_portrait hcols convergedSet_fixed p

/-- The converged region is the greatest in-field fixed point: every fixed `S ⊆ inFieldStates`
sits inside it. So the construction loses no certifiable state. -/
theorem fixed_inField_subset_convergedSet {S : Finset GameState}
    (hfix : F_finite cfg S = S) (hsub : S ⊆ inFieldStates cfg) :
    S ⊆ convergedSet cfg := by
  have key : ∀ n, S ⊆ safeIterFinite cfg (inFieldStates cfg) n := by
    intro n
    induction n with
    | zero => simpa using hsub
    | succ k ih =>
        rw [safeIterFinite_succ]
        calc S = F_finite cfg S := hfix.symm
          _ ⊆ F_finite cfg (safeIterFinite cfg (inFieldStates cfg) k) := F_finite_mono cfg ih
  exact key (inFieldStates cfg).card

/-- Any in-field fixed region containing init feeds the canonical construction to a solver. -/
theorem fixed_inField_init_solvable (hcols : 4 ≤ cfg.cols) {S : Finset GameState}
    (hfix : F_finite cfg S = S) (hsub : S ⊆ inFieldStates cfg) (hinit : GameState.init ∈ S) :
    TetrisSolvableValidFor cfg :=
  init_mem_convergedSet_solvable hcols (fixed_inField_subset_convergedSet hfix hsub hinit)

/-- The converged region characterized: an in-field fixed point that dominates all such. -/
theorem convergedSet_is_greatest_inField_fixed :
    (F_finite cfg (convergedSet cfg) = convergedSet cfg ∧
      convergedSet cfg ⊆ inFieldStates cfg) ∧
    (∀ S, F_finite cfg S = S → S ⊆ inFieldStates cfg → S ⊆ convergedSet cfg) :=
  ⟨⟨convergedSet_fixed, convergedSet_subset_inFieldStates cfg⟩,
   fun _ hfix hsub => fixed_inField_subset_convergedSet hfix hsub⟩

/-- A converged-solver move adds at most four cells. -/
theorem convergedSolver_move_count_le (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) :
    ((convergedSolver (cfg := cfg) g p).applyStep cfg b).count ≤ b.count + 4 :=
  buildSolver_move_count_le hcols convergedSet_fixed hp hWF

/-- A converged-solver move raises max height by at most four. -/
theorem convergedSolver_move_maxHeight_le (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) (b : Board) :
    Board.maxHeight cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b)
      ≤ Board.maxHeight cfg b + 4 :=
  buildSolver_move_maxHeight_le hcols convergedSet_fixed hp b

/-- On a low stack the converged solver's move is non-losing. -/
theorem convergedSolver_lowstack_move_safe (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {g : GameState} {p : Piece} (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b)
    (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4) :
    ¬ Board.isLost cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b) :=
  buildSolver_lowstack_move_safe hcols convergedSet_fixed hrows hp hWF hlow

/-- The converged solver's opening move never loses. -/
theorem convergedSolver_opening_safe (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {p : Piece} (hp : p ∈ GameState.init.bag) :
    ¬ Board.isLost cfg
        ((convergedSolver (cfg := cfg) GameState.init p).applyStep cfg GameState.init.board) :=
  buildSolver_opening_safe hcols convergedSet_fixed hrows hp

/-- Re-running the construction from its own output is a fixed point: nothing more is pruned. -/
theorem safeIterFinite_convergedSet_eq (n : ℕ) :
    safeIterFinite cfg (convergedSet cfg) n = convergedSet cfg := by
  induction n with
  | zero => rfl
  | succ k ih => rw [safeIterFinite_succ, ih, convergedSet_fixed]

/-- The converged solver's raw placement lays exactly four cells. -/
theorem convergedSolver_places_four (b : Board) (g : GameState) (p : Piece) :
    ((convergedSolver (cfg := cfg) g p).place b).count = b.count + 4 :=
  buildSolver_places_four convergedSet_fixed b g p

/-- The converged solver's output code is bounded by `4·cols`. -/
theorem convergedSolver_output_code_lt (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) :
    4 * (convergedSolver (cfg := cfg) g p).col + ((convergedSolver (cfg := cfg) g p).rot : ℕ)
      < 4 * cfg.cols :=
  buildSolver_output_code_lt hcols convergedSet_fixed hp

/-- The converged solver reads only board and bag, not any other state. -/
theorem convergedSolver_reads_board_bag (g₁ g₂ : GameState) (p : Piece)
    (hb : g₁.board = g₂.board) (hbag : g₁.bag = g₂.bag) :
    convergedSolver (cfg := cfg) g₁ p = convergedSolver (cfg := cfg) g₂ p :=
  buildSolver_reads_board_bag convergedSet_fixed g₁ g₂ p hb hbag

/-- Every state the converged solver visits is reachable from the empty board. -/
theorem convergedSolver_trace_reachable (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg convergedSolver s GameState.init n) :=
  buildSolver_trace_reachable hcols convergedSet_fixed hinit hl n

/-- The converged solver's output sits in the rotation×column grid. -/
theorem convergedSolver_output_in_grid (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) :
    ((convergedSolver (cfg := cfg) g p).rot, (convergedSolver (cfg := cfg) g p).col)
      ∈ (Finset.univ : Finset Rotation) ×ˢ Finset.range cfg.cols :=
  buildSolver_output_in_grid hcols convergedSet_fixed hp

/-- The converged solver's output is the canonical triple `⟨p, rot, col⟩`. -/
theorem convergedSolver_output_eq_mk (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) :
    convergedSolver (cfg := cfg) g p
      = ⟨p, (convergedSolver (cfg := cfg) g p).rot, (convergedSolver (cfg := cfg) g p).col⟩ :=
  buildSolver_output_eq_mk hcols convergedSet_fixed hp

/-- The converged solver is exactly `buildSolver` over the converged-region fixed point. -/
theorem convergedSolver_eq_buildSolver :
    (convergedSolver : Solver cfg) = buildSolver convergedSet_fixed :=
  rfl

/-- The construction halts: one more round past `|inFieldStates|` adds nothing. -/
theorem construct_halts_in_card_rounds :
    safeIterFinite cfg (inFieldStates cfg) ((inFieldStates cfg).card + 1) = convergedSet cfg :=
  safeIterFinite_inFieldStates_stable_at_card cfg

/-- Impossibility detection: if Tetris is unsolvable, init is pruned from the converged region. -/
theorem init_notMem_convergedSet_of_not_solvable (hcols : 4 ≤ cfg.cols)
    (h : ¬ TetrisSolvableValidFor cfg) : GameState.init ∉ convergedSet cfg :=
  fun hinit => h (init_mem_convergedSet_solvable hcols hinit)

/-- Footprint portrait of the converged solver: reachable in-field and finitely many placements. -/
theorem convergedSolver_footprint_portrait (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    (∀ g, solverReachable (convergedSolver (cfg := cfg)) g → g ∈ inFieldStates cfg) ∧
    {pl : Placement | ∃ g p, p ∈ g.bag ∧ convergedSolver (cfg := cfg) g p = pl}.Finite :=
  buildSolver_footprint_portrait hcols convergedSet_fixed hinit

/-- Reachability invariant: each converged-reachable state is in-region, safe, in-field, alive. -/
theorem convergedSolver_invariant (hinit : GameState.init ∈ convergedSet cfg) {g : GameState}
    (hr : solverReachable (convergedSolver (cfg := cfg)) g) :
    g ∈ convergedSet cfg ∧ g ∈ safe cfg ∧ g ∈ inFieldStates cfg ∧ ¬ g.lost cfg := by
  have hc := convergedSolver_confined hinit hr
  exact ⟨hc, convergedSet_subset_safe (Finset.mem_coe.mpr hc),
    convergedSet_subset_inFieldStates cfg hc, convergedSet_not_lost hc⟩

/-- Near capacity, any surviving converged-solver move must clear a line. -/
theorem convergedSolver_move_must_clear (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) (hnear : cfg.cols * cfg.rows < b.count + 4)
    (hsurv : ¬ Board.isLost cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b)) :
    0 < Board.linesCleared cfg ((convergedSolver (cfg := cfg) g p).place b) :=
  buildSolver_move_must_clear hcols convergedSet_fixed hp hWF hnear hsurv

/-- Clearing in a converged-solver move never increases hole-debt. -/
theorem convergedSolver_clear_reduces_debt (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) :
    HoleDebt.debt cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b)
      ≤ HoleDebt.debt cfg ((convergedSolver (cfg := cfg) g p).place b) :=
  buildSolver_clear_reduces_debt hcols convergedSet_fixed hp hWF

/-- Drop geometry of the converged solver: 4 cells, disjoint, additive on the board. -/
theorem convergedSolver_drop_geometry (g : GameState) (p : Piece) (b : Board) :
    ((convergedSolver (cfg := cfg) g p).dropped b).card = 4 ∧
    Disjoint b ((convergedSolver (cfg := cfg) g p).dropped b) ∧
    b ⊆ (convergedSolver (cfg := cfg) g p).place b :=
  ⟨buildSolver_output_dropped_card convergedSet_fixed b g p,
   buildSolver_dropped_disjoint convergedSet_fixed b g p,
   buildSolver_move_superset convergedSet_fixed b g p⟩

/-- The converged region is contained in every earlier iterate (descending chain). -/
theorem convergedSet_subset_iterate {n : ℕ} (hn : n ≤ (inFieldStates cfg).card) :
    convergedSet cfg ⊆ safeIterFinite cfg (inFieldStates cfg) n :=
  construct_death_propagation_monotone hn

/-- The converged solver's surface area never exceeds capacity. -/
theorem convergedSolver_trace_surfaceArea (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    HoleDebt.surfaceArea cfg
      (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.cols * cfg.rows :=
  buildSolver_trace_surfaceArea hcols convergedSet_fixed hinit hl n

/-- Along converged play, every queried piece is in the current bag. -/
theorem convergedSolver_queried_in_bag (s : ℕ → Piece) (hl : LegalSequence s) (n : ℕ) :
    s n ∈ (adversarialTrace cfg convergedSolver s GameState.init n).bag :=
  buildSolver_queried_in_bag convergedSet_fixed s hl n

/-- Along legal play the converged solver's chosen move is on the menu. -/
theorem convergedSolver_play_outputs_in_menu (hcols : 4 ≤ cfg.cols) (s : ℕ → Piece)
    (hl : LegalSequence s) (n : ℕ) :
    convergedSolver (cfg := cfg)
        (adversarialTrace cfg convergedSolver s GameState.init n) (s n)
      ∈ Placement.allValidFor cfg (s n) :=
  buildSolver_play_outputs_in_menu hcols convergedSet_fixed s hl n

/-- Every cell the converged solver lays sits within the playfield rectangle. -/
theorem convergedSolver_board_in_field (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ)
    {c : Coord} (hc : c ∈ (adversarialTrace cfg convergedSolver s GameState.init n).board) :
    c.1 < cfg.cols ∧ c.2 < cfg.rows :=
  buildSolver_board_in_field hcols convergedSet_fixed hinit hl n hc

/-- When init is in the converged region, every opening piece has a safe response. -/
theorem convergedSolver_handles_init (hinit : GameState.init ∈ convergedSet cfg) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      adversarialStep cfg GameState.init p pl ∈ safe cfg :=
  construct_handles_all_seven_at_init (convergedSet_subset_safe (Finset.mem_coe.mpr hinit)) p

/-- Trace-invariant bundle: each visited state is WF, reachable, and inside the region. -/
theorem buildSolver_trace_invariant_bundle {S : Finset GameState} (hcols : 4 ≤ cfg.cols)
    (hfix : F_finite cfg S = S) (hinit : GameState.init ∈ S) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n).board ∧
    Reachable cfg (adversarialTrace cfg (buildSolver hfix) s GameState.init n) ∧
    adversarialTrace cfg (buildSolver hfix) s GameState.init n ∈ S :=
  ⟨buildSolver_trace_wf hcols hfix hinit hl n,
   buildSolver_trace_reachable hcols hfix hinit hl n,
   buildSolver_trace_mem hfix hinit hl n⟩

/-- The converged solver maps distinct bag pieces to distinct placements. -/
theorem convergedSolver_outputs_differ_by_piece (hcols : 4 ≤ cfg.cols) {g : GameState}
    {p p' : Piece} (hp : p ∈ g.bag) (hp' : p' ∈ g.bag) (hne : p ≠ p') :
    convergedSolver (cfg := cfg) g p ≠ convergedSolver (cfg := cfg) g p' :=
  buildSolver_outputs_differ_by_piece hcols convergedSet_fixed hp hp' hne

/-- The converged solver's piece-slice image is contained in that piece's menu. -/
theorem convergedSolver_image_per_piece_subset (hcols : 4 ≤ cfg.cols) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    T.image (fun g => convergedSolver (cfg := cfg) g p) ⊆ Placement.allValidFor cfg p :=
  buildSolver_image_per_piece_subset hcols convergedSet_fixed T hT

/-- The placements the converged solver realizes along any legal play form a finite set. -/
theorem convergedSolver_realized_outputs_finite (hcols : 4 ≤ cfg.cols) (s : ℕ → Piece)
    (hl : LegalSequence s) :
    (Set.range fun n =>
      convergedSolver (cfg := cfg)
        (adversarialTrace cfg convergedSolver s GameState.init n) (s n)).Finite :=
  buildSolver_realized_outputs_finite hcols convergedSet_fixed s hl

/-- The converged solver's per-piece image has at most `4·cols` distinct outputs. -/
theorem convergedSolver_image_card (hcols : 4 ≤ cfg.cols) {p : Piece} (T : Finset GameState)
    (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => convergedSolver (cfg := cfg) g p)).card ≤ cfg.cols * 4 :=
  buildSolver_image_card hcols convergedSet_fixed T hT

/-- The converged solver's response table at a state has at most `bag.card` entries. -/
theorem convergedSolver_response_table_card_le (g : GameState) :
    (g.bag.image (fun p => convergedSolver (cfg := cfg) g p)).card ≤ g.bag.card :=
  buildSolver_response_table_card_le convergedSet_fixed g

/-- The converged solver reads only board and bag (eta over the state record). -/
theorem convergedSolver_eta (g : GameState) (p : Piece) :
    convergedSolver (cfg := cfg) g p = convergedSolver (cfg := cfg) ⟨g.board, g.bag⟩ p :=
  buildSolver_eta convergedSet_fixed g p

/-- For each piece the converged solver maps in-bag states into that piece's menu. -/
theorem convergedSolver_slice_mapsTo (hcols : 4 ≤ cfg.cols) (p : Piece) :
    Set.MapsTo (fun g => convergedSolver (cfg := cfg) g p) {g | p ∈ g.bag}
      ↑(Placement.allValidFor cfg p) :=
  buildSolver_slice_mapsTo hcols convergedSet_fixed p

/-- Atlas portrait of the converged solver: total, some-valued, round-trips, injective. -/
theorem convergedSolver_atlas_portrait :
    (∀ g p, (convergedSolver (cfg := cfg)).toAtlas g p
        = some (convergedSolver (cfg := cfg) g p)) ∧
    (∀ g p, ((convergedSolver (cfg := cfg)).toAtlas g p).isSome = true) ∧
    ((convergedSolver (cfg := cfg)).toAtlas.toSolver = convergedSolver (cfg := cfg)) ∧
    (∀ (σ₁ σ₂ : Solver cfg), σ₁.toAtlas = σ₂.toAtlas → σ₁ = σ₂) :=
  buildSolver_atlas_portrait convergedSet_fixed

/-- If the converged solver's trace and the piece stream both repeat, the orbit is periodic. -/
theorem convergedSolver_periodic_play (s : ℕ → Piece) (g0 : GameState) {b d : ℕ}
    (htrace : adversarialTrace cfg convergedSolver s g0 b
        = adversarialTrace cfg convergedSolver s g0 (b + d))
    (hs : ∀ k, s (b + k) = s (b + d + k)) (k : ℕ) :
    adversarialTrace cfg convergedSolver s g0 (b + k)
      = adversarialTrace cfg convergedSolver s g0 (b + d + k) :=
  buildSolver_periodic_play convergedSet_fixed s g0 htrace hs k

/-- Trace composition (semigroup law) for the converged solver's orbit. -/
theorem convergedSolver_trace_compose (s : ℕ → Piece) (g0 : GameState) (n m : ℕ) :
    adversarialTrace cfg convergedSolver s g0 (n + m) =
      adversarialTrace cfg convergedSolver (fun k => s (n + k))
        (adversarialTrace cfg convergedSolver s g0 n) m :=
  buildSolver_trace_compose convergedSet_fixed s g0 n m

/-- One-step unfolding of the converged solver's trace. -/
theorem convergedSolver_trace_eq_solverStep (s : ℕ → Piece) (n : ℕ) :
    adversarialTrace cfg convergedSolver s GameState.init (n + 1)
      = solverStep cfg convergedSolver (s n)
          (adversarialTrace cfg convergedSolver s GameState.init n) :=
  buildSolver_trace_eq_solverStep convergedSet_fixed s n

/-- Under a constant piece stream the converged solver's trace is an iterated step map. -/
theorem convergedSolver_trace_const_eq_iterate (p : Piece) (n : ℕ) :
    adversarialTrace cfg convergedSolver (fun _ => p) GameState.init n
      = (solverStep cfg convergedSolver p)^[n] GameState.init :=
  buildSolver_trace_const_eq_iterate convergedSet_fixed p n

/-- The converged solver's out-degree at a state is at most the bag size. -/
theorem convergedSolver_outdegree_le (g : GameState) :
    (g.bag.image (fun p => solverStep cfg convergedSolver p g)).card ≤ g.bag.card :=
  buildSolver_outdegree_le convergedSet_fixed g

/-- The constructed atlas is a complete M4 artifact: closed, total, containing init. -/
theorem buildSolver_M4_artifact {S : Finset GameState} (hfix : F_finite cfg S = S)
    (hinit : GameState.init ∈ S) :
    (buildSolver hfix).toAtlas.IsClosedOn cfg S ∧
    (∀ g p, ((buildSolver hfix).toAtlas g p).isSome = true) ∧
    GameState.init ∈ S :=
  ⟨buildSolver_atlas_closed hfix, fun g p => buildSolver_toAtlas_isSome hfix g p, hinit⟩

/-- The canonical converged atlas is a complete M4 artifact for the converged region. -/
theorem convergedSolver_M4_artifact (hinit : GameState.init ∈ convergedSet cfg) :
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) ∧
    (∀ g p, ((convergedSolver (cfg := cfg)).toAtlas g p).isSome = true) ∧
    GameState.init ∈ convergedSet cfg :=
  ⟨convergedSolver_atlas_closed cfg,
   fun g p => buildSolver_toAtlas_isSome convergedSet_fixed g p, hinit⟩

/-- **Discovery by construction.** A single membership test `init ∈ convergedSet` yields, at once,
solvability, an explicit winning solver, and a closed atlas — existence found constructively. -/
theorem solvability_discovered_by_construction (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    TetrisSolvableValidFor cfg ∧
    SolvesTetrisValid cfg convergedSolver ∧
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) :=
  ⟨init_mem_convergedSet_solvable hcols hinit,
   convergedSolver_solves hcols hinit,
   convergedSolver_atlas_closed cfg⟩

/-- The converged solver as an uncurried function agrees with its curried form. -/
theorem convergedSolver_uncurry (g : GameState) (p : Piece) :
    Function.uncurry (convergedSolver (cfg := cfg)) (g, p) = convergedSolver (cfg := cfg) g p :=
  buildSolver_uncurry convergedSet_fixed g p

/-- The converged solver round-trips through its atlas representation. -/
theorem convergedSolver_toAtlas_toSolver :
    (convergedSolver (cfg := cfg)).toAtlas.toSolver = convergedSolver (cfg := cfg) :=
  buildSolver_toAtlas_toSolver convergedSet_fixed

/-- The converged solver's reachable footprint is bounded by the converged region. -/
theorem convergedSolver_footprint_card_le (hinit : GameState.init ∈ convergedSet cfg) :
    ∀ g, solverReachable (convergedSolver (cfg := cfg)) g → g ∈ convergedSet cfg :=
  buildSolver_footprint_card_le convergedSet_fixed hinit

/-- The converged solver, as a per-state section, returns the queried piece. -/
theorem convergedSolver_section (hcols : 4 ≤ cfg.cols) (g : GameState) {p : Piece}
    (hp : p ∈ g.bag) :
    (Placement.piece ∘ convergedSolver (cfg := cfg) g) p = p :=
  buildSolver_section hcols convergedSet_fixed g hp

/-- The converged solver's next board is its move applied to the current board. -/
theorem convergedSolver_trace_board_succ (hcols : 4 ≤ cfg.cols) {s : ℕ → Piece}
    (hl : LegalSequence s) (n : ℕ) :
    (adversarialTrace cfg convergedSolver s GameState.init (n + 1)).board
      = (convergedSolver (cfg := cfg)
          (adversarialTrace cfg convergedSolver s GameState.init n) (s n)).applyStep cfg
          (adversarialTrace cfg convergedSolver s GameState.init n).board :=
  buildSolver_trace_board_succ hcols convergedSet_fixed hl n

/-- Bridge from M2: an in-field closed cycle through init lands init in the converged region. -/
theorem init_mem_convergedSet_of_inField_cycle (C : AdversarialClosedCycle cfg)
    (hS : C.states ⊆ inFieldStates cfg) (hinit : GameState.init ∈ C.states) :
    GameState.init ∈ convergedSet cfg :=
  C.init_mem_safeIterFinite hS hinit (inFieldStates cfg).card

/-- An in-field init-cycle hands the canonical converged solver a Tetris win. -/
theorem inField_cycle_yields_convergedSolver (hcols : 4 ≤ cfg.cols)
    (C : AdversarialClosedCycle cfg) (hS : C.states ⊆ inFieldStates cfg)
    (hinit : GameState.init ∈ C.states) :
    SolvesTetrisValid cfg convergedSolver :=
  convergedSolver_solves hcols (init_mem_convergedSet_of_inField_cycle C hS hinit)

/-- Bounds bundle: along converged play, height, count, debt, surface all stay in budget. -/
theorem convergedSolver_trace_bounds_bundle (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Board.maxHeight cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.rows ∧
    (adversarialTrace cfg convergedSolver s GameState.init n).board.count
        ≤ cfg.cols * cfg.rows ∧
    HoleDebt.debt cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.cols * cfg.rows ∧
    HoleDebt.surfaceArea cfg
        (adversarialTrace cfg convergedSolver s GameState.init n).board ≤ cfg.cols * cfg.rows :=
  ⟨convergedSolver_trace_maxHeight hcols hinit hl n,
   convergedSolver_trace_count hcols hinit hl n,
   convergedSolver_trace_debt hcols hinit hl n,
   convergedSolver_trace_surfaceArea hcols hinit hl n⟩

/-- If init survives, the converged region is nonempty. -/
theorem convergedSet_nonempty_of_init_mem (hinit : GameState.init ∈ convergedSet cfg) :
    (convergedSet cfg).Nonempty :=
  ⟨GameState.init, hinit⟩

/-- The converged solver is an explicit existence witness when init survives. -/
theorem convergedSolver_witnesses_existence (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  ⟨convergedSolver, convergedSolver_solves hcols hinit⟩

/-- There exists not just a solver but a region-confined one when init survives. -/
theorem exists_confined_solver_of_init_mem_convergedSet (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    ∃ σ : Solver cfg, SolvesTetrisValid cfg σ ∧
      (∀ g, solverReachable σ g → g ∈ convergedSet cfg) :=
  ⟨convergedSolver, convergedSolver_solves hcols hinit,
   fun _ hr => convergedSolver_confined hinit hr⟩

/-- The converged region sits inside both the safe set and the in-field universe. -/
theorem convergedSet_subset_safe_inter_inField :
    ↑(convergedSet cfg) ⊆ safe cfg ∧ convergedSet cfg ⊆ inFieldStates cfg :=
  ⟨convergedSet_subset_safe, convergedSet_subset_inFieldStates cfg⟩

/-- Every converged state is safe, in-field, and non-losing. -/
theorem convergedSet_mem_props {g : GameState} (hg : g ∈ convergedSet cfg) :
    g ∈ safe cfg ∧ g ∈ inFieldStates cfg ∧ ¬ g.lost cfg :=
  ⟨convergedSet_subset_safe (Finset.mem_coe.mpr hg),
   convergedSet_subset_inFieldStates cfg hg, convergedSet_not_lost hg⟩

/-- Memorylessness bundle: the converged solver depends only on the current board and bag. -/
theorem convergedSolver_memoryless_bundle (g₁ g₂ : GameState) (p : Piece)
    (hb : g₁.board = g₂.board) (hbag : g₁.bag = g₂.bag) :
    convergedSolver (cfg := cfg) g₁ p = convergedSolver (cfg := cfg) g₂ p ∧
    convergedSolver (cfg := cfg) g₁ p = convergedSolver (cfg := cfg) ⟨g₁.board, g₁.bag⟩ p :=
  ⟨convergedSolver_reads_board_bag g₁ g₂ p hb hbag, convergedSolver_eta g₁ p⟩

/-- Compression bundle: a converged piece-slice lands in the menu and is bounded by `4·cols`. -/
theorem convergedSolver_compression_bundle (hcols : 4 ≤ cfg.cols) {p : Piece}
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    T.image (fun g => convergedSolver (cfg := cfg) g p) ⊆ Placement.allValidFor cfg p ∧
    (T.image (fun g => convergedSolver (cfg := cfg) g p)).card ≤ cfg.cols * 4 :=
  ⟨convergedSolver_image_per_piece_subset hcols T hT, convergedSolver_image_card hcols T hT⟩

/-- The converged solver's output rotation is one of four. -/
theorem convergedSolver_rot_lt_four (g : GameState) (p : Piece) :
    ((convergedSolver (cfg := cfg) g p).rot : ℕ) < 4 :=
  buildSolver_rot_lt_four convergedSet_fixed g p

/-- On the empty board the converged solver drops to the floor (offset zero). -/
theorem convergedSolver_empty_drop_zero (g : GameState) (p : Piece) :
    (convergedSolver (cfg := cfg) g p).dropOffset Board.empty = 0 :=
  buildSolver_empty_drop_zero convergedSet_fixed g p

/-- A converged-solver placement only raises per-column heights. -/
theorem convergedSolver_place_raises_columns (g : GameState) (p : Piece) (b : Board) (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight ((convergedSolver (cfg := cfg) g p).place b) j :=
  buildSolver_place_raises_columns convergedSet_fixed g p b j

/-- A converged-solver placement is skyline-monotone in the dominance order. -/
theorem convergedSolver_move_skyline_monotone (g : GameState) (p : Piece) {b β : Board}
    (h : WqoCarrier.domLE b β) :
    WqoCarrier.domLE ((convergedSolver (cfg := cfg) g p).place b)
      ((convergedSolver (cfg := cfg) g p).place β) :=
  buildSolver_move_skyline_monotone convergedSet_fixed g p h

/-- Two distinct witnesses: when init survives, the explicit converged solver and the
canonical `safeSolver` both win. -/
theorem convergedSolver_and_safeSolver_both_solve (hcols : 4 ≤ cfg.cols)
    (hinit : GameState.init ∈ convergedSet cfg) :
    SolvesTetrisValid cfg convergedSolver ∧ SolvesTetrisValid cfg (safeSolver cfg) :=
  ⟨convergedSolver_solves hcols hinit,
   init_safe_implies_solvesTetrisValid hcols
     (convergedSet_subset_safe (Finset.mem_coe.mpr hinit))⟩

/-- Every converged state has a placement for each bag piece keeping the game safe. -/
theorem convergedSet_safe_response {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧ adversarialStep cfg g p pl ∈ safe cfg :=
  construct_safe_response_each_piece (convergedSet_subset_safe (Finset.mem_coe.mpr hg)) hp

/-- **The canonical construction, end to end**: it terminates, is sound, yields an explicit
confined winning solver with a closed atlas exactly when init survives, and otherwise certifies
impossibility — all decided by the single membership test `init ∈ convergedSet`. -/
theorem the_canonical_construction (hcols : 4 ≤ cfg.cols) :
    (safeIterFinite cfg (inFieldStates cfg) ((inFieldStates cfg).card + 1) = convergedSet cfg) ∧
    (↑(convergedSet cfg) ⊆ safe cfg) ∧
    (GameState.init ∈ convergedSet cfg →
      SolvesTetrisValid cfg convergedSolver ∧
      (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg)) ∧
    (¬ TetrisSolvableValidFor cfg → GameState.init ∉ convergedSet cfg) :=
  ⟨construct_halts_in_card_rounds,
   convergedSet_subset_safe,
   fun hinit => ⟨convergedSolver_solves hcols hinit, convergedSolver_atlas_closed cfg⟩,
   init_notMem_convergedSet_of_not_solvable hcols⟩

/-- A fixed region is invariant under the induced self-map of the constructed solver. -/
theorem buildSolver_solverStep_mem {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    solverStep cfg (buildSolver hfix) p g ∈ S :=
  buildSolver_step_mem hfix hg hp

/-- The converged region is invariant under the converged solver's induced self-map. -/
theorem convergedSolver_solverStep_mem {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    solverStep cfg convergedSolver p g ∈ convergedSet cfg :=
  buildSolver_step_mem convergedSet_fixed hg hp

/-- One converged step from a converged state lands in the safe set. -/
theorem convergedSolver_solverStep_safe {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    solverStep cfg convergedSolver p g ∈ safe cfg :=
  convergedSet_subset_safe (Finset.mem_coe.mpr (convergedSolver_solverStep_mem hg hp))

/-- One converged step from a converged state is non-losing. -/
theorem convergedSolver_solverStep_not_lost {g : GameState} (hg : g ∈ convergedSet cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ¬ (solverStep cfg convergedSolver p g).lost cfg :=
  convergedSet_not_lost (convergedSolver_solverStep_mem hg hp)

/-- A constructed step from any fixed region lands in the safe set. -/
theorem buildSolver_solverStep_safe {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    solverStep cfg (buildSolver hfix) p g ∈ safe cfg :=
  fixed_point_subset_safe hfix (Finset.mem_coe.mpr (buildSolver_step_mem hfix hg hp))

/-- A constructed step from any fixed region is non-losing. -/
theorem buildSolver_step_not_lost {S : Finset GameState} (hfix : F_finite cfg S = S)
    {g : GameState} (hg : g ∈ S) {p : Piece} (hp : p ∈ g.bag) :
    ¬ (adversarialStep cfg g p (buildSolver hfix g p)).lost cfg :=
  safe_not_lost
    (fixed_point_subset_safe hfix (Finset.mem_coe.mpr (buildSolver_step_mem hfix hg hp)))

/-- Step bundle: a converged step stays in-region, stays safe, and never loses. -/
theorem convergedSolver_step_bundle {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    solverStep cfg convergedSolver p g ∈ convergedSet cfg ∧
    solverStep cfg convergedSolver p g ∈ safe cfg ∧
    ¬ (solverStep cfg convergedSolver p g).lost cfg :=
  ⟨convergedSolver_solverStep_mem hg hp,
   convergedSolver_solverStep_safe hg hp,
   convergedSolver_solverStep_not_lost hg hp⟩

/-- The converged solver is deterministic and total: it answers every query with one placement. -/
theorem convergedSolver_total (g : GameState) (p : Piece) :
    ∃! pl : Placement, convergedSolver (cfg := cfg) g p = pl :=
  ⟨convergedSolver (cfg := cfg) g p, rfl, fun _ h => h.symm⟩

/-- A converged move recovers at most `cols·4` cells against the +4 added. -/
theorem convergedSolver_move_recovery_bounded (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    b.count + 4 ≤ ((convergedSolver (cfg := cfg) g p).applyStep cfg b).count + cfg.cols * 4 :=
  buildSolver_move_recovery_bounded hcols convergedSet_fixed hp hWF hnf

/-- Energy split holds after a converged move: debt + count = surface area. -/
theorem convergedSolver_move_energy_split (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) :
    HoleDebt.debt cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b)
        + ((convergedSolver (cfg := cfg) g p).applyStep cfg b).count
      = HoleDebt.surfaceArea cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b) :=
  buildSolver_move_energy_split hcols convergedSet_fixed hp hWF

/-- A non-clearing converged move adds exactly four cells. -/
theorem convergedSolver_move_count_no_clear (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b)
    (h0 : Board.linesCleared cfg ((convergedSolver (cfg := cfg) g p).place b) = 0) :
    ((convergedSolver (cfg := cfg) g p).applyStep cfg b).count = b.count + 4 :=
  buildSolver_move_count_no_clear hcols convergedSet_fixed hp hWF h0

/-- A converged move clears at most four lines from a non-full board. -/
theorem convergedSolver_move_clears_le_four (g : GameState) (p : Piece) {b : Board}
    (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    Board.linesCleared cfg ((convergedSolver (cfg := cfg) g p).place b) ≤ 4 :=
  buildSolver_move_clears_le_four convergedSet_fixed g p hnf

/-- The converged solver's placement skyline is hole-independent. -/
theorem convergedSolver_move_skyline_hole_independent (g : GameState) (p : Piece) {b β : Board}
    (h : ∀ j, b.colHeight j = β.colHeight j) (j : ℕ) :
    ((convergedSolver (cfg := cfg) g p).place b).colHeight j
      = ((convergedSolver (cfg := cfg) g p).place β).colHeight j :=
  buildSolver_move_skyline_hole_independent convergedSet_fixed g p h j

/-- The converged solver's placement never removes holes. -/
theorem convergedSolver_place_never_removes_holes (g : GameState) (p : Piece) (b : Board) :
    (HoleyCarrier.holes cfg b).card
      ≤ (HoleyCarrier.holes cfg ((convergedSolver (cfg := cfg) g p).place b)).card :=
  buildSolver_place_never_removes_holes convergedSet_fixed g p b

/-- The construction is stable beyond `|inFieldStates|` rounds: any extra round adds nothing. -/
theorem convergedSet_eq_iterate_card_add (k : ℕ) :
    safeIterFinite cfg (inFieldStates cfg) ((inFieldStates cfg).card + k) = convergedSet cfg :=
  safeIterFinite_stable cfg (inFieldStates cfg)
    (safeIterFinite_inFieldStates_stable_at_card cfg) k

/-- The converged region is self-sustaining: each member is alive with an in-region response. -/
theorem convergedSet_self_sustaining :
    ∀ g ∈ convergedSet cfg, ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ convergedSet cfg :=
  (round_self_sustaining_iff (convergedSet cfg)).mp convergedSet_fixed

/-- Each converged state has, for every bag piece, an in-region valid response. -/
theorem convergedSet_member_response {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    ∃ pl ∈ Placement.allValidFor cfg p, adversarialStep cfg g p pl ∈ convergedSet cfg :=
  (convergedSet_self_sustaining g hg).2 p hp

/-- Move-effect portrait of the converged solver: count +≤4, height +≤4, place +4, WF kept. -/
theorem convergedSolver_move_effect_portrait (hcols : 4 ≤ cfg.cols) {g : GameState} {p : Piece}
    (hp : p ∈ g.bag) {b : Board} (hWF : Board.WF cfg b) :
    (((convergedSolver (cfg := cfg) g p).applyStep cfg b).count ≤ b.count + 4) ∧
    (Board.maxHeight cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b)
        ≤ Board.maxHeight cfg b + 4) ∧
    (((convergedSolver (cfg := cfg) g p).place b).count = b.count + 4) ∧
    Board.WF cfg ((convergedSolver (cfg := cfg) g p).applyStep cfg b) :=
  buildSolver_move_effect_portrait hcols convergedSet_fixed hp hWF

/-- Input portrait: along legal play the converged solver is fed nonempty in-bag pieces. -/
theorem convergedSolver_input_portrait (hcols : 4 ≤ cfg.cols) (s : ℕ → Piece)
    (hl : LegalSequence s) :
    (∀ n, (adversarialTrace cfg convergedSolver s GameState.init n).bag.Nonempty) ∧
    (∀ n, s n ∈ (adversarialTrace cfg convergedSolver s GameState.init n).bag) ∧
    (∀ n, convergedSolver (cfg := cfg)
        (adversarialTrace cfg convergedSolver s GameState.init n) (s n)
      ∈ Placement.allValidFor cfg (s n)) :=
  buildSolver_input_portrait hcols convergedSet_fixed s hl

/-- Collapse portrait: a converged piece-slice fits the menu, ≤ cols·4, and collides past it. -/
theorem convergedSolver_collapse_portrait (hcols : 4 ≤ cfg.cols) (p : Piece)
    (T : Finset GameState) (hT : ∀ g ∈ T, p ∈ g.bag) :
    (T.image (fun g => convergedSolver (cfg := cfg) g p) ⊆ Placement.allValidFor cfg p) ∧
    ((T.image (fun g => convergedSolver (cfg := cfg) g p)).card ≤ cfg.cols * 4) ∧
    ((Placement.allValidFor cfg p).card < T.card →
      ∃ g₁ ∈ T, ∃ g₂ ∈ T, g₁ ≠ g₂ ∧
        convergedSolver (cfg := cfg) g₁ p = convergedSolver (cfg := cfg) g₂ p) :=
  buildSolver_collapse_portrait hcols convergedSet_fixed p T hT

/-- The converged construction realizes the milestone hierarchy: M1 (never tops out),
M3 (init in a closed invariant region), M4 (a closed atlas). -/
theorem convergedSolver_milestones (hinit : GameState.init ∈ convergedSet cfg) :
    (∀ (s : ℕ → Piece) (n : ℕ), LegalSequence s →
      ¬ (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg) ∧
    (GameState.init ∈ convergedSet cfg ∧
      ∀ g, solverReachable (convergedSolver (cfg := cfg)) g → g ∈ convergedSet cfg) ∧
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) :=
  ⟨fun _ n hl => convergedSolver_survives hinit hl n,
   ⟨hinit, fun _ hr => convergedSolver_confined hinit hr⟩,
   convergedSolver_atlas_closed cfg⟩

/-- Membership in the converged region is decidable — the existence test is effective. -/
noncomputable def convergedSet_decidableMem (g : GameState) : Decidable (g ∈ convergedSet cfg) :=
  inferInstance

/-- Function portrait of the converged solver: per-state menu section, injective, board+bag read. -/
theorem convergedSolver_function_portrait (hcols : 4 ≤ cfg.cols) (g : GameState) :
    (∀ p ∈ g.bag, convergedSolver (cfg := cfg) g p ∈ Placement.allValidFor cfg p) ∧
    Set.InjOn (fun p => convergedSolver (cfg := cfg) g p) g.bag ∧
    (∀ g₂ p, g.board = g₂.board → g.bag = g₂.bag →
      convergedSolver (cfg := cfg) g p = convergedSolver (cfg := cfg) g₂ p) :=
  ⟨(convergedSolver_response_portrait hcols g).1,
   (convergedSolver_response_portrait hcols g).2.2,
   fun g₂ p hb hbag => convergedSolver_reads_board_bag g g₂ p hb hbag⟩

/-- The converged atlas is a complete regional lookup table: closed, total on the region, with
init inside — the M4 proof artifact realized for the converged region. -/
theorem convergedSolver_complete_lookup_table (hinit : GameState.init ∈ convergedSet cfg) :
    (convergedSolver (cfg := cfg)).toAtlas.IsClosedOn cfg (convergedSet cfg) ∧
    (∀ g ∈ convergedSet cfg, ∀ p ∈ g.bag,
      ((convergedSolver (cfg := cfg)).toAtlas g p).isSome = true) ∧
    GameState.init ∈ convergedSet cfg :=
  ⟨convergedSolver_atlas_closed cfg,
   fun g _ p _ => buildSolver_toAtlas_isSome convergedSet_fixed g p, hinit⟩

/-- Play forever by lookup: at every step the atlas has a move and the game stays alive. -/
theorem convergedSolver_play_by_lookup (hinit : GameState.init ∈ convergedSet cfg)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    ((convergedSolver (cfg := cfg)).toAtlas
       (adversarialTrace cfg convergedSolver s GameState.init n) (s n)).isSome = true ∧
    ¬ (adversarialTrace cfg convergedSolver s GameState.init n).lost cfg :=
  ⟨buildSolver_toAtlas_isSome convergedSet_fixed _ _,
   convergedSolver_survives hinit hl n⟩

/-- The converged region is a closed subgraph: every chosen edge stays inside it. -/
theorem convergedSet_closed_subgraph {g : GameState} (hg : g ∈ convergedSet cfg) {p : Piece}
    (hp : p ∈ g.bag) :
    adversarialStep cfg g p (convergedSolver (cfg := cfg) g p) ∈ convergedSet cfg :=
  buildSolver_step_mem convergedSet_fixed hg hp

end Tetris
