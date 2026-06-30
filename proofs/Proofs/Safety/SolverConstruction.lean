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

end Tetris
