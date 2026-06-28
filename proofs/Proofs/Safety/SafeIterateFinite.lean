import Mathlib
import Proofs.Safety.SafeSet
import Proofs.Invariants.StateSpace
import Proofs.Invariants.Gameplay

/-!
# Computable safety operator on `Finset GameState`

`F_finite cfg S` is the computable analog of `safeOp cfg` restricted to a
candidate finite set `S`: a state `g ∈ S` survives the iteration iff it is not
already lost and, for every piece in its bag, some valid placement leads back
into `S`.

This is the core of the algorithmic GFP search underlying Route 2: starting
from the in-field universe and iterating `F_finite` shrinks the candidate set
down toward the (finite, decidable) restriction of `safe` to in-field states.
-/

namespace Tetris

/-- One filter step of the algorithmic GFP search. A state `g ∈ S` is kept
exactly when it is not lost and every legal next piece admits at least one
valid placement landing back in `S`. -/
def F_finite (cfg : GameConfig) (S : Finset GameState) : Finset GameState :=
  S.filter (fun g => ¬ g.lost cfg ∧
    ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ S)

/-- `F_finite` only ever removes states — the iteration is monotonically
decreasing in the subset order. (Immediate from `Finset.filter_subset`.) -/
theorem F_finite_subset (cfg : GameConfig) (S : Finset GameState) :
    F_finite cfg S ⊆ S := Finset.filter_subset _ _

/-- `F_finite`'s membership is `S`-membership ∧ not-lost ∧ step-condition. -/
theorem mem_F_finite_iff (cfg : GameConfig) (S : Finset GameState) (g : GameState) :
    g ∈ F_finite cfg S ↔ g ∈ S ∧ ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ S := by
  rw [F_finite, Finset.mem_filter]

/-- `mem_F_finite_iff` accessor: a member of `F_finite cfg S` is not lost. -/
theorem not_lost_of_mem_F_finite {cfg : GameConfig} {S : Finset GameState}
    {g : GameState} (h : g ∈ F_finite cfg S) : ¬ g.lost cfg :=
  ((mem_F_finite_iff cfg S g).mp h).2.1

/-- `mem_F_finite_iff` accessor: a member of `F_finite cfg S` is in `S`. -/
theorem mem_of_mem_F_finite {cfg : GameConfig} {S : Finset GameState}
    {g : GameState} (h : g ∈ F_finite cfg S) : g ∈ S :=
  ((mem_F_finite_iff cfg S g).mp h).1

/-- `mem_F_finite_iff` accessor: a member of `F_finite cfg S` has a valid
step for every piece in its bag landing in `S`. -/
theorem step_exists_of_mem_F_finite {cfg : GameConfig} {S : Finset GameState}
    {g : GameState} (h : g ∈ F_finite cfg S) :
    ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ S :=
  ((mem_F_finite_iff cfg S g).mp h).2.2

/-- Constructor: from the three obligations, get `g ∈ F_finite cfg S`. -/
theorem mem_F_finite_of {cfg : GameConfig} {S : Finset GameState} {g : GameState}
    (h1 : g ∈ S) (h2 : ¬ g.lost cfg)
    (h3 : ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
            adversarialStep cfg g p pl ∈ S) :
    g ∈ F_finite cfg S :=
  (mem_F_finite_iff cfg S g).mpr ⟨h1, h2, h3⟩

/-- `F_finite`'s output card is at most the input's card. -/
theorem F_finite_card_le (cfg : GameConfig) (S : Finset GameState) :
    (F_finite cfg S).card ≤ S.card :=
  Finset.card_le_card (F_finite_subset cfg S)

/-- `F_finite cfg S = S` iff every element of `S` is in `F_finite cfg S`
(i.e., satisfies the not-lost-and-step-condition obligations w.r.t. `S`). -/
theorem F_finite_eq_self_iff (cfg : GameConfig) (S : Finset GameState) :
    F_finite cfg S = S ↔ ∀ g ∈ S, g ∈ F_finite cfg S :=
  ⟨fun h g hg => by rw [h]; exact hg,
   fun h => Finset.Subset.antisymm (F_finite_subset cfg S) h⟩

/-- Constructor: from the obligations on every `g ∈ S`, `F_finite cfg S = S`. -/
theorem F_finite_eq_self_of {cfg : GameConfig} {S : Finset GameState}
    (h : ∀ g ∈ S, ¬ g.lost cfg ∧
      ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
        adversarialStep cfg g p pl ∈ S) :
    F_finite cfg S = S :=
  (F_finite_eq_self_iff cfg S).mpr
    (fun g hg => mem_F_finite_of hg (h g hg).1 (h g hg).2)

/-- **`AdversarialClosedCycle` ⇒ `F_finite` fixed point**: an adversarial
closed cycle's `states` is a fixed point of `F_finite cfg`. -/
theorem AdversarialClosedCycle.F_finite_states_eq_states {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) :
    F_finite cfg C.states = C.states := by
  apply F_finite_eq_self_of
  intro g hg
  refine ⟨C.not_lost g hg, ?_⟩
  intro p hp
  refine ⟨C.solver g p, ?_, C.closed g hg p hp⟩
  exact (Placement.mem_allValidFor _ _ _).mpr (C.valid g hg p hp)


/-- `F_finite` is monotone in its candidate set: enlarging the candidate set
weakens the "lands in S" obligation, so more states survive the filter. -/
theorem F_finite_mono (cfg : GameConfig) {S T : Finset GameState} (h : S ⊆ T) :
    F_finite cfg S ⊆ F_finite cfg T := by
  intro g hg
  rw [F_finite, Finset.mem_filter] at hg
  rw [F_finite, Finset.mem_filter]
  refine ⟨h hg.1, hg.2.1, ?_⟩
  intro p hp
  obtain ⟨pl, hpl, hstep⟩ := hg.2.2 p hp
  exact ⟨pl, hpl, h hstep⟩

/-- Computable iteration of `F_finite` from a starting candidate set `S₀`.
The sequence is **decreasing** (each step is a filter); on a finite universe
it eventually stabilises at a fixed point of `F_finite`. -/
def safeIterFinite (cfg : GameConfig) (S₀ : Finset GameState) :
    ℕ → Finset GameState
  | 0     => S₀
  | n + 1 => F_finite cfg (safeIterFinite cfg S₀ n)

@[simp] theorem safeIterFinite_zero (cfg : GameConfig) (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ 0 = S₀ := rfl

theorem safeIterFinite_succ (cfg : GameConfig) (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) = F_finite cfg (safeIterFinite cfg S₀ n) := rfl

/-- The iteration is **monotone-decreasing** by one step: each iterate is a
subset of the previous one. -/
theorem safeIterFinite_succ_subset (cfg : GameConfig) (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) ⊆ safeIterFinite cfg S₀ n := by
  rw [safeIterFinite_succ]; exact F_finite_subset cfg _

/-- The iteration is **antitone**: a later iterate is contained in any earlier
iterate. Iterating `F_finite` only ever shrinks the candidate set. -/
theorem safeIterFinite_antitone (cfg : GameConfig) (S₀ : Finset GameState)
    {m n : ℕ} (h : m ≤ n) :
    safeIterFinite cfg S₀ n ⊆ safeIterFinite cfg S₀ m := by
  induction h with
  | refl => exact subset_refl _
  | step _ ih => exact subset_trans (safeIterFinite_succ_subset cfg S₀ _) ih

/-- Cardinality is non-increasing along the chain. -/
theorem safeIterFinite_card_succ_le (cfg : GameConfig) (S₀ : Finset GameState) (n : ℕ) :
    (safeIterFinite cfg S₀ (n + 1)).card ≤ (safeIterFinite cfg S₀ n).card :=
  Finset.card_le_card (safeIterFinite_succ_subset cfg S₀ n)

/-- **Stabilisation**: once one step is a no-op, all subsequent steps are too.
A fixed point of the iteration persists forever. -/
theorem safeIterFinite_stable (cfg : GameConfig) (S₀ : Finset GameState)
    {n : ℕ} (h : safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n) (k : ℕ) :
    safeIterFinite cfg S₀ (n + k) = safeIterFinite cfg S₀ n := by
  induction k with
  | zero => rfl
  | succ j ih =>
      change safeIterFinite cfg S₀ ((n + j) + 1) = safeIterFinite cfg S₀ n
      rw [safeIterFinite_succ, ih, ← safeIterFinite_succ]
      exact h

/-- `≥ n` form of `safeIterFinite_stable`. -/
theorem safeIterFinite_eq_of_stable_of_le (cfg : GameConfig)
    (S₀ : Finset GameState) {n : ℕ}
    (h : safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n)
    {k : ℕ} (hk : n ≤ k) :
    safeIterFinite cfg S₀ k = safeIterFinite cfg S₀ n := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hk
  exact safeIterFinite_stable cfg S₀ h j

/-- **Cycle persists in iterates**: if `C.states ⊆ S₀`, then
`C.states ⊆ safeIterFinite cfg S₀ n` for every `n`. The cycle is a
fixed point under `F_finite`, so it survives every iteration. -/
theorem AdversarialClosedCycle.states_subset_safeIterFinite {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {S₀ : Finset GameState}
    (h : C.states ⊆ S₀) (n : ℕ) :
    C.states ⊆ safeIterFinite cfg S₀ n := by
  induction n with
  | zero => exact h
  | succ k ih =>
    rw [safeIterFinite_succ, ← C.F_finite_states_eq_states]
    exact F_finite_mono cfg ih

/-- **Empty iterate ⇒ empty cycle**: if some iterate is empty and
`C.states ⊆ S₀`, then `C.states = ∅`. Direct contrapositive of
`states_subset_safeIterFinite`. -/
theorem AdversarialClosedCycle.states_eq_empty_of_safeIterFinite_eq_empty
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {S₀ : Finset GameState} (hS : C.states ⊆ S₀)
    {n : ℕ} (hn : safeIterFinite cfg S₀ n = ∅) :
    C.states = ∅ :=
  Finset.subset_empty.mp (hn ▸ C.states_subset_safeIterFinite hS n)

/-- **Empty iterate ⇒ no init-containing cycle (subset form)**: if
`safeIterFinite cfg S₀ n = ∅` and `C.states ⊆ S₀`, then `init ∉
C.states`. Direct consequence of Tick 1415. -/
theorem AdversarialClosedCycle.init_not_mem_states_of_safeIterFinite_eq_empty
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {S₀ : Finset GameState} (hS : C.states ⊆ S₀)
    {n : ℕ} (hn : safeIterFinite cfg S₀ n = ∅) :
    GameState.init ∉ C.states := by
  rw [C.states_eq_empty_of_safeIterFinite_eq_empty hS hn]
  exact Finset.notMem_empty _

/-- **Top-level negative result**: if `safeIterFinite cfg S₀ n = ∅`, then
no adversarial closed cycle (with `states ⊆ S₀`) contains `init`. -/
theorem not_exists_init_adversarialClosedCycle_subset_of_safeIterFinite_eq_empty
    {cfg : GameConfig} {S₀ : Finset GameState} {n : ℕ}
    (hn : safeIterFinite cfg S₀ n = ∅) :
    ¬ ∃ C : AdversarialClosedCycle cfg,
        C.states ⊆ S₀ ∧ GameState.init ∈ C.states := by
  rintro ⟨C, hS, hinit⟩
  exact C.init_not_mem_states_of_safeIterFinite_eq_empty hS hn hinit

/-- Pointwise version: no `AdversarialClosedCycle` over `cfg` has both
`states ⊆ S₀` and `init ∈ states` when the iterate evacuates. -/
theorem forall_not_init_and_subset_of_safeIterFinite_eq_empty
    {cfg : GameConfig} {S₀ : Finset GameState} {n : ℕ}
    (hn : safeIterFinite cfg S₀ n = ∅) :
    ∀ C : AdversarialClosedCycle cfg,
      ¬ (C.states ⊆ S₀ ∧ GameState.init ∈ C.states) :=
  fun C ⟨hS, hinit⟩ =>
    C.init_not_mem_states_of_safeIterFinite_eq_empty hS hn hinit

/-- Symmetric form: any cycle `C` with empty iterate-seeded universe
satisfies `init ∈ C.states → ¬ C.states ⊆ S₀`. -/
theorem not_subset_of_init_mem_states_of_safeIterFinite_eq_empty
    {cfg : GameConfig} {S₀ : Finset GameState} {n : ℕ}
    (hn : safeIterFinite cfg S₀ n = ∅)
    (C : AdversarialClosedCycle cfg)
    (hinit : GameState.init ∈ C.states) :
    ¬ C.states ⊆ S₀ :=
  fun hS => C.init_not_mem_states_of_safeIterFinite_eq_empty hS hn hinit

/-- Membership in `safeIterFinite cfg S₀ (n+1)` unfolds via `mem_F_finite_iff`
applied to the previous iterate. -/
theorem mem_safeIterFinite_succ_iff (cfg : GameConfig) (S₀ : Finset GameState)
    (n : ℕ) (g : GameState) :
    g ∈ safeIterFinite cfg S₀ (n + 1) ↔
      g ∈ safeIterFinite cfg S₀ n ∧ ¬ g.lost cfg ∧
        ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
          adversarialStep cfg g p pl ∈ safeIterFinite cfg S₀ n := by
  rw [safeIterFinite_succ]
  exact mem_F_finite_iff cfg _ g

/-- A state in `safeIterFinite cfg S₀ (n+1)` is not lost. -/
theorem not_lost_of_mem_safeIterFinite_succ {cfg : GameConfig}
    {S₀ : Finset GameState} {n : ℕ} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ (n + 1)) : ¬ g.lost cfg :=
  ((mem_safeIterFinite_succ_iff cfg S₀ n g).mp h).2.1

/-- A state in `safeIterFinite cfg S₀ (n+1)` is in the prior iterate. -/
theorem mem_of_mem_safeIterFinite_succ {cfg : GameConfig}
    {S₀ : Finset GameState} {n : ℕ} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ (n + 1)) :
    g ∈ safeIterFinite cfg S₀ n :=
  ((mem_safeIterFinite_succ_iff cfg S₀ n g).mp h).1

/-- A state in `safeIterFinite cfg S₀ (n+1)` has a valid step for every
piece in its bag landing in the prior iterate. -/
theorem step_exists_of_mem_safeIterFinite_succ {cfg : GameConfig}
    {S₀ : Finset GameState} {n : ℕ} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ (n + 1)) :
    ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
      adversarialStep cfg g p pl ∈ safeIterFinite cfg S₀ n :=
  ((mem_safeIterFinite_succ_iff cfg S₀ n g).mp h).2.2

/-- Constructor: from the three obligations, `g ∈ safeIterFinite cfg S₀
(n+1)`. -/
theorem mem_safeIterFinite_succ_of {cfg : GameConfig}
    {S₀ : Finset GameState} {n : ℕ} {g : GameState}
    (h1 : g ∈ safeIterFinite cfg S₀ n)
    (h2 : ¬ g.lost cfg)
    (h3 : ∀ p ∈ g.bag, ∃ pl ∈ Placement.allValidFor cfg p,
            adversarialStep cfg g p pl ∈ safeIterFinite cfg S₀ n) :
    g ∈ safeIterFinite cfg S₀ (n + 1) :=
  (mem_safeIterFinite_succ_iff cfg S₀ n g).mpr ⟨h1, h2, h3⟩

/-- A cycle state lies in every iterate seeded from a containing universe.
Pointwise form of `AdversarialClosedCycle.states_subset_safeIterFinite`. -/
theorem AdversarialClosedCycle.mem_safeIterFinite {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {S₀ : Finset GameState}
    (h : C.states ⊆ S₀) {g : GameState} (hg : g ∈ C.states) (n : ℕ) :
    g ∈ safeIterFinite cfg S₀ n :=
  C.states_subset_safeIterFinite h n hg

/-- If `init ∈ C.states` and `C.states ⊆ S₀`, then `init ∈
safeIterFinite cfg S₀ n` for every `n`. -/
theorem AdversarialClosedCycle.init_mem_safeIterFinite {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg) {S₀ : Finset GameState}
    (hS : C.states ⊆ S₀) (hinit : GameState.init ∈ C.states) (n : ℕ) :
    GameState.init ∈ safeIterFinite cfg S₀ n :=
  C.mem_safeIterFinite hS hinit n

/-- Top-level form of Tick 1425: existence of an init-containing cycle with
`C.states ⊆ S₀` implies `init ∈ safeIterFinite cfg S₀ n` for every `n`. -/
theorem init_mem_safeIterFinite_of_exists_init_adversarialClosedCycle_subset
    {cfg : GameConfig} {S₀ : Finset GameState}
    (h : ∃ C : AdversarialClosedCycle cfg,
        C.states ⊆ S₀ ∧ GameState.init ∈ C.states)
    (n : ℕ) :
    GameState.init ∈ safeIterFinite cfg S₀ n := by
  obtain ⟨C, hS, hinit⟩ := h
  exact C.init_mem_safeIterFinite hS hinit n

/-- `safeIterFinite cfg S₀ (n+1) = safeIterFinite cfg S₀ n` iff
the iterate is a fixed point of `F_finite`. -/
theorem safeIterFinite_succ_eq_iff_F_finite_fixed (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n ↔
      F_finite cfg (safeIterFinite cfg S₀ n) = safeIterFinite cfg S₀ n := by
  rw [safeIterFinite_succ]

/-- **Strict-decrease in cardinality**: if the iterate has not stabilised at
`n`, then `card` strictly drops at the next step. -/
theorem safeIterFinite_card_lt_of_succ_ne (cfg : GameConfig)
    (S₀ : Finset GameState) {n : ℕ}
    (h : safeIterFinite cfg S₀ (n + 1) ≠ safeIterFinite cfg S₀ n) :
    (safeIterFinite cfg S₀ (n + 1)).card < (safeIterFinite cfg S₀ n).card :=
  Finset.card_lt_card
    (Finset.ssubset_iff_subset_ne.mpr ⟨safeIterFinite_succ_subset cfg S₀ n, h⟩)

/-- Strict-subset form of Tick 1428: if not stabilised, next iterate is a
strict subset. -/
theorem safeIterFinite_succ_ssubset_of_ne (cfg : GameConfig)
    (S₀ : Finset GameState) {n : ℕ}
    (h : safeIterFinite cfg S₀ (n + 1) ≠ safeIterFinite cfg S₀ n) :
    safeIterFinite cfg S₀ (n + 1) ⊂ safeIterFinite cfg S₀ n :=
  Finset.ssubset_iff_subset_ne.mpr ⟨safeIterFinite_succ_subset cfg S₀ n, h⟩

/-- `safeIterFinite cfg S₀ (n+1) ⊂ safeIterFinite cfg S₀ n` iff the
iterate has not stabilised at `n`. -/
theorem safeIterFinite_succ_ssubset_iff_ne (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) ⊂ safeIterFinite cfg S₀ n ↔
      safeIterFinite cfg S₀ (n + 1) ≠ safeIterFinite cfg S₀ n :=
  ⟨fun h => h.ne, safeIterFinite_succ_ssubset_of_ne cfg S₀⟩

/-- Stabilisation by cardinality: `safeIterFinite cfg S₀ (n+1) =
safeIterFinite cfg S₀ n` iff card unchanged. (Forward direction is
`congrArg`; reverse uses card-equality on a subset to derive set
equality.) -/
theorem safeIterFinite_succ_eq_iff_card_eq (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n ↔
      (safeIterFinite cfg S₀ (n + 1)).card = (safeIterFinite cfg S₀ n).card := by
  refine ⟨fun h => by rw [h], fun h => ?_⟩
  exact Finset.eq_of_subset_of_card_le (safeIterFinite_succ_subset cfg S₀ n)
    h.ge

/-- `safeIterFinite cfg S₀ n = ∅` iff card is zero. -/
theorem safeIterFinite_eq_empty_iff_card_eq_zero (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ n = ∅ ↔ (safeIterFinite cfg S₀ n).card = 0 :=
  Finset.card_eq_zero.symm

/-- Card monotonicity in `n`: `m ≤ n ⇒ card (iterate n) ≤ card (iterate m)`. -/
theorem safeIterFinite_card_le_of_le (cfg : GameConfig)
    (S₀ : Finset GameState) {m n : ℕ} (h : m ≤ n) :
    (safeIterFinite cfg S₀ n).card ≤ (safeIterFinite cfg S₀ m).card :=
  Finset.card_le_card (safeIterFinite_antitone cfg S₀ h)

/-- Every iterate's card is bounded by `S₀.card`. -/
theorem safeIterFinite_card_le_S₀_card (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    (safeIterFinite cfg S₀ n).card ≤ S₀.card :=
  safeIterFinite_card_le_of_le cfg S₀ (Nat.zero_le n)

/-- Every iterate is a subset of `S₀`. -/
theorem safeIterFinite_subset_S₀ (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    safeIterFinite cfg S₀ n ⊆ S₀ :=
  safeIterFinite_antitone cfg S₀ (Nat.zero_le n)

/-- Pointwise form of `safeIterFinite_subset_S₀`. -/
theorem mem_S₀_of_mem_safeIterFinite {cfg : GameConfig}
    {S₀ : Finset GameState} {n : ℕ} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ n) : g ∈ S₀ :=
  safeIterFinite_subset_S₀ cfg S₀ n h

/-- If `S₀ = ∅`, every iterate is empty. -/
theorem safeIterFinite_empty_eq_empty (cfg : GameConfig) (n : ℕ) :
    safeIterFinite cfg ∅ n = ∅ :=
  Finset.subset_empty.mp (safeIterFinite_subset_S₀ cfg ∅ n)

/-- `safeIterFinite cfg ∅ n` has card 0. -/
@[simp] theorem safeIterFinite_empty_card (cfg : GameConfig) (n : ℕ) :
    (safeIterFinite cfg ∅ n).card = 0 := by
  rw [safeIterFinite_empty_eq_empty]
  rfl

/-- Monotonicity in seed: `S₀ ⊆ T₀ ⇒ safeIterFinite cfg S₀ n ⊆ safeIterFinite
cfg T₀ n` for every `n`. -/
theorem safeIterFinite_mono_seed (cfg : GameConfig)
    {S₀ T₀ : Finset GameState} (h : S₀ ⊆ T₀) (n : ℕ) :
    safeIterFinite cfg S₀ n ⊆ safeIterFinite cfg T₀ n := by
  induction n with
  | zero => exact h
  | succ k ih =>
    rw [safeIterFinite_succ, safeIterFinite_succ]
    exact F_finite_mono cfg ih

/-- `safeIterFinite cfg S₀ 1 = F_finite cfg S₀` (`@[simp]` companion to
`safeIterFinite_zero`). -/
@[simp] theorem safeIterFinite_one (cfg : GameConfig) (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ 1 = F_finite cfg S₀ := rfl

/-- `safeIterFinite cfg S₀ 2 = F_finite cfg (F_finite cfg S₀)`. -/
@[simp] theorem safeIterFinite_two (cfg : GameConfig) (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ 2 = F_finite cfg (F_finite cfg S₀) := rfl

/-- One iteration step **either strictly shrinks the cardinality, or fixes the
set**. The dichotomy underlying the pigeonhole convergence argument. -/
theorem safeIterFinite_strict_or_stable (cfg : GameConfig) (S₀ : Finset GameState) (n : ℕ) :
    (safeIterFinite cfg S₀ (n + 1)).card < (safeIterFinite cfg S₀ n).card ∨
    safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n := by
  by_cases h : safeIterFinite cfg S₀ (n + 1) = safeIterFinite cfg S₀ n
  · exact Or.inr h
  · left
    exact Finset.card_lt_card
      (Finset.ssubset_iff_subset_ne.mpr ⟨safeIterFinite_succ_subset cfg S₀ n, h⟩)

/-- Pigeonhole bookkeeping: after `N` iterations, either some earlier step has
already stabilised, or the cardinality has dropped by at least `N` from
`S₀.card`. -/
theorem safeIterFinite_card_decrease (cfg : GameConfig) (S₀ : Finset GameState) (N : ℕ) :
    (∃ M ≤ N, safeIterFinite cfg S₀ (M + 1) = safeIterFinite cfg S₀ M) ∨
    (safeIterFinite cfg S₀ N).card + N ≤ S₀.card := by
  induction N with
  | zero =>
      right
      simp
  | succ k ih =>
      rcases ih with ⟨M, hM, hMeq⟩ | hbound
      · exact Or.inl ⟨M, Nat.le_succ_of_le hM, hMeq⟩
      · rcases safeIterFinite_strict_or_stable cfg S₀ k with hstrict | hstable
        · right; omega
        · exact Or.inl ⟨k, Nat.le_succ k, hstable⟩

/-- `F_finite` of the empty set is empty (trivial filter). -/
@[simp] theorem F_finite_empty (cfg : GameConfig) : F_finite cfg ∅ = ∅ :=
  Finset.filter_empty _

/-- **Convergence theorem.** Starting from any candidate set `S₀`, iterating
`F_finite` reaches a fixed point within `S₀.card` steps. This makes safe-set
membership decidable in principle on a finite state space — the algorithmic
backbone of Route 2. -/
theorem safeIterFinite_converges (cfg : GameConfig) (S₀ : Finset GameState) :
    ∃ N, N ≤ S₀.card ∧ safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N := by
  rcases safeIterFinite_card_decrease cfg S₀ S₀.card with ⟨M, hM, hMeq⟩ | hbound
  · exact ⟨M, hM, hMeq⟩
  · refine ⟨S₀.card, le_refl _, ?_⟩
    have hcard : (safeIterFinite cfg S₀ S₀.card).card = 0 := by omega
    have hempty : safeIterFinite cfg S₀ S₀.card = ∅ := Finset.card_eq_zero.mp hcard
    rw [safeIterFinite_succ, hempty, F_finite_empty]

/-- **Soundness**: a fixed point of the finite iteration is contained in the
abstract `safe` set. Combined with `safeIterFinite_converges`, this bridges the
algorithmic search to the formal `safe`/`TetrisSolvable` framework. -/
theorem safeIterFinite_subset_safe (cfg : GameConfig) (S₀ : Finset GameState) (N : ℕ)
    (h : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (↑(safeIterFinite cfg S₀ N) : Set GameState) ⊆ safe cfg := by
  apply safe_greatest
  intro g hg
  rw [Finset.mem_coe] at hg
  have hg' : g ∈ F_finite cfg (safeIterFinite cfg S₀ N) := by
    rw [← safeIterFinite_succ, h]; exact hg
  rw [F_finite, Finset.mem_filter] at hg'
  obtain ⟨_, hnlost, hmoves⟩ := hg'
  refine ⟨hnlost, ?_⟩
  intro p hp
  obtain ⟨pl, hpl_mem, hstep⟩ := hmoves p hp
  rw [Placement.mem_allValidFor] at hpl_mem
  exact ⟨pl, hpl_mem.1, hpl_mem.2, Finset.mem_coe.mpr hstep⟩

/-- Convenience corollary: if `init` is in a fixed point of the finite
iteration, then `init` is in the abstract `safe` set. Ready to feed into
`safe_extract` for `TetrisSolvable`. -/
theorem init_safe_of_mem_fixedPoint (cfg : GameConfig) (S₀ : Finset GameState) (N : ℕ)
    (h : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (hi : GameState.init ∈ safeIterFinite cfg S₀ N) :
    GameState.init ∈ safe cfg :=
  safeIterFinite_subset_safe cfg S₀ N h (Finset.mem_coe.mpr hi)

/-- **The full algorithmic → solvability bridge.** A successful finite search
(finding `N`, a fixed-point witness, and `init` inside the resulting set)
proves `TetrisSolvable`. -/
theorem tetrisSolvable_of_init_mem_fixedPoint
    (S₀ : Finset GameState) (N : ℕ)
    (h : safeIterFinite GameConfig.standard S₀ (N + 1)
       = safeIterFinite GameConfig.standard S₀ N)
    (hi : GameState.init ∈ safeIterFinite GameConfig.standard S₀ N) :
    TetrisSolvable :=
  safe_extract (init_safe_of_mem_fixedPoint GameConfig.standard S₀ N h hi)

/-- **Completeness**: when the starting set `S₀` contains every safe state,
every iterate still contains every safe state. Combined with soundness
(`safeIterFinite_subset_safe`), this pins the finite-fixed-point set as a
two-sided characterisation of `safe ∩ ↑S₀`. -/
theorem safe_subset_safeIterFinite (cfg : GameConfig) (S₀ : Finset GameState)
    (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ) :
    safe cfg ⊆ ↑(safeIterFinite cfg S₀ n) := by
  induction n with
  | zero => exact hS₀
  | succ k ih =>
      intro g hg
      have hgk : g ∈ safeIterFinite cfg S₀ k := Finset.mem_coe.mp (ih hg)
      rw [Finset.mem_coe, safeIterFinite_succ, F_finite, Finset.mem_filter]
      refine ⟨hgk, safe_not_lost hg, ?_⟩
      intro p hp
      obtain ⟨pl, hpiece, hvalid, hstep⟩ := safe_step hg hp
      refine ⟨pl, ?_, Finset.mem_coe.mp (ih hstep)⟩
      rw [Placement.mem_allValidFor]
      exact ⟨hpiece, hvalid⟩

/-- **Two-sided characterisation**: at a fixed point of `F_finite` over a
candidate set `S₀` that contains every safe state, `safe`-set membership is
exactly membership in the iterate. This collapses the open `init ∈ safe`
question to a *decidable* finite-set membership check (given a suitable `S₀`). -/
theorem safe_iff_mem_fixedPoint (cfg : GameConfig) (S₀ : Finset GameState) (N : ℕ)
    (hS₀ : safe cfg ⊆ ↑S₀)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    g ∈ safe cfg ↔ g ∈ (↑(safeIterFinite cfg S₀ N) : Set GameState) :=
  ⟨fun h => safe_subset_safeIterFinite cfg S₀ hS₀ N h,
   fun h => safeIterFinite_subset_safe cfg S₀ N hfix h⟩

/-- **Decision procedure** for `safe`-set membership, given a witness universe
`S₀` containing every safe state and a fixed-point witness `(N, hfix)`. The
returned `Decidable` reduces the (otherwise noncomputable) `safe`-membership
to a finite-set membership lookup. -/
def decideSafeFromUniverse (cfg : GameConfig) (S₀ : Finset GameState) (N : ℕ)
    (hS₀ : safe cfg ⊆ ↑S₀)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) (g : GameState) :
    Decidable (g ∈ safe cfg) :=
  decidable_of_iff (g ∈ safeIterFinite cfg S₀ N)
    (((safe_iff_mem_fixedPoint cfg S₀ N hS₀ hfix g).trans Finset.mem_coe).symm)

/-- **Empty `safeIterFinite` implies empty `safe`**. If a finite iteration
ever reaches `∅`, the abstract safe set is empty too. Atlas: if we
computationally confirm `safeIterFinite cfg S₀ n = ∅` for some valid `S₀`
witness and `n`, we've proven there's no surviving strategy. -/
theorem safe_eq_empty_of_safeIterFinite_empty {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {n : ℕ}
    (hempty : safeIterFinite cfg S₀ n = ∅) : safe cfg = ∅ := by
  ext g
  refine ⟨fun hg => ?_, fun h => absurd h (Set.notMem_empty g)⟩
  have hin := safe_subset_safeIterFinite cfg S₀ hS₀ n hg
  rw [hempty] at hin
  exact absurd (Finset.mem_coe.mp hin) (Finset.notMem_empty g)

/-- **`init ∉ safeIterFinite n` ⇒ `init ∉ safe`**. If `init` ever falls out
of the iteration approximation, it's not safe. -/
theorem init_not_safe_of_init_not_mem_safeIterFinite {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (n : ℕ)
    (h : GameState.init ∉ safeIterFinite cfg S₀ n) :
    GameState.init ∉ safe cfg := fun hsafe =>
  h (Finset.mem_coe.mp (safe_subset_safeIterFinite cfg S₀ hS₀ n hsafe))

/-- **At a fixed point, `init` is safe iff it's in the iteration**. The
**positive bridge**: computationally, run `safeIterFinite` until a fixed
point, then check `init ∈` — this decides solvability. -/
theorem init_safe_iff_init_mem_safeIterFinite {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) (N : ℕ)
    (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    GameState.init ∈ safe cfg ↔ GameState.init ∈ safeIterFinite cfg S₀ N := by
  rw [← Finset.mem_coe]
  exact safe_iff_mem_fixedPoint cfg S₀ N hS₀ hfix GameState.init

/-- **THE MASTER DECIDABLE CHARACTERIZATION**: Tetris solvability ↔ init
in the safe-iter fixed point. Chains Tick 124 (safe ↔ solvable) with the
computational iter (Tick 187). When `safeIterFinite` reaches a fixed
point, solvability is *fully decidable* as a Finset membership check. -/
theorem tetrisSolvableValidFor_iff_init_mem_safeIterFinite {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀)
    (N : ℕ) (hfix : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    TetrisSolvableValidFor cfg ↔
      GameState.init ∈ safeIterFinite cfg S₀ N := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      init_safe_iff_init_mem_safeIterFinite hS₀ N hfix]

/-- **Solvability certificate**: a triple `(S₀, N, witnesses)` that proves
Tetris solvability via Tick 188. The structure bundles the universe
soundness, fixed-point convergence, and init's membership into a single
discharge target. -/
structure IsSolvabilityCertificate (cfg : GameConfig) (S₀ : Finset GameState)
    (N : ℕ) : Prop where
  /-- Universe soundness: `safe cfg ⊆ ↑S₀`. -/
  universe_sound : safe cfg ⊆ ↑S₀
  /-- Fixed-point convergence at step `N`. -/
  fixed_point : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N
  /-- `init` is in the (fixed-point) iteration set. -/
  init_in : GameState.init ∈ safeIterFinite cfg S₀ N

/-- A solvability certificate proves Tetris is solvable. The positive
construction route. -/
theorem solvable_of_certificate {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    {S₀ : Finset GameState} {N : ℕ}
    (h : IsSolvabilityCertificate cfg S₀ N) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_mem_safeIterFinite hcols h.universe_sound N
    h.fixed_point).mpr h.init_in

/-- **Unsolvability certificate**: a `(S₀, N)` witness that
`init ∉ safeIterFinite cfg S₀ N`. Doesn't need a fixed-point — any
iteration step where init is dropped suffices. -/
structure IsUnsolvabilityCertificate (cfg : GameConfig)
    (S₀ : Finset GameState) (N : ℕ) : Prop where
  universe_sound : safe cfg ⊆ ↑S₀
  init_out : GameState.init ∉ safeIterFinite cfg S₀ N

/-- An unsolvability certificate disproves Tetris solvability. The
negative computational route. -/
theorem not_solvable_of_unsolvability_certificate {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState} {N : ℕ}
    (h : IsUnsolvabilityCertificate cfg S₀ N) :
    ¬ TetrisSolvableValidFor cfg := by
  intro hsolv
  have hinit : GameState.init ∈ safe cfg :=
    (tetrisSolvableValidFor_iff_init_safe hcols).mp hsolv
  exact init_not_safe_of_init_not_mem_safeIterFinite h.universe_sound N
    h.init_out hinit

/-- **One-shot positive route**: from a `(states, solver)` candidate
satisfying the cycle obligations and `init ∈ states`, derive
`TetrisSolvableValidFor cfg` in one call. The Valid solver is
`safeSolver cfg`, not the cycle's `solver` — this avoids requiring the
cycle's solver to be Valid globally. -/
theorem solvable_valid_via_cycle {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    {states : Finset GameState} {solver : Solver cfg}
    (hobl : AdversarialClosedCycle.Obligations cfg states solver)
    (hinit : GameState.init ∈ states) :
    TetrisSolvableValidFor cfg := by
  have hsafe : GameState.init ∈ safe cfg :=
    (AdversarialClosedCycle.mkChecked states solver hobl).subset_safe
      (Finset.mem_coe.mpr hinit)
  exact ⟨safeSolver cfg,
         init_safe_implies_solvesTetrisValid hcols hsafe⟩

/-- **`IsInitCycle` ⇒ solvable (Valid)**: the bundled `IsInitCycle`
predicate directly discharges Valid solvability. -/
theorem solvable_valid_of_isInitCycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {states : Finset GameState} {solver : Solver cfg}
    (h : AdversarialClosedCycle.IsInitCycle cfg states solver) :
    TetrisSolvableValidFor cfg :=
  solvable_valid_via_cycle hcols h.1 h.2

/-- Config-specific: standard solvability via `IsInitCycle`. -/
theorem standard_solvable_of_isInitCycle
    {states : Finset GameState} {solver : Solver GameConfig.standard}
    (h : AdversarialClosedCycle.IsInitCycle GameConfig.standard states solver) :
    TetrisSolvableValid :=
  solvable_valid_of_isInitCycle (by decide) h

/-- Config-specific: tiny solvability via `IsInitCycle`. -/
theorem tiny_solvable_of_isInitCycle
    {states : Finset GameState} {solver : Solver GameConfig.tiny}
    (h : AdversarialClosedCycle.IsInitCycle GameConfig.tiny states solver) :
    TetrisSolvableValidFor GameConfig.tiny :=
  solvable_valid_of_isInitCycle (by decide) h

/-- **trivialSolver-specific positive entry point**: if `(states,
trivialSolver cfg)` satisfies the cycle obligations and contains `init`,
derive `TetrisSolvableValidFor cfg`. The atlas-search target where the
solver is the fully-computable `trivialSolver`. -/
theorem solvable_valid_via_trivialSolver_cycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {states : Finset GameState}
    (hobl : AdversarialClosedCycle.Obligations cfg states (trivialSolver cfg))
    (hinit : GameState.init ∈ states) :
    TetrisSolvableValidFor cfg :=
  solvable_valid_via_cycle hcols hobl hinit


/-- The canonical universe Finset: every game state whose board is in-field,
paired with every bag state. The natural starting set for `safeIterFinite`. -/
def inFieldStates (cfg : GameConfig) : Finset GameState :=
  (Finset.univ : Finset (InFieldBoard cfg × Bag)).image
    (fun p => ⟨p.1.val, p.2⟩)

/-- `GameState.init` lives in the universe. (Empty board is trivially WF
and in-field; `Bag.full` is a bag.) Useful as a sanity check and base case
for finite-state computations. -/
theorem init_mem_inFieldStates (cfg : GameConfig) :
    GameState.init ∈ inFieldStates cfg := by
  unfold inFieldStates
  rw [Finset.mem_image]
  let b : InFieldBoard cfg :=
    ⟨Board.empty, Board.empty_wf cfg, fun _ hp => absurd hp (Finset.notMem_empty _)⟩
  refine ⟨(b, Bag.full), Finset.mem_univ _, ?_⟩
  rfl

/-- `inFieldStates cfg` is always nonempty (contains `init`). -/
theorem inFieldStates_nonempty (cfg : GameConfig) :
    (inFieldStates cfg).Nonempty :=
  ⟨GameState.init, init_mem_inFieldStates cfg⟩

/-- `inFieldStates cfg` has at least 1 state — concrete trivial lower bound. -/
theorem inFieldStates_card_pos (cfg : GameConfig) :
    0 < (inFieldStates cfg).card :=
  Finset.card_pos.mpr (inFieldStates_nonempty cfg)

/-- `inFieldStates cfg ≠ ∅`. -/
theorem inFieldStates_ne_empty (cfg : GameConfig) :
    inFieldStates cfg ≠ ∅ :=
  (inFieldStates_nonempty cfg).ne_empty

/-- `inFieldStates cfg` has card ≥ 1. -/
theorem inFieldStates_card_ge_one (cfg : GameConfig) :
    1 ≤ (inFieldStates cfg).card :=
  inFieldStates_card_pos cfg

/-- `inFieldStates cfg` has nonzero card. -/
theorem inFieldStates_card_ne_zero (cfg : GameConfig) :
    (inFieldStates cfg).card ≠ 0 :=
  Nat.ne_of_gt (inFieldStates_card_pos cfg)

/-- **Universe cardinality**: `|inFieldStates cfg| = 2^(cols*rows) * 128`. -/
theorem inFieldStates_card (cfg : GameConfig) :
    (inFieldStates cfg).card = 2 ^ (cfg.cols * cfg.rows) * 128 := by
  unfold inFieldStates
  rw [Finset.card_image_of_injective _ ?inj]
  case inj =>
    intro p q hpq
    have hb : p.1.val = q.1.val := congrArg GameState.board hpq
    have hbg : p.2 = q.2 := congrArg GameState.bag hpq
    exact Prod.ext (Subtype.ext hb) hbg
  rw [Finset.card_univ, Fintype.card_prod, InFieldBoard.fintype_card,
      Bag.fintype_card]

/-- Standard config: universe has 2^200 * 128 = 2^207 states. -/
theorem standard_inFieldStates_card :
    (inFieldStates GameConfig.standard).card = 2 ^ 200 * 128 := by
  rw [inFieldStates_card, GameConfig.standard_cols, GameConfig.standard_rows]

/-- Standard universe size as `2^207` (since `128 = 2^7`). -/
theorem standard_inFieldStates_card_eq_two_pow_207 :
    (inFieldStates GameConfig.standard).card = 2 ^ 207 := by
  rw [standard_inFieldStates_card]
  rw [show (128 : ℕ) = 2 ^ 7 from by decide]
  rw [← pow_add]

/-- Parametric universe size as a single power of two: `2^(cols*rows + 7)`. -/
theorem inFieldStates_card_eq_two_pow (cfg : GameConfig) :
    (inFieldStates cfg).card = 2 ^ (cfg.cols * cfg.rows + 7) := by
  rw [inFieldStates_card]
  rw [show (128 : ℕ) = 2 ^ 7 from by decide]
  rw [← pow_add]


/-- Tiny (4×4) config: universe has 2^16 * 128 = 8 388 608 states. -/
theorem tiny_inFieldStates_card :
    (inFieldStates GameConfig.tiny).card = 8388608 := by
  rw [inFieldStates_card, GameConfig.tiny_cols, GameConfig.tiny_rows]
  decide

/-- Tiny universe size as `2^23`. -/
theorem tiny_inFieldStates_card_eq_two_pow_23 :
    (inFieldStates GameConfig.tiny).card = 2 ^ 23 := by
  rw [tiny_inFieldStates_card]; decide

/-- `tiny_inFieldStates_card < standard_inFieldStates_card`. -/
theorem tiny_inFieldStates_card_lt_standard_inFieldStates_card :
    (inFieldStates GameConfig.tiny).card <
      (inFieldStates GameConfig.standard).card := by
  rw [tiny_inFieldStates_card_eq_two_pow_23,
      standard_inFieldStates_card_eq_two_pow_207]
  exact Nat.pow_lt_pow_right (by decide) (by decide)

/-- Non-strict (`≤`) companion to Tick 1541. -/
theorem tiny_inFieldStates_card_le_standard_inFieldStates_card :
    (inFieldStates GameConfig.tiny).card ≤
      (inFieldStates GameConfig.standard).card :=
  le_of_lt tiny_inFieldStates_card_lt_standard_inFieldStates_card

/-- Distinctness (`≠`) companion to Tick 1541. -/
theorem tiny_inFieldStates_card_ne_standard_inFieldStates_card :
    (inFieldStates GameConfig.tiny).card ≠
      (inFieldStates GameConfig.standard).card :=
  Nat.ne_of_lt tiny_inFieldStates_card_lt_standard_inFieldStates_card

/-- `safeIterFinite cfg (inFieldStates cfg) N ⊆ inFieldStates cfg` for
every `N` (specialisation of `safeIterFinite_subset_S₀`). -/
theorem safeIterFinite_inFieldStates_subset (cfg : GameConfig) (N : ℕ) :
    safeIterFinite cfg (inFieldStates cfg) N ⊆ inFieldStates cfg :=
  safeIterFinite_subset_S₀ cfg (inFieldStates cfg) N

/-- Card bound for in-field-seed iterates. -/
theorem safeIterFinite_inFieldStates_card_le (cfg : GameConfig) (N : ℕ) :
    (safeIterFinite cfg (inFieldStates cfg) N).card ≤
      (inFieldStates cfg).card :=
  safeIterFinite_card_le_S₀_card cfg (inFieldStates cfg) N

/-- Each in-field iterate step is a subset of the previous. -/
theorem safeIterFinite_inFieldStates_succ_subset (cfg : GameConfig) (n : ℕ) :
    safeIterFinite cfg (inFieldStates cfg) (n + 1) ⊆
      safeIterFinite cfg (inFieldStates cfg) n :=
  safeIterFinite_succ_subset cfg (inFieldStates cfg) n

/-- Antitone form (in `n`) for in-field iterates. -/
theorem safeIterFinite_inFieldStates_antitone (cfg : GameConfig)
    {m n : ℕ} (h : m ≤ n) :
    safeIterFinite cfg (inFieldStates cfg) n ⊆
      safeIterFinite cfg (inFieldStates cfg) m :=
  safeIterFinite_antitone cfg (inFieldStates cfg) h

/-- Card monotonicity in `n` for in-field iterates. -/
theorem safeIterFinite_inFieldStates_card_succ_le (cfg : GameConfig) (n : ℕ) :
    (safeIterFinite cfg (inFieldStates cfg) (n + 1)).card ≤
      (safeIterFinite cfg (inFieldStates cfg) n).card :=
  safeIterFinite_card_succ_le cfg (inFieldStates cfg) n

/-- `≤`-form card monotonicity for in-field iterates. -/
theorem safeIterFinite_inFieldStates_card_le_of_le (cfg : GameConfig)
    {m n : ℕ} (h : m ≤ n) :
    (safeIterFinite cfg (inFieldStates cfg) n).card ≤
      (safeIterFinite cfg (inFieldStates cfg) m).card :=
  safeIterFinite_card_le_of_le cfg (inFieldStates cfg) h


/-- `inFieldStates cfg` has cardinality `≥ 128` (each of the 128 bags pairs
with at least the empty in-field board). -/
theorem inFieldStates_card_ge_128 (cfg : GameConfig) :
    128 ≤ (inFieldStates cfg).card := by
  rw [inFieldStates_card]
  have h1 : 1 ≤ 2 ^ (cfg.cols * cfg.rows) := Nat.one_le_two_pow
  calc 128 = 1 * 128 := by ring
    _ ≤ 2 ^ (cfg.cols * cfg.rows) * 128 := Nat.mul_le_mul_right _ h1

/-- `inFieldStates cfg` has card `> 1` (in particular, ≠ 1). -/
theorem inFieldStates_card_ne_one (cfg : GameConfig) :
    (inFieldStates cfg).card ≠ 1 := by
  have h := inFieldStates_card_ge_128 cfg
  omega

/-- Parametric universe size `≥ 2 ^ 7 = 128`, phrased as a power. -/
theorem inFieldStates_card_ge_two_pow_seven (cfg : GameConfig) :
    2 ^ 7 ≤ (inFieldStates cfg).card := by
  rw [show (2 ^ 7 : ℕ) = 128 from by decide]
  exact inFieldStates_card_ge_128 cfg

/-- Parametric universe size `≥ 1` (a power of 2 is always at least 1). -/
theorem inFieldStates_card_ge_one' (cfg : GameConfig) :
    1 ≤ (inFieldStates cfg).card := by
  rw [inFieldStates_card_eq_two_pow]
  exact Nat.one_le_two_pow

/-- Universe cardinality is divisible by 128. -/
theorem inFieldStates_card_mod_128_eq_zero (cfg : GameConfig) :
    (inFieldStates cfg).card % 128 = 0 := by
  rw [inFieldStates_card]; simp

/-- `Dvd` form of `inFieldStates_card_mod_128_eq_zero`. -/
theorem dvd_inFieldStates_card_128 (cfg : GameConfig) :
    128 ∣ (inFieldStates cfg).card :=
  Nat.dvd_of_mod_eq_zero (inFieldStates_card_mod_128_eq_zero cfg)

/-- Universe cardinality is even. -/
theorem inFieldStates_card_even (cfg : GameConfig) :
    (inFieldStates cfg).card % 2 = 0 := by
  have h128 := dvd_inFieldStates_card_128 cfg
  have h2_128 : (2 : ℕ) ∣ 128 := by decide
  exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans h2_128 h128)

/-- `Dvd` form of `inFieldStates_card_even`. -/
theorem dvd_inFieldStates_card_2 (cfg : GameConfig) :
    2 ∣ (inFieldStates cfg).card :=
  Nat.dvd_of_mod_eq_zero (inFieldStates_card_even cfg)

/-- `inFieldStates cfg` has card `≥ 2`. -/
theorem inFieldStates_card_ge_two (cfg : GameConfig) :
    2 ≤ (inFieldStates cfg).card := by
  have h := inFieldStates_card_ge_128 cfg
  omega

/-- `inFieldStates cfg` has card `≥ 28` (subsumes the cycle bound). -/
theorem inFieldStates_card_ge_28 (cfg : GameConfig) :
    28 ≤ (inFieldStates cfg).card := by
  have h := inFieldStates_card_ge_128 cfg
  omega

/-- Standard: universe has card `≥ 28`. -/
theorem standard_inFieldStates_card_ge_28 :
    28 ≤ (inFieldStates GameConfig.standard).card :=
  inFieldStates_card_ge_28 _

/-- Tiny: universe has card `≥ 28`. -/
theorem tiny_inFieldStates_card_ge_28 :
    28 ≤ (inFieldStates GameConfig.tiny).card :=
  inFieldStates_card_ge_28 _

/-- Tiny: universe has card `≥ 128`. -/
theorem tiny_inFieldStates_card_ge_128 :
    128 ≤ (inFieldStates GameConfig.tiny).card :=
  inFieldStates_card_ge_128 _

/-- Standard: universe has card `≥ 128`. -/
theorem standard_inFieldStates_card_ge_128 :
    128 ≤ (inFieldStates GameConfig.standard).card :=
  inFieldStates_card_ge_128 _


/-- A safe state's board has no cell at row `≥ cfg.rows` (the "doesn't overflow
upward" half of in-field-ness — the easier direction). -/
theorem safe_not_overflow {cfg : GameConfig} {g : GameState} (h : g ∈ safe cfg) :
    ∀ p ∈ g.board, p.2 < cfg.rows := by
  intro p hp
  by_contra hge
  exact safe_not_lost h ⟨p, hp, Nat.not_lt.mp hge⟩

/-- **Safe column-height bound**: every column of a safe state's board has
height at most `cfg.rows`. Direct consequence of `safe_not_overflow`. The
first atlas-construction-relevant invariant: safe states never have
overflowing stacks. -/
theorem safe_colHeight_le {cfg : GameConfig} {g : GameState} (h : g ∈ safe cfg)
    (j : ℕ) : Board.colHeight g.board j ≤ cfg.rows :=
  Board.colHeight_le_rows_of_in_field (safe_not_overflow h) j

/-- Reachable + not-lost version: any reachable non-lost state has bounded
column heights. -/
theorem reachable_not_lost_colHeight_le {cfg : GameConfig} {g : GameState}
    (_hr : Reachable cfg g) (hnl : ¬ g.lost cfg) (j : ℕ) :
    Board.colHeight g.board j ≤ cfg.rows := by
  apply Board.colHeight_le_rows_of_in_field _ j
  intro p hp
  exact Nat.lt_of_not_le (fun h => hnl ⟨p, hp, h⟩)

/-- **Contrapositive — overflowing columns ⇒ not safe**. Any state with a
column height exceeding `cfg.rows` is provably not safe. Atlas: instant
pruning of overflow candidate states. -/
theorem not_safe_of_colHeight_exceeds {cfg : GameConfig} {g : GameState}
    (j : ℕ) (h : cfg.rows < Board.colHeight g.board j) :
    g ∉ safe cfg := fun hsafe => by
  have := safe_colHeight_le hsafe j
  omega

/-- Concrete: safe states in `standard` have column heights ≤ 20. -/
theorem safe_standard_colHeight_le_twenty {g : GameState}
    (h : g ∈ safe GameConfig.standard) (j : ℕ) :
    Board.colHeight g.board j ≤ 20 := by
  have := safe_colHeight_le h j
  rw [GameConfig.standard_rows] at this
  exact this

/-- Concrete: safe states in `tiny` have column heights ≤ 4. -/
theorem safe_tiny_colHeight_le_four {g : GameState}
    (h : g ∈ safe GameConfig.tiny) (j : ℕ) :
    Board.colHeight g.board j ≤ 4 := by
  have := safe_colHeight_le h j
  rw [GameConfig.tiny_rows] at this
  exact this

/-- **Reachable + not-lost board cell-count bound**: weaker than safety;
only needs `¬ g.lost cfg` for the row bound. Atlas-relevant: any reachable
state we'd want to play from has bounded cell count. -/
theorem Reachable.not_lost_board_card_le {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.card ≤ cfg.cols * cfg.rows := by
  have hsub : g.board ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
    refine ⟨hr.wf p hp, ?_⟩
    exact Nat.lt_of_not_le (fun h => hnl ⟨p, hp, h⟩)
  calc g.board.card
      ≤ (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).card :=
        Finset.card_le_card hsub
    _ = cfg.cols * cfg.rows := by
        rw [Finset.card_product, Finset.card_range, Finset.card_range]

/-- `Board.count` alias for `Reachable.not_lost_board_card_le`. -/
theorem Reachable.count_le_of_not_lost {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.count ≤ cfg.cols * cfg.rows :=
  hr.not_lost_board_card_le hnl

/-- For init-seeded cycle traces, the board count never exceeds capacity. -/
theorem ClosedCycle.init_trace_count_le_capacity {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace cfg C.policy GameState.init n).board.count ≤ cfg.cols * cfg.rows :=
  (C.init_trace_reachable h n).count_le_of_not_lost (C.trace_not_lost h n)

/-- Standard specialisation: init-seeded trace board count ≤ 200. -/
theorem ClosedCycle.init_trace_count_le_two_hundred_standard
    (C : ClosedCycle GameConfig.standard) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace GameConfig.standard C.policy GameState.init n).board.count ≤ 200 := by
  have := C.init_trace_count_le_capacity h n
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at this
  exact this

/-- Tiny specialisation: init-seeded trace board count ≤ 16. -/
theorem ClosedCycle.init_trace_count_le_sixteen_tiny
    (C : ClosedCycle GameConfig.tiny) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace GameConfig.tiny C.policy GameState.init n).board.count ≤ 16 := by
  have := C.init_trace_count_le_capacity h n
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at this
  exact this


/-- **Safe board cell-count bound**: a reachable safe state has at most
`cols * rows` filled cells. Direct embedding into the in-field grid.
Atlas: prunes states with implausibly many cells. -/
theorem reachable_safe_board_card_le {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hs : g ∈ safe cfg) :
    g.board.card ≤ cfg.cols * cfg.rows := by
  have hsub : g.board ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
    exact ⟨hr.wf p hp, safe_not_overflow hs p hp⟩
  calc g.board.card
      ≤ (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).card :=
        Finset.card_le_card hsub
    _ = cfg.cols * cfg.rows := by
        rw [Finset.card_product, Finset.card_range, Finset.card_range]

/-- **Reachable-safe joint invariant**: standard cell count is *even* and
*at most 200*. Combines Tick 156 (parity) with Tick 150 (cell-count bound). -/
theorem reachable_safe_standard_count_invariant {g : GameState}
    (hr : Reachable GameConfig.standard g) (hs : g ∈ safe GameConfig.standard) :
    g.board.count % 2 = 0 ∧ g.board.count ≤ 200 := by
  refine ⟨Reachable.standard_count_mod_two hr, ?_⟩
  have h := reachable_safe_board_card_le hr hs
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  exact h

/-- **Reachable-safe joint invariant for tiny**: cell count is *divisible
by 4* and *at most 16*. -/
theorem reachable_safe_tiny_count_invariant {g : GameState}
    (hr : Reachable GameConfig.tiny g) (hs : g ∈ safe GameConfig.tiny) :
    g.board.count % 4 = 0 ∧ g.board.count ≤ 16 := by
  refine ⟨Reachable.tiny_count_mod_four hr, ?_⟩
  have h := reachable_safe_board_card_le hr hs
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at h
  exact h

/-- **Tiny reachable cell count is in {0, 4, 8, 12, 16}**. Direct
enumeration from `count % 4 = 0` + `count ≤ 16`. Atlas: only 5 possible
cell counts to consider for tiny. -/
theorem Reachable.tiny_count_in_set {g : GameState}
    (h : Reachable GameConfig.tiny g) (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count = 0 ∨ g.board.count = 4 ∨ g.board.count = 8 ∨
    g.board.count = 12 ∨ g.board.count = 16 := by
  have h1 := Reachable.tiny_count_mod_four h
  have h2 := h.not_lost_board_card_le hnl
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at h2
  change g.board.card = 0 ∨ g.board.card = 4 ∨ g.board.card = 8 ∨
       g.board.card = 12 ∨ g.board.card = 16
  unfold Board.count at h1
  omega

/-- Safe variant: reachable + safe tiny states have count in the 5-set. -/
theorem reachable_safe_tiny_count_in_set {g : GameState}
    (hr : Reachable GameConfig.tiny g) (hs : g ∈ safe GameConfig.tiny) :
    g.board.count = 0 ∨ g.board.count = 4 ∨ g.board.count = 8 ∨
    g.board.count = 12 ∨ g.board.count = 16 :=
  Reachable.tiny_count_in_set hr (safe_not_lost hs)

/-- **Standard reachable cell count is bounded by 200**. Concrete bound
specialised to the canonical 10×20 config. -/
theorem Reachable.standard_count_le_two_hundred {g : GameState}
    (h : Reachable GameConfig.standard g)
    (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count ≤ 200 := by
  have := h.not_lost_board_card_le hnl
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at this
  exact this

/-- **Tiny count cap**: a reachable non-lost tiny state has at most 16 cells. -/
theorem Reachable.tiny_count_le_sixteen {g : GameState}
    (h : Reachable GameConfig.tiny g)
    (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count ≤ 16 := by
  have := h.not_lost_board_card_le hnl
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at this
  exact this

/-- Tiny: a reachable state with `count > 16` must be lost. -/
theorem Reachable.tiny_lost_of_count_gt_sixteen {g : GameState}
    (h : Reachable GameConfig.tiny g) (hgt : 16 < g.board.count) :
    g.lost GameConfig.tiny := by
  by_contra hnl
  have hle := h.tiny_count_le_sixteen hnl
  omega

/-- Standard: a reachable state with `count > 200` must be lost. -/
theorem Reachable.standard_lost_of_count_gt_two_hundred {g : GameState}
    (h : Reachable GameConfig.standard g) (hgt : 200 < g.board.count) :
    g.lost GameConfig.standard := by
  by_contra hnl
  have hle := h.standard_count_le_two_hundred hnl
  omega

/-- Standard: any alive reachable state has `count ≤ 200`. -/
theorem Reachable.standard_count_le_two_hundred_of_alive {g : GameState}
    (h : Reachable GameConfig.standard g)
    (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count ≤ 200 :=
  h.standard_count_le_two_hundred hnl

/-- Tiny: any alive reachable state has `count ≤ 16`. -/
theorem Reachable.tiny_count_le_sixteen_of_alive {g : GameState}
    (h : Reachable GameConfig.tiny g)
    (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count ≤ 16 :=
  h.tiny_count_le_sixteen hnl

/-- Parametric `_of_alive` alias: any alive reachable state has
`count ≤ cfg.cols * cfg.rows`. -/
theorem Reachable.count_le_capacity_of_alive {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.count ≤ cfg.cols * cfg.rows :=
  h.count_le_of_not_lost hnl

/-- Standard: any alive reachable state has `count ≤ 200` and `count` is even. -/
theorem Reachable.standard_count_le_two_hundred_and_even_of_alive {g : GameState}
    (h : Reachable GameConfig.standard g)
    (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count ≤ 200 ∧ g.board.count % 2 = 0 :=
  ⟨h.standard_count_le_two_hundred hnl, h.standard_count_mod_two⟩

/-- Tiny: any alive reachable state has `count ≤ 16` and `count % 4 = 0`. -/
theorem Reachable.tiny_count_le_sixteen_and_mod_four_of_alive {g : GameState}
    (h : Reachable GameConfig.tiny g)
    (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count ≤ 16 ∧ g.board.count % 4 = 0 :=
  ⟨h.tiny_count_le_sixteen hnl, h.tiny_count_mod_four⟩

/-- Tiny: any alive reachable state has `count ∈ {0, 4, 8, 12, 16}`. -/
theorem Reachable.tiny_count_mem_multiples_of_four_of_alive {g : GameState}
    (h : Reachable GameConfig.tiny g)
    (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count = 0 ∨ g.board.count = 4 ∨ g.board.count = 8 ∨
      g.board.count = 12 ∨ g.board.count = 16 := by
  have ⟨hle, hmod⟩ := h.tiny_count_le_sixteen_and_mod_four_of_alive hnl
  omega

/-- Parametric: a reachable state with `count > cfg.cols * cfg.rows` must be lost. -/
theorem Reachable.lost_of_count_gt_capacity {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hgt : cfg.cols * cfg.rows < g.board.count) :
    g.lost cfg := by
  by_contra hnl
  have hle := h.count_le_of_not_lost hnl
  omega

/-- For every reachable state, either `count ≤ capacity` or the state is lost. -/
theorem Reachable.count_le_capacity_or_lost {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    g.board.count ≤ cfg.cols * cfg.rows ∨ g.lost cfg := by
  by_cases hnl : g.lost cfg
  · exact Or.inr hnl
  · exact Or.inl (h.count_le_of_not_lost hnl)

/-- Standard specialisation of `count_le_capacity_or_lost`. -/
theorem Reachable.standard_count_le_two_hundred_or_lost {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g.board.count ≤ 200 ∨ g.lost GameConfig.standard := by
  have := h.count_le_capacity_or_lost
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at this
  exact this

/-- Tiny specialisation of `count_le_capacity_or_lost`. -/
theorem Reachable.tiny_count_le_sixteen_or_lost {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g.board.count ≤ 16 ∨ g.lost GameConfig.tiny := by
  have := h.count_le_capacity_or_lost
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at this
  exact this

/-- Tiny: any reachable state with `count = 17` is lost. -/
theorem Reachable.tiny_lost_of_count_eq_seventeen {g : GameState}
    (h : Reachable GameConfig.tiny g) (h17 : g.board.count = 17) :
    g.lost GameConfig.tiny :=
  h.tiny_lost_of_count_gt_sixteen (by rw [h17]; decide)

/-- Standard: any reachable state with `count = 201` is lost. -/
theorem Reachable.standard_lost_of_count_eq_two_hundred_one {g : GameState}
    (h : Reachable GameConfig.standard g) (h201 : g.board.count = 201) :
    g.lost GameConfig.standard :=
  h.standard_lost_of_count_gt_two_hundred (by rw [h201]; decide)

/-- **Tiny full-board characterization**: a reachable non-lost tiny state
with cell count 16 must have exactly the full 4×4 grid as its board.
Atlas: pin down the unique "saturated" tiny state. Now a one-line corollary
of Tick 182's general `Board.eq_full_grid_of_card_eq`. -/
theorem tiny_count_16_implies_board_full {g : GameState}
    (hr : Reachable GameConfig.tiny g) (hnl : ¬ g.lost GameConfig.tiny)
    (h16 : g.board.count = 16) :
    g.board = Finset.range 4 ×ˢ Finset.range 4 := by
  have heq := Board.eq_full_grid_of_card_eq hr.wf
    (fun p hp => Nat.lt_of_not_le (fun h => hnl ⟨p, hp, h⟩))
    (by rw [GameConfig.tiny_cols, GameConfig.tiny_rows]; exact h16)
  rw [GameConfig.tiny_cols, GameConfig.tiny_rows] at heq
  exact heq

/-- Converse of `tiny_count_16_implies_board_full`: a board equal to the
full 4×4 grid has count 16. (No reachability/non-lost hypothesis needed.) -/
theorem tiny_board_full_implies_count_16 {g : GameState}
    (h : g.board = Finset.range 4 ×ˢ Finset.range 4) :
    g.board.count = 16 := by
  rw [h]
  change (Finset.range 4 ×ˢ Finset.range 4).card = 16
  simp [Finset.card_product, Finset.card_range]

/-- Iff bundling: for alive reachable tiny states,
`board = full grid ↔ count = 16`. -/
theorem tiny_count_16_iff_board_full {g : GameState}
    (hr : Reachable GameConfig.tiny g) (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count = 16 ↔ g.board = Finset.range 4 ×ˢ Finset.range 4 :=
  ⟨tiny_count_16_implies_board_full hr hnl, tiny_board_full_implies_count_16⟩

/-- Tiny: alive reachable states have `count ≠ 17` (cap is 16). -/
theorem Reachable.tiny_count_ne_seventeen_of_alive {g : GameState}
    (h : Reachable GameConfig.tiny g) (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count ≠ 17 := by
  have := h.tiny_count_le_sixteen hnl
  omega

/-- Standard: alive reachable states have `count ≠ 201` (cap is 200). -/
theorem Reachable.standard_count_ne_201_of_alive {g : GameState}
    (h : Reachable GameConfig.standard g) (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count ≠ 201 := by
  have := h.standard_count_le_two_hundred hnl
  omega

/-- Parametric: alive reachable states cannot have count exactly
`cfg.cols * cfg.rows + 1`. -/
theorem Reachable.count_ne_capacity_succ_of_alive {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.count ≠ cfg.cols * cfg.rows + 1 := by
  have := h.count_le_of_not_lost hnl
  omega

/-- Parametric: alive reachable states satisfy `count < cfg.cols * cfg.rows + 1`. -/
theorem Reachable.count_lt_capacity_succ_of_alive {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g.board.count < cfg.cols * cfg.rows + 1 := by
  have := h.count_le_of_not_lost hnl
  omega

/-- Standard: alive reachable states satisfy `count < 201`. -/
theorem Reachable.standard_count_lt_201_of_alive {g : GameState}
    (h : Reachable GameConfig.standard g) (hnl : ¬ g.lost GameConfig.standard) :
    g.board.count < 201 := by
  have := h.standard_count_le_two_hundred hnl
  omega

/-- Tiny: alive reachable states satisfy `count < 17`. -/
theorem Reachable.tiny_count_lt_17_of_alive {g : GameState}
    (h : Reachable GameConfig.tiny g) (hnl : ¬ g.lost GameConfig.tiny) :
    g.board.count < 17 := by
  have := h.tiny_count_le_sixteen hnl
  omega

/-- **Reachable safe standard states have `Even` cell count.** Repackages
Tick 156 in the `Even` form, convenient for `Even`/`Odd`-style reasoning. -/
theorem reachable_safe_standard_even_count {g : GameState}
    (hr : Reachable GameConfig.standard g) (_hs : g ∈ safe GameConfig.standard) :
    Even g.board.count :=
  Nat.even_iff.mpr (Reachable.standard_count_mod_two hr)


/-- **Reachable + safe states live in `inFieldStates`.** Combines
`Reachable.wf` (Tick 87, column-bound) with `safe_not_overflow`
(row-bound) to build the `InFieldBoard` witness. -/
theorem reachable_safe_mem_inFieldStates {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hs : g ∈ safe cfg) :
    g ∈ inFieldStates cfg := by
  unfold inFieldStates
  rw [Finset.mem_image]
  let b : InFieldBoard cfg := ⟨g.board, hr.wf, safe_not_overflow hs⟩
  refine ⟨(b, g.bag), Finset.mem_univ _, rfl⟩

/-- Set form: the intersection of `safe` and `Reachable` lives inside
the `inFieldStates` universe. -/
theorem safe_inter_reachable_subset_inFieldStates (cfg : GameConfig) :
    {g | g ∈ safe cfg ∧ Reachable cfg g} ⊆ (inFieldStates cfg : Finset GameState) :=
  fun _ ⟨hs, hr⟩ => reachable_safe_mem_inFieldStates hr hs

/-- A reachable non-lost state lives in `inFieldStates`. Weaker hypothesis
than safety: only needs `¬ g.lost cfg` for the row bound, plus
`Reachable.wf` for the column bound. -/
theorem reachable_not_lost_mem_inFieldStates {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g ∈ inFieldStates cfg := by
  unfold inFieldStates
  rw [Finset.mem_image]
  have hrows : ∀ p ∈ g.board, p.2 < cfg.rows := fun p hp =>
    Nat.lt_of_not_le (fun h => hnl ⟨p, hp, h⟩)
  let b : InFieldBoard cfg := ⟨g.board, hr.wf, hrows⟩
  refine ⟨(b, g.bag), Finset.mem_univ _, rfl⟩

/-- **Dichotomy**: every reachable state either lives in `inFieldStates`
or is lost. Atlas: classify each candidate state as in-field or
already-lost — no third category. -/
theorem Reachable.mem_inFieldStates_or_lost {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g ∈ inFieldStates cfg ∨ g.lost cfg := by
  by_cases hlost : g.lost cfg
  · right; exact hlost
  · left; exact reachable_not_lost_mem_inFieldStates h hlost

/-- Dot-form: `Reachable + ¬ lost ⇒ g ∈ inFieldStates`. -/
theorem Reachable.mem_inFieldStates_of_not_lost {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g ∈ inFieldStates cfg :=
  reachable_not_lost_mem_inFieldStates h hnl

/-- Dot-form: `Reachable + safe ⇒ g ∈ inFieldStates`. -/
theorem Reachable.mem_inFieldStates_of_safe {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hs : g ∈ safe cfg) :
    g ∈ inFieldStates cfg :=
  reachable_safe_mem_inFieldStates h hs

/-- Init-seeded trace state lives in the universe `inFieldStates`. -/
theorem ClosedCycle.init_trace_mem_inFieldStates {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ) :
    trace cfg C.policy GameState.init n ∈ inFieldStates cfg :=
  (C.init_trace_reachable h n).mem_inFieldStates_of_not_lost (C.trace_not_lost h n)

/-- Parametric: any Reachable-seeded cycle trace state lives in `inFieldStates`. -/
theorem ClosedCycle.trace_mem_inFieldStates {cfg : GameConfig}
    (C : ClosedCycle cfg) {g0 : GameState} (h0 : g0 ∈ C.states)
    (hr : Reachable cfg g0) (n : ℕ) :
    trace cfg C.policy g0 n ∈ inFieldStates cfg :=
  (C.trace_reachable h0 hr n).mem_inFieldStates_of_not_lost (C.trace_not_lost h0 n)

/-- Init-seeded trace board has `count ≠ cfg.cols * cfg.rows + 1`. -/
theorem ClosedCycle.init_trace_count_ne_capacity_succ {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace cfg C.policy GameState.init n).board.count ≠ cfg.cols * cfg.rows + 1 := by
  have := C.init_trace_count_le_capacity h n
  omega

/-- Init-seeded trace board has `count < cfg.cols * cfg.rows + 1`. -/
theorem ClosedCycle.init_trace_count_lt_capacity_succ {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace cfg C.policy GameState.init n).board.count < cfg.cols * cfg.rows + 1 := by
  have := C.init_trace_count_le_capacity h n
  omega

/-- For init-seeded standard cycle traces, the count is even. -/
theorem ClosedCycle.init_trace_count_even_standard
    (C : ClosedCycle GameConfig.standard) (h : GameState.init ∈ C.states) (n : ℕ) :
    Even (trace GameConfig.standard C.policy GameState.init n).board.count :=
  (C.init_trace_reachable h n).standard_count_even

/-- For init-seeded tiny cycle traces, the count is divisible by 4. -/
theorem ClosedCycle.init_trace_count_mod_four_tiny
    (C : ClosedCycle GameConfig.tiny) (h : GameState.init ∈ C.states) (n : ℕ) :
    (trace GameConfig.tiny C.policy GameState.init n).board.count % 4 = 0 :=
  (C.init_trace_reachable h n).tiny_count_mod_four

/-- For init-seeded tiny cycle traces, the count is in `{0, 4, 8, 12, 16}`. -/
theorem ClosedCycle.init_trace_count_mem_multiples_of_four_tiny
    (C : ClosedCycle GameConfig.tiny) (h : GameState.init ∈ C.states) (n : ℕ) :
    let c := (trace GameConfig.tiny C.policy GameState.init n).board.count
    c = 0 ∨ c = 4 ∨ c = 8 ∨ c = 12 ∨ c = 16 := by
  have hmod := C.init_trace_count_mod_four_tiny h n
  have hle := C.init_trace_count_le_sixteen_tiny h n
  omega

/-- Standard init-seeded trace count bundle: `even ∧ ≤ 200`. -/
theorem ClosedCycle.init_trace_count_even_and_le_two_hundred_standard
    (C : ClosedCycle GameConfig.standard) (h : GameState.init ∈ C.states) (n : ℕ) :
    Even (trace GameConfig.standard C.policy GameState.init n).board.count ∧
      (trace GameConfig.standard C.policy GameState.init n).board.count ≤ 200 :=
  ⟨C.init_trace_count_even_standard h n,
    C.init_trace_count_le_two_hundred_standard h n⟩

/-- If an init-seeded trace state has both `count = 0` and `bag = full`,
it equals `init`. -/
theorem ClosedCycle.init_trace_eq_init_of_count_zero_and_bag_full
    {cfg : GameConfig} (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (h0 : (trace cfg C.policy GameState.init n).board.count = 0)
    (hbag : (trace cfg C.policy GameState.init n).bag = Bag.full) :
    trace cfg C.policy GameState.init n = GameState.init :=
  (C.init_trace_reachable h n).eq_init_of_bag_full_of_count_zero h0 hbag

/-- For an init-seeded trace, `trace_n = init ↔ count = 0 ∧ bag = full`. -/
theorem ClosedCycle.init_trace_eq_init_iff_count_zero_and_bag_full
    {cfg : GameConfig} (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ) :
    trace cfg C.policy GameState.init n = GameState.init ↔
      (trace cfg C.policy GameState.init n).board.count = 0 ∧
        (trace cfg C.policy GameState.init n).bag = Bag.full :=
  (C.init_trace_reachable h n).eq_init_iff_count_zero_and_bag_full

/-- An init-seeded trace with positive count is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_count_pos {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hpos : 0 < (trace cfg C.policy GameState.init n).board.count) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_board_count_pos hpos

/-- An init-seeded trace with `bag ≠ full` is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_bag_ne_full {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hne : (trace cfg C.policy GameState.init n).bag ≠ Bag.full) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_bag_ne_full hne

/-- An init-seeded trace with `bag.card < 7` is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_bag_card_lt_seven {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hlt : (trace cfg C.policy GameState.init n).bag.card < 7) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_bag_card_lt_seven hlt

/-- An init-seeded trace with `bag.card ≤ 6` is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_bag_card_le_six {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hle : (trace cfg C.policy GameState.init n).bag.card ≤ 6) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_bag_card_le_six hle

/-- An init-seeded trace with `board ≠ empty` is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_board_ne_empty {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hne : (trace cfg C.policy GameState.init n).board ≠ Board.empty) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_board_ne_empty hne

/-- An init-seeded trace with `board.Nonempty` is not `init`. -/
theorem ClosedCycle.init_trace_ne_init_of_board_nonempty {cfg : GameConfig}
    (C : ClosedCycle cfg) (h : GameState.init ∈ C.states) (n : ℕ)
    (hne : (trace cfg C.policy GameState.init n).board.Nonempty) :
    trace cfg C.policy GameState.init n ≠ GameState.init :=
  (C.init_trace_reachable h n).ne_init_of_board_nonempty hne


/-- Contrapositive of `Reachable.mem_inFieldStates_or_lost`: a state outside
the universe and not lost cannot be reachable. -/
theorem not_reachable_of_notMem_inFieldStates_of_not_lost {cfg : GameConfig}
    {g : GameState} (hne : g ∉ inFieldStates cfg) (hnl : ¬ g.lost cfg) :
    ¬ Reachable cfg g := by
  intro h
  rcases h.mem_inFieldStates_or_lost with hin | hlost
  · exact hne hin
  · exact hnl hlost

/-- Variant: a safe state outside the universe cannot be reachable. -/
theorem not_reachable_of_notMem_inFieldStates_of_safe {cfg : GameConfig}
    {g : GameState} (hne : g ∉ inFieldStates cfg) (hs : g ∈ safe cfg) :
    ¬ Reachable cfg g :=
  not_reachable_of_notMem_inFieldStates_of_not_lost hne (safe_not_lost hs)

/-- **In-field-state invariants bundle**: every reachable non-lost state
satisfies (1) WF, (2) all cells in field rows, (3) at most `cols*rows`
cells, (4) nonempty bag, (5) bag.card ≤ 7. Convenient atlas precondition
pack. -/
theorem Reachable.in_field_invariants {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    Board.WF cfg g.board ∧
    (∀ p ∈ g.board, p.2 < cfg.rows) ∧
    g.board.card ≤ cfg.cols * cfg.rows ∧
    g.bag.Nonempty ∧
    g.bag.card ≤ 7 :=
  ⟨h.wf,
   fun p hp => Nat.lt_of_not_le (fun ho => hnl ⟨p, hp, ho⟩),
   h.not_lost_board_card_le hnl,
   h.bag_nonempty,
   h.bag_card_le_seven⟩

/-- **Every trace state from `init` (via `safeSolver`) lives in
`inFieldStates`.** Direct corollary of Tick 92's
`adversarialTrace_reachable_and_safe` plus
`reachable_safe_mem_inFieldStates`. -/
theorem adversarialTrace_mem_inFieldStates {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (s : ℕ → Piece) (hl : LegalSequence s)
    (n : ℕ) :
    adversarialTrace cfg (safeSolver cfg) s GameState.init n ∈
      inFieldStates cfg := by
  obtain ⟨hr, hs⟩ := adversarialTrace_reachable_and_safe h s hl n
  exact reachable_safe_mem_inFieldStates hr hs

/-- **The safe-reachable Finset**: the cycle candidate built from
`init ∈ safe cfg`. Consists of all `inFieldStates` states that are both
`Reachable` from `init` and live in `safe`. This is the natural cycle
extracted from safety (by `Reachable.step` closure + `safeSolver_spec`). -/
noncomputable def safeReachableCycle (cfg : GameConfig) : Finset GameState := by
  classical
  exact (inFieldStates cfg).filter (fun g => Reachable cfg g ∧ g ∈ safe cfg)

theorem mem_safeReachableCycle (cfg : GameConfig) (g : GameState) :
    g ∈ safeReachableCycle cfg ↔
      g ∈ inFieldStates cfg ∧ Reachable cfg g ∧ g ∈ safe cfg := by
  unfold safeReachableCycle
  classical
  exact Finset.mem_filter

theorem init_mem_safeReachableCycle {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    GameState.init ∈ safeReachableCycle cfg :=
  (mem_safeReachableCycle cfg GameState.init).mpr
    ⟨init_mem_inFieldStates cfg, Reachable.init, h⟩

/-- **Simplified membership criterion**: the `inFieldStates`-membership
clause is redundant — it's implied by `Reachable ∧ safe` via
`reachable_safe_mem_inFieldStates`. -/
theorem mem_safeReachableCycle_iff_reachable_and_safe
    {cfg : GameConfig} {g : GameState} :
    g ∈ safeReachableCycle cfg ↔ Reachable cfg g ∧ g ∈ safe cfg := by
  rw [mem_safeReachableCycle]
  exact ⟨fun ⟨_, hr, hs⟩ => ⟨hr, hs⟩,
         fun ⟨hr, hs⟩ => ⟨reachable_safe_mem_inFieldStates hr hs, hr, hs⟩⟩

/-- Negated form of `mem_safeReachableCycle_iff_reachable_and_safe`. -/
theorem not_mem_safeReachableCycle_iff_not_reachable_or_not_safe
    {cfg : GameConfig} {g : GameState} :
    g ∉ safeReachableCycle cfg ↔ ¬ Reachable cfg g ∨ g ∉ safe cfg := by
  rw [mem_safeReachableCycle_iff_reachable_and_safe]
  rw [not_and_or]

/-- **Reachable ∧ safe ⇒ in atlas**: the canonical atlas absorbs any
reachable-and-safe state. Direct from
`mem_safeReachableCycle_iff_reachable_and_safe`. -/
theorem reachable_safe_mem_safeReachableCycle (cfg : GameConfig)
    {g : GameState} (hr : Reachable cfg g) (hs : g ∈ safe cfg) :
    g ∈ safeReachableCycle cfg :=
  mem_safeReachableCycle_iff_reachable_and_safe.mpr ⟨hr, hs⟩

/-- **Set-level atlas characterization**: as a `Set GameState`, the
canonical atlas is exactly the intersection `Reachable ∩ safe`. -/
theorem safeReachableCycle_coe_eq (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) =
      {g | Reachable cfg g} ∩ safe cfg := by
  ext g
  exact mem_safeReachableCycle_iff_reachable_and_safe

/-- Set-level subset form of Tick 265. -/
theorem safeReachableCycle_coe_subset_reachable_inter_safe
    (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) ⊆
      {g | Reachable cfg g} ∩ safe cfg :=
  (safeReachableCycle_coe_eq cfg).le

/-- Set-level reverse subset form. -/
theorem reachable_inter_safe_subset_safeReachableCycle_coe
    (cfg : GameConfig) :
    {g | Reachable cfg g} ∩ safe cfg ⊆
      (↑(safeReachableCycle cfg) : Set GameState) :=
  (safeReachableCycle_coe_eq cfg).ge

/-- **Atlas coercion as `safe ∩ reachable`** (commuted order of
Tick 265). -/
theorem safeReachableCycle_coe_eq_safe_inter (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) =
      safe cfg ∩ {g | Reachable cfg g} := by
  rw [safeReachableCycle_coe_eq, Set.inter_comm]

/-- **Atlas-empty iff no reachable-and-safe state exists.** -/
theorem safeReachableCycle_eq_empty_iff (cfg : GameConfig) :
    safeReachableCycle cfg = ∅ ↔ ¬ ∃ g, Reachable cfg g ∧ g ∈ safe cfg := by
  rw [← Finset.not_nonempty_iff_eq_empty]
  refine ⟨?_, ?_⟩
  · intro hempty ⟨g, hr, hs⟩
    exact hempty ⟨g, mem_safeReachableCycle_iff_reachable_and_safe.mpr ⟨hr, hs⟩⟩
  · rintro hne ⟨g, hg⟩
    obtain ⟨hr, hs⟩ := mem_safeReachableCycle_iff_reachable_and_safe.mp hg
    exact hne ⟨g, hr, hs⟩

/-- **Atlas nonempty iff some reachable-and-safe state exists.** -/
theorem safeReachableCycle_nonempty_iff (cfg : GameConfig) :
    (safeReachableCycle cfg).Nonempty ↔
      ∃ g, Reachable cfg g ∧ g ∈ safe cfg := by
  refine ⟨?_, ?_⟩
  · rintro ⟨g, hg⟩
    exact ⟨g, (mem_safeReachableCycle_iff_reachable_and_safe.mp hg).1,
           (mem_safeReachableCycle_iff_reachable_and_safe.mp hg).2⟩
  · rintro ⟨g, hr, hs⟩
    exact ⟨g, mem_safeReachableCycle_iff_reachable_and_safe.mpr ⟨hr, hs⟩⟩

/-- **Atlas card positive iff some reachable-and-safe state exists.** -/
theorem safeReachableCycle_card_pos_iff (cfg : GameConfig) :
    0 < (safeReachableCycle cfg).card ↔
      ∃ g, Reachable cfg g ∧ g ∈ safe cfg := by
  rw [Finset.card_pos]; exact safeReachableCycle_nonempty_iff cfg

/-- **The empty-bag empty-board state is in the universe but never
reachable** — a concrete out-of-atlas witness. `⟨Board.empty, ∅⟩` is
trivially well-formed and in-field, so it lives in `inFieldStates`;
but `Reachable.bag_nonempty` rules out any reachable state with an
empty bag, so it cannot be in the canonical atlas. -/
theorem empty_bag_empty_board_not_reachable (cfg : GameConfig) :
    ¬ Reachable cfg ⟨Board.empty, (∅ : Bag)⟩ := by
  intro hr
  have hne := hr.bag_nonempty
  rcases hne with ⟨_, hmem⟩
  exact absurd hmem (Finset.notMem_empty _)

/-- **The empty-bag empty-board state is not in the canonical atlas.**
Composes Tick 267 (`¬ Reachable`) with Tick 264 (atlas iff `Reachable
∧ safe`). -/
theorem empty_bag_empty_board_not_mem_safeReachableCycle (cfg : GameConfig) :
    (⟨Board.empty, (∅ : Bag)⟩ : GameState) ∉ safeReachableCycle cfg := by
  rw [mem_safeReachableCycle_iff_reachable_and_safe]
  exact fun ⟨hr, _⟩ => empty_bag_empty_board_not_reachable cfg hr

/-- **Empty-bag empty-board lives in the universe.** Witnessed by the
trivially-WF empty board paired with the empty bag. -/
theorem empty_bag_empty_board_mem_inFieldStates (cfg : GameConfig) :
    (⟨Board.empty, (∅ : Bag)⟩ : GameState) ∈ inFieldStates cfg := by
  unfold inFieldStates
  rw [Finset.mem_image]
  refine ⟨(⟨Board.empty, Board.empty_wf cfg,
            fun _ hp => absurd hp (Finset.notMem_empty _)⟩, (∅ : Bag)),
          Finset.mem_univ _, rfl⟩

/-- **Finiteness of `Reachable ∩ safe`**: the Set-level
reachable-and-safe slice is always finite, witnessed by the canonical
atlas Finset. Lifts Finset finiteness through `safeReachableCycle_coe_eq`. -/
theorem reachable_inter_safe_finite (cfg : GameConfig) :
    ({g | Reachable cfg g} ∩ safe cfg).Finite := by
  rw [← safeReachableCycle_coe_eq]
  exact (safeReachableCycle cfg).finite_toSet

/-- **Nonemptiness of `Reachable ∩ safe`** when init is safe: init
witnesses the intersection. -/
theorem reachable_inter_safe_nonempty_of_init_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ({g | Reachable cfg g} ∩ safe cfg).Nonempty :=
  ⟨GameState.init, Reachable.init, h⟩

/-- **Set-level finiteness** of the canonical atlas Finset coerced as
a Set. Direct from `Finset.finite_toSet`. -/
theorem safeReachableCycle_coe_finite (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState).Finite :=
  (safeReachableCycle cfg).finite_toSet

/-- **Set-level finiteness** of the universe `inFieldStates` Finset
coerced as a Set. -/
theorem inFieldStates_coe_finite (cfg : GameConfig) :
    (↑(inFieldStates cfg) : Set GameState).Finite :=
  (inFieldStates cfg).finite_toSet


theorem safeReachableCycle_subset_inFieldStates (cfg : GameConfig) :
    safeReachableCycle cfg ⊆ inFieldStates cfg := fun g hg =>
  ((mem_safeReachableCycle cfg g).mp hg).1

/-- **Set-level inclusion** of canonical atlas in universe. -/
theorem safeReachableCycle_coe_subset_inFieldStates_coe (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) ⊆ ↑(inFieldStates cfg) :=
  Finset.coe_subset.mpr (safeReachableCycle_subset_inFieldStates cfg)


theorem safeReachableCycle_subset_safe (cfg : GameConfig) :
    ∀ g ∈ safeReachableCycle cfg, g ∈ safe cfg := fun g hg =>
  ((mem_safeReachableCycle cfg g).mp hg).2.2

/-- Projection: every atlas state is `Reachable`. -/
theorem safeReachableCycle_subset_reachable (cfg : GameConfig) :
    ∀ g ∈ safeReachableCycle cfg, Reachable cfg g := fun g hg =>
  ((mem_safeReachableCycle cfg g).mp hg).2.1

/-- Bag dichotomy for atlas states: bag = full or card < 7. -/
theorem safeReachableCycle_bag_eq_full_or_card_lt_seven {cfg : GameConfig}
    {g : GameState} (_h : g ∈ safeReachableCycle cfg) :
    g.bag = Bag.full ∨ g.bag.card < 7 :=
  g.bag_eq_full_or_card_lt_seven

/-- Bag dichotomy for atlas states: bag = full or card ≤ 6. -/
theorem safeReachableCycle_bag_eq_full_or_card_le_six {cfg : GameConfig}
    {g : GameState} (_h : g ∈ safeReachableCycle cfg) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 :=
  g.bag_eq_full_or_card_le_six

/-- Bag-card-7-iff-full for atlas states. -/
theorem safeReachableCycle_bag_card_eq_seven_iff_bag_eq_full
    {cfg : GameConfig} {g : GameState}
    (_h : g ∈ safeReachableCycle cfg) :
    g.bag.card = 7 ↔ g.bag = Bag.full :=
  Bag.card_eq_seven_iff_eq_full g.bag


/-- **Atlas coe is a subset of `safe`** (as a Set). Trivial via
`safeReachableCycle_subset_safe`. -/
theorem safeReachableCycle_coe_subset_safe (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) ⊆ safe cfg :=
  fun _ hg => safeReachableCycle_subset_safe cfg _ (Finset.mem_coe.mp hg)

/-- Atlas coe ⊆ Reachable Set. -/
theorem safeReachableCycle_coe_subset_reachable (cfg : GameConfig) :
    (↑(safeReachableCycle cfg) : Set GameState) ⊆ {g | Reachable cfg g} :=
  fun _ hg => safeReachableCycle_subset_reachable cfg _ (Finset.mem_coe.mp hg)


theorem safeReachableCycle_nonempty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableCycle cfg).Nonempty :=
  ⟨GameState.init, init_mem_safeReachableCycle h⟩

/-- `safeReachableCycle cfg ≠ ∅` when `init ∈ safe`. -/
theorem safeReachableCycle_ne_empty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    safeReachableCycle cfg ≠ ∅ :=
  (safeReachableCycle_nonempty h).ne_empty

/-- Contrapositive: if init isn't safe, it's not in the canonical
atlas (since membership ⇒ safety, via subset_safe). -/
theorem not_init_mem_safeReachableCycle_of_init_not_safe
    {cfg : GameConfig} (h : GameState.init ∉ safe cfg) :
    GameState.init ∉ safeReachableCycle cfg :=
  fun hmem => h (safeReachableCycle_subset_safe cfg _ hmem)

theorem safeReachableCycle_card_pos {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) : 0 < (safeReachableCycle cfg).card :=
  Finset.card_pos.mpr (safeReachableCycle_nonempty h)

/-- The canonical atlas card is nonzero. Nat alias of card_pos. -/
theorem safeReachableCycle_card_ne_zero {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableCycle cfg).card ≠ 0 :=
  Nat.ne_of_gt (safeReachableCycle_card_pos h)

/-- `1 ≤ card` form of the positivity bound. -/
theorem safeReachableCycle_card_ge_one {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    1 ≤ (safeReachableCycle cfg).card :=
  safeReachableCycle_card_pos h

theorem safeReachableCycle_card_le_inFieldStates (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤ (inFieldStates cfg).card :=
  Finset.card_le_card (safeReachableCycle_subset_inFieldStates cfg)

/-- Closed-form upper bound on the canonical atlas card. -/
theorem safeReachableCycle_card_le_pow (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤ 2 ^ (cfg.cols * cfg.rows) * 128 :=
  inFieldStates_card cfg ▸ safeReachableCycle_card_le_inFieldStates cfg


/-- **Strict atlas-inclusion**: the canonical atlas is *strictly* smaller
than `inFieldStates cfg` — the empty-board / empty-bag state is the
explicit out-of-atlas witness. -/
theorem safeReachableCycle_strict_subset_inFieldStates (cfg : GameConfig) :
    safeReachableCycle cfg ⊂ inFieldStates cfg := by
  refine ⟨safeReachableCycle_subset_inFieldStates cfg, ?_⟩
  intro h
  exact empty_bag_empty_board_not_mem_safeReachableCycle cfg
    (h (empty_bag_empty_board_mem_inFieldStates cfg))

/-- **Strict card inequality**: the atlas has *strictly* fewer states
than the universe. Card-level form of
`safeReachableCycle_strict_subset_inFieldStates`. -/
theorem safeReachableCycle_card_lt_inFieldStates_card (cfg : GameConfig) :
    (safeReachableCycle cfg).card < (inFieldStates cfg).card :=
  Finset.card_lt_card (safeReachableCycle_strict_subset_inFieldStates cfg)

/-- Closed-form strict upper bound on the canonical atlas card. -/
theorem safeReachableCycle_card_lt_pow (cfg : GameConfig) :
    (safeReachableCycle cfg).card < 2 ^ (cfg.cols * cfg.rows) * 128 :=
  inFieldStates_card cfg ▸ safeReachableCycle_card_lt_inFieldStates_card cfg

/-- **Standard strict upper bound**: the standard atlas has strictly
fewer than `2^207` states. -/
theorem safeReachableCycle_standard_card_lt :
    (safeReachableCycle GameConfig.standard).card < 2 ^ 200 * 128 :=
  standard_inFieldStates_card ▸
    safeReachableCycle_card_lt_inFieldStates_card GameConfig.standard

/-- **Tiny strict upper bound**: the tiny atlas has strictly fewer
than `8 388 608` states. -/
theorem safeReachableCycle_tiny_card_lt :
    (safeReachableCycle GameConfig.tiny).card < 8388608 :=
  tiny_inFieldStates_card ▸
    safeReachableCycle_card_lt_inFieldStates_card GameConfig.tiny


/-- `init` is in the safe-reachable cycle Finset **iff** init is safe.
Atlas: the cycle Finset's membership for init *exactly* characterises
solvability (modulo the cfg.cols hyp). -/
theorem init_mem_safeReachableCycle_iff (cfg : GameConfig) :
    GameState.init ∈ safeReachableCycle cfg ↔ GameState.init ∈ safe cfg :=
  ⟨safeReachableCycle_subset_safe cfg _, init_mem_safeReachableCycle⟩

/-- `Obligations` + `init ∈ states` puts `init` in the canonical
atlas. Bridges Tick 570 (`Obligations.init_safe`) to the
`safeReachableCycle` Finset. -/
theorem AdversarialClosedCycle.Obligations.init_mem_safeReachableCycle
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : AdversarialClosedCycle.Obligations cfg states solver)
    (hinit : GameState.init ∈ states) :
    GameState.init ∈ safeReachableCycle cfg :=
  (init_mem_safeReachableCycle_iff cfg).mpr (h.init_safe hinit)

/-- `IsInitCycle` puts `init` in the canonical atlas. -/
theorem AdversarialClosedCycle.IsInitCycle.init_mem_safeReachableCycle
    {cfg : GameConfig} {states : Finset GameState} {solver : Solver cfg}
    (h : AdversarialClosedCycle.IsInitCycle cfg states solver) :
    GameState.init ∈ safeReachableCycle cfg :=
  h.obligations.init_mem_safeReachableCycle h.init_mem

/-- An adversarial closed cycle containing `init` puts `init` in
the canonical atlas. The full bridge from existence of a cycle
through init to atlas membership. -/
theorem AdversarialClosedCycle.init_mem_safeReachableCycle_of_init_mem
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hinit : GameState.init ∈ C.states) :
    GameState.init ∈ safeReachableCycle cfg :=
  (init_mem_safeReachableCycle_iff cfg).mpr (C.init_safe_of_init_mem hinit)

/-- Solvability ⇒ init in canonical atlas (under `4 ≤ cols`). -/
theorem TetrisSolvableValidFor.init_mem_safeReachableCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    GameState.init ∈ safeReachableCycle cfg :=
  (init_mem_safeReachableCycle_iff cfg).mpr (h.init_safe hcols)



/-- **Generic parametric iff**: Tetris-valid-solvable iff init is in the
canonical `safeReachableCycle` Finset. The fully parametric version of
Ticks 232 (standard) and 233 (tiny). -/
theorem tetrisSolvableValidFor_iff_init_mem_safeReachableCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
    GameState.init ∈ safeReachableCycle cfg := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      ← init_mem_safeReachableCycle_iff]

/-- **The atlas-Finset membership *is* Tetris solvability** (for standard).
Now a one-line corollary of the parametric
`tetrisSolvableValidFor_iff_init_mem_safeReachableCycle`. -/
theorem standard_init_mem_safeReachableCycle_iff_solvable :
    GameState.init ∈ safeReachableCycle GameConfig.standard ↔
    TetrisSolvableValid :=
  (tetrisSolvableValidFor_iff_init_mem_safeReachableCycle
    (cfg := GameConfig.standard) (by decide)).symm

/-- **The atlas-Finset membership *is* Tetris solvability** (for tiny). -/
theorem tiny_init_mem_safeReachableCycle_iff_solvable :
    GameState.init ∈ safeReachableCycle GameConfig.tiny ↔
    TetrisSolvableValidFor GameConfig.tiny :=
  (tetrisSolvableValidFor_iff_init_mem_safeReachableCycle
    (cfg := GameConfig.tiny) (by decide)).symm

/-- **Atlas closure**: the `safeReachableCycle` Finset is closed under
`safeSolver` steps. From any state in the atlas domain and any drawable
piece, the canonical successor is also in the atlas domain. -/
theorem safeReachableCycle_safeSolver_step {cfg : GameConfig} {g : GameState}
    (hg : g ∈ safeReachableCycle cfg) {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (safeSolver cfg g p) ∈ safeReachableCycle cfg := by
  rw [mem_safeReachableCycle] at hg
  obtain ⟨_, hr, hs⟩ := hg
  obtain ⟨hr', hs'⟩ := reachable_safe_safeSolver_step hr hs hp
  rw [mem_safeReachableCycle]
  exact ⟨reachable_safe_mem_inFieldStates hr' hs', hr', hs'⟩

/-- **Init-step stays in atlas**: starting from init (when safe), any
safeSolver-step lands in the atlas Finset. Direct instance of the closure
above with `g = init`. -/
theorem safeReachableCycle_init_step {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (p : Piece) :
    adversarialStep cfg GameState.init p (safeSolver cfg GameState.init p)
      ∈ safeReachableCycle cfg :=
  safeReachableCycle_safeSolver_step (init_mem_safeReachableCycle h)
    (GameState.init_bag.symm ▸ Bag.mem_full p)

/-- **Full trace stays in atlas**: every state along the trace from `init`
via `safeSolver` against any legal sequence is in `safeReachableCycle`.
Inductive lift of `safeReachableCycle_init_step`. -/
theorem safeReachableCycle_trace_mem {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (s : ℕ → Piece) (hl : LegalSequence s)
    (n : ℕ) :
    adversarialTrace cfg (safeSolver cfg) s GameState.init n
      ∈ safeReachableCycle cfg := by
  induction n with
  | zero => exact init_mem_safeReachableCycle h
  | succ k ih =>
    rw [adversarialTrace_succ]
    apply safeReachableCycle_safeSolver_step ih
    rw [adversarialTrace_bag]
    exact hl k



/-- **Pigeonhole**: the trace from `init` via `safeSolver` cannot all be
distinct — `|inFieldStates cfg|+1` indices must contain a collision in
the finite universe. -/
theorem exists_repeat_trace_state {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (s : ℕ → Piece) (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧
      adversarialTrace cfg (safeSolver cfg) s GameState.init i =
      adversarialTrace cfg (safeSolver cfg) s GameState.init j := by
  let N := (inFieldStates cfg).card
  have hmap : ∀ a ∈ Finset.range (N + 1),
      adversarialTrace cfg (safeSolver cfg) s GameState.init a ∈ inFieldStates cfg :=
    fun a _ => adversarialTrace_mem_inFieldStates h s hl a
  have hlt : (inFieldStates cfg).card < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨x, _, y, _, hne, hfxy⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hlt hmap
  rcases lt_or_gt_of_ne hne with hlt' | hgt'
  · exact ⟨x, y, hlt', hfxy⟩
  · exact ⟨y, x, hgt', hfxy.symm⟩

/-- **Universe containment for cycle states.** A cycle whose boards are all
WF and in-field has all states living in `inFieldStates cfg`. This is the
*conditional* upper-bound bridge: if we can establish WF of cycle boards
(currently an open obligation), the cycle's cardinality is bounded by
`|inFieldStates cfg|`, giving a closed search interval `28 ≤ card ≤ |U|`
together with the depth-28 lower bound. -/
theorem AdversarialClosedCycle.cycle_subset_inFieldStates_of_wf {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg)
    (hwf : ∀ s ∈ C.states, Board.WF cfg s.board) :
    C.states ⊆ inFieldStates cfg := by
  intro s hs
  unfold inFieldStates
  rw [Finset.mem_image]
  let b : InFieldBoard cfg := ⟨s.board, hwf s hs, C.cycle_state_in_field hs⟩
  refine ⟨(b, s.bag), Finset.mem_univ _, ?_⟩
  rfl

/-- **WF-strengthened cycle** — same data as `AdversarialClosedCycle` with the
additional invariant that every state's board is well-formed. Discharges the
hypothesis of `cycle_subset_inFieldStates_of_wf` unconditionally, yielding
the absolute upper bound `C.states.card ≤ |inFieldStates cfg|`. -/
structure AdversarialClosedCycleWF (cfg : GameConfig)
    extends AdversarialClosedCycle cfg where
  wf : ∀ s ∈ toAdversarialClosedCycle.states, Board.WF cfg s.board

/-- Lift an existing `AdversarialClosedCycle` to a `WF` cycle by supplying
the missing WF invariant. -/
def AdversarialClosedCycle.withWF {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hwf : ∀ s ∈ C.states, Board.WF cfg s.board) : AdversarialClosedCycleWF cfg where
  toAdversarialClosedCycle := C
  wf := hwf

namespace AdversarialClosedCycleWF

/-- Direct upper bound: the WF cycle is contained in `inFieldStates`. -/
theorem subset_inFieldStates {cfg : GameConfig}
    (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states ⊆ inFieldStates cfg :=
  C.toAdversarialClosedCycle.cycle_subset_inFieldStates_of_wf C.wf

/-- Closed-form upper bound: card ≤ |inFieldStates cfg|. -/
theorem states_card_le {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  Finset.card_le_card C.subset_inFieldStates

/-- WF cycle with `init`: `28 ≤ states.card ≤ |inFieldStates cfg|`. -/
theorem init_states_card_in_28_inFieldStates {cfg : GameConfig}
    (C : AdversarialClosedCycleWF cfg) (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  ⟨C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight h, C.states_card_le⟩

/-- WF cycle: `states.card ≤ 2 ^ (cfg.cols * cfg.rows + 7)`. -/
theorem states_card_le_two_pow {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states.card ≤ 2 ^ (cfg.cols * cfg.rows + 7) := by
  rw [← inFieldStates_card_eq_two_pow]
  exact C.states_card_le

/-- WF init-cycle: `28 ≤ states.card ≤ 2 ^ (cfg.cols * cfg.rows + 7)`. -/
theorem init_states_card_in_28_two_pow {cfg : GameConfig}
    (C : AdversarialClosedCycleWF cfg)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 2 ^ (cfg.cols * cfg.rows + 7) :=
  ⟨C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight h,
    C.states_card_le_two_pow⟩

/-- WF init-cycle on tiny: `28 ≤ states.card ≤ 8388608`. -/
theorem init_states_card_in_28_tiny
    (C : AdversarialClosedCycleWF GameConfig.tiny)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 8388608 := by
  have := C.init_states_card_in_28_inFieldStates h
  rw [tiny_inFieldStates_card] at this
  exact this

/-- WF init-cycle on standard: `28 ≤ states.card ≤ 2^200 * 128`. -/
theorem init_states_card_in_28_standard
    (C : AdversarialClosedCycleWF GameConfig.standard)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 2 ^ 200 * 128 := by
  have := C.init_states_card_in_28_inFieldStates h
  rw [standard_inFieldStates_card] at this
  exact this

/-- WF init-cycle on standard, `2^207` form: `28 ≤ states.card ≤ 2^207`. -/
theorem init_states_card_in_28_two_pow_207_standard
    (C : AdversarialClosedCycleWF GameConfig.standard)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 2 ^ 207 := by
  have := C.init_states_card_in_28_inFieldStates h
  rw [standard_inFieldStates_card_eq_two_pow_207] at this
  exact this

/-- WF init-cycle on tiny, `2^23` form: `28 ≤ states.card ≤ 2^23`. -/
theorem init_states_card_in_28_two_pow_23_tiny
    (C : AdversarialClosedCycleWF GameConfig.tiny)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 2 ^ 23 := by
  have := C.init_states_card_in_28_inFieldStates h
  rw [tiny_inFieldStates_card_eq_two_pow_23] at this
  exact this

/-- Decidable WF check on a Finset of game states. -/
instance (cfg : GameConfig) (states : Finset GameState) :
    Decidable (∀ s ∈ states, Board.WF cfg s.board) := by
  unfold Board.WF; infer_instance

/-- Bundled WF obligations: the standard cycle obligations *plus* WF on
every state's board. Single decidable predicate for searching WF cycles. -/
def Obligations (cfg : GameConfig) (states : Finset GameState)
    (solver : Solver cfg) : Prop :=
  AdversarialClosedCycle.Obligations cfg states solver ∧
  ∀ s ∈ states, Board.WF cfg s.board

instance (cfg : GameConfig) (states : Finset GameState) (solver : Solver cfg) :
    Decidable (Obligations cfg states solver) := by
  unfold Obligations; infer_instance

/-- Smart constructor for `AdversarialClosedCycleWF` from the decidable
bundled obligations. -/
def mkChecked {cfg : GameConfig} (states : Finset GameState) (solver : Solver cfg)
    (h : Obligations cfg states solver) : AdversarialClosedCycleWF cfg where
  toAdversarialClosedCycle := AdversarialClosedCycle.mkChecked states solver h.1
  wf := h.2

/-- Combined search target for WF cycles through `init`. Single decidable
predicate whose witness implies `TetrisSolvableFor cfg`. -/
def IsInitCycle (cfg : GameConfig) (states : Finset GameState)
    (solver : Solver cfg) : Prop :=
  Obligations cfg states solver ∧ GameState.init ∈ states

instance (cfg : GameConfig) (states : Finset GameState) (solver : Solver cfg) :
    Decidable (IsInitCycle cfg states solver) := by
  unfold IsInitCycle; infer_instance

/-- **WF one-shot reduction**: a decidable WF-cycle witness through `init`
implies `TetrisSolvableFor cfg`. The WF version of
`AdversarialClosedCycle.solvable_of_isInitCycle`. -/
theorem solvable_of_isInitCycle {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : IsInitCycle cfg states solver) :
    TetrisSolvableFor cfg :=
  (mkChecked states solver h.1).toAdversarialClosedCycle.init_solves_for h.2

@[simp] theorem mkChecked_states {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver) :
    (mkChecked states solver h).toAdversarialClosedCycle.states = states := rfl

@[simp] theorem mkChecked_solver {cfg : GameConfig} {states : Finset GameState}
    {solver : Solver cfg} (h : Obligations cfg states solver) :
    (mkChecked states solver h).toAdversarialClosedCycle.solver = solver := rfl

/-- Cycle states are *fully* in-field for a WF cycle: both column and row
bounds. -/
theorem cycle_state_inField {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg)
    {s : GameState} (hs : s ∈ C.toAdversarialClosedCycle.states) :
    (∀ p ∈ s.board, p.1 < cfg.cols) ∧ (∀ p ∈ s.board, p.2 < cfg.rows) :=
  ⟨C.wf s hs, C.toAdversarialClosedCycle.cycle_state_in_field hs⟩

/-- **The cycle-size search interval**: combining Tick 64's lower bound
with Tick 76's upper bound, any WF cycle through `init` satisfies
`28 ≤ |states| ≤ |inFieldStates|`. -/
theorem init_states_card_interval {cfg : GameConfig}
    (C : AdversarialClosedCycleWF cfg)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
    C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  ⟨C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight h, C.states_card_le⟩

end AdversarialClosedCycleWF

/-- **Reachability-strengthened cycle.** A cycle where every state is
`Reachable` from `init` (i.e. lives in the actual game's forward closure).
This automatically discharges WF via `Reachable.wf` (Tick 87) — cycle
authors need only prove reachability of each state. -/
structure ReachableClosedCycle (cfg : GameConfig)
    extends AdversarialClosedCycle cfg where
  reachable : ∀ s ∈ toAdversarialClosedCycle.states, Reachable cfg s

namespace ReachableClosedCycle

/-- A reachable cycle is automatically a WF cycle. -/
def toWF {cfg : GameConfig} (C : ReachableClosedCycle cfg) :
    AdversarialClosedCycleWF cfg where
  toAdversarialClosedCycle := C.toAdversarialClosedCycle
  wf := fun s hs => (C.reachable s hs).wf

/-- The cycle is contained in `inFieldStates`, hence its cardinality is
bounded by `|inFieldStates|`. -/
theorem states_card_le {cfg : GameConfig} (C : ReachableClosedCycle cfg) :
    C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  C.toWF.states_card_le

/-- A reachable init-cycle implies `TetrisSolvableFor cfg`. The cleanest
authoring path: define a `ReachableClosedCycle` and apply this. -/
theorem init_solves {cfg : GameConfig} (C : ReachableClosedCycle cfg)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    TetrisSolvableFor cfg :=
  C.toAdversarialClosedCycle.init_solves_for h

/-- **Full cycle-size search interval** for reachable init-cycles. -/
theorem init_states_card_interval {cfg : GameConfig}
    (C : ReachableClosedCycle cfg)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    28 ≤ C.toAdversarialClosedCycle.states.card ∧
    C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card :=
  ⟨C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight h, C.states_card_le⟩

end ReachableClosedCycle

/-- Lift an existing `AdversarialClosedCycle` to a `Reachable` cycle by
supplying the missing reachability invariant. The WF invariant is then
auto-derived via `Reachable.wf`. -/
def AdversarialClosedCycle.withReachable {cfg : GameConfig}
    (C : AdversarialClosedCycle cfg)
    (hr : ∀ s ∈ C.states, Reachable cfg s) : ReachableClosedCycle cfg where
  toAdversarialClosedCycle := C
  reachable := hr

/-- **The cycle from safety**: `init ∈ safe → ReachableClosedCycle`.
Constructed from `safeReachableCycle` with `safeSolver` as the solver.
The closure obligation chains `Reachable.step` and `safeSolver_spec` to
keep successors in `safe ∩ Reachable ∩ inFieldStates`. -/
noncomputable def safeReachableClosedCycle {cfg : GameConfig}
    (_h : GameState.init ∈ safe cfg) : ReachableClosedCycle cfg where
  toAdversarialClosedCycle :=
    { states := safeReachableCycle cfg
      solver := safeSolver cfg
      not_lost := fun s hs => by
        rw [mem_safeReachableCycle] at hs
        exact safe_not_lost hs.2.2
      valid := fun s hs p hp => by
        rw [mem_safeReachableCycle] at hs
        exact ⟨(safeSolver_spec hs.2.2 hp).1, (safeSolver_spec hs.2.2 hp).2.1⟩
      closed := fun s hs p hp => by
        rw [mem_safeReachableCycle] at hs
        obtain ⟨_, hr, hsafe⟩ := hs
        obtain ⟨hpiece, hvalid, hstep_safe⟩ := safeSolver_spec hsafe hp
        have heta : ({safeSolver cfg s p with piece := p} : Placement)
            = safeSolver cfg s p := Placement.eta_of_piece_eq hpiece
        have hstep_eq : adversarialStep cfg s p (safeSolver cfg s p)
            = s.step cfg (safeSolver cfg s p) := by
          rw [adversarialStep_eq_step, heta]
        have hstep_reach :
            Reachable cfg (s.step cfg (safeSolver cfg s p)) := by
          refine Reachable.step hr ?_ hvalid (safe_not_lost hsafe)
          rw [hpiece]; exact hp
        rw [mem_safeReachableCycle, hstep_eq]
        refine ⟨?_, hstep_reach, ?_⟩
        · exact reachable_safe_mem_inFieldStates hstep_reach
            (by rw [← hstep_eq]; exact hstep_safe)
        · rw [← hstep_eq]; exact hstep_safe }
  reachable := fun s hs => by
    rw [mem_safeReachableCycle] at hs
    exact hs.2.1

@[simp] theorem safeReachableClosedCycle_states {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.states =
      safeReachableCycle cfg := rfl

/-- The constructed cycle from safety contains `init`. -/
theorem init_mem_safeReachableClosedCycle {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    GameState.init ∈
      (safeReachableClosedCycle h).toAdversarialClosedCycle.states := by
  rw [safeReachableClosedCycle_states]
  exact init_mem_safeReachableCycle h

/-- **The WF-strengthened cycle from safety.** Composes `safeReachableClosedCycle`
with `ReachableClosedCycle.toWF` to lift a `safe`-witness directly into an
`AdversarialClosedCycleWF` — every cycle constructed from `init ∈ safe cfg`
automatically carries the WF invariant via `Reachable.wf`. Connects the
safe-set theory (Tick 1500 family) to the WF-cycle theory (1910+) without
manual WF discharge. -/
noncomputable def safeReachableClosedCycleWF {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) : AdversarialClosedCycleWF cfg :=
  (safeReachableClosedCycle h).toWF

/-- The WF-strengthened cycle from safety contains `init`. -/
theorem init_mem_safeReachableClosedCycleWF {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    GameState.init ∈
      (safeReachableClosedCycleWF h).toAdversarialClosedCycle.states :=
  init_mem_safeReachableClosedCycle h

/-- **Existence of an init-containing WF cycle from safety.** Direct
existential form of `safeReachableClosedCycleWF`. -/
theorem exists_init_AdversarialClosedCycleWF_of_init_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states :=
  ⟨safeReachableClosedCycleWF h, init_mem_safeReachableClosedCycleWF h⟩

/-- **Headline WF-cycle Iff**: `init ∈ safe cfg ↔ ∃ init-containing
WF cycle`. WF-strengthened counterpart of
`init_safe_iff_exists_init_adversarialClosedCycle`. Forward direction is
Tick 1583's existential; reverse uses the embedded `AdversarialClosedCycle`
together with the standard `init_safe_of_init_mem`. Strengthens the
M2/M3 bridge: not only does `init ∈ safe` produce *some* cycle, it
produces a WF cycle (with all the structural goodies of
`AdversarialClosedCycleWF` — card bound by `|inFieldStates cfg|`, etc.). -/
theorem init_safe_iff_exists_init_adversarialClosedCycleWF (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
      ∃ C : AdversarialClosedCycleWF cfg,
        GameState.init ∈ C.toAdversarialClosedCycle.states :=
  ⟨exists_init_AdversarialClosedCycleWF_of_init_safe,
   fun ⟨C, hinit⟩ => C.toAdversarialClosedCycle.init_safe_of_init_mem hinit⟩

/-- **Solvability ↔ init-containing WF cycle.** Composes
`tetrisSolvableValidFor_iff_init_safe` with the WF-strengthened headline
Iff (Tick 1584). For configs with `4 ≤ cfg.cols`,
`TetrisSolvableValidFor cfg` is *equivalent* to the existence of an
init-containing **WF** cycle — strictly stronger than the existing
`tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle`. Means:
to prove solvability, it suffices to exhibit a WF cycle witness; the
full `AdversarialClosedCycleWF` API ships with the witness. -/
theorem tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycleWF
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
      ∃ C : AdversarialClosedCycleWF cfg,
        GameState.init ∈ C.toAdversarialClosedCycle.states := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      init_safe_iff_exists_init_adversarialClosedCycleWF]

/-- **Negated WF-cycle Iff**: `¬ TetrisSolvableValidFor` ↔ *no* WF cycle
contains init. Contrapositive of Tick 1585 via `not_exists`. This is the
**universal-quantification** form of unsolvability — to prove a config
is unsolvable, one must show *every* WF cycle excludes init. -/
theorem not_tetrisSolvableValidFor_iff_forall_no_init_adversarialClosedCycleWF
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ¬ TetrisSolvableValidFor cfg ↔
      ∀ C : AdversarialClosedCycleWF cfg,
        GameState.init ∉ C.toAdversarialClosedCycle.states := by
  rw [tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycleWF hcols,
      not_exists]

/-- **Cardinality envelope from safety.** From `init ∈ safe cfg` we directly
extract a concrete bound on the canonical WF cycle:
`28 ≤ card ≤ |inFieldStates cfg|`. Combines Tick 1583's
`safeReachableClosedCycleWF` with the existing
`AdversarialClosedCycleWF.init_states_card_in_28_inFieldStates` envelope.
Promises the M2/M3 search space is non-trivially bounded the moment safety
is established. -/
theorem safe_init_cycle_card_envelope {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableClosedCycleWF h).toAdversarialClosedCycle.states.card ∧
    (safeReachableClosedCycleWF h).toAdversarialClosedCycle.states.card
      ≤ (inFieldStates cfg).card :=
  (safeReachableClosedCycleWF h).init_states_card_in_28_inFieldStates
    (init_mem_safeReachableClosedCycleWF h)

/-- **Solvability gives a bounded init-cycle.** For configs with
`4 ≤ cfg.cols`, `TetrisSolvableValidFor cfg` directly produces a WF cycle
through `init` whose card is sandwiched in `[28, |inFieldStates cfg|]`.
Composes `tetrisSolvableValidFor_iff_init_safe` with Tick 1583/1587. The
solvability premise is the entry point most external consumers have; this
strips out the `safe`-set intermediate. -/
theorem tetrisSolvableValidFor_gives_cycle_card_envelope
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (hsol : TetrisSolvableValidFor cfg) :
    ∃ C : AdversarialClosedCycleWF cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card := by
  have hsafe := (tetrisSolvableValidFor_iff_init_safe hcols).mp hsol
  refine ⟨safeReachableClosedCycleWF hsafe,
          init_mem_safeReachableClosedCycleWF hsafe, ?_⟩
  exact safe_init_cycle_card_envelope hsafe

/-- **Distinct-boards bound for WF cycles.** The image of `GameState.board`
across a WF cycle's states is contained in the InFieldBoard-projected
image of `Finset.univ : Finset (InFieldBoard cfg)`. Constructive: each
`g ∈ C.states` carries WF (from the cycle's WF invariant) and an in-field
row bound (from `C.not_lost` via `not_lost_iff_forall_row_lt`), packaging
into an `InFieldBoard` witness. -/
theorem AdversarialClosedCycleWF.image_board_subset_inField_image
    {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states.image GameState.board ⊆
      (Finset.univ : Finset (InFieldBoard cfg)).image (fun b => b.val) := by
  intro b hb
  rw [Finset.mem_image] at hb
  obtain ⟨g, hg, hgb⟩ := hb
  rw [Finset.mem_image]
  refine ⟨⟨g.board, C.wf g hg, ?_⟩, Finset.mem_univ _, hgb⟩
  exact (GameState.not_lost_iff_forall_row_lt cfg g).mp
    (C.toAdversarialClosedCycle.not_lost g hg)

/-- **Distinct-boards card bound**: a WF cycle's states have at most
`2 ^ (cfg.cols * cfg.rows)` distinct boards. Sharper than the trivial
`|image board| ≤ |C.states|` bound (which itself is ≤ `|inFieldStates cfg|
= 128 * 2^(cols*rows)`): the board-projection drops the factor of 128
(bag values). Composes the in-field subset (above) with
`InFieldBoard.fintype_card`. -/
theorem AdversarialClosedCycleWF.image_board_card_le_two_pow
    {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    (C.toAdversarialClosedCycle.states.image GameState.board).card ≤
      2 ^ (cfg.cols * cfg.rows) := by
  calc (C.toAdversarialClosedCycle.states.image GameState.board).card
      ≤ ((Finset.univ : Finset (InFieldBoard cfg)).image (fun b => b.val)).card :=
        Finset.card_le_card C.image_board_subset_inField_image
    _ ≤ (Finset.univ : Finset (InFieldBoard cfg)).card := Finset.card_image_le
    _ = Fintype.card (InFieldBoard cfg) := Finset.card_univ
    _ = 2 ^ (cfg.cols * cfg.rows) := InFieldBoard.fintype_card

/-- **Distinct-bags card bound for WF cycles** (and for any cycle, really —
the WF wrapper is not used): the image of `GameState.bag` across cycle
states has at most `128 = 2^7` elements. Since `Bag := Finset Piece` and
`|Piece| = 7`, the entire type has only 128 inhabitants. Combined with
Tick 1589's board bound, gives a complete factorisation
`|C.states| ≤ |image board| * |image bag| ≤ 2^(cols*rows) * 128`. -/
theorem AdversarialClosedCycle.image_bag_card_le_128
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    (C.states.image GameState.bag).card ≤ 128 := by
  calc (C.states.image GameState.bag).card
      ≤ (Finset.univ : Finset Bag).card :=
        Finset.card_le_card (Finset.subset_univ _)
    _ = Fintype.card Bag := Finset.card_univ
    _ = 2 ^ Fintype.card Piece := Fintype.card_finset
    _ = 128 := by decide

/-- **Product upper bound on cycle state count.** Since each `GameState` is
uniquely determined by its `(board, bag)` pair, a cycle's state count is at
most the product of its distinct-board count and its distinct-bag count.
Proof: inject `g ↦ (g.board, g.bag)` from `C.states` into
`(image board) ×ˢ (image bag)`, using `Finset.card_le_card_of_injOn`. -/
theorem AdversarialClosedCycle.states_card_le_image_board_mul_image_bag
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    C.states.card ≤
      (C.states.image GameState.board).card *
      (C.states.image GameState.bag).card := by
  have hmap : ∀ g ∈ C.states,
      (g.board, g.bag) ∈
        (C.states.image GameState.board) ×ˢ (C.states.image GameState.bag) := by
    intro g hg
    rw [Finset.mem_product]
    exact ⟨Finset.mem_image.mpr ⟨g, hg, rfl⟩,
           Finset.mem_image.mpr ⟨g, hg, rfl⟩⟩
  have hinj : Set.InjOn (fun g : GameState => (g.board, g.bag)) C.states := by
    intro g₁ _ g₂ _ heq
    obtain ⟨hb, hbag⟩ := Prod.mk.inj heq
    cases g₁; cases g₂
    simp_all
  calc C.states.card
      ≤ ((C.states.image GameState.board) ×ˢ
         (C.states.image GameState.bag)).card :=
        Finset.card_le_card_of_injOn _ hmap hinj
    _ = (C.states.image GameState.board).card *
        (C.states.image GameState.bag).card :=
        Finset.card_product _ _

/-- **Init-cycle product envelope.** Bundles the universal lower bound
`28 ≤ |C.states|` (existing) with Tick 1591's product upper bound, giving
a tight closed-form sandwich for any cycle through `init`:
`28 ≤ |C.states| ≤ |image board| * |image bag|`. The upper side is now
*structurally tight* — equal to `|image board| * |image bag|` whenever
every (board, bag) combination in the projections is realised. -/
theorem AdversarialClosedCycle.init_states_card_envelope_via_image_product
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    28 ≤ C.states.card ∧
    C.states.card ≤
      (C.states.image GameState.board).card *
      (C.states.image GameState.bag).card :=
  ⟨C.init_states_card_ge_twenty_eight h,
   C.states_card_le_image_board_mul_image_bag⟩

/-- **Single-board cycle has bounded card.** If a cycle uses only one
distinct board across all its states, its total state count is at most 128
(the bag-count bound). Composes Tick 1591 (product bound) at
`|image board| = 1` with Tick 1590 (`|image bag| ≤ 128`). -/
theorem AdversarialClosedCycle.states_card_le_128_of_single_board
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hsingle : (C.states.image GameState.board).card = 1) :
    C.states.card ≤ 128 := by
  have := C.states_card_le_image_board_mul_image_bag
  rw [hsingle, one_mul] at this
  exact le_trans this C.image_bag_card_le_128

/-- **Contrapositive: large cycles use multiple boards.** If a cycle has
more than 128 states, it must use at least 2 distinct boards. Direct
contrapositive of `states_card_le_128_of_single_board`. -/
theorem AdversarialClosedCycle.image_board_card_ge_two_of_states_card_gt_128
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (hlarge : 128 < C.states.card) :
    2 ≤ (C.states.image GameState.board).card := by
  by_contra h
  have h1 : (C.states.image GameState.board).card ≤ 1 :=
    Nat.lt_succ_iff.mp (Nat.not_le.mp h)
  interval_cases hboard : (C.states.image GameState.board).card
  · -- card = 0 → image empty → states empty → 0 ≤ 128 < |states| contradicts
    have : (C.states.image GameState.board) = ∅ := Finset.card_eq_zero.mp hboard
    have : C.states = ∅ := by
      apply Finset.eq_empty_of_forall_notMem
      intro g hg
      apply Finset.notMem_empty g.board
      rw [← this]
      exact Finset.mem_image.mpr ⟨g, hg, rfl⟩
    rw [this] at hlarge
    simp at hlarge
  · have := C.states_card_le_128_of_single_board hboard
    omega

/-- **States ≤ image-board × 128.** Specialisation of Tick 1591's product
bound that "pre-applies" the universal bag-count bound 128. Cleaner form
when one only wants to reason about the board projection. -/
theorem AdversarialClosedCycle.states_card_le_image_board_mul_128
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    C.states.card ≤ (C.states.image GameState.board).card * 128 := by
  calc C.states.card
      ≤ (C.states.image GameState.board).card *
        (C.states.image GameState.bag).card :=
        C.states_card_le_image_board_mul_image_bag
    _ ≤ (C.states.image GameState.board).card * 128 :=
        Nat.mul_le_mul_left _ C.image_bag_card_le_128

/-- **Floor-form board-diversity lower bound.** From the product bound,
`|image board| ≥ ⌊|C.states| / 128⌋`. Strictly stronger than the
"single-board → small" specialisation: any cycle is forced to use at
least `⌊|C.states| / 128⌋` distinct boards. -/
theorem AdversarialClosedCycle.image_board_card_ge_states_card_div_128
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) :
    C.states.card / 128 ≤ (C.states.image GameState.board).card := by
  have h1 := C.states_card_le_image_board_mul_128
  calc C.states.card / 128
      ≤ ((C.states.image GameState.board).card * 128) / 128 :=
        Nat.div_le_div_right h1
    _ = (C.states.image GameState.board).card :=
        Nat.mul_div_cancel _ (by norm_num)

/-- **WF-cycle states ≤ 2^(cols*rows) × |image bag|.** Companion of Tick
1594's board-form bound, requiring WF (via Tick 1589's
`image_board_card_le_two_pow`). Reading: the cycle's state count is
bounded by the product of the maximum possible distinct boards
(`2^(cols*rows)`) and the actual distinct-bag count. -/
theorem AdversarialClosedCycleWF.states_card_le_two_pow_mul_image_bag
    {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states.card ≤
      2 ^ (cfg.cols * cfg.rows) *
      (C.toAdversarialClosedCycle.states.image GameState.bag).card := by
  calc C.toAdversarialClosedCycle.states.card
      ≤ (C.toAdversarialClosedCycle.states.image GameState.board).card *
        (C.toAdversarialClosedCycle.states.image GameState.bag).card :=
        C.toAdversarialClosedCycle.states_card_le_image_board_mul_image_bag
    _ ≤ 2 ^ (cfg.cols * cfg.rows) *
        (C.toAdversarialClosedCycle.states.image GameState.bag).card :=
        Nat.mul_le_mul_right _ C.image_board_card_le_two_pow

/-- **Floor-form bag-diversity lower bound.** For WF cycles, the
distinct-bag count is at least `⌊|C.states| / 2^(cols*rows)⌋`. Divides
both sides of `states_card_le_two_pow_mul_image_bag` by
`2^(cfg.cols * cfg.rows)`. Symmetric counterpart to Tick 1594's
`image_board_card_ge_states_card_div_128`. -/
theorem AdversarialClosedCycleWF.image_bag_card_ge_states_card_div_two_pow
    {cfg : GameConfig} (C : AdversarialClosedCycleWF cfg) :
    C.toAdversarialClosedCycle.states.card / (2 ^ (cfg.cols * cfg.rows)) ≤
      (C.toAdversarialClosedCycle.states.image GameState.bag).card := by
  have h1 := C.states_card_le_two_pow_mul_image_bag
  have hpos : 0 < 2 ^ (cfg.cols * cfg.rows) := Nat.two_pow_pos _
  calc C.toAdversarialClosedCycle.states.card / (2 ^ (cfg.cols * cfg.rows))
      ≤ (2 ^ (cfg.cols * cfg.rows) *
         (C.toAdversarialClosedCycle.states.image GameState.bag).card) /
        (2 ^ (cfg.cols * cfg.rows)) :=
        Nat.div_le_div_right h1
    _ = (C.toAdversarialClosedCycle.states.image GameState.bag).card :=
        Nat.mul_div_cancel_left _ hpos

/-- **Generic projection-bound contradiction lemma.** From any pair of
upper bounds on the marginal projections, derive a product bound on the
state count. Schematic form of Ticks 1593, 1594, 1595 — single rule that
specialises to all of them by choosing `k1`, `k2`. Use case: rule out
candidate cycles by exhibiting any pair of marginal bounds whose product
contradicts the lower bound on `|C.states|`. -/
theorem AdversarialClosedCycle.states_card_le_of_image_bounds
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) {k1 k2 : ℕ}
    (hb : (C.states.image GameState.board).card ≤ k1)
    (hg : (C.states.image GameState.bag).card ≤ k2) :
    C.states.card ≤ k1 * k2 :=
  le_trans C.states_card_le_image_board_mul_image_bag (Nat.mul_le_mul hb hg)

/-- **Projection product ≥ 28 for init cycles.** Composes
`init_states_card_ge_twenty_eight` with Tick 1591 — for any cycle through
`init`, the product of its two marginal projection cardinalities is at
least 28. This is the *projection-form* of the universal init-cycle lower
bound. -/
theorem AdversarialClosedCycle.init_image_board_mul_image_bag_ge_28
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    28 ≤ (C.states.image GameState.board).card *
         (C.states.image GameState.bag).card :=
  le_trans (C.init_states_card_ge_twenty_eight h)
    C.states_card_le_image_board_mul_image_bag

/-- **Init non-membership from small projections.** If a cycle's two
marginal projection cardinalities multiply to less than 28, then `init`
cannot be in the cycle. Concrete non-existence rule — given any pair of
projection bounds whose product is `< 28`, the cycle is immediately
disqualified as a candidate init-cycle. -/
theorem AdversarialClosedCycle.init_not_mem_of_image_bounds_lt_28
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg) {k1 k2 : ℕ}
    (hb : (C.states.image GameState.board).card ≤ k1)
    (hg : (C.states.image GameState.bag).card ≤ k2)
    (hk : k1 * k2 < 28) :
    GameState.init ∉ C.states := by
  intro h
  have h1 := C.init_states_card_ge_twenty_eight h
  have h2 := C.states_card_le_of_image_bounds hb hg
  omega

/-- **Single-board init-cycle has ≥ 28 distinct bags.** If an init-cycle
uses only a single board configuration across all states (or fewer), the
bag-projection must carry the entire diversity — at least 28 distinct bags.
Specialises Tick 1597's `init_image_board_mul_image_bag_ge_28` at
`|image board| ≤ 1`. -/
theorem AdversarialClosedCycle.init_image_bag_card_ge_28_of_image_board_card_le_one
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    (hb : (C.states.image GameState.board).card ≤ 1) :
    28 ≤ (C.states.image GameState.bag).card := by
  have hprod := C.init_image_board_mul_image_bag_ge_28 h
  have hineq :
      (C.states.image GameState.board).card *
        (C.states.image GameState.bag).card ≤
      1 * (C.states.image GameState.bag).card :=
    Nat.mul_le_mul_right _ hb
  omega

/-- **Init-cycle bag-diversity unconditional lower bound**:
`|image bag| ≥ 2` for any cycle through `init`. Proof: `init.bag = Bag.full`
(card 7) is in the image, and the closed-under-step successor of `init`
under any piece (use `Piece.O`) has bag `Bag.full.draw p` (card 6, by
`Bag.draw_full_ne_full`) — a distinct element of the image. So
`Finset.one_lt_card.mpr` gives 2 ≤ card. *Asymmetric* to the
`|image board| ≥ 2` question, which is false in general (line-clearing
can return to empty board). -/
theorem AdversarialClosedCycle.init_image_bag_card_ge_two
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    2 ≤ (C.states.image GameState.bag).card := by
  set p : Piece := Piece.O with hp_def
  have hp : p ∈ GameState.init.bag := by
    rw [GameState.init_bag]; exact Bag.mem_full _
  set g' := adversarialStep cfg GameState.init p (C.solver GameState.init p)
      with hg'_def
  have hg' : g' ∈ C.states := C.closed _ h p hp
  have hg'_bag : g'.bag = (Bag.full : Bag).draw p := by
    change (adversarialStep cfg GameState.init p (C.solver GameState.init p)).bag = _
    rw [adversarialStep_bag, GameState.init_bag]
  have hbag_ne : g'.bag ≠ GameState.init.bag := by
    rw [hg'_bag, GameState.init_bag]
    exact Bag.draw_full_ne_full p
  apply Finset.one_lt_card.mpr
  refine ⟨GameState.init.bag, ?_, g'.bag, ?_, hbag_ne.symm⟩
  · exact Finset.mem_image.mpr ⟨GameState.init, h, rfl⟩
  · exact Finset.mem_image.mpr ⟨g', hg', rfl⟩

/-- **Minimum-bag init-cycle needs ≥ 14 boards.** Combining Tick 1599's
unconditional `|image bag| ≥ 2` with the product constraint
`|image board| * |image bag| ≥ 28`: if a cycle saturates the bag minimum
(`|image bag| ≤ 2`, hence `= 2`), then the board projection must carry the
rest of the 28-state lower bound, needing at least 14 distinct boards.
The bag-minimum case is the *real* extremal scenario (per the asymmetry
noted at Tick 1599 — `|image bag| = 1` is impossible). -/
theorem AdversarialClosedCycle.init_image_board_card_ge_14_of_image_bag_card_le_two
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    (hg : (C.states.image GameState.bag).card ≤ 2) :
    14 ≤ (C.states.image GameState.board).card := by
  have hprod := C.init_image_board_mul_image_bag_ge_28 h
  have hineq :
      (C.states.image GameState.board).card *
        (C.states.image GameState.bag).card ≤
      (C.states.image GameState.board).card * 2 :=
    Nat.mul_le_mul_left _ hg
  omega

/-- **The full positive equivalence**: an `IsInitCycle` witness exists
iff `init ∈ safe cfg`. The "discovery" target for atlas construction. -/
theorem exists_isInitCycle_iff_init_safe (cfg : GameConfig) :
    (∃ states : Finset GameState, ∃ solver : Solver cfg,
      AdversarialClosedCycle.IsInitCycle cfg states solver) ↔
    GameState.init ∈ safe cfg := by
  constructor
  · rintro ⟨states, solver, h⟩
    exact (AdversarialClosedCycle.mkChecked states solver h.1).subset_safe
      (Finset.mem_coe.mpr h.2)
  · intro h
    let C := safeReachableClosedCycle h
    exact ⟨C.toAdversarialClosedCycle.states, C.toAdversarialClosedCycle.solver,
           C.toAdversarialClosedCycle.isInitCycle_of
             (init_mem_safeReachableClosedCycle h)⟩

/-- Negation form: `init ∉ safe` iff no `IsInitCycle` exists. -/
theorem not_exists_isInitCycle_iff_init_not_safe (cfg : GameConfig) :
    (¬ ∃ states : Finset GameState, ∃ solver : Solver cfg,
      AdversarialClosedCycle.IsInitCycle cfg states solver) ↔
    GameState.init ∉ safe cfg :=
  not_congr (exists_isInitCycle_iff_init_safe cfg)

/-- Negation form: `init ∉ safe` iff no `Obligations + init ∈ states`
exists. Unfolds `IsInitCycle = Obligations ∧ init ∈ states` so the
witness structure is exposed. -/
theorem not_exists_obligations_init_mem_iff_init_not_safe (cfg : GameConfig) :
    (¬ ∃ states : Finset GameState, ∃ solver : Solver cfg,
      AdversarialClosedCycle.Obligations cfg states solver ∧
      GameState.init ∈ states) ↔
    GameState.init ∉ safe cfg :=
  not_exists_isInitCycle_iff_init_not_safe cfg

/-- Positive form: `init ∈ safe cfg` iff there exists an
`Obligations + init ∈ states` witness. Unfolds `IsInitCycle`. -/
theorem exists_obligations_init_mem_iff_init_safe (cfg : GameConfig) :
    (∃ states : Finset GameState, ∃ solver : Solver cfg,
      AdversarialClosedCycle.Obligations cfg states solver ∧
      GameState.init ∈ states) ↔
    GameState.init ∈ safe cfg :=
  exists_isInitCycle_iff_init_safe cfg

/-- **THE MASTER POSITIVE IFF**: Tetris is `Valid`-solvable iff an
`IsInitCycle` candidate exists. Combines Tick 124 (safe ↔ solvable) with
Tick 195 (cycle ↔ safe). Atlas: this is the single iff for which a
constructive witness yields solvability. -/
theorem tetrisSolvableValidFor_iff_exists_isInitCycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
    ∃ states : Finset GameState, ∃ solver : Solver cfg,
      AdversarialClosedCycle.IsInitCycle cfg states solver := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      ← exists_isInitCycle_iff_init_safe]

/-- Standard specialisation of the master iff. -/
theorem TetrisSolvableValid_iff_exists_isInitCycle :
    TetrisSolvableValid ↔
    ∃ states : Finset GameState,
      ∃ solver : Solver GameConfig.standard,
        AdversarialClosedCycle.IsInitCycle GameConfig.standard states solver :=
  tetrisSolvableValidFor_iff_exists_isInitCycle (by decide)

/-- Tiny specialisation of the master iff. -/
theorem TetrisSolvableValidFor_tiny_iff_exists_isInitCycle :
    TetrisSolvableValidFor GameConfig.tiny ↔
    ∃ states : Finset GameState,
      ∃ solver : Solver GameConfig.tiny,
        AdversarialClosedCycle.IsInitCycle GameConfig.tiny states solver :=
  tetrisSolvableValidFor_iff_exists_isInitCycle (by decide)

/-- **Contrapositive of the master iff**: Tetris is unsolvable iff no
`IsInitCycle` witness exists for *any* `(states, solver)` pair. -/
theorem not_tetrisSolvableValidFor_iff_forall_not_isInitCycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    ¬ TetrisSolvableValidFor cfg ↔
    ∀ states : Finset GameState, ∀ solver : Solver cfg,
      ¬ AdversarialClosedCycle.IsInitCycle cfg states solver := by
  rw [tetrisSolvableValidFor_iff_exists_isInitCycle hcols]
  push Not
  rfl

/-- **The canonical IsInitCycle from safety**: when `init ∈ safe cfg`,
the pair `(safeReachableCycle cfg, safeSolver cfg)` constitutes an
`IsInitCycle`. -/
theorem safeReachableCycle_isInitCycle {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    AdversarialClosedCycle.IsInitCycle cfg (safeReachableCycle cfg)
      (safeSolver cfg) :=
  ⟨(safeReachableClosedCycle h).toAdversarialClosedCycle.obligations_of,
   init_mem_safeReachableCycle h⟩

/-- **Tight iff**: init is safe iff the canonical `(safeReachableCycle,
safeSolver)` pair is an init-cycle. -/
theorem init_safe_iff_safeReachableCycle_isInitCycle (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
    AdversarialClosedCycle.IsInitCycle cfg (safeReachableCycle cfg)
      (safeSolver cfg) := by
  refine ⟨safeReachableCycle_isInitCycle, ?_⟩
  intro h
  rw [← init_mem_safeReachableCycle_iff]
  exact h.2

/-- **Unconditional Obligations**: the canonical pair `(safeReachableCycle
cfg, safeSolver cfg)` *always* satisfies the cycle Obligations (for
cfg.cols ≥ 4), regardless of whether init is safe. -/
theorem safeReachableCycle_obligations {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    AdversarialClosedCycle.Obligations cfg (safeReachableCycle cfg)
      (safeSolver cfg) := by
  refine ⟨?_, ?_, ?_⟩
  · intro s hs
    exact safe_not_lost (safeReachableCycle_subset_safe cfg s hs)
  · intro s _ p hp
    exact ⟨safeSolver_piece cfg s p, (safeSolver_validSolver hcols s p hp).2⟩
  · intro _ hs _ hp
    exact safeReachableCycle_safeSolver_step hs hp

/-- **Canonical-pair solvability iff**: Tetris-valid-solvable iff the
canonical `(safeReachableCycle, safeSolver)` pair satisfies `IsInitCycle`.
The tightest characterization of solvability via a specific pair. -/
theorem tetrisSolvableValidFor_iff_canonical_isInitCycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
    AdversarialClosedCycle.IsInitCycle cfg (safeReachableCycle cfg)
      (safeSolver cfg) := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      init_safe_iff_safeReachableCycle_isInitCycle]

/-- The constructed cycle is nonempty. -/
theorem safeReachableClosedCycle_states_nonempty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ((safeReachableClosedCycle h).toAdversarialClosedCycle.states).Nonempty :=
  ⟨GameState.init, init_mem_safeReachableClosedCycle h⟩

/-- The constructed cycle has positive `states.card`. -/
theorem safeReachableClosedCycle_states_card_pos {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    0 < ((safeReachableClosedCycle h).toAdversarialClosedCycle.states).card :=
  Finset.card_pos.mpr (safeReachableClosedCycle_states_nonempty h)

/-- The constructed cycle has nonzero `states.card`. -/
theorem safeReachableClosedCycle_states_card_ne_zero {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ((safeReachableClosedCycle h).toAdversarialClosedCycle.states).card ≠ 0 :=
  Nat.ne_of_gt (safeReachableClosedCycle_states_card_pos h)

/-- The constructed cycle has `states.card ≥ 1`. -/
theorem safeReachableClosedCycle_states_one_le_card {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    1 ≤ ((safeReachableClosedCycle h).toAdversarialClosedCycle.states).card :=
  safeReachableClosedCycle_states_card_pos h

/-- The constructed cycle's `states` are not equal to `∅`. -/
theorem safeReachableClosedCycle_states_ne_empty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.states ≠ ∅ :=
  (safeReachableClosedCycle_states_nonempty h).ne_empty

/-- The constructed cycle's `states` are a subset of `inFieldStates cfg`. -/
theorem safeReachableClosedCycle_states_subset_inFieldStates {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.states ⊆
      inFieldStates cfg := by
  rw [safeReachableClosedCycle_states]
  exact safeReachableCycle_subset_inFieldStates cfg

/-- **If `init ∈ safe cfg`, the in-field iterate is never empty.** Chains
the cycle construction (`safeReachableClosedCycle`) with
`states_subset_safeIterFinite` and `Finset.subset_empty` — the constructed
nonempty cycle persists in every iterate from `inFieldStates`. -/
theorem safeIterFinite_inFieldStates_ne_empty_of_init_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (n : ℕ) :
    safeIterFinite cfg (inFieldStates cfg) n ≠ ∅ := by
  intro hempty
  have hne :=
    (safeReachableClosedCycle h).toAdversarialClosedCycle.states_eq_empty_of_safeIterFinite_eq_empty
      (safeReachableClosedCycle_states_subset_inFieldStates h) hempty
  exact safeReachableClosedCycle_states_ne_empty h hne

/-- **Contrapositive of Tick 1445**: if `safeIterFinite cfg (inFieldStates
cfg) n = ∅`, then `init ∉ safe cfg`. -/
theorem init_not_mem_safe_of_safeIterFinite_inFieldStates_eq_empty
    {cfg : GameConfig} {n : ℕ}
    (h : safeIterFinite cfg (inFieldStates cfg) n = ∅) :
    GameState.init ∉ safe cfg :=
  fun hsafe => safeIterFinite_inFieldStates_ne_empty_of_init_safe hsafe n h

/-- **Decidable unsolvability**: if the in-field iterate ever evacuates and
`4 ≤ cfg.cols`, then `¬ TetrisSolvableValidFor cfg`. Chains Tick 1446 with
`tetrisSolvableValidFor_iff_init_safe`. -/
theorem not_tetrisSolvableValidFor_of_safeIterFinite_inFieldStates_eq_empty
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {n : ℕ}
    (h : safeIterFinite cfg (inFieldStates cfg) n = ∅) :
    ¬ TetrisSolvableValidFor cfg := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols]
  exact init_not_mem_safe_of_safeIterFinite_inFieldStates_eq_empty h

/-- **Standard-config decidable unsolvability**: if the in-field iterate
ever evacuates on `GameConfig.standard`, then `¬ TetrisSolvableValid`. -/
theorem not_tetrisSolvableValid_of_safeIterFinite_standard_inFieldStates_eq_empty
    {n : ℕ}
    (h : safeIterFinite GameConfig.standard
        (inFieldStates GameConfig.standard) n = ∅) :
    ¬ TetrisSolvableValid :=
  not_tetrisSolvableValidFor_of_safeIterFinite_inFieldStates_eq_empty (by decide) h

/-- **Tiny-config decidable unsolvability**: if the in-field iterate ever
evacuates on `GameConfig.tiny`, then `¬ TetrisSolvableValidFor tiny`. -/
theorem not_tetrisSolvableValidFor_tiny_of_safeIterFinite_inFieldStates_eq_empty
    {n : ℕ}
    (h : safeIterFinite GameConfig.tiny
        (inFieldStates GameConfig.tiny) n = ∅) :
    ¬ TetrisSolvableValidFor GameConfig.tiny :=
  not_tetrisSolvableValidFor_of_safeIterFinite_inFieldStates_eq_empty (by decide) h

/-- The proposition `safeIterFinite cfg S₀ n = ∅` is decidable, by
`Finset.instDecidableEq`. (Explicit instance for downstream `decide`-based
usage.) -/
instance instDecidableSafeIterFiniteEqEmpty (cfg : GameConfig)
    (S₀ : Finset GameState) (n : ℕ) :
    Decidable (safeIterFinite cfg S₀ n = ∅) :=
  inferInstance

/-- Unfolded form: no solver `σ` is `SolvesTetrisValid cfg σ` when the
iterate evacuates and `4 ≤ cfg.cols`. -/
theorem not_exists_solver_solvesTetrisValid_of_safeIterFinite_inFieldStates_eq_empty
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {n : ℕ}
    (h : safeIterFinite cfg (inFieldStates cfg) n = ∅) :
    ¬ ∃ σ : Solver cfg, SolvesTetrisValid cfg σ :=
  not_tetrisSolvableValidFor_of_safeIterFinite_inFieldStates_eq_empty hcols h

/-- **No init-containing adversarial closed cycle when the iterate
evacuates** (any-`cfg` form via `safe`-set route). -/
theorem not_exists_init_adversarialClosedCycle_of_safeIterFinite_inFieldStates_eq_empty
    {cfg : GameConfig} {n : ℕ}
    (h : safeIterFinite cfg (inFieldStates cfg) n = ∅) :
    ¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states := by
  rintro ⟨C, hinit⟩
  exact init_not_mem_safe_of_safeIterFinite_inFieldStates_eq_empty h
    (C.init_safe_of_init_mem hinit)

/-- Standard-config specialisation of Tick 1452. -/
theorem not_exists_init_adversarialClosedCycle_standard_of_safeIterFinite_inFieldStates_eq_empty
    {n : ℕ}
    (h : safeIterFinite GameConfig.standard
        (inFieldStates GameConfig.standard) n = ∅) :
    ¬ ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  not_exists_init_adversarialClosedCycle_of_safeIterFinite_inFieldStates_eq_empty h

/-- Tiny-config specialisation of Tick 1452. -/
theorem not_exists_init_adversarialClosedCycle_tiny_of_safeIterFinite_inFieldStates_eq_empty
    {n : ℕ}
    (h : safeIterFinite GameConfig.tiny
        (inFieldStates GameConfig.tiny) n = ∅) :
    ¬ ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states :=
  not_exists_init_adversarialClosedCycle_of_safeIterFinite_inFieldStates_eq_empty h

/-- **Bundled convergence + soundness**: combining `safeIterFinite_converges`
with `safeIterFinite_subset_safe`. There exists an `N ≤ S₀.card` such that
the iterate stabilises and is contained in `safe cfg`. -/
theorem exists_stable_safeIterFinite_subset_safe (cfg : GameConfig)
    (S₀ : Finset GameState) :
    ∃ N, N ≤ S₀.card ∧
      safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N ∧
      (↑(safeIterFinite cfg S₀ N) : Set GameState) ⊆ safe cfg := by
  obtain ⟨N, hN, hstable⟩ := safeIterFinite_converges cfg S₀
  exact ⟨N, hN, hstable, safeIterFinite_subset_safe cfg S₀ N hstable⟩

/-- Membership in a stable iterate implies membership in `safe`. -/
theorem mem_safe_of_mem_stable_safeIterFinite {cfg : GameConfig}
    {S₀ : Finset GameState} {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    {g : GameState} (h : g ∈ safeIterFinite cfg S₀ N) : g ∈ safe cfg :=
  safeIterFinite_subset_safe cfg S₀ N hstable (Finset.mem_coe.mpr h)

/-- Init-specialisation: if `init ∈ safeIterFinite cfg S₀ N` at a stable
`N`, then `init ∈ safe cfg`. -/
theorem init_safe_of_init_mem_stable_safeIterFinite {cfg : GameConfig}
    {S₀ : Finset GameState} {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (h : GameState.init ∈ safeIterFinite cfg S₀ N) :
    GameState.init ∈ safe cfg :=
  mem_safe_of_mem_stable_safeIterFinite hstable h

/-- **Positive decision procedure (any-`cfg`)**: if `4 ≤ cfg.cols`, `N` is
stable, and `init ∈ safeIterFinite cfg S₀ N`, then `TetrisSolvableValidFor
cfg`. -/
theorem tetrisSolvableValidFor_of_init_mem_stable_safeIterFinite
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState} {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (h : GameState.init ∈ safeIterFinite cfg S₀ N) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mpr
    (init_safe_of_init_mem_stable_safeIterFinite hstable h)

/-- Standard-config specialisation of Tick 1458. -/
theorem tetrisSolvableValid_of_init_mem_stable_safeIterFinite_standard
    {S₀ : Finset GameState} {N : ℕ}
    (hstable : safeIterFinite GameConfig.standard S₀ (N + 1) =
      safeIterFinite GameConfig.standard S₀ N)
    (h : GameState.init ∈ safeIterFinite GameConfig.standard S₀ N) :
    TetrisSolvableValid :=
  tetrisSolvableValidFor_of_init_mem_stable_safeIterFinite (by decide) hstable h

/-- Tiny-config specialisation of Tick 1458. -/
theorem tetrisSolvableValidFor_tiny_of_init_mem_stable_safeIterFinite
    {S₀ : Finset GameState} {N : ℕ}
    (hstable : safeIterFinite GameConfig.tiny S₀ (N + 1) =
      safeIterFinite GameConfig.tiny S₀ N)
    (h : GameState.init ∈ safeIterFinite GameConfig.tiny S₀ N) :
    TetrisSolvableValidFor GameConfig.tiny :=
  tetrisSolvableValidFor_of_init_mem_stable_safeIterFinite (by decide) hstable h

/-- **Bidirectional init membership iff** at a stable iterate, given a
universe `S₀` containing all of `safe cfg`. -/
theorem init_mem_safeIterFinite_iff_init_safe {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    GameState.init ∈ safeIterFinite cfg S₀ N ↔ GameState.init ∈ safe cfg := by
  have := safe_iff_mem_fixedPoint cfg S₀ N hS₀ hstable GameState.init
  exact this.symm

/-- **Bidirectional `TetrisSolvableValidFor` iff** at a stable iterate. -/
theorem tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState}
    (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    TetrisSolvableValidFor cfg ↔ GameState.init ∈ safeIterFinite cfg S₀ N := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols,
      ← init_mem_safeIterFinite_iff_init_safe hS₀ hstable]

/-- Negated form of Tick 1461: `init ∉ safeIterFinite ↔ init ∉ safe`. -/
theorem init_not_mem_safeIterFinite_iff_init_not_safe {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    GameState.init ∉ safeIterFinite cfg S₀ N ↔ GameState.init ∉ safe cfg :=
  not_congr (init_mem_safeIterFinite_iff_init_safe hS₀ hstable)

/-- Negated form of Tick 1462: `¬ TetrisSolvableValidFor cfg ↔ init ∉
safeIterFinite cfg S₀ N` at a stable iterate. -/
theorem not_tetrisSolvableValidFor_iff_init_not_mem_stable_safeIterFinite
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState}
    (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    ¬ TetrisSolvableValidFor cfg ↔
      GameState.init ∉ safeIterFinite cfg S₀ N :=
  not_congr (tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite hcols hS₀ hstable)

/-- Standard-config specialisation of Tick 1462. -/
theorem tetrisSolvableValid_iff_init_mem_stable_safeIterFinite_standard
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.standard ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.standard S₀ (N + 1) =
      safeIterFinite GameConfig.standard S₀ N) :
    TetrisSolvableValid ↔
      GameState.init ∈ safeIterFinite GameConfig.standard S₀ N :=
  tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite (by decide) hS₀ hstable

/-- Tiny-config specialisation of Tick 1462. -/
theorem tetrisSolvableValidFor_tiny_iff_init_mem_stable_safeIterFinite
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.tiny ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.tiny S₀ (N + 1) =
      safeIterFinite GameConfig.tiny S₀ N) :
    TetrisSolvableValidFor GameConfig.tiny ↔
      GameState.init ∈ safeIterFinite GameConfig.tiny S₀ N :=
  tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite (by decide) hS₀ hstable

/-- **General-`g` form of Tick 1461**: at a stable iterate with `safe ⊆ ↑S₀`,
`g ∈ safeIterFinite cfg S₀ N ↔ g ∈ safe cfg`. Just unwraps
`safe_iff_mem_fixedPoint` to a Finset-membership form. -/
theorem mem_safeIterFinite_iff_mem_safe {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (g : GameState) :
    g ∈ safeIterFinite cfg S₀ N ↔ g ∈ safe cfg :=
  (safe_iff_mem_fixedPoint cfg S₀ N hS₀ hstable g).symm

/-- **Set-coerced stable iterate equals `safe`** (the `safe ⊆ ↑S₀` form).
Direct from Tick 1467 by `Set.ext`. -/
theorem coe_safeIterFinite_eq_safe_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (↑(safeIterFinite cfg S₀ N) : Set GameState) = safe cfg := by
  ext g
  rw [Finset.mem_coe]
  exact mem_safeIterFinite_iff_mem_safe hS₀ hstable g

/-- Symmetric form of Tick 1468. -/
theorem safe_eq_coe_safeIterFinite_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    safe cfg = (↑(safeIterFinite cfg S₀ N) : Set GameState) :=
  (coe_safeIterFinite_eq_safe_of_stable hS₀ hstable).symm

/-- `safe cfg` is finite, given a stable iterate witness containing it. -/
theorem safe_finite_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (safe cfg).Finite := by
  rw [safe_eq_coe_safeIterFinite_of_stable hS₀ hstable]
  exact (safeIterFinite cfg S₀ N).finite_toSet

/-- `safe cfg` is finite given any Finset `S₀` containing it. Uses
`safeIterFinite_converges` to discharge the stability hypothesis. -/
theorem safe_finite_of_subset {cfg : GameConfig} {S₀ : Finset GameState}
    (hS₀ : safe cfg ⊆ ↑S₀) : (safe cfg).Finite := by
  obtain ⟨N, _, hstable⟩ := safeIterFinite_converges cfg S₀
  exact safe_finite_of_stable hS₀ hstable

/-- `safe cfg` is finite iff some Finset contains it. -/
theorem safe_finite_iff_exists_finset_subset (cfg : GameConfig) :
    (safe cfg).Finite ↔ ∃ S₀ : Finset GameState, safe cfg ⊆ ↑S₀ := by
  refine ⟨fun h => ?_, fun ⟨S₀, hS₀⟩ => safe_finite_of_subset hS₀⟩
  exact ⟨h.toFinset, by rw [Set.Finite.coe_toFinset]⟩

/-- Bound: when `safe cfg ⊆ ↑S₀`, `(safe_finite_of_subset hS₀).toFinset.card`
is at most `S₀.card`. -/
theorem safe_finite_toFinset_card_le {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) :
    (safe_finite_of_subset hS₀).toFinset.card ≤ S₀.card := by
  apply Finset.card_le_card
  intro g hg
  rw [Set.Finite.mem_toFinset] at hg
  exact Finset.mem_coe.mp (hS₀ hg)

/-- Strictness: `(safe_finite_of_subset hS₀).toFinset.card ≤ S₀.card` may
hold with equality, but in general is `≤`. We also get the stable-iterate
form bounded by `S₀.card` (via `safeIterFinite_card_le_S₀_card`). -/
theorem safe_finite_toFinset_eq_safeIterFinite_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (safe_finite_of_subset hS₀).toFinset = safeIterFinite cfg S₀ N := by
  ext g
  rw [Set.Finite.mem_toFinset, ← Finset.mem_coe,
      coe_safeIterFinite_eq_safe_of_stable hS₀ hstable]

/-- Card equality at stable iterate: the safe-Finset card equals the stable
iterate's card. -/
theorem safe_finite_toFinset_card_eq_safeIterFinite_card_of_stable
    {cfg : GameConfig} {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (safe_finite_of_subset hS₀).toFinset.card =
      (safeIterFinite cfg S₀ N).card := by
  rw [safe_finite_toFinset_eq_safeIterFinite_of_stable hS₀ hstable]

/-- Decidability of `init ∈ safe cfg` given a Finset universe and a
stability witness. Forward via Tick 1461; reverse via `Finset.decidableMem`
on the stable iterate. -/
def decidable_init_safe_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    Decidable (GameState.init ∈ safe cfg) :=
  decidable_of_iff _ (init_mem_safeIterFinite_iff_init_safe hS₀ hstable)

/-- Decidability of `TetrisSolvableValidFor cfg` given a stability witness
and `4 ≤ cfg.cols`. Chains Tick 1476 with
`tetrisSolvableValidFor_iff_init_safe`. -/
def decidable_tetrisSolvableValidFor_of_stable {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀)
    {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    Decidable (TetrisSolvableValidFor cfg) :=
  decidable_of_iff _ (tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite
    hcols hS₀ hstable).symm

/-- Standard-config decidability of `TetrisSolvableValid` from a stability
witness. -/
def decidable_tetrisSolvableValid_of_stable_standard
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.standard ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.standard S₀ (N + 1) =
      safeIterFinite GameConfig.standard S₀ N) :
    Decidable TetrisSolvableValid :=
  decidable_tetrisSolvableValidFor_of_stable (by decide) hS₀ hstable

/-- Tiny-config decidability of `TetrisSolvableValidFor tiny` from a
stability witness. -/
def decidable_tetrisSolvableValidFor_tiny_of_stable
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.tiny ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.tiny S₀ (N + 1) =
      safeIterFinite GameConfig.tiny S₀ N) :
    Decidable (TetrisSolvableValidFor GameConfig.tiny) :=
  decidable_tetrisSolvableValidFor_of_stable (by decide) hS₀ hstable

/-- General-`g` decidability: `Decidable (g ∈ safe cfg)` given a Finset
universe and a stability witness. -/
def decidable_mem_safe_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N)
    (g : GameState) :
    Decidable (g ∈ safe cfg) :=
  decidable_of_iff _ (mem_safeIterFinite_iff_mem_safe hS₀ hstable g)

/-- `safe cfg = ∅ ↔ stable iterate = ∅`, given `safe ⊆ ↑S₀`. -/
theorem safe_eq_empty_iff_safeIterFinite_eq_empty_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    safe cfg = ∅ ↔ safeIterFinite cfg S₀ N = ∅ := by
  rw [safe_eq_coe_safeIterFinite_of_stable hS₀ hstable, Finset.coe_eq_empty]

/-- Nonemptiness Iff: `(safe cfg).Nonempty ↔ stable iterate is nonempty`. -/
theorem safe_nonempty_iff_safeIterFinite_nonempty_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (safe cfg).Nonempty ↔ (safeIterFinite cfg S₀ N).Nonempty := by
  rw [safe_eq_coe_safeIterFinite_of_stable hS₀ hstable, Finset.coe_nonempty]

/-- Decidability of `safe cfg = ∅` via Tick 1481. -/
def decidable_safe_eq_empty_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    Decidable (safe cfg = ∅) :=
  decidable_of_iff _ (safe_eq_empty_iff_safeIterFinite_eq_empty_of_stable hS₀ hstable).symm

/-- `¬ ∃ init-cycle ↔ init ∉ stable iterate` (forward via cycle ⇒ safe;
reverse via Tick 1452's no-cycle-when-evac route adapted to stable). -/
theorem not_exists_init_adversarialClosedCycle_iff_init_not_mem_stable_safeIterFinite
    {cfg : GameConfig} {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀)
    {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) ↔
      GameState.init ∉ safeIterFinite cfg S₀ N := by
  refine ⟨fun hno hmem => hno ?_, fun hni ⟨C, hinit⟩ => hni ?_⟩
  · -- This direction is hard; we'd need to construct a cycle from
    -- init ∈ stable iterate ⊆ safe. Use `safeReachableClosedCycle`.
    have hsafe : GameState.init ∈ safe cfg :=
      init_safe_of_init_mem_stable_safeIterFinite hstable hmem
    exact ⟨(safeReachableClosedCycle hsafe).toAdversarialClosedCycle,
           init_mem_safeReachableClosedCycle hsafe⟩
  · exact (init_mem_safeIterFinite_iff_init_safe hS₀ hstable).mpr
      (C.init_safe_of_init_mem hinit)

/-- Positive form: `∃ init-cycle ↔ init ∈ stable iterate`. Negation of
Tick 1484, simplified via `not_not`. -/
theorem exists_init_adversarialClosedCycle_iff_init_mem_stable_safeIterFinite
    {cfg : GameConfig} {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀)
    {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    (∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) ↔
      GameState.init ∈ safeIterFinite cfg S₀ N := by
  rw [← Classical.not_not (a := ∃ _, _),
      ← Classical.not_not (a := GameState.init ∈ _),
      not_exists_init_adversarialClosedCycle_iff_init_not_mem_stable_safeIterFinite
        hS₀ hstable]

/-- Decidability of `∃ init-cycle` via Tick 1485. -/
def decidable_exists_init_adversarialClosedCycle_of_stable {cfg : GameConfig}
    {S₀ : Finset GameState} (hS₀ : safe cfg ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite cfg S₀ (N + 1) = safeIterFinite cfg S₀ N) :
    Decidable (∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) :=
  decidable_of_iff _
    (exists_init_adversarialClosedCycle_iff_init_mem_stable_safeIterFinite
      hS₀ hstable).symm

/-- Standard-config: decidability of init-cycle existence. -/
def decidable_exists_init_adversarialClosedCycle_standard_of_stable
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.standard ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.standard S₀ (N + 1) =
      safeIterFinite GameConfig.standard S₀ N) :
    Decidable
      (∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states) :=
  decidable_exists_init_adversarialClosedCycle_of_stable hS₀ hstable

/-- Tiny-config: decidability of init-cycle existence. -/
def decidable_exists_init_adversarialClosedCycle_tiny_of_stable
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.tiny ⊆ ↑S₀) {N : ℕ}
    (hstable : safeIterFinite GameConfig.tiny S₀ (N + 1) =
      safeIterFinite GameConfig.tiny S₀ N) :
    Decidable
      (∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states) :=
  decidable_exists_init_adversarialClosedCycle_of_stable hS₀ hstable

/-- **`TetrisSolvableValidFor` ↔ `∃ init-cycle`** (when `4 ≤ cfg.cols`). -/
theorem tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
      ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols]
  exact ⟨fun h => ⟨(safeReachableClosedCycle h).toAdversarialClosedCycle,
                   init_mem_safeReachableClosedCycle h⟩,
         fun ⟨C, hinit⟩ => C.init_safe_of_init_mem hinit⟩

/-- Standard-config specialisation of Tick 1489. -/
theorem tetrisSolvableValid_iff_exists_init_adversarialClosedCycle_standard :
    TetrisSolvableValid ↔
      ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle (by decide)

/-- Tiny-config specialisation of Tick 1489. -/
theorem tetrisSolvableValidFor_tiny_iff_exists_init_adversarialClosedCycle :
    TetrisSolvableValidFor GameConfig.tiny ↔
      ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states :=
  tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle (by decide)

/-- Negated form of Tick 1489: `¬ TetrisSolvableValidFor cfg ↔ ¬ ∃ init-cycle`. -/
theorem not_tetrisSolvableValidFor_iff_not_exists_init_adversarialClosedCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ¬ TetrisSolvableValidFor cfg ↔
      ¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  not_congr (tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle hcols)

/-- Standard-config specialisation of Tick 1491. -/
theorem not_tetrisSolvableValid_iff_not_exists_init_adversarialClosedCycle_standard :
    ¬ TetrisSolvableValid ↔
      ¬ ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  not_tetrisSolvableValidFor_iff_not_exists_init_adversarialClosedCycle (by decide)

/-- Tiny-config specialisation of Tick 1491. -/
theorem not_tetrisSolvableValidFor_tiny_iff_not_exists_init_adversarialClosedCycle :
    ¬ TetrisSolvableValidFor GameConfig.tiny ↔
      ¬ ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states :=
  not_tetrisSolvableValidFor_iff_not_exists_init_adversarialClosedCycle (by decide)

/-- Direct corollary: from a single init-containing cycle to
`TetrisSolvableValidFor cfg`. -/
theorem tetrisSolvableValidFor_of_init_mem_states {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_exists_init_adversarialClosedCycle hcols).mpr ⟨C, h⟩

/-- Standard-config: a single init-containing cycle ⇒ `TetrisSolvableValid`. -/
theorem tetrisSolvableValid_of_init_mem_states_standard
    (C : AdversarialClosedCycle GameConfig.standard)
    (h : GameState.init ∈ C.states) :
    TetrisSolvableValid :=
  tetrisSolvableValidFor_of_init_mem_states (by decide) C h

/-- Tiny-config: a single init-containing cycle ⇒ `TetrisSolvableValidFor tiny`. -/
theorem tetrisSolvableValidFor_tiny_of_init_mem_states
    (C : AdversarialClosedCycle GameConfig.tiny)
    (h : GameState.init ∈ C.states) :
    TetrisSolvableValidFor GameConfig.tiny :=
  tetrisSolvableValidFor_of_init_mem_states (by decide) C h

/-- Contrapositive of Tick 1493: if `¬ TetrisSolvableValidFor cfg` (and `4
≤ cfg.cols`), then `init ∉ C.states` for every `C`. -/
theorem init_not_mem_states_of_not_tetrisSolvableValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : ¬ TetrisSolvableValidFor cfg)
    (C : AdversarialClosedCycle cfg) :
    GameState.init ∉ C.states :=
  fun hmem => h (tetrisSolvableValidFor_of_init_mem_states hcols C hmem)

/-- Standard-config: contrapositive of Tick 1494's standard version. -/
theorem init_not_mem_states_of_not_tetrisSolvableValid
    (h : ¬ TetrisSolvableValid)
    (C : AdversarialClosedCycle GameConfig.standard) :
    GameState.init ∉ C.states :=
  init_not_mem_states_of_not_tetrisSolvableValidFor (by decide) h C

/-- Tiny-config: contrapositive of Tick 1494's tiny version. -/
theorem init_not_mem_states_of_not_tetrisSolvableValidFor_tiny
    (h : ¬ TetrisSolvableValidFor GameConfig.tiny)
    (C : AdversarialClosedCycle GameConfig.tiny) :
    GameState.init ∉ C.states :=
  init_not_mem_states_of_not_tetrisSolvableValidFor (by decide) h C

/-- ¬∃ form: `¬ TetrisSolvableValidFor cfg ⇒ ¬ ∃ init-cycle`. Direct
consequence of Tick 1491. -/
theorem not_exists_init_adversarialClosedCycle_of_not_tetrisSolvableValidFor
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : ¬ TetrisSolvableValidFor cfg) :
    ¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  (not_tetrisSolvableValidFor_iff_not_exists_init_adversarialClosedCycle hcols).mp h

/-- Standard-config: `¬ TetrisSolvableValid ⇒ ¬ ∃ init-cycle`. -/
theorem not_exists_init_adversarialClosedCycle_standard_of_not_tetrisSolvableValid
    (h : ¬ TetrisSolvableValid) :
    ¬ ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  not_exists_init_adversarialClosedCycle_of_not_tetrisSolvableValidFor (by decide) h

/-- Tiny-config: `¬ TetrisSolvableValidFor tiny ⇒ ¬ ∃ init-cycle`. -/
theorem not_exists_init_adversarialClosedCycle_tiny_of_not_tetrisSolvableValidFor
    (h : ¬ TetrisSolvableValidFor GameConfig.tiny) :
    ¬ ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states :=
  not_exists_init_adversarialClosedCycle_of_not_tetrisSolvableValidFor (by decide) h

/-- **Headline Iff**: `init ∈ safe cfg ↔ ∃ init-cycle`. Pure
characterisation without iterate intermediate steps. Forward via
`safeReachableClosedCycle`; reverse via `init_safe_of_init_mem`. -/
theorem init_safe_iff_exists_init_adversarialClosedCycle (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
      ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  ⟨fun h => ⟨(safeReachableClosedCycle h).toAdversarialClosedCycle,
             init_mem_safeReachableClosedCycle h⟩,
   fun ⟨C, hinit⟩ => C.init_safe_of_init_mem hinit⟩

/-- Negated form of Tick 1499. -/
theorem init_not_safe_iff_not_exists_init_adversarialClosedCycle
    (cfg : GameConfig) :
    GameState.init ∉ safe cfg ↔
      ¬ ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  not_congr (init_safe_iff_exists_init_adversarialClosedCycle cfg)

/-- **M4 ↔ TetrisSolvableValidFor** (generic config with `4 ≤ cfg.cols`):
the headline iff bridging the M4 final-form artifact with the project's
canonical solvability predicate. Composes Tick 1641's
`nonempty_M4Artifact_iff_exists_init_cycle`, Tick 1499's
`init_safe_iff_exists_init_adversarialClosedCycle`, and Tick 515's
`tetrisSolvableValidFor_iff_init_safe`.

This is the **end-to-end equivalence**: producing an M4 artifact is
exactly proving `TetrisSolvableValidFor cfg`, for any config with at
least 4 columns. -/
theorem nonempty_M4Artifact_iff_tetrisSolvableValidFor
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    Nonempty (AdversarialClosedCycle.M4Artifact cfg) ↔
      TetrisSolvableValidFor cfg := by
  rw [nonempty_M4Artifact_iff_exists_init_cycle,
      ← init_safe_iff_exists_init_adversarialClosedCycle,
      ← tetrisSolvableValidFor_iff_init_safe hcols]

/-- **Standard-config M4 iff `TetrisSolvableValid`**: specialisation of
Tick 1645 to `GameConfig.standard`. This is the **headline AGENTS.md
statement**: producing an M4 atlas for canonical 10×20 Tetris is
*exactly* proving `TetrisSolvableValid`. -/
theorem nonempty_M4Artifact_standard_iff_tetrisSolvableValid :
    Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.standard) ↔
      TetrisSolvableValid :=
  nonempty_M4Artifact_iff_tetrisSolvableValidFor
    GameConfig.standard_four_le_cols

/-- **Tiny-config M4 iff `TetrisSolvableValidFor tiny`**: specialisation
of Tick 1645 to `GameConfig.tiny`. -/
theorem nonempty_M4Artifact_tiny_iff_tetrisSolvableValidFor_tiny :
    Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.tiny) ↔
      TetrisSolvableValidFor GameConfig.tiny :=
  nonempty_M4Artifact_iff_tetrisSolvableValidFor
    GameConfig.tiny_four_le_cols

/-- **Negated M4 ↔ TetrisSolvableValidFor**: contrapositive form of
Tick 1645. Useful for decidability arguments showing the M4 artifact
doesn't exist via `¬ TetrisSolvableValidFor cfg`. -/
theorem not_nonempty_M4Artifact_iff_not_tetrisSolvableValidFor
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ¬ Nonempty (AdversarialClosedCycle.M4Artifact cfg) ↔
      ¬ TetrisSolvableValidFor cfg :=
  not_congr (nonempty_M4Artifact_iff_tetrisSolvableValidFor hcols)

/-- **Standard-config negated M4 iff**: contrapositive of Tick 1646. -/
theorem not_nonempty_M4Artifact_standard_iff_not_tetrisSolvableValid :
    ¬ Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.standard) ↔
      ¬ TetrisSolvableValid :=
  not_congr nonempty_M4Artifact_standard_iff_tetrisSolvableValid

/-- **Tiny-config negated M4 iff**: contrapositive of Tick 1646's tiny
variant. -/
theorem not_nonempty_M4Artifact_tiny_iff_not_tetrisSolvableValidFor_tiny :
    ¬ Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.tiny) ↔
      ¬ TetrisSolvableValidFor GameConfig.tiny :=
  not_congr nonempty_M4Artifact_tiny_iff_tetrisSolvableValidFor_tiny

/-- `init ∈ safe cfg` is invariant under the cycle-construction round-trip:
`safeReachableClosedCycle h` produces a cycle, and `init_safe_of_init_mem`
on it recovers `h`. -/
theorem init_safe_of_safeReachableClosedCycle_round_trip {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.init_safe_of_init_mem
        (init_mem_safeReachableClosedCycle h) = h := rfl

/-- `init ∈ safe cfg ↔ init ∈ safeReachableCycle cfg`. The constructed
cycle's states equal `safeReachableCycle cfg`, so init-membership
correspondences. (Forward: from `init ∈ safe`, the constructed cycle has
init; this exits the bridge through the Finset `safeReachableCycle cfg`.) -/
theorem init_safe_iff_init_mem_safeReachableCycle (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔ GameState.init ∈ safeReachableCycle cfg :=
  ⟨fun h => init_mem_safeReachableCycle h,
   fun h => safeReachableCycle_subset_safe cfg _ h⟩

/-- Decidability of `init ∈ safe cfg` via Tick 1502 — bypasses the need
for a stability witness by going through `safeReachableCycle cfg` (which
itself is `noncomputable`, so this instance is also `noncomputable`). -/
noncomputable def decidable_init_safe (cfg : GameConfig) :
    Decidable (GameState.init ∈ safe cfg) :=
  decidable_of_iff _ (init_safe_iff_init_mem_safeReachableCycle cfg).symm

/-- Unconditional `noncomputable` decidability of `TetrisSolvableValidFor
cfg` (when `4 ≤ cfg.cols`). Composes Tick 1503 with
`tetrisSolvableValidFor_iff_init_safe`. -/
noncomputable def decidable_tetrisSolvableValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    Decidable (TetrisSolvableValidFor cfg) :=
  let _ := decidable_init_safe cfg
  decidable_of_iff _ (tetrisSolvableValidFor_iff_init_safe hcols).symm

/-- Standard-config: `noncomputable` decidability of `TetrisSolvableValid`. -/
noncomputable def decidable_tetrisSolvableValid :
    Decidable TetrisSolvableValid :=
  decidable_tetrisSolvableValidFor (by decide)

/-- Tiny-config: `noncomputable` decidability of `TetrisSolvableValidFor
tiny`. -/
noncomputable def decidable_tetrisSolvableValidFor_tiny :
    Decidable (TetrisSolvableValidFor GameConfig.tiny) :=
  decidable_tetrisSolvableValidFor (by decide)

/-- **Generic-config noncomputable decidability of M4 artifact existence**:
for any `cfg` with `4 ≤ cfg.cols`, `Nonempty (M4Artifact cfg)` is decidable.
Reduces to `Decidable (TetrisSolvableValidFor cfg)` via Tick 1645. -/
noncomputable def decidable_nonempty_M4Artifact {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    Decidable (Nonempty (AdversarialClosedCycle.M4Artifact cfg)) :=
  let _ := decidable_tetrisSolvableValidFor hcols
  decidable_of_iff _ (nonempty_M4Artifact_iff_tetrisSolvableValidFor hcols).symm

/-- **Standard-config noncomputable decidability of M4 artifact existence**. -/
noncomputable def decidable_nonempty_M4Artifact_standard :
    Decidable
      (Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.standard)) :=
  decidable_nonempty_M4Artifact GameConfig.standard_four_le_cols

/-- **Tiny-config noncomputable decidability of M4 artifact existence**:
the M4 artifact existence question is decidable on the tractable tiny
config (`|inFieldStates GameConfig.tiny| = 2^16`). -/
noncomputable def decidable_nonempty_M4Artifact_tiny :
    Decidable
      (Nonempty (AdversarialClosedCycle.M4Artifact GameConfig.tiny)) :=
  decidable_nonempty_M4Artifact GameConfig.tiny_four_le_cols

/-- Unconditional `noncomputable` decidability of `∃ init-cycle`. -/
noncomputable def decidable_exists_init_adversarialClosedCycle
    (cfg : GameConfig) :
    Decidable (∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states) :=
  let _ := decidable_init_safe cfg
  decidable_of_iff _ (init_safe_iff_exists_init_adversarialClosedCycle cfg)

/-- Standard-config specialisation of Tick 1506. -/
noncomputable def decidable_exists_init_adversarialClosedCycle_standard :
    Decidable
      (∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states) :=
  decidable_exists_init_adversarialClosedCycle GameConfig.standard

/-- Tiny-config specialisation of Tick 1506. -/
noncomputable def decidable_exists_init_adversarialClosedCycle_tiny :
    Decidable
      (∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states) :=
  decidable_exists_init_adversarialClosedCycle GameConfig.tiny

/-- Unconditional `noncomputable` decidability of `safe cfg = ∅` (via Tick
1503 routed through `init`-mem decidability + safe-emptiness Iff). -/
noncomputable def decidable_safe_eq_empty (cfg : GameConfig) :
    Decidable (safe cfg = ∅) :=
  Classical.dec _

/-- Unconditional `noncomputable` decidability of `(safe cfg).Nonempty`. -/
noncomputable def decidable_safe_nonempty (cfg : GameConfig) :
    Decidable (safe cfg).Nonempty :=
  Classical.dec _

/-- **Guaranteed stability at `S₀.card`**: the iterate is stable at index
`S₀.card` (no later step changes it). Combines
`safeIterFinite_converges` with stability propagation. -/
theorem safeIterFinite_stable_at_S₀_card (cfg : GameConfig)
    (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ (S₀.card + 1) = safeIterFinite cfg S₀ S₀.card := by
  obtain ⟨N, hN, hstable⟩ := safeIterFinite_converges cfg S₀
  -- The iterate is stable at N ≤ S₀.card. Propagate to S₀.card via
  -- safeIterFinite_eq_of_stable_of_le.
  have h1 : safeIterFinite cfg S₀ S₀.card = safeIterFinite cfg S₀ N :=
    safeIterFinite_eq_of_stable_of_le cfg S₀ hstable hN
  have h2 : safeIterFinite cfg S₀ (S₀.card + 1) = safeIterFinite cfg S₀ N :=
    safeIterFinite_eq_of_stable_of_le cfg S₀ hstable (by omega)
  rw [h1, h2]

/-- In-field iterate stabilises at index `(inFieldStates cfg).card`. -/
theorem safeIterFinite_inFieldStates_stable_at_card (cfg : GameConfig) :
    safeIterFinite cfg (inFieldStates cfg)
        ((inFieldStates cfg).card + 1) =
      safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card :=
  safeIterFinite_stable_at_S₀_card cfg (inFieldStates cfg)

/-- The in-field iterate at `(inFieldStates cfg).card` is an `F_finite`
fixed point. -/
theorem safeIterFinite_inFieldStates_F_finite_fixed_at_card (cfg : GameConfig) :
    F_finite cfg (safeIterFinite cfg (inFieldStates cfg)
        (inFieldStates cfg).card) =
      safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card :=
  (safeIterFinite_succ_eq_iff_F_finite_fixed cfg
      (inFieldStates cfg) (inFieldStates cfg).card).mp
    (safeIterFinite_inFieldStates_stable_at_card cfg)

/-- For `inFieldStates cfg ⊆ S₀.card`-iterations beyond
`(inFieldStates cfg).card`, the iterate is constant. -/
theorem safeIterFinite_inFieldStates_eq_of_le (cfg : GameConfig) {k : ℕ}
    (hk : (inFieldStates cfg).card ≤ k) :
    safeIterFinite cfg (inFieldStates cfg) k =
      safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card :=
  safeIterFinite_eq_of_stable_of_le cfg (inFieldStates cfg)
    (safeIterFinite_inFieldStates_stable_at_card cfg) hk

/-- The in-field iterate at `(inFieldStates cfg).card` is contained in
`inFieldStates cfg`. -/
theorem safeIterFinite_inFieldStates_at_card_subset (cfg : GameConfig) :
    safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card ⊆
      inFieldStates cfg :=
  safeIterFinite_inFieldStates_subset cfg (inFieldStates cfg).card

/-- Card bound at `(inFieldStates cfg).card`. -/
theorem safeIterFinite_inFieldStates_at_card_card_le (cfg : GameConfig) :
    (safeIterFinite cfg (inFieldStates cfg) (inFieldStates cfg).card).card ≤
      (inFieldStates cfg).card :=
  safeIterFinite_inFieldStates_card_le cfg (inFieldStates cfg).card

/-- `standard_inFieldStates_card > 0`. -/
theorem standard_inFieldStates_card_pos :
    0 < (inFieldStates GameConfig.standard).card :=
  inFieldStates_card_pos GameConfig.standard

/-- `tiny_inFieldStates_card > 0`. -/
theorem tiny_inFieldStates_card_pos :
    0 < (inFieldStates GameConfig.tiny).card :=
  inFieldStates_card_pos GameConfig.tiny

/-- `standard_inFieldStates.Nonempty`. -/
theorem standard_inFieldStates_nonempty :
    (inFieldStates GameConfig.standard).Nonempty :=
  Finset.card_pos.mp standard_inFieldStates_card_pos

/-- `tiny_inFieldStates.Nonempty`. -/
theorem tiny_inFieldStates_nonempty :
    (inFieldStates GameConfig.tiny).Nonempty :=
  Finset.card_pos.mp tiny_inFieldStates_card_pos

/-- **Decision procedure at index `S₀.card`**: removes the
existential `N` from Tick 1462 by fixing `N := S₀.card`. -/
theorem tetrisSolvableValidFor_iff_init_mem_safeIterFinite_at_S₀_card
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState}
    (hS₀ : safe cfg ⊆ ↑S₀) :
    TetrisSolvableValidFor cfg ↔
      GameState.init ∈ safeIterFinite cfg S₀ S₀.card :=
  tetrisSolvableValidFor_iff_init_mem_stable_safeIterFinite hcols hS₀
    (safeIterFinite_stable_at_S₀_card cfg S₀)

/-- Standard-config specialisation of Tick 1511. -/
theorem tetrisSolvableValid_iff_init_mem_safeIterFinite_at_S₀_card_standard
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.standard ⊆ ↑S₀) :
    TetrisSolvableValid ↔
      GameState.init ∈ safeIterFinite GameConfig.standard S₀ S₀.card :=
  tetrisSolvableValidFor_iff_init_mem_safeIterFinite_at_S₀_card (by decide) hS₀

/-- Tiny-config specialisation of Tick 1511. -/
theorem tetrisSolvableValidFor_tiny_iff_init_mem_safeIterFinite_at_S₀_card
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.tiny ⊆ ↑S₀) :
    TetrisSolvableValidFor GameConfig.tiny ↔
      GameState.init ∈ safeIterFinite GameConfig.tiny S₀ S₀.card :=
  tetrisSolvableValidFor_iff_init_mem_safeIterFinite_at_S₀_card (by decide) hS₀

/-- **Computable decidability** of `TetrisSolvableValidFor cfg`, given a
seed `S₀` containing `safe cfg`. The decidability instance is computable
because it's directly through `Finset.decidableMem` on the iterate. -/
def decidable_tetrisSolvableValidFor_at_S₀_card {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState}
    (hS₀ : safe cfg ⊆ ↑S₀) :
    Decidable (TetrisSolvableValidFor cfg) :=
  decidable_of_iff _
    (tetrisSolvableValidFor_iff_init_mem_safeIterFinite_at_S₀_card hcols hS₀).symm

/-- Standard-config (computable!) decidability of `TetrisSolvableValid`. -/
def decidable_tetrisSolvableValid_at_S₀_card_standard
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.standard ⊆ ↑S₀) :
    Decidable TetrisSolvableValid :=
  decidable_tetrisSolvableValidFor_at_S₀_card (by decide) hS₀

/-- Tiny-config (computable!) decidability of
`TetrisSolvableValidFor tiny`. -/
def decidable_tetrisSolvableValidFor_tiny_at_S₀_card
    {S₀ : Finset GameState} (hS₀ : safe GameConfig.tiny ⊆ ↑S₀) :
    Decidable (TetrisSolvableValidFor GameConfig.tiny) :=
  decidable_tetrisSolvableValidFor_at_S₀_card (by decide) hS₀

/-- The iterate at index `S₀.card` (always stable per Tick 1510) is
contained in `safe cfg`. -/
theorem safeIterFinite_at_S₀_card_subset_safe (cfg : GameConfig)
    (S₀ : Finset GameState) :
    (↑(safeIterFinite cfg S₀ S₀.card) : Set GameState) ⊆ safe cfg :=
  safeIterFinite_subset_safe cfg S₀ S₀.card
    (safeIterFinite_stable_at_S₀_card cfg S₀)

/-- Pointwise form of Tick 1515. -/
theorem mem_safe_of_mem_safeIterFinite_at_S₀_card {cfg : GameConfig}
    {S₀ : Finset GameState} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ S₀.card) : g ∈ safe cfg :=
  safeIterFinite_at_S₀_card_subset_safe cfg S₀ (Finset.mem_coe.mpr h)

/-- Init-specialisation of Tick 1516. -/
theorem init_safe_of_init_mem_safeIterFinite_at_S₀_card {cfg : GameConfig}
    {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite cfg S₀ S₀.card) :
    GameState.init ∈ safe cfg :=
  mem_safe_of_mem_safeIterFinite_at_S₀_card h

/-- **Unconditional positive direction**: from `init ∈ safeIterFinite cfg
S₀ S₀.card` (no `safe ⊆ ↑S₀` needed), get `TetrisSolvableValidFor cfg`
(under `4 ≤ cfg.cols`). -/
theorem tetrisSolvableValidFor_of_init_mem_safeIterFinite_at_S₀_card
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite cfg S₀ S₀.card) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_safe hcols).mpr
    (init_safe_of_init_mem_safeIterFinite_at_S₀_card h)

/-- Standard-config: a positive iterate witness ⇒ `TetrisSolvableValid`. -/
theorem tetrisSolvableValid_of_init_mem_safeIterFinite_at_S₀_card_standard
    {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite GameConfig.standard S₀ S₀.card) :
    TetrisSolvableValid :=
  tetrisSolvableValidFor_of_init_mem_safeIterFinite_at_S₀_card (by decide) h

/-- Tiny-config: a positive iterate witness ⇒ `TetrisSolvableValidFor tiny`. -/
theorem tetrisSolvableValidFor_tiny_of_init_mem_safeIterFinite_at_S₀_card
    {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite GameConfig.tiny S₀ S₀.card) :
    TetrisSolvableValidFor GameConfig.tiny :=
  tetrisSolvableValidFor_of_init_mem_safeIterFinite_at_S₀_card (by decide) h

/-- **Unconditional cycle-existence positive direction**: from a positive
iterate witness, get an init-containing cycle directly (no `4 ≤ cfg.cols`
hypothesis needed). -/
theorem exists_init_adversarialClosedCycle_of_init_mem_safeIterFinite_at_S₀_card
    {cfg : GameConfig} {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite cfg S₀ S₀.card) :
    ∃ C : AdversarialClosedCycle cfg, GameState.init ∈ C.states :=
  (init_safe_iff_exists_init_adversarialClosedCycle cfg).mp
    (init_safe_of_init_mem_safeIterFinite_at_S₀_card h)

/-- Standard-config specialisation of Tick 1520. -/
theorem exists_init_adversarialClosedCycle_standard_of_init_mem_safeIterFinite_at_S₀_card
    {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite GameConfig.standard S₀ S₀.card) :
    ∃ C : AdversarialClosedCycle GameConfig.standard,
        GameState.init ∈ C.states :=
  exists_init_adversarialClosedCycle_of_init_mem_safeIterFinite_at_S₀_card h

/-- Tiny-config specialisation of Tick 1520. -/
theorem exists_init_adversarialClosedCycle_tiny_of_init_mem_safeIterFinite_at_S₀_card
    {S₀ : Finset GameState}
    (h : GameState.init ∈ safeIterFinite GameConfig.tiny S₀ S₀.card) :
    ∃ C : AdversarialClosedCycle GameConfig.tiny,
        GameState.init ∈ C.states :=
  exists_init_adversarialClosedCycle_of_init_mem_safeIterFinite_at_S₀_card h

/-- **Computable** `safeIterFinite cfg S₀ S₀.card` Decidable instance for
`init`-membership (via `Finset.decidableMem`). -/
instance instDecidableInitMemSafeIterFiniteAtS₀Card (cfg : GameConfig)
    (S₀ : Finset GameState) :
    Decidable (GameState.init ∈ safeIterFinite cfg S₀ S₀.card) :=
  inferInstance

/-- Convenience eta: `S₀.card`-iterated stable inhabitant. -/
theorem safeIterFinite_at_S₀_card_eq_stable_iterate (cfg : GameConfig)
    (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ (S₀.card + 1) = safeIterFinite cfg S₀ S₀.card :=
  safeIterFinite_stable_at_S₀_card cfg S₀

/-- Beyond-`S₀.card`: `safeIterFinite cfg S₀ (S₀.card + k) = safeIterFinite
cfg S₀ S₀.card` for every `k`. -/
theorem safeIterFinite_S₀_card_add_eq_S₀_card (cfg : GameConfig)
    (S₀ : Finset GameState) (k : ℕ) :
    safeIterFinite cfg S₀ (S₀.card + k) = safeIterFinite cfg S₀ S₀.card :=
  safeIterFinite_stable cfg S₀ (safeIterFinite_stable_at_S₀_card cfg S₀) k

/-- `≥ S₀.card` form: every iterate at index `≥ S₀.card` equals the
iterate at `S₀.card`. -/
theorem safeIterFinite_eq_S₀_card_of_le (cfg : GameConfig)
    (S₀ : Finset GameState) {k : ℕ} (hk : S₀.card ≤ k) :
    safeIterFinite cfg S₀ k = safeIterFinite cfg S₀ S₀.card :=
  safeIterFinite_eq_of_stable_of_le cfg S₀
    (safeIterFinite_stable_at_S₀_card cfg S₀) hk

/-- The iterate at `S₀.card` is a subset of `S₀`. -/
theorem safeIterFinite_at_S₀_card_subset (cfg : GameConfig)
    (S₀ : Finset GameState) :
    safeIterFinite cfg S₀ S₀.card ⊆ S₀ :=
  safeIterFinite_subset_S₀ cfg S₀ S₀.card

/-- The iterate at `S₀.card` has card ≤ `S₀.card`. -/
theorem safeIterFinite_at_S₀_card_card_le (cfg : GameConfig)
    (S₀ : Finset GameState) :
    (safeIterFinite cfg S₀ S₀.card).card ≤ S₀.card :=
  safeIterFinite_card_le_S₀_card cfg S₀ S₀.card

/-- The iterate at `S₀.card` is a fixed point of `F_finite`. -/
theorem safeIterFinite_at_S₀_card_F_finite_fixed (cfg : GameConfig)
    (S₀ : Finset GameState) :
    F_finite cfg (safeIterFinite cfg S₀ S₀.card) =
      safeIterFinite cfg S₀ S₀.card :=
  (safeIterFinite_succ_eq_iff_F_finite_fixed cfg S₀ S₀.card).mp
    (safeIterFinite_stable_at_S₀_card cfg S₀)

/-- No state in the closed-form stable iterate is lost. -/
theorem not_lost_of_mem_safeIterFinite_at_S₀_card_succ {cfg : GameConfig}
    {S₀ : Finset GameState} {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ (S₀.card + 1)) : ¬ g.lost cfg :=
  not_lost_of_mem_safeIterFinite_succ h

/-- For `1 ≤ N`, members of `safeIterFinite cfg S₀ N` are not lost. -/
theorem not_lost_of_mem_safeIterFinite_pos {cfg : GameConfig}
    {S₀ : Finset GameState} {N : ℕ} (hN : 1 ≤ N) {g : GameState}
    (h : g ∈ safeIterFinite cfg S₀ N) : ¬ g.lost cfg := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.one_le_iff_ne_zero.mp hN)
  exact not_lost_of_mem_safeIterFinite_succ h

/-- Contrapositive of Tick 1531: for `1 ≤ N`, lost states are not in
`safeIterFinite cfg S₀ N`. -/
theorem not_mem_safeIterFinite_pos_of_lost {cfg : GameConfig}
    {S₀ : Finset GameState} {N : ℕ} (hN : 1 ≤ N) {g : GameState}
    (h : g.lost cfg) : g ∉ safeIterFinite cfg S₀ N :=
  fun hmem => not_lost_of_mem_safeIterFinite_pos hN hmem h

/-- For `1 ≤ N`, the iterate's Set-coercion is disjoint from
`{g | g.lost cfg}`. -/
theorem safeIterFinite_pos_disjoint_lost (cfg : GameConfig)
    (S₀ : Finset GameState) {N : ℕ} (hN : 1 ≤ N) :
    Disjoint (↑(safeIterFinite cfg S₀ N) : Set GameState)
      { g | g.lost cfg } := by
  rw [Set.disjoint_left]
  intro g hmem hlost
  exact not_lost_of_mem_safeIterFinite_pos hN (Finset.mem_coe.mp hmem) hlost

/-- `Disjoint`-form companion to Tick 1533: lost-states are disjoint from
the iterate at `1 ≤ N`. -/
theorem lost_disjoint_safeIterFinite_pos (cfg : GameConfig)
    (S₀ : Finset GameState) {N : ℕ} (hN : 1 ≤ N) :
    Disjoint ({ g | g.lost cfg } : Set GameState)
      (↑(safeIterFinite cfg S₀ N)) :=
  (safeIterFinite_pos_disjoint_lost cfg S₀ hN).symm


/-- The cycle's `next` reduces to the canonical `safeSolver` step. -/
@[simp] theorem safeReachableClosedCycle_next {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) (s : GameState) (p : Piece) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.next s p =
      adversarialStep cfg s p (safeSolver cfg s p) := rfl

/-- The constructed `safeReachableCycle` has card in `[28, |inFieldStates|]`. -/
theorem safeReachableCycle_card_interval {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card ≤ (inFieldStates cfg).card := by
  refine ⟨?_, safeReachableCycle_card_le_inFieldStates cfg⟩
  have hC :=
    (safeReachableClosedCycle h).toAdversarialClosedCycle.init_states_card_ge_twenty_eight
      (init_mem_safeReachableClosedCycle h)
  rwa [safeReachableClosedCycle_states] at hC

/-- **Strict interval for the canonical atlas card**: combines the
existing `28 ≤ card` lower bound with the strict `< universe` upper
bound from Tick 270. -/
theorem safeReachableCycle_card_strict_interval {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card < (inFieldStates cfg).card :=
  ⟨(safeReachableCycle_card_interval h).1,
   safeReachableCycle_card_lt_inFieldStates_card cfg⟩

/-- Closed-form atlas card interval `[28, 2^(cols*rows)*128)`. -/
theorem safeReachableCycle_card_pow_interval {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card < 2 ^ (cfg.cols * cfg.rows) * 128 :=
  ⟨(safeReachableCycle_card_interval h).1,
   safeReachableCycle_card_lt_pow cfg⟩

/-- Named projection of the lower bound: the canonical atlas
contains at least 28 states whenever init is safe. -/
theorem safeReachableCycle_card_ge_28 {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableCycle cfg).card :=
  (safeReachableCycle_card_interval h).1

/-- The canonical atlas has more than one state. Strict form of the
`2 ≤ card` weakening; useful for cycle/pigeonhole arguments. -/
theorem safeReachableCycle_one_lt_card {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    1 < (safeReachableCycle cfg).card := by
  have := safeReachableCycle_card_ge_28 h
  omega

/-- Atlas card is not 1 — it has at least 28 elements. -/
theorem safeReachableCycle_card_ne_one {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableCycle cfg).card ≠ 1 := by
  have := safeReachableCycle_one_lt_card h
  omega

/-- Atlas contains two distinct states. -/
theorem safeReachableCycle_exists_two_distinct {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ a ∈ safeReachableCycle cfg, ∃ b ∈ safeReachableCycle cfg, a ≠ b :=
  Finset.one_lt_card.mp (safeReachableCycle_one_lt_card h)


@[simp] theorem safeReachableClosedCycle_solver {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    (safeReachableClosedCycle h).toAdversarialClosedCycle.solver =
      safeSolver cfg := rfl

/-- **The safe ↔ cycle equivalence**: `init` is in the safe set iff there
exists a `ReachableClosedCycle` containing `init`.
- (→) `safeReachableClosedCycle` constructs the cycle from safety.
- (←) `subset_safe` on the cycle's underlying `AdversarialClosedCycle`. -/
theorem init_safe_iff_exists_init_cycle (cfg : GameConfig) :
    GameState.init ∈ safe cfg ↔
    ∃ C : ReachableClosedCycle cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states := by
  constructor
  · intro h
    exact ⟨safeReachableClosedCycle h, init_mem_safeReachableCycle h⟩
  · rintro ⟨C, hinit⟩
    exact C.toAdversarialClosedCycle.subset_safe (Finset.mem_coe.mpr hinit)

/-- **Decidability transport**: if `init ∈ safe cfg` is decidable, so is
`TetrisSolvableValidFor cfg`. Useful when combined with a concrete
decidability witness from `decideSafeFromUniverse` (Tick 25). -/
def TetrisSolvableValidFor.decidable {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    [Decidable (GameState.init ∈ safe cfg)] :
    Decidable (TetrisSolvableValidFor cfg) :=
  decidable_of_iff _ (tetrisSolvableValidFor_iff_init_safe hcols).symm

/-- **Headline equivalence**: `TetrisSolvableValidFor cfg ↔ ∃ init-cycle`. -/
theorem tetrisSolvableValidFor_iff_exists_init_cycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    TetrisSolvableValidFor cfg ↔
    ∃ C : ReachableClosedCycle cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states := by
  rw [tetrisSolvableValidFor_iff_init_safe hcols, init_safe_iff_exists_init_cycle]

/-- Specialised to `standard`. -/
theorem tetrisSolvableValid_iff_exists_init_cycle :
    TetrisSolvableValid ↔
    ∃ C : ReachableClosedCycle GameConfig.standard,
      GameState.init ∈ C.toAdversarialClosedCycle.states :=
  tetrisSolvableValidFor_iff_exists_init_cycle (by decide)

/-- **Contrapositive**: if `init` is not safe, no `ValidSolver` solves Tetris. -/
theorem not_tetrisSolvableValidFor_of_init_not_safe {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : GameState.init ∉ safe cfg) :
    ¬ TetrisSolvableValidFor cfg := fun hs =>
  h ((tetrisSolvableValidFor_iff_init_safe hcols).mp hs)

/-- Direct cycle-to-solver path with the canonical `safeSolver`. -/
theorem solvesTetrisValid_of_init_cycle {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (C : ReachableClosedCycle cfg)
    (h : GameState.init ∈ C.toAdversarialClosedCycle.states) :
    SolvesTetrisValid cfg (safeSolver cfg) :=
  init_safe_implies_solvesTetrisValid hcols
    (C.toAdversarialClosedCycle.subset_safe (Finset.mem_coe.mpr h))

/-- **Solvability implies a large cycle**: if Tetris is valid-solvable,
there's an init-cycle with at least 28 states. Combines Tick 64's lower
bound with the existence equivalence (Tick 128). -/
theorem tetrisSolvableValidFor_implies_cycle_card_ge_28 {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : TetrisSolvableValidFor cfg) :
    ∃ C : ReachableClosedCycle cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card := by
  obtain ⟨C, hinit⟩ :=
    (tetrisSolvableValidFor_iff_exists_init_cycle hcols).mp h
  exact ⟨C, hinit,
    C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight hinit⟩

/-- **Solvability implies a bounded cycle**: the cycle has card in
`[28, |inFieldStates cfg|]`. -/
theorem tetrisSolvableValidFor_implies_cycle_card_interval {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : TetrisSolvableValidFor cfg) :
    ∃ C : ReachableClosedCycle cfg,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ (inFieldStates cfg).card := by
  obtain ⟨C, hinit⟩ :=
    (tetrisSolvableValidFor_iff_exists_init_cycle hcols).mp h
  exact ⟨C, hinit,
    C.toAdversarialClosedCycle.init_states_card_ge_twenty_eight hinit,
    C.states_card_le⟩

/-- **Concrete interval for standard**: any valid solving solver implies an
init-cycle of size in `[28, 2^200 * 128]`. -/
theorem tetrisSolvableValid_implies_cycle_card_interval_standard
    (h : TetrisSolvableValid) :
    ∃ C : ReachableClosedCycle GameConfig.standard,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 2 ^ 200 * 128 := by
  obtain ⟨C, hinit, h1, h2⟩ :=
    tetrisSolvableValidFor_implies_cycle_card_interval (cfg := GameConfig.standard)
      (by decide) h
  exact ⟨C, hinit, h1, standard_inFieldStates_card ▸ h2⟩

/-- **Concrete interval for tiny**: any valid solving solver implies an
init-cycle of size in `[28, 8 388 608]`. -/
theorem tetrisSolvableValidFor_tiny_implies_cycle_card_interval
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    ∃ C : ReachableClosedCycle GameConfig.tiny,
      GameState.init ∈ C.toAdversarialClosedCycle.states ∧
      28 ≤ C.toAdversarialClosedCycle.states.card ∧
      C.toAdversarialClosedCycle.states.card ≤ 8388608 := by
  obtain ⟨C, hinit, h1, h2⟩ :=
    tetrisSolvableValidFor_implies_cycle_card_interval (cfg := GameConfig.tiny)
      (by decide) h
  exact ⟨C, hinit, h1, tiny_inFieldStates_card ▸ h2⟩

/-- **Solvability ⇒ concrete `safeReachableCycle` card bounds**. Combining
the Finset equivalence with `safeReachableCycle_card_interval`: any
`TetrisSolvableValidFor cfg` proof furnishes the closed search interval
`28 ≤ |safeReachableCycle cfg| ≤ |inFieldStates cfg|`. -/
theorem tetrisSolvableValidFor_implies_safeReachableCycle_card_interval
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card ≤ (inFieldStates cfg).card :=
  safeReachableCycle_card_interval
    ((init_mem_safeReachableCycle_iff cfg).mp
      ((tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp h))

/-- **Strict solvability bound**: solvability implies the strict
two-sided interval `28 ≤ card < |inFieldStates|`. Upgrades Tick 246
with the strict upper bound from Tick 270 / Tick 272. -/
theorem tetrisSolvableValidFor_implies_safeReachableCycle_card_strict_interval
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card < (inFieldStates cfg).card :=
  safeReachableCycle_card_strict_interval
    ((init_mem_safeReachableCycle_iff cfg).mp
      ((tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp h))

/-- **Closed-form solvability bound**: solvability implies the strict
interval `28 ≤ card < 2^(cols*rows) * 128`. -/
theorem tetrisSolvableValidFor_implies_safeReachableCycle_card_pow_interval
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card < 2 ^ (cfg.cols * cfg.rows) * 128 :=
  safeReachableCycle_card_pow_interval
    ((init_mem_safeReachableCycle_iff cfg).mp
      ((tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp h))

/-- Standard `card_pow_interval` from solvability. -/
theorem tetrisSolvableValid_implies_standard_card_pow_interval
    (h : TetrisSolvableValid) :
    28 ≤ (safeReachableCycle GameConfig.standard).card ∧
    (safeReachableCycle GameConfig.standard).card <
      2 ^ (GameConfig.standard.cols * GameConfig.standard.rows) * 128 :=
  tetrisSolvableValidFor_implies_safeReachableCycle_card_pow_interval
    GameConfig.standard_four_le_cols h

/-- Tiny `card_pow_interval` from solvability. -/
theorem tetrisSolvableValidFor_tiny_implies_card_pow_interval
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    28 ≤ (safeReachableCycle GameConfig.tiny).card ∧
    (safeReachableCycle GameConfig.tiny).card <
      2 ^ (GameConfig.tiny.cols * GameConfig.tiny.rows) * 128 :=
  tetrisSolvableValidFor_implies_safeReachableCycle_card_pow_interval
    GameConfig.tiny_four_le_cols h

/-- **Atlas-emptiness is an unsolvability certificate** (cols ≥ 4). If
the canonical `safeReachableCycle` Finset is empty, then no valid solver
exists. Contrapositive of the iff: a solver would put `init` inside the
empty Finset. Decidable membership (`safeReachableCycle.decEq ∅`) makes
this a directly checkable witness against solvability. -/
theorem safeReachableCycle_eq_empty_implies_not_solvable
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : safeReachableCycle cfg = ∅) :
    ¬ TetrisSolvableValidFor cfg := by
  intro hsol
  have hmem : GameState.init ∈ safeReachableCycle cfg :=
    (tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp hsol
  rw [h] at hmem
  exact absurd hmem (Finset.notMem_empty _)

/-- **Card-zero unsolvability certificate** (cols ≥ 4). Numerical
restatement of Tick 248: if a search procedure proves the canonical
atlas Finset has cardinality 0, the game is not validly solvable. -/
theorem safeReachableCycle_card_zero_implies_not_solvable
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : (safeReachableCycle cfg).card = 0) :
    ¬ TetrisSolvableValidFor cfg :=
  safeReachableCycle_eq_empty_implies_not_solvable hcols
    (Finset.card_eq_zero.mp h)

/-- **Standard atlas-empty refutes solvability**. Specialization of
Tick 248 to `GameConfig.standard`. -/
theorem safeReachableCycle_standard_eq_empty_implies_not_solvable
    (h : safeReachableCycle GameConfig.standard = ∅) :
    ¬ TetrisSolvableValid :=
  safeReachableCycle_eq_empty_implies_not_solvable (by decide) h

/-- **Tiny atlas-empty refutes solvability**. Specialization of Tick 248
to `GameConfig.tiny`. -/
theorem safeReachableCycle_tiny_eq_empty_implies_not_solvable
    (h : safeReachableCycle GameConfig.tiny = ∅) :
    ¬ TetrisSolvableValidFor GameConfig.tiny :=
  safeReachableCycle_eq_empty_implies_not_solvable (by decide) h

/-- **Standard concrete bound**: solvability of standard Tetris implies
`28 ≤ |safeReachableCycle standard| ≤ 2^207`. -/
theorem tetrisSolvableValid_implies_standard_safeReachableCycle_card_interval
    (h : TetrisSolvableValid) :
    28 ≤ (safeReachableCycle GameConfig.standard).card ∧
    (safeReachableCycle GameConfig.standard).card ≤ 2 ^ 200 * 128 := by
  obtain ⟨h1, h2⟩ :=
    tetrisSolvableValidFor_implies_safeReachableCycle_card_interval
      (cfg := GameConfig.standard) (by decide) h
  exact ⟨h1, standard_inFieldStates_card ▸ h2⟩

/-- **Tiny concrete bound**: solvability of tiny implies
`28 ≤ |safeReachableCycle tiny| ≤ 8 388 608`. -/
theorem tetrisSolvableValidFor_tiny_implies_safeReachableCycle_card_interval
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    28 ≤ (safeReachableCycle GameConfig.tiny).card ∧
    (safeReachableCycle GameConfig.tiny).card ≤ 8388608 := by
  obtain ⟨h1, h2⟩ :=
    tetrisSolvableValidFor_implies_safeReachableCycle_card_interval
      (cfg := GameConfig.tiny) (by decide) h
  exact ⟨h1, tiny_inFieldStates_card ▸ h2⟩

/-- **Standard strict concrete bound**: solvability of standard implies
`28 ≤ |safeReachableCycle standard| < 2^207`. -/
theorem tetrisSolvableValid_implies_standard_safeReachableCycle_card_strict_interval
    (h : TetrisSolvableValid) :
    28 ≤ (safeReachableCycle GameConfig.standard).card ∧
    (safeReachableCycle GameConfig.standard).card < 2 ^ 200 * 128 := by
  obtain ⟨h1, h2⟩ :=
    tetrisSolvableValidFor_implies_safeReachableCycle_card_strict_interval
      (cfg := GameConfig.standard) (by decide) h
  exact ⟨h1, standard_inFieldStates_card ▸ h2⟩

/-- **Tiny strict concrete bound**: solvability of tiny implies
`28 ≤ |safeReachableCycle tiny| < 8 388 608`. -/
theorem tetrisSolvableValidFor_tiny_implies_safeReachableCycle_card_strict_interval
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    28 ≤ (safeReachableCycle GameConfig.tiny).card ∧
    (safeReachableCycle GameConfig.tiny).card < 8388608 := by
  obtain ⟨h1, h2⟩ :=
    tetrisSolvableValidFor_implies_safeReachableCycle_card_strict_interval
      (cfg := GameConfig.tiny) (by decide) h
  exact ⟨h1, tiny_inFieldStates_card ▸ h2⟩

/-- **Card-`<28` unsolvability certificate** (cols ≥ 4). Strengthens
the card-zero refutation (Tick 249): if the canonical atlas has fewer
than 28 elements, the game cannot be validly solved. -/
theorem safeReachableCycle_card_lt_28_implies_not_solvable
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : (safeReachableCycle cfg).card < 28) :
    ¬ TetrisSolvableValidFor cfg := by
  intro hsol
  have h28 :=
    (tetrisSolvableValidFor_implies_safeReachableCycle_card_interval
      hcols hsol).1
  omega

/-- **Parametric `2 ≤ card` lower bound**: any solvable config has at
least 2 distinct atlas states. Ergonomic corollary of the `28 ≤ card`
interval theorem. -/
theorem tetrisSolvableValidFor_implies_safeReachableCycle_card_ge_two
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    2 ≤ (safeReachableCycle cfg).card := by
  have h28 :=
    (tetrisSolvableValidFor_implies_safeReachableCycle_card_interval
      hcols h).1
  omega

/-- **Standard card-`<28` unsolvability**: specialization of Tick 261. -/
theorem safeReachableCycle_standard_card_lt_28_implies_not_solvable
    (h : (safeReachableCycle GameConfig.standard).card < 28) :
    ¬ TetrisSolvableValid :=
  safeReachableCycle_card_lt_28_implies_not_solvable (by decide) h

/-- **Tiny card-`<28` unsolvability**: specialization of Tick 261. -/
theorem safeReachableCycle_tiny_card_lt_28_implies_not_solvable
    (h : (safeReachableCycle GameConfig.tiny).card < 28) :
    ¬ TetrisSolvableValidFor GameConfig.tiny :=
  safeReachableCycle_card_lt_28_implies_not_solvable (by decide) h

/-- **Standard atlas lower bound**: solvability of standard Tetris
implies the canonical atlas holds at least 28 states. -/
theorem tetrisSolvableValid_implies_standard_safeReachableCycle_card_ge_28
    (h : TetrisSolvableValid) :
    28 ≤ (safeReachableCycle GameConfig.standard).card :=
  (tetrisSolvableValid_implies_standard_safeReachableCycle_card_interval h).1

/-- **Tiny atlas lower bound**: solvability of tiny implies the canonical
atlas holds at least 28 states. -/
theorem tetrisSolvableValidFor_tiny_implies_safeReachableCycle_card_ge_28
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    28 ≤ (safeReachableCycle GameConfig.tiny).card :=
  (tetrisSolvableValidFor_tiny_implies_safeReachableCycle_card_interval h).1

/-- **Canonical-pair `IsInitCycle` collapses to membership** (cols ≥ 4).
Combining the unconditional Obligations theorem
`safeReachableCycle_obligations` with the structural fact that
`IsInitCycle = Obligations ∧ init ∈ states`, deciding whether the
canonical pair is an init-cycle reduces to a single membership check. -/
theorem canonical_isInitCycle_iff_init_mem {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) :
    AdversarialClosedCycle.IsInitCycle cfg (safeReachableCycle cfg)
        (safeSolver cfg) ↔
    GameState.init ∈ safeReachableCycle cfg := by
  refine ⟨fun h => h.2, fun h => ⟨safeReachableCycle_obligations hcols, h⟩⟩

/-- Parametric atlas parity: every state in the canonical atlas has
cell-count `≡ 4k (mod cfg.cols)` for some k. Corollary of
`Reachable.exists_count_mod_cols`. -/
theorem safeReachableCycle_exists_count_mod_cols {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    ∃ n, g.board.count % cfg.cols = 4 * n % cfg.cols :=
  (safeReachableCycle_subset_reachable cfg g h).exists_count_mod_cols

/-- **Every atlas state has a valid placement for every bag piece.**
For any `g ∈ safeReachableCycle cfg` and `p ∈ g.bag`, there exists an
in-bounds placement playing piece `p`. Direct corollary of
`safe_has_valid_for_each_bag_piece` via the atlas → safe projection. -/
theorem safeReachableCycle_has_valid_placement {cfg : GameConfig}
    {g : GameState} (hg : g ∈ safeReachableCycle cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  safe_has_valid_for_each_bag_piece
    (safeReachableCycle_subset_safe cfg g hg) hp

/-- **Constructive atlas-closure witness** (cols ≥ 4). For any atlas
state and any bag piece, there exists a placement that
(i) plays the right piece, (ii) is in-bounds, (iii) lands in the
atlas. The witness is `safeSolver cfg g p`, packaging the canonical
extractor together with its three properties. -/
theorem safeReachableCycle_safeSolver_witness {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {g : GameState}
    (hg : g ∈ safeReachableCycle cfg) {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement,
      pl.piece = p ∧ pl.Valid cfg ∧
        adversarialStep cfg g p pl ∈ safeReachableCycle cfg :=
  ⟨safeSolver cfg g p, safeSolver_piece cfg g p,
   (safeSolver_validSolver hcols g p hp).2,
   safeReachableCycle_safeSolver_step hg hp⟩

/-- **`solverReachable` ⊆ `inFieldStates` under a `SolvesTetrisValid`
solver.** Composes Ticks 298/297 with
`reachable_not_lost_mem_inFieldStates`. -/
theorem solverReachable_mem_inFieldStates_of_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) :
    g ∈ inFieldStates cfg :=
  reachable_not_lost_mem_inFieldStates
    (solverReachable_reachable_of_solvesTetrisValid hsv hr)
    (solverReachable_not_lost_of_solvesTetrisValid hsv hr)

/-- **Finiteness of `solverReachable` under a `SolvesTetrisValid`
solver.** The forward-closure set is bounded by the universe Finset
and therefore finite. -/
theorem solverReachable_finite_of_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) :
    {g | solverReachable σ g}.Finite :=
  Set.Finite.subset (inFieldStates cfg).finite_toSet
    (fun _ hr =>
      solverReachable_mem_inFieldStates_of_solvesTetrisValid hsv hr)

/-- **`solverReachable` card bound**: under a valid solving solver,
the forward-closure set's `ncard` is bounded by `|inFieldStates cfg|`. -/
theorem solverReachable_ncard_le_of_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) :
    {g | solverReachable σ g}.ncard ≤ (inFieldStates cfg).card := by
  rw [← Set.ncard_coe_finset]
  exact Set.ncard_le_ncard
    (fun _ hr =>
      solverReachable_mem_inFieldStates_of_solvesTetrisValid hsv hr)
    (inFieldStates cfg).finite_toSet

/-- **Finset version of `solverReachable` under a `SolvesTetrisValid`
solver.** Materialises Tick 302's `Set.Finite` as a concrete Finset
via `Set.Finite.toFinset`. -/
noncomputable def solverReachableFinset {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    Finset GameState :=
  (solverReachable_finite_of_solvesTetrisValid hsv).toFinset

/-- Membership in `solverReachableFinset` is exactly `solverReachable`. -/
@[simp] theorem mem_solverReachableFinset {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) (g : GameState) :
    g ∈ solverReachableFinset hsv ↔ solverReachable σ g := by
  unfold solverReachableFinset
  rw [Set.Finite.mem_toFinset]; rfl

/-- `init` is always in `solverReachableFinset`. -/
theorem init_mem_solverReachableFinset {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    GameState.init ∈ solverReachableFinset hsv :=
  (mem_solverReachableFinset hsv _).mpr solverReachable.init

/-- `solverReachableFinset hsv ⊆ inFieldStates cfg`. -/
theorem solverReachableFinset_subset_inFieldStates {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ⊆ inFieldStates cfg := fun g hg =>
  solverReachable_mem_inFieldStates_of_solvesTetrisValid hsv
    ((mem_solverReachableFinset hsv g).mp hg)

/-- Bag dichotomy for solver-atlas states: bag = full or card < 7. -/
theorem solverReachableFinset_bag_eq_full_or_card_lt_seven
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_h : g ∈ solverReachableFinset hsv) :
    g.bag = Bag.full ∨ g.bag.card < 7 :=
  g.bag_eq_full_or_card_lt_seven

/-- Solver-atlas dichotomy in `≤ 6` form. -/
theorem solverReachableFinset_bag_eq_full_or_card_le_six
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_h : g ∈ solverReachableFinset hsv) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 :=
  g.bag_eq_full_or_card_le_six

/-- Solver-atlas states have nonempty bags. -/
theorem solverReachableFinset_bag_nonempty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) : g.bag.Nonempty :=
  solverReachable_bag_nonempty ((mem_solverReachableFinset hsv g).mp h)

/-- Solver-atlas states have non-empty bags (`≠ ∅` form). -/
theorem solverReachableFinset_bag_ne_empty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) : g.bag ≠ ∅ :=
  (solverReachableFinset_bag_nonempty hsv h).ne_empty

/-- Every solver-atlas state has a drawable piece. -/
theorem solverReachableFinset_exists_canDraw {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    ∃ p : Piece, g.bag.canDraw p :=
  Bag.exists_canDraw_of_nonempty (solverReachableFinset_bag_nonempty hsv h)

/-- Canonical drawable-piece witness for a solver-atlas state. -/
noncomputable def solverReachableFinset_someBagElement {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) : Piece :=
  g.bag.someElement (solverReachableFinset_bag_nonempty hsv h)

theorem solverReachableFinset_someBagElement_mem {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    solverReachableFinset_someBagElement hsv h ∈ g.bag :=
  Bag.someElement_mem _ _

theorem solverReachableFinset_someBagElement_canDraw {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    g.bag.canDraw (solverReachableFinset_someBagElement hsv h) :=
  Bag.someElement_canDraw _ _

/-- Solver-atlas state's bag card is positive. -/
theorem solverReachableFinset_bag_card_pos {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) : 0 < g.bag.card :=
  Finset.card_pos.mpr (solverReachableFinset_bag_nonempty hsv h)

/-- Solver-atlas state's bag card ≤ 7. -/
theorem solverReachableFinset_bag_card_le_seven {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (_h : g ∈ solverReachableFinset hsv) : g.bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- Solver-atlas state's bag is either ⊂ `full` or = `full`. -/
theorem solverReachableFinset_bag_ssubset_full_or_eq_full {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    g.bag ⊂ Bag.full ∨ g.bag = Bag.full :=
  ((mem_solverReachableFinset hsv g).mp h).bag_ssubset_full_or_eq_full

/-- Bundled `[1, 7]` solver-atlas bag-card interval. -/
theorem solverReachableFinset_bag_card_interval {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    1 ≤ g.bag.card ∧ g.bag.card ≤ 7 :=
  ⟨solverReachableFinset_bag_card_pos hsv h,
   solverReachableFinset_bag_card_le_seven hsv h⟩

/-- Bag-card-7-iff-full for solver-atlas states. -/
theorem solverReachableFinset_bag_card_eq_seven_iff_bag_eq_full
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_h : g ∈ solverReachableFinset hsv) :
    g.bag.card = 7 ↔ g.bag = Bag.full :=
  Bag.card_eq_seven_iff_eq_full g.bag

/-- `solverReachableFinset hsv` has card ≤ `|inFieldStates cfg|`. -/
theorem solverReachableFinset_card_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤ (inFieldStates cfg).card :=
  Finset.card_le_card (solverReachableFinset_subset_inFieldStates hsv)

/-- Closed-form upper bound on the solver-atlas card. -/
theorem solverReachableFinset_card_le_pow {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤ 2 ^ (cfg.cols * cfg.rows) * 128 :=
  inFieldStates_card cfg ▸ solverReachableFinset_card_le hsv

/-- **Set-level characterization** of `solverReachableFinset`: as a
Set, it equals `{g | solverReachable σ g}`. -/
theorem solverReachableFinset_coe_eq {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState) =
      {g | solverReachable σ g} := by
  ext g
  exact mem_solverReachableFinset hsv g

/-- Set-level inclusion of solver atlas in universe. -/
theorem solverReachableFinset_coe_subset_inFieldStates_coe
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState) ⊆ ↑(inFieldStates cfg) :=
  Finset.coe_subset.mpr (solverReachableFinset_subset_inFieldStates hsv)


/-- `solverReachableFinset hsv` is nonempty (contains `init`). -/
theorem solverReachableFinset_nonempty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).Nonempty :=
  ⟨GameState.init, init_mem_solverReachableFinset hsv⟩

/-- `solverReachableFinset hsv ≠ ∅`. -/
theorem solverReachableFinset_ne_empty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ≠ ∅ :=
  (solverReachableFinset_nonempty hsv).ne_empty

/-- `solverReachableFinset hsv` has positive card. -/
theorem solverReachableFinset_card_pos {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    0 < (solverReachableFinset hsv).card :=
  Finset.card_pos.mpr (solverReachableFinset_nonempty hsv)

/-- `solverReachableFinset hsv` has nonzero card (Nat alias). -/
theorem solverReachableFinset_card_ne_zero {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≠ 0 :=
  Nat.ne_of_gt (solverReachableFinset_card_pos hsv)

/-- `1 ≤ card` form of solver-atlas positivity. -/
theorem solverReachableFinset_card_ge_one {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    1 ≤ (solverReachableFinset hsv).card :=
  solverReachableFinset_card_pos hsv

/-- `solverReachableFinset` is closed under one σ-step. -/
theorem solverReachableFinset_step {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    adversarialStep cfg g p (σ g p) ∈ solverReachableFinset hsv :=
  (mem_solverReachableFinset hsv _).mpr
    (solverReachable.step p ((mem_solverReachableFinset hsv g).mp hg) hp)

/-- **Cycle Obligations for `solverReachableFinset`.** Any valid
solving solver `σ` together with its `solverReachableFinset` satisfies
the three `AdversarialClosedCycle.Obligations` clauses:
- no atlas state is lost (Tick 297);
- `σ`'s output has the correct piece + is Valid (`ValidSolver`);
- closed under σ-step (Tick 307). -/
theorem solverReachableFinset_obligations {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    AdversarialClosedCycle.Obligations cfg
      (solverReachableFinset hsv) σ := by
  refine ⟨?_, ?_, ?_⟩
  · intro g hg
    exact solverReachable_not_lost_of_solvesTetrisValid hsv
      ((mem_solverReachableFinset hsv g).mp hg)
  · intro g _ p hp
    exact hsv.1 g p hp
  · intro _ hg _ hp
    exact solverReachableFinset_step hsv hg hp

/-- **`solverReachableFinset` is an init-cycle for any valid solving
solver.** Combines `solverReachableFinset_obligations` (Tick 308) with
`init_mem_solverReachableFinset` (Tick 305). Every `SolvesTetrisValid`
solver automatically witnesses an `AdversarialClosedCycle.IsInitCycle`
via its own forward-closure. -/
theorem solverReachableFinset_isInitCycle {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    AdversarialClosedCycle.IsInitCycle cfg
      (solverReachableFinset hsv) σ :=
  ⟨solverReachableFinset_obligations hsv,
   init_mem_solverReachableFinset hsv⟩

/-- **`solverReachableFinset` has card ≥ 28.** Any valid solving solver's
forward-closure Finset is at least 28 states large — direct consequence of
Tick 309 and `AdversarialClosedCycle.isInitCycle_card_ge_28`. -/
theorem solverReachableFinset_card_ge_28 {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    28 ≤ (solverReachableFinset hsv).card :=
  AdversarialClosedCycle.isInitCycle_card_ge_28
    (solverReachableFinset_isInitCycle hsv)

/-- **Two-sided card interval for `solverReachableFinset`**: any valid
solving solver's forward-closure Finset satisfies
`28 ≤ card ≤ |inFieldStates cfg|`. -/
theorem solverReachableFinset_card_interval {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card ≤ (inFieldStates cfg).card :=
  ⟨solverReachableFinset_card_ge_28 hsv,
   solverReachableFinset_card_le hsv⟩

/-- **Strict subsetting for `solverReachableFinset`**: the empty-bag /
empty-board state lives in `inFieldStates` (Tick 269) but is not
`Reachable`, hence not in any valid solving solver's forward-closure
Finset (Tick 298). -/
theorem solverReachableFinset_strict_subset_inFieldStates
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ⊂ inFieldStates cfg := by
  refine ⟨solverReachableFinset_subset_inFieldStates hsv, ?_⟩
  intro hsub
  have hmem := hsub (empty_bag_empty_board_mem_inFieldStates cfg)
  rw [mem_solverReachableFinset] at hmem
  exact empty_bag_empty_board_not_reachable cfg
    (solverReachable_reachable_of_solvesTetrisValid hsv hmem)

/-- **Strict card inequality** for `solverReachableFinset`: any valid
solving solver's atlas Finset has *strictly* fewer states than the
universe. -/
theorem solverReachableFinset_card_lt {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card < (inFieldStates cfg).card :=
  Finset.card_lt_card (solverReachableFinset_strict_subset_inFieldStates hsv)

/-- Closed-form strict bound for solver atlas card. -/
theorem solverReachableFinset_card_lt_pow {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card < 2 ^ (cfg.cols * cfg.rows) * 128 :=
  inFieldStates_card cfg ▸ solverReachableFinset_card_lt hsv

/-- **Strict two-sided interval for `solverReachableFinset`**: any valid
solving solver's atlas Finset has card in `[28, |inFieldStates|)`. -/
theorem solverReachableFinset_card_strict_interval {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card < (inFieldStates cfg).card :=
  ⟨solverReachableFinset_card_ge_28 hsv,
   solverReachableFinset_card_lt hsv⟩

/-- Closed-form solver-atlas card interval `[28, 2^(cols*rows)*128)`. -/
theorem solverReachableFinset_card_pow_interval {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card < 2 ^ (cfg.cols * cfg.rows) * 128 :=
  ⟨solverReachableFinset_card_ge_28 hsv,
   solverReachableFinset_card_lt_pow hsv⟩

/-- **`solverReachableFinset` is contained in `safeReachableCycle`** —
under any valid solving solver, every `solverReachable` state is also
in the canonical safe atlas (it is reachable, in-field, and safe via
the solver's own no-loss guarantee). Bridges arbitrary-solver atlases
to the canonical one. -/
theorem solverReachableFinset_subset_safeReachableCycle {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ⊆ safeReachableCycle cfg := by
  intro g hg
  have hr := (mem_solverReachableFinset hsv g).mp hg
  rw [mem_safeReachableCycle_iff_reachable_and_safe]
  refine ⟨solverReachable_reachable_of_solvesTetrisValid hsv hr, ?_⟩
  -- need: g ∈ safe cfg
  -- via init_safe_iff_exists_solvesTetrisValid : init safe ↔ ∃ ValidSolver+SolvesTetris.
  -- Strategy: any IsInitCycle's states ⊆ safe (Tick 195 style).
  -- Here (solverReachableFinset hsv, σ) is an IsInitCycle (Tick 309).
  exact AdversarialClosedCycle.isInitCycle_states_subset_safe
    (solverReachableFinset_isInitCycle hsv) (Finset.mem_coe.mpr hg)

/-- **Card bound**: any valid solving solver's atlas Finset is at most
the canonical safe atlas in size. Card-level form of Tick 314. -/
theorem solverReachableFinset_card_le_safeReachableCycle {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤ (safeReachableCycle cfg).card :=
  Finset.card_le_card (solverReachableFinset_subset_safeReachableCycle hsv)

/-- Set-level inclusion of solver atlas in canonical atlas. -/
theorem solverReachableFinset_coe_subset_safeReachableCycle_coe
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState) ⊆
      ↑(safeReachableCycle cfg) :=
  Finset.coe_subset.mpr (solverReachableFinset_subset_safeReachableCycle hsv)

/-- Set-level finiteness of `solverReachableFinset` coerced to a Set. -/
theorem solverReachableFinset_coe_finite {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState).Finite :=
  (solverReachableFinset hsv).finite_toSet


/-- **Every `solverReachable` state under a valid solving solver is
safe.** Predicate-level form of Tick 314, bypassing the Finset
materialisation. Useful when only the safe-set membership is needed. -/
theorem solverReachable_safe_of_solvesTetrisValid {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    g ∈ safe cfg :=
  safeReachableCycle_subset_safe cfg g
    (solverReachableFinset_subset_safeReachableCycle hsv
      ((mem_solverReachableFinset hsv g).mpr hr))

/-- **Full 5-tuple invariants for `solverReachable` under a valid
solving solver**: Reachable, board WF, bag nonempty, not lost,
*and* in the safe set. Extends Tick 300 with the safe-set
membership from Tick 316. -/
theorem solverReachable_full_invariants_of_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (hr : solverReachable σ g) :
    Reachable cfg g ∧ Board.WF cfg g.board ∧ g.bag.Nonempty ∧
      ¬ g.lost cfg ∧ g ∈ safe cfg :=
  let ⟨h1, h2, h3, h4⟩ :=
    solverReachable_invariants_of_solvesTetrisValid hsv hr
  ⟨h1, h2, h3, h4, solverReachable_safe_of_solvesTetrisValid hsv hr⟩

/-- **`solverReachableFinset` is contained in `safe cfg`.** Finset-level
form of Tick 316. -/
theorem solverReachableFinset_subset_safe {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∀ g ∈ solverReachableFinset hsv, g ∈ safe cfg := fun g hg =>
  solverReachable_safe_of_solvesTetrisValid hsv
    ((mem_solverReachableFinset hsv g).mp hg)

/-- Solver atlas coe ⊆ `safe`. -/
theorem solverReachableFinset_coe_subset_safe {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState) ⊆ safe cfg :=
  fun _ hg => solverReachableFinset_subset_safe hsv _ (Finset.mem_coe.mp hg)

/-- Solver atlas coe ⊆ Reachable Set. -/
theorem solverReachableFinset_coe_subset_reachable {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (↑(solverReachableFinset hsv) : Set GameState) ⊆ {g | Reachable cfg g} :=
  fun _ hg => solverReachable_reachable_of_solvesTetrisValid hsv
    ((mem_solverReachableFinset hsv _).mp (Finset.mem_coe.mp hg))

/-- **Empty-bag empty-board state is not in any solver's atlas**.
Combines Tick 319 with Tick 304's membership iff. -/
theorem empty_bag_empty_board_not_mem_solverReachableFinset
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (⟨Board.empty, (∅ : Bag)⟩ : GameState) ∉ solverReachableFinset hsv :=
  fun h => empty_bag_empty_board_not_solverReachable
    ((mem_solverReachableFinset hsv _).mp h)

/-- **Contrapositive — `¬ Reachable → ¬ solverReachable` under any
valid solving solver.** Pairs with parity/contrapositive lemmas in
`Theorems/Gameplay.lean` (e.g. `not_reachable_standard_of_count_odd`)
to refute `solverReachable` from cell-count arithmetic. -/
theorem not_solverReachable_of_not_reachable_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : ¬ Reachable cfg g) : ¬ solverReachable σ g := fun hr =>
  h (solverReachable_reachable_of_solvesTetrisValid hsv hr)

/-- **Standard parity pruning**: under any valid solving solver on the
standard config, states with odd cell counts are not `solverReachable`. -/
theorem not_solverReachable_standard_of_count_odd
    {σ : Solver GameConfig.standard}
    (hsv : SolvesTetrisValid GameConfig.standard σ)
    {g : GameState} (hodd : g.board.count % 2 = 1) :
    ¬ solverReachable σ g :=
  not_solverReachable_of_not_reachable_solvesTetrisValid hsv
    (not_reachable_standard_of_count_odd hodd)

/-- **Tiny parity pruning**: under any valid solving solver on the
tiny config, states whose count is not divisible by 4 are not
`solverReachable`. -/
theorem not_solverReachable_tiny_of_count_not_mod_four
    {σ : Solver GameConfig.tiny}
    (hsv : SolvesTetrisValid GameConfig.tiny σ)
    {g : GameState} (h4 : g.board.count % 4 ≠ 0) :
    ¬ solverReachable σ g :=
  not_solverReachable_of_not_reachable_solvesTetrisValid hsv
    (not_reachable_tiny_of_count_not_mod_four h4)

/-- **Every adversarial-trace state lives in `solverReachableFinset`**
under any valid solving solver. Direct corollary of
`adversarialTrace_solverReachable` + the Tick 304 membership iff. -/
theorem adversarialTrace_mem_solverReachableFinset
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ solverReachableFinset hsv :=
  (mem_solverReachableFinset hsv _).mpr
    (adversarialTrace_solverReachable σ hl n)

/-- **Every adversarial-trace state under a valid solving solver lives
in the canonical atlas** `safeReachableCycle cfg`. Composes Tick 344
with Tick 314. -/
theorem adversarialTrace_mem_safeReachableCycle
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ safeReachableCycle cfg :=
  solverReachableFinset_subset_safeReachableCycle hsv
    (adversarialTrace_mem_solverReachableFinset hsv hl n)

/-- **Every adversarial-trace state under any `SolvesTetrisValid`
solver is `safe`.** Generalises `adversarialTrace_mem_safe` (which is
`safeSolver`-specific) to arbitrary valid solving solvers. Direct
corollary of Tick 345 + `safeReachableCycle_subset_safe`. -/
theorem adversarialTrace_solvesTetrisValid_mem_safe
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg σ s GameState.init n ∈ safe cfg :=
  safeReachableCycle_subset_safe cfg _
    (adversarialTrace_mem_safeReachableCycle hsv hl n)

/-- **Every adversarial-trace state under any `SolvesTetrisValid`
solver is `Reachable`.** Generalises the `safeSolver`-specific
`adversarialTrace_reachable` (SafeSet.lean:368). -/
theorem adversarialTrace_solvesTetrisValid_reachable
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg σ s GameState.init n) :=
  safeReachableCycle_subset_reachable cfg _
    (adversarialTrace_mem_safeReachableCycle hsv hl n)

/-- **Bundled adversarial-trace invariants under `SolvesTetrisValid`**:
`Reachable ∧ ∈ safe ∧ ∈ inFieldStates ∧ ¬ lost`. -/
theorem adversarialTrace_solvesTetrisValid_invariants
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    Reachable cfg (adversarialTrace cfg σ s GameState.init n) ∧
    adversarialTrace cfg σ s GameState.init n ∈ safe cfg ∧
    adversarialTrace cfg σ s GameState.init n ∈ inFieldStates cfg ∧
    ¬ (adversarialTrace cfg σ s GameState.init n).lost cfg :=
  let hmem := adversarialTrace_mem_safeReachableCycle hsv hl n
  ⟨safeReachableCycle_subset_reachable cfg _ hmem,
   safeReachableCycle_subset_safe cfg _ hmem,
   safeReachableCycle_subset_inFieldStates cfg hmem,
   safe_not_lost (safeReachableCycle_subset_safe cfg _ hmem)⟩

/-- **Pigeonhole on `adversarialTrace`** under any valid solving solver:
for any legal piece sequence, the trace length `|solverReachableFinset|
+ 1` must revisit some state. -/
theorem adversarialTrace_pigeonhole {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i ≤ (solverReachableFinset hsv).card ∧
      j ≤ (solverReachableFinset hsv).card ∧ i ≠ j ∧
      adversarialTrace cfg σ s GameState.init i =
        adversarialTrace cfg σ s GameState.init j := by
  set N := (solverReachableFinset hsv).card
  have hmap : ∀ k ∈ Finset.range (N + 1),
      adversarialTrace cfg σ s GameState.init k ∈ solverReachableFinset hsv :=
    fun k _ => (mem_solverReachableFinset hsv _).mpr
      (adversarialTrace_solverReachable σ hl k)
  have hcard : N < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; omega
  obtain ⟨i, hi, j, hj, hne, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rw [Finset.mem_range] at hi hj
  exact ⟨i, j, by omega, by omega, hne, heq⟩

/-- **Ordered pigeonhole on `adversarialTrace`**: same as Tick 342 but
with `i < j`. -/
theorem adversarialTrace_pigeonhole_lt {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i j : ℕ, i < j ∧ j ≤ (solverReachableFinset hsv).card ∧
      adversarialTrace cfg σ s GameState.init i =
        adversarialTrace cfg σ s GameState.init j := by
  obtain ⟨i, j, hi, hj, hne, heq⟩ := adversarialTrace_pigeonhole hsv hl
  rcases lt_or_gt_of_ne hne with h | h
  · exact ⟨i, j, h, hj, heq⟩
  · exact ⟨j, i, h, hi, heq.symm⟩

/-- **Adversarial-trace period existence**: there is a base index `i`
and a positive period `d ≤ |solverReachableFinset|` such that
`trace_i = trace_(i+d)`. Mirrors Tick 337 (`ClosedCycle.trace_exists_period`)
for adversarial traces. -/
theorem adversarialTrace_exists_period {cfg : GameConfig} {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {s : ℕ → Piece} (hl : LegalSequence s) :
    ∃ i d : ℕ, 0 < d ∧ d ≤ (solverReachableFinset hsv).card ∧
      adversarialTrace cfg σ s GameState.init i =
        adversarialTrace cfg σ s GameState.init (i + d) := by
  obtain ⟨i, j, hlt, hj, heq⟩ := adversarialTrace_pigeonhole_lt hsv hl
  refine ⟨i, j - i, ?_, ?_, ?_⟩
  · omega
  · omega
  · rw [show i + (j - i) = j from by omega]; exact heq

/-- **Standard concrete bound**: any valid solving solver on the
standard config has its atlas Finset in `[28, 2^207)`. -/
theorem solverReachableFinset_standard_card_strict_interval
    {σ : Solver GameConfig.standard}
    (hsv : SolvesTetrisValid GameConfig.standard σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card < 2 ^ 200 * 128 := by
  obtain ⟨h1, h2⟩ := solverReachableFinset_card_strict_interval hsv
  exact ⟨h1, standard_inFieldStates_card ▸ h2⟩

/-- **Tiny concrete bound**: any valid solving solver on the tiny
config has its atlas Finset in `[28, 8 388 608)`. -/
theorem solverReachableFinset_tiny_card_strict_interval
    {σ : Solver GameConfig.tiny}
    (hsv : SolvesTetrisValid GameConfig.tiny σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card < 8388608 := by
  obtain ⟨h1, h2⟩ := solverReachableFinset_card_strict_interval hsv
  exact ⟨h1, tiny_inFieldStates_card ▸ h2⟩

/-- **Ergonomic weak bound**: any valid solving solver's atlas Finset
has at least 2 distinct states (follows from `28 ≤ card`). -/
theorem solverReachableFinset_card_ge_two {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    2 ≤ (solverReachableFinset hsv).card := by
  have := solverReachableFinset_card_ge_28 hsv
  omega

/-- Strict `1 < card` form for the solver atlas (Tick 507 analog). -/
theorem solverReachableFinset_one_lt_card {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    1 < (solverReachableFinset hsv).card :=
  solverReachableFinset_card_ge_two hsv

/-- Solver-atlas card is not 1 (Tick 509 analog). -/
theorem solverReachableFinset_card_ne_one {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≠ 1 := by
  have := solverReachableFinset_one_lt_card hsv
  omega

/-- Solver-atlas contains two distinct states (Tick 511 analog). -/
theorem solverReachableFinset_exists_two_distinct {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ a ∈ solverReachableFinset hsv,
    ∃ b ∈ solverReachableFinset hsv, a ≠ b :=
  Finset.one_lt_card.mp (solverReachableFinset_one_lt_card hsv)

/-- **Solvability ⇒ canonical atlas nonempty.** Any
`TetrisSolvableValidFor cfg` proof (with `4 ≤ cfg.cols`) furnishes a
nonempty canonical atlas. -/
theorem safeReachableCycle_nonempty_of_solvable {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : TetrisSolvableValidFor cfg) :
    (safeReachableCycle cfg).Nonempty :=
  ⟨GameState.init,
   (tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp h⟩

/-- **Solvability ⇒ atlas card positive.** Numerical form of Tick 324. -/
theorem safeReachableCycle_card_pos_of_solvable {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (h : TetrisSolvableValidFor cfg) :
    0 < (safeReachableCycle cfg).card :=
  Finset.card_pos.mpr (safeReachableCycle_nonempty_of_solvable hcols h)

/-- **Atlas board count bound (parametric)**: every state in the
canonical atlas has board cell count `≤ cfg.cols * cfg.rows`. Composes
`Reachable.not_lost_board_card_le` with the atlas projections. -/
theorem safeReachableCycle_board_card_le {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.board.card ≤ cfg.cols * cfg.rows :=
  (safeReachableCycle_subset_reachable cfg g h).not_lost_board_card_le
    (safe_not_lost (safeReachableCycle_subset_safe cfg g h))

/-- `.count` alias of `safeReachableCycle_board_card_le`. -/
theorem safeReachableCycle_board_count_le {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.board.count ≤ cfg.cols * cfg.rows :=
  safeReachableCycle_board_card_le h

/-- Per-step count bound at any atlas state. -/
theorem safeReachableCycle_step_count_le {cfg : GameConfig}
    {g : GameState} (_h : g ∈ safeReachableCycle cfg) (pl : Placement) :
    (g.step cfg pl).board.count ≤ g.board.count + 4 :=
  GameState.step_count_le g pl

/-- Solver-atlas analog. -/
theorem solverReachableFinset_step_count_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_h : g ∈ solverReachableFinset hsv) (pl : Placement) :
    (g.step cfg pl).board.count ≤ g.board.count + 4 :=
  GameState.step_count_le g pl

/-- **Atlas contains a max-count state**: when init is safe, the
atlas (nonempty) has a state attaining the supremum of board
counts within the atlas. -/
theorem safeReachableCycle_exists_max_count {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, ∀ g' ∈ safeReachableCycle cfg,
      g'.board.count ≤ g.board.count :=
  Finset.exists_max_image (safeReachableCycle cfg)
    (fun g => g.board.count) (safeReachableCycle_nonempty h)

/-- **Atlas contains a min-count state** (dual of Tick 678). The
min is realised at init itself (count = 0). -/
theorem safeReachableCycle_exists_min_count {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, ∀ g' ∈ safeReachableCycle cfg,
      g.board.count ≤ g'.board.count :=
  Finset.exists_min_image (safeReachableCycle cfg)
    (fun g => g.board.count) (safeReachableCycle_nonempty h)

/-- Solver-atlas analog of Tick 678 (max-count). -/
theorem solverReachableFinset_exists_max_count {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, ∀ g' ∈ solverReachableFinset hsv,
      g'.board.count ≤ g.board.count :=
  Finset.exists_max_image (solverReachableFinset hsv)
    (fun g => g.board.count) (solverReachableFinset_nonempty hsv)

/-- Solver-atlas analog of Tick 679 (min-count). -/
theorem solverReachableFinset_exists_min_count {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, ∀ g' ∈ solverReachableFinset hsv,
      g.board.count ≤ g'.board.count :=
  Finset.exists_min_image (solverReachableFinset hsv)
    (fun g => g.board.count) (solverReachableFinset_nonempty hsv)

/-- Atlas contains a max-bag-card state. -/
theorem safeReachableCycle_exists_max_bag_card {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, ∀ g' ∈ safeReachableCycle cfg,
      g'.bag.card ≤ g.bag.card :=
  Finset.exists_max_image (safeReachableCycle cfg)
    (fun g => g.bag.card) (safeReachableCycle_nonempty h)

/-- Atlas contains a min-bag-card state. -/
theorem safeReachableCycle_exists_min_bag_card {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, ∀ g' ∈ safeReachableCycle cfg,
      g.bag.card ≤ g'.bag.card :=
  Finset.exists_min_image (safeReachableCycle cfg)
    (fun g => g.bag.card) (safeReachableCycle_nonempty h)

/-- Solver-atlas analog: max-bag-card state exists. -/
theorem solverReachableFinset_exists_max_bag_card {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, ∀ g' ∈ solverReachableFinset hsv,
      g'.bag.card ≤ g.bag.card :=
  Finset.exists_max_image (solverReachableFinset hsv)
    (fun g => g.bag.card) (solverReachableFinset_nonempty hsv)

/-- Solver-atlas analog: min-bag-card state exists. -/
theorem solverReachableFinset_exists_min_bag_card {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, ∀ g' ∈ solverReachableFinset hsv,
      g.bag.card ≤ g'.bag.card :=
  Finset.exists_min_image (solverReachableFinset hsv)
    (fun g => g.bag.card) (solverReachableFinset_nonempty hsv)

/-- **Atlas min count = 0**: every atlas state has count ≥ 0 (Nat),
and init realises the minimum (count 0). Concrete extremum
identification. -/
theorem safeReachableCycle_min_count_eq_zero {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, g.board.count = 0 ∧
    ∀ g' ∈ safeReachableCycle cfg, g.board.count ≤ g'.board.count := by
  refine ⟨GameState.init, init_mem_safeReachableCycle h, ?_, fun _ _ => ?_⟩
  · rw [GameState.init_board]; rfl
  · simp [GameState.init_board]

/-- **Atlas max bag-card = 7**: every atlas state has bag.card ≤ 7,
and init realises the maximum (bag = full, card = 7). -/
theorem safeReachableCycle_max_bag_card_eq_seven {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, g.bag.card = 7 ∧
    ∀ g' ∈ safeReachableCycle cfg, g'.bag.card ≤ g.bag.card := by
  refine ⟨GameState.init, init_mem_safeReachableCycle h, ?_, fun g' hg' => ?_⟩
  · rw [GameState.init_bag, Bag.full_card]
  · rw [GameState.init_bag, Bag.full_card]
    exact Bag.card_le_seven _

/-- Solver-atlas analog: init realises max bag.card = 7. -/
theorem solverReachableFinset_max_bag_card_eq_seven {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, g.bag.card = 7 ∧
    ∀ g' ∈ solverReachableFinset hsv, g'.bag.card ≤ g.bag.card := by
  refine ⟨GameState.init, init_mem_solverReachableFinset hsv, ?_,
          fun g' _ => ?_⟩
  · rw [GameState.init_bag, Bag.full_card]
  · rw [GameState.init_bag, Bag.full_card]
    exact Bag.card_le_seven _

/-- Solver-atlas analog: init realises min count = 0. -/
theorem solverReachableFinset_min_count_eq_zero {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, g.board.count = 0 ∧
    ∀ g' ∈ solverReachableFinset hsv, g.board.count ≤ g'.board.count := by
  refine ⟨GameState.init, init_mem_solverReachableFinset hsv, ?_,
          fun _ _ => ?_⟩
  · rw [GameState.init_board]; rfl
  · simp [GameState.init_board]

/-- **Atlas init-uniqueness**: within the atlas, only `init` has
both count = 0 and bag = full. -/
theorem safeReachableCycle_eq_init_of_count_zero_and_bag_full {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg)
    (hc : g.board.count = 0) (hb : g.bag = Bag.full) :
    g = GameState.init :=
  (safeReachableCycle_subset_reachable cfg g h).eq_init_of_bag_full_of_count_zero hc hb

/-- Solver-atlas analog of init-uniqueness. -/
theorem solverReachableFinset_eq_init_of_count_zero_and_bag_full {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (_h : g ∈ solverReachableFinset hsv)
    (hc : g.board.count = 0) (hb : g.bag = Bag.full) :
    g = GameState.init :=
  (GameState.eq_init_iff_count_zero_and_bag_full g).mpr ⟨hc, hb⟩

/-- **Atlas filter card = 1**: when init is safe, the filter of
the atlas by `count = 0 ∧ bag = full` is exactly `{init}` and
has card 1. -/
theorem safeReachableCycle_filter_count_zero_and_bag_full_card_eq_one
    {cfg : GameConfig} (h : GameState.init ∈ safe cfg) :
    ((safeReachableCycle cfg).filter
        (fun g => g.board.count = 0 ∧ g.bag = Bag.full)).card = 1 := by
  rw [Finset.card_eq_one]
  refine ⟨GameState.init, Finset.ext fun g => ?_⟩
  simp only [Finset.mem_filter, Finset.mem_singleton]
  refine ⟨fun ⟨hmem, hc, hb⟩ =>
    safeReachableCycle_eq_init_of_count_zero_and_bag_full hmem hc hb, ?_⟩
  rintro rfl
  exact ⟨init_mem_safeReachableCycle h,
         by rw [GameState.init_board]; rfl,
         GameState.init_bag⟩

/-- Solver-atlas analog of Tick 689. -/
theorem solverReachableFinset_filter_count_zero_and_bag_full_card_eq_one
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).filter
        (fun g => g.board.count = 0 ∧ g.bag = Bag.full)).card = 1 := by
  rw [Finset.card_eq_one]
  refine ⟨GameState.init, Finset.ext fun g => ?_⟩
  simp only [Finset.mem_filter, Finset.mem_singleton]
  refine ⟨fun ⟨hmem, hc, hb⟩ =>
    solverReachableFinset_eq_init_of_count_zero_and_bag_full hsv hmem hc hb, ?_⟩
  rintro rfl
  exact ⟨init_mem_solverReachableFinset hsv,
         by rw [GameState.init_board]; rfl,
         GameState.init_bag⟩

/-- **Atlas distinct-board card bound**: the number of distinct
boards appearing among atlas states is bounded by the atlas size
(trivial via Finset.card_image_le). -/
theorem safeReachableCycle_image_board_card_le (cfg : GameConfig) :
    ((safeReachableCycle cfg).image GameState.board).card ≤
      (safeReachableCycle cfg).card :=
  Finset.card_image_le

/-- **Atlas distinct-bag card bound**: distinct bags among atlas
states ≤ atlas size. -/
theorem safeReachableCycle_image_bag_card_le (cfg : GameConfig) :
    ((safeReachableCycle cfg).image GameState.bag).card ≤
      (safeReachableCycle cfg).card :=
  Finset.card_image_le

/-- Solver-atlas analog: distinct boards ≤ solver atlas size. -/
theorem solverReachableFinset_image_board_card_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.board).card ≤
      (solverReachableFinset hsv).card :=
  Finset.card_image_le

/-- Solver-atlas analog: distinct bags ≤ solver atlas size. -/
theorem solverReachableFinset_image_bag_card_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.bag).card ≤
      (solverReachableFinset hsv).card :=
  Finset.card_image_le

/-- **Distinct atlas bags ≤ 2^7 = 128**: bags are `Finset Piece`
with 7 underlying values; the powerset bound applies. -/
theorem safeReachableCycle_image_bag_card_le_univ_card (cfg : GameConfig) :
    ((safeReachableCycle cfg).image GameState.bag).card ≤
      (Finset.univ : Finset Bag).card :=
  Finset.card_le_univ _

/-- Solver-atlas analog. -/
theorem solverReachableFinset_image_bag_card_le_univ_card {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.bag).card ≤
      (Finset.univ : Finset Bag).card :=
  Finset.card_le_univ _

/-- **Atlas distinct-(board, bag) pairs**: distinct
`(board, bag)` images bounded by atlas card. -/
theorem safeReachableCycle_image_state_pair_card_le (cfg : GameConfig) :
    ((safeReachableCycle cfg).image (fun g => (g.board, g.bag))).card ≤
      (safeReachableCycle cfg).card :=
  Finset.card_image_le

/-- **Atlas pair-image equality**: the `(board, bag)` map is
injective on atlas (since `GameState` is just `⟨board, bag⟩`),
so the image card equals the atlas card. -/
theorem safeReachableCycle_image_state_pair_card_eq (cfg : GameConfig) :
    ((safeReachableCycle cfg).image (fun g => (g.board, g.bag))).card =
      (safeReachableCycle cfg).card := by
  apply Finset.card_image_of_injective
  intro g g' h
  cases g; cases g'
  simp_all

/-- Solver-atlas analog of Tick 697. -/
theorem solverReachableFinset_image_state_pair_card_eq {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image (fun g => (g.board, g.bag))).card =
      (solverReachableFinset hsv).card := by
  apply Finset.card_image_of_injective
  intro g g' h
  cases g; cases g'
  simp_all

/-- **Atlas card ≤ distinct-boards × distinct-bags**: the pair
projection lives in the product of distinct boards and distinct
bags. -/
theorem safeReachableCycle_card_le_image_board_mul_image_bag
    (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤
      ((safeReachableCycle cfg).image GameState.board).card *
      ((safeReachableCycle cfg).image GameState.bag).card := by
  rw [← safeReachableCycle_image_state_pair_card_eq]
  have hsub :
      (safeReachableCycle cfg).image (fun g => (g.board, g.bag)) ⊆
      ((safeReachableCycle cfg).image GameState.board) ×ˢ
      ((safeReachableCycle cfg).image GameState.bag) := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨g, hg, rfl⟩ := hp
    simp only [Finset.mem_product, Finset.mem_image]
    exact ⟨⟨g, hg, rfl⟩, ⟨g, hg, rfl⟩⟩
  calc ((safeReachableCycle cfg).image (fun g => (g.board, g.bag))).card
      ≤ (((safeReachableCycle cfg).image GameState.board) ×ˢ
         ((safeReachableCycle cfg).image GameState.bag)).card :=
        Finset.card_le_card hsub
    _ = _ := Finset.card_product _ _

/-- Solver-atlas analog of Tick 699. -/
theorem solverReachableFinset_card_le_image_board_mul_image_bag
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤
      ((solverReachableFinset hsv).image GameState.board).card *
      ((solverReachableFinset hsv).image GameState.bag).card := by
  rw [← solverReachableFinset_image_state_pair_card_eq hsv]
  have hsub :
      (solverReachableFinset hsv).image (fun g => (g.board, g.bag)) ⊆
      ((solverReachableFinset hsv).image GameState.board) ×ˢ
      ((solverReachableFinset hsv).image GameState.bag) := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨g, hg, rfl⟩ := hp
    simp only [Finset.mem_product, Finset.mem_image]
    exact ⟨⟨g, hg, rfl⟩, ⟨g, hg, rfl⟩⟩
  calc ((solverReachableFinset hsv).image (fun g => (g.board, g.bag))).card
      ≤ (((solverReachableFinset hsv).image GameState.board) ×ˢ
         ((solverReachableFinset hsv).image GameState.bag)).card :=
        Finset.card_le_card hsub
    _ = _ := Finset.card_product _ _

/-- Chained bound: atlas card ≤ distinct-boards × |Finset Bag|.
Substitutes the second factor by its universe bound. -/
theorem safeReachableCycle_card_le_image_board_mul_bag_univ
    (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤
      ((safeReachableCycle cfg).image GameState.board).card *
      (Finset.univ : Finset Bag).card := by
  calc (safeReachableCycle cfg).card
      ≤ _ := safeReachableCycle_card_le_image_board_mul_image_bag cfg
    _ ≤ _ := Nat.mul_le_mul_left _
              (safeReachableCycle_image_bag_card_le_univ_card cfg)

/-- Solver-atlas analog. -/
theorem solverReachableFinset_card_le_image_board_mul_bag_univ
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤
      ((solverReachableFinset hsv).image GameState.board).card *
      (Finset.univ : Finset Bag).card := by
  calc (solverReachableFinset hsv).card
      ≤ _ := solverReachableFinset_card_le_image_board_mul_image_bag hsv
    _ ≤ _ := Nat.mul_le_mul_left _
              (solverReachableFinset_image_bag_card_le_univ_card hsv)

/-- Atlas distinct boards have empty board among them when init
is safe (board of init is `∅`). -/
theorem empty_mem_safeReachableCycle_image_board {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    Board.empty ∈ (safeReachableCycle cfg).image GameState.board := by
  rw [Finset.mem_image]
  refine ⟨GameState.init, init_mem_safeReachableCycle h, ?_⟩
  rfl

/-- Atlas distinct bags contain `Bag.full` when init is safe. -/
theorem full_mem_safeReachableCycle_image_bag {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    Bag.full ∈ (safeReachableCycle cfg).image GameState.bag := by
  rw [Finset.mem_image]
  refine ⟨GameState.init, init_mem_safeReachableCycle h, ?_⟩
  rfl

/-- Solver-atlas: empty board in image. -/
theorem empty_mem_solverReachableFinset_image_board {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    Board.empty ∈ (solverReachableFinset hsv).image GameState.board := by
  rw [Finset.mem_image]
  refine ⟨GameState.init, init_mem_solverReachableFinset hsv, ?_⟩
  rfl

/-- Solver-atlas: Bag.full in image. -/
theorem full_mem_solverReachableFinset_image_bag {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    Bag.full ∈ (solverReachableFinset hsv).image GameState.bag := by
  rw [Finset.mem_image]
  refine ⟨GameState.init, init_mem_solverReachableFinset hsv, ?_⟩
  rfl

/-- The image of `board` over atlas is nonempty (contains `∅`)
when init is safe. -/
theorem safeReachableCycle_image_board_nonempty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ((safeReachableCycle cfg).image GameState.board).Nonempty :=
  ⟨Board.empty, empty_mem_safeReachableCycle_image_board h⟩

/-- The image of `bag` over atlas is nonempty (contains `Bag.full`)
when init is safe. -/
theorem safeReachableCycle_image_bag_nonempty {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ((safeReachableCycle cfg).image GameState.bag).Nonempty :=
  ⟨Bag.full, full_mem_safeReachableCycle_image_bag h⟩

/-- Solver-atlas analog of image-board nonempty. -/
theorem solverReachableFinset_image_board_nonempty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.board).Nonempty :=
  ⟨Board.empty, empty_mem_solverReachableFinset_image_board hsv⟩

/-- Solver-atlas analog of image-bag nonempty. -/
theorem solverReachableFinset_image_bag_nonempty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.bag).Nonempty :=
  ⟨Bag.full, full_mem_solverReachableFinset_image_bag hsv⟩

/-- Atlas distinct-board image has positive card. -/
theorem safeReachableCycle_image_board_card_pos {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    0 < ((safeReachableCycle cfg).image GameState.board).card :=
  Finset.card_pos.mpr (safeReachableCycle_image_board_nonempty h)

/-- Atlas distinct-bag image has positive card. -/
theorem safeReachableCycle_image_bag_card_pos {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    0 < ((safeReachableCycle cfg).image GameState.bag).card :=
  Finset.card_pos.mpr (safeReachableCycle_image_bag_nonempty h)

/-- Solver-atlas board image card positive. -/
theorem solverReachableFinset_image_board_card_pos {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    0 < ((solverReachableFinset hsv).image GameState.board).card :=
  Finset.card_pos.mpr (solverReachableFinset_image_board_nonempty hsv)

/-- Solver-atlas bag image card positive. -/
theorem solverReachableFinset_image_bag_card_pos {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    0 < ((solverReachableFinset hsv).image GameState.bag).card :=
  Finset.card_pos.mpr (solverReachableFinset_image_bag_nonempty hsv)

/-- Distinct atlas boards ≤ inFieldStates card. Composes
`image_card_le` with `card_le_inFieldStates`. -/
theorem safeReachableCycle_image_board_card_le_inFieldStates_card
    (cfg : GameConfig) :
    ((safeReachableCycle cfg).image GameState.board).card ≤
      (inFieldStates cfg).card :=
  le_trans (safeReachableCycle_image_board_card_le cfg)
           (safeReachableCycle_card_le_inFieldStates cfg)

/-- Distinct atlas bags ≤ inFieldStates card. -/
theorem safeReachableCycle_image_bag_card_le_inFieldStates_card
    (cfg : GameConfig) :
    ((safeReachableCycle cfg).image GameState.bag).card ≤
      (inFieldStates cfg).card :=
  le_trans (safeReachableCycle_image_bag_card_le cfg)
           (safeReachableCycle_card_le_inFieldStates cfg)

/-- Solver atlas distinct boards ≤ |inFieldStates|. -/
theorem solverReachableFinset_image_board_card_le_inFieldStates_card
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.board).card ≤
      (inFieldStates cfg).card :=
  le_trans (solverReachableFinset_image_board_card_le hsv)
           (le_trans (solverReachableFinset_card_le_safeReachableCycle hsv)
                     (safeReachableCycle_card_le_inFieldStates cfg))

/-- Solver atlas distinct bags ≤ |inFieldStates|. -/
theorem solverReachableFinset_image_bag_card_le_inFieldStates_card
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ((solverReachableFinset hsv).image GameState.bag).card ≤
      (inFieldStates cfg).card :=
  le_trans (solverReachableFinset_image_bag_card_le hsv)
           (le_trans (solverReachableFinset_card_le_safeReachableCycle hsv)
                     (safeReachableCycle_card_le_inFieldStates cfg))

/-- Every atlas-projected board is WF. -/
theorem safeReachableCycle_image_board_wf {cfg : GameConfig}
    {b : Board} (h : b ∈ (safeReachableCycle cfg).image GameState.board) :
    Board.WF cfg b := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g hg).wf

/-- Every atlas-projected bag is nonempty. -/
theorem safeReachableCycle_image_bag_nonempty_pointwise {cfg : GameConfig}
    {bag : Bag} (h : bag ∈ (safeReachableCycle cfg).image GameState.bag) :
    bag.Nonempty := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g hg).bag_nonempty

/-- Every atlas-projected bag is a subset of `Bag.full`. -/
theorem safeReachableCycle_image_bag_subset_full {cfg : GameConfig}
    {bag : Bag} (h : bag ∈ (safeReachableCycle cfg).image GameState.bag) :
    bag ⊆ Bag.full := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g hg).bag_subset_full

/-- Solver-atlas: every projected board is WF. -/
theorem solverReachableFinset_image_board_wf {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {b : Board}
    (h : b ∈ (solverReachableFinset hsv).image GameState.board) :
    Board.WF cfg b := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g
          (solverReachableFinset_subset_safeReachableCycle hsv hg)).wf

/-- Solver-atlas: every projected bag is nonempty. -/
theorem solverReachableFinset_image_bag_nonempty_pointwise {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {bag : Bag}
    (h : bag ∈ (solverReachableFinset hsv).image GameState.bag) :
    bag.Nonempty := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g
          (solverReachableFinset_subset_safeReachableCycle hsv hg)).bag_nonempty

/-- Solver-atlas: every projected bag ⊆ full. -/
theorem solverReachableFinset_image_bag_subset_full {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) {bag : Bag}
    (h : bag ∈ (solverReachableFinset hsv).image GameState.bag) :
    bag ⊆ Bag.full := by
  rw [Finset.mem_image] at h
  obtain ⟨g, hg, rfl⟩ := h
  exact (safeReachableCycle_subset_reachable cfg g
          (solverReachableFinset_subset_safeReachableCycle hsv hg)).bag_subset_full

/-- Distinct atlas bag-card values are at most 7 (bag.card ∈ {1,...,7}). -/
theorem safeReachableCycle_image_bag_card_image_card_le_seven (cfg : GameConfig) :
    (((safeReachableCycle cfg).image GameState.bag).image Finset.card).card ≤ 7 := by
  calc (((safeReachableCycle cfg).image GameState.bag).image Finset.card).card
      ≤ ((Finset.Icc 1 7 : Finset ℕ)).card := by
        apply Finset.card_le_card
        intro n hn
        rw [Finset.mem_image] at hn
        obtain ⟨bag, hbag, rfl⟩ := hn
        rw [Finset.mem_image] at hbag
        obtain ⟨g, hg, rfl⟩ := hbag
        rw [Finset.mem_Icc]
        refine ⟨?_, Bag.card_le_seven _⟩
        have : 0 < g.bag.card :=
          (safeReachableCycle_subset_reachable cfg g hg).bag_card_pos
        omega
    _ = 7 := by decide

/-- Atlas contains a state equal to init (when init is safe).
Trivial but useful as an exists-pattern. -/
theorem safeReachableCycle_exists_eq_init {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    ∃ g ∈ safeReachableCycle cfg, g = GameState.init :=
  ⟨GameState.init, init_mem_safeReachableCycle h, rfl⟩

/-- Solver-atlas analog: contains a state equal to init. -/
theorem solverReachableFinset_exists_eq_init {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    ∃ g ∈ solverReachableFinset hsv, g = GameState.init :=
  ⟨GameState.init, init_mem_solverReachableFinset hsv, rfl⟩

/-- Solver-atlas analog. -/
theorem solverReachableFinset_image_bag_card_image_card_le_seven
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (((solverReachableFinset hsv).image GameState.bag).image Finset.card).card ≤ 7 := by
  calc (((solverReachableFinset hsv).image GameState.bag).image Finset.card).card
      ≤ ((Finset.Icc 1 7 : Finset ℕ)).card := by
        apply Finset.card_le_card
        intro n hn
        rw [Finset.mem_image] at hn
        obtain ⟨bag, hbag, rfl⟩ := hn
        rw [Finset.mem_image] at hbag
        obtain ⟨g, hg, rfl⟩ := hbag
        rw [Finset.mem_Icc]
        refine ⟨?_, Bag.card_le_seven _⟩
        have : 0 < g.bag.card :=
          (safeReachableCycle_subset_reachable cfg g
            (solverReachableFinset_subset_safeReachableCycle hsv hg)).bag_card_pos
        omega
    _ = 7 := by decide


/-- **`safeSolver` from init yields 7 distinct one-step states**.
Direct corollary of `init_step_image_card` (Tick 359) applied with
`f p = safeSolver cfg init p`; `safeSolver_piece` discharges the
piece-matching hypothesis. -/
theorem safeSolver_init_step_image_card (cfg : GameConfig) :
    ((Finset.univ : Finset Piece).image (fun p =>
        GameState.init.step cfg (safeSolver cfg GameState.init p))).card = 7 :=
  GameState.init_step_image_card (fun p => safeSolver cfg GameState.init p)
    (safeSolver_piece cfg GameState.init)

/-- **State-fanout for arbitrary valid solving solvers**: any state in
`solverReachableFinset` has `|g.bag|` distinct in-atlas successors
under σ. Uses ValidSolver for piece + Valid; Tick 307 for closure;
Tick 364 for cardinality. -/
theorem solverReachableFinset_state_fanout {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hg : g ∈ solverReachableFinset hsv) :
    (g.bag.image (fun p => g.step cfg (σ g p))) ⊆
      solverReachableFinset hsv ∧
    (g.bag.image (fun p => g.step cfg (σ g p))).card = g.bag.card := by
  refine ⟨?_, ?_⟩
  · intro g' hg'
    rw [Finset.mem_image] at hg'
    obtain ⟨p, hp, rfl⟩ := hg'
    have hpiece := (hsv.1 g p hp).1
    have heta : ({σ g p with piece := p} : Placement) = σ g p := by
      rcases hpl : σ g p with ⟨pc, r, c⟩
      rw [hpl] at hpiece; cases hpiece; rfl
    have hstep : g.step cfg (σ g p)
        = adversarialStep cfg g p (σ g p) := by
      rw [adversarialStep_eq_step, heta]
    rw [hstep]
    exact solverReachableFinset_step hsv hg hp
  · apply Finset.card_image_of_injOn
    intro p hp q hq heq
    by_contra hne
    have hpp : (σ g p).piece = p := (hsv.1 g p hp).1
    have hqp : (σ g q).piece = q := (hsv.1 g q hq).1
    refine GameState.step_inj_on_bag (cfg := cfg) (pl1 := σ g p)
      (pl2 := σ g q) ?_ ?_ ?_ heq
    · rw [hpp]; exact hp
    · rw [hqp]; exact hq
    · rw [hpp, hqp]; exact hne
theorem safeReachableCycle_state_fanout {cfg : GameConfig}
    {g : GameState} (hg : g ∈ safeReachableCycle cfg) :
    (g.bag.image (fun p => g.step cfg (safeSolver cfg g p))) ⊆
      safeReachableCycle cfg ∧
    (g.bag.image (fun p => g.step cfg (safeSolver cfg g p))).card =
      g.bag.card := by
  refine ⟨?_, ?_⟩
  · intro g' hg'
    rw [Finset.mem_image] at hg'
    obtain ⟨p, hp, rfl⟩ := hg'
    have hpiece := safeSolver_piece cfg g p
    have heta : ({safeSolver cfg g p with piece := p} : Placement)
        = safeSolver cfg g p := by
      rcases hpl : safeSolver cfg g p with ⟨pc, r, c⟩
      rw [hpl] at hpiece; cases hpiece; rfl
    have hstep : g.step cfg (safeSolver cfg g p)
        = adversarialStep cfg g p (safeSolver cfg g p) := by
      rw [adversarialStep_eq_step, heta]
    rw [hstep]
    exact safeReachableCycle_safeSolver_step hg hp
  · exact GameState.step_image_card g _ (safeSolver_piece cfg g)

/-- **When init is safe, the 7 distinct one-step `safeSolver` images
all live in the atlas.** Composes Tick 360 (cardinality 7) with
`safeReachableCycle_init_step` (Tick 162-ish). -/
theorem safeSolver_init_step_image_subset_safeReachableCycle
    {cfg : GameConfig} (h : GameState.init ∈ safe cfg) :
    ((Finset.univ : Finset Piece).image (fun p =>
        GameState.init.step cfg (safeSolver cfg GameState.init p)))
      ⊆ safeReachableCycle cfg := by
  intro g hg
  rw [Finset.mem_image] at hg
  obtain ⟨p, _, rfl⟩ := hg
  -- Reduce init.step cfg (safeSolver cfg init p) to adversarialStep form via eta.
  have hpiece := safeSolver_piece cfg GameState.init p
  have heta : ({safeSolver cfg GameState.init p with piece := p} : Placement)
      = safeSolver cfg GameState.init p := by
    rcases hpl : safeSolver cfg GameState.init p with ⟨pc, r, c⟩
    rw [hpl] at hpiece; cases hpiece; rfl
  have hstep : GameState.init.step cfg (safeSolver cfg GameState.init p)
      = adversarialStep cfg GameState.init p
          (safeSolver cfg GameState.init p) := by
    rw [adversarialStep_eq_step, heta]
  rw [hstep]
  exact safeReachableCycle_init_step h p

/-- **`safeReachableCycle` card ≥ 8 when init is safe** — direct
constructive bound: init + the 7 distinct one-step images. Stronger
than `card_ge_28` *as a constructive certificate*: the 8 states are
explicitly exhibited. -/
theorem safeReachableCycle_card_ge_eight {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    8 ≤ (safeReachableCycle cfg).card := by
  -- Image of size 7, init in atlas but not in image (init ≠ step result).
  set F := (Finset.univ : Finset Piece).image (fun p =>
      GameState.init.step cfg (safeSolver cfg GameState.init p)) with hF
  have hFcard : F.card = 7 := safeSolver_init_step_image_card cfg
  have hFsub : F ⊆ safeReachableCycle cfg :=
    safeSolver_init_step_image_subset_safeReachableCycle h
  have hinit : GameState.init ∈ safeReachableCycle cfg :=
    init_mem_safeReachableCycle h
  have hinit_notin : GameState.init ∉ F := by
    intro hmem
    rw [hF, Finset.mem_image] at hmem
    obtain ⟨p, _, hp⟩ := hmem
    exact GameState.init_step_ne_init _ hp
  -- F ∪ {init} ⊆ atlas, with card 8.
  have hins_sub : insert GameState.init F ⊆ safeReachableCycle cfg :=
    Finset.insert_subset hinit hFsub
  have hins_card : (insert GameState.init F).card = 8 := by
    rw [Finset.card_insert_of_notMem hinit_notin, hFcard]
  calc 8 = (insert GameState.init F).card := hins_card.symm
    _ ≤ (safeReachableCycle cfg).card := Finset.card_le_card hins_sub

/-- **`card = 0` refutes init safety.** Contrapositive of Tick 362's
`init ∈ safe → card ≥ 8`. -/
theorem safeReachableCycle_card_eq_zero_implies_init_not_safe
    {cfg : GameConfig} (h : (safeReachableCycle cfg).card = 0) :
    GameState.init ∉ safe cfg := by
  intro hsafe
  have := safeReachableCycle_card_ge_eight hsafe
  omega

/-- **`card < 8` refutes init safety.** Stronger refutation tool. -/
theorem safeReachableCycle_card_lt_8_implies_init_not_safe
    {cfg : GameConfig} (h : (safeReachableCycle cfg).card < 8) :
    GameState.init ∉ safe cfg := by
  intro hsafe
  have := safeReachableCycle_card_ge_eight hsafe
  omega

/-- **`card < 28` refutes init safety.** Strongest refutation tool. -/
theorem safeReachableCycle_card_lt_28_implies_init_not_safe
    {cfg : GameConfig} (h : (safeReachableCycle cfg).card < 28) :
    GameState.init ∉ safe cfg := by
  intro hsafe
  have := (safeReachableCycle_card_interval hsafe).1
  omega

/-- **Solvability implies atlas card ≥ 8 (constructive)**: any
`TetrisSolvableValidFor cfg` proof (with `4 ≤ cfg.cols`) yields atlas
containing init + 7 distinct one-step images. -/
theorem tetrisSolvableValidFor_implies_safeReachableCycle_card_ge_eight
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : TetrisSolvableValidFor cfg) :
    8 ≤ (safeReachableCycle cfg).card :=
  safeReachableCycle_card_ge_eight
    ((tetrisSolvableValidFor_iff_init_safe hcols).mp h)

/-- **`safeReachableCycle_card_ge_eight` weak form**: 2 ≤ card. -/
theorem safeReachableCycle_card_ge_two_of_init_safe {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    2 ≤ (safeReachableCycle cfg).card := by
  have := safeReachableCycle_card_ge_eight h
  omega

/-- Standard specialization of Tick 373. -/
theorem tetrisSolvableValid_implies_standard_safeReachableCycle_card_ge_eight
    (h : TetrisSolvableValid) :
    8 ≤ (safeReachableCycle GameConfig.standard).card :=
  tetrisSolvableValidFor_implies_safeReachableCycle_card_ge_eight
    GameConfig.standard_four_le_cols h

/-- Tiny specialization of Tick 373. -/
theorem tetrisSolvableValidFor_tiny_implies_safeReachableCycle_card_ge_eight
    (h : TetrisSolvableValidFor GameConfig.tiny) :
    8 ≤ (safeReachableCycle GameConfig.tiny).card :=
  tetrisSolvableValidFor_implies_safeReachableCycle_card_ge_eight
    GameConfig.tiny_four_le_cols h

/-- **Parametric parity for solver-reachable states** under any valid
solving solver: count mod cfg.cols equals `4*n mod cfg.cols` for some
`n`. Inherits from `Reachable.exists_count_mod_cols` via Tick 298. -/
theorem solverReachable_exists_count_mod_cols_of_solvesTetrisValid
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hr : solverReachable σ g) :
    ∃ n, g.board.count % cfg.cols = 4 * n % cfg.cols :=
  (solverReachable_reachable_of_solvesTetrisValid hsv hr).exists_count_mod_cols

/-- **Solver atlas board count bound**: every state in any valid
solving solver's atlas has board cell count `≤ cfg.cols * cfg.rows`. -/
theorem solverReachableFinset_board_card_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    g.board.card ≤ cfg.cols * cfg.rows := by
  have hr := (mem_solverReachableFinset hsv g).mp h
  exact (solverReachable_reachable_of_solvesTetrisValid hsv hr).not_lost_board_card_le
    (solverReachable_not_lost_of_solvesTetrisValid hsv hr)

/-- `.count` alias of `solverReachableFinset_board_card_le`. -/
theorem solverReachableFinset_board_count_le {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    g.board.count ≤ cfg.cols * cfg.rows :=
  solverReachableFinset_board_card_le hsv h


/-- **Bundled atlas-membership invariants.** Five core projections at
once: every atlas state is reachable, safe, has a WF board, a
nonempty bag, and is not lost. Useful single-source destructuring for
proofs that need multiple properties simultaneously. -/
theorem safeReachableCycle_invariants {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    Reachable cfg g ∧ g ∈ safe cfg ∧ Board.WF cfg g.board ∧
      g.bag.Nonempty ∧ ¬ g.lost cfg :=
  ⟨safeReachableCycle_subset_reachable cfg g h,
   safeReachableCycle_subset_safe cfg g h,
   (safeReachableCycle_subset_reachable cfg g h).wf,
   (safeReachableCycle_subset_reachable cfg g h).bag_nonempty,
   safe_not_lost (safeReachableCycle_subset_safe cfg g h)⟩

/-- **No atlas state is lost.** Direct inheritance from `safe_not_lost`
via `safeReachableCycle_subset_safe`. The atlas contains only live
states — none have overflowed above the visible field. -/
theorem safeReachableCycle_not_lost {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) : ¬ g.lost cfg :=
  safe_not_lost (safeReachableCycle_subset_safe cfg g h)

/-- **Every atlas state has a WF board.** Direct inheritance from
`Reachable.wf`. -/
theorem safeReachableCycle_board_wf {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    Board.WF cfg g.board :=
  (safeReachableCycle_subset_reachable cfg g h).wf

/-- **Every atlas state has a nonempty bag.** Direct inheritance from
`Reachable.bag_nonempty`. -/
theorem safeReachableCycle_bag_nonempty {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) : g.bag.Nonempty :=
  (safeReachableCycle_subset_reachable cfg g h).bag_nonempty

/-- Atlas states have nonempty bags. -/
theorem safeReachableCycle_bag_ne_empty {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) : g.bag ≠ ∅ :=
  (safeReachableCycle_bag_nonempty h).ne_empty

/-- Every atlas state has a drawable piece. -/
theorem safeReachableCycle_exists_canDraw {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    ∃ p : Piece, g.bag.canDraw p :=
  Bag.exists_canDraw_of_nonempty (safeReachableCycle_bag_nonempty h)

/-- Canonical drawable-piece witness for an atlas state. -/
noncomputable def safeReachableCycle_someBagElement {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) : Piece :=
  g.bag.someElement (safeReachableCycle_bag_nonempty h)

theorem safeReachableCycle_someBagElement_mem {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    safeReachableCycle_someBagElement h ∈ g.bag :=
  Bag.someElement_mem _ _

theorem safeReachableCycle_someBagElement_canDraw {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.bag.canDraw (safeReachableCycle_someBagElement h) :=
  Bag.someElement_canDraw _ _

/-- Atlas state's bag card is positive. -/
theorem safeReachableCycle_bag_card_pos {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) : 0 < g.bag.card :=
  Finset.card_pos.mpr (safeReachableCycle_bag_nonempty h)

/-- Atlas state's bag card ≤ 7. -/
theorem safeReachableCycle_bag_card_le_seven {cfg : GameConfig}
    {g : GameState} (_h : g ∈ safeReachableCycle cfg) : g.bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- Atlas state's bag is either ⊂ `full` or = `full`. -/
theorem safeReachableCycle_bag_ssubset_full_or_eq_full {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.bag ⊂ Bag.full ∨ g.bag = Bag.full :=
  (safeReachableCycle_subset_reachable cfg g h).bag_ssubset_full_or_eq_full

/-- **Tick-500 milestone**: when init is safe, the canonical atlas
*both* satisfies the `IsInitCycle` predicate (Tick 234) and contains
at least 28 states (Tick 156). The "atlas exists and is large
enough" summary fact. -/
theorem safeReachableCycle_isInitCycle_and_card_ge_28
    {cfg : GameConfig} (h : GameState.init ∈ safe cfg) :
    AdversarialClosedCycle.IsInitCycle cfg (safeReachableCycle cfg)
        (safeSolver cfg) ∧
    28 ≤ (safeReachableCycle cfg).card :=
  ⟨safeReachableCycle_isInitCycle h, (safeReachableCycle_card_interval h).1⟩

/-- Solver-atlas analog of Tick 500. -/
theorem solverReachableFinset_isInitCycle_and_card_ge_28
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    AdversarialClosedCycle.IsInitCycle cfg (solverReachableFinset hsv) σ ∧
    28 ≤ (solverReachableFinset hsv).card :=
  ⟨solverReachableFinset_isInitCycle hsv,
   solverReachableFinset_card_ge_28 hsv⟩

/-- **Solver-atlas guarantees nonempty `allValidFor`.** Solver
analog of Tick 252/safeReachableCycle_allValidFor_nonempty:
the solver's placement witnesses non-emptiness. -/
theorem solverReachableFinset_allValidFor_nonempty {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    (Placement.allValidFor cfg p).Nonempty :=
  ⟨σ g p, hsv.toValidSolver.mem_allValidFor g hp⟩

/-- **Solver atlas has a valid placement for each bag piece.**
Solver analog of `safeReachableCycle_has_valid_placement`. The
witness is the solver's own choice `σ g p`. -/
theorem solverReachableFinset_has_valid_placement {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg :=
  ⟨σ g p, hsv.toValidSolver g p hp⟩

/-- The solver's placement plays the announced piece (lift of
`ValidSolver` field `.1` to the Finset form). -/
theorem solverReachableFinset_solver_piece {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).piece = p :=
  (hsv.toValidSolver g p hp).1

/-- The solver's placement is in-bounds (lift of `ValidSolver`
field `.2` to the Finset form). -/
theorem solverReachableFinset_solver_valid {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    (σ g p).Valid cfg :=
  (hsv.toValidSolver g p hp).2

/-- The solver's placement always lands in `Placement.allValidFor`,
i.e. it's one of the precomputed search candidates. Scoped to the
Finset form for ergonomics. -/
theorem solverReachableFinset_solver_mem_allValidFor {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (_hg : g ∈ solverReachableFinset hsv)
    {p : Piece} (hp : p ∈ g.bag) :
    σ g p ∈ Placement.allValidFor cfg p :=
  hsv.toValidSolver.mem_allValidFor g hp

/-- **Every atlas state has bag cardinality in `[1, 7]`.** Numerical
form, useful for bounding the atlas's branching factor. -/
theorem safeReachableCycle_bag_card_interval {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    1 ≤ g.bag.card ∧ g.bag.card ≤ 7 :=
  ⟨(safeReachableCycle_subset_reachable cfg g h).bag_card_pos,
   (safeReachableCycle_subset_reachable cfg g h).bag_card_le_seven⟩

/-- **Atlas guarantees nonempty `allValidFor`.** For any state `g` in the
canonical atlas and any bag piece `p`, the Finset of valid placements
for `p` (the search enumeration) is nonempty. Lifts Tick 252 to the
Finset characterization. -/
theorem safeReachableCycle_allValidFor_nonempty {cfg : GameConfig}
    {g : GameState} (hg : g ∈ safeReachableCycle cfg)
    {p : Piece} (hp : p ∈ g.bag) :
    (Placement.allValidFor cfg p).Nonempty := by
  obtain ⟨pl, hpiece, hvalid⟩ := safeReachableCycle_has_valid_placement hg hp
  exact ⟨pl, (Placement.mem_allValidFor cfg p pl).mpr ⟨hpiece, hvalid⟩⟩

/-- **Canonical-solver mem_allValidFor for atlas states**. For any
state in the canonical atlas, `safeSolver`'s choice for each bag
piece sits in the precomputed search Finset. Atlas-scoped form of
`safeSolver_mem_allValidFor`. -/
theorem safeReachableCycle_safeSolver_mem_allValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {g : GameState}
    (_hg : g ∈ safeReachableCycle cfg) {p : Piece} (hp : p ∈ g.bag) :
    safeSolver cfg g p ∈ Placement.allValidFor cfg p :=
  safeSolver_mem_allValidFor hcols hp

/-- Atlas-scoped `safeSolver` piece accessor. -/
theorem safeReachableCycle_safeSolver_piece {cfg : GameConfig}
    {g : GameState} (_hg : g ∈ safeReachableCycle cfg) (p : Piece) :
    (safeSolver cfg g p).piece = p :=
  safeSolver_piece cfg g p

/-- Atlas-scoped `safeSolver` validity accessor. -/
theorem safeReachableCycle_safeSolver_valid {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) {g : GameState}
    (_hg : g ∈ safeReachableCycle cfg) {p : Piece} (hp : p ∈ g.bag) :
    (safeSolver cfg g p).Valid cfg :=
  (safeSolver_validSolver hcols g p hp).2

/-- **Init-state canonical-solver validity**: when `init` is in the
safe set, `safeSolver`'s placement for any piece is in-bounds.
Bag-piece hypothesis is discharged via `Bag.mem_full`. -/
theorem init_safeSolver_valid {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    (safeSolver cfg GameState.init p).Valid cfg :=
  (safeSolver_validSolver hcols GameState.init p
    (by simp [GameState.init, Bag.mem_full])).2

/-- Init-state version of `safeSolver_mem_allValidFor`: every
piece's `safeSolver` choice at `init` lies in the search Finset. -/
theorem init_safeSolver_mem_allValidFor {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (p : Piece) :
    safeSolver cfg GameState.init p ∈ Placement.allValidFor cfg p :=
  safeSolver_mem_allValidFor hcols
    (by simp [GameState.init, Bag.mem_full])

/-- **Init-state ValidSolver projection**: at `init`, any
`ValidSolver` produces a placement that plays the right piece
and is in-bounds, for any piece (since `init.bag = full`). -/
theorem ValidSolver.init {cfg : GameConfig} {σ : Solver cfg}
    (h : ValidSolver cfg σ) (p : Piece) :
    (σ GameState.init p).piece = p ∧ (σ GameState.init p).Valid cfg :=
  h GameState.init p (by simp [GameState.init, Bag.mem_full])

/-- The "piece" half of `ValidSolver.init`. -/
theorem ValidSolver.init_piece {cfg : GameConfig} {σ : Solver cfg}
    (h : ValidSolver cfg σ) (p : Piece) :
    (σ GameState.init p).piece = p :=
  (h.init p).1

/-- The "valid" half of `ValidSolver.init`. -/
theorem ValidSolver.init_valid {cfg : GameConfig} {σ : Solver cfg}
    (h : ValidSolver cfg σ) (p : Piece) :
    (σ GameState.init p).Valid cfg :=
  (h.init p).2

/-- Init-instance of `ValidSolver.mem_allValidFor`. -/
theorem ValidSolver.init_mem_allValidFor {cfg : GameConfig} {σ : Solver cfg}
    (h : ValidSolver cfg σ) (p : Piece) :
    σ GameState.init p ∈ Placement.allValidFor cfg p :=
  h.mem_allValidFor GameState.init
    (by simp [GameState.init, Bag.mem_full])

/-- Init-instance dot accessor for a `SolvesTetrisValid` solver,
forwarding through `toValidSolver`. -/
theorem SolvesTetrisValid.init {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) (p : Piece) :
    (σ GameState.init p).piece = p ∧ (σ GameState.init p).Valid cfg :=
  h.toValidSolver.init p

/-- "piece" half of `SolvesTetrisValid.init`. -/
theorem SolvesTetrisValid.init_piece {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) (p : Piece) :
    (σ GameState.init p).piece = p := (h.init p).1

/-- "valid" half of `SolvesTetrisValid.init`. -/
theorem SolvesTetrisValid.init_valid {cfg : GameConfig} {σ : Solver cfg}
    (h : SolvesTetrisValid cfg σ) (p : Piece) :
    (σ GameState.init p).Valid cfg := (h.init p).2

/-- Init-instance `mem_allValidFor` accessor for
`SolvesTetrisValid`. -/
theorem SolvesTetrisValid.init_mem_allValidFor {cfg : GameConfig}
    {σ : Solver cfg} (h : SolvesTetrisValid cfg σ) (p : Piece) :
    σ GameState.init p ∈ Placement.allValidFor cfg p :=
  h.toValidSolver.init_mem_allValidFor p

/-- **Every solver-atlas state is not lost.** Direct lift of
`solverReachable_not_lost_of_solvesTetrisValid` to the Finset
form. -/
theorem solverReachableFinset_not_lost {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    ¬ g.lost cfg :=
  solverReachable_not_lost_of_solvesTetrisValid hsv
    ((mem_solverReachableFinset hsv g).mp h)

/-- **Pointwise lift**: every solver-atlas state is `Reachable`.
Direct lift of `solverReachable_reachable_of_solvesTetrisValid`. -/
theorem solverReachableFinset_reachable {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    Reachable cfg g :=
  solverReachable_reachable_of_solvesTetrisValid hsv
    ((mem_solverReachableFinset hsv g).mp h)

/-- Every solver-atlas state has a WF board. Direct projection
through `Reachable.wf`. -/
theorem solverReachableFinset_board_wf {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    Board.WF cfg g.board :=
  (solverReachableFinset_reachable hsv h).wf

/-- Every solver-atlas state's bag is a subset of the full bag. -/
theorem solverReachableFinset_bag_subset_full {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    g.bag ⊆ Bag.full :=
  (solverReachableFinset_reachable hsv h).bag_subset_full

/-- Every solver-atlas state's bag has nonzero cardinality (Nat
form). -/
theorem solverReachableFinset_bag_card_ne_zero {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    g.bag.card ≠ 0 :=
  Nat.ne_of_gt (solverReachableFinset_bag_card_pos hsv h)

/-- `1 ≤ bag.card` form for solver-atlas states. -/
theorem solverReachableFinset_bag_card_ge_one {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    1 ≤ g.bag.card :=
  solverReachableFinset_bag_card_pos hsv h

/-- `1 ≤ bag.card` form for canonical-atlas states. -/
theorem safeReachableCycle_bag_card_ge_one {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    1 ≤ g.bag.card :=
  safeReachableCycle_bag_card_pos h

/-- `bag.card ≠ 0` form for canonical-atlas states. -/
theorem safeReachableCycle_bag_card_ne_zero {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.bag.card ≠ 0 :=
  Nat.ne_of_gt (safeReachableCycle_bag_card_pos h)

/-- `bag.card < 8` strict-`<` form for canonical-atlas states. -/
theorem safeReachableCycle_bag_card_lt_eight {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g.bag.card < 8 := by
  have := safeReachableCycle_bag_card_le_seven h
  omega

/-- `bag.card < 8` strict-`<` form for solver-atlas states. -/
theorem solverReachableFinset_bag_card_lt_eight {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    g.bag.card < 8 := by
  have := solverReachableFinset_bag_card_le_seven hsv h
  omega


/-- **Refined atlas inclusion**: the canonical atlas sits inside the
nonempty-bag subset of the universe Finset. Strengthens
`safeReachableCycle_subset_inFieldStates` by using
`safeReachableCycle_bag_nonempty`. -/
theorem safeReachableCycle_subset_nonempty_bag_inFieldStates
    (cfg : GameConfig) :
    safeReachableCycle cfg ⊆
      (inFieldStates cfg).filter (fun g => g.bag.Nonempty) := by
  intro g hg
  rw [Finset.mem_filter]
  exact ⟨safeReachableCycle_subset_inFieldStates cfg hg,
         safeReachableCycle_bag_nonempty hg⟩

/-- **Card-form of the refined inclusion**: atlas card is bounded by
the count of nonempty-bag universe states. -/
theorem safeReachableCycle_card_le_nonempty_bag_inFieldStates_card
    (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤
      ((inFieldStates cfg).filter (fun g => g.bag.Nonempty)).card :=
  Finset.card_le_card
    (safeReachableCycle_subset_nonempty_bag_inFieldStates cfg)

/-- **Sharper atlas inclusion**: also rules out lost states. The
canonical atlas sits inside the not-lost ∧ nonempty-bag subset of
the universe. Combines `not_lost`, `bag_nonempty`, and
`subset_inFieldStates`. -/
theorem safeReachableCycle_subset_alive_inFieldStates (cfg : GameConfig) :
    safeReachableCycle cfg ⊆
      (inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty) := by
  intro g hg
  rw [Finset.mem_filter]
  exact ⟨safeReachableCycle_subset_inFieldStates cfg hg,
         safeReachableCycle_not_lost hg,
         safeReachableCycle_bag_nonempty hg⟩

/-- Card-form of the sharper inclusion: atlas card is bounded by
the count of alive (not-lost + nonempty-bag) universe states.
This is a tighter upper bound than
`safeReachableCycle_card_le_nonempty_bag_inFieldStates_card`. -/
theorem safeReachableCycle_card_le_alive_inFieldStates_card (cfg : GameConfig) :
    (safeReachableCycle cfg).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty)).card :=
  Finset.card_le_card (safeReachableCycle_subset_alive_inFieldStates cfg)

/-- Solver-atlas analog: solver-reachable Finset ⊆ alive-states. -/
theorem solverReachableFinset_subset_alive_inFieldStates
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ⊆
      (inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty) := by
  intro g hg
  have hatlas := solverReachableFinset_subset_safeReachableCycle hsv hg
  exact (Finset.mem_filter).mpr
    ⟨safeReachableCycle_subset_inFieldStates cfg hatlas,
     solverReachableFinset_not_lost hsv hg,
     solverReachableFinset_bag_nonempty hsv hg⟩

/-- Card-form: solver-atlas card ≤ alive count. -/
theorem solverReachableFinset_card_le_alive_inFieldStates_card
    {cfg : GameConfig} {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty)).card :=
  Finset.card_le_card (solverReachableFinset_subset_alive_inFieldStates hsv)

/-- **Atlas-pruning contrapositive**: any state that is lost is
not in the canonical atlas. -/
theorem not_mem_safeReachableCycle_of_lost {cfg : GameConfig}
    {g : GameState} (h : g.lost cfg) : g ∉ safeReachableCycle cfg :=
  fun hmem => safeReachableCycle_not_lost hmem h

/-- **Atlas-pruning contrapositive**: any state with empty bag is
not in the canonical atlas. -/
theorem not_mem_safeReachableCycle_of_bag_empty {cfg : GameConfig}
    {g : GameState} (h : g.bag = ∅) : g ∉ safeReachableCycle cfg :=
  fun hmem => safeReachableCycle_bag_ne_empty hmem h

/-- **Atlas-pruning contrapositive**: any state outside the
in-field universe is not in the canonical atlas. -/
theorem not_mem_safeReachableCycle_of_not_inFieldStates {cfg : GameConfig}
    {g : GameState} (h : g ∉ inFieldStates cfg) :
    g ∉ safeReachableCycle cfg :=
  fun hmem => h (safeReachableCycle_subset_inFieldStates cfg hmem)

/-- **Atlas-pruning contrapositive**: any non-reachable state is
not in the canonical atlas. -/
theorem not_mem_safeReachableCycle_of_not_reachable {cfg : GameConfig}
    {g : GameState} (h : ¬ Reachable cfg g) : g ∉ safeReachableCycle cfg :=
  fun hmem => h (safeReachableCycle_subset_reachable cfg g hmem)

/-- **Parametric parity pruning**: for any config with even cols,
every atlas state has an even board count. Composes
`safeReachableCycle_subset_reachable` with `Reachable.even_count`. -/
theorem safeReachableCycle_board_count_even {cfg : GameConfig}
    (hcol : Even cfg.cols) {g : GameState}
    (h : g ∈ safeReachableCycle cfg) :
    Even g.board.count :=
  (safeReachableCycle_subset_reachable cfg g h).even_count hcol

/-- Contrapositive: a state with odd board count is not in the
canonical atlas (when cols is even). -/
theorem not_mem_safeReachableCycle_of_count_not_even {cfg : GameConfig}
    (hcol : Even cfg.cols) {g : GameState}
    (hne : ¬ Even g.board.count) : g ∉ safeReachableCycle cfg :=
  fun hmem => hne (safeReachableCycle_board_count_even hcol hmem)

/-- **Lifted count-relation**: every atlas state satisfies
`board.count + cols * k = 4 * n` for some `n, k`. Useful for
deriving config-specific count constraints (e.g. mod-4 for
tiny, mod-2 for standard). -/
theorem safeReachableCycle_exists_count_eq {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    ∃ n k : ℕ, g.board.count + cfg.cols * k = 4 * n :=
  (safeReachableCycle_subset_reachable cfg g h).exists_count_eq

/-- Solver-atlas analog: count-relation lifted to
`solverReachableFinset`. -/
theorem solverReachableFinset_exists_count_eq {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    ∃ n k : ℕ, g.board.count + cfg.cols * k = 4 * n :=
  (solverReachableFinset_reachable hsv h).exists_count_eq

/-- Solver-atlas analog: parity (`Even cols` config). -/
theorem solverReachableFinset_board_count_even {cfg : GameConfig}
    (hcol : Even cfg.cols) {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) {g : GameState}
    (h : g ∈ solverReachableFinset hsv) :
    Even g.board.count :=
  (solverReachableFinset_reachable hsv h).even_count hcol

/-- Contrapositive: odd-count states ruled out of the solver
atlas (when cols even). -/
theorem not_mem_solverReachableFinset_of_count_not_even {cfg : GameConfig}
    (hcol : Even cfg.cols) {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (hne : ¬ Even g.board.count) :
    g ∉ solverReachableFinset hsv :=
  fun hmem => hne (solverReachableFinset_board_count_even hcol hsv hmem)

/-- **Atlas full invariants** (bundled): every atlas state
satisfies the structural quartet — in-field, Reachable, safe,
non-empty bag — and the dynamic invariants: not lost. Convenient
one-shot destructor for downstream proofs. -/
theorem safeReachableCycle_full_invariants {cfg : GameConfig}
    {g : GameState} (h : g ∈ safeReachableCycle cfg) :
    g ∈ inFieldStates cfg ∧ Reachable cfg g ∧ g ∈ safe cfg ∧
    ¬ g.lost cfg ∧ g.bag.Nonempty :=
  ⟨safeReachableCycle_subset_inFieldStates cfg h,
   safeReachableCycle_subset_reachable cfg g h,
   safeReachableCycle_subset_safe cfg g h,
   safeReachableCycle_not_lost h,
   safeReachableCycle_bag_nonempty h⟩

/-- **Init-state full invariants** (when init is safe): bundled
invariants at init. Trivially init is reachable and has empty
board + full bag, plus by hypothesis init is safe and in the
atlas. -/
theorem safeReachableCycle_init_full_invariants {cfg : GameConfig}
    (h : GameState.init ∈ safe cfg) :
    GameState.init ∈ safeReachableCycle cfg ∧
    Reachable cfg GameState.init ∧
    GameState.init ∈ safe cfg ∧
    GameState.init.bag = Bag.full ∧
    GameState.init.board = ∅ :=
  ⟨init_mem_safeReachableCycle h, Reachable.init, h,
   GameState.init_bag, GameState.init_board⟩

/-- **Sharper atlas inclusion via parity**: when `cfg.cols` is
even, the atlas sits inside the alive + even-count subset of
the universe. Composes Ticks 645/646 (alive) with Tick 647
(parity). -/
theorem safeReachableCycle_subset_alive_even_inFieldStates
    {cfg : GameConfig} (hcol : Even cfg.cols) :
    safeReachableCycle cfg ⊆
      (inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count) := by
  intro g hg
  rw [Finset.mem_filter]
  exact ⟨safeReachableCycle_subset_inFieldStates cfg hg,
         safeReachableCycle_not_lost hg,
         safeReachableCycle_bag_nonempty hg,
         safeReachableCycle_board_count_even hcol hg⟩

/-- Card form of the sharper inclusion. -/
theorem safeReachableCycle_card_le_alive_even_inFieldStates_card
    {cfg : GameConfig} (hcol : Even cfg.cols) :
    (safeReachableCycle cfg).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count)).card :=
  Finset.card_le_card (safeReachableCycle_subset_alive_even_inFieldStates hcol)

/-- Solver-atlas analog: when cols is even, solver atlas ⊆
alive + even-count subset. -/
theorem solverReachableFinset_subset_alive_even_inFieldStates
    {cfg : GameConfig} (hcol : Even cfg.cols) {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) :
    solverReachableFinset hsv ⊆
      (inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count) := by
  intro g hg
  have hatlas := solverReachableFinset_subset_safeReachableCycle hsv hg
  rw [Finset.mem_filter]
  exact ⟨safeReachableCycle_subset_inFieldStates cfg hatlas,
         solverReachableFinset_not_lost hsv hg,
         solverReachableFinset_bag_nonempty hsv hg,
         solverReachableFinset_board_count_even hcol hsv hg⟩

/-- Card form. -/
theorem solverReachableFinset_card_le_alive_even_inFieldStates_card
    {cfg : GameConfig} (hcol : Even cfg.cols) {σ : Solver cfg}
    (hsv : SolvesTetrisValid cfg σ) :
    (solverReachableFinset hsv).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count)).card :=
  Finset.card_le_card
    (solverReachableFinset_subset_alive_even_inFieldStates hcol hsv)

/-- **Atlas-card interval with parity** (when cols even):
`28 ≤ atlas card ≤ alive-even-count`. Tightest known interval
combining lower bound (Tick 156) with upper bound (Tick 661). -/
theorem safeReachableCycle_card_interval_alive_even {cfg : GameConfig}
    (hcol : Even cfg.cols) (h : GameState.init ∈ safe cfg) :
    28 ≤ (safeReachableCycle cfg).card ∧
    (safeReachableCycle cfg).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count)).card :=
  ⟨safeReachableCycle_card_ge_28 h,
   safeReachableCycle_card_le_alive_even_inFieldStates_card hcol⟩

/-- Solver-atlas analog of the alive+even card interval. -/
theorem solverReachableFinset_card_interval_alive_even {cfg : GameConfig}
    (hcol : Even cfg.cols) {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ) :
    28 ≤ (solverReachableFinset hsv).card ∧
    (solverReachableFinset hsv).card ≤
      ((inFieldStates cfg).filter
        (fun g => ¬ g.lost cfg ∧ g.bag.Nonempty ∧
                  Even g.board.count)).card :=
  ⟨solverReachableFinset_card_ge_28 hsv,
   solverReachableFinset_card_le_alive_even_inFieldStates_card hcol hsv⟩

/-- Solver-atlas bundled invariants — Reachable, not lost,
nonempty bag, WF board, and safe (via
`solverReachableFinset_subset_safe`). -/
theorem solverReachableFinset_full_invariants {cfg : GameConfig}
    {σ : Solver cfg} (hsv : SolvesTetrisValid cfg σ)
    {g : GameState} (h : g ∈ solverReachableFinset hsv) :
    Reachable cfg g ∧ ¬ g.lost cfg ∧ g.bag.Nonempty ∧
    Board.WF cfg g.board ∧ g ∈ safe cfg :=
  ⟨solverReachableFinset_reachable hsv h,
   solverReachableFinset_not_lost hsv h,
   solverReachableFinset_bag_nonempty hsv h,
   solverReachableFinset_board_wf hsv h,
   solverReachableFinset_subset_safe hsv g h⟩



/-- **Atlas-init ⇒ solvable** (cols ≥ 4): direct implication form of
Tick 242's iff. A search procedure proving `init ∈ safeReachableCycle
cfg` immediately certifies `TetrisSolvableValidFor cfg`. -/
theorem init_mem_safeReachableCycle_implies_solvable
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : GameState.init ∈ safeReachableCycle cfg) :
    TetrisSolvableValidFor cfg :=
  (tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mpr h

/-- **Contrapositive**: unsolvability implies `init` is not in the
canonical atlas. Together with Tick 276 forms the full iff. -/
theorem not_solvable_implies_init_not_mem_safeReachableCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (h : ¬ TetrisSolvableValidFor cfg) :
    GameState.init ∉ safeReachableCycle cfg := fun hmem =>
  h (init_mem_safeReachableCycle_implies_solvable hcols hmem)

/-- **Unsolvability iff `init` not in atlas.** Bundles Tick 277
(forward) with the contrapositive of Tick 276 (reverse). -/
theorem not_solvable_iff_init_not_mem_safeReachableCycle
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ¬ TetrisSolvableValidFor cfg ↔
      GameState.init ∉ safeReachableCycle cfg :=
  ⟨not_solvable_implies_init_not_mem_safeReachableCycle hcols,
   fun h hsol => h
     ((tetrisSolvableValidFor_iff_init_mem_safeReachableCycle hcols).mp hsol)⟩

/-- Standard specialization of Tick 376. -/
theorem not_tetrisSolvableValid_iff_init_not_mem_standard_safeReachableCycle :
    ¬ TetrisSolvableValid ↔
      GameState.init ∉ safeReachableCycle GameConfig.standard :=
  not_solvable_iff_init_not_mem_safeReachableCycle
    GameConfig.standard_four_le_cols

/-- Tiny specialization of Tick 376. -/
theorem not_tetrisSolvableValidFor_tiny_iff_init_not_mem_safeReachableCycle :
    ¬ TetrisSolvableValidFor GameConfig.tiny ↔
      GameState.init ∉ safeReachableCycle GameConfig.tiny :=
  not_solvable_iff_init_not_mem_safeReachableCycle
    GameConfig.tiny_four_le_cols


/-- **Reachable non-lost states live in `inFieldStates`.** A foundational
structural realisation: the gameplay-defined universe of "states a real game
could be in" (reachable AND non-lost) is a subset of the syntactic enumeration
universe `inFieldStates cfg`. The two ingredients are `Reachable.wf` (board
well-formedness) and `GameState.not_lost_iff_forall_row_lt` (in-field row
bound). This lets reachability arguments cross into the finite-Finset world
where decidability and `Finset.card` bounds apply. -/
theorem mem_inFieldStates_of_reachable_of_not_lost
    {cfg : GameConfig} {g : GameState}
    (hr : Reachable cfg g) (hnl : ¬ g.lost cfg) :
    g ∈ inFieldStates cfg := by
  unfold inFieldStates
  rw [Finset.mem_image]
  refine ⟨(⟨g.board, hr.wf, (GameState.not_lost_iff_forall_row_lt cfg g).mp hnl⟩,
          g.bag), Finset.mem_univ _, ?_⟩
  rfl

/-- **`inFieldStates` membership characterisation.** A `GameState` lives in
`inFieldStates cfg` iff its board is well-formed AND has every cell strictly
below row `cfg.rows`. Exposes the two predicates packaged in `InFieldBoard`
as a direct Iff on game states. The bag is unconstrained (`inFieldStates`
pairs every in-field board with every bag). -/
theorem mem_inFieldStates_iff (cfg : GameConfig) (g : GameState) :
    g ∈ inFieldStates cfg ↔
      Board.WF cfg g.board ∧ ∀ p ∈ g.board, p.2 < cfg.rows := by
  unfold inFieldStates
  rw [Finset.mem_image]
  constructor
  · rintro ⟨⟨b, bag⟩, _, hgeq⟩
    subst hgeq
    exact ⟨b.prop.1, b.prop.2⟩
  · rintro ⟨hwf, hif⟩
    exact ⟨(⟨g.board, hwf, hif⟩, g.bag), Finset.mem_univ _, rfl⟩

/-- **`inFieldStates` membership extracts WF + non-lost.** Reformulation of
the forward direction of `mem_inFieldStates_iff` using
`GameState.not_lost_iff_forall_row_lt` (Tick 1557) to expose the in-field
row bound as the *gameplay* loss predicate. The reverse implication
(`WF + ¬ lost → g ∈ inFieldStates`) also holds; together they give a clean
characterisation independent of reachability. -/
theorem wf_and_not_lost_of_mem_inFieldStates
    {cfg : GameConfig} {g : GameState} (h : g ∈ inFieldStates cfg) :
    Board.WF cfg g.board ∧ ¬ g.lost cfg := by
  rw [mem_inFieldStates_iff] at h
  exact ⟨h.1, (GameState.not_lost_iff_forall_row_lt cfg g).mpr h.2⟩

/-- **Reverse direction**: WF board + non-lost state ⇒ state lives in
`inFieldStates`. Pairs with `wf_and_not_lost_of_mem_inFieldStates` to give
a reachability-free Iff. -/
theorem mem_inFieldStates_of_wf_and_not_lost
    {cfg : GameConfig} {g : GameState}
    (hwf : Board.WF cfg g.board) (hnl : ¬ g.lost cfg) :
    g ∈ inFieldStates cfg := by
  rw [mem_inFieldStates_iff]
  exact ⟨hwf, (GameState.not_lost_iff_forall_row_lt cfg g).mp hnl⟩

/-- **Reachability-free `inFieldStates` characterisation**: `g ∈ inFieldStates cfg
↔ Board.WF cfg g.board ∧ ¬ g.lost cfg`. Stronger than
`mem_inFieldStates_of_reachable_of_not_lost` (which threads reachability
for the WF witness); here WF is taken as a direct hypothesis. -/
theorem mem_inFieldStates_iff_wf_and_not_lost
    (cfg : GameConfig) (g : GameState) :
    g ∈ inFieldStates cfg ↔ Board.WF cfg g.board ∧ ¬ g.lost cfg :=
  ⟨wf_and_not_lost_of_mem_inFieldStates,
   fun ⟨hwf, hnl⟩ => mem_inFieldStates_of_wf_and_not_lost hwf hnl⟩

/-- **Adversarial trace from init lives in `inFieldStates`.** Direct corollary
of the 1557→1559 arc: `Reachable` (Tick 1559's
`adversarialTrace_reachable_of_init`) + `¬ lost`
(`adversarialTrace_not_lost_of_init`) ⇒ membership in `inFieldStates`. Bounds
the entire adversarial-trace cone of any init-containing cycle by the finite
`inFieldStates cfg`. -/
theorem AdversarialClosedCycle.adversarialTrace_mem_inFieldStates_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (n : ℕ) :
    adversarialTrace cfg C.solver s GameState.init n ∈ inFieldStates cfg :=
  (C.adversarialTrace_reachable_of_init h hl n).mem_inFieldStates_of_not_lost
    (C.adversarialTrace_not_lost_of_init h hl n)

/-- **Finite trace-cone bound (Finset form).** The Finset image of an
adversarial trace from `init` under an init-containing cycle solver, taken
over any prefix `Finset.range N` of step counts, is a subset of
`inFieldStates cfg`. Hence its card is bounded by `|inFieldStates cfg|` —
a brute-force enumeration bound on how many distinct states the trace can
visit. Combines `Finset.image_subset_iff` with
`adversarialTrace_mem_inFieldStates_of_init`. -/
theorem AdversarialClosedCycle.adversarialTrace_range_image_subset_inFieldStates_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (N : ℕ) :
    (Finset.range N).image (fun n => adversarialTrace cfg C.solver s GameState.init n)
      ⊆ inFieldStates cfg := by
  rw [Finset.image_subset_iff]
  intro n _
  exact C.adversarialTrace_mem_inFieldStates_of_init h hl n

/-- Card-form of the trace-cone bound: any prefix of the trace cone visits at
most `|inFieldStates cfg|` distinct states. -/
theorem AdversarialClosedCycle.adversarialTrace_range_image_card_le_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) (N : ℕ) :
    ((Finset.range N).image
        (fun n => adversarialTrace cfg C.solver s GameState.init n)).card
      ≤ (inFieldStates cfg).card :=
  Finset.card_le_card (C.adversarialTrace_range_image_subset_inFieldStates_of_init h hl N)

/-- **Pigeonhole on the adversarial trace cone.** When the prefix length
`N + 1` strictly exceeds `|inFieldStates cfg|`, the trace from `init` under
an init-containing cycle's solver must revisit a state: there exist indices
`i < j ≤ N` with `trace i = trace j`. Combines Tick 1560's pointwise
membership in `inFieldStates` with `Finset.exists_ne_map_eq_of_card_lt_of_maps_to`.
This is the first true cycle-detection result: from the *existence* of a
cycle and a long enough trace prefix we extract a *concrete* duplicate
index pair in the trace itself. -/
theorem AdversarialClosedCycle.adversarialTrace_pigeonhole_via_inFieldStates_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) {N : ℕ}
    (hN : (inFieldStates cfg).card < N + 1) :
    ∃ i j : ℕ, i < j ∧ j ≤ N ∧
      adversarialTrace cfg C.solver s GameState.init i
        = adversarialTrace cfg C.solver s GameState.init j := by
  have hmap : ∀ n ∈ Finset.range (N + 1),
      adversarialTrace cfg C.solver s GameState.init n ∈ inFieldStates cfg := by
    intro n _
    exact C.adversarialTrace_mem_inFieldStates_of_init h hl n
  have hcard : (inFieldStates cfg).card < (Finset.range (N + 1)).card := by
    rw [Finset.card_range]; exact hN
  obtain ⟨x, hx, y, hy, hne, hfeq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmap
  rcases lt_or_gt_of_ne hne with hxy | hyx
  · exact ⟨x, y, hxy, Nat.lt_succ_iff.mp (Finset.mem_range.mp hy), hfeq⟩
  · exact ⟨y, x, hyx, Nat.lt_succ_iff.mp (Finset.mem_range.mp hx), hfeq.symm⟩

/-- **Base+period form of the `inFieldStates`-pigeonhole.** Repackages
`adversarialTrace_pigeonhole_via_inFieldStates_of_init`'s duplicate `i < j ≤ N`
into a base/period witness `1 ≤ period ≤ N` with
`trace base = trace (base + period)`. Universal-bound counterpart of
`SafeSet.lean`'s `adversarialTrace_pigeonhole_base_period_of_init` (which uses
the cycle-specific `|C.states|` bound). -/
theorem AdversarialClosedCycle.adversarialTrace_pigeonhole_base_period_via_inFieldStates_of_init
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    (h : GameState.init ∈ C.states)
    {s : ℕ → Piece} (hl : LegalSequence s) {N : ℕ}
    (hN : (inFieldStates cfg).card < N + 1) :
    ∃ base period : ℕ, 1 ≤ period ∧ period ≤ N ∧
      adversarialTrace cfg C.solver s GameState.init base =
        adversarialTrace cfg C.solver s GameState.init (base + period) := by
  obtain ⟨i, j, hij, hj, heq⟩ :=
    C.adversarialTrace_pigeonhole_via_inFieldStates_of_init h hl hN
  refine ⟨i, j - i, ?_, ?_, ?_⟩
  · omega
  · omega
  · rw [Nat.add_sub_cancel' (le_of_lt hij)]; exact heq

/-- **M3 cycle-entry-via-splicing (constructive).** Given a cycle `C`, a
legal sequence `s` whose trace under `C.solver` enters `C.states` at index
`N`, and any continuation `t` that is legal from the entry state's bag,
the **spliced sequence**
`sp := fun k => if k < N then s k else t (k - N)` is itself a
`LegalSequence`, and its trace under `C.solver` from `init` lies in
`C.states` at every index `k ≥ N`. Packages the entire splice toolkit
(Ticks 1572-1576) into a single constructive cycle-entry theorem — given
ANY witness of "a legal sequence whose trace reaches a cycle state", we
extract an explicit legal sequence that stays in the cycle forever. -/
theorem AdversarialClosedCycle.cycle_entry_via_splice
    {cfg : GameConfig} (C : AdversarialClosedCycle cfg)
    {s : ℕ → Piece} (hs : LegalSequence s) {N : ℕ}
    (hN : adversarialTrace cfg C.solver s GameState.init N ∈ C.states)
    {t : ℕ → Piece}
    (ht : LegalSequenceFrom
      (adversarialTrace cfg C.solver s GameState.init N).bag t) :
    LegalSequence (fun k => if k < N then s k else t (k - N)) ∧
    ∀ k, N ≤ k →
      adversarialTrace cfg C.solver
        (fun k => if k < N then s k else t (k - N)) GameState.init k
          ∈ C.states := by
  have hbag : (adversarialTrace cfg C.solver s GameState.init N).bag
      = bagAt Bag.full s N :=
    adversarialTrace_bag cfg C.solver s N
  have ht' : LegalSequenceFrom (bagAt Bag.full s N) t := hbag ▸ ht
  refine ⟨LegalSequenceFrom.splice hs ht', ?_⟩
  intro k hk
  rw [adversarialTrace_splice_past_N cfg C.solver s t GameState.init hk]
  exact C.adversarialTrace_mem_states_from_mem hN ht (k - N)


end Tetris
