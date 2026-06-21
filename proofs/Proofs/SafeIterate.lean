import Mathlib
import Proofs.SafeSet

/-!
# Iterating the safety operator from the top

`safeIterate cfg n` is the result of applying the safety operator `safeOp cfg`
to `Set.univ` (the "everything is potentially safe" starting point) `n` times.
This is the *abstract form* of the iterative computation underlying any
mechanical search for the safe set:

```
safeIterate 0       = ⊤             -- everything
safeIterate (n+1)   = safeOp safeIterate n
```

We prove the iterates form a **decreasing chain** and that `safe` lies inside
every iterate (the "upper bound" direction):

```
safe ⊆ safeIterate n      for every n
safe ⊆ ⋂ n, safeIterate n
safeIterate (n+1) ⊆ safeIterate n
```

The matching lower bound (`safeIterate n ⊆ safe` for some `n` — convergence in
`ω` steps) requires that the operator be `ω`-co-continuous, which holds when
the state space (restricted appropriately, e.g. to in-field states) is finite.
That decidable / convergent computation is the algorithmic content of the GFP
search referenced in the project's prior work, and is **further work** here.
-/

namespace Tetris

/-- Iterated application of the safety operator starting from `Set.univ`. -/
def safeIterate (cfg : GameConfig) : ℕ → Set GameState
  | 0     => Set.univ
  | n + 1 => safeOp cfg (safeIterate cfg n)

@[simp] theorem safeIterate_zero (cfg : GameConfig) :
    safeIterate cfg 0 = Set.univ := rfl

theorem safeIterate_succ (cfg : GameConfig) (n : ℕ) :
    safeIterate cfg (n + 1) = safeOp cfg (safeIterate cfg n) := rfl

/-- The iterates form a **decreasing chain**: each iterate refines the last. -/
theorem safeIterate_succ_subset (cfg : GameConfig) (n : ℕ) :
    safeIterate cfg (n + 1) ⊆ safeIterate cfg n := by
  induction n with
  | zero => exact Set.subset_univ _
  | succ k ih => exact (safeOp cfg).monotone' ih

/-- **The safe set is contained in every iterate from the top.** Any state in
the greatest fixed point survives every finite "depth-bounded refutation" —
none of the iterates can rule it out. -/
theorem safe_subset_safeIterate (cfg : GameConfig) (n : ℕ) :
    safe cfg ⊆ safeIterate cfg n := by
  induction n with
  | zero => exact Set.subset_univ _
  | succ k ih =>
      intro g hg
      have h1 : g ∈ safeOp cfg (safe cfg) := (safe_eq cfg).symm ▸ hg
      exact (safeOp cfg).monotone' ih h1

/-- The safe set sits inside the intersection of all iterates. -/
theorem safe_subset_iInter (cfg : GameConfig) :
    safe cfg ⊆ ⋂ n, safeIterate cfg n :=
  Set.subset_iInter (safe_subset_safeIterate cfg)

/-- Iterates are monotone in the index (decreasing): `m ≤ n ⇒ iterate n ⊆ iterate m`. -/
theorem safeIterate_antitone (cfg : GameConfig) {m n : ℕ} (h : m ≤ n) :
    safeIterate cfg n ⊆ safeIterate cfg m := by
  induction n, h using Nat.le_induction with
  | base => exact Set.Subset.refl _
  | succ k hkm ih => exact (safeIterate_succ_subset cfg k).trans ih

/-- The first iterate is `safeOp cfg Set.univ` (a definitional spelling). -/
theorem safeIterate_one (cfg : GameConfig) :
    safeIterate cfg 1 = safeOp cfg Set.univ := rfl

/-- Every state is in `safeIterate cfg 0` (it's `Set.univ`). -/
theorem mem_safeIterate_zero (cfg : GameConfig) (g : GameState) :
    g ∈ safeIterate cfg 0 := by
  rw [safeIterate_zero]; trivial

/-- Membership in `safeIterate cfg (n+1)` unfolds to `safeOp` application. -/
theorem mem_safeIterate_succ (cfg : GameConfig) (g : GameState) (n : ℕ) :
    g ∈ safeIterate cfg (n + 1) ↔ g ∈ safeOp cfg (safeIterate cfg n) :=
  Iff.rfl

/-- A state in `safeIterate cfg (n+1)` is not lost. -/
theorem not_lost_of_mem_safeIterate_succ (cfg : GameConfig)
    {g : GameState} {n : ℕ} (h : g ∈ safeIterate cfg (n + 1)) :
    ¬ g.lost cfg :=
  h.1

/-- A lost state is not in `safeIterate cfg (n+1)`. -/
theorem not_mem_safeIterate_succ_of_lost (cfg : GameConfig)
    {g : GameState} {n : ℕ} (h : g.lost cfg) :
    g ∉ safeIterate cfg (n + 1) :=
  fun hmem => not_lost_of_mem_safeIterate_succ cfg hmem h

/-- States in `safeIterate cfg n` for `1 ≤ n` are not lost. -/
theorem not_lost_of_mem_safeIterate_pos (cfg : GameConfig)
    {g : GameState} {n : ℕ} (hn : 1 ≤ n) (h : g ∈ safeIterate cfg n) :
    ¬ g.lost cfg := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.one_le_iff_ne_zero.mp hn)
  exact not_lost_of_mem_safeIterate_succ cfg h

/-- A lost state is not in `safeIterate cfg n` for `1 ≤ n`. -/
theorem not_mem_safeIterate_pos_of_lost (cfg : GameConfig)
    {g : GameState} {n : ℕ} (hn : 1 ≤ n) (h : g.lost cfg) :
    g ∉ safeIterate cfg n :=
  fun hmem => not_lost_of_mem_safeIterate_pos cfg hn hmem h

/-- The intersection of all positive iterates excludes lost states. -/
theorem not_lost_of_mem_iInter_safeIterate (cfg : GameConfig)
    {g : GameState} (h : g ∈ ⋂ n, safeIterate cfg n) :
    ¬ g.lost cfg :=
  not_lost_of_mem_safeIterate_succ cfg (Set.mem_iInter.mp h 1)

/-- Contrapositive: lost states are not in the intersection of all iterates. -/
theorem not_mem_iInter_safeIterate_of_lost (cfg : GameConfig)
    {g : GameState} (h : g.lost cfg) :
    g ∉ ⋂ n, safeIterate cfg n :=
  fun hmem => not_lost_of_mem_iInter_safeIterate cfg hmem h

/-- States in the intersection of iterates are not lost (set-form). -/
theorem iInter_safeIterate_subset_compl_lost (cfg : GameConfig) :
    (⋂ n, safeIterate cfg n) ⊆ { g | ¬ g.lost cfg } :=
  fun _ h => not_lost_of_mem_iInter_safeIterate cfg h

/-- Membership in `safe cfg` implies membership in every iterate. -/
theorem mem_safeIterate_of_mem_safe (cfg : GameConfig) (g : GameState)
    (h : g ∈ safe cfg) (n : ℕ) : g ∈ safeIterate cfg n :=
  safe_subset_safeIterate cfg n h

/-- Membership in `safe cfg` implies membership in the iterates' intersection. -/
theorem mem_iInter_safeIterate_of_mem_safe (cfg : GameConfig) (g : GameState)
    (h : g ∈ safe cfg) : g ∈ ⋂ n, safeIterate cfg n :=
  safe_subset_iInter cfg h

/-- If any iterate is empty, the safe set is empty. -/
theorem safe_eq_empty_of_safeIterate_eq_empty (cfg : GameConfig) {n : ℕ}
    (h : safeIterate cfg n = ∅) :
    safe cfg = ∅ :=
  Set.subset_eq_empty (safe_subset_safeIterate cfg n) h

/-- Contrapositive: the safe set is nonempty when any iterate is nonempty
*above* it. (Trivially, `safe ≠ ∅ → ∀ n, safeIterate cfg n ≠ ∅`.) -/
theorem safeIterate_ne_empty_of_safe_ne_empty (cfg : GameConfig) {n : ℕ}
    (h : safe cfg ≠ ∅) : safeIterate cfg n ≠ ∅ :=
  fun he => h (safe_eq_empty_of_safeIterate_eq_empty cfg he)

/-- If a state is excluded from any iterate, it's not in the safe set. -/
theorem not_mem_safe_of_not_mem_safeIterate (cfg : GameConfig) {g : GameState}
    {n : ℕ} (h : g ∉ safeIterate cfg n) : g ∉ safe cfg :=
  fun hsafe => h (mem_safeIterate_of_mem_safe cfg g hsafe n)

/-- The init state is in every iterate when it's in the safe set. -/
theorem init_mem_safeIterate_of_mem_safe (cfg : GameConfig)
    (h : GameState.init ∈ safe cfg) (n : ℕ) :
    GameState.init ∈ safeIterate cfg n :=
  mem_safeIterate_of_mem_safe cfg GameState.init h n

/-- The init state is excluded from `safe` if it's excluded from any iterate. -/
theorem init_not_mem_safe_of_not_mem_safeIterate (cfg : GameConfig) {n : ℕ}
    (h : GameState.init ∉ safeIterate cfg n) :
    GameState.init ∉ safe cfg :=
  not_mem_safe_of_not_mem_safeIterate cfg h

/-- The iterates' intersection is contained in any iterate. -/
theorem iInter_safeIterate_subset_safeIterate (cfg : GameConfig) (n : ℕ) :
    (⋂ k, safeIterate cfg k) ⊆ safeIterate cfg n :=
  Set.iInter_subset _ n

/-- A state in the iterates' intersection is in every iterate. -/
theorem mem_safeIterate_of_mem_iInter_safeIterate (cfg : GameConfig)
    {g : GameState} (h : g ∈ ⋂ k, safeIterate cfg k) (n : ℕ) :
    g ∈ safeIterate cfg n :=
  Set.mem_iInter.mp h n

/-- A state is in `safeIterate cfg 1` iff it's in `safeOp cfg Set.univ`. -/
theorem mem_safeIterate_one_iff (cfg : GameConfig) (g : GameState) :
    g ∈ safeIterate cfg 1 ↔ g ∈ safeOp cfg Set.univ := by
  rw [safeIterate_one]

/-- A state in `safeIterate cfg 1` is not lost. -/
theorem not_lost_of_mem_safeIterate_one (cfg : GameConfig) {g : GameState}
    (h : g ∈ safeIterate cfg 1) : ¬ g.lost cfg :=
  not_lost_of_mem_safeIterate_succ cfg h

/-- A lost state is not in `safeIterate cfg 1`. -/
theorem not_mem_safeIterate_one_of_lost (cfg : GameConfig) {g : GameState}
    (h : g.lost cfg) : g ∉ safeIterate cfg 1 :=
  not_mem_safeIterate_succ_of_lost cfg h

/-- `safeIterate cfg 2 = safeOp cfg (safeOp cfg Set.univ)`. -/
theorem safeIterate_two (cfg : GameConfig) :
    safeIterate cfg 2 = safeOp cfg (safeOp cfg Set.univ) := rfl

/-- A state in `safeIterate cfg 2` is not lost. -/
theorem not_lost_of_mem_safeIterate_two (cfg : GameConfig) {g : GameState}
    (h : g ∈ safeIterate cfg 2) : ¬ g.lost cfg :=
  not_lost_of_mem_safeIterate_succ cfg h

/-- A lost state is not in `safeIterate cfg 2`. -/
theorem not_mem_safeIterate_two_of_lost (cfg : GameConfig) {g : GameState}
    (h : g.lost cfg) : g ∉ safeIterate cfg 2 :=
  not_mem_safeIterate_succ_of_lost cfg h

/-- A state is in `safeIterate cfg 2` iff it's in
`safeOp cfg (safeIterate cfg 1)`. -/
theorem mem_safeIterate_two_iff (cfg : GameConfig) (g : GameState) :
    g ∈ safeIterate cfg 2 ↔ g ∈ safeOp cfg (safeIterate cfg 1) :=
  mem_safeIterate_succ cfg g 1

/-- Every iterate is a subset of `Set.univ` (trivially). -/
theorem safeIterate_subset_univ (cfg : GameConfig) (n : ℕ) :
    safeIterate cfg n ⊆ Set.univ := Set.subset_univ _

/-- `safeIterate cfg (n+1)` is disjoint from `{g | g.lost cfg}`. -/
theorem safeIterate_succ_disjoint_lost (cfg : GameConfig) (n : ℕ) :
    Disjoint (safeIterate cfg (n + 1)) { g | g.lost cfg } := by
  rw [Set.disjoint_left]
  exact fun _ h hlost => not_lost_of_mem_safeIterate_succ cfg h hlost

/-- Symmetric form: `{g | g.lost cfg}` is disjoint from `safeIterate cfg
(n+1)`. -/
theorem lost_disjoint_safeIterate_succ (cfg : GameConfig) (n : ℕ) :
    Disjoint { g | g.lost cfg } (safeIterate cfg (n + 1)) :=
  (safeIterate_succ_disjoint_lost cfg n).symm

/-- The intersection of all iterates is disjoint from the lost-state set. -/
theorem iInter_safeIterate_disjoint_lost (cfg : GameConfig) :
    Disjoint (⋂ n, safeIterate cfg n) { g | g.lost cfg } := by
  rw [Set.disjoint_left]
  exact fun _ h hlost => not_lost_of_mem_iInter_safeIterate cfg h hlost

/-- Symmetric form of `iInter_safeIterate_disjoint_lost`. -/
theorem lost_disjoint_iInter_safeIterate (cfg : GameConfig) :
    Disjoint { g | g.lost cfg } (⋂ n, safeIterate cfg n) :=
  (iInter_safeIterate_disjoint_lost cfg).symm

/-- `safeIterate cfg 3 = safeOp cfg (safeIterate cfg 2)`. -/
theorem safeIterate_three (cfg : GameConfig) :
    safeIterate cfg 3 = safeOp cfg (safeIterate cfg 2) := rfl

/-- A state in `safeIterate cfg 3` is not lost. -/
theorem not_lost_of_mem_safeIterate_three (cfg : GameConfig) {g : GameState}
    (h : g ∈ safeIterate cfg 3) : ¬ g.lost cfg :=
  not_lost_of_mem_safeIterate_succ cfg h

/-- A lost state is not in `safeIterate cfg 3`. -/
theorem not_mem_safeIterate_three_of_lost (cfg : GameConfig) {g : GameState}
    (h : g.lost cfg) : g ∉ safeIterate cfg 3 :=
  not_mem_safeIterate_succ_of_lost cfg h

/-- A state is in `safeIterate cfg 3` iff it's in
`safeOp cfg (safeIterate cfg 2)`. -/
theorem mem_safeIterate_three_iff (cfg : GameConfig) (g : GameState) :
    g ∈ safeIterate cfg 3 ↔ g ∈ safeOp cfg (safeIterate cfg 2) :=
  mem_safeIterate_succ cfg g 2

/-- `safeIterate cfg 3 ⊆ safeIterate cfg 2`. -/
theorem safeIterate_three_subset_two (cfg : GameConfig) :
    safeIterate cfg 3 ⊆ safeIterate cfg 2 := safeIterate_succ_subset cfg 2

/-- `safeIterate cfg 3 ⊆ safeIterate cfg 1` (transitivity). -/
theorem safeIterate_three_subset_one (cfg : GameConfig) :
    safeIterate cfg 3 ⊆ safeIterate cfg 1 :=
  safeIterate_antitone cfg (by decide : (1 : ℕ) ≤ 3)

/-- `safeIterate cfg 3 ⊆ safeIterate cfg 0`. -/
theorem safeIterate_three_subset_zero (cfg : GameConfig) :
    safeIterate cfg 3 ⊆ safeIterate cfg 0 :=
  safeIterate_antitone cfg (Nat.zero_le 3)

/-- `safeIterate cfg 1 ⊆ safeIterate cfg 0`. -/
theorem safeIterate_one_subset_zero (cfg : GameConfig) :
    safeIterate cfg 1 ⊆ safeIterate cfg 0 := safeIterate_succ_subset cfg 0

/-- `safeIterate cfg 2 ⊆ safeIterate cfg 1`. -/
theorem safeIterate_two_subset_one (cfg : GameConfig) :
    safeIterate cfg 2 ⊆ safeIterate cfg 1 := safeIterate_succ_subset cfg 1

/-- `safeIterate cfg 2 ⊆ safeIterate cfg 0` (via transitivity). -/
theorem safeIterate_two_subset_zero (cfg : GameConfig) :
    safeIterate cfg 2 ⊆ safeIterate cfg 0 :=
  safeIterate_antitone cfg (Nat.zero_le 2)

/-- If the iterates' intersection equals `safe`, equivalent (`safe`-as-glb)
form: every state in every iterate lies in `safe`. The reverse of
`safe_subset_iInter` requires `ω`-co-continuity; this is the
forward direction in a packaged set-equality. -/
theorem iInter_safeIterate_eq_safe_of_subset {cfg : GameConfig}
    (h : (⋂ n, safeIterate cfg n) ⊆ safe cfg) :
    (⋂ n, safeIterate cfg n) = safe cfg :=
  Set.Subset.antisymm h (safe_subset_iInter cfg)

/-- **Stable iterate = safe.** If `safeIterate cfg (n+1) = safeIterate cfg n`,
then this stable iterate is a fixed point of `safeOp`, hence (by
`safe_greatest`) contained in `safe`. Combined with
`safe_subset_safeIterate`, this gives equality. -/
theorem safeIterate_eq_safe_of_succ_eq {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg (n + 1) = safeIterate cfg n) :
    safeIterate cfg n = safe cfg := by
  refine Set.Subset.antisymm ?_ (safe_subset_safeIterate cfg n)
  apply safe_greatest
  -- need: safeIterate cfg n ⊆ safeOp cfg (safeIterate cfg n)
  have hfix : safeOp cfg (safeIterate cfg n) = safeIterate cfg n := by
    rw [← safeIterate_succ]; exact h
  rw [hfix]

/-- **Stabilisation propagates upward**: once
`safeIterate cfg (n+1) = safeIterate cfg n`, the chain is constant
from index `n` onward. -/
theorem safeIterate_add_eq_of_succ_eq {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg (n + 1) = safeIterate cfg n) (k : ℕ) :
    safeIterate cfg (n + k) = safeIterate cfg n := by
  induction k with
  | zero => rfl
  | succ j ih =>
    rw [show n + (j + 1) = (n + j) + 1 from by omega,
        safeIterate_succ, ih,
        ← safeIterate_succ, h]

/-- **`≥ n`-form**: combining Ticks 1334 and 1335, every iterate beyond
the stabilisation point equals `safe cfg`. -/
theorem safeIterate_eq_safe_of_succ_eq_of_le {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg (n + 1) = safeIterate cfg n) {k : ℕ} (hk : n ≤ k) :
    safeIterate cfg k = safe cfg := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hk
  rw [safeIterate_add_eq_of_succ_eq h j, safeIterate_eq_safe_of_succ_eq h]

/-- **Stabilisation iff**: `safeIterate cfg (n+1) = safeIterate cfg n`
iff that iterate equals `safe cfg`. Forward direction is Tick 1334;
reverse uses the `safeOp`-fixed-point property of `safe`. -/
theorem safeIterate_succ_eq_iff_eq_safe {cfg : GameConfig} {n : ℕ} :
    safeIterate cfg (n + 1) = safeIterate cfg n ↔
      safeIterate cfg n = safe cfg := by
  refine ⟨safeIterate_eq_safe_of_succ_eq, fun h => ?_⟩
  rw [safeIterate_succ, h, safe_eq]

/-- **Strict-decrease version**: if no iterate has stabilised through
index `n`, every iterate up through `n+1` is a strict subset of its
predecessor. (Contrapositive of Tick 1337's stabilisation iff in
strict-subset form.) -/
theorem safeIterate_succ_ssubset_of_ne_safe {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg n ≠ safe cfg) :
    safeIterate cfg (n + 1) ⊂ safeIterate cfg n := by
  refine ⟨safeIterate_succ_subset cfg n, ?_⟩
  intro hge
  exact h (safeIterate_succ_eq_iff_eq_safe.mp
    (Set.Subset.antisymm (safeIterate_succ_subset cfg n) hge))

/-- **Strict-decrease iff**: `safeIterate cfg (n+1) ⊂ safeIterate cfg n`
iff `safeIterate cfg n ≠ safe cfg`. Combines Tick 1338 with
`safe_subset_safeIterate` (strict subset rules out equality). -/
theorem safeIterate_succ_ssubset_iff_ne_safe {cfg : GameConfig} {n : ℕ} :
    safeIterate cfg (n + 1) ⊂ safeIterate cfg n ↔
      safeIterate cfg n ≠ safe cfg := by
  refine ⟨fun hss heq => ?_, safeIterate_succ_ssubset_of_ne_safe⟩
  exact hss.ne (safeIterate_succ_eq_iff_eq_safe.mpr heq)

/-- **Forward propagation**: `safeIterate cfg n = safe cfg ⇒
safeIterate cfg (n+1) = safe cfg`. Direct via `safeIterate_succ`
and `safe_eq`. (The reverse direction is false in general:
`safeIterate cfg (n+1) = safe` can hold while `safeIterate cfg n`
remains strictly larger than `safe`.) -/
theorem safeIterate_succ_eq_safe_of_eq_safe {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg n = safe cfg) :
    safeIterate cfg (n + 1) = safe cfg := by
  rw [safeIterate_succ, h, safe_eq]

/-- **Forward propagation to any later index**: once `safeIterate cfg n
= safe cfg`, every later iterate equals `safe cfg`. Iterates Tick 1340. -/
theorem safeIterate_add_eq_safe_of_eq_safe {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg n = safe cfg) (k : ℕ) :
    safeIterate cfg (n + k) = safe cfg := by
  induction k with
  | zero => exact h
  | succ j ih =>
    rw [show n + (j + 1) = (n + j) + 1 from by omega]
    exact safeIterate_succ_eq_safe_of_eq_safe ih

/-- **`≥ n` form**: once `safeIterate cfg n = safe cfg`, every iterate
at index `k ≥ n` equals `safe cfg`. Derived from Tick 1341. -/
theorem safeIterate_eq_safe_of_eq_safe_of_le {cfg : GameConfig} {n : ℕ}
    (h : safeIterate cfg n = safe cfg) {k : ℕ} (hk : n ≤ k) :
    safeIterate cfg k = safe cfg := by
  obtain ⟨j, rfl⟩ := Nat.exists_eq_add_of_le hk
  exact safeIterate_add_eq_safe_of_eq_safe h j

end Tetris
