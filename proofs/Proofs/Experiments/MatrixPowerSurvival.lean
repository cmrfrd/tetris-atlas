import Mathlib

/-!
# MatrixPowerSurvival — survival as a finite Boolean linear dynamical system

A re-presentation of the Tetris survival problem in the language of **Boolean matrix powers**
over a finite state space, following the matrix-power formulation. The point is to recast
"there is an infinite safe play" as a *standard* finite-graph / linear-algebra statement:

* every complete state is a basis vector `eᵢ ∈ {0,1}ᴺ`;
* each "place piece `a`, draw next piece `r`" transition is a Boolean matrix `T_{a,r}`;
* the "try-all-placements, try-all-legal-draws" graph is the adjacency matrix `A = ⋁ T_{a,r}`;
* death `⊥` is an absorbing sink, projected out into the **safe** traversal relation `B`;
* traversal of length `t` is the matrix power `Bᵗ`; reachability is `⋁ₜ Bᵗ`;
* a **safe cycle** is a diagonal entry `(Bᵏ)_{i,i} = 1`;
* **infinite safe play ⟺ a safe cycle is reachable from the start** (the weak/existential view), and
* **robust 7-bag survival ⟺ a reachable safe *recurrent support*** `X` (the strong view).

This file develops the framework **abstractly**, over an arbitrary state type. Per the development
plan we lead with **relations** as the primary object (a relation `R : State → State → Prop` *is*
the Boolean adjacency matrix; relation powers *are* matrix powers); the literal `Matrix`-over-a-
semiring correspondence is an optional layer added later (walk-counting: `0 < (Aⁿ) i j ↔ Rⁿ i j`).

## Honest scope

The abstract theorems here — *reachable cycle ⇒ infinite safe path*, *reachable recurrent support
⇒ survival* — are pure finite-graph mathematics and go through cleanly. They are the **easy half**:
they re-package, in standard matrix-power vocabulary, the survival reduction the green library
already proves concretely (`Tetris.safe`, `Tetris.AdversarialClosedCycle`, `closed_cycle_survives`).
What this reframing does **not** do is dissolve the project's irreducible crux: it relocates
"survival" to "**exhibit a concrete nonempty `X`** that is reachable and closed under every 7-bag
draw" — i.e. the support-indicator vector `x` must satisfy `x ⊨ x` under the `B_r`'s. That is the
I-drain realization crux (#66/#72) every prior route converged onto. The matrix powers are a
*presentation* of reachability, not a smaller `N`.

## Layout

* **§1 Phase 1** — abstract reachability (`RelPow`, `Reachable`), cycles (`OnCycle`,
  `ReachableCycle`), infinite paths (`InfiniteSafePath`), and the headline
  `reachable_cycle_implies_infinite_safe_path`.
* later phases (recurrent support, faithful all-sequences survival, matrix correspondence,
  concrete instantiation) are added incrementally below.
-/

namespace MatrixPowerSurvival

universe u

variable {State : Type u}

/-! ## §1 Phase 1 — abstract finite-graph reachability and cycles

Everything in this section is generic over a relation `R : State → State → Prop`, read as the
Boolean adjacency matrix of the transition graph (`R s t` ⇔ entry `(t,s) = 1`). No finiteness is
needed for the headline theorem; finiteness enters only for the matrix-power / pigeonhole layer. -/

/-- `RelPow R n s t`: an `R`-walk of **length exactly `n`** from `s` to `t`. This is the relational
reading of the Boolean matrix power: `RelPow R n s t` ⇔ `(Bⁿ)_{t,s} = 1`. The walk is built
front-to-back (`succ` prepends an edge `R s u`). -/
inductive RelPow (R : State → State → Prop) : ℕ → State → State → Prop
  | zero (s : State) : RelPow R 0 s s
  | succ {n : ℕ} {s u t : State} : R s u → RelPow R n u t → RelPow R (n + 1) s t

/-- Reachability is the reflexive-transitive closure of `R` — i.e. `⋁ₙ Bⁿ`. We use Mathlib's
`Relation.ReflTransGen` so the whole closure API (including `head_induction_on`) is available;
`reachable_iff_exists_relPow` shows it equals the `∃ n, RelPow` form of the matrix-power union. -/
abbrev Reachable (R : State → State → Prop) : State → State → Prop :=
  Relation.ReflTransGen R

/-- `OnCycle R s`: the state `s` lies on a closed `R`-walk of **positive** length — exactly a
nonzero diagonal entry `(Bᵏ)_{s,s} = 1` for some `k > 0`. -/
def OnCycle (R : State → State → Prop) (s : State) : Prop :=
  ∃ k, 0 < k ∧ RelPow R k s s

/-- `ReachableCycle R s₀`: some cycle vertex is reachable from `s₀`. The matrix-power statement
`∃ i m k, (k>0) ∧ (Bᵐ)_{i,s₀}=1 ∧ (Bᵏ)_{i,i}=1`. -/
def ReachableCycle (R : State → State → Prop) (s₀ : State) : Prop :=
  ∃ c, Reachable R s₀ c ∧ OnCycle R c

/-- An infinite `R`-path out of `s₀` that never visits `death`: the certificate of infinite safe
play. (Here `R` is meant to be the *safe* traversal relation `B`.) -/
def InfiniteSafePath (R : State → State → Prop) (death : State) (s₀ : State) : Prop :=
  ∃ path : ℕ → State, path 0 = s₀ ∧ (∀ n, path n ≠ death) ∧ ∀ n, R (path n) (path (n + 1))

/-! ### Relation powers ↔ reachability -/

/-- A length-`n` walk is in particular a reachability witness (`Bⁿ ≤ ⋁ₘ Bᵐ`). -/
theorem RelPow.reachable {R : State → State → Prop} :
    ∀ {n : ℕ} {s t : State}, RelPow R n s t → Reachable R s t := by
  intro n s t h
  induction h with
  | zero s => exact Relation.ReflTransGen.refl
  | succ hsu _ ih => exact Relation.ReflTransGen.head hsu ih

/-- Append an edge at the end of a walk (`Bⁿ · B = Bⁿ⁺¹`). -/
theorem RelPow.snoc {R : State → State → Prop} :
    ∀ {n : ℕ} {s t u : State}, RelPow R n s t → R t u → RelPow R (n + 1) s u := by
  intro n s t u h htu
  induction h with
  | zero s => exact RelPow.succ htu (RelPow.zero u)
  | succ hab _ ih => exact RelPow.succ hab (ih htu)

/-- Reachability is exactly the union of all relation powers: `ReflTransGen R = ⋁ₙ RelPow R n`. -/
theorem reachable_iff_exists_relPow {R : State → State → Prop} {s t : State} :
    Reachable R s t ↔ ∃ n, RelPow R n s t := by
  constructor
  · intro h
    induction h with
    | refl => exact ⟨0, RelPow.zero s⟩
    | tail _ hbc ih =>
      obtain ⟨n, hn⟩ := ih
      exact ⟨n + 1, hn.snoc hbc⟩
  · rintro ⟨n, hn⟩
    exact hn.reachable

/-- **Chain extraction.** A length-`n` walk `RelPow R n s t` is witnessed by an explicit vertex
sequence `w : ℕ → State` with `w 0 = s`, `w n = t`, and `R (w i) (w (i+1))` for every `i < n`. This
materializes the abstract power as a concrete finite path — the bridge to tiling cycles. -/
theorem RelPow.exists_chain {R : State → State → Prop} :
    ∀ {n : ℕ} {s t : State}, RelPow R n s t →
      ∃ w : ℕ → State, w 0 = s ∧ w n = t ∧ ∀ i, i < n → R (w i) (w (i + 1)) := by
  intro n s t h
  induction h with
  | zero s => exact ⟨fun _ => s, rfl, rfl, fun i hi => (Nat.not_lt_zero i hi).elim⟩
  | @succ m s u t hsu _htail ih =>
    obtain ⟨w, hw0, hwm, hstep⟩ := ih
    refine ⟨fun i => match i with | 0 => s | j + 1 => w j, rfl, hwm, ?_⟩
    intro i hi
    match i with
    | 0 => change R s (w 0); rw [hw0]; exact hsu
    | j + 1 => change R (w j) (w (j + 1)); exact hstep j (Nat.lt_of_succ_lt_succ hi)

/-! ### The two path-construction engines -/

/-- **Closed-set continuation engine.** If every vertex of `X` has an `R`-successor inside `X`
(`X` is a B-invariant support), then from any `c ∈ X` there is an infinite `R`-path staying in `X`.
Built by dependent choice on the successor map of the finite-branching relation. -/
theorem exists_infinite_continuation {R : State → State → Prop} {X : Set State}
    (hclosed : ∀ s ∈ X, ∃ t ∈ X, R s t) {c : State} (hc : c ∈ X) :
    ∃ f : ℕ → State, f 0 = c ∧ (∀ n, f n ∈ X) ∧ ∀ n, R (f n) (f (n + 1)) := by
  classical
  let next : {s // s ∈ X} → {s // s ∈ X} := fun p =>
    ⟨Classical.choose (hclosed p.1 p.2), (Classical.choose_spec (hclosed p.1 p.2)).1⟩
  refine ⟨fun n => (next^[n] ⟨c, hc⟩).1, rfl, fun n => (next^[n] ⟨c, hc⟩).2, fun n => ?_⟩
  change R ((next^[n] ⟨c, hc⟩).1) ((next^[n + 1] ⟨c, hc⟩).1)
  rw [Function.iterate_succ_apply' next n ⟨c, hc⟩]
  exact (Classical.choose_spec (hclosed _ (next^[n] ⟨c, hc⟩).2)).2

/-- **Prefix-prepend engine.** If `c` is reachable from `s₀` and `c` starts an infinite `R`-path,
then `s₀` starts an infinite `R`-path. Proved by head-induction on the reachability witness (the
right endpoint `c` stays fixed, so the continuation `g` is never generalized). -/
theorem infinite_path_of_reachable_of_continuation {R : State → State → Prop} {s₀ c : State}
    (hreach : Reachable R s₀ c)
    {g : ℕ → State} (hg0 : g 0 = c) (hgstep : ∀ n, R (g n) (g (n + 1))) :
    ∃ h : ℕ → State, h 0 = s₀ ∧ ∀ n, R (h n) (h (n + 1)) := by
  induction hreach using Relation.ReflTransGen.head_induction_on with
  | refl => exact ⟨g, hg0, hgstep⟩
  | @head a mid hmid _htail ih =>
    obtain ⟨h, hh0, hhstep⟩ := ih
    refine ⟨fun k => match k with | 0 => a | j + 1 => h j, rfl, ?_⟩
    intro k
    match k with
    | 0 => change R a (h 0); rw [hh0]; exact hmid
    | j + 1 => change R (h j) (h (j + 1)); exact hhstep j

/-- A cycle yields an infinite path starting at the cycle vertex (tile the cycle forever). The
cycle's finite vertex set is a closed support, so this is a special case of the continuation
engine. -/
theorem OnCycle.continuation {R : State → State → Prop} {c : State} (h : OnCycle R c) :
    ∃ g : ℕ → State, g 0 = c ∧ ∀ n, R (g n) (g (n + 1)) := by
  obtain ⟨k, hk, hcyc⟩ := h
  obtain ⟨w, hw0, hwk, hstep⟩ := hcyc.exists_chain
  set X : Set State := {x | ∃ i, i < k ∧ w i = x} with hX
  have hcX : c ∈ X := ⟨0, hk, hw0⟩
  have hclosed : ∀ s ∈ X, ∃ t ∈ X, R s t := by
    rintro x ⟨i, hik, rfl⟩
    by_cases hi : i + 1 < k
    · exact ⟨w (i + 1), ⟨i + 1, hi, rfl⟩, hstep i hik⟩
    · have hik1 : i + 1 = k := by omega
      refine ⟨w 0, ⟨0, hk, rfl⟩, ?_⟩
      have hr := hstep i hik
      rw [hik1, hwk] at hr
      rwa [hw0]
  obtain ⟨g, hg0, _, hgstep⟩ := exists_infinite_continuation hclosed hcX
  exact ⟨g, hg0, hgstep⟩

/-! ### Headline: a reachable safe cycle ⇒ infinite safe play -/

/-- **Reachable safe cycle ⇒ infinite safe path.** If every `R`-edge avoids `death` on both ends
(`R` is the *safe* traversal relation `B`) and a cycle is reachable from `s₀`, then there is an
infinite `R`-path from `s₀` that never dies. This is the matrix-power statement
`(∃ i m k>0, (Bᵐ)_{i,s₀}=1 ∧ (Bᵏ)_{i,i}=1) ⇒ ∃ infinite safe orbit`, and it is pure finite-graph
math (no `Fintype` needed). -/
theorem reachable_cycle_implies_infinite_safe_path {R : State → State → Prop} {death : State}
    (hSafeEdges : ∀ s t, R s t → s ≠ death ∧ t ≠ death)
    {s₀ : State} (hcyc : ReachableCycle R s₀) :
    InfiniteSafePath R death s₀ := by
  obtain ⟨c, hreach, hOnCycle⟩ := hcyc
  obtain ⟨g, hg0, hgstep⟩ := hOnCycle.continuation
  obtain ⟨h, hh0, hhstep⟩ := infinite_path_of_reachable_of_continuation hreach hg0 hgstep
  exact ⟨h, hh0, fun n => (hSafeEdges _ _ (hhstep n)).1, hhstep⟩

end MatrixPowerSurvival
