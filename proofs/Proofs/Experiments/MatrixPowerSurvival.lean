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

/-- An infinite `R`-path out of `s₀` that never enters the dead set `Dead`: the certificate of
infinite safe play. (`R` is meant to be the *safe* traversal relation `B`.) We take death as a
**predicate** `Dead : State → Prop`, not a single absorbing state — strictly more general (the doc's
sink `⊥` is the instance `Dead := (· = ⊥)`) and a match for Tetris, where loss is the predicate
`GameState.lost`, not one state. -/
def InfiniteSafePath (R : State → State → Prop) (Dead : State → Prop) (s₀ : State) : Prop :=
  ∃ path : ℕ → State, path 0 = s₀ ∧ (∀ n, ¬ Dead (path n)) ∧ ∀ n, R (path n) (path (n + 1))

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

/-- **Reachable safe cycle ⇒ infinite safe path.** If every `R`-edge avoids the dead set on both
ends (`R` is the *safe* traversal relation `B`) and a cycle is reachable from `s₀`, then there is an
infinite `R`-path from `s₀` that never dies. This is the matrix-power statement
`(∃ i m k>0, (Bᵐ)_{i,s₀}=1 ∧ (Bᵏ)_{i,i}=1) ⇒ ∃ infinite safe orbit`, and it is pure finite-graph
math (no `Fintype` needed). -/
theorem reachable_cycle_implies_infinite_safe_path {R : State → State → Prop}
    {Dead : State → Prop}
    (hSafeEdges : ∀ s t, R s t → ¬ Dead s ∧ ¬ Dead t)
    {s₀ : State} (hcyc : ReachableCycle R s₀) :
    InfiniteSafePath R Dead s₀ := by
  obtain ⟨c, hreach, hOnCycle⟩ := hcyc
  obtain ⟨g, hg0, hgstep⟩ := hOnCycle.continuation
  obtain ⟨h, hh0, hhstep⟩ := infinite_path_of_reachable_of_continuation hreach hg0 hgstep
  exact ⟨h, hh0, fun n => (hSafeEdges _ _ (hhstep n)).1, hhstep⟩

/-- **Combined reachable-closed engine.** A dead-avoiding closed support `X` (every vertex has an
in-`X` successor) with some vertex reachable from `s₀` yields an infinite safe path from `s₀`. This
is the general workhorse behind every recurrent-support theorem below: reach `X`, then stay. -/
theorem infinite_safe_path_of_reachable_closed {R : State → State → Prop} {Dead : State → Prop}
    (hSafeEdges : ∀ s t, R s t → ¬ Dead s ∧ ¬ Dead t)
    {s₀ : State} {X : Set State}
    (hclosed : ∀ s ∈ X, ∃ t ∈ X, R s t)
    (hReach : ∃ c ∈ X, Reachable R s₀ c) :
    InfiniteSafePath R Dead s₀ := by
  obtain ⟨c, hcX, hreach⟩ := hReach
  obtain ⟨g, hg0, _, hgstep⟩ := exists_infinite_continuation hclosed hcX
  obtain ⟨h, hh0, hhstep⟩ := infinite_path_of_reachable_of_continuation hreach hg0 hgstep
  exact ⟨h, hh0, fun n => (hSafeEdges _ _ (hhstep n)).1, hhstep⟩

/-! ## §2 Phase 4 — safe recurrent support (the policy-free survival certificate)

We now equip the graph with Tetris-flavoured structure: a placement `Action`, a next-`Piece` draw,
a legal-draw set `legalDraws s` (the current 7-bag), and a transition `step s a r` (place piece via
`a`, then draw `r`). Two quantifier orders for a *closed support* `X` matter, and they are NOT the
same condition:

* **weak** `SafeRecurrentSupport` (`∀ r, ∃ a`): for each next piece there is *some* placement
  staying in `X`. The placement may peek at `r`, so this is a relaxation — useful as a warm-up.
* **online** `OnlineSafeRecurrentSupport` (`∃ a, ∀ r`): *one* placement of the current piece keeps
  us in `X` no matter what is drawn next. This is the **faithful** survival condition — you commit
  before the draw. It is strictly stronger (`online_implies_weak`).

The "matrix form" of either is a fixed-point on the support indicator `x ∈ {0,1}ᴺ`: `x ⊨ x` under
the piece-labeled transition matrices. Exhibiting such an `x` is the open Tetris crux; here we prove
that *given* one (reachable, dead-avoiding), infinite safe play follows. -/

/-- The "try-all-placements" edge for a fixed next draw is rolled into one existential edge:
`EdgeExists s t` ⇔ some legal next piece `r` and some placement `a` send `s` to `t`. This is the
adjacency `A = ⋁_{a,r} T_{a,r}` of the full traversal graph. -/
def EdgeExists {Piece Action : Type*} (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (s t : State) : Prop :=
  ∃ r ∈ legalDraws s, ∃ a : Action, step s a r = t

/-- The safe traversal relation `B = P_safe · A · P_safe`: an `EdgeExists` edge whose endpoints both
avoid the dead set. -/
def SafeEdge {Piece Action : Type*} (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (s t : State) : Prop :=
  ¬ Dead s ∧ ¬ Dead t ∧ EdgeExists legalDraws step s t

/-- **Weak (`∀ r, ∃ a`) safe recurrent support.** `X` avoids the dead set and from every `s ∈ X`,
for each legal next piece there is *some* placement landing back in `X` (placement may peek at the
piece). -/
def SafeRecurrentSupport {Piece Action : Type*} (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State)
    (X : Set State) : Prop :=
  ∀ s ∈ X, ¬ Dead s ∧ ∀ r ∈ legalDraws s, ∃ a : Action, step s a r ∈ X

/-- **Online (`∃ a, ∀ r`) safe recurrent support — the faithful condition.** `X` avoids the dead set
and from every `s ∈ X` there is *one* placement that keeps us in `X` under *every* legal next draw.
Commit-before-draw: this is what real online survival requires. -/
def OnlineSafeRecurrentSupport {Piece Action : Type*} (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State)
    (X : Set State) : Prop :=
  ∀ s ∈ X, ¬ Dead s ∧ ∃ a : Action, ∀ r ∈ legalDraws s, step s a r ∈ X

/-- The support is reached from the start: some `x ∈ X` lies in `⋁ₘ Bᵐ e_{s₀}`. -/
def SupportReachable (R : State → State → Prop) (s₀ : State) (X : Set State) : Prop :=
  ∃ x ∈ X, Reachable R s₀ x

/-- A safe edge avoids the dead set at both endpoints (`P_safe` projection). -/
theorem safeEdge_endpoints {Piece Action : Type*} {Dead : State → Prop}
    {legalDraws : State → Finset Piece} {step : State → Action → Piece → State}
    {s t : State} (h : SafeEdge Dead legalDraws step s t) : ¬ Dead s ∧ ¬ Dead t :=
  ⟨h.1, h.2.1⟩

/-- **Online support is stronger than weak support** (`∃ a, ∀ r` ⟹ `∀ r, ∃ a`). -/
theorem online_implies_weak {Piece Action : Type*} {Dead : State → Prop}
    {legalDraws : State → Finset Piece} {step : State → Action → Piece → State}
    {X : Set State} (h : OnlineSafeRecurrentSupport Dead legalDraws step X) :
    SafeRecurrentSupport Dead legalDraws step X := by
  intro s hs
  obtain ⟨hsne, a, ha⟩ := h s hs
  exact ⟨hsne, fun r hr => ⟨a, ha r hr⟩⟩

/-- **Weak recurrent support ⇒ infinite survival.** A reachable, dead-avoiding weak support `X`
(with nonempty legal-draw sets, as in any real bag) yields an infinite safe `B`-path from `s₀`. The
support is a closed set for `SafeEdge`, so this is `infinite_safe_path_of_reachable_closed`. -/
theorem reachable_safe_recurrent_support_implies_infinite_survival
    {Piece Action : Type*} {Dead : State → Prop}
    {legalDraws : State → Finset Piece} {step : State → Action → Piece → State}
    {s₀ : State} {X : Set State}
    (hLegal : ∀ s ∈ X, (legalDraws s).Nonempty)
    (hX : SafeRecurrentSupport Dead legalDraws step X)
    (hReach : SupportReachable (SafeEdge Dead legalDraws step) s₀ X) :
    InfiniteSafePath (SafeEdge Dead legalDraws step) Dead s₀ := by
  refine infinite_safe_path_of_reachable_closed (fun _ _ h => safeEdge_endpoints h) ?_ hReach
  intro s hs
  obtain ⟨hsne, hsucc⟩ := hX s hs
  obtain ⟨r, hr⟩ := hLegal s hs
  obtain ⟨a, hax⟩ := hsucc r hr
  exact ⟨step s a r, hax, hsne, (hX (step s a r) hax).1, ⟨r, hr, a, rfl⟩⟩

/-- **Online recurrent support ⇒ infinite survival (faithful headline).** A reachable, dead-avoiding
online support `X` yields an infinite safe `B`-path from `s₀`. Reduces to the weak theorem via
`online_implies_weak`. -/
theorem reachable_online_safe_recurrent_support_implies_infinite_survival
    {Piece Action : Type*} {Dead : State → Prop}
    {legalDraws : State → Finset Piece} {step : State → Action → Piece → State}
    {s₀ : State} {X : Set State}
    (hLegal : ∀ s ∈ X, (legalDraws s).Nonempty)
    (hX : OnlineSafeRecurrentSupport Dead legalDraws step X)
    (hReach : SupportReachable (SafeEdge Dead legalDraws step) s₀ X) :
    InfiniteSafePath (SafeEdge Dead legalDraws step) Dead s₀ :=
  reachable_safe_recurrent_support_implies_infinite_survival hLegal (online_implies_weak hX) hReach

/-! ## §3 Faithful all-sequences survival (the adversarial headline)

§2 produces *one* infinite safe orbit — existence of *a* surviving trajectory. The faithful 7-bag
claim is stronger: a single committed policy must survive **every** legal piece sequence the
adversary can draw. The online support is exactly the certificate for that. We make the adversarial
play explicit and prove the abstract analogue of `closed_cycle_survives`: a dead-avoiding online
support is invariant under the policy that picks its witnessed placement, for all adversaries. -/

/-- The adversarial trace: from `s₀`, at each step place the current piece via policy `π`, then the
adversary reveals the next piece `seq n`. Abstract mirror of `Tetris.adversarialTrace`. -/
def absTrace {Piece Action : Type*} (step : State → Action → Piece → State)
    (π : State → Action) (seq : ℕ → Piece) (s₀ : State) : ℕ → State
  | 0 => s₀
  | n + 1 => step (absTrace step π seq s₀ n) (π (absTrace step π seq s₀ n)) (seq n)

@[simp] theorem absTrace_zero {Piece Action : Type*} (step : State → Action → Piece → State)
    (π : State → Action) (seq : ℕ → Piece) (s₀ : State) :
    absTrace step π seq s₀ 0 = s₀ := rfl

theorem absTrace_succ {Piece Action : Type*} (step : State → Action → Piece → State)
    (π : State → Action) (seq : ℕ → Piece) (s₀ : State) (n : ℕ) :
    absTrace step π seq s₀ (n + 1) =
      step (absTrace step π seq s₀ n) (π (absTrace step π seq s₀ n)) (seq n) := rfl

/-- **Online support ⇒ a policy invariant under every adversary.** From a dead-avoiding online
support `X` and an entry `s₀ ∈ X`, there is a policy `π` whose adversarial trace stays inside `X`
for *every* legal piece sequence. (Legality is "along the trace": each draw lies in the current
`legalDraws`.) This is the abstract `closed_cycle_survives` — the all-`∀r` strength of the online
condition is exactly what makes the invariant hold against an adversary, not just one sequence. -/
theorem online_support_invariant_all_sequences {Piece Action : Type*} [Nonempty Action]
    {Dead : State → Prop} {legalDraws : State → Finset Piece}
    {step : State → Action → Piece → State}
    {X : Set State} (hX : OnlineSafeRecurrentSupport Dead legalDraws step X)
    {s₀ : State} (hs₀ : s₀ ∈ X) :
    ∃ π : State → Action, ∀ seq : ℕ → Piece,
      (∀ n, seq n ∈ legalDraws (absTrace step π seq s₀ n)) →
      ∀ n, absTrace step π seq s₀ n ∈ X := by
  classical
  -- A placement choice for every state: the online witness on `X`, arbitrary off `X`.
  have hpick : ∀ s, ∃ a : Action, s ∈ X → ∀ r ∈ legalDraws s, step s a r ∈ X := by
    intro s
    by_cases h : s ∈ X
    · obtain ⟨a, ha⟩ := (hX s h).2
      exact ⟨a, fun _ => ha⟩
    · exact ⟨Classical.arbitrary Action, fun hc => absurd hc h⟩
  refine ⟨fun s => Classical.choose (hpick s), fun seq hlegal n => ?_⟩
  induction n with
  | zero => exact hs₀
  | succ k ih =>
    rw [absTrace_succ]
    exact Classical.choose_spec (hpick _) ih (seq k) (hlegal k)

/-- **Faithful adversarial survival.** Same hypotheses as above: the witnessed policy never tops
out against any legal adversary — `¬ Dead (absTrace … n)` for all `n`. Corollary of the invariant
plus dead-avoidance of `X`. The abstract, online (`∃a∀r`) survival statement the Atlas mission
targets. -/
theorem online_support_survives_all_sequences {Piece Action : Type*} [Nonempty Action]
    {Dead : State → Prop} {legalDraws : State → Finset Piece}
    {step : State → Action → Piece → State}
    {X : Set State} (hX : OnlineSafeRecurrentSupport Dead legalDraws step X)
    {s₀ : State} (hs₀ : s₀ ∈ X) :
    ∃ π : State → Action, ∀ seq : ℕ → Piece,
      (∀ n, seq n ∈ legalDraws (absTrace step π seq s₀ n)) →
      ∀ n, ¬ Dead (absTrace step π seq s₀ n) := by
  obtain ⟨π, hπ⟩ := online_support_invariant_all_sequences hX hs₀
  exact ⟨π, fun seq hlegal n => (hX _ (hπ seq hlegal n)).1⟩

/-! ## §4 The Boolean matrix-power correspondence

Everything above led with *relations*. Here we make the title literal: a relation `R` on a finite
state space is the support of its adjacency matrix `adj R : Matrix State State ℕ` (0/1 entries), and
its relation powers are exactly the *positivity pattern* of the matrix powers `adjⁿ`. This is the
classical theorem that **adjacency-matrix powers count walks**, in Boolean (positivity) form, built
on Mathlib's matrix monoid (`pow_succ'`, `Matrix.mul_apply`). It justifies reading every preceding
statement as a Boolean linear-algebra statement:

* reachability `⋁ₙ Bⁿ`  ↔  `∃ n, 0 < (adjⁿ) i j`  (`reachable_iff_exists_adj_pow_pos`)
* a safe cycle `(Bᵏ)_{s,s}=1`  ↔  `∃ k>0, 0 < (adjᵏ) s s`  (`onCycle_iff_exists_pos_diag`). -/

section MatrixPower

variable (R : State → State → Prop) [DecidableRel R]

/-- The adjacency matrix of `R` as an ℕ-matrix with 0/1 entries: `adj i j = 1 ⟺ R i j`. (We use ℕ
rather than a Boolean semiring so that Mathlib's matrix-power monoid and walk-counting are directly
available; only the *positivity* of entries is ever used, so this is the Boolean matrix `B`.) -/
def adj : Matrix State State ℕ := fun i j => if R i j then 1 else 0

/-- An adjacency entry is positive exactly on edges. -/
theorem adj_pos_iff (i j : State) : 0 < adj R i j ↔ R i j := by
  unfold adj; split <;> simp_all

variable [Fintype State] [DecidableEq State]

/-- **Walk-counting (Boolean form).** `0 < (adjⁿ)_{i,j}` iff there is an `R`-walk of length `n` from
`i` to `j` — i.e. the matrix power `Bⁿ` has a nonzero `(i,j)` entry iff `RelPow R n i j`. Proved by
induction on `n` via `adjⁿ⁺¹ = adj · adjⁿ` and "a finite ℕ-sum is positive iff some summand is". -/
theorem relPow_iff_adj_pow_pos (n : ℕ) (i j : State) :
    0 < (adj R ^ n) i j ↔ RelPow R n i j := by
  induction n generalizing i with
  | zero =>
    rw [pow_zero]
    constructor
    · intro h
      by_cases hij : i = j
      · subst hij; exact RelPow.zero i
      · rw [Matrix.one_apply_ne hij] at h; exact absurd h (lt_irrefl 0)
    · intro h
      cases h
      rw [Matrix.one_apply_eq]; exact Nat.one_pos
  | succ k ih =>
    rw [pow_succ', Matrix.mul_apply]
    constructor
    · intro h
      obtain ⟨x, _, hx⟩ := Finset.exists_ne_zero_of_sum_ne_zero (Nat.pos_iff_ne_zero.mp h)
      rw [mul_ne_zero_iff] at hx
      exact RelPow.succ ((adj_pos_iff R i x).mp (Nat.pos_of_ne_zero hx.1))
        ((ih x).mp (Nat.pos_of_ne_zero hx.2))
    · intro h
      cases h with
      | succ hiu huj =>
        rename_i u
        refine Finset.sum_pos' (fun y _ => Nat.zero_le _) ⟨u, Finset.mem_univ u, ?_⟩
        exact mul_pos ((adj_pos_iff R i u).mpr hiu) ((ih u).mpr huj)

/-- **Reachability is the union of matrix powers.** `Reachable R i j ↔ ∃ n, 0 < (adjⁿ)_{i,j}`, the
literal `⋁ₙ Bⁿ` reading of the transitive-reflexive closure. -/
theorem reachable_iff_exists_adj_pow_pos (i j : State) :
    Reachable R i j ↔ ∃ n, 0 < (adj R ^ n) i j := by
  rw [reachable_iff_exists_relPow]
  exact exists_congr fun n => (relPow_iff_adj_pow_pos R n i j).symm

/-- **A cycle is a nonzero diagonal entry of a matrix power.** Exactly the matrix-power statement
`(Bᵏ)_{s,s}=1` for some `k>0`: `OnCycle R s ↔ ∃ k>0, 0 < (adjᵏ)_{s,s}`. -/
theorem onCycle_iff_exists_pos_diag (s : State) :
    OnCycle R s ↔ ∃ k, 0 < k ∧ 0 < (adj R ^ k) s s :=
  exists_congr fun k => and_congr_right fun _ => (relPow_iff_adj_pow_pos R k s s).symm

end MatrixPower

end MatrixPowerSurvival
