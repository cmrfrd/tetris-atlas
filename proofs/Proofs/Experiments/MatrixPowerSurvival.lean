import Mathlib
import Proofs.Survival.Survival
import Proofs.Safety.SafeSet

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

/-- **Composition of walks**: `Bᵃ · Bᵇ = Bᵃ⁺ᵇ`. Concatenate a length-`a` walk with a length-`b`
walk — used to iterate a cycle to arbitrary length. -/
theorem RelPow.trans {R : State → State → Prop} :
    ∀ {a : ℕ} {s u : State}, RelPow R a s u →
      ∀ {b : ℕ} {t : State}, RelPow R b u t → RelPow R (a + b) s t := by
  intro a s u h
  induction h with
  | zero s => intro b t h2; simpa using h2
  | @succ n s' x _u' hsx _hxu ih =>
    intro b t h2
    rw [show n + 1 + b = (n + b) + 1 from by omega]
    exact RelPow.succ hsx (ih h2)

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

/-- **Sub-walk extraction.** Any contiguous stretch of a vertex sequence whose consecutive pairs are
`R`-edges (up to index `N`) is itself a walk: `RelPow R m (w c) (w (c+m))` for `c + m ≤ N`. The
converse of `exists_chain`, used to splice out loops. -/
theorem RelPow.subchain {R : State → State → Prop} {w : ℕ → State} {N : ℕ}
    (hstep : ∀ i, i < N → R (w i) (w (i + 1))) :
    ∀ c m, c + m ≤ N → RelPow R m (w c) (w (c + m)) := by
  intro c m
  induction m with
  | zero => intro _; simpa using RelPow.zero (w c)
  | succ k ih =>
    intro hcm
    rw [show c + (k + 1) = (c + k) + 1 from by omega]
    exact (ih (by omega)).snoc (hstep (c + k) (by omega))

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

/-! ### Perron–Frobenius / subshift converse: an infinite walk forces a reachable cycle

On a *finite* state space the headline reverses. Any infinite `R`-walk visits `card + 1` states in
its first `card + 1` steps, so by pigeonhole it repeats one — its prefix reaches that state and the
segment between the two visits is a positive-length cycle. Hence reachable cycle ⟺ infinite safe
path: the subshift of safe walks is nonempty iff a cycle is reachable iff `ρ(B) ≥ 1`. -/

/-- **Infinite walk ⇒ reachable cycle** (the finite-state Perron–Frobenius / subshift direction).
Pigeonhole on the first `card + 1` states of the path. Needs no `Dead` hypothesis — a cycle is
forced in any infinite walk on a finite graph. -/
theorem infiniteSafePath_imp_reachableCycle [Finite State] {R : State → State → Prop}
    {Dead : State → Prop} {s₀ : State} (h : InfiniteSafePath R Dead s₀) :
    ReachableCycle R s₀ := by
  letI := Fintype.ofFinite State
  obtain ⟨path, hp0, _hsafe, hpstep⟩ := h
  obtain ⟨i, j, hij, heq⟩ := Fintype.exists_ne_map_eq_of_card_lt
    (fun k : Fin (Fintype.card State + 1) => path (k : ℕ)) (by rw [Fintype.card_fin]; omega)
  have key : ∀ a b : ℕ, a < b → path a = path b → ReachableCycle R s₀ := by
    intro a b hab hpab
    refine ⟨path a, ?_, b - a, by omega, ?_⟩
    · have hw := RelPow.subchain (N := a) (fun k _ => hpstep k) 0 a (by omega)
      rw [Nat.zero_add, hp0] at hw
      exact hw.reachable
    · have hw := RelPow.subchain (N := b) (fun k _ => hpstep k) a (b - a) (by omega)
      rw [show a + (b - a) = b from by omega, ← hpab] at hw
      exact hw
  have hne : (i : ℕ) ≠ (j : ℕ) := fun e => hij (Fin.ext e)
  rcases Nat.lt_or_ge (i : ℕ) (j : ℕ) with hlt | hge
  · exact key i j hlt heq
  · exact key j i (by omega) heq.symm

/-- **Reachable cycle ⟺ infinite safe play** on a finite state space — the `ρ(B) ≥ 1`
characterization. Forward is the headline; backward is `infiniteSafePath_imp_reachableCycle`. -/
theorem reachableCycle_iff_infiniteSafePath [Finite State] {R : State → State → Prop}
    {Dead : State → Prop} (hSafeEdges : ∀ s t, R s t → ¬ Dead s ∧ ¬ Dead t) {s₀ : State} :
    ReachableCycle R s₀ ↔ InfiniteSafePath R Dead s₀ :=
  ⟨reachable_cycle_implies_infinite_safe_path hSafeEdges, infiniteSafePath_imp_reachableCycle⟩

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

/-- **Domination: survival transfers up from a sub-dynamics.** If the full game `step` can realize
every move of a restricted dynamics `step'` (the player has *at least* as many options under `step`,
`∀ s a' r, ∃ a, step s a r = step' s a' r`), then any recurrent support under `step'` is a recurrent
support under `step`. So **a winning sub-strategy proves the full game winnable**, and the
sub-strategy may be as simple as you like (a deterministic policy, a structured discipline on a few
classes). In matrix terms `step'` simulating means `adjFor step' r ≤ adjFor step r` entrywise, and
`mulVec_mono` transfers the common sub-eigenvector upward — the witness-shrinking lever. -/
theorem safeRecurrentSupport_mono {Piece Action Action' : Type*} {Dead : State → Prop}
    {legalDraws : State → Finset Piece}
    {step : State → Action → Piece → State} {step' : State → Action' → Piece → State}
    (hsim : ∀ s a' r, ∃ a, step s a r = step' s a' r)
    {X : Set State} (h : SafeRecurrentSupport Dead legalDraws step' X) :
    SafeRecurrentSupport Dead legalDraws step X := by
  intro s hs
  obtain ⟨hsne, hsucc⟩ := h s hs
  refine ⟨hsne, fun r hr => ?_⟩
  obtain ⟨a', ha'⟩ := hsucc r hr
  obtain ⟨a, ha⟩ := hsim s a' r
  exact ⟨a, by rw [ha]; exact ha'⟩

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

/-- **A cycle exists iff some matrix power has positive trace.** `tr(adjⁿ) = Σᵢ (adjⁿ)ᵢᵢ` counts the
length-`n` closed walks, so `(∃ s, OnCycle R s) ↔ ∃ n>0, 0 < tr(adjⁿ)` — the trace / dynamical-zeta
view of cyclicity (`tr(adjⁿ) = Σ λᵢⁿ`, the power-sum of the eigenvalues). -/
theorem exists_cycle_iff_trace_pow_pos :
    (∃ s, OnCycle R s) ↔ ∃ n, 0 < n ∧ 0 < (adj R ^ n).trace := by
  have htr : ∀ n, (adj R ^ n).trace = ∑ i, (adj R ^ n) i i := fun _ => rfl
  constructor
  · rintro ⟨s, k, hk, hcyc⟩
    refine ⟨k, hk, ?_⟩
    rw [htr]
    exact Finset.sum_pos' (fun i _ => Nat.zero_le _)
      ⟨s, Finset.mem_univ s, (relPow_iff_adj_pow_pos R k s s).mpr hcyc⟩
  · rintro ⟨n, hn, htrpos⟩
    rw [htr] at htrpos
    obtain ⟨i, _, hi⟩ := Finset.exists_ne_zero_of_sum_ne_zero (Nat.pos_iff_ne_zero.mp htrpos)
    exact ⟨i, n, hn, (relPow_iff_adj_pow_pos R n i i).mp (Nat.pos_of_ne_zero hi)⟩

/-- **Nilpotence ⇒ no survival (the matrix-power no-go criterion).** If some power of the adjacency
matrix vanishes (`adjᴺ = 0` — no walk of length `N`, i.e. the safe graph is a DAG with all plays
dying within `N-1` steps), then there is no infinite safe play from any start. An infinite path
would supply a length-`N` walk `RelPow R N s₀ (path N)`, contradicting `adjᴺ = 0`. The exact dual of
the cycle/diagonal criterion: nilpotent ⟺ no cycle ⟺ no survival. -/
theorem not_infiniteSafePath_of_pow_eq_zero {N : ℕ} (hN : adj R ^ N = 0)
    {Dead : State → Prop} {s₀ : State} : ¬ InfiniteSafePath R Dead s₀ := by
  rintro ⟨path, hp0, _, hstep⟩
  have hwalk : RelPow R N s₀ (path N) := by
    have h := RelPow.subchain (w := path) (N := N) (fun k _ => hstep k) 0 N (by omega)
    rw [Nat.zero_add, hp0] at h
    exact h
  have hpos := (relPow_iff_adj_pow_pos R N s₀ (path N)).mpr hwalk
  rw [hN] at hpos
  simp at hpos

/-- **A cycle makes nilpotence impossible (the contradiction).** If any state lies on a cycle then
*no* power of `adj` vanishes: iterate the cycle to length `k·N ≥ N`, a positive diagonal entry
`(adj^{k·N})_{s,s} > 0`, whereas `adjᴺ = 0` would force `adj^{k·N} = 0`. So exhibiting **one cycle
refutes nilpotence** — equivalently refutes "no survival." This is the lever for proving survival by
contradiction: assume `adjᴺ = 0`, produce a cycle, contradiction. -/
theorem onCycle_imp_pow_ne_zero {s : State} (h : OnCycle R s) (N : ℕ) : adj R ^ N ≠ 0 := by
  intro hN
  obtain ⟨k, hk, hcyc⟩ := h
  have hiter : ∀ m, RelPow R (k * m) s s := by
    intro m
    induction m with
    | zero => simpa using RelPow.zero s
    | succ j ih => rw [Nat.mul_succ]; exact ih.trans hcyc
  have hle : N ≤ k * N := le_mul_of_one_le_left (Nat.zero_le N) hk
  have hpos := (relPow_iff_adj_pow_pos R (k * N) s s).mpr (hiter N)
  rw [pow_eq_zero_of_le hle hN] at hpos
  simp at hpos

/-- **Recurrent support = Boolean Perron sub-eigenvector (Collatz–Wielandt).** A finite set `X` is
closed under `R` (every state of `X` has an in-`X` successor) iff its indicator `1_X` is a Boolean
*sub-eigenvector* of the adjacency matrix — for each `i ∈ X` the row action `(adj ⬝ᵥ 1_X) i =
∑ⱼ adjᵢⱼ·(1_X)ⱼ` is `≥ 1 = (1_X) i`, i.e. `adj ⬝ᵥ 1_X ≥ 1_X`. So a `SafeRecurrentSupport` is exactly
a nonzero `x ≥ 0` with `Bx ≥ x`: the Collatz–Wielandt witness certifying `ρ(B) ≥ 1`. -/
theorem recurrentSupport_iff_subEigenvector (X : Finset State) :
    (∀ i ∈ X, ∃ j ∈ X, R i j) ↔
      ∀ i ∈ X, 1 ≤ ∑ j, adj R i j * (if j ∈ X then (1 : ℕ) else 0) := by
  constructor
  · intro hclosed i hi
    obtain ⟨j, hjX, hij⟩ := hclosed i hi
    refine Finset.sum_pos' (fun _ _ => Nat.zero_le _) ⟨j, Finset.mem_univ j, ?_⟩
    rw [if_pos hjX, mul_one]
    exact (adj_pos_iff R i j).mpr hij
  · intro hsub i hi
    obtain ⟨j, _, hj⟩ := Finset.exists_ne_zero_of_sum_ne_zero
      (Nat.one_le_iff_ne_zero.mp (hsub i hi))
    rw [mul_ne_zero_iff] at hj
    exact ⟨j, by by_contra hjX; exact hj.2 (if_neg hjX),
              by by_contra hRij; exact hj.1 (if_neg hRij)⟩

end MatrixPower

/-! ### Cycles via Mathlib's transitive closure

`OnCycle R s` (`∃ k>0, Bᵏ` has a nonzero `(s,s)` entry) is exactly "`s` is `R`-related to itself in
the **transitive closure** `⋁_{k≥1} Bᵏ = Relation.TransGen R`". This ties the matrix-diagonal view
to Mathlib's closure API, and with `RelPow.trans` gives the composition algebra `Bᵃ · Bᵇ = Bᵃ⁺ᵇ`. -/

/-- Peel the first edge of a positive-length walk (`Bⁿ⁺¹ = B · Bⁿ`). -/
theorem RelPow.inv_succ {R : State → State → Prop} {n : ℕ} {s t : State}
    (h : RelPow R (n + 1) s t) : ∃ u, R s u ∧ RelPow R n u t := by
  cases h with
  | succ hsu htail => exact ⟨_, hsu, htail⟩

/-- The transitive closure is the union of positive relation powers: `TransGen R = ⋁_{n≥1} Bⁿ`. -/
theorem transGen_iff_exists_relPow_succ {R : State → State → Prop} {s t : State} :
    Relation.TransGen R s t ↔ ∃ n, RelPow R (n + 1) s t := by
  constructor
  · intro h
    induction h with
    | single hst => exact ⟨0, RelPow.succ hst (RelPow.zero _)⟩
    | tail _ hbc ih => obtain ⟨n, hn⟩ := ih; exact ⟨n + 1, hn.snoc hbc⟩
  · rintro ⟨n, hn⟩
    induction n generalizing s with
    | zero =>
      obtain ⟨u, hsu, htail⟩ := hn.inv_succ
      cases htail
      exact Relation.TransGen.single hsu
    | succ k ih =>
      obtain ⟨u, hsu, htail⟩ := hn.inv_succ
      exact Relation.TransGen.head hsu (ih htail)

/-- **A cycle is a self-loop in the transitive closure.** `OnCycle R s ↔ Relation.TransGen R s s` —
the closure-API counterpart of the matrix-diagonal `onCycle_iff_exists_pos_diag`. -/
theorem onCycle_iff_transGen {R : State → State → Prop} {s : State} :
    OnCycle R s ↔ Relation.TransGen R s s := by
  rw [transGen_iff_exists_relPow_succ]
  constructor
  · rintro ⟨k, hk, hcyc⟩
    obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
    exact ⟨m, hcyc⟩
  · rintro ⟨n, hn⟩
    exact ⟨n + 1, Nat.succ_pos n, hn⟩

/-! ### Finite stabilization — the matrix powers saturate at `N = card`

The deepest classical matrix-power fact: on a finite state space the powers `Bⁿ` stabilize. Any
reachable pair is reachable by a walk of length `< card`, because a longer walk repeats a vertex
(pigeonhole) and the loop splices out (`subchain` + `RelPow.trans`). Equivalently the transitive
closure is the *finite* union `⋁_{k < N} Bᵏ` — computing `B, B², …, B^{N-1}` already saturates
reachability, making it decidable. (This is what makes the abstract `safe` set a *finite* fixpoint;
it is also the honest ceiling — saturation needs all `N` powers, and for Tetris `N` is astronomical,
so the matrix view reorganizes the search without shrinking it.) -/

/-- **Reachability saturates below `card`.** If `t` is reachable from `s`, it is reachable by a walk
of length strictly less than `Fintype.card State` — a shortest such walk visits no vertex twice, so
spans at most `card` distinct vertices. Proof: take the shortest walk (`Nat.find`); were it
`≥ card`, pigeonhole forces a repeated vertex and splicing the loop yields a shorter walk. -/
theorem exists_relPow_lt_card [Fintype State] {R : State → State → Prop} {s t : State}
    (h : Reachable R s t) : ∃ n, n < Fintype.card State ∧ RelPow R n s t := by
  classical
  have hex : ∃ n, RelPow R n s t := reachable_iff_exists_relPow.mp h
  refine ⟨Nat.find hex, ?_, Nat.find_spec hex⟩
  by_contra hge
  rw [not_lt] at hge
  obtain ⟨w, hw0, hwn, hstep⟩ := (Nat.find_spec hex).exists_chain
  set n₀ := Nat.find hex with hn₀
  -- A length-`n₀ ≥ card` walk has `n₀ + 1 > card` vertices, hence a repeat (pigeonhole).
  obtain ⟨i, j, hij, heq⟩ := Fintype.exists_ne_map_eq_of_card_lt
    (fun k : Fin (n₀ + 1) => w (k : ℕ)) (by rw [Fintype.card_fin]; omega)
  -- Splicing out the loop between two equal vertices gives a strictly shorter walk to `t`.
  have key : ∀ a b : ℕ, a < b → b ≤ n₀ → w a = w b → False := by
    intro a b hab hbn hwab
    have hpre : RelPow R a s (w b) := by
      have hw := RelPow.subchain hstep 0 a (by omega)
      rw [Nat.zero_add, hw0, hwab] at hw
      exact hw
    have hsuf : RelPow R (n₀ - b) (w b) t := by
      have hw := RelPow.subchain hstep b (n₀ - b) (by omega)
      rw [show b + (n₀ - b) = n₀ from by omega, hwn] at hw
      exact hw
    exact Nat.find_min hex (show a + (n₀ - b) < n₀ from by omega) (hpre.trans hsuf)
  have hi : (i : ℕ) ≤ n₀ := by have := i.isLt; omega
  have hj : (j : ℕ) ≤ n₀ := by have := j.isLt; omega
  have hne : (i : ℕ) ≠ (j : ℕ) := fun e => hij (Fin.ext e)
  rcases Nat.lt_or_ge (i : ℕ) (j : ℕ) with hlt | hgt
  · exact key i j hlt hj heq
  · exact key j i (by omega) hi heq.symm

/-! ## §5 Encoding the ∀-over-the-bag: the per-piece matrix family

The single existential matrix `B = ⋁_{a,r} T_{a,r}` cannot carry the adversary: a walk in `B` lets
you choose the next *piece* as well as the placement, collapsing the faithful `∀ piece ∈ bag` into
`∃ piece`. The fix is to **not collapse** — keep the whole **family of per-piece matrices** `{A_r}`
and combine them by *conjunction over the bag*:

* `adjFor r` (`= A_r`): `(A_r)_{s,t} = 1 ⟺ ∃ placement, step s · r = t` — "if the adversary draws
  `r`, where can I go". The row action `(A_r ⬝ᵥ 1_X)_s ≥ 1` is exactly the `∃ placement` clause.
* The adversarial one-step operator is the **controllable predecessor**
  `s ↦ ¬Dead s ∧ ⋀_{r ∈ bag(s)} (A_r ⬝ᵥ 1_X)_s ≥ 1`: the bag's ∀ is the *conjunction* of the
  per-piece matrix actions, not a single power. (This is `safeOp` from the green library.)
* So a **recurrent support is a COMMON sub-eigenvector of the whole family** — `1_X ≤ A_r ⬝ᵥ 1_X`
  for every `r ∈ bag`, simultaneously. Existential survival needed one matrix (`Bx ≥ x`,
  Collatz–Wielandt); adversarial survival needs `x` to sub-fix *every* `A_r` at once. The governing
  quantity is the **joint spectral radius** of `{A_r}` (the worst case over all adversarial piece
  sequences `A_{rₙ}···A_{r₁}`), not a single `ρ` — the matrix face of the mean-payoff game. -/

/-! ### Reaching the gfp by finite iteration (the descending chain stabilizes)

A general fact about any monotone `f` on a complete lattice: iterating downward from `⊤`,
`gfpIter f n = f (f (… ⊤))`, gives a decreasing chain that lies above the gfp. On a *finite* lattice
the chain stabilizes, and the stable value **is** the gfp. Applied to `cpreHom`, this turns
"`adversarialSafe` is nonempty" — the existence of a cyclic sub-graph — into a *finite, detectable*
computation: iterate until `gfpIter (N+1) = gfpIter N`, and that set is the cyclic sub-graph. -/

section IterateGfp

variable {α : Type*} [CompleteLattice α] (f : α →o α)

/-- The downward iteration of `f` from `⊤`: `gfpIter f n = fⁿ ⊤`. -/
def gfpIter : ℕ → α
  | 0 => ⊤
  | n + 1 => f (gfpIter n)

/-- The gfp lies below every downward iterate. -/
theorem gfp_le_gfpIter : ∀ n, OrderHom.gfp f ≤ gfpIter f n := by
  intro n
  induction n with
  | zero => exact le_top
  | succ k ih => exact (OrderHom.map_gfp f).symm.le.trans (f.mono ih)

/-- The downward iteration decreases. -/
theorem gfpIter_succ_le : ∀ n, gfpIter f (n + 1) ≤ gfpIter f n := by
  intro n
  induction n with
  | zero => exact le_top
  | succ k ih => exact f.mono ih

/-- **Stabilization ⇒ the gfp.** If the downward iteration stops changing at step `N`
(`gfpIter (N+1) = gfpIter N`), the stable value is exactly the greatest fixed point. -/
theorem gfpIter_eq_gfp_of_fixed {N : ℕ} (h : gfpIter f (N + 1) = gfpIter f N) :
    gfpIter f N = OrderHom.gfp f :=
  le_antisymm (OrderHom.le_gfp _ (le_of_eq h.symm)) (gfp_le_gfpIter f N)

/-- **On a finite lattice the iteration stabilizes.** The decreasing chain cannot strictly descend
forever, so some step is a fixed point (pigeonhole: the iterate `ℕ → α` into a finite `α` repeats,
and the chain between equal values is squeezed constant). -/
theorem exists_gfpIter_fixed [Finite α] : ∃ N, gfpIter f (N + 1) = gfpIter f N := by
  have hanti : Antitone (gfpIter f) := antitone_nat_of_succ_le (gfpIter_succ_le f)
  obtain ⟨i, j, hij, heq⟩ := Finite.exists_ne_map_eq_of_infinite (gfpIter f)
  rcases Nat.lt_or_ge i j with hlt | hge
  · refine ⟨i, le_antisymm (gfpIter_succ_le f i) ?_⟩
    calc gfpIter f i = gfpIter f j := heq
      _ ≤ gfpIter f (i + 1) := hanti hlt
  · have hji : j < i := lt_of_le_of_ne hge (Ne.symm hij)
    refine ⟨j, le_antisymm (gfpIter_succ_le f j) ?_⟩
    calc gfpIter f j = gfpIter f i := heq.symm
      _ ≤ gfpIter f (j + 1) := hanti hji

/-- **The gfp is reached by finite iteration** on a finite lattice. -/
theorem exists_gfpIter_eq_gfp [Finite α] : ∃ N, gfpIter f N = OrderHom.gfp f := by
  obtain ⟨N, hN⟩ := exists_gfpIter_fixed f
  exact ⟨N, gfpIter_eq_gfp_of_fixed f hN⟩

end IterateGfp

section ControllablePredecessor

variable {Piece Action : Type*}

/-- The **controllable predecessor** `CPre(X)`: states that are live and from which, whatever piece
the adversary draws from the bag, some placement stays in `X`. The `∀ r ∈ bag` is a *conjunction*
(the bag's universal quantifier); this is the adversarial one-step operator `safeOp`, abstractly. -/
def CPre (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (X : Set State) : Set State :=
  {s | ¬ Dead s ∧ ∀ r ∈ legalDraws s, ∃ a, step s a r ∈ X}

/-- `CPre` is monotone: more room downstream ⇒ at least as many states survive the round. -/
theorem cpre_mono (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) {X Y : Set State} (h : X ⊆ Y) :
    CPre Dead legalDraws step X ⊆ CPre Dead legalDraws step Y :=
  fun _ hs => ⟨hs.1, fun r hr => (hs.2 r hr).imp fun _ ha => h ha⟩

/-- `CPre` as a monotone self-map of the lattice `Set State`. -/
def cpreHom (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) : Set State →o Set State where
  toFun := CPre Dead legalDraws step
  monotone' _ _ h := cpre_mono Dead legalDraws step h

/-- The **adversarial safe set** as the greatest fixed point of `CPre` — the largest set on which
the player can keep surviving the round against the bag. The abstract analogue of `Tetris.safe`. -/
def adversarialSafe (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) : Set State :=
  OrderHom.gfp (cpreHom Dead legalDraws step)

/-- **A recurrent support is exactly a sub-fixpoint of `CPre`.** `X ⊆ CPre(X)` is definitionally the
`SafeRecurrentSupport` closure (every state of `X` is live and handles every drawable piece into
`X`). Chained with `recurrentSupport_iff_common_subEigenvector` below, this reads: recurrent support
⟺ `X ⊆ CPre(X)` ⟺ `1_X` is a common sub-eigenvector of the per-piece family `{A_r}`. -/
theorem subset_cpre_iff_recurrentSupport (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (X : Set State) :
    X ⊆ CPre Dead legalDraws step X ↔ SafeRecurrentSupport Dead legalDraws step X :=
  Iff.rfl

/-- **CPre membership is the meet over the bag of the per-piece existential edges.** `s ∈ CPre X`
unfolds to `¬Dead s ∧ ⋀_{r ∈ bag s} (∃ a, step s a r ∈ X)` — a *conjunction across pieces* of the
per-piece "some placement lands in `X`" predicate. The bag's `∀` is this meet (AND); the companion
`mem_cpre_iff_meet_perPiece` reads it as the Hadamard product of the per-piece matrices `A_r` on
`1_X`. -/
theorem mem_cpre_iff_meet (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (X : Set State) (s : State) :
    s ∈ CPre Dead legalDraws step X ↔
      ¬ Dead s ∧ ∀ r ∈ legalDraws s, ∃ a, step s a r ∈ X :=
  Iff.rfl

/-- **Coinduction: every recurrent support lies inside the adversarial safe set.** The abstract
`safe_greatest` for the conjunctive operator — the gfp is the *maximal* recurrent support / common
sub-eigenvector of the per-piece matrix family. -/
theorem subset_adversarialSafe_of_recurrentSupport (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) {X : Set State}
    (h : SafeRecurrentSupport Dead legalDraws step X) :
    X ⊆ adversarialSafe Dead legalDraws step :=
  OrderHom.le_gfp _ ((subset_cpre_iff_recurrentSupport Dead legalDraws step X).mpr h)

/-- **The gfp is itself a recurrent support — the maximal cyclic sub-graph.** `adversarialSafe` is a
fixed point of `CPre` (`OrderHom.map_gfp`), so it is closed, i.e. a `SafeRecurrentSupport`. Together
with the coinduction above, it is *the largest* one: every recurrent support is contained in it, and
it is one. This is the object whose nonemptiness *is* the existence of a cyclic sub-graph. -/
theorem adversarialSafe_isRecurrentSupport (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) :
    SafeRecurrentSupport Dead legalDraws step (adversarialSafe Dead legalDraws step) := by
  apply (subset_cpre_iff_recurrentSupport Dead legalDraws step _).mp
  exact le_of_eq (OrderHom.map_gfp (cpreHom Dead legalDraws step)).symm

/-- **Existence reduction.** A *nonempty* recurrent support (a cyclic sub-graph) exists iff the
canonical gfp `adversarialSafe` is nonempty. The whole existence question concentrates into the
nonemptiness of one fixpoint-defined object — the matrix-powers target for "a cyclic sub-graph
exists". -/
theorem exists_nonempty_recurrentSupport_iff (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) :
    (∃ X : Set State, X.Nonempty ∧ SafeRecurrentSupport Dead legalDraws step X) ↔
      (adversarialSafe Dead legalDraws step).Nonempty := by
  constructor
  · rintro ⟨X, ⟨x, hx⟩, hX⟩
    exact ⟨x, subset_adversarialSafe_of_recurrentSupport Dead legalDraws step hX hx⟩
  · intro h
    exact ⟨_, h, adversarialSafe_isRecurrentSupport Dead legalDraws step⟩

/-- **Entry version.** A state `s₀` lies in a recurrent support (a cyclic sub-graph it can never be
forced out of) iff `s₀ ∈ adversarialSafe`. Reduces "does the start enter a cyclic sub-graph" to one
membership in the canonical gfp. -/
theorem mem_recurrentSupport_iff (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (s₀ : State) :
    (∃ X : Set State, s₀ ∈ X ∧ SafeRecurrentSupport Dead legalDraws step X) ↔
      s₀ ∈ adversarialSafe Dead legalDraws step := by
  constructor
  · rintro ⟨X, hs₀, hX⟩
    exact subset_adversarialSafe_of_recurrentSupport Dead legalDraws step hX hs₀
  · intro h
    exact ⟨_, h, adversarialSafe_isRecurrentSupport Dead legalDraws step⟩

/-- **Search-termination certificate.** If the downward iteration of `CPre` stops changing at step
`N` (`gfpIter (N+1) = gfpIter N`), that finite set *is* `adversarialSafe` — the maximal cyclic
sub-graph. This is what a retrograde/death-propagation search detects: iterate until no change. -/
theorem gfpIter_cpre_eq_adversarialSafe_of_fixed (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) {N : ℕ}
    (h : gfpIter (cpreHom Dead legalDraws step) (N + 1)
      = gfpIter (cpreHom Dead legalDraws step) N) :
    gfpIter (cpreHom Dead legalDraws step) N = adversarialSafe Dead legalDraws step :=
  gfpIter_eq_gfp_of_fixed _ h

/-- **`adversarialSafe` is a finite iterate.** On a finite state space the downward `CPre`-iteration
reaches the gfp in finitely many steps: `∃ N, gfpIter CPre N = adversarialSafe`. So existence of a
cyclic sub-graph (`adversarialSafe` nonempty, via `mem_recurrentSupport_iff`) is a **finite,
computable check**: surviving `N` adversarial rounds from `⊤` settles surviving
forever. (`N` is bounded by the lattice height, astronomical for real Tetris — this
makes existence *finite/decidable*, not *cheap*.) -/
theorem exists_gfpIter_cpre_eq_adversarialSafe [Finite State] (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) :
    ∃ N, gfpIter (cpreHom Dead legalDraws step) N = adversarialSafe Dead legalDraws step :=
  exists_gfpIter_eq_gfp (cpreHom Dead legalDraws step)

/-- **Adversarial nilpotence at `s₀`** (the no-survival side, operator form): the downward
`CPre`-iteration eliminates `s₀` at some finite stage — the adversary forces `s₀` to death within
finitely many rounds (`s₀ ∉ CPreᴺ ⊤`). -/
def AdvNilpotent (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (s₀ : State) : Prop :=
  ∃ N, s₀ ∉ gfpIter (cpreHom Dead legalDraws step) N

/-- **The goal, posed exactly: adversarial non-nilpotence ⟺ survival.** On a finite state space,
`s₀` is *never* eliminated by any finite stage of the adversarial `CPre`-iteration iff
`s₀ ∈ adversarialSafe`. So "prove the adversarial operator non-nilpotent at `s₀`" **is** "prove `s₀`
survives forever." -/
theorem not_advNilpotent_iff [Finite State] (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (s₀ : State) :
    ¬ AdvNilpotent Dead legalDraws step s₀ ↔ s₀ ∈ adversarialSafe Dead legalDraws step := by
  unfold AdvNilpotent
  rw [not_exists]
  constructor
  · intro h
    obtain ⟨N, hN⟩ := exists_gfpIter_eq_gfp (cpreHom Dead legalDraws step)
    have hmem := h N
    rw [not_not] at hmem
    rw [hN] at hmem
    exact hmem
  · intro h N
    rw [not_not]
    exact gfp_le_gfpIter (cpreHom Dead legalDraws step) N h

/-! ### The whole problem as one matrix: fix the policy, the adversary becomes reachability

With the player's memoryless policy `π : State → Piece → Action` fixed, the adversary's *only*
freedom is which bag piece to draw, so the dynamics is a **single Boolean matrix** `M_π` (the policy
graph). The adversary forcing death = *some* draw-sequence reaching a dead state = **death reachable
in `M_π`** = a matrix-power statement (`∃ n, (M_πⁿ)_{init,⊥} > 0`). Survival under `π` is its
negation — `M_π` is *death-unreachable* (death-nilpotent) from `init`. The full problem then reads:

> `init` survives  ⟺  `∃ π`, the matrix `M_π` never reaches death from `init`.

The adversary's `∀`-over-infinite-sequences is fully linearized into reachability on the one matrix
`M_π`; the only genuinely non-linear ingredient that remains is the existential `∃ π` (the policy
search). -/

/-- The **policy graph** `M_π`: with memoryless policy `π` fixed, `s → t` iff some drawable piece
`r` sends `s` to `t` under the policy's placement `π s r`. A single Boolean transition relation. -/
def policyGraph (legalDraws : State → Finset Piece) (step : State → Action → Piece → State)
    (π : State → Piece → Action) : State → State → Prop :=
  fun s t => ∃ r ∈ legalDraws s, step s (π s r) r = t

/-- `π` **survives** from `s₀`: no dead state is reachable from `s₀` in the policy graph `M_π` —
`M_π` is death-unreachable (death-nilpotent) from `s₀`, a reachability/matrix-power property of one
Boolean matrix. -/
def survivesUnder (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (π : State → Piece → Action) (s₀ : State) : Prop :=
  ∀ t, Reachable (policyGraph legalDraws step π) s₀ t → ¬ Dead t

/-- **The whole adversarial problem as matrix powers.** `s₀ ∈ adversarialSafe` (survives forever
against the adversary) iff there is a memoryless policy `π` whose policy graph `M_π` never reaches
death from `s₀`. The adversary's `∀`-over-sequences is reachability in the single matrix `M_π`; the
sole non-linear ingredient is the `∃ π`. -/
theorem mem_adversarialSafe_iff_exists_surviving_policy [Nonempty Action] (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (s₀ : State) :
    s₀ ∈ adversarialSafe Dead legalDraws step ↔
      ∃ π : State → Piece → Action, survivesUnder Dead legalDraws step π s₀ := by
  classical
  have hrs := adversarialSafe_isRecurrentSupport Dead legalDraws step
  constructor
  · intro hs₀
    have hpick : ∀ s r, ∃ a, s ∈ adversarialSafe Dead legalDraws step → r ∈ legalDraws s →
        step s a r ∈ adversarialSafe Dead legalDraws step := by
      intro s r
      by_cases hs : s ∈ adversarialSafe Dead legalDraws step
      · by_cases hr : r ∈ legalDraws s
        · obtain ⟨a, ha⟩ := (hrs s hs).2 r hr
          exact ⟨a, fun _ _ => ha⟩
        · exact ⟨Classical.arbitrary Action, fun _ hr' => absurd hr' hr⟩
      · exact ⟨Classical.arbitrary Action, fun hs' _ => absurd hs' hs⟩
    refine ⟨fun s r => Classical.choose (hpick s r), fun t ht => ?_⟩
    have hmem : t ∈ adversarialSafe Dead legalDraws step := by
      induction ht with
      | refl => exact hs₀
      | @tail u v _ hstep ih =>
        obtain ⟨r, hr, hv⟩ := hstep
        have := Classical.choose_spec (hpick u r) ih hr
        rwa [hv] at this
    exact (hrs t hmem).1
  · rintro ⟨π, hπ⟩
    refine subset_adversarialSafe_of_recurrentSupport Dead legalDraws step
      (X := {t | Reachable (policyGraph legalDraws step π) s₀ t}) ?_ Relation.ReflTransGen.refl
    intro s hs
    exact ⟨hπ s hs, fun r hr => ⟨π s r, hs.tail ⟨r, hr, rfl⟩⟩⟩

end ControllablePredecessor

section AdversarialMatrix

variable {Piece Action : Type*} [DecidableEq State] [Fintype Action]

/-- Per-piece adjacency matrix `A_r`: `(A_r)_{s,t} = 1` iff some placement of piece `r` carries `s`
to `t`. The "the adversary drew `r`" transition, kept as its *own* Boolean matrix rather than merged
into `B`. -/
def adjFor (step : State → Action → Piece → State) (r : Piece) : Matrix State State ℕ :=
  fun s t => if ∃ a, step s a r = t then 1 else 0

theorem adjFor_apply (step : State → Action → Piece → State) (r : Piece) (s t : State) :
    adjFor step r s t = if ∃ a, step s a r = t then 1 else 0 := rfl

variable [Fintype State]

/-- **The per-piece matrix encodes the `∃ placement` clause.** The row action `(A_r ⬝ᵥ 1_X)_s ≥ 1`
holds iff some placement of `r` from `s` lands in `X`. -/
theorem adjFor_reaches_iff (step : State → Action → Piece → State) (r : Piece)
    (X : Finset State) (s : State) :
    1 ≤ ∑ t, adjFor step r s t * (if t ∈ X then (1 : ℕ) else 0) ↔ ∃ a, step s a r ∈ X := by
  constructor
  · intro h
    obtain ⟨t, _, ht⟩ := Finset.exists_ne_zero_of_sum_ne_zero (Nat.one_le_iff_ne_zero.mp h)
    rw [mul_ne_zero_iff] at ht
    obtain ⟨a, ha⟩ : ∃ a, step s a r = t := by
      by_contra hne; exact ht.1 (by rw [adjFor_apply]; exact if_neg hne)
    have htX : t ∈ X := by by_contra h'; exact ht.2 (if_neg h')
    exact ⟨a, by rw [ha]; exact htX⟩
  · rintro ⟨a, ha⟩
    refine Finset.sum_pos' (fun _ _ => Nat.zero_le _) ⟨step s a r, Finset.mem_univ _, ?_⟩
    refine Nat.mul_pos ?_ ?_
    · rw [adjFor_apply, if_pos ⟨a, rfl⟩]; exact Nat.one_pos
    · rw [if_pos ha]; exact Nat.one_pos

/-- **Recurrent support = common sub-eigenvector of the per-piece family (the ∀-over-the-bag
encoding).** A finite set `X` is closed under the adversarial `∀ r ∈ bag, ∃ placement` iff its
indicator `1_X` is a sub-eigenvector of *every* per-piece matrix `A_r` over the bag at once:
`1 ≤ (A_r ⬝ᵥ 1_X)_s` for all `s ∈ X`, `r ∈ bag(s)`. A single existential `B` cannot express this;
the conjunction over `r` is the matrix face of the bag's universal quantifier. -/
theorem recurrentSupport_iff_common_subEigenvector
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (X : Finset State) :
    (∀ s ∈ X, ∀ r ∈ legalDraws s, ∃ a, step s a r ∈ X) ↔
      ∀ s ∈ X, ∀ r ∈ legalDraws s,
        1 ≤ ∑ t, adjFor step r s t * (if t ∈ X then (1 : ℕ) else 0) := by
  constructor
  · intro h s hs r hr; exact (adjFor_reaches_iff step r X s).mpr (h s hs r hr)
  · intro h s hs r hr; exact (adjFor_reaches_iff step r X s).mp (h s hs r hr)

/-- **`CPre` is the Hadamard meet `⋀_r A_r` of the per-piece matrices.** For a finite target `X`,
`s ∈ CPre X ↔ ¬Dead s ∧ ∀ r ∈ bag s, 1 ≤ (A_r ⬝ᵥ 1_X)_s` — membership is `¬Dead` together with the
*pointwise AND over the bag* of the per-piece row actions `(A_r ⬝ᵥ 1_X)_s ≥ 1`. This is the exact
matrix form of the bag's `∀`: a meet of the per-piece Boolean matrix-vector products, never the
action of one matrix (see §`BarrierNonlinear`). -/
theorem mem_cpre_iff_meet_perPiece (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (X : Finset State) (s : State) :
    s ∈ CPre Dead legalDraws step (↑X) ↔
      ¬ Dead s ∧ ∀ r ∈ legalDraws s,
        1 ≤ ∑ t, adjFor step r s t * (if t ∈ X then (1 : ℕ) else 0) := by
  rw [mem_cpre_iff_meet]
  simp only [Finset.mem_coe, adjFor_reaches_iff]

/-! ### Products of per-piece matrices — the joint-spectral-radius view

The adversary draws a *sequence* of pieces, so the relevant object is **not** a power `Bⁿ` but a
*product* `A_{rₙ}···A_{r₁}` — a different product for each adversarial sequence. Applying that
product to a vector is just applying the per-piece matrices one after another (a `foldr`), which is
what we formalize (forming the literal product matrix `Matrix.mul` over the huge `State` is
unnecessary and elaboration-heavy). The worst-case growth rate over all such products is the **joint
spectral radius** of `{A_r}`; the fact below: a common sub-eigenvector survives *every* one. -/

omit [DecidableEq State] [Fintype Action] in
/-- `mulVec` by a nonnegative (here ℕ) matrix is monotone in the vector. -/
theorem mulVec_mono (M : Matrix State State ℕ) {y z : State → ℕ} (h : y ≤ z) :
    M.mulVec y ≤ M.mulVec z := by
  intro s
  change ∑ t, M s t * y t ≤ ∑ t, M s t * z t
  exact Finset.sum_le_sum fun t _ => Nat.mul_le_mul (le_refl (M s t)) (h t)

/-- **A common sub-eigenvector survives every adversarial product.** If `x ≤ A_r ⬝ᵥ x` for every
piece `r` in a list `l` (a common sub-eigenvector of that sub-family), then applying the whole
product to `x` — `foldr (A_r ⬝ᵥ ·)` down the list — never pushes `x` below itself:
`x ≤ A_{r₁} ⬝ᵥ (A_{r₂} ⬝ᵥ (··· ⬝ᵥ x))`, in any order. This is the witness that the family's **joint
spectral radius is `≥ 1`** on `x`'s support — survival against *all* piece sequences at once, not a
single power `Bⁿ`. -/
theorem le_foldr_mulVec (step : State → Action → Piece → State) {x : State → ℕ} :
    ∀ l : List Piece, (∀ r ∈ l, x ≤ (adjFor step r).mulVec x) →
      x ≤ l.foldr (fun r (v : State → ℕ) => (adjFor step r).mulVec v) x := by
  intro l
  induction l with
  | nil => intro _; exact le_refl x
  | cons r rest ih =>
    intro h
    have hrest := ih fun r' hr' => h r' (List.mem_cons_of_mem r hr')
    exact le_trans (h r (by simp)) (mulVec_mono _ hrest)

/-! #### The reverse direction: a sub-eigenvector's support is recurrent

The product theorem above is the *forward* half (a common sub-eigenvector witnesses JSR `≥ 1`). The
reverse closes the loop: the **positive support of any common sub-eigenvector is a recurrent
support**. So "a common sub-eigenvector exists" ⟺ "a recurrent support exists" ⟺ JSR `≥ 1`, the full
matrix characterization of survival. -/

omit [DecidableEq State] [Fintype Action] in
/-- A positive coordinate of a sub-eigenvector has a positive matrix-successor: if `0 < x s` and
`x s ≤ (M ⬝ᵥ x) s`, some `t` has `0 < M s t` and `0 < x t`. -/
theorem pos_succ_of_subEig (M : Matrix State State ℕ) {x : State → ℕ} {s : State}
    (h : x s ≤ M.mulVec x s) (hs : 0 < x s) : ∃ t, 0 < M s t ∧ 0 < x t := by
  have hpos : 0 < ∑ t, M s t * x t := lt_of_lt_of_le hs h
  obtain ⟨t, _, ht⟩ := Finset.exists_ne_zero_of_sum_ne_zero (Nat.pos_iff_ne_zero.mp hpos)
  rw [mul_ne_zero_iff] at ht
  exact ⟨t, Nat.pos_of_ne_zero ht.1, Nat.pos_of_ne_zero ht.2⟩

/-- **The support of a common sub-eigenvector is recurrent (reverse direction).** If `x` sub-fixes
`A_r` (`x s ≤ (A_r ⬝ᵥ x) s`) at a support state (`0 < x s`), then some placement of `r` lands in the
support. Family-wise (over `r ∈ bag`) `{s | 0 < x s}` is a recurrent support, so any JSR `≥ 1`
witness `x` yields a genuine survival set. -/
theorem exists_placement_of_subEig (step : State → Action → Piece → State) (r : Piece)
    {x : State → ℕ} {s : State} (h : x s ≤ (adjFor step r).mulVec x s) (hs : 0 < x s) :
    ∃ a, 0 < x (step s a r) := by
  obtain ⟨t, hM, hxt⟩ := pos_succ_of_subEig (adjFor step r) h hs
  rw [adjFor_apply] at hM
  obtain ⟨a, ha⟩ : ∃ a, step s a r = t := by
    by_contra hne; rw [if_neg hne] at hM; exact absurd hM (lt_irrefl 0)
  exact ⟨a, ha ▸ hxt⟩

end AdversarialMatrix

/-! ## §5c Barrier — the bag-meet `CPre` is not a single Boolean matrix

These sections recast survival in matrix vocabulary. This section makes the **negative**
fact precise and final: the bag's `∀ piece` operator `CPre` is a *meet* of the per-piece matrices
(`mem_cpre_iff_meet_perPiece`), and a meet is **not** the action of any single Boolean matrix. So no
amount of single-matrix spectral theory (powers, eigenvalues, nilpotence of one `B`) can compute
adversarial survival — the right object is irreducibly the family `{A_r}` under conjunction, whose
governing quantity is a *joint* spectral radius, not a single `ρ`. We prove two self-contained
barriers over tiny explicit instances.

* **Non-linearity (`cpre_not_matrixInduced`).** Every single-Boolean-matrix operator preserves
  binary unions (`MatrixInduced.union` — Boolean mat-vec distributes over `∨`). `CPre` does not: a
  3-state, 2-piece routing instance puts a state in `CPre(X ∪ Y)` but in neither `CPre X` nor
  `CPre Y` (route piece `r₁` into `X` and `r₂` into `Y` — impossible for one piece alone). Hence
  `CPre` is induced by **no** matrix.
* **Cooperative ≠ adversarial (`coop_nonNilpotent_but_not_safe`).** The single existential matrix
  `B = ⋁_r A_r` can be non-nilpotent (a cooperative cycle `0 → 0` under the safe piece, so `Bᴺ ≠ 0`
  for every `N`) while the start is **not** adversarially safe (the adversary draws the deadly piece
  and tops out). So `B`'s cyclicity — the one-matrix criterion traditional linear algebra reads off
  `B` alone — is no survival certificate.

Together: linearise (drop the `∀`) and you get a true-but-irrelevant statement about `B`; keep the
`∀` and the operator leaves single-matrix linear algebra entirely. This is *why* the matrix-power
route stalls here, stated as theorems rather than a claim. -/

section BarrierNonlinear

/-- An operator `F : Set State → Set State` is **matrix-induced** (Boolean-linear) when it is the
Boolean matrix-vector action of some relation `M`: `F X = {s | ∃ t, M s t ∧ t ∈ X}` (equivalently
`s ∈ F X ⟺ (M ⬝ᵥ 1_X)_s ≥ 1`). This is the most general single-Boolean-matrix operator on sets. -/
def MatrixInduced (F : Set State → Set State) : Prop :=
  ∃ M : State → State → Prop, ∀ X : Set State, F X = {s | ∃ t, M s t ∧ t ∈ X}

/-- **Every matrix-induced operator preserves binary unions.** Boolean mat-vec distributes over `∪`
(`∃ t, M s t ∧ (t ∈ X ∨ t ∈ Y)` splits over the disjunction), so `F (X ∪ Y) = F X ∪ F Y`. This
additivity is the defining linearity a single Boolean matrix must have. -/
theorem MatrixInduced.union {F : Set State → Set State} (hF : MatrixInduced F) (X Y : Set State) :
    F (X ∪ Y) = F X ∪ F Y := by
  obtain ⟨M, hM⟩ := hF
  rw [hM (X ∪ Y), hM X, hM Y]
  ext s
  simp only [Set.mem_setOf_eq, Set.mem_union]
  constructor
  · rintro ⟨t, hMt, ht | ht⟩
    · exact Or.inl ⟨t, hMt, ht⟩
    · exact Or.inr ⟨t, hMt, ht⟩
  · rintro (⟨t, hMt, ht⟩ | ⟨t, hMt, ht⟩)
    · exact ⟨t, hMt, Or.inl ht⟩
    · exact ⟨t, hMt, Or.inr ht⟩

end BarrierNonlinear

section BarrierWitnessNonlinear

/-- Routing-instance states: `0` routes, `1` is the `X`-home, `2` the `Y`-home. -/
abbrev RState := Fin 3
/-- Two pieces: `r₁ = 0` routes into `X`, `r₂ = 1` routes into `Y`. -/
abbrev RPiece := Fin 2

/-- One forced placement per `(state, piece)`: from `0`, piece `0 ↦ 1` (the `X`-home) and piece
`1 ↦ 2` (the `Y`-home); the homes stay put. -/
def rstep : RState → Unit → RPiece → RState := fun s _ r =>
  if s = 0 then (if r = 0 then 1 else 2) else s

/-- Both pieces are always drawable. -/
def rlegal : RState → Finset RPiece := fun _ => Finset.univ
/-- Nobody dies in the routing instance. -/
def rdead : RState → Prop := fun _ => False

/-- The router survives the **union**: piece `r₁` lands in `{1}`, piece `r₂` in `{2}`. -/
theorem router_mem_union :
    (0 : RState) ∈ CPre rdead rlegal rstep (({1} : Set RState) ∪ {2}) := by
  refine ⟨not_false, fun r _ => ⟨(), ?_⟩⟩
  fin_cases r <;> simp only [Set.mem_union, Set.mem_singleton_iff] <;> decide

/-- …but not `CPre {1}`: piece `r₂` has no placement into `{1}`. -/
theorem router_not_mem_X :
    (0 : RState) ∉ CPre rdead rlegal rstep ({1} : Set RState) := by
  rintro ⟨-, h⟩
  obtain ⟨a, ha⟩ := h 1 (Finset.mem_univ 1)
  rw [Set.mem_singleton_iff] at ha
  revert ha; cases a; decide

/-- …and not `CPre {2}`: piece `r₁` has no placement into `{2}`. -/
theorem router_not_mem_Y :
    (0 : RState) ∉ CPre rdead rlegal rstep ({2} : Set RState) := by
  rintro ⟨-, h⟩
  obtain ⟨a, ha⟩ := h 0 (Finset.mem_univ 0)
  rw [Set.mem_singleton_iff] at ha
  revert ha; cases a; decide

/-- **`CPre` fails union-preservation** — the router is in `CPre(X ∪ Y)` but in neither piece's set,
because routing different pieces to different homes is invisible to any single (additive) matrix. -/
theorem cpre_not_union_preserving :
    CPre rdead rlegal rstep (({1} : Set RState) ∪ {2}) ≠
      CPre rdead rlegal rstep ({1} : Set RState)
        ∪ CPre rdead rlegal rstep ({2} : Set RState) := by
  intro heq
  have h0 := heq ▸ router_mem_union
  rcases h0 with h | h
  · exact router_not_mem_X h
  · exact router_not_mem_Y h

/-- **Barrier 1: `CPre` is induced by no single Boolean matrix.** If it were, it would preserve
unions (`MatrixInduced.union`), contradicting `cpre_not_union_preserving`. The bag's `∀`-meet is
genuinely outside single-matrix linear algebra. -/
theorem cpre_not_matrixInduced : ¬ MatrixInduced (CPre rdead rlegal rstep) :=
  fun hMI => cpre_not_union_preserving (hMI.union _ _)

end BarrierWitnessNonlinear

section BarrierWitnessCooperative

/-- Cooperative-instance states: `0` is live (and self-loops on the safe piece), `1` is dead. -/
abbrev DState := Fin 2
/-- Two pieces: `0` is safe (self-loops), `1` is deadly (forces the dead state). -/
abbrev DPiece := Fin 2

/-- From the live state `0`: the safe piece `0` self-loops (cooperative cycle), the deadly piece `1`
forces death `1`; the dead state absorbs. -/
def dstep : DState → Unit → DPiece → DState := fun s _ r =>
  if s = 0 then (if r = 0 then 0 else 1) else 1

/-- Both pieces always drawable. -/
def dlegal : DState → Finset DPiece := fun _ => Finset.univ
/-- State `1` is the dead / top-out state. -/
def ddead : DState → Prop := fun s => s = 1

/-- A classical decidability instance so the existential graph `B = EdgeExists` has an honest
adjacency matrix `adj B` (base-axiom clean: only `Classical.choice`, never `native_decide`). -/
noncomputable instance : DecidableRel (EdgeExists dlegal dstep) :=
  fun s t => Classical.propDecidable (EdgeExists dlegal dstep s t)

/-- The existential graph `B = ⋁_r A_r` has a self-loop at `0` (the safe piece returns to `0`). -/
theorem coop_cycle : OnCycle (EdgeExists dlegal dstep) 0 :=
  ⟨1, Nat.one_pos, RelPow.succ ⟨0, Finset.mem_univ 0, (), by decide⟩ (RelPow.zero 0)⟩

/-- Hence **`B` is non-nilpotent**: every power is nonzero (`Bᴺ ≠ 0` for all `N`). This is the
single-matrix cyclicity criterion `ρ(B) ≥ 1` that traditional linear algebra reads off `B` alone. -/
theorem coop_B_non_nilpotent (N : ℕ) : adj (EdgeExists dlegal dstep) ^ N ≠ 0 :=
  onCycle_imp_pow_ne_zero (EdgeExists dlegal dstep) coop_cycle N

/-- **Yet `0 ∉ adversarialSafe`**: any recurrent support containing `0` must, for the deadly piece,
land in itself — but the only successor is the dead state `1`, which no recurrent support contains.
So the adversary forces death; the cooperative cycle is no survival certificate. -/
theorem coop_not_adversarialSafe : (0 : DState) ∉ adversarialSafe ddead dlegal dstep := by
  intro h0
  have hrs := adversarialSafe_isRecurrentSupport ddead dlegal dstep
  obtain ⟨-, hsucc⟩ := hrs 0 h0
  obtain ⟨a, ha⟩ := hsucc 1 (Finset.mem_univ 1)
  have h1 : (1 : DState) ∈ adversarialSafe ddead dlegal dstep := by
    have he : dstep 0 a 1 = 1 := by cases a; decide
    rwa [he] at ha
  exact (hrs 1 h1).1 rfl

/-- **Barrier 2: `B` non-nilpotent ⇏ adversarial survival.** A cooperative cycle in the single
existential matrix `B` (so `Bᴺ ≠ 0` for every `N`) coexists with the start being doomed against the
adversary. Single-matrix non-nilpotence is detectable but *irrelevant* to the `∀`-piece safety the
meet `CPre` encodes — the precise gap between `∃`-piece reachability and `∀`-piece survival. -/
theorem coop_nonNilpotent_but_not_safe :
    (∀ N, adj (EdgeExists dlegal dstep) ^ N ≠ 0) ∧
      (0 : DState) ∉ adversarialSafe ddead dlegal dstep :=
  ⟨coop_B_non_nilpotent, coop_not_adversarialSafe⟩

end BarrierWitnessCooperative

/-! ## §5d The Koopman lift — make it fully linear by enlarging the space

Barrier 1 says the `∀`-meet `CPre` is not a single matrix *on `Set State`*. But it becomes a genuine
**linear operator** if we enlarge the space: lift to the (`2^|State|`-dimensional) space of
observables `Set State → ℚ` — one coordinate per subset, "all candidate safe-sets at once" — with
the operator being *precomposition by* `CPre`. This is the Koopman / transfer-operator
linearization, the "trade nonlinearity for dimension" move. It genuinely linearizes everything
(`koopman`
is a bona-fide `ℚ`-linear map, Mathlib's `LinearMap.funLeft`), and survival becomes a single
**coordinate of a linear orbit** (`koopman_orbit_survives_iff`).

HONEST CEILING (why this is a repackaging, not a shortcut): the lifted space is doubly exponential
(`2^|State|`, and `|State|` already astronomical), and `koopman` is *precomposition* by `CPre`, so
its powers merely re-run the iteration `CPreⁿ` (`koopmanFun_iterate`) — there is no eigenvalue that
reads off survival more cheaply than iterating `CPre`. The linearity is a relabeling; the difficulty
moves into the dimension. -/

section KoopmanLift

variable {Piece Action : Type*}

/-- Raw action of the Koopman lift on the observable space `Set State → ℚ`: precomposition by the
(nonlinear) controllable predecessor `CPre`. Each subset is a coordinate; this map advances *all* of
them one adversarial round at once. -/
def koopmanFun (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (φ : Set State → ℚ) : Set State → ℚ :=
  fun X => φ (CPre Dead legalDraws step X)

/-- **The lift is genuinely `ℚ`-linear.** `koopman` is Mathlib's `LinearMap.funLeft` (precomposition
by `CPre`), a bundled linear map — so "everything is linear in the bigger space" is certified by the
type, not merely asserted. Its underlying function is `koopmanFun` (`koopman_eq_koopmanFun`). -/
def koopman (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) : (Set State → ℚ) →ₗ[ℚ] (Set State → ℚ) :=
  LinearMap.funLeft ℚ ℚ (CPre Dead legalDraws step)

theorem koopman_eq_koopmanFun (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (φ : Set State → ℚ) :
    koopman Dead legalDraws step φ = koopmanFun Dead legalDraws step φ := by
  ext X; exact LinearMap.funLeft_apply ℚ ℚ (CPre Dead legalDraws step) φ X

/-- **Linear powers = the survival iteration.** The `n`-th iterate of the linear lift precomposes by
the `n`-fold `CPre`: `koopmanFunⁿ φ X = φ (CPreⁿ X)`. So the lifted linear dynamics carries exactly
`CPreⁿ` — the honest ceiling: matrix powers here *are* the original iteration, relabeled. -/
theorem koopmanFun_iterate (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (n : ℕ) (φ : Set State → ℚ) (X : Set State) :
    (koopmanFun Dead legalDraws step)^[n] φ X = φ ((CPre Dead legalDraws step)^[n] X) := by
  induction n generalizing φ with
  | zero => rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply, ih, Function.iterate_succ_apply']
    rfl

/-- The `CPre` iteration from the top set is exactly the greatest-fixpoint downward iteration
`gfpIter cpreHom` — the bridge from the lifted orbit to the existing gfp machinery. -/
theorem cpre_iterate_univ_eq_gfpIter (Dead : State → Prop) (legalDraws : State → Finset Piece)
    (step : State → Action → Piece → State) (n : ℕ) :
    (CPre Dead legalDraws step)^[n] Set.univ = gfpIter (cpreHom Dead legalDraws step) n := by
  induction n with
  | zero => simp [gfpIter, Set.top_eq_univ]
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih]
    rfl

/-- The survival observable: `1` on candidate sets containing `init`, `0` otherwise — the coordinate
functional that reads "is `init` still in the safe set". -/
noncomputable def survObs (init : State) : Set State → ℚ := fun X => Set.indicator X 1 init

theorem survObs_eq_one_iff (init : State) (S : Set State) : survObs init S = 1 ↔ init ∈ S := by
  unfold survObs
  by_cases h : init ∈ S
  · rw [Set.indicator_of_mem h]; simp [h]
  · rw [Set.indicator_of_notMem h]; simp [h]

/-- **Survival is a coordinate of the linear orbit.** On a finite state space, iterating the
*linear* operator `koopman` on the single observable `survObs init` from the top set stabilizes, its
value at `Set.univ` is `survObs init` on the true safe set `adversarialSafe`. Everything is
linear, in the enlarged space `Set State → ℚ`; `N` is the gfp stabilization stage. -/
theorem koopman_orbit_decides_survival [Finite State] (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (init : State) :
    ∃ N, (koopmanFun Dead legalDraws step)^[N] (survObs init) Set.univ
        = survObs init (adversarialSafe Dead legalDraws step) := by
  obtain ⟨N, hN⟩ := exists_gfpIter_eq_gfp (cpreHom Dead legalDraws step)
  refine ⟨N, ?_⟩
  rw [koopmanFun_iterate, cpre_iterate_univ_eq_gfpIter, hN]
  rfl

/-- **Read-off: the stabilized coordinate is `1` iff `init` survives forever.** Combining the orbit
value with `survObs_eq_one_iff`: at the stabilization stage `N`, the single coordinate
`koopmanᴺ (survObs init) (univ)` equals `1` exactly when `init ∈ adversarialSafe` — i.e. the linear
lift *decides* adversarial survival, at the honest cost of a `2^|State|`-dimensional space. -/
theorem koopman_orbit_survives_iff [Finite State] (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (init : State) :
    ∃ N, ((koopmanFun Dead legalDraws step)^[N] (survObs init) Set.univ = 1
            ↔ init ∈ adversarialSafe Dead legalDraws step) := by
  obtain ⟨N, hN⟩ := koopman_orbit_decides_survival Dead legalDraws step init
  exact ⟨N, by rw [hN, survObs_eq_one_iff]⟩

/-! ### The spectral reframe: solvability as a `1`-eigenvector of the transfer matrix

The transfer matrix `T` on the `2^|State|` subset-coordinates advances each candidate set one
adversarial round: `T e_X = e_{CPre X}`. So coordinate `X` carries a **`1` on the diagonal**
(`T e_X = e_X`, an eigenvalue-`1` eigenvector) exactly when `CPre X = X` — `X` is a **closed set**.
The whole problem becomes one spectral statement: Tetris is solvable iff `T` has an eigenvalue-`1`
eigenvector whose coordinate contains `init`. -/

/-- **A good `1` on the diagonal ⇒ solvable.** If `CPre X = X` (`e_X` is a `1`-eigenvector of the
transfer matrix — a `1` on the diagonal at coordinate `X`) and `init ∈ X`, then
`init ∈ adversarialSafe`: Tetris survives forever from `init`. Exhibit *one* diagonal `1` at an
`init`-containing coordinate and you are done (coinduction / `OrderHom.le_gfp`). -/
theorem mem_adversarialSafe_of_cpre_fixed (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) {X : Set State}
    (hfix : CPre Dead legalDraws step X = X) {init : State} (hinit : init ∈ X) :
    init ∈ adversarialSafe Dead legalDraws step :=
  subset_adversarialSafe_of_recurrentSupport Dead legalDraws step
    ((subset_cpre_iff_recurrentSupport Dead legalDraws step X).mp (le_of_eq hfix.symm)) hinit

/-- **The spectral reframe, complete: solvable ⟺ a `1`-eigenvector contains `init`.** `init`
survives forever iff the transfer matrix has an eigenvalue-`1` eigenvector `e_X` (`CPre X = X`, a
`1` on the diagonal at `X`) with `init ∈ X`. Forward: `adversarialSafe` is itself such a fixed point
(`OrderHom.map_gfp`) and contains `init`. So "prove Tetris solvable" is *exactly* "prove the
transfer matrix has a good `1` on the diagonal" — the whole problem as one spectral existence claim.
(Honest residual: that eigenvector's coordinate IS the carrier; the reframe is complete, the
existence is the open crux — spectral theory hands you eigenvalue-`1` eigenvectors for free, e.g.
`e_∅`, but not a nonempty `init`-containing one.) -/
theorem mem_adversarialSafe_iff_exists_cpre_fixed (Dead : State → Prop)
    (legalDraws : State → Finset Piece) (step : State → Action → Piece → State) (init : State) :
    init ∈ adversarialSafe Dead legalDraws step ↔
      ∃ X : Set State, CPre Dead legalDraws step X = X ∧ init ∈ X := by
  constructor
  · intro h
    exact ⟨adversarialSafe Dead legalDraws step,
      OrderHom.map_gfp (cpreHom Dead legalDraws step), h⟩
  · rintro ⟨X, hfix, hinit⟩
    exact mem_adversarialSafe_of_cpre_fixed Dead legalDraws step hfix hinit

end KoopmanLift

/-! ## §6 Connecting the abstract layer to the concrete safe-set results

The abstract theorems above are not new survival *content* — they re-present, in matrix-power
vocabulary, certificates the green library already proves over `GameState`. We make that precise for
the single-player closed cycle (`Tetris.ClosedCycle`, the M2 artifact): its state set, under the
deterministic policy-step relation gated to the cycle, is a *reachable cycle* in the abstract sense,
so the abstract `reachable_cycle_implies_infinite_safe_path` yields an `InfiniteSafePath`. The
concrete certificate *is* an abstract reachable-cycle certificate; the recurrent state is extracted
by the library's existing pigeonhole `trace_exists_period`. -/

section ConcreteSinglePlayer

open Tetris

variable {cfg : GameConfig} (C : ClosedCycle cfg)

/-- The deterministic policy-step relation of a closed cycle, **gated** to the cycle's states so its
edges provably avoid loss. The concrete instance of the abstract relation `R`. -/
def cycleRel : GameState → GameState → Prop :=
  fun g g' => g ∈ C.states ∧ g' = g.step cfg (C.policy g)

/-- The cycle trace realizes an `R`-walk: `RelPow (cycleRel C) n g0 (trace … n)`. -/
theorem relPow_cycleRel_trace {g0 : GameState} (h0 : g0 ∈ C.states) (n : ℕ) :
    RelPow (cycleRel C) n g0 (trace cfg C.policy g0 n) := by
  induction n with
  | zero => exact RelPow.zero g0
  | succ k ih => exact ih.snoc ⟨C.trace_mem_states h0 k, C.trace_succ_eq g0 k⟩

/-- Edges of `cycleRel C` avoid loss at both ends (cycle membership + closure + `not_lost`). -/
theorem cycleRel_safe {g g' : GameState} (h : cycleRel C g g') :
    ¬ g.lost cfg ∧ ¬ g'.lost cfg := by
  obtain ⟨hmem, heq⟩ := h
  refine ⟨C.not_lost g hmem, ?_⟩
  rw [heq]; exact C.not_lost _ (C.closed g hmem)

/-- **A concrete closed cycle is an abstract reachable cycle.** From any entry `g0 ∈ C.states`, some
trace state is on a `cycleRel`-cycle and is reachable from `g0`. The recurrent state and period come
from the library's pigeonhole `ClosedCycle.trace_exists_period`. -/
theorem closedCycle_reachableCycle {g0 : GameState} (h0 : g0 ∈ C.states) :
    ReachableCycle (cycleRel C) g0 := by
  obtain ⟨i, d, hd, _hle, hper⟩ := C.trace_exists_period h0
  refine ⟨trace cfg C.policy g0 i, (relPow_cycleRel_trace C h0 i).reachable, d, hd, ?_⟩
  have hwalk := relPow_cycleRel_trace C (C.trace_mem_states h0 i) d
  rw [← C.trace_add g0 i d, ← hper] at hwalk
  exact hwalk

/-- **The abstract cycle theorem recovers concrete infinite play.** A single-player closed cycle
yields an abstract `InfiniteSafePath` from any entry state — an instance of
`reachable_cycle_implies_infinite_safe_path`, with `GameState.lost` as the dead predicate. -/
theorem closedCycle_infiniteSafePath {g0 : GameState} (h0 : g0 ∈ C.states) :
    InfiniteSafePath (cycleRel C) (fun g => g.lost cfg) g0 :=
  reachable_cycle_implies_infinite_safe_path (fun _ _ h => cycleRel_safe C h)
    (closedCycle_reachableCycle C h0)

end ConcreteSinglePlayer

/-! ### The faithful adversarial side: `safe` is an abstract recurrent support

The library's `Tetris.safe cfg` is the greatest fixed point of the adversarial "∀ piece ∈ bag,
∃ valid placement landing back in the set" operator. Under the repo's `(board, bag)` convention the
piece is revealed *before* placement, so this `∀p ∃pl` order is exactly the abstract **weak**
`SafeRecurrentSupport` (with `step g pl p := adversarialStep g p pl`, `legalDraws := GameState.bag`,
`Dead := GameState.lost`). We exhibit `safe cfg` as an instance — the matrix-power presentation of
the gfp — and derive an abstract `InfiniteSafePath` from `init ∈ safe`, complementing the library's
own `safe_extract`. (We drop the concrete `pl.piece = p ∧ pl.Valid` data: the abstract layer needs
only that the witnessed placement lands back in the support.) -/

section ConcreteAdversarial

open Tetris

/-- **The safe set is an abstract (weak) recurrent support.** `Tetris.safe cfg` instantiates
`SafeRecurrentSupport` for the adversarial transition — the matrix-power reading of the gfp. -/
theorem safe_isSafeRecurrentSupport (cfg : GameConfig) :
    SafeRecurrentSupport (fun g => g.lost cfg) (fun g => g.bag)
      (fun g pl p => adversarialStep cfg g p pl) (safe cfg) := by
  intro g hg
  rw [mem_safe_iff] at hg
  refine ⟨hg.1, fun p hp => ?_⟩
  obtain ⟨pl, _hpiece, _hv, hmem⟩ := hg.2 p hp
  exact ⟨pl, hmem⟩

/-- The safe states with a nonempty bag form a recurrent support whose legal-draw sets are all
nonempty — the successor's bag is `g.bag.draw p`, always nonempty (`Bag.draw_nonempty`). This is the
sub-support on which `InfiniteSafePath` applies (a safe state with empty bag would be vacuously safe
but have no successor edge). -/
theorem safe_nonemptyBag_isSafeRecurrentSupport (cfg : GameConfig) :
    SafeRecurrentSupport (fun g => g.lost cfg) (fun g => g.bag)
      (fun g pl p => adversarialStep cfg g p pl)
      {g | g ∈ safe cfg ∧ g.bag.Nonempty} := by
  rintro g ⟨hsafe, _hne⟩
  rw [mem_safe_iff] at hsafe
  refine ⟨hsafe.1, fun p hp => ?_⟩
  obtain ⟨pl, _hpiece, _hv, hmem⟩ := hsafe.2 p hp
  exact ⟨pl, hmem, Bag.draw_nonempty g.bag p⟩

/-- **`init ∈ safe` ⇒ abstract infinite safe play.** An instance of
`reachable_safe_recurrent_support_implies_infinite_survival` on the nonempty-bag safe support,
seeded at `init` (whose bag is `Bag.full`). Complements `Tetris.safe_extract`: the same membership
that the library turns into `TetrisSolvable` is, in matrix-power terms, a reachable recurrent
support, hence an `InfiniteSafePath`. -/
theorem safe_init_infiniteSafePath {cfg : GameConfig} (h : GameState.init ∈ safe cfg) :
    InfiniteSafePath
      (SafeEdge (fun g => g.lost cfg) (fun g => g.bag)
        (fun g pl p => adversarialStep cfg g p pl))
      (fun g => g.lost cfg) GameState.init := by
  have hinit : GameState.init ∈ {g : GameState | g ∈ safe cfg ∧ g.bag.Nonempty} := by
    refine ⟨h, ?_⟩
    rw [GameState.init_bag]; exact Bag.full_nonempty
  exact reachable_safe_recurrent_support_implies_infinite_survival
    (fun _ hg => hg.2) (safe_nonemptyBag_isSafeRecurrentSupport cfg)
    ⟨GameState.init, hinit, Relation.ReflTransGen.refl⟩

/-! ### The per-piece family on real Tetris, and the `Fintype` boundary

Instantiating §5's controllable predecessor with the actual move `step g pl p := adversarialStep
cfg g p pl`, the per-piece transition `A_p` is "every placement of piece `p`", and the bag's ∀ is
the conjunction over `p ∈ g.bag`. The concrete `safe` set is a common sub-invariant of this family
and sits inside its greatest fixed point.

**Boundary (honest).** The *literal* ℕ-matrix `Matrix GameState GameState ℕ` and its sub-eigenvector
*sum* need `Fintype GameState`, which is **false** in this model: `Coord = ℕ × ℕ` and a placement's
`col : ℕ`, so `GameState` is an *infinite ambient type* — finiteness lives in the `WF`/reachability
predicates, not the type. So on concrete Tetris the per-piece *matrices* stay abstract; their
faithful concrete shadow is the per-piece *relations*, and "common sub-eigenvector" reads as the
relational `∀ p ∈ bag, ∃ placement, stays in X` (i.e. `mem_safe_iff`). -/

/-- **`safe` is a common sub-invariant of the per-piece transitions** — the relational
sub-eigenvector. For every safe state and every drawable piece `p`, the per-piece transition `A_p`
(some placement of `p`) lands back in `safe`: `1_safe ≤ A_p ⬝ᵥ 1_safe` for each `p ∈ bag`, in
relational form. This is `mem_safe_iff` read through the per-piece family. -/
theorem safe_common_subInvariant (cfg : GameConfig) {g : GameState} (hg : g ∈ safe cfg)
    {p : Piece} (hp : p ∈ g.bag) : ∃ pl, adversarialStep cfg g p pl ∈ safe cfg := by
  rw [mem_safe_iff] at hg
  obtain ⟨pl, _, _, hmem⟩ := hg.2 p hp
  exact ⟨pl, hmem⟩

/-- **`safe` lies in the adversarial gfp of the per-piece family.** Wiring §5's controllable
predecessor to the Tetris move function, the concrete `safe` set is contained in `adversarialSafe`,
the greatest common sub-invariant of the per-piece transitions `{A_p}`. (Containment, not equality:
the abstract `step` does not re-impose `pl.piece = p ∧ pl.Valid`, so `adversarialSafe` is at least
as big; `safe` is exactly the *valid* part.) -/
theorem safe_subset_adversarialSafe (cfg : GameConfig) :
    safe cfg ⊆ adversarialSafe (fun g => g.lost cfg) (fun g => g.bag)
      (fun g pl p => adversarialStep cfg g p pl) :=
  subset_adversarialSafe_of_recurrentSupport _ _ _ (safe_isSafeRecurrentSupport cfg)

end ConcreteAdversarial

/-! ### Materializing the matrices: a finite bounded Tetris state space

Option 1 hit the wall that `GameState` is an infinite ambient type. Here we carve out the genuinely
finite *live* state space — boards confined to the `cols × rows` grid — so the per-piece matrices
become honest `Matrix`-objects over a `Fintype`. A board is *in-grid* iff every cell lies in
`region cfg = [0,cols) × [0,rows)`; that is exactly `WF` (`p.1 < cols`) together with not-lost
(`p.2 < rows`). The in-grid boards form a `Fintype` (subsets of a fixed finite grid), and the action
space `Rotation × Fin cols` is finite — so `adjFor (boundedStep cfg) p` is a genuine finite matrix
and the §5 common-sub-eigenvector encoding lands on a concrete `Fintype`. -/

section FiniteBoundedTetris

open Tetris

/-- The `cols × rows` playable grid as a finite cell set. -/
def region (cfg : GameConfig) : Finset Coord :=
  Finset.range cfg.cols ×ˢ Finset.range cfg.rows

/-- In-grid boards: every cell inside `region` — `WF` (`col < cols`) ∧ not-lost (`row < rows`). -/
abbrev BoundedBoard (cfg : GameConfig) := {b : Board // b ⊆ region cfg}

/-- There are finitely many in-grid boards (subsets of the fixed grid). -/
instance (cfg : GameConfig) : Fintype (BoundedBoard cfg) :=
  Fintype.subtype (region cfg).powerset (fun _ => Finset.mem_powerset)

/-- A finite bounded game state: an in-grid board with a 7-bag. -/
abbrev BoundedState (cfg : GameConfig) := BoundedBoard cfg × Bag

/-- The finite action space: a rotation and an in-bounds target column. -/
abbrev BoundedAction (cfg : GameConfig) := Rotation × Fin cfg.cols

/-- The bounded transition: run the real `adversarialStep` for the drawn piece at the chosen
rotation/column, then clamp the board back into the grid (out-of-grid overflow is dropped — a
placeholder; the faithful version would route overflow to a lost sink). Total on the finite type. -/
def boundedStep (cfg : GameConfig) (s : BoundedState cfg) (a : BoundedAction cfg) (p : Piece) :
    BoundedState cfg :=
  let g' := adversarialStep cfg ⟨s.1.1, s.2⟩ p ⟨p, a.1, a.2.1⟩
  (⟨g'.board ∩ region cfg, Finset.inter_subset_right⟩, g'.bag)

/-- **The per-piece matrices materialize on the finite bounded state space.** §5's encoding
instantiated at `State := BoundedState cfg`, `Action := BoundedAction cfg`: a finite set `X` of
bounded states is closed under the adversarial `∀ piece ∈ bag, ∃ (rotation, column)` iff `1_X` is a
common sub-eigenvector of the now-genuine per-piece matrices `adjFor (boundedStep cfg) p` — the
matrix form of the survival condition, on a concrete `Fintype`. -/
theorem bounded_recurrentSupport_iff_common_subEigenvector (cfg : GameConfig)
    (X : Finset (BoundedState cfg)) :
    (∀ s ∈ X, ∀ p ∈ s.2, ∃ a, boundedStep cfg s a p ∈ X) ↔
      ∀ s ∈ X, ∀ p ∈ s.2,
        1 ≤ ∑ t, adjFor (boundedStep cfg) p s t * (if t ∈ X then (1 : ℕ) else 0) :=
  recurrentSupport_iff_common_subEigenvector (fun s => s.2) (boundedStep cfg) X

end FiniteBoundedTetris

end MatrixPowerSurvival
