import Mathlib

/-!
# Tetris as a topical dynamical system (the object, not the orbit graph)

Every prior route treated survival as a search over a state *graph*. This file studies the
**dynamical object** whose orbits that graph merely samples. The key structural fact:

> The Tetris **surface map** — drop a piece onto a column-height vector — is **topical**
> (monotone + additively homogeneous), i.e. a *min-max function* in Gunawardena's sense.

Topical maps are exactly the nonexpansive maps for the sup-norm, and they carry a
**spectral theory**: a *cycle time* `χ = lim Tⁿ(h)/n` (the asymptotic growth rate of the
stack) and **eigen-surfaces** `T(v) = v + λ·𝟙` (the recurrent shapes). This recasts the
whole problem:

* **Survival ⟺ the player can force cycle time `≤ 0`** — a *mean-payoff game on topical
  maps*, whose value is a single number. Our proven clearing equilibrium (`~4/cols`
  lines/piece, `EnergyGame.survival_forces_clears`) is precisely this cycle time.
* **The carrier is an eigen-surface, not a set.** The recurrent surface the optimal play
  settles into is a max-plus eigenvector `v` with `T(v) = v + λ·𝟙`. A single vector `v` and
  rate `λ ≤ 0` *replaces* the >5·10⁶-node carrier graph and certifies bounded play
  (`eigen_surface_bounded` below).
* **Holes are the homogeneity defect.** Lifting the whole board commutes with the dynamics
  *only when the board is hole-free*; a buried hole is exactly where homogeneity fails. So
  `debt` (`HoleDebt.lean`) measures the distance from the topical manifold, and the energy
  invariant `debt ≤ K` says "stay within a bounded defect of the topical object."

## What is proven here (sorry-free)

The abstract topical calculus, and that the Tetris piece-drop on surfaces is an instance:
`Topical.nonexpansive`, `Topical.comp`, `eigen_iterate`, `eigen_surface_bounded`,
`dropMap_topical`. The concrete link: the *monotone* half is already proven for the real
engine (`WqoCarrier.place_domLE_mono`); the *homogeneous* half — the genuinely new
structural content — is `dropMap_homog` here.
-/

namespace Tetris.Topical

/-- A **surface**: a column-height vector. Heights live in `ℤ` so that translation (lifting
the whole stack) is clean — the homogeneity that makes the dynamics topical. -/
abbrev Surface (n : ℕ) := Fin n → ℤ

/-- Lift an entire surface by a constant `c` (raise the stack by `c` rows). -/
def shift {n : ℕ} (h : Surface n) (c : ℤ) : Surface n := fun j => h j + c

@[simp] theorem shift_apply {n : ℕ} (h : Surface n) (c : ℤ) (j : Fin n) :
    shift h c j = h j + c := rfl

@[simp] theorem shift_zero {n : ℕ} (h : Surface n) : shift h 0 = h := by
  funext j; simp [shift]

theorem shift_shift {n : ℕ} (h : Surface n) (c d : ℤ) :
    shift (shift h c) d = shift h (c + d) := by
  funext j; simp [shift]; ring

/-! ## The abstract topical calculus -/

/-- **Homogeneous**: the map commutes with lifting the stack — `T(h + c·𝟙) = T(h) + c·𝟙`.
This is the structural heart: the dynamics only sees *relative* heights. -/
def Homogeneous {n : ℕ} (T : Surface n → Surface n) : Prop :=
  ∀ (h : Surface n) (c : ℤ), T (shift h c) = shift (T h) c

/-- **Topical** = monotone + homogeneous (a min-max / Gunawardena function). -/
structure Topical {n : ℕ} (T : Surface n → Surface n) : Prop where
  mono : Monotone T
  homog : Homogeneous T

/-- **Nonexpansive** (additive, order form): if `h` is everywhere `≤ h' + c`, so are their
images. Equivalent to `‖T h − T h'‖∞ ≤ ‖h − h'‖∞`. -/
def Nonexpansive {n : ℕ} (T : Surface n → Surface n) : Prop :=
  ∀ (h h' : Surface n) (c : ℤ), (∀ j, h j ≤ h' j + c) → (∀ j, T h j ≤ T h' j + c)

/-- **Topical ⇒ nonexpansive** — the theorem that makes the cycle time exist. From
`h ≤ h' + c·𝟙`, monotonicity gives `T h ≤ T(h' + c·𝟙)`, and homogeneity rewrites the right
side to `T h' + c·𝟙`. -/
theorem Topical.nonexpansive {n : ℕ} {T : Surface n → Surface n} (ht : Topical T) :
    Nonexpansive T := by
  intro h h' c hle j
  have h1 : h ≤ shift h' c := fun k => hle k
  have h2 := ht.mono h1
  rw [ht.homog] at h2
  simpa using h2 j

/-- Topical maps compose (so a whole **bag-strategy** — seven piece-drops chained — is
again topical, and has its own cycle time and eigen-surfaces). -/
theorem Topical.comp {n : ℕ} {T S : Surface n → Surface n}
    (hT : Topical T) (hS : Topical S) : Topical (T ∘ S) where
  mono := hT.mono.comp hS.mono
  homog := by
    intro h c
    show T (S (shift h c)) = shift (T (S h)) c
    rw [hS.homog h c, hT.homog (S h) c]

/-! ## Eigen-surfaces: the carrier as a single vector -/

/-- `v` is an **eigen-surface** of `T` with rate `λ` when one application lifts it uniformly:
`T(v) = v + λ·𝟙`. The recurrent shape of the play; the spectral replacement for the carrier
set. -/
def IsEigen {n : ℕ} (T : Surface n → Surface n) (v : Surface n) (lam : ℤ) : Prop :=
  T v = shift v lam

/-- Iterating an eigen-surface just keeps lifting it: `Tᵏ(v) = v + (k·λ)·𝟙`. -/
theorem eigen_iterate {n : ℕ} {T : Surface n → Surface n} {v : Surface n} {lam : ℤ}
    (hT : Topical T) (he : IsEigen T v lam) (k : ℕ) :
    T^[k] v = shift v (k * lam) := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, hT.homog, he, shift_shift]
    funext j; simp [shift]; ring

/-- **The spectral survival certificate.** An eigen-surface with rate `λ ≤ 0` keeps every
column height bounded forever (`Tᵏ(v) ≤ v` pointwise, for all `k`). So a *single* vector
`v` and a sign condition on `λ` certify infinite bounded play — no enumeration. This is the
topical form of "the carrier exists": it is the max-plus eigenvector. -/
theorem eigen_surface_bounded {n : ℕ} {T : Surface n → Surface n} {v : Surface n} {lam : ℤ}
    (hT : Topical T) (he : IsEigen T v lam) (hlam : lam ≤ 0) (k : ℕ) (j : Fin n) :
    (T^[k] v) j ≤ v j := by
  rw [eigen_iterate hT he k]
  have hk : (k : ℤ) * lam ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by positivity) hlam
  simp only [shift_apply]
  linarith

/-! ## The Tetris piece-drop IS topical

A piece placement is abstracted by its **profile** over the columns it occupies: the bottom
and top cell offsets per occupied column. The hard drop is the classic max-plus operation —
the landing level is `max` over the footprint of `height − bottom_offset`, and the new
heights are that level plus the top offsets. We prove this map is monotone *and* homogeneous
(hence topical, hence nonexpansive). -/

/-- A placed piece's column profile: occupied columns, with bottom/top cell offsets. -/
structure PieceProfile (n : ℕ) where
  occ : Finset (Fin n)
  ne : occ.Nonempty
  bot : Fin n → ℤ
  top : Fin n → ℤ

/-- Adding a constant commutes with `Finset.sup'` over `ℤ` (a clean max-plus identity). -/
theorem sup'_add_const {α : Type*} {s : Finset α} (hs : s.Nonempty) (g : α → ℤ) (c : ℤ) :
    (s.sup' hs fun k => g k + c) = s.sup' hs g + c := by
  apply le_antisymm
  · apply Finset.sup'_le
    intro k hk
    have := Finset.le_sup' g hk
    linarith
  · have h2 : s.sup' hs g ≤ (s.sup' hs fun k => g k + c) - c := by
      apply Finset.sup'_le
      intro k hk
      have := Finset.le_sup' (fun k => g k + c) hk
      simp only at this
      linarith
    linarith

/-- The **landing level** of the drop: `max` over the footprint of `height − bottom_offset`.
This is the max-plus inner product at the heart of the dynamics. -/
def base {n : ℕ} (p : PieceProfile n) (h : Surface n) : ℤ :=
  p.occ.sup' p.ne fun k => h k - p.bot k

/-- **The surface drop map**: occupied columns rise to `base + top_offset + 1`; others
unchanged. (Hole-free model — the topical object; holes are the defect that perturbs it.) -/
def dropMap {n : ℕ} (p : PieceProfile n) (h : Surface n) : Surface n :=
  fun j => if j ∈ p.occ then base p h + p.top j + 1 else h j

theorem base_mono {n : ℕ} (p : PieceProfile n) {h h' : Surface n} (hle : h ≤ h') :
    base p h ≤ base p h' := by
  apply Finset.sup'_le
  intro k hk
  have : h k - p.bot k ≤ h' k - p.bot k := by have := hle k; linarith
  exact le_trans this (Finset.le_sup' (fun k => h' k - p.bot k) hk)

theorem base_shift {n : ℕ} (p : PieceProfile n) (h : Surface n) (c : ℤ) :
    base p (shift h c) = base p h + c := by
  unfold base
  rw [show (fun k => shift h c k - p.bot k) = (fun k => (h k - p.bot k) + c) by
        funext k; simp [shift]; ring]
  exact sup'_add_const p.ne (fun k => h k - p.bot k) c

/-- **The drop is monotone.** (For the real engine this is `WqoCarrier.place_domLE_mono`.) -/
theorem dropMap_mono {n : ℕ} (p : PieceProfile n) : Monotone (dropMap p) := by
  intro h h' hle j
  unfold dropMap
  by_cases hj : j ∈ p.occ
  · simp only [if_pos hj]
    have := base_mono p hle
    linarith
  · simp only [if_neg hj]
    exact hle j

/-- **The drop is homogeneous** — the genuinely new structural fact: lifting the whole
surface lifts the outcome equally, because the drop reads only *relative* heights. -/
theorem dropMap_homog {n : ℕ} (p : PieceProfile n) : Homogeneous (dropMap p) := by
  intro h c
  funext j
  unfold dropMap
  by_cases hj : j ∈ p.occ
  · simp only [if_pos hj, shift_apply, base_shift]
    ring
  · simp only [if_neg hj, shift_apply]

/-- **The Tetris surface drop is topical** — monotone + homogeneous. Therefore nonexpansive,
therefore it has a cycle time and eigen-surfaces. This is the object. -/
theorem dropMap_topical {n : ℕ} (p : PieceProfile n) : Topical (dropMap p) :=
  ⟨dropMap_mono p, dropMap_homog p⟩

/-- Corollary: the Tetris drop is sup-norm nonexpansive. -/
theorem dropMap_nonexpansive {n : ℕ} (p : PieceProfile n) : Nonexpansive (dropMap p) :=
  (dropMap_topical p).nonexpansive

/-! ## The bag-strategy map, and existence of the eigen-surface

A bag-strategy is the player's seven placements chained — a composition of `dropMap`s, hence
itself topical. The **carrier-as-eigenvector** question becomes: does the bag map have an
eigen-surface `T(v) = v + λ·𝟙`? We attack it the way tropical/topical spectral theory does —
on the *projective quotient* (heights mod translation): if the surface-*shape* orbit stays in
a finite set (bounded roughness — exactly our search finding), then by pigeonhole an
**eigen-cycle** exists. That periodic eigen-surface, with rate `≤ 0`, *is* the carrier. -/

/-- The bag-strategy surface map: drop the listed pieces in order. -/
def bagMap {n : ℕ} (ps : List (PieceProfile n)) (h : Surface n) : Surface n :=
  ps.foldl (fun acc p => dropMap p acc) h

theorem bagMap_cons {n : ℕ} (p : PieceProfile n) (ps : List (PieceProfile n)) :
    bagMap (p :: ps) = bagMap ps ∘ dropMap p := by
  funext h; simp [bagMap, Function.comp]

/-- A whole bag-strategy is topical — so it has a cycle time and eigen-surfaces of its own. -/
theorem bagMap_topical {n : ℕ} (ps : List (PieceProfile n)) : Topical (bagMap ps) := by
  induction ps with
  | nil => exact ⟨monotone_id, fun h c => rfl⟩
  | cons p ps ih => rw [bagMap_cons]; exact ih.comp (dropMap_topical p)

/-- Iterates of a topical map are topical. -/
theorem Topical.iterate {n : ℕ} {T : Surface n → Surface n} (hT : Topical T) :
    ∀ k, Topical (T^[k])
  | 0 => ⟨monotone_id, fun h c => rfl⟩
  | k + 1 => by rw [Function.iterate_succ]; exact (hT.iterate k).comp hT

/-- A **periodic eigen-surface** (eigen-cycle): `Tᵏ(w) = w + c·𝟙` with period `k > 0` and
rate `c`. The recurrent carrier shape, returning to itself (up to a uniform lift) every `k`
steps. -/
def IsEigenCycle {n : ℕ} (T : Surface n → Surface n) (w : Surface n) (k : ℕ) (c : ℤ) : Prop :=
  0 < k ∧ T^[k] w = shift w c

/-- **Constructive existence: a repeated *shape* yields an eigen-cycle.** If two iterates
`Tᵃ(v)` and `Tᵇ(v)` (`a < b`) differ only by a uniform lift, then `Tᵃ(v)` is a periodic
eigen-surface of period `b − a`. (Pure iteration algebra — no topical hypothesis needed.) -/
theorem eigen_cycle_of_sameShape {n : ℕ} (T : Surface n → Surface n) (v : Surface n)
    {a b : ℕ} {c : ℤ} (hab : a < b) (hs : T^[b] v = shift (T^[a] v) c) :
    IsEigenCycle T (T^[a] v) (b - a) c := by
  refine ⟨Nat.sub_pos_of_lt hab, ?_⟩
  rw [← Function.iterate_add_apply, Nat.sub_add_cancel hab.le]
  exact hs

/-- **An eigen-cycle with non-positive rate keeps every height bounded forever.** Every
iterate `Tᵐ(v)` is bounded by one of the first `k` iterates (`Tᵐ ᵐᵒᵈ ᵏ(v)`), because
`m = k·(m/k) + m%k` and the `k`-periodic lift contributes `(m/k)·c ≤ 0`. So the orbit lives
in the finite set `{T⁰v, …, Tᵏ⁻¹v}` shifted down — a bounded invariant region, i.e. survival.
-/
theorem eigen_cycle_bounded {n : ℕ} {T : Surface n → Surface n} {v : Surface n} {k : ℕ} {c : ℤ}
    (hT : Topical T) (he : IsEigenCycle T v k c) (hc : c ≤ 0) (m : ℕ) (j : Fin n) :
    (T^[m] v) j ≤ (T^[m % k] v) j := by
  obtain ⟨hk, heq⟩ := he
  have hmul : T^[k * (m / k)] v = shift v ((m / k : ℤ) * c) := by
    rw [Function.iterate_mul]
    have h := eigen_iterate (hT.iterate k) heq (m / k)
    rw [h]; push_cast; ring_nf
  have hstep : (T^[m] v) j = (T^[m % k] v) j + (m / k : ℤ) * c := by
    conv_lhs => rw [show m = m % k + k * (m / k) from (Nat.mod_add_div m k).symm]
    rw [Function.iterate_add_apply, hmul, (hT.iterate (m % k)).homog]
    simp [shift]
  rw [hstep]
  have : (m / k : ℤ) * c ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by positivity) hc
  linarith

/-- **The existence theorem (pigeonhole on the projective quotient).** If every iterate of
`v` is a uniform lift of some surface in a *finite* set `F` (the surface-shape orbit is
bounded — bounded roughness), then an **eigen-cycle exists**. This is the spectral form of
"the carrier exists": bounded shapes ⇒ a recurrent eigen-surface. Combined with
`eigen_cycle_bounded`, a non-positive rate then certifies infinite bounded play. -/
theorem exists_eigen_cycle {n : ℕ} (T : Surface n → Surface n) (v : Surface n)
    (F : Finset (Surface n)) (hF : ∀ k, ∃ f ∈ F, ∃ c, T^[k] v = shift f c) :
    ∃ (a k : ℕ) (c : ℤ), IsEigenCycle T (T^[a] v) k c := by
  classical
  choose g hg cc hcc using hF
  obtain ⟨a, b, hab, hgab⟩ :=
    Finite.exists_ne_map_eq_of_infinite (fun k => (⟨g k, hg k⟩ : {x // x ∈ F}))
  have hgeq : g a = g b := congrArg Subtype.val hgab
  have mk : ∀ {x y : ℕ}, x < y → g x = g y →
      ∃ (a k : ℕ) (c : ℤ), IsEigenCycle T (T^[a] v) k c := by
    intro x y hxy hxyg
    have hgx : g x = shift (T^[x] v) (- cc x) := by
      rw [hcc x, shift_shift]; funext j; simp [shift]
    have hb : T^[y] v = shift (T^[x] v) (cc y - cc x) := by
      rw [hcc y, ← hxyg, hgx, shift_shift]
      funext j; simp [shift]; ring
    exact ⟨x, y - x, cc y - cc x, eigen_cycle_of_sameShape T v hxy hb⟩
  rcases lt_or_gt_of_ne hab with h | h
  · exact mk h hgeq
  · exact mk h hgeq.symm

#print axioms eigen_surface_bounded
#print axioms dropMap_topical
#print axioms bagMap_topical
#print axioms exists_eigen_cycle
#print axioms eigen_cycle_bounded

/-! ## Where this points (the analytic program, no enumeration)

* **Find the eigen-surface.** Exhibit `v, λ` with `dropMap_bag(v) = v + λ·𝟙` for the
  bag-strategy map (a composition of seven `dropMap`s with the player's placement choices)
  and `λ ≤ 0`. By `eigen_surface_bounded` that is a *finite, vector-sized* survival
  certificate — the carrier-as-eigenvector, importable into the `EnergyGame` reduction.
* **Cycle time = clearing rate.** The mean-payoff value of the player-min/adversary-max
  topical game is the proven `~4/cols` equilibrium; surviving ⟺ value `≤ 0`.
* **Defect theory for holes.** Quantify how a bounded-debt perturbation moves the cycle
  time — the perturbation theory of topical maps off the homogeneous manifold (`debt`). -/

end Tetris.Topical
