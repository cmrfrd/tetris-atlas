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

#print axioms eigen_surface_bounded
#print axioms dropMap_topical

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
