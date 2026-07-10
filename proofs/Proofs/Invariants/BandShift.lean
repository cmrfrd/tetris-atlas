import Proofs.Invariants.HoledSkyline
import Proofs.Invariants.Confluence

/-!
# Well-anchored band lifts: the translation-quotient transport layer

The translation symmetry of the debt-1 board algebra is *well-anchored*: one
well column is pinned at height `0` (which blocks every line clear) and the
remaining band columns shift up by `c`. `bandLift` lifts a base-0 profile to
band base `c`; `holeLift` rides the (optional) buried cell along. The
transport theorems (`place_debtBoard_bandLift`, `drain_debtBoard_bandLift`)
let a `DebtCertificate`-style closure obligation be proven once at the
representative and reused at every base — see
`Proofs/Safety/ShiftCertificate.lean` for the certificate that packages this.
-/

namespace Tetris

/-- Transport an optional buried cell up by `c` rows. -/
def holeLift (c : ℕ) : Option Coord → Option Coord :=
  Option.map (fun x => (x.1, x.2 + c))

@[simp] theorem holeLift_none (c : ℕ) : holeLift c none = none := rfl

@[simp] theorem holeLift_some (c : ℕ) (x : Coord) :
    holeLift c (some x) = some (x.1, x.2 + c) := rfl

namespace Board

/-- The well-anchored band lift: column `w` pinned at `0`, every other
column raised by `c`. `bandLift w 0 h = h` exactly when `h w = 0`. -/
def bandLift (w c : ℕ) (h : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 0 else h j + c

@[simp] theorem bandLift_well (w c : ℕ) (h : ℕ → ℕ) : bandLift w c h w = 0 :=
  if_pos rfl

theorem bandLift_ne (w c : ℕ) (h : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    bandLift w c h j = h j + c := if_neg hj

theorem bandLift_zero {w : ℕ} {h : ℕ → ℕ} (hw : h w = 0) :
    bandLift w 0 h = h := by
  funext j
  by_cases hj : j = w
  · subst hj; simp [bandLift, hw]
  · simp [bandLift, hj]

/-- Uniform membership for debt-≤1 boards, covering both `Option` cases. -/
theorem mem_debtBoard {cfg : GameConfig} {h : ℕ → ℕ} {ho : Option Coord}
    (p : Coord) :
    p ∈ debtBoard cfg h ho
      ↔ (∀ x, ho = some x → p ≠ x) ∧ p.1 < cfg.cols ∧ p.2 < h p.1 := by
  cases ho with
  | none => simp [debtBoard_none, mem_skyline']
  | some x =>
      rw [debtBoard_some, mem_holedSkyline]
      constructor
      · rintro ⟨hne, hc, hr⟩
        exact ⟨fun y hy => by cases hy; exact hne, hc, hr⟩
      · rintro ⟨hne, hc, hr⟩
        exact ⟨hne x rfl, hc, hr⟩

/-- Truncated-subtraction sup shift: raising every value by `c` raises the
sup of `value − row` by exactly `c`, provided some cell sits at row `0`. -/
theorem sup_sub_add_shift (s : Finset Coord) (f : Coord → ℕ) (c : ℕ)
    (hbot : ∃ cell ∈ s, cell.2 = 0) :
    s.sup (fun cell => f cell + c - cell.2)
      = s.sup (fun cell => f cell - cell.2) + c := by
  obtain ⟨c₀, hc₀, hc₀0⟩ := hbot
  refine le_antisymm (Finset.sup_le fun cell hcell => ?_) ?_
  · have hb : f cell - cell.2 ≤ s.sup (fun cell => f cell - cell.2) :=
      Finset.le_sup (f := fun cell => f cell - cell.2) hcell
    omega
  · rcases Finset.exists_mem_eq_sup s ⟨c₀, hc₀⟩ (fun cell => f cell - cell.2)
      with ⟨cm, hcm, heq⟩
    have heq' : s.sup (fun cell => f cell - cell.2) = f cm - cm.2 := heq
    rw [heq']
    by_cases hz : f cm - cm.2 = 0
    · rw [hz]
      have hb : f c₀ + c - c₀.2 ≤ s.sup (fun cell => f cell + c - cell.2) := by
        exact Finset.le_sup (f := fun cell => f cell + c - cell.2) hc₀
      omega
    · have hb : f cm + c - cm.2 ≤ s.sup (fun cell => f cell + c - cell.2) := by
        exact Finset.le_sup (f := fun cell => f cell + c - cell.2) hcm
      omega

/-- **Shift-equivariance of the hard drop on debt-1 boards.** For a
band placement (in-bounds, avoiding the well) the drop offset at band
base `c` is the base-0 offset plus `c`; the strictly covered hole is
invisible on both boards. -/
theorem dropOffset_debtBoard_bandLift {cfg : GameConfig} {w c : ℕ}
    {ρ : ℕ → ℕ} {ho : Option Coord} {pl : Placement}
    (hcols : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols)
    (havoid : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ w)
    (hho : ∀ x, ho = some x →
      x.1 ≠ w ∧ x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    pl.dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = pl.dropOffset (debtBoard cfg ρ ho) + c := by
  have hcov : ∀ x, ho = some x → x.2 + 1 < ρ x.1 := fun x hx => (hho x hx).2.2
  have hcovL : ∀ x, holeLift c ho = some x → x.2 + 1 < bandLift w c ρ x.1 := by
    rintro x hx
    cases ho with
    | none => simp at hx
    | some x₀ =>
        rw [holeLift_some, Option.some.injEq] at hx
        subst hx
        rw [bandLift_ne w c ρ (hho x₀ rfl).1]
        have := (hho x₀ rfl).2.2
        omega
  have hL : pl.dropOffset (debtBoard cfg (bandLift w c ρ) (holeLift c ho))
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) + c - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcovL, if_pos (hcols cell hcell),
      bandLift_ne w c ρ (havoid cell hcell)]
  have hR : pl.dropOffset (debtBoard cfg ρ ho)
      = pl.shapeUp.sup (fun cell => ρ (pl.col + cell.1) - cell.2) := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_congr rfl fun cell hcell => ?_
    rw [colHeight_debtBoard hcov, if_pos (hcols cell hcell)]
  rw [hL, hR]
  exact sup_sub_add_shift pl.shapeUp (fun cell => ρ (pl.col + cell.1)) c
    (Placement.shapeUp_exists_bottom pl.piece pl.rot)

end Board
end Tetris
