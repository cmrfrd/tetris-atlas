import Proofs.Invariants.HoledSkyline

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

end Board
end Tetris
