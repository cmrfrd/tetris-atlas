import Proofs.Invariants.BandShift
import Proofs.Invariants.BandMechanisms

/-!
# The plinth: floor-1 transport for the entombed-hole regime

Findings D1/D2 (see PROGRESS.md, 2026-07-12): the `bandLift` certificates
cannot be inhabited — the base never rises, and the forced row-0 bootstrap
hole blocks re-anchoring. The plinth regime fixes both: one bag-1 placement
(`place_wellPlug_flat`) plugs the well's row 0; row 0 becomes a permanent
floor of nine cells plus the entombed hole, which keeps row 0 from ever
being full — the floor is immortal. The well operates at height 1, drains
fill and clear rows 1–4 (`drain_debtBoard_plinthLift`), and the active band
rides `c + 1` above the floor in coordinates the hole never touches, so
re-anchoring (`ReanchorsTo`) is unobstructed.
-/

namespace Tetris
namespace Board

/-- The plinth lift: well pinned at height 1 (the plug), band riding `c + 1`
above the floor. -/
def plinthLift (w c : ℕ) (ρ : ℕ → ℕ) : ℕ → ℕ :=
  fun j => if j = w then 1 else ρ j + c + 1

@[simp] theorem plinthLift_well (w c : ℕ) (ρ : ℕ → ℕ) :
    plinthLift w c ρ w = 1 := if_pos rfl

theorem plinthLift_ne (w c : ℕ) (ρ : ℕ → ℕ) {j : ℕ} (hj : j ≠ w) :
    plinthLift w c ρ j = ρ j + c + 1 := if_neg hj

/-- Re-anchoring: the same absolute band at a different base split. The
board-level no-op that lets the base rise as the pattern grows (fix for
finding D1). -/
def ReanchorsTo (well : ℕ) (ρ : ℕ → ℕ) (c : ℕ) (ρ' : ℕ → ℕ) (c' : ℕ) : Prop :=
  (∀ j, j ≠ well → ρ j + c = ρ' j + c') ∧ ρ' well = 0

/-- Re-anchored splits denote the same plinth profile. -/
theorem plinthLift_congr_reanchor {well : ℕ} {ρ ρ' : ℕ → ℕ} {c c' : ℕ}
    (h : ReanchorsTo well ρ c ρ' c') :
    plinthLift well c ρ = plinthLift well c' ρ' := by
  funext j
  by_cases hj : j = well
  · subst hj; simp [plinthLift]
  · rw [plinthLift_ne well c ρ hj, plinthLift_ne well c' ρ' hj]
    have := h.1 j hj
    omega

/-- **The floor is immortal.** A plinth board has no full rows: the well
column stops at height 1 (blocking every row ≥ 1) and the entombed hole
blocks row 0. -/
theorem fullRows_plinth_eq_empty {cfg : GameConfig} {w hx c : ℕ}
    {ρ : ℕ → ℕ} (hw : w < cfg.cols) (hxw : hx ≠ w) (hxc : hx < cfg.cols) :
    fullRows cfg (debtBoard cfg (plinthLift w c ρ) (some (hx, 0))) = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro r hr
  unfold fullRows at hr
  rw [Finset.mem_filter] at hr
  have hfull := hr.2
  by_cases hr0 : r = 0
  · subst hr0
    have hmem := hfull hx (Finset.mem_range.mpr hxc)
    rw [mem_debtBoard] at hmem
    exact (hmem.1 (hx, 0) rfl) rfl
  · have hmem := hfull w (Finset.mem_range.mpr hw)
    rw [mem_debtBoard] at hmem
    have hlt := hmem.2.2
    simp only [plinthLift_well] at hlt
    omega

/-- The well-plug shape: `J` rot 3 — one cell in its left column at row 0,
three cells in the right column rows 0–2. -/
theorem shapeUp_wellPlugJ (c : ℕ) :
    ({ piece := Piece.J, rot := 3, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (1, 1), (1, 2)} := by
  show Piece.shapeUp Piece.J 3 = _
  decide

/-- **The well plug**: `J` rot 3 straddling the well boundary on flat ground
— exactly one cell lands at `(w, base)` (the plug), three seat flush on the
neighbor. No hole is created. The bag-1 entombment move (finding D2). -/
theorem place_wellPlug_flat (cfg : GameConfig) (base w : ℕ)
    (hw1 : w + 1 < cfg.cols) :
    ({ piece := Piece.J, rot := 3, col := w } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j =>
          if j = w then base + 1 else if j = w + 1 then base + 3 else base) := by
  have hsh := shapeUp_wellPlugJ w
  have hc0 : w + 0 < cfg.cols := by omega
  have hd : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hw1]
    omega
  have hdr : ({ piece := Piece.J, rot := 3, col := w } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(w, base), (w + 1, base), (w + 1, base + 1), (w + 1, base + 2)} := by
    rw [Placement.dropped_eq_image, hsh, hd]
    simp only [Finset.image_insert, Finset.image_singleton]
    norm_num
  rw [Placement.place_eq_union_dropped, hdr]
  ext ⟨a, b⟩
  simp only [Finset.mem_union, mem_skyline', Finset.mem_insert,
    Finset.mem_singleton, Prod.mk.injEq]
  split_ifs <;> omega

end Board
end Tetris
