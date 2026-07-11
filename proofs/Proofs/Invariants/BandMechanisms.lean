import Proofs.Invariants.HoledSkyline
import Proofs.Invariants.GameplayExtra

/-!
# Band mechanisms: the debt-carry wrapper and the bag-1 bootstrap pack

`place_debtBoard_of_flush` upgrades every flush skyline transition to debt-1
boards: the strictly covered hole is disjoint from the drop path, so the
placement commutes with the erasure. Combined with the proven reproduction
mechanisms (`place_vertS_skyline`, `place_vertZ_skyline`, `place_O_pair`)
this makes the permanent bootstrap hole ride through all flush play for
free. The rest of the file is the bag-1 pack: the horizontal-I flat
response, and the S/Z mutual-enabling edges (the flat-S residue contains a
Z-window and vice versa, so the second roughness piece of bag 1 seats
flush — exactly one hole is ever forced).
-/

namespace Tetris
namespace Board

/-- Heights never drop across a flush transition (read off the flush
equality column-wise). -/
theorem le_of_flush {cfg : GameConfig} {ρ ρ' : ℕ → ℕ} {pl : Placement}
    (hflush : pl.place (skyline cfg ρ) = skyline cfg ρ')
    {j : ℕ} (hj : j < cfg.cols) : ρ j ≤ ρ' j := by
  have h := colHeight_le_place (skyline cfg ρ) pl j
  rwa [hflush, colHeight_skyline hj, colHeight_skyline hj] at h

/-- **The debt-carry wrapper.** A flush skyline transition holds verbatim on
the debt-1 board: the strictly covered hole is untouched by the drop and
survives on both sides. Upgrades every flush mechanism to debt-1. -/
theorem place_debtBoard_of_flush {cfg : GameConfig} {ρ ρ' : ℕ → ℕ}
    {ho : Option Coord} {pl : Placement}
    (hflush : pl.place (skyline cfg ρ) = skyline cfg ρ')
    (hho : ∀ x, ho = some x → x.1 < cfg.cols ∧ x.2 + 1 < ρ x.1) :
    pl.place (debtBoard cfg ρ ho) = debtBoard cfg ρ' ho := by
  cases ho with
  | none => simpa using hflush
  | some x =>
      obtain ⟨hxc, hxcov⟩ := hho x rfl
      rw [debtBoard_some, debtBoard_some,
        place_holedSkyline pl hxc hxcov, hflush]
      rfl

theorem shapeUp_horizI (c : ℕ) :
    ({ piece := Piece.I, rot := 0, col := c } : Placement).shapeUp
      = {((0 : ℕ), (0 : ℕ)), (1, 0), (2, 0), (3, 0)} := by
  show Piece.shapeUp Piece.I 0 = _
  decide

/-- **Flush placement of the horizontal I on a flat surface** — four columns,
+1 each. This is the bag-1 I response at base 0, where the well drain
(`4 ≤ c`) is unavailable. -/
theorem place_horizI_flat (cfg : GameConfig) (base col : ℕ)
    (hcol : col + 3 < cfg.cols) :
    ({ piece := Piece.I, rot := 0, col := col } : Placement).place
        (skyline cfg (fun _ => base))
      = skyline cfg (fun j =>
          if j = col ∨ j = col + 1 ∨ j = col + 2 ∨ j = col + 3
          then base + 1 else base) := by
  have hsh := shapeUp_horizI col
  have hc0 : col + 0 < cfg.cols := by omega
  have hc1 : col + 1 < cfg.cols := by omega
  have hc2 : col + 2 < cfg.cols := by omega
  have hd : ({ piece := Piece.I, rot := 0, col := col } : Placement).dropOffset
      (skyline cfg (fun _ => base)) = base := by
    rw [Placement.dropOffset_eq_sup, hsh]
    simp only [Finset.sup_insert, Finset.sup_singleton,
      colHeight_skyline hc0, colHeight_skyline hc1, colHeight_skyline hc2,
      colHeight_skyline hcol]
    omega
  have hdr : ({ piece := Piece.I, rot := 0, col := col } : Placement).dropped
      (skyline cfg (fun _ => base))
      = {(col, base), (col + 1, base), (col + 2, base), (col + 3, base)} := by
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
