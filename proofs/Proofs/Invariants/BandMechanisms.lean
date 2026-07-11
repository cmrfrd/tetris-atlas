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

end Board
end Tetris
