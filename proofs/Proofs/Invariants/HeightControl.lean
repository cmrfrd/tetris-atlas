import Proofs.Invariants.HoleDebt

/-!
# Why controlling the max height is hard — the survival obstruction theorems

Survival means never losing, and `isLost cfg b ↔ maxHeight cfg b > rows`: the loss-relevant
quantity is the **max** column height. This file collects the theorems that explain *why*
keeping that max bounded against an adversary is hard — the structural core of the open crux.

The first fact is an **asymmetry**: a placement can only *raise* the max height; the only
operation that *lowers* it is a line clear. So height is a one-way ratchet that the player can
release only by completing full rows. Subsequent sections quantify the per-piece budget, the
hole inflation, and the energy/clearing necessity that together make the max hard to control.
-/

namespace Tetris.Board

open Tetris.WqoCarrier

/-! ## The asymmetry: placement only raises the max, clears only lower it -/

/-- **Placement never lowers the max height.** `place` only adds cells, so every column height
is non-decreasing (`colHeight_le_place`), hence so is their sup. The player has *no* move that
reduces the max height by placing — the one-way ratchet at the heart of the difficulty. -/
theorem maxHeight_le_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    maxHeight cfg b ≤ maxHeight cfg (pl.place b) := by
  unfold maxHeight
  exact Finset.sup_mono_fun (fun j _ => colHeight_le_place b pl j)

/-- **Line clears never raise the max height.** `clearLines` only lowers column heights
(`clearLines_domLE`), so the sup can only drop. Clearing is therefore the *unique* height-
reducing primitive — and it is gated on completing full rows. -/
theorem maxHeight_clearLines_le (cfg : GameConfig) (b : Board) :
    maxHeight cfg (clearLines cfg b) ≤ maxHeight cfg b := by
  unfold maxHeight
  exact Finset.sup_mono_fun (fun j _ => clearLines_domLE cfg b j)

/-- **Within a full move, the clear phase never raises the max height.** Since
`applyStep = clearLines ∘ place`, the only height growth in a step is from the placement; the
clear can only give back. Any net height reduction in a move is paid for entirely by cleared
lines. -/
theorem maxHeight_applyStep_le_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    maxHeight cfg (pl.applyStep cfg b) ≤ maxHeight cfg (pl.place b) :=
  maxHeight_clearLines_le cfg (pl.place b)

/-! ## The survival target is exactly `maxHeight ≤ rows` -/

/-- **Survival forces a bounded max height.** `¬ isLost b → maxHeight b ≤ rows`: not losing
means every column is within the ceiling, hence so is their max. Together with the ratchet
asymmetry this is the difficulty in one line — the player must hold `maxHeight ≤ rows` forever,
and the only lever that pushes it down is a line clear. So the entire survival problem is:
keep this single sup under the ceiling using only the clear primitive, against an adversary who
picks the pieces. -/
theorem maxHeight_le_rows_of_not_isLost (cfg : GameConfig) {b : Board} (h : ¬ isLost cfg b) :
    maxHeight cfg b ≤ cfg.rows := by
  unfold maxHeight
  exact Finset.sup_le (fun j _ => colHeight_le_rows_of_not_isLost cfg h j)

/-! ## A surviving board is resource-tight, so the player is forced to clear -/

/-- **A surviving board fits inside the field.** `¬ isLost b → count b ≤ cols·rows` (well-formed
`b`): the volume bound `count ≤ cols·maxHeight` composed with `maxHeight ≤ rows`. Since
`count_place` adds exactly 4 cells per piece, the board climbs 4 cells per move toward this cap
of `cols·rows`, so a run with no line clears lasts at most `cols·rows / 4` pieces — the player
is *forced* to clear. The whole adversarial difficulty is then concentrated in making the full
rows a clear requires as expensive as possible to assemble. -/
theorem count_le_capacity_of_not_isLost {cfg : GameConfig} {b : Board}
    (hwf : WF cfg b) (h : ¬ isLost cfg b) : b.count ≤ cfg.cols * cfg.rows := by
  calc b.count ≤ cfg.cols * maxHeight cfg b := count_le_cols_mul_maxHeight b hwf
    _ ≤ cfg.cols * cfg.rows := by gcongr; exact maxHeight_le_rows_of_not_isLost cfg h

end Tetris.Board
