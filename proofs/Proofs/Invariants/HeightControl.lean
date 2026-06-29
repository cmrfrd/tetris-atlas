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

/-! ## The per-piece height budget: the ratchet climbs by at most 4 -/

/-- The dropped piece is at most 4 tall above where it rests. Every dropped cell sits at row
`dropOffset + c.2` with `c.2 < 4` (`shapeUp_row_lt_four`), so the column it fills reaches at most
`dropOffset + 4`. -/
theorem colHeight_dropped_le (b : Board) (pl : Placement) (j : ℕ) :
    (pl.dropped b).colHeight j ≤ pl.dropOffset b + 4 := by
  unfold Placement.dropped Board.colHeight
  apply Finset.sup_le
  intro r hr
  rw [Board.colRows, Finset.mem_image] at hr
  obtain ⟨x, hxf, hxr⟩ := hr
  rw [Finset.mem_filter] at hxf
  obtain ⟨hxmem, _⟩ := hxf
  rw [Placement.cellsAt, Finset.mem_image] at hxmem
  obtain ⟨c, hc, hcx⟩ := hxmem
  have hx2 : x.2 = pl.dropOffset b + c.2 := by rw [← hcx]
  have hcr : c.2 < 4 := Piece.shapeUp_row_lt_four pl.piece pl.rot c hc
  have hr_eq : r = pl.dropOffset b + c.2 := by rw [← hxr, hx2]
  change r + 1 ≤ pl.dropOffset b + 4
  omega

/-- A valid piece rests no higher than the tallest column: `dropOffset ≤ maxHeight`. Each drop
candidate `colHeight (col+c.1) − c.2 ≤ colHeight (col+c.1) ≤ maxHeight` (the column is in range
by `Valid`), so their sup is bounded by `maxHeight`. -/
theorem dropOffset_le_maxHeight {cfg : GameConfig} (b : Board) {pl : Placement}
    (hv : pl.Valid cfg) : pl.dropOffset b ≤ maxHeight cfg b := by
  unfold Placement.dropOffset
  apply Finset.sup_le
  intro c hc
  calc b.colHeight (pl.col + c.1) - c.2 ≤ b.colHeight (pl.col + c.1) := Nat.sub_le _ _
    _ ≤ maxHeight cfg b := colHeight_le_maxHeight (hv c hc)

/-- **The ratchet climbs by at most 4 per piece.** A valid placement raises the max height by at
most 4: `maxHeight (place b) ≤ maxHeight b + 4`. Each column of `place b = b ∪ dropped` is the max
of the old height (`≤ maxHeight`) and the dropped contribution (`≤ dropOffset + 4 ≤ maxHeight +
4`). So from a board of max height `H` the player has at least `(rows − H)/4` moves before a
forced top-out *if it never clears* — and clearing is the only way to buy more. -/
theorem maxHeight_place_le_add_four {cfg : GameConfig} (b : Board) {pl : Placement}
    (hv : pl.Valid cfg) : maxHeight cfg (pl.place b) ≤ maxHeight cfg b + 4 := by
  apply Finset.sup_le
  intro j hj
  rw [Placement.place_eq_union_dropped, Board.colHeight_union]
  apply max_le
  · exact le_trans (colHeight_le_maxHeight (Finset.mem_range.mp hj)) (Nat.le_add_right _ _)
  · calc (pl.dropped b).colHeight j ≤ pl.dropOffset b + 4 := colHeight_dropped_le b pl j
      _ ≤ maxHeight cfg b + 4 := by gcongr; exact dropOffset_le_maxHeight b hv

/-- **A full move raises the max height by at most 4.** `maxHeight (applyStep b) ≤ maxHeight b +
4`: the placement climbs by ≤4 and the clear phase only gives back. So the per-move height budget
is a hard `+4` ceiling — the rate at which the irreversible ratchet can advance. -/
theorem maxHeight_applyStep_le_add_four {cfg : GameConfig} (b : Board) {pl : Placement}
    (hv : pl.Valid cfg) : maxHeight cfg (pl.applyStep cfg b) ≤ maxHeight cfg b + 4 :=
  le_trans (maxHeight_applyStep_le_place cfg b pl) (maxHeight_place_le_add_four b hv)

end Tetris.Board
