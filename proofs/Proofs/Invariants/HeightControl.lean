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

/-! ## Near capacity, clearing is forced -/

/-- **A surviving move near capacity must clear a line.** If a well-formed board is within 4
cells of the field capacity (`cols·rows < count + 4`) and a valid move keeps it alive, then that
move cleared at least one line. Proof: the cell ledger `applyStep_count`
(`count' + cols·linesCleared = count + 4`) with no clear gives `count' = count + 4 > cols·rows`,
contradicting `count_le_capacity_of_not_isLost`. So the player cannot merely stack near the top —
survival *requires* completing full rows there, exactly where the board is most constrained.
This is the bind the adversary exploits: force the stack up, then make the needed full rows hard
to assemble. -/
theorem must_clear_near_capacity {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg)
    (hfull : cfg.cols * cfg.rows < b.count + 4)
    (hsurvive : ¬ isLost cfg (pl.applyStep cfg b)) :
    0 < Board.linesCleared cfg (pl.place b) := by
  by_contra hzero
  have hz : Board.linesCleared cfg (pl.place b) = 0 := by omega
  have hcount := applyStep_count cfg b pl hwf hv
  rw [hz, Nat.mul_zero, Nat.add_zero] at hcount
  have hcap := count_le_capacity_of_not_isLost (Placement.applyStep_wf hwf hv) hsurvive
  omega

/-! ## Holes block clearing — the adversary's roughness obstructs the one height lever -/

/-- **A hole's row is not full.** A buried empty cell `p ∈ holes` leaves its row `p.2` incomplete
(`p ∉ b`, in range), so that row cannot be a full row. The S/Z pieces manufacture exactly such
holes, and a clear needs the *whole* row — so adversarial roughness disables the player's sole
height-reducing primitive at every row a hole touches. -/
theorem not_isFull_of_mem_holes {cfg : GameConfig} {b : Board} {p : Coord}
    (hp : p ∈ HoleyCarrier.holes cfg b) : ¬ isFull cfg b p.2 := by
  rw [SurfaceFiber.mem_holes_iff] at hp
  exact not_isFull_of_notMem (Finset.mem_range.mp (Finset.mem_product.mp hp.1).1) hp.2.1

/-- **A hole's row is never cleared.** Strengthening: the row of a hole is not even in `fullRows`,
so `clearLines` leaves it — and the hole — in place. A buried hole is removable only by first
clearing rows *above* it (lowering the column past the hole), never directly: holes are sticky
debt that the player must dig out from the top. -/
theorem notMem_fullRows_of_mem_holes {cfg : GameConfig} {b : Board} {p : Coord}
    (hp : p ∈ HoleyCarrier.holes cfg b) : p.2 ∉ fullRows cfg b := by
  rw [SurfaceFiber.mem_holes_iff] at hp
  exact notMem_fullRows_of_notMem (Finset.mem_range.mp (Finset.mem_product.mp hp.1).1) hp.2.1

/-! ## Energy and material do not control survival — a single cell at the brink -/

/-- **One cell can sit at the loss boundary.** On a nonempty field the one-cell board
`{(0, rows-1)}` has `count = 1` yet `maxHeight = rows`: a lone cell floating at the top (which
buried holes beneath it permit) is one move from a top-out. Survival is governed by the *position
of the highest cell*, not the amount of material — `cols·rows` capacity gives no safety if a
single cell sits at the ceiling. So the `count`/`surfaceArea` budgets (here tiny: energy `= rows`,
far below `cols·rows`) cannot certify safety; the loss metric `maxHeight` is decoupled from them,
exactly the slack in `maxHeight ≤ surfaceArea ≤ cols·maxHeight`. This is *why* a Lyapunov function
on a sum/budget alone is provably insufficient — the adversary aims a single cell at the top. -/
theorem exists_one_cell_at_brink {cfg : GameConfig}
    (hcols : 0 < cfg.cols) (hrows : 0 < cfg.rows) :
    ∃ b : Board, WF cfg b ∧ b.count = 1 ∧ maxHeight cfg b = cfg.rows := by
  refine ⟨{(0, cfg.rows - 1)}, ?_, ?_, ?_⟩
  · intro p hp
    rw [Finset.mem_singleton] at hp; subst hp
    exact hcols
  · exact Finset.card_singleton _
  · apply le_antisymm
    · apply maxHeight_le_rows_of_not_isLost
      rw [not_isLost_iff_forall_row_lt]
      intro p hp
      rw [Finset.mem_singleton] at hp; subst hp
      exact Nat.sub_lt hrows Nat.one_pos
    · have hlt := Board.lt_colHeight (Finset.mem_singleton_self ((0 : ℕ), cfg.rows - 1))
      exact le_trans (by omega) (colHeight_le_maxHeight hcols)

/-! ## Clearing is a coordinated, expensive operation -/

/-- **A clearable row costs a full width of cells.** A full row contains at least `cols` cells
(one in every column). So the player's only height-reducing primitive requires *coordinating* a
complete `cols`-wide layer, whereas each piece deposits only 4 cells anywhere it lands — a clear
needs the contributions of `≥ cols/4` pieces aimed at one row, and the adversary can keep
scattering those contributions across different rows (and burying holes, crux 6) to prevent any
single row from completing. The asymmetry "4 local cells in, a coordinated `cols`-wide row out"
is the engine of the difficulty. -/
theorem cols_le_card_row_of_isFull {cfg : GameConfig} {b : Board} {r : ℕ}
    (h : isFull cfg b r) : cfg.cols ≤ (b.filter (fun p => p.2 = r)).card := by
  have hsub : (Finset.range cfg.cols).image (fun c => (c, r)) ⊆ b.filter (fun p => p.2 = r) := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨c, hc, rfl⟩ := hp
    rw [Finset.mem_filter]
    exact ⟨h c hc, rfl⟩
  calc cfg.cols
      = ((Finset.range cfg.cols).image (fun c => (c, r))).card := by
        rw [Finset.card_image_of_injective _ (fun a b hab => by simpa using hab), Finset.card_range]
    _ ≤ (b.filter (fun p => p.2 = r)).card := Finset.card_le_card hsub

end Tetris.Board
