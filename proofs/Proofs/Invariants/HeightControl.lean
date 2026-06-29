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

/-- **A hole is buried under a filled cell.** Every hole `p` has a filled cell strictly above it in
its column (`∃ r > p.2, (p.1, r) ∈ b`): the column's top cell. So a hole cannot be reached from
above — to fill or expose it the player must first clear the cell(s) covering it, which means
clearing rows *above* the hole. Holes are sticky debt that can only be dug out from the top, never
patched directly; this is the dynamical content of `clearLines_holes_le_false` (clears can even
create holes). -/
theorem exists_cover_of_hole {cfg : GameConfig} {b : Board} {p : Coord}
    (hp : p ∈ HoleyCarrier.holes cfg b) : ∃ r, p.2 < r ∧ (p.1, r) ∈ b := by
  rw [SurfaceFiber.mem_holes_iff] at hp
  obtain ⟨_, hpnb, hplt⟩ := hp
  have hpos : 0 < b.colHeight p.1 := Nat.lt_of_le_of_lt (Nat.zero_le _) hplt
  obtain ⟨r0, hr0⟩ := (colHeight_pos_iff_exists_mem b p.1).mp hpos
  have hne : (b.colRows p.1).Nonempty := by
    refine ⟨r0, ?_⟩
    rw [Board.colRows, Finset.mem_image]
    exact ⟨(p.1, r0), Finset.mem_filter.mpr ⟨hr0, rfl⟩, rfl⟩
  obtain ⟨r, hrmem, hrsup⟩ := Finset.exists_mem_eq_sup (b.colRows p.1) hne (· + 1)
  have hmem : (p.1, r) ∈ b := by
    rw [Board.colRows, Finset.mem_image] at hrmem
    obtain ⟨x, hx, hxr⟩ := hrmem
    rw [Finset.mem_filter] at hx
    exact (Prod.ext hx.2 hxr : x = (p.1, r)) ▸ hx.1
  have hcol : b.colHeight p.1 = r + 1 := hrsup
  refine ⟨r, ?_, hmem⟩
  rcases Nat.lt_or_ge p.2 r with h | h
  · exact h
  · exfalso
    have hpeq : p.2 = r := by omega
    have hpe : p = (p.1, r) := Prod.ext rfl hpeq
    rw [hpe] at hpnb
    exact hpnb hmem

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

/-- Every row of the full rectangle is full (hence clearable): `(c, r)` is present for all
in-range `c`, `r`. The full board is hole-free — the opposite extreme from a board of holes. -/
theorem isFull_full_board {cfg : GameConfig} {r : ℕ} (hr : r < cfg.rows) :
    isFull cfg (Finset.range cfg.cols ×ˢ Finset.range cfg.rows) r := by
  intro c hc
  exact Finset.mem_product.mpr ⟨hc, Finset.mem_range.mpr hr⟩

/-- **A completely full board reaches the boundary too — but is the SAFE extreme.** The full
rectangle has `count = cols·rows` (maximal material) and `maxHeight = rows`, yet — unlike the
one-cell board — it is hole-free, so *every* row is full and clearable (`isFull_full_board`): a
piece placed on top leaves rows `0…rows-1` complete, so the move clears them all and the board
survives. Two lessons. (a) `maxHeight = rows` is reached at *every* material level from 1 cell
(`exists_one_cell_at_brink`) to `cols·rows`, so the loss metric is decoupled from `count`/energy.
(b) At that same boundary the hole-free board is *safe* while the holey one-cell board is
*precarious* — danger is governed by `maxHeight` **together with** clearability (holes), neither
alone. So even tracking `maxHeight` is insufficient; a survival invariant must also see the hole
structure — exactly the two-axis difficulty `HoleyCarrier`'s non-congruence refutations expose. -/
theorem exists_full_board_at_brink {cfg : GameConfig}
    (hcols : 0 < cfg.cols) (hrows : 0 < cfg.rows) :
    ∃ b : Board, WF cfg b ∧ b.count = cfg.cols * cfg.rows ∧
      ¬ isLost cfg b ∧ maxHeight cfg b = cfg.rows := by
  refine ⟨Finset.range cfg.cols ×ˢ Finset.range cfg.rows, ?_, ?_, ?_, ?_⟩
  · intro p hp
    exact Finset.mem_range.mp (Finset.mem_product.mp hp).1
  · unfold Board.count
    rw [Finset.card_product, Finset.card_range, Finset.card_range]
  · rw [not_isLost_iff_forall_row_lt]
    intro p hp
    exact Finset.mem_range.mp (Finset.mem_product.mp hp).2
  · apply le_antisymm
    · apply maxHeight_le_rows_of_not_isLost
      rw [not_isLost_iff_forall_row_lt]
      intro p hp
      exact Finset.mem_range.mp (Finset.mem_product.mp hp).2
    · have hmem : ((0 : ℕ), cfg.rows - 1) ∈ Finset.range cfg.cols ×ˢ Finset.range cfg.rows :=
        Finset.mem_product.mpr ⟨Finset.mem_range.mpr hcols, Finset.mem_range.mpr (by omega)⟩
      have hlt := Board.lt_colHeight hmem
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

/-! ## The death zone: any cell at row ≥ rows is an immediate top-out -/

/-- **Reaching row `rows` is death.** The playfield is rows `0 … rows-1`; a single cell at any
row `r ≥ rows` makes its column overflow (`colHeight > rows`), so the board is lost. This is the
target the irreversible +4-per-piece ratchet drives every column toward — and (crux 1) the only
way to retreat is a clear, which (crux 6, crux 8) the adversary obstructs by burying holes and
scattering the cells a full row needs. The whole game is a race between the ratchet pushing the
top cell into row `rows` and the player assembling clears to pull it back. -/
theorem isLost_of_mem_row_ge {cfg : GameConfig} {b : Board} {j r : ℕ}
    (hr : cfg.rows ≤ r) (hmem : (j, r) ∈ b) : isLost cfg b := by
  by_contra hcon
  rw [not_isLost_iff_forall_colHeight_le] at hcon
  have h1 := hcon j
  have h2 := Board.lt_colHeight hmem
  omega

/-- **The survival threshold is exactly `maxHeight ≤ rows`.** The converse of
`maxHeight_le_rows_of_not_isLost`: for a well-formed board, `maxHeight ≤ rows → ¬ isLost`. In-range
columns are `≤ maxHeight ≤ rows`; out-of-range columns are empty by `WF`. So (together with the
forward direction) `¬ isLost b ↔ maxHeight b ≤ rows` for WF boards: the single scalar `maxHeight` is
*precisely* the loss predicate. The entire game is keeping this one number under the ceiling — yet
(crux 5–18) the only lever to lower it is an obstructed, coordinated clear, and (crux 13) no
additive budget tracks it. The danger is one number; controlling it is the whole problem. -/
theorem not_isLost_of_maxHeight_le {cfg : GameConfig} {b : Board}
    (hwf : WF cfg b) (h : maxHeight cfg b ≤ cfg.rows) : ¬ isLost cfg b := by
  rw [not_isLost_iff_forall_colHeight_le]
  intro j
  rcases Nat.lt_or_ge j cfg.cols with hj | hj
  · exact le_trans (colHeight_le_maxHeight hj) h
  · have hz : b.colHeight j = 0 := by
      rw [colHeight_eq_zero_iff_forall_not_mem]
      intro r hr
      exact absurd (hwf (j, r) hr) (by omega)
    omega

/-! ## The per-move material speed limit -/

/-- **Cells grow by at most 4 per move.** `count (applyStep b) ≤ count b + 4`: a piece deposits 4
cells and clears only remove. The material analogue of the +4 height ceiling
(`maxHeight_applyStep_le_add_four`) — both the stack height and the cell count advance by a hard
`+4` per move, so the board's danger budgets tick up at the same bounded rate. -/
theorem count_applyStep_le_add_four {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg) : (pl.applyStep cfg b).count ≤ b.count + 4 := by
  have := applyStep_count cfg b pl hwf hv; omega

/-- **A single move cannot outrun the accumulation.** `cols·linesCleared ≤ count + 4`: the cells
one move removes are bounded by those then present (plus the new 4). So clearing only ever chips
away — the player can never erase the stack in one move, only balance the steady +4 inflow over a
long horizon. This is why survival is an unbounded balancing act, not a finite puzzle: there is no
single move that resets the danger. -/
theorem cols_mul_linesCleared_le {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg) :
    cfg.cols * Board.linesCleared cfg (pl.place b) ≤ b.count + 4 := by
  have := applyStep_count cfg b pl hwf hv; omega

/-- **Only clearing makes real progress.** A move that clears at least one line strictly beats the
`+4` cell gain: `count (applyStep b) < count b + 4`. So the steady material inflow is reversed only
on the moves that clear — and (`cols_le_card_row_of_isFull`) those require assembling a full
`cols`-wide row the adversary fights to prevent. -/
theorem count_applyStep_lt_of_clear {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg) (hcols : 0 < cfg.cols)
    (hclear : 0 < Board.linesCleared cfg (pl.place b)) :
    (pl.applyStep cfg b).count < b.count + 4 := by
  have hstep := applyStep_count cfg b pl hwf hv
  have hge : cfg.cols ≤ cfg.cols * Board.linesCleared cfg (pl.place b) :=
    Nat.le_mul_of_pos_right _ hclear
  omega

/-- **No clear means strict accumulation.** A move that clears no line adds exactly 4 cells:
`count (applyStep b) = count b + 4`. So between clears the board climbs monotonically by 4 toward
the `cols·rows` cap (`count_le_capacity_of_not_isLost`) — a deterministic countdown to a forced
clear (`must_clear_near_capacity`). The player has no way to *stall*; every non-clearing move spends
irreplaceable headroom. -/
theorem count_applyStep_eq_of_no_clear {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg) (h0 : Board.linesCleared cfg (pl.place b) = 0) :
    (pl.applyStep cfg b).count = b.count + 4 := by
  have hstep := applyStep_count cfg b pl hwf hv
  rw [h0, Nat.mul_zero] at hstep
  omega

/-- **Recovery is bounded: even a Tetris removes only `4·cols` cells.** From a settled board (no
pending full rows), one move clears at most 4 lines (`linesCleared_place_le_four`, since a piece
spans ≤4 rows — and only the I-piece reaches 4), so it removes at most `4·cols` cells:
`count b + 4 ≤ count (applyStep b) + 4·cols`. The player cannot erase an arbitrary accumulated
deficit in a single move; survival is necessarily *incremental*, chipping at most a Tetris per
move against a steady `+4` inflow — and a Tetris itself needs four full rows the adversary fights
to deny. -/
theorem count_le_count_applyStep_add {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : WF cfg b) (hv : pl.Valid cfg) (hnf : ∀ r, ¬ isFull cfg b r) :
    b.count + 4 ≤ (pl.applyStep cfg b).count + cfg.cols * 4 := by
  have hstep := applyStep_count cfg b pl hwf hv
  have hle : Board.linesCleared cfg (pl.place b) ≤ 4 := linesCleared_place_le_four cfg b pl hnf
  have hmul : cfg.cols * Board.linesCleared cfg (pl.place b) ≤ cfg.cols * 4 := by gcongr
  omega

/-! ## The adversary's concrete weapon: S/Z roughness manufactures unclearable holes -/

/-- **S on flat ground buries a hole from nothing.** Dropping the S-piece (rotation 0, column 0)
onto the empty standard board leaves `(2, 0)` a buried empty — a hole created on flat ground, with
no prior roughness. This is the adversary's lever: the 7-bag hands it two such roughness pieces
(S and Z) every bag, each able to plant a hole the player did not choose. -/
theorem S_buries_hole :
    ((2 : ℕ), (0 : ℕ)) ∈ HoleyCarrier.holes GameConfig.standard
      (Placement.place ∅ ⟨Piece.S, 0, 0⟩) := by decide

/-- **The buried hole blocks its row.** The hole S plants makes row `0` of the result permanently
unfull (`not_isFull_of_mem_holes` applied to `S_buries_hole`), so that row cannot be cleared while
the hole stands — the adversary converts one piece into a standing obstruction to the player's only
height-reducing primitive. -/
theorem S_hole_row_not_isFull :
    ¬ isFull GameConfig.standard (Placement.place ∅ ⟨Piece.S, 0, 0⟩) 0 :=
  not_isFull_of_mem_holes S_buries_hole

/-- **Z too buries a hole on flat ground.** The mirror roughness piece behaves identically: a Z
dropped on the empty board also creates a buried empty. So *both* of the 7-bag's two-per-bag
roughness pieces (`BagBurst.countP_isSZ = 2`) are hole-injectors — the adversary is guaranteed two
forced hole-plantings every bag, against the player's single guaranteed I-drain. The roughness
budget exceeds the drain budget in count; that the drains can still keep up is a question of
*geometry*, not budget — the open crux. -/
theorem Z_buries_hole :
    0 < (HoleyCarrier.holes GameConfig.standard (Placement.place ∅ ⟨Piece.Z, 0, 0⟩)).card := by
  decide

/-! ## Synthesis — why the crux is hard, assembled from the above

The survival problem is exactly to hold `maxHeight ≤ rows` forever
(`maxHeight_le_rows_of_not_isLost`, `isLost_of_mem_row_ge`). The theorems above show why that is
hard against a piece-picking adversary:

1. **One-way ratchet.** Placement only *raises* the max (`maxHeight_le_place`); the *sole* way to
   lower it is a line clear (`maxHeight_clearLines_le`). Height is irreversible except by clearing.
2. **Bounded but relentless climb.** Each move pushes the ceiling up by a hard `+4`
   (`maxHeight_applyStep_le_add_four`) and the cell count likewise (`count_applyStep_le_add_four`);
   no single move resets the danger (`cols_mul_linesCleared_le`).
3. **Clearing is forced.** A surviving board fits the field (`count_le_capacity_of_not_isLost`), so
   near capacity the player *must* clear (`must_clear_near_capacity`) — stacking is not an option.
4. **Clearing is obstructed.** A clear needs a coordinated full `cols`-wide row
   (`cols_le_card_row_of_isFull`), and the adversary's S/Z holes make their rows permanently
   unclearable (`not_isFull_of_mem_holes`, `notMem_fullRows_of_mem_holes`); only clearing makes
   real progress (`count_applyStep_lt_of_clear`).
5. **The danger is decoupled from every budget — and is two-axis.** `maxHeight ≤ surfaceArea ≤
   cols·maxHeight` (in `HoleDebt`) and the witnesses `exists_one_cell_at_brink` /
   `exists_full_board_at_brink` show `maxHeight = rows` occurs at *any* material/energy, so no
   additive (sum) potential tracks the loss metric. Worse, at that same boundary the hole-free full
   board is *safe* (every row clears, `isFull_full_board`) while the holey one-cell board is
   *precarious* — danger needs `maxHeight` **and** clearability (holes). So even a max-height
   potential is insufficient: a survival invariant must jointly track the height envelope and the
   hole structure, and (`HoleyCarrier`) holes are non-congruent under the dynamics.

Together: the player is forced to keep clearing a height that only clears can lower, while the
adversary buries holes and scatters fills to deny the coordinated full rows clearing needs — and
no single scalar (height, energy, count) tracks the joint height-and-hole danger. That is the
crux: it demands a two-axis, non-congruent invariant, which is why every scalar-potential route in
this project floored. -/

end Tetris.Board
