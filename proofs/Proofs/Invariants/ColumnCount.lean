import Mathlib
import Proofs.Invariants.GameplayExtra

/-!
# Per-column cell counts

The total cell count obeys `+4` per placement and `-cols` per cleared line
(`applyStep_count`). This file refines that law to **each column**: a placement
adds the piece's per-column profile, and every line clear removes exactly
**one** cell from every valid column — a full row holds exactly one cell in
each column `j < cols`, and line-clear gravity moves cells straight down,
never sideways.

Headline: `applyStep_colCount`, the per-column analogue of `applyStep_count`:

```
(applyStep cfg b pl).colCount j + linesCleared cfg (pl.place b)
    = b.colCount j + pl.colProfile j        (j < cfg.cols)
```

Unlike the total-count law this needs *no* well-formedness hypothesis.
-/

namespace Tetris
namespace Board

/-- Number of filled cells in column `j`. -/
def colCount (b : Board) (j : ℕ) : ℕ := (b.filter (fun p => p.1 = j)).card

@[simp] theorem colCount_empty (j : ℕ) : colCount (∅ : Board) j = 0 := by
  simp [colCount]

/-- A column never holds more cells than the whole board. -/
theorem colCount_le_count (b : Board) (j : ℕ) : b.colCount j ≤ b.count :=
  Finset.card_filter_le b _

/-- On a well-formed board the per-column counts sum to the total count. -/
theorem sum_colCount {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    ∑ j ∈ Finset.range cfg.cols, b.colCount j = b.count := by
  unfold colCount count
  exact (Finset.card_eq_sum_card_fiberwise (fun p hp => Finset.mem_range.2 (hb p hp))).symm

/-- **Each line clear removes exactly one cell from every valid column.**
Column `j < cols` of a full row holds exactly one cell, and gravity relocates
the surviving cells straight down (injectively, by `row_map_lt`). -/
theorem colCount_clearLines_add (cfg : GameConfig) (b : Board) {j : ℕ}
    (hj : j < cfg.cols) :
    b.colCount j = (clearLines cfg b).colCount j + (fullRows cfg b).card := by
  classical
  -- Split column `j` into full-row cells and surviving cells.
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := b.filter (fun p => p.1 = j)) (p := fun p => isFull cfg b p.2)
  -- Full-row part: exactly one cell per full row.
  have hfull_eq : (b.filter (fun p => p.1 = j)).filter (fun p => isFull cfg b p.2)
      = (fullRows cfg b).image (fun r => (j, r)) := by
    ext p
    simp only [Finset.mem_filter, Finset.mem_image, fullRows]
    constructor
    · rintro ⟨⟨hpb, hpj⟩, hpfull⟩
      exact ⟨p.2, ⟨⟨p, hpb, rfl⟩, hpfull⟩, Prod.ext hpj.symm rfl⟩
    · rintro ⟨r, ⟨⟨q, hqb, hqr⟩, hrfull⟩, rfl⟩
      exact ⟨⟨hrfull j (Finset.mem_range.2 hj), rfl⟩, hrfull⟩
  have hfull_card : ((b.filter (fun p => p.1 = j)).filter
      (fun p => isFull cfg b p.2)).card = (fullRows cfg b).card := by
    rw [hfull_eq]
    exact Finset.card_image_of_injective _ (fun r1 r2 h => by simpa using h)
  -- Surviving part: gravity within a column is injective.
  have hnonfull_eq : (clearLines cfg b).filter (fun p => p.1 = j)
      = ((b.filter (fun p => p.1 = j)).filter (fun p => ¬ isFull cfg b p.2)).image
          (fun p => (p.1, p.2 - clearedBelow cfg b p.2)) := by
    ext p
    simp only [clearLines, Finset.mem_filter, Finset.mem_image]
    constructor
    · rintro ⟨⟨q, ⟨hqb, hqnf⟩, rfl⟩, hpj⟩
      exact ⟨q, ⟨⟨hqb, hpj⟩, hqnf⟩, rfl⟩
    · rintro ⟨q, ⟨⟨hqb, hqj⟩, hqnf⟩, rfl⟩
      exact ⟨⟨q, ⟨hqb, hqnf⟩, rfl⟩, hqj⟩
  have hnonfull_card : ((clearLines cfg b).filter (fun p => p.1 = j)).card
      = ((b.filter (fun p => p.1 = j)).filter (fun p => ¬ isFull cfg b p.2)).card := by
    rw [hnonfull_eq]
    apply Finset.card_image_of_injOn
    intro p hp q hq hpq
    simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
    simp only [Prod.mk.injEq] at hpq
    obtain ⟨h1, h2⟩ := hpq
    have heq2 : p.2 = q.2 := row_map_eq_of_not_full hp.2 hq.2 h2
    exact Prod.ext_iff.mpr ⟨h1, heq2⟩
  unfold colCount
  omega

/-! ### Per-row cell counts (`rowCount`)

The row-indexed analogue of `colCount`: `rowCount b r` is the number of filled
cells in row `r`. A full row holds exactly `cols` cells, so per-row under-fill
(`rowCount b r < cols`) blocks a line clear — the per-row gate that fires at
arbitrarily large *total* counts, unlike `clearLines_eq_self_of_count_lt`. -/

/-- Number of filled cells in row `r` (the per-row analogue of `Board.colCount`). -/
def rowCount (b : Board) (r : ℕ) : ℕ := (b.filter (fun p => p.2 = r)).card

@[simp] theorem rowCount_empty (r : ℕ) : rowCount (∅ : Board) r = 0 := by
  simp [rowCount]

/-- **Per-row under-fill blocks a clear: a row holding fewer than `cols` cells is not full.**
A full row carries one cell in every column `c < cols`, so the injection `c ↦ (c, r)` embeds
`range cols` into row `r`'s cell-fiber, forcing `rowCount b r ≥ cols`. Contrapositive:
`rowCount b r < cols ⇒ ¬ isFull`. This is the per-row gate the *total*-count argument lacks:
`clearLines_eq_self_of_count_lt` needs the whole board under `cols`, but a clear is actually
governed by per-*row* fill, so this fires at arbitrarily large total counts as long as each row
stays under-full. -/
theorem not_isFull_of_rowCount_lt (cfg : GameConfig) (b : Board) (r : ℕ)
    (h : rowCount b r < cfg.cols) : ¬ Board.isFull cfg b r := by
  intro hfull
  have hsub : (Finset.range cfg.cols).image (fun c => ((c, r) : ℕ × ℕ))
      ⊆ b.filter (fun p => p.2 = r) := by
    intro p hp
    rw [Finset.mem_image] at hp
    obtain ⟨c, hc, rfl⟩ := hp
    rw [Finset.mem_range] at hc
    rw [Finset.mem_filter]
    exact ⟨Board.mem_of_isFull cfg hc hfull, rfl⟩
  have hinj : Function.Injective (fun c => ((c, r) : ℕ × ℕ)) := by
    intro a b hab; simpa using hab
  have hcard : cfg.cols ≤ rowCount b r := by
    unfold rowCount
    have h1 : ((Finset.range cfg.cols).image (fun c => ((c, r) : ℕ × ℕ))).card
        ≤ (b.filter (fun p => p.2 = r)).card := Finset.card_le_card hsub
    rwa [Finset.card_image_of_injective _ hinj, Finset.card_range] at h1
  omega

/-- **Uniformly under-full board has no full rows.** If *every* row holds fewer than `cols`
cells then `fullRows = ∅`, so (via `clearLines_eq_self_of_no_fullRows`) `clearLines` is the
identity. The per-row replacement for `clearLines_eq_self_of_count_lt` that survives past the
total-count wall `count < cols` (the obstruction that shut the no-clear floor ladder at
`count = 8`): a board can carry many cells yet clear nothing as long as no single row fills. -/
theorem fullRows_eq_empty_of_forall_rowCount_lt (cfg : GameConfig) (b : Board)
    (h : ∀ r, rowCount b r < cfg.cols) : Board.fullRows cfg b = ∅ := by
  rcases Finset.eq_empty_or_nonempty (Board.fullRows cfg b) with he | hne
  · exact he
  · exfalso
    obtain ⟨r, hr⟩ := hne
    exact not_isFull_of_rowCount_lt cfg b r (h r) (Board.isFull_of_mem_fullRows hr)

/-- **Per-row placement bound: a single placement adds at most 4 cells to any row.**
A tetromino is `4` cells, so its dropped image contributes `≤ 4` to row `r` at *any* drop height
(the per-row profile is drop-dependent, unlike `colProfile`, but the crude `≤ 4` cap needs no
profile — just `dropped.card ≤ shapeUp.card = 4`). The per-row analogue of `count_place`'s `+4`,
restricted to one row. Composed with `fullRows_eq_empty_of_forall_rowCount_lt`, this is the engine
for "no clear at depth `k`": as long as each row stays under `cols`, the board never clears no
matter how large the *total* count grows — exactly the regime the total-count wall
(`clearLines_eq_self_of_count_lt`, shut at `count = 8`) cannot reach. -/
theorem rowCount_place_le (b : Board) (pl : Placement) (r : ℕ) :
    rowCount (pl.place b) r ≤ rowCount b r + 4 := by
  unfold rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union]
  refine le_trans (Finset.card_union_le _ _) ?_
  have hd : ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 4 := by
    rw [Placement.dropped_eq_image]
    refine le_trans (Finset.card_filter_le _ _) ?_
    exact le_trans Finset.card_image_le (le_of_eq pl.shapeUp_card)
  omega

end Board

namespace Placement

/-- Number of cells the piece contributes to column `j`. Independent of the
drop height: a hard drop is purely vertical. -/
def colProfile (pl : Placement) (j : ℕ) : ℕ :=
  (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j)).card

/-- The translated piece occupies `colProfile j` cells of column `j`, at any
drop height. -/
theorem colCount_cellsAt (pl : Placement) (d j : ℕ) :
    (pl.cellsAt d).colCount j = pl.colProfile j := by
  have himg : (pl.cellsAt d).filter (fun p => p.1 = j)
      = (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j)).image
          (fun cell => (pl.col + cell.1, d + cell.2)) := by
    ext p
    simp only [cellsAt, Finset.mem_filter, Finset.mem_image]
    constructor
    · rintro ⟨⟨cell, hcell, rfl⟩, hpj⟩
      exact ⟨cell, ⟨hcell, hpj⟩, rfl⟩
    · rintro ⟨cell, ⟨hcell, hpj⟩, rfl⟩
      exact ⟨⟨cell, hcell, rfl⟩, hpj⟩
  unfold Board.colCount colProfile
  rw [himg, Finset.card_image_of_injective _ (cellsAt_injective pl d)]

/-- **Placing a piece adds its column profile to every column.** -/
theorem colCount_place (b : Board) (pl : Placement) (j : ℕ) :
    (pl.place b).colCount j = b.colCount j + pl.colProfile j := by
  have hdisj : Disjoint (b.filter (fun p => p.1 = j))
      ((pl.dropped b).filter (fun p => p.1 = j)) :=
    Finset.disjoint_filter_filter (pl.dropped_disjoint b).symm
  have hdrop : (pl.dropped b).colCount j = pl.colProfile j :=
    colCount_cellsAt pl (pl.dropOffset b) j
  unfold Board.colCount at hdrop ⊢
  unfold place
  rw [Finset.filter_union, Finset.card_union_of_disjoint hdisj, hdrop]

/-- A valid placement's column profile sums to 4 over the board's columns. -/
theorem sum_colProfile {cfg : GameConfig} {pl : Placement} (hv : pl.Valid cfg) :
    ∑ j ∈ Finset.range cfg.cols, pl.colProfile j = 4 := by
  unfold colProfile
  rw [← Finset.card_eq_sum_card_fiberwise
    (fun cell hcell => Finset.mem_range.2 (hv cell hcell))]
  exact pl.shapeUp_card

end Placement

/-- **Per-column cell-count law for a full move**: a move adds the piece's
column profile to column `j` and removes one cell per cleared line. The
per-column analogue of `applyStep_count`, with no well-formedness hypothesis. -/
theorem applyStep_colCount (cfg : GameConfig) (b : Board) (pl : Placement)
    {j : ℕ} (hj : j < cfg.cols) :
    (Placement.applyStep cfg b pl).colCount j
      + Board.linesCleared cfg (pl.place b)
      = b.colCount j + pl.colProfile j := by
  have h1 := Board.colCount_clearLines_add cfg (pl.place b) hj
  have h2 := Placement.colCount_place b pl j
  unfold Placement.applyStep Board.linesCleared
  omega

end Tetris
