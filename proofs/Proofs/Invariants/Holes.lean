import Mathlib
import Proofs.Invariants.ColumnCount

/-!
# Column holes

A **hole** is a buried empty cell: an empty cell sitting strictly below the
topmost filled cell of its column. This file defines holes column-by-column
(`colHoles`) and board-wide (`holes`), and proves the general algebra that any
hole argument rests on:

* **Decomposition** — `colRows_card_add_colHoles`: a column's filled cells plus
  its holes equal its height. This is the conservation law underlying every
  holes argument; holes are exactly the height not accounted for by filled
  cells.
* **Bounds** — `colRows_card_le_colHeight`, `colHoles_le_colHeight`,
  `holes_le_sum_colHeight`: holes are dominated by the geometric stack profile.
* **Zero-characterizations** — `colHoles_eq_zero_iff`, `holes_eq_zero_iff`,
  `holes_pos_iff`: a column/board is hole-free exactly when it is gravity-packed.

These are general-purpose facts about the holes count; piece-specific
hole-creation witnesses (S/Z overhangs, tuck/overhang dichotomies, adversarial
caps) live with the experiment that uses them.
-/

namespace Tetris
namespace Board

/-- Buried cells in column `j`: empty cells strictly below the topmost
filled cell of column `j`. Zero for empty columns. -/
def colHoles (b : Board) (j : ℕ) : ℕ :=
  b.colHeight j - (b.colRows j).card

@[simp] theorem colHoles_empty (j : ℕ) : colHoles (∅ : Board) j = 0 := by
  simp [colHoles]

/-- Total holes across `cfg.cols` columns. -/
def holes (cfg : GameConfig) (b : Board) : ℕ :=
  ∑ j ∈ Finset.range cfg.cols, colHoles b j

@[simp] theorem holes_empty (cfg : GameConfig) : holes cfg ∅ = 0 := by
  simp [holes]

/-- A column whose filled rows all lie strictly above row 0 contributes at
least one hole — the empty row 0 sits below the topmost filled cell. -/
theorem one_le_colHoles_of_zero_notMem_of_mem_pos
    {b : Board} {j h : ℕ} (hpos : 0 < h)
    (hmem : (j, h) ∈ b) (hzero : (j, 0) ∉ b) :
    1 ≤ colHoles b j := by
  classical
  -- The filled rows of column `j` are a subset of `Finset.range (b.colHeight j)`
  -- (every filled cell sits strictly below its column's height) but skip row 0.
  have hsubset : b.colRows j ⊆ (Finset.range (b.colHeight j)).erase 0 := by
    intro r hr
    rcases Finset.mem_image.mp hr with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol
    subst heq
    refine Finset.mem_erase.mpr ⟨?_, ?_⟩
    · intro hr0
      subst hr0
      exact hzero hcell
    · exact Finset.mem_range.mpr (Board.lt_colHeight hcell)
  have hcardLe : (b.colRows j).card ≤ ((Finset.range (b.colHeight j)).erase 0).card :=
    Finset.card_le_card hsubset
  -- The column has a cell at row `h > 0`, so its height is ≥ h + 1 ≥ 2.
  have hheight : 2 ≤ b.colHeight j := by
    have : h + 1 ≤ b.colHeight j := by
      have := Board.lt_colHeight hmem
      omega
    omega
  -- erase 0 from a range of size ≥ 2 cuts the cardinality by exactly 1.
  have herasecard : ((Finset.range (b.colHeight j)).erase 0).card =
      b.colHeight j - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_range.mpr (by omega))]
    rw [Finset.card_range]
  rw [herasecard] at hcardLe
  unfold colHoles
  omega

/-- **Filled rows fit under the column height.** The number of filled cells in column `j` is at
most the column height `b.colHeight j`, since every filled cell of column `j` sits at a row
strictly below the column height (`Board.lt_colHeight`). The filled rows therefore embed into
`Finset.range (b.colHeight j)`. -/
theorem colRows_card_le_colHeight (b : Board) (j : ℕ) :
    (b.colRows j).card ≤ b.colHeight j := by
  classical
  have hsubset : b.colRows j ⊆ Finset.range (b.colHeight j) := by
    intro r hr
    rcases Finset.mem_image.mp hr with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol
    subst heq
    exact Finset.mem_range.mpr (Board.lt_colHeight hcell)
  calc (b.colRows j).card ≤ (Finset.range (b.colHeight j)).card := Finset.card_le_card hsubset
    _ = b.colHeight j := Finset.card_range _

/-- **Per-column hole/height decomposition.** A column's filled rows split its height into the
filled cells plus the buried (hole) cells: `(b.colRows j).card + colHoles b j = b.colHeight j`.
The key arithmetic fact is `colRows_card_le_colHeight` (every filled cell sits strictly below the
column height), which turns the truncated subtraction in `colHoles` into an exact complement. This
is the conservation law underlying every holes argument — holes are exactly the height not
accounted for by filled cells. -/
theorem colRows_card_add_colHoles (b : Board) (j : ℕ) :
    (b.colRows j).card + colHoles b j = b.colHeight j := by
  unfold colHoles
  have hle : (b.colRows j).card ≤ b.colHeight j := colRows_card_le_colHeight b j
  omega

/-- **Holes never exceed column height.** `colHoles b j ≤ b.colHeight j` — buried cells are a
subset of the column's vertical extent. Immediate from `colRows_card_add_colHoles`. -/
theorem colHoles_le_colHeight (b : Board) (j : ℕ) :
    colHoles b j ≤ b.colHeight j := by
  unfold colHoles
  exact Nat.sub_le _ _

/-- **Board-level holes bounded by total stack height.** Total holes across the playfield are at
most the sum of the column heights: `holes cfg b ≤ ∑ⱼ b.colHeight j`. Summing the per-column bound
`colHoles_le_colHeight`. A clean envelope linking the hole count to the geometric profile — useful
for any `height + holes` Lyapunov candidate, which needs holes controlled by height. -/
theorem holes_le_sum_colHeight (cfg : GameConfig) (b : Board) :
    holes cfg b ≤ ∑ j ∈ Finset.range cfg.cols, b.colHeight j := by
  unfold holes
  exact Finset.sum_le_sum (fun j _ => colHoles_le_colHeight b j)

/-- **A column is hole-free iff its filled cells fill its height.** `colHoles b j = 0` exactly when
the filled-cell count equals the column height — i.e. the stack in column `j` is gravity-packed with
no buried gap. Immediate from the decomposition `colRows_card_add_colHoles`. -/
theorem colHoles_eq_zero_iff (b : Board) (j : ℕ) :
    colHoles b j = 0 ↔ (b.colRows j).card = b.colHeight j := by
  have h := colRows_card_add_colHoles b j
  omega

/-- **A board is hole-free iff every column is.** `holes cfg b = 0` exactly when no column has a
buried cell. The pointwise characterization of the holes-0 universe used by the impossibility
theorems (`no_init_closed_atlas_in_holes_zero_universe`): a board lies in that universe iff each of
its `cfg.cols` columns is gravity-packed. From `Finset.sum_eq_zero_iff` on `holes = ∑ colHoles`. -/
theorem holes_eq_zero_iff (cfg : GameConfig) (b : Board) :
    holes cfg b = 0 ↔ ∀ j ∈ Finset.range cfg.cols, colHoles b j = 0 := by
  unfold holes
  exact Finset.sum_eq_zero_iff

/-- **Board-level fill/hole/height conservation.** Summing the per-column decomposition over the
playfield: total filled cells (counted column-wise) plus total holes equals total stack height,
`(∑ⱼ (b.colRows j).card) + holes cfg b = ∑ⱼ b.colHeight j`. The global conservation law: the stack
volume bounded by the column tops is exactly the filled cells plus the buried gaps. -/
theorem sum_colRows_card_add_holes (cfg : GameConfig) (b : Board) :
    (∑ j ∈ Finset.range cfg.cols, (b.colRows j).card) + holes cfg b
      = ∑ j ∈ Finset.range cfg.cols, b.colHeight j := by
  unfold holes
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun j _ => colRows_card_add_colHoles b j)

/-- **Holes positivity is witnessed by a single buried column.** `0 < holes cfg b` exactly when some
column in the playfield has a buried cell, `∃ j < cfg.cols, 0 < colHoles b j`. The contrapositive of
`holes_eq_zero_iff`: a board has *some* hole iff it is not gravity-packed in at least one column.
Lets hole-creation witnesses (e.g. `one_le_holes_S_place_empty`) certify global holes-positivity. -/
theorem holes_pos_iff (cfg : GameConfig) (b : Board) :
    0 < holes cfg b ↔ ∃ j ∈ Finset.range cfg.cols, 0 < colHoles b j := by
  rw [Nat.pos_iff_ne_zero, ne_eq, holes_eq_zero_iff]
  push Not
  simp only [Nat.pos_iff_ne_zero, ne_eq]

end Board
end Tetris
