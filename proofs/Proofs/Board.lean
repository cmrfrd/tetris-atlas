import Mathlib
import Proofs.Config

/-!
# Boards

A board is the finite set of filled cells, each a `(column, row)` pair with
`row 0` the bottom (a clean mathematical model rather than the Rust bit-column
representation). Columns/rows are plain `ℕ`; the configured dimensions enter
only through predicates (`isLost`, `isFull`, well-formedness), which keeps the
hard-drop and line-clear arithmetic free of `Fin` bookkeeping.
-/

namespace Tetris

/-- A board coordinate: `(column, row)`, with `row 0` the bottom. -/
abbrev Coord := ℕ × ℕ

/-- A board is the finite set of filled cells. -/
abbrev Board := Finset Coord

namespace Board

/-- The number of filled cells. -/
def count (b : Board) : ℕ := b.card

/-- The empty board. -/
def empty : Board := (∅ : Finset Coord)

/-- Fill a single cell. -/
def set (b : Board) (c : Coord) : Board := insert c b

/-- Clear a single cell. -/
def clear (b : Board) (c : Coord) : Board := b.erase c

/-- The cells of row `r` across the first `cols` columns. -/
def row (cols : ℕ) (r : ℕ) : Board := (Finset.range cols).image (fun c => (c, r))

/-- The filled rows in column `j`. -/
def colRows (b : Board) (j : ℕ) : Finset ℕ := (b.filter (·.1 = j)).image (·.2)

/-- Stack height of column `j`: `0` if empty, else one above the highest filled
row. Every filled cell in the column sits strictly below this height. -/
def colHeight (b : Board) (j : ℕ) : ℕ := (b.colRows j).sup (· + 1)

/-- Every filled cell sits strictly below its column's height. -/
theorem lt_colHeight {b : Board} {j r : ℕ} (h : (j, r) ∈ b) : r < b.colHeight j := by
  have hr : r ∈ b.colRows j := by
    simp only [colRows, Finset.mem_image, Finset.mem_filter]
    exact ⟨(j, r), ⟨h, rfl⟩, rfl⟩
  have h1 : r + 1 ≤ b.colHeight j := Finset.le_sup (f := (· + 1)) hr
  omega

/-- A board is lost when any filled cell reaches or exceeds row `rows`
(i.e. spills above the visible field). Mirrors the Rust `is_lost` check. -/
def isLost (cfg : GameConfig) (b : Board) : Prop := ∃ p ∈ b, cfg.rows ≤ p.2

instance (cfg : GameConfig) (b : Board) : Decidable (isLost cfg b) := by
  unfold isLost; infer_instance

/-- **Foundational characterisation of "not lost"**: a board is non-lost iff
every filled cell sits strictly below row `cfg.rows`, i.e. inside the visible
playing field. This is the structural bridge between the gameplay loss
predicate (`isLost`) and the in-field universe used to enumerate states
(`InFieldBoard.prop.2`). -/
theorem not_isLost_iff_forall_row_lt (cfg : GameConfig) (b : Board) :
    ¬ isLost cfg b ↔ ∀ p ∈ b, p.2 < cfg.rows := by
  unfold isLost
  constructor
  · intro h p hp
    by_contra hlt
    exact h ⟨p, hp, Nat.not_lt.mp hlt⟩
  · rintro h ⟨p, hp, hle⟩
    exact (Nat.not_lt.mpr hle) (h p hp)

/-- Row `r` is full when every column `< cols` is filled in that row. -/
def isFull (cfg : GameConfig) (b : Board) (r : ℕ) : Prop :=
  ∀ c ∈ Finset.range cfg.cols, (c, r) ∈ b

instance (cfg : GameConfig) (b : Board) (r : ℕ) : Decidable (isFull cfg b r) := by
  unfold isFull; infer_instance

/-- The set of completely filled rows. -/
def fullRows (cfg : GameConfig) (b : Board) : Finset ℕ :=
  (b.image (·.2)).filter (fun r => isFull cfg b r)

/-- Number of full rows strictly below row `r`. -/
def clearedBelow (cfg : GameConfig) (b : Board) (r : ℕ) : ℕ :=
  ((fullRows cfg b).filter (· < r)).card

/-- Apply line clears: drop every surviving cell down by the number of cleared
rows below it (gravity). -/
def clearLines (cfg : GameConfig) (b : Board) : Board :=
  (b.filter (fun p => ¬ isFull cfg b p.2)).image
    (fun p => (p.1, p.2 - clearedBelow cfg b p.2))

/-- Well-formed: every filled cell lies in a valid column. -/
def WF (cfg : GameConfig) (b : Board) : Prop := ∀ p ∈ b, p.1 < cfg.cols

/-- `clearLines` preserves well-formedness: cells are only filtered out or
shifted down, never moved horizontally. -/
theorem clearLines_wf {cfg : GameConfig} {b : Board} (h : WF cfg b) :
    WF cfg (clearLines cfg b) := by
  intro p hp
  unfold clearLines at hp
  rw [Finset.mem_image] at hp
  obtain ⟨q, hq, rfl⟩ := hp
  exact h q (Finset.mem_filter.mp hq).1

/-- The empty board is vacuously WF (and in-field). -/
theorem empty_wf (cfg : GameConfig) : WF cfg (∅ : Board) := by
  intro _ hp; exact absurd hp (Finset.notMem_empty _)

/-- The empty board has height 0 in every column. -/
@[simp] theorem colHeight_empty (j : ℕ) : colHeight (∅ : Board) j = 0 := by
  unfold colHeight colRows
  simp

/-- General column-height bound: if all cells are below `cfg.rows`, every
column's height is at most `cfg.rows`. Generalises `safe_colHeight_le`. -/
theorem colHeight_le_rows_of_in_field {cfg : GameConfig} {b : Board}
    (h : ∀ p ∈ b, p.2 < cfg.rows) (j : ℕ) :
    colHeight b j ≤ cfg.rows := by
  unfold colHeight
  apply Finset.sup_le
  intro r hr
  unfold colRows at hr
  rw [Finset.mem_image] at hr
  obtain ⟨p, hp, hrp⟩ := hr
  rw [Finset.mem_filter] at hp
  have hover := h p hp.1
  rw [hrp] at hover
  omega

/-- **Height bound rules out loss.** If every column height is at most
`cfg.rows`, the board is not lost: every filled cell sits strictly below its
column height (`lt_colHeight`), hence strictly below `cfg.rows`. The clean
converse-direction companion to `colHeight_le_rows_of_in_field`, letting any
height-bounded invariant discharge the `¬ isLost` obligation directly. -/
theorem not_isLost_of_colHeight_le {cfg : GameConfig} {b : Board}
    (h : ∀ j, colHeight b j ≤ cfg.rows) : ¬ isLost cfg b := by
  rw [not_isLost_iff_forall_row_lt]
  rintro ⟨j, r⟩ hp
  have hlt := lt_colHeight hp
  have hle := h j
  omega

/-- An in-field board with `cols * rows` cells is the full grid. -/
theorem eq_full_grid_of_card_eq {cfg : GameConfig} {b : Board}
    (hwf : WF cfg b) (hif : ∀ p ∈ b, p.2 < cfg.rows)
    (hcard : b.card = cfg.cols * cfg.rows) :
    b = Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
  have hsub : b ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product, Finset.mem_range, Finset.mem_range]
    exact ⟨hwf p hp, hif p hp⟩
  apply Finset.eq_of_subset_of_card_le hsub
  rw [Finset.card_product, Finset.card_range, Finset.card_range]
  omega

end Board
end Tetris
