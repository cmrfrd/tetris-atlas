import Mathlib
import Proofs.Model.Board

/-!
# Board cell-count theorems

Basic facts about the number of filled cells on a board.
-/

namespace Tetris
namespace Board

/-- The empty board has no filled cells. -/
@[simp] theorem count_empty : (empty).count = 0 := by
  simp [count, empty]

/-- Filling an empty cell increases the count by one. -/
theorem count_set_of_not_filled (b : Board) (c : Coord) (h : c ∉ b) :
    (b.set c).count = b.count + 1 := by
  simp [count, set, Finset.card_insert_of_notMem h]

/-- Clearing a filled cell decreases the count by one. -/
theorem count_clear_of_filled (b : Board) (c : Coord) (h : c ∈ b) :
    (b.clear c).count = b.count - 1 := by
  simp [count, clear, Finset.card_erase_of_mem h]

/-- A full row contains exactly `cols` filled cells. -/
theorem count_row (cols r : ℕ) : (Board.row cols r).count = cols := by
  unfold Board.row Board.count
  rw [Finset.card_image_of_injective _ (by intro a b h; simpa using h)]
  simp

/-- A board has zero count iff it equals the empty board. -/
theorem count_eq_zero_iff_eq_empty (b : Board) : b.count = 0 ↔ b = ∅ :=
  Finset.card_eq_zero

/-- A board has positive count iff it is nonempty. -/
theorem count_pos_iff_nonempty (b : Board) : 0 < b.count ↔ b.Nonempty :=
  Finset.card_pos

/-- A board has nonzero count iff it is nonempty. -/
theorem count_ne_zero_iff_nonempty (b : Board) : b.count ≠ 0 ↔ b.Nonempty := by
  rw [← Nat.pos_iff_ne_zero, count_pos_iff_nonempty]

/-- A board has nonzero count iff it is not the empty board. -/
theorem count_ne_zero_iff_ne_empty (b : Board) : b.count ≠ 0 ↔ b ≠ ∅ := by
  rw [Ne, count_eq_zero_iff_eq_empty]

/-- A board has positive count iff it is not the empty board. -/
theorem count_pos_iff_ne_empty (b : Board) : 0 < b.count ↔ b ≠ ∅ := by
  rw [Nat.pos_iff_ne_zero, count_ne_zero_iff_ne_empty]

/-- A row of width 0 has count 0. -/
theorem count_row_zero (r : ℕ) : (Board.row 0 r).count = 0 := by
  rw [count_row]

/-- A row's count is positive iff its width is positive. -/
theorem count_row_pos_iff_cols_pos (cols r : ℕ) :
    0 < (Board.row cols r).count ↔ 0 < cols := by
  rw [count_row]

/-- A row equals empty iff its width is zero. -/
theorem row_eq_empty_iff_cols_eq_zero (cols r : ℕ) :
    Board.row cols r = ∅ ↔ cols = 0 := by
  rw [← count_eq_zero_iff_eq_empty, count_row]

/-- A row is nonempty iff its width is positive. -/
theorem row_nonempty_iff_cols_pos (cols r : ℕ) :
    (Board.row cols r).Nonempty ↔ 0 < cols := by
  rw [← count_pos_iff_nonempty, count_row]

/-- A row's count is `≠ 0` iff its width is positive. -/
theorem count_row_ne_zero_iff_cols_pos (cols r : ℕ) :
    (Board.row cols r).count ≠ 0 ↔ 0 < cols := by
  rw [count_row, Nat.pos_iff_ne_zero]

/-- A row of zero width is empty. -/
theorem row_zero_eq_empty (r : ℕ) : Board.row 0 r = ∅ := by
  rw [row_eq_empty_iff_cols_eq_zero]

/-- A row's count is `≤ cols` (trivially since `= cols`). -/
theorem count_row_le (cols r : ℕ) : (Board.row cols r).count ≤ cols := by
  rw [count_row]

/-- A row's count is `≥ cols` (trivially since `= cols`). -/
theorem count_row_ge (cols r : ℕ) : cols ≤ (Board.row cols r).count := by
  rw [count_row]

/-- A row's count is `< cols + 1`. -/
theorem count_row_lt_succ (cols r : ℕ) : (Board.row cols r).count < cols + 1 := by
  rw [count_row]; omega

/-- A row's count interval: `[cols, cols]` (degenerate). -/
theorem count_row_interval (cols r : ℕ) :
    cols ≤ (Board.row cols r).count ∧ (Board.row cols r).count ≤ cols :=
  ⟨count_row_ge cols r, count_row_le cols r⟩

/-- After setting a not-yet-filled cell, the count is positive. -/
theorem count_set_pos (b : Board) (c : Coord) (h : c ∉ b) :
    0 < (b.set c).count := by
  rw [count_set_of_not_filled b c h]; omega

/-- After setting a not-yet-filled cell, the board is nonempty. -/
theorem set_nonempty (b : Board) (c : Coord) (h : c ∉ b) :
    (b.set c).Nonempty :=
  Finset.card_pos.mp (count_set_pos b c h)

/-- After setting a not-yet-filled cell, the board is not empty. -/
theorem set_ne_empty (b : Board) (c : Coord) (h : c ∉ b) :
    b.set c ≠ ∅ :=
  (set_nonempty b c h).ne_empty

/-- After setting a not-yet-filled cell, `1 ≤ (b.set c).count`. -/
theorem count_set_ge_one (b : Board) (c : Coord) (h : c ∉ b) :
    1 ≤ (b.set c).count :=
  count_set_pos b c h

/-- After setting a not-yet-filled cell, `(b.set c).count ≠ 0`. -/
theorem count_set_ne_zero (b : Board) (c : Coord) (h : c ∉ b) :
    (b.set c).count ≠ 0 :=
  Nat.ne_of_gt (count_set_pos b c h)

/-- After setting a not-yet-filled cell, the count grew by exactly 1. -/
theorem count_set_sub_count (b : Board) (c : Coord) (h : c ∉ b) :
    (b.set c).count - b.count = 1 := by
  rw [count_set_of_not_filled b c h]; omega

/-- After setting a not-yet-filled cell, the prior count is `< (b.set c).count`. -/
theorem count_lt_count_set (b : Board) (c : Coord) (h : c ∉ b) :
    b.count < (b.set c).count := by
  rw [count_set_of_not_filled b c h]; omega

/-- After clearing a filled cell, the new count is `< b.count`. -/
theorem count_clear_lt_count (b : Board) (c : Coord) (h : c ∈ b) :
    (b.clear c).count < b.count := by
  rw [count_clear_of_filled b c h]
  exact Nat.sub_lt (Finset.card_pos.mpr ⟨c, h⟩) Nat.one_pos

/-- After clearing a filled cell, the count drops by exactly 1. -/
theorem count_sub_count_clear (b : Board) (c : Coord) (h : c ∈ b) :
    b.count - (b.clear c).count = 1 := by
  rw [count_clear_of_filled b c h]
  exact Nat.sub_sub_self (Finset.card_pos.mpr ⟨c, h⟩)

/-- Clearing any cell never increases the count. -/
theorem count_clear_le_count (b : Board) (c : Coord) :
    (b.clear c).count ≤ b.count :=
  Finset.card_erase_le

/-- Setting any cell never decreases the count. -/
theorem count_le_count_set (b : Board) (c : Coord) :
    b.count ≤ (b.set c).count :=
  Finset.card_le_card (Finset.subset_insert _ _)

/-- `b.clear c ⊆ b`: clearing only removes cells. -/
theorem clear_subset (b : Board) (c : Coord) : b.clear c ⊆ b :=
  Finset.erase_subset _ _

/-- `b ⊆ b.set c`: setting only adds cells. -/
theorem subset_set (b : Board) (c : Coord) : b ⊆ b.set c :=
  Finset.subset_insert _ _

/-- Membership in `b.set c` is `= c` or already in `b`. -/
theorem mem_set (b : Board) (c d : Coord) : d ∈ b.set c ↔ d = c ∨ d ∈ b :=
  Finset.mem_insert

/-- Membership in `b.clear c` is "in `b` and not `= c`". -/
theorem mem_clear (b : Board) (c d : Coord) : d ∈ b.clear c ↔ d ≠ c ∧ d ∈ b :=
  Finset.mem_erase

/-- The cell just set is in the resulting board. -/
theorem mem_set_self (b : Board) (c : Coord) : c ∈ b.set c :=
  Finset.mem_insert_self _ _

/-- The cell just cleared is gone from the resulting board. -/
theorem not_mem_clear_self (b : Board) (c : Coord) : c ∉ b.clear c := by
  simp [clear]

/-- Setting an already-filled cell is a no-op. -/
theorem set_of_mem (b : Board) (c : Coord) (h : c ∈ b) : b.set c = b :=
  Finset.insert_eq_self.mpr h

/-- Clearing a not-filled cell is a no-op. -/
theorem clear_of_not_mem (b : Board) (c : Coord) (h : c ∉ b) : b.clear c = b :=
  Finset.erase_eq_of_notMem h

/-- Setting an already-filled cell preserves the count. -/
theorem count_set_of_mem (b : Board) (c : Coord) (h : c ∈ b) :
    (b.set c).count = b.count := by
  rw [set_of_mem b c h]

/-- Clearing a not-filled cell preserves the count. -/
theorem count_clear_of_not_mem (b : Board) (c : Coord) (h : c ∉ b) :
    (b.clear c).count = b.count := by
  rw [clear_of_not_mem b c h]

/-- Clear after set on the same cell agrees with clear alone. -/
theorem clear_set (b : Board) (c : Coord) : (b.set c).clear c = b.clear c := by
  simp [set, clear, Finset.erase_insert_eq_erase]

/-- Set after clear on the same cell agrees with set alone. -/
theorem set_clear (b : Board) (c : Coord) : (b.clear c).set c = b.set c := by
  ext d
  simp [set, clear, Finset.mem_insert, Finset.mem_erase]
  by_cases h : d = c
  · simp [h]
  · simp [h]

/-- Setting the same cell twice is the same as setting it once. -/
theorem set_set (b : Board) (c : Coord) : (b.set c).set c = b.set c :=
  set_of_mem _ c (mem_set_self b c)

/-- Clearing the same cell twice is the same as clearing it once. -/
theorem clear_clear (b : Board) (c : Coord) : (b.clear c).clear c = b.clear c :=
  clear_of_not_mem _ c (not_mem_clear_self b c)

/-- Setting two cells commutes. -/
theorem set_comm (b : Board) (c d : Coord) :
    (b.set c).set d = (b.set d).set c := by
  ext x; simp [set, Finset.mem_insert]; tauto

/-- Clearing two cells commutes. -/
theorem clear_comm (b : Board) (c d : Coord) :
    (b.clear c).clear d = (b.clear d).clear c := by
  ext x; simp [clear, Finset.mem_erase]; tauto

/-- Setting a cell into the empty board yields the singleton `{c}`. -/
@[simp] theorem set_empty (c : Coord) : (Board.empty).set c = {c} := by
  simp [empty, set]

/-- Clearing any cell from the empty board leaves it empty. -/
@[simp] theorem clear_empty (c : Coord) : (Board.empty).clear c = Board.empty := by
  simp [empty, clear]

/-- The count of a singleton-set board is exactly 1. -/
theorem count_set_empty (c : Coord) : (Board.empty.set c).count = 1 := by
  rw [set_empty]; simp [count]

/-- `b.set c` is never empty. -/
theorem set_ne_empty_unconditional (b : Board) (c : Coord) : b.set c ≠ ∅ :=
  (Finset.insert_nonempty c b).ne_empty

/-- `b.set c` is always nonempty. -/
theorem set_nonempty_unconditional (b : Board) (c : Coord) : (b.set c).Nonempty :=
  Finset.insert_nonempty c b

/-- `0 < (b.set c).count` unconditionally. -/
theorem count_set_pos_unconditional (b : Board) (c : Coord) :
    0 < (b.set c).count :=
  Finset.card_pos.mpr (set_nonempty_unconditional b c)

/-- `(b.set c).count ≤ b.count + 1` unconditionally. -/
theorem count_set_le_succ (b : Board) (c : Coord) :
    (b.set c).count ≤ b.count + 1 := by
  change (insert c b).card ≤ b.card + 1
  exact Finset.card_insert_le _ _

/-- `b.count ≤ (b.clear c).count + 1` unconditionally. -/
theorem count_le_count_clear_succ (b : Board) (c : Coord) :
    b.count ≤ (b.clear c).count + 1 := by
  rcases Decidable.em (c ∈ b) with h | h
  · rw [count_clear_of_filled b c h]
    have hp : 1 ≤ b.count := Finset.card_pos.mpr ⟨c, h⟩
    omega
  · rw [count_clear_of_not_mem b c h]; omega

/-- The empty board has empty `colRows` in every column. -/
@[simp] theorem colRows_empty (j : ℕ) : (∅ : Board).colRows j = ∅ := by
  simp [colRows]

/-- Board.empty has empty `colRows` in every column. -/
theorem colRows_Board_empty (j : ℕ) : (Board.empty).colRows j = ∅ := by
  simp [empty]

/-- Board.empty has height 0 in every column. -/
theorem colHeight_Board_empty (j : ℕ) : (Board.empty).colHeight j = 0 := by
  simp [empty]

/-- `colRows` is monotone under board subset. -/
theorem colRows_subset_of_subset {b₁ b₂ : Board} (h : b₁ ⊆ b₂) (j : ℕ) :
    b₁.colRows j ⊆ b₂.colRows j := by
  intro r hr
  simp only [colRows, Finset.mem_image, Finset.mem_filter] at hr ⊢
  obtain ⟨p, ⟨hp, hpj⟩, hpr⟩ := hr
  exact ⟨p, ⟨h hp, hpj⟩, hpr⟩

/-- `colHeight` is monotone under board subset. -/
theorem colHeight_le_of_subset {b₁ b₂ : Board} (h : b₁ ⊆ b₂) (j : ℕ) :
    b₁.colHeight j ≤ b₂.colHeight j := by
  unfold colHeight
  exact Finset.sup_mono (colRows_subset_of_subset h j)

/-- Clearing a cell never raises any column height. -/
theorem colHeight_clear_le (b : Board) (c : Coord) (j : ℕ) :
    (b.clear c).colHeight j ≤ b.colHeight j :=
  colHeight_le_of_subset (clear_subset b c) j

/-- Setting a cell never lowers any column height. -/
theorem colHeight_le_colHeight_set (b : Board) (c : Coord) (j : ℕ) :
    b.colHeight j ≤ (b.set c).colHeight j :=
  colHeight_le_of_subset (subset_set b c) j

/-- `isLost` is monotone in the board: subset of a lost board is lost. -/
theorem isLost_of_subset_of_isLost (cfg : GameConfig) {b₁ b₂ : Board}
    (hsub : b₁ ⊆ b₂) (h : isLost cfg b₁) : isLost cfg b₂ := by
  obtain ⟨p, hp, hge⟩ := h
  exact ⟨p, hsub hp, hge⟩

/-- Not-lost is preserved under taking subsets. -/
theorem not_isLost_of_subset (cfg : GameConfig) {b₁ b₂ : Board}
    (hsub : b₁ ⊆ b₂) (h : ¬ isLost cfg b₂) : ¬ isLost cfg b₁ :=
  fun hl => h (isLost_of_subset_of_isLost cfg hsub hl)

/-- Clearing a cell preserves not-lost. -/
theorem not_isLost_clear (cfg : GameConfig) (b : Board) (c : Coord)
    (h : ¬ isLost cfg b) : ¬ isLost cfg (b.clear c) :=
  not_isLost_of_subset cfg (clear_subset b c) h

/-- The empty board is never lost (`∅` form). -/
@[simp] theorem not_isLost_emptyset (cfg : GameConfig) :
    ¬ isLost cfg (∅ : Board) := by
  rintro ⟨p, hp, _⟩
  simp at hp

/-- `¬ isLost cfg b` iff every filled cell lies strictly below `cfg.rows`. -/
theorem not_isLost_iff_forall (cfg : GameConfig) (b : Board) :
    ¬ isLost cfg b ↔ ∀ p ∈ b, p.2 < cfg.rows := by
  unfold isLost
  push Not
  rfl

/-- `isLost cfg b` iff some filled cell sits at or above `cfg.rows`. -/
theorem isLost_iff_exists (cfg : GameConfig) (b : Board) :
    isLost cfg b ↔ ∃ p ∈ b, cfg.rows ≤ p.2 :=
  Iff.rfl

/-- Setting a cell into a safe board preserves safety iff the new cell
sits below `cfg.rows`. -/
theorem not_isLost_set_iff (cfg : GameConfig) (b : Board) (c : Coord)
    (hb : ¬ isLost cfg b) :
    ¬ isLost cfg (b.set c) ↔ c.2 < cfg.rows := by
  rw [not_isLost_iff_forall] at hb ⊢
  constructor
  · intro h
    exact h c (mem_set_self b c)
  · intro hc p hp
    rcases (mem_set b c p).mp hp with rfl | hpb
    · exact hc
    · exact hb p hpb

/-- Setting a cell with low enough row preserves safety. -/
theorem not_isLost_set (cfg : GameConfig) (b : Board) (c : Coord)
    (hb : ¬ isLost cfg b) (hc : c.2 < cfg.rows) :
    ¬ isLost cfg (b.set c) :=
  (not_isLost_set_iff cfg b c hb).mpr hc

/-- A board union is lost iff either side is lost. -/
theorem isLost_union (cfg : GameConfig) (b₁ b₂ : Board) :
    isLost cfg (b₁ ∪ b₂) ↔ isLost cfg b₁ ∨ isLost cfg b₂ := by
  unfold isLost
  constructor
  · rintro ⟨p, hp, hge⟩
    rcases Finset.mem_union.mp hp with h₁ | h₂
    · exact Or.inl ⟨p, h₁, hge⟩
    · exact Or.inr ⟨p, h₂, hge⟩
  · rintro (⟨p, hp, hge⟩ | ⟨p, hp, hge⟩)
    · exact ⟨p, Finset.mem_union.mpr (Or.inl hp), hge⟩
    · exact ⟨p, Finset.mem_union.mpr (Or.inr hp), hge⟩

/-- Union of two safe boards is safe. -/
theorem not_isLost_union (cfg : GameConfig) {b₁ b₂ : Board}
    (h₁ : ¬ isLost cfg b₁) (h₂ : ¬ isLost cfg b₂) :
    ¬ isLost cfg (b₁ ∪ b₂) := by
  rw [isLost_union]
  rintro (h | h)
  · exact h₁ h
  · exact h₂ h

/-- A board is lost iff some column overflows the field. (Local copy of the
characterisation in `Gameplay.lean` to avoid a circular import.) -/
theorem isLost_iff_exists_colHeight_gt (cfg : GameConfig) (b : Board) :
    isLost cfg b ↔ ∃ j, cfg.rows < colHeight b j := by
  constructor
  · rintro ⟨p, hp, hr⟩
    refine ⟨p.1, ?_⟩
    unfold colHeight
    rw [Finset.lt_sup_iff]
    refine ⟨p.2, ?_, by omega⟩
    simp only [colRows, Finset.mem_image, Finset.mem_filter]
    exact ⟨p, ⟨hp, rfl⟩, rfl⟩
  · rintro ⟨j, hj⟩
    unfold colHeight at hj
    rw [Finset.lt_sup_iff] at hj
    obtain ⟨x, hx, hxlt⟩ := hj
    simp only [colRows, Finset.mem_image, Finset.mem_filter] at hx
    obtain ⟨p, ⟨hpb, _⟩, hpe⟩ := hx
    exact ⟨p, hpb, by omega⟩

/-- Column-form of safety: every column's height is `≤ cfg.rows`. -/
theorem not_isLost_iff_forall_colHeight_le (cfg : GameConfig) (b : Board) :
    ¬ isLost cfg b ↔ ∀ j, colHeight b j ≤ cfg.rows := by
  rw [isLost_iff_exists_colHeight_gt]
  push Not
  rfl

/-- Column-form safety constructor: if every column has height `≤ cfg.rows`,
the board is safe. -/
theorem not_isLost_of_forall_colHeight_le (cfg : GameConfig) (b : Board)
    (h : ∀ j, colHeight b j ≤ cfg.rows) : ¬ isLost cfg b :=
  (not_isLost_iff_forall_colHeight_le cfg b).mpr h

/-- Column-form safety eliminator: a safe board has every column `≤ cfg.rows`. -/
theorem colHeight_le_rows_of_not_isLost (cfg : GameConfig) {b : Board}
    (h : ¬ isLost cfg b) (j : ℕ) : colHeight b j ≤ cfg.rows :=
  (not_isLost_iff_forall_colHeight_le cfg b).mp h j

/-- Setting a cell `c` into the board inserts `c.2` into `colRows` at column `c.1`. -/
theorem colRows_set_eq (b : Board) (c : Coord) :
    (b.set c).colRows c.1 = insert c.2 (b.colRows c.1) := by
  unfold colRows set
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_insert]
  constructor
  · rintro ⟨p, ⟨hp | hp, hpj⟩, hpr⟩
    · subst hp; subst hpr; exact Or.inl rfl
    · exact Or.inr ⟨p, ⟨hp, hpj⟩, hpr⟩
  · rintro (rfl | ⟨p, ⟨hp, hpj⟩, hpr⟩)
    · exact ⟨c, ⟨Or.inl rfl, rfl⟩, rfl⟩
    · exact ⟨p, ⟨Or.inr hp, hpj⟩, hpr⟩

/-- Setting a cell `c` at column `c.1` makes the new height equal to
`max (old height) (c.2 + 1)`. -/
theorem colHeight_set_eq (b : Board) (c : Coord) :
    (b.set c).colHeight c.1 = max (b.colHeight c.1) (c.2 + 1) := by
  unfold colHeight
  rw [colRows_set_eq, Finset.sup_insert]
  exact max_comm _ _

/-- After setting `c`, the height of column `c.1` is at most `c.2 + 1`
when the prior height was already ≤ `c.2 + 1`. -/
theorem colHeight_set_le_succ (b : Board) (c : Coord)
    (h : b.colHeight c.1 ≤ c.2 + 1) :
    (b.set c).colHeight c.1 ≤ c.2 + 1 := by
  rw [colHeight_set_eq]; exact max_le h (le_refl _)

/-- After setting `c`, the height of column `c.1` is `> c.2`, i.e.
the new cell counts towards the column. -/
theorem lt_colHeight_set (b : Board) (c : Coord) :
    c.2 < (b.set c).colHeight c.1 := by
  rw [colHeight_set_eq]
  exact lt_of_lt_of_le (Nat.lt_succ_self _) (le_max_right _ _)


/-- Setting a cell in a different column doesn't change that column's height. -/
theorem colHeight_set_of_ne {b : Board} {c : Coord} {j : ℕ} (h : c.1 ≠ j) :
    (b.set c).colHeight j = b.colHeight j := by
  unfold colHeight colRows set
  congr 1
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_insert]
  constructor
  · rintro ⟨p, ⟨hp | hp, hpj⟩, hpr⟩
    · subst hp; exact absurd hpj h
    · exact ⟨p, ⟨hp, hpj⟩, hpr⟩
  · rintro ⟨p, ⟨hp, hpj⟩, hpr⟩
    exact ⟨p, ⟨Or.inr hp, hpj⟩, hpr⟩

/-- Setting a cell into a board makes the result lost iff the board was
already lost or the new cell sits at/above `cfg.rows`. -/
theorem isLost_set_iff (cfg : GameConfig) (b : Board) (c : Coord) :
    isLost cfg (b.set c) ↔ isLost cfg b ∨ cfg.rows ≤ c.2 := by
  unfold isLost
  constructor
  · rintro ⟨p, hp, hge⟩
    rcases (mem_set b c p).mp hp with rfl | hpb
    · exact Or.inr hge
    · exact Or.inl ⟨p, hpb, hge⟩
  · rintro (⟨p, hp, hge⟩ | hge)
    · exact ⟨p, (mem_set b c p).mpr (Or.inr hp), hge⟩
    · exact ⟨c, mem_set_self b c, hge⟩

/-- `set` and `clear` commute on distinct cells. -/
theorem set_clear_comm_of_ne (b : Board) (c d : Coord) (h : c ≠ d) :
    (b.set c).clear d = (b.clear d).set c := by
  ext x
  simp only [set, clear, Finset.mem_erase, Finset.mem_insert]
  constructor
  · rintro ⟨hxd, hxc | hxb⟩
    · exact Or.inl hxc
    · exact Or.inr ⟨hxd, hxb⟩
  · rintro (hxc | ⟨hxd, hxb⟩)
    · subst hxc; exact ⟨h, Or.inl rfl⟩
    · exact ⟨hxd, Or.inr hxb⟩

/-- Universal column-height upper bound under `set`:
`(b.set c).colHeight j ≤ max (b.colHeight j) (c.2 + 1)`. -/
theorem colHeight_set_le (b : Board) (c : Coord) (j : ℕ) :
    (b.set c).colHeight j ≤ max (b.colHeight j) (c.2 + 1) := by
  by_cases h : c.1 = j
  · subst h; rw [colHeight_set_eq]
  · rw [colHeight_set_of_ne h]; exact le_max_left _ _

/-- `colRows` distributes over union. -/
theorem colRows_union (b₁ b₂ : Board) (j : ℕ) :
    (b₁ ∪ b₂).colRows j = b₁.colRows j ∪ b₂.colRows j := by
  unfold colRows
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_union]
  constructor
  · rintro ⟨p, ⟨h₁ | h₂, hpj⟩, hpr⟩
    · exact Or.inl ⟨p, ⟨h₁, hpj⟩, hpr⟩
    · exact Or.inr ⟨p, ⟨h₂, hpj⟩, hpr⟩
  · rintro (⟨p, ⟨hp, hpj⟩, hpr⟩ | ⟨p, ⟨hp, hpj⟩, hpr⟩)
    · exact ⟨p, ⟨Or.inl hp, hpj⟩, hpr⟩
    · exact ⟨p, ⟨Or.inr hp, hpj⟩, hpr⟩

/-- `colHeight` of a union equals the max of the two heights. -/
theorem colHeight_union (b₁ b₂ : Board) (j : ℕ) :
    (b₁ ∪ b₂).colHeight j = max (b₁.colHeight j) (b₂.colHeight j) := by
  unfold colHeight
  rw [colRows_union, Finset.sup_union]

/-- Safety of a union is conjunction of safeties (and-form companion to
Tick 1100's `isLost_union`). -/
theorem not_isLost_union_iff (cfg : GameConfig) (b₁ b₂ : Board) :
    ¬ isLost cfg (b₁ ∪ b₂) ↔ ¬ isLost cfg b₁ ∧ ¬ isLost cfg b₂ := by
  rw [isLost_union]
  push Not
  rfl

/-- If every cell of a board lies strictly below `n`, its column heights
are all `≤ n`. -/
theorem colHeight_le_of_forall_lt {b : Board} {n : ℕ}
    (h : ∀ p ∈ b, p.2 < n) (j : ℕ) : b.colHeight j ≤ n := by
  unfold colHeight
  refine Finset.sup_le ?_
  intro r hr
  simp only [colRows, Finset.mem_image, Finset.mem_filter] at hr
  obtain ⟨p, ⟨hpb, _⟩, hpr⟩ := hr
  subst hpr
  exact h p hpb

/-- Universal column form of `not_isLost_iff_forall`. -/
theorem not_isLost_of_forall_row_lt (cfg : GameConfig) {b : Board}
    (h : ∀ p ∈ b, p.2 < cfg.rows) : ¬ isLost cfg b :=
  (not_isLost_iff_forall cfg b).mpr h

/-- `set` is monotone in the underlying board. -/
theorem set_subset_set {b₁ b₂ : Board} (h : b₁ ⊆ b₂) (c : Coord) :
    b₁.set c ⊆ b₂.set c :=
  Finset.insert_subset_insert _ h

/-- `clear` is monotone in the underlying board. -/
theorem clear_subset_clear {b₁ b₂ : Board} (h : b₁ ⊆ b₂) (c : Coord) :
    b₁.clear c ⊆ b₂.clear c :=
  Finset.erase_subset_erase _ h

/-- `set` is the union with a singleton. -/
theorem set_eq_union_singleton (b : Board) (c : Coord) :
    b.set c = b ∪ {c} := by
  unfold set
  rw [Finset.insert_eq, Finset.union_comm]

/-- Singleton is never lost iff the cell row is below `cfg.rows`. -/
theorem not_isLost_singleton_iff (cfg : GameConfig) (c : Coord) :
    ¬ isLost cfg ({c} : Board) ↔ c.2 < cfg.rows := by
  rw [not_isLost_iff_forall]
  constructor
  · intro h
    exact h c (Finset.mem_singleton.mpr rfl)
  · intro hc p hp
    rw [Finset.mem_singleton.mp hp]; exact hc

/-- Singleton board is lost iff the cell row is at or above `cfg.rows`. -/
theorem isLost_singleton_iff (cfg : GameConfig) (c : Coord) :
    isLost cfg ({c} : Board) ↔ cfg.rows ≤ c.2 := by
  unfold isLost
  constructor
  · rintro ⟨p, hp, hge⟩
    rw [Finset.mem_singleton.mp hp] at hge; exact hge
  · intro hge
    exact ⟨c, Finset.mem_singleton.mpr rfl, hge⟩

/-- The `colRows` of a singleton at column `c.1` is `{c.2}`. -/
theorem colRows_singleton (c : Coord) :
    ({c} : Board).colRows c.1 = {c.2} := by
  unfold colRows
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_singleton]
  constructor
  · rintro ⟨p, ⟨hp, _⟩, hpr⟩
    rw [hp] at hpr; exact hpr.symm
  · rintro rfl
    exact ⟨c, ⟨rfl, rfl⟩, rfl⟩

/-- The `colRows` of a singleton at any other column is empty. -/
theorem colRows_singleton_of_ne {c : Coord} {j : ℕ} (h : c.1 ≠ j) :
    ({c} : Board).colRows j = ∅ := by
  unfold colRows
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_singleton,
    Finset.notMem_empty, iff_false]
  rintro ⟨p, ⟨hp, hpj⟩, _⟩
  rw [hp] at hpj; exact h hpj

/-- The column height of a singleton at its own column is `c.2 + 1`. -/
theorem colHeight_singleton (c : Coord) :
    ({c} : Board).colHeight c.1 = c.2 + 1 := by
  unfold colHeight; rw [colRows_singleton]; simp

/-- The column height of a singleton at any other column is `0`. -/
theorem colHeight_singleton_of_ne {c : Coord} {j : ℕ} (h : c.1 ≠ j) :
    ({c} : Board).colHeight j = 0 := by
  unfold colHeight; rw [colRows_singleton_of_ne h]; simp

/-- A column has positive height iff some cell sits in it. -/
theorem colHeight_pos_iff_exists_mem (b : Board) (j : ℕ) :
    0 < b.colHeight j ↔ ∃ r, (j, r) ∈ b := by
  unfold colHeight
  rw [Finset.lt_sup_iff]
  simp only [colRows, Finset.mem_image, Finset.mem_filter, Nat.lt_succ_iff]
  constructor
  · rintro ⟨r, ⟨p, ⟨hpb, hpj⟩, hpr⟩, _⟩
    refine ⟨r, ?_⟩
    have : p = (j, r) := by
      cases p; simp at hpj hpr; aesop
    rwa [this] at hpb
  · rintro ⟨r, hr⟩
    exact ⟨r, ⟨(j, r), ⟨hr, rfl⟩, rfl⟩, Nat.zero_le _⟩

/-- If `c ∉ b`, set then clear is the identity. -/
theorem clear_set_of_not_mem (b : Board) (c : Coord) (h : c ∉ b) :
    (b.set c).clear c = b := by
  rw [clear_set, clear_of_not_mem b c h]

/-- If `c ∈ b`, clear then set is the identity. -/
theorem set_clear_of_mem (b : Board) (c : Coord) (h : c ∈ b) :
    (b.clear c).set c = b := by
  rw [set_clear, set_of_mem b c h]

/-- Strict column-height bound for a safe board: `colHeight b j < cfg.rows + 1`. -/
theorem colHeight_lt_succ_rows_of_not_isLost (cfg : GameConfig) {b : Board}
    (h : ¬ isLost cfg b) (j : ℕ) : colHeight b j < cfg.rows + 1 :=
  Nat.lt_succ_of_le (colHeight_le_rows_of_not_isLost cfg h j)

/-- Adding a cell at or above the current column top yields exactly `c.2 + 1`. -/
theorem colHeight_set_eq_succ_of_le (b : Board) (c : Coord)
    (h : b.colHeight c.1 ≤ c.2 + 1) :
    (b.set c).colHeight c.1 = c.2 + 1 := by
  rw [colHeight_set_eq]
  exact max_eq_right h

/-- Adding a cell strictly below the current column top doesn't change the height. -/
theorem colHeight_set_eq_of_lt (b : Board) (c : Coord)
    (h : c.2 + 1 ≤ b.colHeight c.1) :
    (b.set c).colHeight c.1 = b.colHeight c.1 := by
  rw [colHeight_set_eq]
  exact max_eq_left h

/-- `b.set c = b` iff `c ∈ b`. -/
theorem set_eq_self_iff (b : Board) (c : Coord) : b.set c = b ↔ c ∈ b :=
  Finset.insert_eq_self

/-- `b.clear c = b` iff `c ∉ b`. -/
theorem clear_eq_self_iff (b : Board) (c : Coord) : b.clear c = b ↔ c ∉ b :=
  Finset.erase_eq_self

/-- If clearing leaves a lost board, the original was already lost. -/
theorem isLost_of_isLost_clear (cfg : GameConfig) (b : Board) (c : Coord)
    (h : isLost cfg (b.clear c)) : isLost cfg b :=
  isLost_of_subset_of_isLost cfg (clear_subset b c) h

/-- `(b₁ ∪ b₂).set c = b₁.set c ∪ b₂`: setting distributes over union (left). -/
theorem set_union_left (b₁ b₂ : Board) (c : Coord) :
    (b₁ ∪ b₂).set c = b₁.set c ∪ b₂ := by
  ext x
  simp only [set, Finset.mem_insert, Finset.mem_union]
  tauto

/-- `(b₁ ∪ b₂).set c = b₁ ∪ b₂.set c`: setting distributes over union (right). -/
theorem set_union_right (b₁ b₂ : Board) (c : Coord) :
    (b₁ ∪ b₂).set c = b₁ ∪ b₂.set c := by
  ext x
  simp only [set, Finset.mem_insert, Finset.mem_union]
  tauto

/-- `clear` distributes over union. -/
theorem clear_union (b₁ b₂ : Board) (c : Coord) :
    (b₁ ∪ b₂).clear c = b₁.clear c ∪ b₂.clear c := by
  ext x
  simp only [clear, Finset.mem_erase, Finset.mem_union]
  tauto

/-- `b.set c ⊆ b'` iff `b ⊆ b'` and `c ∈ b'`. -/
theorem set_subset_iff (b b' : Board) (c : Coord) :
    b.set c ⊆ b' ↔ b ⊆ b' ∧ c ∈ b' := by
  unfold set
  rw [Finset.insert_subset_iff]
  exact and_comm

/-- Placing a piece (as a `Finset Coord`) onto a safe board preserves
safety when every piece cell sits below `cfg.rows`. -/
theorem not_isLost_union_of_forall_lt (cfg : GameConfig) {b piece : Board}
    (hb : ¬ isLost cfg b)
    (hp : ∀ p ∈ piece, p.2 < cfg.rows) :
    ¬ isLost cfg (b ∪ piece) :=
  not_isLost_union cfg hb (not_isLost_of_forall_row_lt cfg hp)

/-- Iff form of piece-placement safety: union safe iff both sides safe and
piece cells all below `cfg.rows`. -/
theorem not_isLost_union_iff_forall_lt (cfg : GameConfig) (b piece : Board) :
    ¬ isLost cfg (b ∪ piece) ↔ ¬ isLost cfg b ∧ ∀ p ∈ piece, p.2 < cfg.rows := by
  rw [not_isLost_union_iff, not_isLost_iff_forall cfg piece]

/-- An image board is safe iff every translated cell has row below `cfg.rows`. -/
theorem not_isLost_image_iff (cfg : GameConfig) {α : Type*}
    (s : Finset α) (f : α → Coord) :
    ¬ isLost cfg (s.image f) ↔ ∀ p ∈ s, (f p).2 < cfg.rows := by
  rw [not_isLost_iff_forall]
  constructor
  · intro h p hp
    exact h (f p) (Finset.mem_image.mpr ⟨p, hp, rfl⟩)
  · rintro h _ hq
    obtain ⟨p, hpb, rfl⟩ := Finset.mem_image.mp hq
    exact h p hpb

/-- A row at row `r` is safe iff its width is zero or `r < cfg.rows`. -/
theorem not_isLost_row_iff (cfg : GameConfig) (cols r : ℕ) :
    ¬ isLost cfg (Board.row cols r) ↔ cols = 0 ∨ r < cfg.rows := by
  rw [Board.row, not_isLost_image_iff]
  constructor
  · intro h
    by_cases hc : cols = 0
    · exact Or.inl hc
    · refine Or.inr ?_
      have h0 : 0 ∈ Finset.range cols := Finset.mem_range.mpr (Nat.pos_of_ne_zero hc)
      exact h 0 h0
  · rintro (rfl | hr)
    · intro c hc; simp at hc
    · intro c _; exact hr

/-- A row at row `r < cfg.rows` is safe (regardless of width). -/
theorem not_isLost_row_of_lt (cfg : GameConfig) (cols : ℕ) {r : ℕ}
    (hr : r < cfg.rows) : ¬ isLost cfg (Board.row cols r) :=
  (not_isLost_row_iff cfg cols r).mpr (Or.inr hr)

/-- Filtering a safe board by any predicate keeps it safe. -/
theorem not_isLost_filter (cfg : GameConfig) {b : Board} (p : Coord → Prop)
    [DecidablePred p] (h : ¬ isLost cfg b) : ¬ isLost cfg (b.filter p) :=
  not_isLost_of_subset cfg (Finset.filter_subset _ _) h

/-- Line-clearing never increases the cell count. -/
theorem count_clearLines_le (cfg : GameConfig) (b : Board) :
    (clearLines cfg b).count ≤ b.count := by
  unfold clearLines count
  refine Nat.le_trans Finset.card_image_le ?_
  exact Finset.card_filter_le _ _

/-- A board with no full rows has `clearedBelow = 0` for every row. -/
theorem clearedBelow_of_no_fullRows (cfg : GameConfig) {b : Board} (r : ℕ)
    (h : fullRows cfg b = ∅) : clearedBelow cfg b r = 0 := by
  unfold clearedBelow
  rw [h]
  simp

/-- The empty board has no full rows. -/
@[simp] theorem fullRows_empty (cfg : GameConfig) :
    fullRows cfg (∅ : Board) = ∅ := by
  unfold fullRows
  simp

/-- The number of full rows below `r` is at most the total number of full rows. -/
theorem clearedBelow_le_card_fullRows (cfg : GameConfig) (b : Board) (r : ℕ) :
    clearedBelow cfg b r ≤ (fullRows cfg b).card := by
  unfold clearedBelow
  exact Finset.card_filter_le _ _

/-- The number of full rows below `r` is at most `r`. -/
theorem clearedBelow_le_row (cfg : GameConfig) (b : Board) (r : ℕ) :
    clearedBelow cfg b r ≤ r := by
  unfold clearedBelow
  refine Nat.le_trans (Finset.card_le_card ?_) (by rw [Finset.card_range])
  intro x hx
  rw [Finset.mem_filter] at hx
  exact Finset.mem_range.mpr hx.2

/-- A board with no full rows is unchanged by `clearLines`. -/
theorem clearLines_eq_self_of_no_fullRows (cfg : GameConfig) {b : Board}
    (h : fullRows cfg b = ∅) : clearLines cfg b = b := by
  have hkeep : ∀ p ∈ b, ¬ isFull cfg b p.2 := by
    intro p hp hfull
    have : p.2 ∈ fullRows cfg b := by
      simp only [fullRows, Finset.mem_filter, Finset.mem_image]
      exact ⟨⟨p, hp, rfl⟩, hfull⟩
    rw [h] at this
    exact Finset.notMem_empty _ this
  unfold clearLines
  ext q
  simp only [Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨p, ⟨hpb, _⟩, hpq⟩
    rw [clearedBelow_of_no_fullRows cfg p.2 h] at hpq
    rw [Nat.sub_zero] at hpq
    have : q = p := by
      rw [← hpq]
    rw [this]; exact hpb
  · intro hq
    refine ⟨q, ⟨hq, hkeep q hq⟩, ?_⟩
    rw [clearedBelow_of_no_fullRows cfg q.2 h]
    rfl

/-- A line clear with no full rows preserves the count. -/
theorem count_clearLines_eq_self_of_no_fullRows (cfg : GameConfig) {b : Board}
    (h : fullRows cfg b = ∅) : (clearLines cfg b).count = b.count := by
  rw [clearLines_eq_self_of_no_fullRows cfg h]

/-- `clearedBelow` at row `0` is always `0` (no full rows sit below the floor). -/
@[simp] theorem clearedBelow_zero (cfg : GameConfig) (b : Board) :
    clearedBelow cfg b 0 = 0 := by
  unfold clearedBelow
  simp

/-- `clearedBelow` is monotone in the row argument. -/
theorem clearedBelow_mono (cfg : GameConfig) (b : Board) {r r' : ℕ}
    (h : r ≤ r') : clearedBelow cfg b r ≤ clearedBelow cfg b r' := by
  unfold clearedBelow
  refine Finset.card_le_card ?_
  intro x hx
  rw [Finset.mem_filter] at hx ⊢
  exact ⟨hx.1, Nat.lt_of_lt_of_le hx.2 h⟩

/-- Membership in `clearLines`: surviving cells are exactly the non-full-row
cells, with their row shifted down by `clearedBelow`. -/
theorem mem_clearLines_iff (cfg : GameConfig) (b : Board) (q : Coord) :
    q ∈ clearLines cfg b ↔
      ∃ p ∈ b, ¬ isFull cfg b p.2 ∧ (p.1, p.2 - clearedBelow cfg b p.2) = q := by
  unfold clearLines
  simp only [Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨p, ⟨hpb, hpf⟩, hpq⟩
    exact ⟨p, hpb, hpf, hpq⟩
  · rintro ⟨p, hpb, hpf, hpq⟩
    exact ⟨p, ⟨hpb, hpf⟩, hpq⟩

/-- The full-rows set is contained in the row image of the board. -/
theorem fullRows_subset_image_row (cfg : GameConfig) (b : Board) :
    fullRows cfg b ⊆ b.image (·.2) :=
  Finset.filter_subset _ _

/-- Membership in `fullRows`: `r` appears in some board cell and the
whole row is filled. -/
theorem mem_fullRows_iff (cfg : GameConfig) (b : Board) (r : ℕ) :
    r ∈ fullRows cfg b ↔ r ∈ b.image (·.2) ∧ isFull cfg b r := by
  unfold fullRows
  exact Finset.mem_filter

/-- A row in `fullRows` is automatically full. -/
theorem isFull_of_mem_fullRows {cfg : GameConfig} {b : Board} {r : ℕ}
    (h : r ∈ fullRows cfg b) : isFull cfg b r :=
  ((mem_fullRows_iff cfg b r).mp h).2

/-- A row in `fullRows` is the row of some filled cell. -/
theorem mem_image_row_of_mem_fullRows {cfg : GameConfig} {b : Board} {r : ℕ}
    (h : r ∈ fullRows cfg b) : r ∈ b.image (·.2) :=
  ((mem_fullRows_iff cfg b r).mp h).1

/-- `isFull cfg ∅ r ↔ cfg.cols = 0`: the empty board has no full rows except
in the degenerate zero-column case. -/
theorem isFull_emptyset_iff (cfg : GameConfig) (r : ℕ) :
    isFull cfg (∅ : Board) r ↔ cfg.cols = 0 := by
  unfold isFull
  constructor
  · intro h
    by_contra hc
    have h0 : 0 ∈ Finset.range cfg.cols :=
      Finset.mem_range.mpr (Nat.pos_of_ne_zero hc)
    exact (Finset.notMem_empty (0, r)) (h 0 h0)
  · intro h c hc
    rw [h] at hc; simp at hc

/-- When `cfg.cols > 0`, no row is full on the empty board. -/
theorem not_isFull_emptyset_of_cols_pos {cfg : GameConfig} (h : 0 < cfg.cols)
    (r : ℕ) : ¬ isFull cfg (∅ : Board) r := by
  rw [isFull_emptyset_iff]
  omega

/-- For standard config (10×20), the empty board has no full rows. -/
theorem not_isFull_emptyset_standard (r : ℕ) :
    ¬ isFull GameConfig.standard (∅ : Board) r :=
  not_isFull_emptyset_of_cols_pos (by decide) r

/-- A row is not full iff some column in `[0, cfg.cols)` is missing in that row. -/
theorem not_isFull_iff_exists_missing (cfg : GameConfig) (b : Board) (r : ℕ) :
    ¬ isFull cfg b r ↔ ∃ c ∈ Finset.range cfg.cols, (c, r) ∉ b := by
  unfold isFull
  push Not
  rfl

/-- If `(c, r) ∉ b` and `c < cfg.cols`, then row `r` is not full. -/
theorem not_isFull_of_not_mem (cfg : GameConfig) {b : Board} {c r : ℕ}
    (hc : c < cfg.cols) (h : (c, r) ∉ b) : ¬ isFull cfg b r :=
  (not_isFull_iff_exists_missing cfg b r).mpr ⟨c, Finset.mem_range.mpr hc, h⟩

/-- If row `r` is full and `c < cfg.cols`, then `(c, r) ∈ b`. -/
theorem mem_of_isFull (cfg : GameConfig) {b : Board} {c r : ℕ}
    (hc : c < cfg.cols) (h : isFull cfg b r) : (c, r) ∈ b :=
  h c (Finset.mem_range.mpr hc)

/-- A board containing a full row has at least `cfg.cols` cells. -/
theorem cols_le_count_of_isFull (cfg : GameConfig) {b : Board} {r : ℕ}
    (h : isFull cfg b r) : cfg.cols ≤ b.count := by
  have hsub : Board.row cfg.cols r ⊆ b := by
    intro p hp
    simp only [Board.row, Finset.mem_image, Finset.mem_range] at hp
    obtain ⟨c, hc, rfl⟩ := hp
    exact mem_of_isFull cfg hc h
  have : (Board.row cfg.cols r).count ≤ b.count := Finset.card_le_card hsub
  rw [count_row] at this
  exact this

/-- A board with at least one full row has `≥ cfg.cols` cells. -/
theorem cols_le_count_of_fullRows_nonempty (cfg : GameConfig) {b : Board}
    (h : (fullRows cfg b).Nonempty) : cfg.cols ≤ b.count := by
  obtain ⟨r, hr⟩ := h
  exact cols_le_count_of_isFull cfg (isFull_of_mem_fullRows hr)

/-- A board with fewer cells than the column count has no full rows. -/
theorem fullRows_eq_empty_of_count_lt (cfg : GameConfig) {b : Board}
    (h : b.count < cfg.cols) : fullRows cfg b = ∅ := by
  rcases Finset.eq_empty_or_nonempty (fullRows cfg b) with he | hne
  · exact he
  · exfalso
    exact Nat.lt_le_asymm h (cols_le_count_of_fullRows_nonempty cfg hne)

/-- A board with fewer cells than the column count is fixed by `clearLines`. -/
theorem clearLines_eq_self_of_count_lt (cfg : GameConfig) {b : Board}
    (h : b.count < cfg.cols) : clearLines cfg b = b :=
  clearLines_eq_self_of_no_fullRows cfg (fullRows_eq_empty_of_count_lt cfg h)

/-- A board with fewer cells than the column count has its count preserved
under `clearLines`. -/
theorem count_clearLines_eq_count_of_count_lt (cfg : GameConfig) {b : Board}
    (h : b.count < cfg.cols) : (clearLines cfg b).count = b.count := by
  rw [clearLines_eq_self_of_count_lt cfg h]


/-- Any cell in `clearLines cfg b` comes from a cell in `b` with row at
least as high (the gravity step only lowers rows). -/
theorem exists_source_of_mem_clearLines (cfg : GameConfig) {b : Board} {q : Coord}
    (h : q ∈ clearLines cfg b) :
    ∃ p ∈ b, q.2 ≤ p.2 ∧ p.1 = q.1 := by
  obtain ⟨p, hpb, _, hpq⟩ := (mem_clearLines_iff cfg b q).mp h
  refine ⟨p, hpb, ?_, ?_⟩
  · rw [show q.2 = p.2 - clearedBelow cfg b p.2 by rw [← hpq]]
    exact Nat.sub_le _ _
  · rw [show q.1 = p.1 by rw [← hpq]]

/-- A line clear preserves safety: the post-clear board's surviving cells
have row no greater than their original row, so the row bound is preserved. -/
theorem not_isLost_clearLines (cfg : GameConfig) (b : Board)
    (h : ¬ isLost cfg b) : ¬ isLost cfg (clearLines cfg b) := by
  rw [not_isLost_iff_forall] at h ⊢
  intro q hq
  simp only [clearLines, Finset.mem_image, Finset.mem_filter] at hq
  obtain ⟨p, ⟨hpb, _⟩, hpq⟩ := hq
  have hpr : p.2 < cfg.rows := h p hpb
  have : q.2 = p.2 - clearedBelow cfg b p.2 := by
    rw [← hpq]
  rw [this]
  exact Nat.lt_of_le_of_lt (Nat.sub_le _ _) hpr

/-- A row is lost iff it has positive width and sits at or above `cfg.rows`. -/
theorem isLost_row_iff (cfg : GameConfig) (cols r : ℕ) :
    isLost cfg (Board.row cols r) ↔ 0 < cols ∧ cfg.rows ≤ r := by
  rw [← not_not (a := isLost _ _), not_isLost_row_iff]
  push Not
  omega

/-- `isLost` after clearing is iff some *other* cell exceeds `cfg.rows`. -/
theorem isLost_clear_iff (cfg : GameConfig) (b : Board) (c : Coord) :
    isLost cfg (b.clear c) ↔ ∃ p ∈ b, p ≠ c ∧ cfg.rows ≤ p.2 := by
  unfold isLost
  constructor
  · rintro ⟨p, hp, hge⟩
    rcases (mem_clear b c p).mp hp with ⟨hne, hpb⟩
    exact ⟨p, hpb, hne, hge⟩
  · rintro ⟨p, hpb, hne, hge⟩
    exact ⟨p, (mem_clear b c p).mpr ⟨hne, hpb⟩, hge⟩

/-- Clearing a cell in a different column doesn't change that column's height. -/
theorem colHeight_clear_of_ne {b : Board} {c : Coord} {j : ℕ} (h : c.1 ≠ j) :
    (b.clear c).colHeight j = b.colHeight j := by
  unfold colHeight colRows clear
  congr 1
  ext r
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_erase]
  constructor
  · rintro ⟨p, ⟨⟨_, hp⟩, hpj⟩, hpr⟩
    exact ⟨p, ⟨hp, hpj⟩, hpr⟩
  · rintro ⟨p, ⟨hp, hpj⟩, hpr⟩
    refine ⟨p, ⟨⟨?_, hp⟩, hpj⟩, hpr⟩
    rintro rfl; exact h hpj

/-- A column has zero height iff no cell sits in it. -/
theorem colHeight_eq_zero_iff_forall_not_mem (b : Board) (j : ℕ) :
    b.colHeight j = 0 ↔ ∀ r, (j, r) ∉ b := by
  constructor
  · intro h r hr
    have hpos : 0 < b.colHeight j :=
      (colHeight_pos_iff_exists_mem b j).mpr ⟨r, hr⟩
    omega
  · intro h
    by_contra hne
    have hpos : 0 < b.colHeight j := Nat.pos_of_ne_zero hne
    obtain ⟨r, hr⟩ := (colHeight_pos_iff_exists_mem b j).mp hpos
    exact h r hr

/-- Predicate "every cell's column is `< cols`". Generalises `Board.WF`
without requiring a full `GameConfig`. -/
def ColWF (cols : ℕ) (b : Board) : Prop := ∀ p ∈ b, p.1 < cols

/-- For a `GameConfig`, `WF cfg b ↔ ColWF cfg.cols b`. -/
theorem colWF_iff_wf {cfg : GameConfig} {b : Board} :
    ColWF cfg.cols b ↔ Board.WF cfg b := Iff.rfl

/-! ### Maximum column height (`maxHeight`) and the volume bound

`maxHeight cfg b` is the height of the tallest in-range column — the per-column
`colHeight` packaged into a single board statistic. It underpins the volume bound
`count ≤ cols * maxHeight` (a well-formed board embeds into the `cols × maxHeight`
rectangle) and the line-clear corollary `#fullRows ≤ maxHeight`. -/

/-- The height of the tallest in-range column. -/
def maxHeight (cfg : GameConfig) (b : Board) : ℕ :=
  (Finset.range cfg.cols).sup (fun j => b.colHeight j)

@[simp] theorem maxHeight_empty (cfg : GameConfig) : maxHeight cfg (∅ : Board) = 0 := by
  unfold maxHeight
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    simp [Board.colHeight_empty]
  · exact Nat.zero_le _

/-- Every in-range column is no taller than the board max-height. -/
theorem colHeight_le_maxHeight {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) : b.colHeight j ≤ maxHeight cfg b := by
  unfold maxHeight
  exact Finset.le_sup (Finset.mem_range.mpr hj)

/-- **Volume bound.** A well-formed board's cell count is at most `cols × maxHeight`:
every filled cell `(j, r)` has `j < cols` and `r < colHeight j ≤ maxHeight`, so the
board embeds into the `cols × maxHeight` rectangle. -/
theorem count_le_cols_mul_maxHeight {cfg : GameConfig} (b : Board)
    (hwf : Board.WF cfg b) :
    b.count ≤ cfg.cols * maxHeight cfg b := by
  have hsub : b ⊆ (Finset.range cfg.cols) ×ˢ (Finset.range (maxHeight cfg b)) := by
    intro c hc
    obtain ⟨j, r⟩ := c
    have hj : j < cfg.cols := hwf (j, r) hc
    have hr : r < b.colHeight j := Board.lt_colHeight hc
    simp only [Finset.mem_product, Finset.mem_range]
    exact ⟨hj, lt_of_lt_of_le hr (colHeight_le_maxHeight hj)⟩
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_product, Finset.card_range, Finset.card_range] at hcard
  exact hcard

/-- Volume bound against any height cap `H`: a board no taller than `H` holds at
most `cols × H` cells. -/
theorem count_le_of_maxHeight_le {cfg : GameConfig} {b : Board} {H : ℕ}
    (hwf : Board.WF cfg b) (hH : maxHeight cfg b ≤ H) :
    b.count ≤ cfg.cols * H :=
  le_trans (count_le_cols_mul_maxHeight b hwf) (Nat.mul_le_mul (le_refl cfg.cols) hH)

/-- Contrapositive of the volume bound: if the count exceeds `cols × H`, some
column rises strictly above `H`. -/
theorem lt_maxHeight_of_cols_mul_lt_count {cfg : GameConfig} {b : Board} {H : ℕ}
    (hwf : Board.WF cfg b) (h : cfg.cols * H < b.count) :
    H < maxHeight cfg b := by
  by_contra hcon
  push Not at hcon
  have hle := count_le_of_maxHeight_le hwf hcon
  omega

/-- An in-field board (every filled cell strictly below `cfg.rows`) is at most
`cfg.rows` tall. -/
theorem maxHeight_le_rows_of_forall_row_lt {cfg : GameConfig} {b : Board}
    (h : ∀ p ∈ b, p.2 < cfg.rows) :
    maxHeight cfg b ≤ cfg.rows := by
  unfold maxHeight
  apply Finset.sup_le
  intro j _
  change (b.colRows j).sup (· + 1) ≤ cfg.rows
  apply Finset.sup_le
  intro r hr
  have hjr : (j, r) ∈ b := by
    rcases Finset.mem_image.mp hr with ⟨⟨c, r'⟩, hmem', heq⟩
    rcases Finset.mem_filter.mp hmem' with ⟨hcell, hcol⟩
    simp only at hcol heq
    subst hcol
    subst heq
    exact hcell
  have hlt : r < cfg.rows := h (j, r) hjr
  change r + 1 ≤ cfg.rows
  omega

/-- A missing in-field cell `(j, r) ∉ b` with `j < cfg.cols` keeps row `r` from
being full: fullness demands every in-range column be filled, and column `j` is empty. -/
theorem not_isFull_of_notMem {cfg : GameConfig} {b : Board} {j r : ℕ}
    (hj : j < cfg.cols) (hmem : (j, r) ∉ b) : ¬ Board.isFull cfg b r := by
  intro hfull
  unfold Board.isFull at hfull
  exact hmem (hfull j (Finset.mem_range.mpr hj))

/-- A missing in-field cell excludes its row from `fullRows`, so that row is not
cleared by `clearLines`. -/
theorem notMem_fullRows_of_notMem {cfg : GameConfig} {b : Board} {j r : ℕ}
    (hj : j < cfg.cols) (hmem : (j, r) ∉ b) : r ∉ Board.fullRows cfg b := by
  intro hr
  unfold Board.fullRows at hr
  rw [Finset.mem_filter] at hr
  exact not_isFull_of_notMem hj hmem hr.2

/-- Every full row sits below the stack height: a full row holds `(0, r) ∈ b`, so
`r < colHeight 0 ≤ maxHeight`. -/
theorem lt_maxHeight_of_mem_fullRows {cfg : GameConfig} {b : Board} {r : ℕ}
    (hcols : 0 < cfg.cols) (hr : r ∈ Board.fullRows cfg b) : r < maxHeight cfg b := by
  unfold Board.fullRows at hr
  rw [Finset.mem_filter] at hr
  have hfull := hr.2
  unfold Board.isFull at hfull
  have hmem : (0, r) ∈ b := hfull 0 (Finset.mem_range.mpr hcols)
  exact lt_of_lt_of_le (Board.lt_colHeight hmem) (colHeight_le_maxHeight (j := 0) hcols)

/-- The full rows live inside `range (maxHeight)`. -/
theorem fullRows_subset_range_maxHeight {cfg : GameConfig} {b : Board}
    (hcols : 0 < cfg.cols) : Board.fullRows cfg b ⊆ Finset.range (maxHeight cfg b) := by
  intro r hr
  exact Finset.mem_range.mpr (lt_maxHeight_of_mem_fullRows hcols hr)

/-- **Clears ≤ height.** At most `maxHeight cfg b` rows are full at once, so a single
move clears at most `maxHeight` lines. -/
theorem fullRows_card_le_maxHeight {cfg : GameConfig} {b : Board}
    (hcols : 0 < cfg.cols) : (Board.fullRows cfg b).card ≤ maxHeight cfg b := by
  calc (Board.fullRows cfg b).card
      ≤ (Finset.range (maxHeight cfg b)).card :=
        Finset.card_le_card (fullRows_subset_range_maxHeight hcols)
    _ = maxHeight cfg b := Finset.card_range _

end Board
end Tetris
