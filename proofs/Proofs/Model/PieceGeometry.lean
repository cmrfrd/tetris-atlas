import Mathlib
import Proofs.Model.Piece

/-!
# Piece geometry theorems

Basic well-formedness facts about the tetrominoes.
-/

namespace Tetris
namespace Piece

/-- Every piece, in every rotation, occupies exactly 4 cells — i.e. each is a
genuine *tetro*-mino. -/
theorem shape_card (p : Piece) (r : Rotation) : (p.shape r).card = 4 := by
  fin_cases r <;> cases p <;> decide

/-- The bottom-up drop profile also has exactly 4 cells. -/
theorem shapeUp_card (p : Piece) (r : Rotation) : (p.shapeUp r).card = 4 := by
  fin_cases r <;> cases p <;> decide

/-- The number of distinct rotations is 1, 2, or 4. -/
theorem numRotations_mem (p : Piece) : p.numRotations ∈ ({1, 2, 4} : Finset Nat) := by
  cases p <;> decide

/-- The bottom-up profile's column offsets stay within the 4-wide bounding box. -/
theorem shapeUp_col_lt_four (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shapeUp r, cell.1 < 4 := by
  fin_cases r <;> cases p <;> decide

/-- The bottom-up profile's row offsets stay within the 4-tall bounding box. -/
theorem shapeUp_row_lt_four (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shapeUp r, cell.2 < 4 := by
  fin_cases r <;> cases p <;> decide

/-- Every piece in every rotation has at least one cell on the leftmost column
of its bounding box. -/
theorem shapeUp_zero_mem (p : Piece) (r : Rotation) :
    ∃ cell ∈ p.shapeUp r, cell.1 = 0 := by
  fin_cases r <;> cases p <;> decide

/-- `shape` is always nonempty (since it has 4 cells). -/
theorem shape_nonempty (p : Piece) (r : Rotation) : (p.shape r).Nonempty :=
  Finset.card_pos.mp (by rw [shape_card]; decide)

/-- `shapeUp` is always nonempty (since it has 4 cells). -/
theorem shapeUp_nonempty (p : Piece) (r : Rotation) : (p.shapeUp r).Nonempty :=
  Finset.card_pos.mp (by rw [shapeUp_card]; decide)

/-- `shape ≠ ∅`. -/
theorem shape_ne_empty (p : Piece) (r : Rotation) : p.shape r ≠ ∅ :=
  (shape_nonempty p r).ne_empty

/-- `shapeUp ≠ ∅`. -/
theorem shapeUp_ne_empty (p : Piece) (r : Rotation) : p.shapeUp r ≠ ∅ :=
  (shapeUp_nonempty p r).ne_empty

/-- `shape.card > 0`. -/
theorem shape_card_pos (p : Piece) (r : Rotation) : 0 < (p.shape r).card := by
  rw [shape_card]; decide

/-- `shapeUp.card > 0`. -/
theorem shapeUp_card_pos (p : Piece) (r : Rotation) : 0 < (p.shapeUp r).card := by
  rw [shapeUp_card]; decide

/-- `shape.card ≠ 0`. -/
theorem shape_card_ne_zero (p : Piece) (r : Rotation) : (p.shape r).card ≠ 0 :=
  Nat.ne_of_gt (shape_card_pos p r)

/-- `shapeUp.card ≠ 0`. -/
theorem shapeUp_card_ne_zero (p : Piece) (r : Rotation) :
    (p.shapeUp r).card ≠ 0 :=
  Nat.ne_of_gt (shapeUp_card_pos p r)

/-- `shape.card ≤ 4` (trivially since `= 4`). -/
theorem shape_card_le_four (p : Piece) (r : Rotation) : (p.shape r).card ≤ 4 := by
  rw [shape_card]

/-- `shapeUp.card ≤ 4` (trivially since `= 4`). -/
theorem shapeUp_card_le_four (p : Piece) (r : Rotation) :
    (p.shapeUp r).card ≤ 4 := by
  rw [shapeUp_card]

/-- `shape.card ≥ 4` (trivially since `= 4`). -/
theorem shape_card_ge_four (p : Piece) (r : Rotation) : 4 ≤ (p.shape r).card := by
  rw [shape_card]

/-- `shapeUp.card ≥ 4` (trivially since `= 4`). -/
theorem shapeUp_card_ge_four (p : Piece) (r : Rotation) :
    4 ≤ (p.shapeUp r).card := by
  rw [shapeUp_card]

/-- `shape.card < 5` (strict). -/
theorem shape_card_lt_five (p : Piece) (r : Rotation) : (p.shape r).card < 5 := by
  rw [shape_card]; decide

/-- `shapeUp.card < 5` (strict). -/
theorem shapeUp_card_lt_five (p : Piece) (r : Rotation) :
    (p.shapeUp r).card < 5 := by
  rw [shapeUp_card]; decide

/-- `shape.card = 4` is in `[1, 4]` (degenerate interval). -/
theorem shape_card_in_one_to_four (p : Piece) (r : Rotation) :
    1 ≤ (p.shape r).card ∧ (p.shape r).card ≤ 4 :=
  ⟨Nat.le_trans (by decide : (1 : ℕ) ≤ 4) (shape_card_ge_four p r),
    shape_card_le_four p r⟩

/-- `shapeUp.card = 4` is in `[1, 4]`. -/
theorem shapeUp_card_in_one_to_four (p : Piece) (r : Rotation) :
    1 ≤ (p.shapeUp r).card ∧ (p.shapeUp r).card ≤ 4 :=
  ⟨Nat.le_trans (by decide : (1 : ℕ) ≤ 4) (shapeUp_card_ge_four p r),
    shapeUp_card_le_four p r⟩

/-- Every piece has at most 4 rotations. -/
theorem numRotations_le_four (p : Piece) : p.numRotations ≤ 4 := by
  cases p <;> decide

/-- Every piece has at least 1 rotation. -/
theorem numRotations_pos (p : Piece) : 0 < p.numRotations := by
  cases p <;> decide

/-- Every piece has at least 1 rotation (≥ 1 form). -/
theorem numRotations_ge_one (p : Piece) : 1 ≤ p.numRotations :=
  numRotations_pos p

/-- Every piece has nonzero `numRotations`. -/
theorem numRotations_ne_zero (p : Piece) : p.numRotations ≠ 0 :=
  Nat.ne_of_gt (numRotations_pos p)

/-- Strict `< 5` form of `numRotations_le_four`. -/
theorem numRotations_lt_five (p : Piece) : p.numRotations < 5 :=
  Nat.lt_succ_of_le (numRotations_le_four p)

/-- Bundled `[1, 4]` interval for `numRotations`. -/
theorem numRotations_interval (p : Piece) :
    1 ≤ p.numRotations ∧ p.numRotations ≤ 4 :=
  ⟨numRotations_ge_one p, numRotations_le_four p⟩

/-- The natural-orientation shape's row offsets stay within the 4-tall
bounding box. -/
theorem shape_row_lt_four (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shape r, cell.2 < 4 := by
  fin_cases r <;> cases p <;> decide

/-- The natural-orientation shape's column offsets stay within the 4-wide
bounding box. -/
theorem shape_col_lt_four (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shape r, cell.1 < 4 := by
  fin_cases r <;> cases p <;> decide

/-- Bundled bounding-box bound on `shape`: every cell has both column and row
strictly below 4. -/
theorem shape_mem_box (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shape r, cell.1 < 4 ∧ cell.2 < 4 :=
  fun cell hc => ⟨shape_col_lt_four p r cell hc, shape_row_lt_four p r cell hc⟩

/-- Bundled bounding-box bound on `shapeUp`. -/
theorem shapeUp_mem_box (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shapeUp r, cell.1 < 4 ∧ cell.2 < 4 :=
  fun cell hc => ⟨shapeUp_col_lt_four p r cell hc, shapeUp_row_lt_four p r cell hc⟩

/-- `shape r ⊆ range 4 ×ˢ range 4`: the shape fits in the 4×4 bounding box. -/
theorem shape_subset_box (p : Piece) (r : Rotation) :
    p.shape r ⊆ Finset.range 4 ×ˢ Finset.range 4 := by
  intro cell hc
  rw [Finset.mem_product]
  obtain ⟨h1, h2⟩ := shape_mem_box p r cell hc
  exact ⟨Finset.mem_range.mpr h1, Finset.mem_range.mpr h2⟩

/-- `shapeUp r ⊆ range 4 ×ˢ range 4`: the shapeUp fits in the 4×4 box. -/
theorem shapeUp_subset_box (p : Piece) (r : Rotation) :
    p.shapeUp r ⊆ Finset.range 4 ×ˢ Finset.range 4 := by
  intro cell hc
  rw [Finset.mem_product]
  obtain ⟨h1, h2⟩ := shapeUp_mem_box p r cell hc
  exact ⟨Finset.mem_range.mpr h1, Finset.mem_range.mpr h2⟩

/-- `shape r`'s row sup is at most 3 (every cell's row is in `[0, 4)`). -/
theorem shape_sup_row_le_three (p : Piece) (r : Rotation) :
    (p.shape r).sup (·.2) ≤ 3 := by
  refine Finset.sup_le ?_
  intro cell hc
  have := shape_row_lt_four p r cell hc
  omega

/-- `shapeUp r`'s row sup is at most 3. -/
theorem shapeUp_sup_row_le_three (p : Piece) (r : Rotation) :
    (p.shapeUp r).sup (·.2) ≤ 3 := by
  refine Finset.sup_le ?_
  intro cell hc
  have := shapeUp_row_lt_four p r cell hc
  omega

/-- `shape r`'s column sup is at most 3. -/
theorem shape_sup_col_le_three (p : Piece) (r : Rotation) :
    (p.shape r).sup (·.1) ≤ 3 := by
  refine Finset.sup_le ?_
  intro cell hc
  have := shape_col_lt_four p r cell hc
  omega

/-- `shapeUp r`'s column sup is at most 3. -/
theorem shapeUp_sup_col_le_three (p : Piece) (r : Rotation) :
    (p.shapeUp r).sup (·.1) ≤ 3 := by
  refine Finset.sup_le ?_
  intro cell hc
  have := shapeUp_col_lt_four p r cell hc
  omega

/-- `shapeUp` is the "bottom-up" profile: every shapeUp has at least one
cell on row 0. -/
theorem shapeUp_row_zero_mem (p : Piece) (r : Rotation) :
    ∃ cell ∈ p.shapeUp r, cell.2 = 0 := by
  fin_cases r <;> cases p <;> decide

/-- Every `shape` has at least one cell on column 0. -/
theorem shape_zero_mem (p : Piece) (r : Rotation) :
    ∃ cell ∈ p.shape r, cell.1 = 0 := by
  fin_cases r <;> cases p <;> decide

/-- Every `shape` has at least one cell on row 0. -/
theorem shape_row_zero_mem (p : Piece) (r : Rotation) :
    ∃ cell ∈ p.shape r, cell.2 = 0 := by
  fin_cases r <;> cases p <;> decide

/-- `shape` and `shapeUp` have the same cell count (both are tetrominoes). -/
theorem shape_card_eq_shapeUp_card (p : Piece) (r : Rotation) :
    (p.shape r).card = (p.shapeUp r).card := by
  rw [shape_card, shapeUp_card]

/-- The 4×4 bounding box for any rotation has card 16. -/
theorem box_card : (Finset.range 4 ×ˢ Finset.range 4).card = 16 := by
  simp [Finset.card_product]

/-! ### Per-piece per-row footprint bounds

How many cells each tetromino can deposit into a single row of its `shapeUp`
drop profile, in any rotation: `O, S, Z ≤ 2` and `T, L, J ≤ 3` (only the `I`
piece reaches a flat 4-bar). Each comes in a plain `c.2 = e` form and a
drop-shifted `d + c.2 = r` form. Pure shape facts (no board); the board-level
per-row placement bounds that consume them live with the placement layer. -/

/-- **Shape fact (S): every row of an S-piece drop profile holds at most 2 cells.**
In all rotations the S tetromino is two horizontally-adjacent dominoes offset by one, so any
single row of `shapeUp` carries `≤ 2` cells. Proved by a finite split: rows `< 3` are checked
per `(rotation, row)` by `decide` (a pure no-board shape evaluation), and rows `≥ 3` are empty
because every `shapeUp` cell sits below row 3. -/
theorem shapeUp_S_row_card_le_two (rot : Rotation) (e : ℕ) :
    ((Piece.S.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 2 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.S.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.S.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The S shape fact in `d + c.2 = r` (drop-shifted) form: at most 2 `shapeUp` cells land in the
absolute row `r` when the piece is dropped at offset `d`. Reduces to `shapeUp_S_row_card_le_two`
via the forward implication `d + c.2 = r ⇒ c.2 = r - d`. -/
theorem shapeUp_S_shift_row_le_two (rot : Rotation) (d r : ℕ) :
    ((Piece.S.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 2 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_S_row_card_le_two rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

/-- **Shape fact (Z): every row of a Z-piece drop profile holds at most 2 cells.** -/
theorem shapeUp_Z_row_card_le_two (rot : Rotation) (e : ℕ) :
    ((Piece.Z.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 2 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.Z.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.Z.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The Z shape fact in `d + c.2 = r` (drop-shifted) form. -/
theorem shapeUp_Z_shift_row_le_two (rot : Rotation) (d r : ℕ) :
    ((Piece.Z.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 2 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_Z_row_card_le_two rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

/-- **Shape fact (O): every row of an O-piece drop profile holds at most 2 cells.**
The O tetromino is a 2×2 box (one rotation), so each occupied row carries exactly 2 cells and
no row holds more. Same finite split as `shapeUp_S_row_card_le_two`: rows `< 3` by `decide`,
rows `≥ 3` empty (all `shapeUp` cells sit below row 3). -/
theorem shapeUp_O_row_card_le_two (rot : Rotation) (e : ℕ) :
    ((Piece.O.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 2 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.O.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.O.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The O shape fact in `d + c.2 = r` (drop-shifted) form. -/
theorem shapeUp_O_shift_row_le_two (rot : Rotation) (d r : ℕ) :
    ((Piece.O.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 2 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_O_row_card_le_two rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

/-- **Shape fact (T): every row of a T-piece drop profile holds at most 3 cells.**
The T tetromino's flat bar puts 3 cells in one row (tight — `T,0` row 0 and `T,2` row 1 each hit
3); no rotation packs 4 into a row (that is the I piece). Same finite split as the `≤ 2` family:
rows `< 3` by `decide` over all 4 rotations, rows `≥ 3` empty. -/
theorem shapeUp_T_row_card_le_three (rot : Rotation) (e : ℕ) :
    ((Piece.T.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 3 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.T.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.T.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The T shape fact in `d + c.2 = r` (drop-shifted) form. -/
theorem shapeUp_T_shift_row_le_three (rot : Rotation) (d r : ℕ) :
    ((Piece.T.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 3 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_T_row_card_le_three rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

/-- **Shape fact (L): every row of an L-piece drop profile holds at most 3 cells.**
The L tetromino, like `T`, packs at most a 3-cell bar into any single row (`L,0` and `L,2`
realise the tight `3`); no rotation reaches the `I`-piece's 4. Same finite split as the `T`
family: rows `< 3` by `decide` over all 4 rotations, rows `≥ 3` empty. -/
theorem shapeUp_L_row_card_le_three (rot : Rotation) (e : ℕ) :
    ((Piece.L.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 3 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.L.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.L.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The L shape fact in `d + c.2 = r` (drop-shifted) form. -/
theorem shapeUp_L_shift_row_le_three (rot : Rotation) (d r : ℕ) :
    ((Piece.L.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 3 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_L_row_card_le_three rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

/-- **Shape fact (J): every row of a J-piece drop profile holds at most 3 cells.**
The J tetromino is the mirror of `L`; rotations `J,0` and `J,2` realise the tight `3`, and no
rotation reaches 4. Same finite split: rows `< 3` by `decide`, rows `≥ 3` empty. -/
theorem shapeUp_J_row_card_le_three (rot : Rotation) (e : ℕ) :
    ((Piece.J.shapeUp rot).filter (fun c => c.2 = e)).card ≤ 3 := by
  by_cases he : e < 3
  · interval_cases e <;> fin_cases rot <;> decide
  · have hempty : (Piece.J.shapeUp rot).filter (fun c => c.2 = e) = ∅ := by
      have hb : ∀ c ∈ Piece.J.shapeUp rot, c.2 < 3 := by fin_cases rot <;> decide
      rw [Finset.filter_eq_empty_iff]
      intro c hc
      have := hb c hc
      omega
    rw [hempty]
    simp

/-- The J shape fact in `d + c.2 = r` (drop-shifted) form. -/
theorem shapeUp_J_shift_row_le_three (rot : Rotation) (d r : ℕ) :
    ((Piece.J.shapeUp rot).filter (fun c => d + c.2 = r)).card ≤ 3 := by
  refine le_trans (Finset.card_le_card ?_) (shapeUp_J_row_card_le_three rot (r - d))
  intro c hc
  rw [Finset.mem_filter] at hc
  obtain ⟨h1, h2⟩ := hc
  rw [Finset.mem_filter]
  exact ⟨h1, by omega⟩

end Piece
end Tetris
