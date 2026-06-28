import Mathlib

/-!
# Game configuration

Board dimensions for a Tetris variant, mirroring the Rust `TetrisGameConfig`
trait in `crates/tetris-game/src/config.rs`. The canonical game is 10 columns
× 20 rows (`GameConfig.standard`).
-/

namespace Tetris

/-- Board dimensions: `cols` columns by `rows` rows, both positive. -/
structure GameConfig where
  cols : Nat
  rows : Nat
  cols_pos : 0 < cols
  rows_pos : 0 < rows

namespace GameConfig

/-- The canonical Tetris configuration: 10 columns × 20 rows. -/
def standard : GameConfig where
  cols := 10
  rows := 20
  cols_pos := by decide
  rows_pos := by decide

@[simp] theorem standard_cols : standard.cols = 10 := rfl
@[simp] theorem standard_rows : standard.rows = 20 := rfl

/-- A minimal config with `cols ≥ 4` (so any piece can be placed at column 0)
and `rows ≥ 4` (so the low-stack invariant is sensible). Used as a test bed
for degenerate-config cycle search (Roadmap sub-task 1.2). -/
def tiny : GameConfig where
  cols := 4
  rows := 4
  cols_pos := by decide
  rows_pos := by decide

@[simp] theorem tiny_cols : tiny.cols = 4 := rfl
@[simp] theorem tiny_rows : tiny.rows = 4 := rfl

/-- `4 ≤ standard.cols` (standard has 10 columns). Useful as a named
hypothesis to avoid `by decide` in downstream lemmas. -/
theorem standard_four_le_cols : 4 ≤ standard.cols := by decide

/-- `4 ≤ standard.rows` (standard has 20 rows). -/
theorem standard_four_le_rows : 4 ≤ standard.rows := by decide

/-- `4 ≤ tiny.cols` (tiny has 4 columns). -/
theorem tiny_four_le_cols : 4 ≤ tiny.cols := by decide

/-- `tiny.cols = 4` (exact form). -/
theorem tiny_cols_eq_four : tiny.cols = 4 := rfl

/-- `tiny.rows = 4` (exact form). -/
theorem tiny_rows_eq_four : tiny.rows = 4 := rfl

/-- `standard.cols = 10` (named exact form). -/
theorem standard_cols_eq_ten : standard.cols = 10 := rfl

/-- `standard.rows = 20` (named exact form). -/
theorem standard_rows_eq_twenty : standard.rows = 20 := rfl

/-- `standard.cols ≠ tiny.cols`. -/
theorem standard_cols_ne_tiny_cols : standard.cols ≠ tiny.cols := by decide

/-- `standard.rows ≠ tiny.rows`. -/
theorem standard_rows_ne_tiny_rows : standard.rows ≠ tiny.rows := by decide

/-- `tiny.cols < standard.cols` (4 < 10). -/
theorem tiny_cols_lt_standard_cols : tiny.cols < standard.cols := by decide

/-- `tiny.rows < standard.rows` (4 < 20). -/
theorem tiny_rows_lt_standard_rows : tiny.rows < standard.rows := by decide

/-- The tiny grid is smaller than the standard grid (16 < 200). -/
theorem tiny_cols_mul_rows_lt_standard_cols_mul_rows :
    tiny.cols * tiny.rows < standard.cols * standard.rows := by decide

/-- `tiny.cols ≤ standard.cols` (non-strict form of Tick 1397's `<`). -/
theorem tiny_cols_le_standard_cols : tiny.cols ≤ standard.cols :=
  le_of_lt tiny_cols_lt_standard_cols

/-- `tiny.rows ≤ standard.rows` (non-strict form of Tick 1397's `<`). -/
theorem tiny_rows_le_standard_rows : tiny.rows ≤ standard.rows :=
  le_of_lt tiny_rows_lt_standard_rows

/-- Non-strict form of the grid-area comparison. -/
theorem tiny_cols_mul_rows_le_standard_cols_mul_rows :
    tiny.cols * tiny.rows ≤ standard.cols * standard.rows :=
  le_of_lt tiny_cols_mul_rows_lt_standard_cols_mul_rows

/-- `4 ≤ tiny.rows` (tiny has 4 rows). -/
theorem tiny_four_le_rows : 4 ≤ tiny.rows := by decide

/-- Standard config's column count is positive. -/
theorem standard_cols_pos : 0 < standard.cols := by decide

/-- Standard config's row count is positive. -/
theorem standard_rows_pos : 0 < standard.rows := by decide

/-- Tiny config's column count is positive. -/
theorem tiny_cols_pos : 0 < tiny.cols := by decide

/-- Tiny config's row count is positive. -/
theorem tiny_rows_pos : 0 < tiny.rows := by decide

/-- Standard config's total cell count is 200. -/
theorem standard_cols_mul_rows : standard.cols * standard.rows = 200 := by decide

/-- Tiny config's total cell count is 16. -/
theorem tiny_cols_mul_rows : tiny.cols * tiny.rows = 16 := by decide

/-- Standard config: column count is not zero. -/
theorem standard_cols_ne_zero : standard.cols ≠ 0 := by decide

/-- Standard config: row count is not zero. -/
theorem standard_rows_ne_zero : standard.rows ≠ 0 := by decide

/-- Tiny config: column count is not zero. -/
theorem tiny_cols_ne_zero : tiny.cols ≠ 0 := by decide

/-- Tiny config: row count is not zero. -/
theorem tiny_rows_ne_zero : tiny.rows ≠ 0 := by decide

/-- Standard config: `1 ≤ cols`. -/
theorem standard_one_le_cols : 1 ≤ standard.cols := by decide

/-- Standard config: `1 ≤ rows`. -/
theorem standard_one_le_rows : 1 ≤ standard.rows := by decide

/-- Tiny config: `1 ≤ cols`. -/
theorem tiny_one_le_cols : 1 ≤ tiny.cols := by decide

/-- Tiny config: `1 ≤ rows`. -/
theorem tiny_one_le_rows : 1 ≤ tiny.rows := by decide

/-- Standard has more rows than columns (20 > 10). -/
theorem standard_cols_lt_rows : standard.cols < standard.rows := by decide

/-- Tiny has equal rows and columns (4 = 4). -/
theorem tiny_cols_eq_rows : tiny.cols = tiny.rows := by decide

/-- Standard config is not tiny config (their col counts differ). -/
theorem standard_ne_tiny : standard ≠ tiny := by
  intro h
  have : standard.cols = tiny.cols := by rw [h]
  exact absurd this (by decide)

end GameConfig
end Tetris
