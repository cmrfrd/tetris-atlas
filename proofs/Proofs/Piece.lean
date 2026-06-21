import Mathlib

/-!
# Tetrominoes

The 7 standard pieces and their rotation shapes, transcribed from the column
bitmask data in `crates/tetris-game/src/tetris.rs` (`tetris_piece_data`).

Each shape is the set of occupied cells within the piece's 4×4 bounding box,
given as `(column, rowFromTop)` offsets. Rotations are indexed `0..3` clockwise;
pieces with fewer than 4 distinct rotations repeat (O has 1; I, S, Z have 2).

`shapeUp` re-expresses a shape bottom-up (lowest occupied cell at row offset 0),
which is the orientation used for hard-drop placement.
-/

namespace Tetris

/-- The 7 standard tetrominoes. -/
inductive Piece
  | O | I | S | Z | T | L | J
deriving DecidableEq, Repr, Fintype

/-- A clockwise rotation state, `0 = 0°`, `1 = 90°`, `2 = 180°`, `3 = 270°`. -/
abbrev Rotation := Fin 4

/-- A cell offset: `(column, row)`. -/
abbrev PieceCell := ℕ × ℕ

namespace Piece

/-- Number of distinct rotations for each piece. -/
def numRotations : Piece → Nat
  | O => 1
  | I | S | Z => 2
  | T | L | J => 4

/-- Occupied cells of piece `p` at rotation `r`, as `(column, rowFromTop)`
offsets. Transcribed directly from the reference engine's bitmasks. -/
def shape (p : Piece) (r : Rotation) : Finset PieceCell :=
  match p, r.val with
  | .O, _          => {(0, 0), (0, 1), (1, 0), (1, 1)}
  | .I, 0 | .I, 2  => {(0, 0), (1, 0), (2, 0), (3, 0)}
  | .I, _          => {(0, 0), (0, 1), (0, 2), (0, 3)}
  | .S, 0 | .S, 2  => {(0, 1), (1, 0), (1, 1), (2, 0)}
  | .S, _          => {(0, 0), (0, 1), (1, 1), (1, 2)}
  | .Z, 0 | .Z, 2  => {(0, 0), (1, 0), (1, 1), (2, 1)}
  | .Z, _          => {(0, 1), (0, 2), (1, 0), (1, 1)}
  | .T, 0          => {(0, 0), (1, 0), (1, 1), (2, 0)}
  | .T, 1          => {(0, 1), (1, 0), (1, 1), (1, 2)}
  | .T, 2          => {(0, 1), (1, 0), (1, 1), (2, 1)}
  | .T, _          => {(0, 0), (0, 1), (0, 2), (1, 1)}
  | .L, 0          => {(0, 1), (1, 1), (2, 0), (2, 1)}
  | .L, 1          => {(0, 0), (0, 1), (0, 2), (1, 2)}
  | .L, 2          => {(0, 0), (0, 1), (1, 0), (2, 0)}
  | .L, _          => {(0, 0), (1, 0), (1, 1), (1, 2)}
  | .J, 0          => {(0, 0), (0, 1), (1, 1), (2, 1)}
  | .J, 1          => {(0, 0), (0, 1), (0, 2), (1, 0)}
  | .J, 2          => {(0, 0), (1, 0), (2, 0), (2, 1)}
  | .J, _          => {(0, 2), (1, 0), (1, 1), (1, 2)}

/-- The shape re-expressed bottom-up: the lowest occupied cell sits at row
offset 0. Used as the drop profile for hard-drop placement. -/
def shapeUp (p : Piece) (r : Rotation) : Finset PieceCell :=
  let s := p.shape r
  let maxT := s.sup (·.2)
  s.image (fun cell => (cell.1, maxT - cell.2))

/-- `numRotations Piece.O = 1`. -/
@[simp] theorem numRotations_O : Piece.O.numRotations = 1 := rfl

/-- `numRotations Piece.I = 2`. -/
@[simp] theorem numRotations_I : Piece.I.numRotations = 2 := rfl

/-- `numRotations Piece.S = 2`. -/
@[simp] theorem numRotations_S : Piece.S.numRotations = 2 := rfl

/-- `numRotations Piece.Z = 2`. -/
@[simp] theorem numRotations_Z : Piece.Z.numRotations = 2 := rfl

/-- `numRotations Piece.T = 4`. -/
@[simp] theorem numRotations_T : Piece.T.numRotations = 4 := rfl

/-- `numRotations Piece.L = 4`. -/
@[simp] theorem numRotations_L : Piece.L.numRotations = 4 := rfl

/-- `numRotations Piece.J = 4`. -/
@[simp] theorem numRotations_J : Piece.J.numRotations = 4 := rfl

/-- Every piece has at least 1 rotation. (Companion to `numRotations_pos`
in `Theorems/PieceGeometry.lean`.) -/
theorem one_le_numRotations (p : Piece) : 1 ≤ p.numRotations := by
  cases p <;> decide

/-- There are exactly 7 distinct Tetris pieces. -/
@[simp] theorem card_eq_seven : Fintype.card Piece = 7 := by decide

/-- The universe of `Piece` has card 7. -/
@[simp] theorem univ_card : (Finset.univ : Finset Piece).card = 7 := by
  rw [Finset.card_univ, card_eq_seven]

/-- There are exactly 7 distinct Tetris pieces — `Nat.lt` form. -/
theorem card_lt_eight : Fintype.card Piece < 8 := by decide

/-- There are exactly 7 distinct Tetris pieces — positive form. -/
theorem card_pos : 0 < Fintype.card Piece := by decide

/-- The universe of `Piece` is nonempty. -/
theorem univ_nonempty : (Finset.univ : Finset Piece).Nonempty :=
  ⟨O, Finset.mem_univ O⟩

/-- The universe of `Piece` is not empty. -/
theorem univ_ne_empty : (Finset.univ : Finset Piece) ≠ ∅ :=
  univ_nonempty.ne_empty

/-- There exist two distinct pieces. -/
theorem exists_distinct_pair : ∃ p q : Piece, p ≠ q :=
  ⟨O, I, by decide⟩

/-- The universe of `Piece` has more than one element. -/
theorem one_lt_univ_card : 1 < (Finset.univ : Finset Piece).card := by
  rw [univ_card]; decide

/-- `numRotations p = 1 ↔ p = O`. -/
theorem numRotations_eq_one_iff (p : Piece) : p.numRotations = 1 ↔ p = O := by
  cases p <;> simp

/-- `numRotations p = 2 ↔ p ∈ {I, S, Z}`. -/
theorem numRotations_eq_two_iff (p : Piece) :
    p.numRotations = 2 ↔ p = I ∨ p = S ∨ p = Z := by
  cases p <;> simp

/-- `numRotations p = 4 ↔ p ∈ {T, L, J}`. -/
theorem numRotations_eq_four_iff (p : Piece) :
    p.numRotations = 4 ↔ p = T ∨ p = L ∨ p = J := by
  cases p <;> simp

/-- `numRotations p` is `1`, `2`, or `4`. -/
theorem numRotations_eq_one_or_two_or_four (p : Piece) :
    p.numRotations = 1 ∨ p.numRotations = 2 ∨ p.numRotations = 4 := by
  cases p <;> decide

/-- `numRotations p` divides 4. -/
theorem numRotations_dvd_four (p : Piece) : p.numRotations ∣ 4 := by
  cases p <;> decide

/-- `4 % numRotations p = 0`. -/
theorem four_mod_numRotations_eq_zero (p : Piece) :
    4 % p.numRotations = 0 := by
  cases p <;> decide

end Piece
end Tetris
