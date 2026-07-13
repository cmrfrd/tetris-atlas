import Mathlib
import Proofs.Model.Piece

/-!
# The slot algebra: flush-landing profiles and their exclusivity theorems

`colBot p r i` / `colTop p r i` — the lowest/highest occupied row of column
`i` in `p`'s drop profile. A rotation seats flush on a surface segment iff
the segment's height differences match the bottom profile; the landing
raises column `i` to `off + colTop + 1`. The exclusivity theorems decide,
once and for all, which pieces can consume which local surface shapes:

- a standing ±1 step admits ONLY vertical S/Z and vertical T (`sStep_exclusive`,
  `zStep_exclusive`) — so schedule designs that redirect O/L/J onto step
  zones are impossible (finding F1, 2026-07-12);
- 2-wide flat pairs admit exactly O, L (rot 1), J (rot 3)
  (`flatPair_receivers`);
- ±2 steps are the exclusive L/J currency (`twoStep_left_only_L`,
  `twoStep_right_only_J`), and both landings FLATTEN the step into a flat
  pair (the tops lemmas) — the O/L/J flat/±2-step economy.
-/

namespace Tetris
namespace Piece

/-- Lowest occupied row of column `i` in the drop profile (0 if empty).
Tetromino drop profiles occupy rows 0–3 only, so a membership chain
suffices — and, unlike `Finset.min`, it reduces under `decide`. -/
def colBot (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  if (i, 0) ∈ p.shapeUp r then 0
  else if (i, 1) ∈ p.shapeUp r then 1
  else if (i, 2) ∈ p.shapeUp r then 2
  else if (i, 3) ∈ p.shapeUp r then 3 else 0

/-- Highest occupied row of column `i` in the drop profile (0 if empty). -/
def colTop (p : Piece) (r : Rotation) (i : ℕ) : ℕ :=
  if (i, 3) ∈ p.shapeUp r then 3
  else if (i, 2) ∈ p.shapeUp r then 2
  else if (i, 1) ∈ p.shapeUp r then 1
  else if (i, 0) ∈ p.shapeUp r then 0 else 0

/-- A rotation is 2-wide when it occupies exactly columns 0 and 1. -/
def TwoWide (p : Piece) (r : Rotation) : Prop :=
  (∀ cell ∈ p.shapeUp r, cell.1 < 2) ∧ (∃ cell ∈ p.shapeUp r, cell.1 = 1)

instance (p : Piece) (r : Rotation) : Decidable (TwoWide p r) := by
  unfold TwoWide; infer_instance

/-- **A standing S-step admits only S and T.** Any 2-wide rotation whose
bottom profile is `(1, 0)` belongs to S (vertical) or T (rot 1). -/
theorem sStep_exclusive : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 + 1 → p = Piece.S ∨ p = Piece.T := by
  decide

/-- **A standing Z-step admits only Z and T** (mirror). -/
theorem zStep_exclusive : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 1 = colBot p r 0 + 1 → p = Piece.Z ∨ p = Piece.T := by
  decide

/-- **A 2-wide flat pair admits only O, L, J.** -/
theorem flatPair_receivers : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 → p = Piece.O ∨ p = Piece.L ∨ p = Piece.J := by
  decide

/-- **A left-high ±2 step is exclusively L's** (rot 3). -/
theorem twoStep_left_only_L : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 0 = colBot p r 1 + 2 → p = Piece.L := by
  decide

/-- **A right-high ±2 step is exclusively J's** (rot 1). -/
theorem twoStep_right_only_J : ∀ (p : Piece) (r : Rotation), TwoWide p r →
    colBot p r 1 = colBot p r 0 + 2 → p = Piece.J := by
  decide

/-! ## Tops: what each 2-wide landing leaves behind -/

/-- Vertical S preserves the S-step: tops `(2, 1)`. -/
theorem tops_vertS : colTop Piece.S 1 0 = 2 ∧ colTop Piece.S 1 1 = 1 := by
  decide

/-- Vertical Z preserves the Z-step: tops `(1, 2)`. -/
theorem tops_vertZ : colTop Piece.Z 1 0 = 1 ∧ colTop Piece.Z 1 1 = 2 := by
  decide

/-- T rot 1 consumes an S-step and leaves a Z-step: tops `(1, 2)`. -/
theorem tops_T1 : colTop Piece.T 1 0 = 1 ∧ colTop Piece.T 1 1 = 2 := by
  decide

/-- T rot 3 consumes a Z-step and leaves an S-step: tops `(2, 1)`. -/
theorem tops_T3 : colTop Piece.T 3 0 = 2 ∧ colTop Piece.T 3 1 = 1 := by
  decide

/-- O preserves the flat pair: tops `(1, 1)`. -/
theorem tops_O : colTop Piece.O 0 0 = 1 ∧ colTop Piece.O 0 1 = 1 := by
  decide

/-- L rot 1 on a flat pair leaves a left-high ±2 step: tops `(2, 0)`. -/
theorem tops_L1 : colTop Piece.L 1 0 = 2 ∧ colTop Piece.L 1 1 = 0 := by
  decide

/-- **L rot 3 consumes a left-high ±2 step and FLATTENS it**: tops `(2, 2)`. -/
theorem tops_L3 : colTop Piece.L 3 0 = 2 ∧ colTop Piece.L 3 1 = 2 := by
  decide

/-- J rot 3 on a flat pair leaves a right-high ±2 step: tops `(0, 2)`. -/
theorem tops_J3 : colTop Piece.J 3 0 = 0 ∧ colTop Piece.J 3 1 = 2 := by
  decide

/-- **J rot 1 consumes a right-high ±2 step and FLATTENS it**: tops `(2, 2)`. -/
theorem tops_J1 : colTop Piece.J 1 0 = 2 ∧ colTop Piece.J 1 1 = 2 := by
  decide

end Piece
end Tetris
