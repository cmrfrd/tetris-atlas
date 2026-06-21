import Mathlib
import Proofs.Piece
import Proofs.Board
import Proofs.Theorems.PieceGeometry

/-!
# Hard-drop placements

A placement chooses a piece, a rotation, and a leftmost target column. The
piece falls straight down (`only hard drop`) until it rests on the stack or the
floor, then merges into the board. Line clears are applied separately by
`Board.clearLines` (see `applyStep`).

The hard-drop offset is computed cell-wise: for each piece cell, the largest
fall keeps that cell at or above the relevant column height; the actual offset
is the supremum, which guarantees the dropped cells never overlap the stack.
-/

namespace Tetris

/-- A hard-drop placement: a piece, a rotation, and the leftmost target column. -/
structure Placement where
  piece : Piece
  rot : Rotation
  col : ℕ
deriving DecidableEq, Repr

namespace Placement

/-- The piece's bottom-up drop profile. -/
def shapeUp (pl : Placement) : Finset Coord := pl.piece.shapeUp pl.rot

/-- The drop profile translated to absolute columns at vertical offset `d`. -/
def cellsAt (pl : Placement) (d : ℕ) : Board :=
  pl.shapeUp.image (fun cell => (pl.col + cell.1, d + cell.2))

/-- Hard-drop offset: the furthest the piece can fall onto board `b`. -/
def dropOffset (b : Board) (pl : Placement) : ℕ :=
  pl.shapeUp.sup (fun cell => b.colHeight (pl.col + cell.1) - cell.2)

/-- The cells occupied after hard-dropping onto `b`. -/
def dropped (b : Board) (pl : Placement) : Board := pl.cellsAt (pl.dropOffset b)

/-- Merge the dropped piece into the board (before line clears). -/
def place (b : Board) (pl : Placement) : Board := b ∪ pl.dropped b

/-- One full move: hard-drop the piece, then clear completed lines. -/
def applyStep (cfg : GameConfig) (b : Board) (pl : Placement) : Board :=
  Board.clearLines cfg (pl.place b)

/-- Generic membership lower bound for the drop-offset supremum. Stated with an
opaque height function `H` so that elaboration never unfolds `colHeight` (which
is itself a `Finset.sup`) inside the `le_sup` unification. -/
theorem le_sup_sub (s : Finset Coord) (H : ℕ → ℕ) (col : ℕ) {cell : Coord}
    (h : cell ∈ s) :
    H (col + cell.1) - cell.2 ≤ s.sup (fun c => H (col + c.1) - c.2) :=
  Finset.le_sup (f := fun c => H (col + c.1) - c.2) h

/-- The translation of the drop profile into absolute coordinates is injective. -/
theorem cellsAt_injective (pl : Placement) (d : ℕ) :
    Function.Injective (fun cell : Coord => (pl.col + cell.1, d + cell.2)) := by
  intro x y hxy
  simp only [Prod.mk.injEq, add_right_inj] at hxy
  exact Prod.ext hxy.1 hxy.2

/-- A dropped piece occupies exactly 4 cells. -/
theorem card_dropped (b : Board) (pl : Placement) : (pl.dropped b).card = 4 := by
  unfold dropped cellsAt shapeUp
  rw [Finset.card_image_of_injective _ (cellsAt_injective pl _)]
  exact Piece.shapeUp_card pl.piece pl.rot

-- The hard-dropped cells never overlap the existing stack: every dropped cell
-- sits at or above its column height, while every filled cell sits strictly
-- below it.
theorem dropped_disjoint (b : Board) (pl : Placement) : Disjoint (pl.dropped b) b := by
  refine Finset.disjoint_left.2 ?_
  intro c hc hcb
  have hc2 : c ∈ Finset.image (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2))
      pl.shapeUp := hc
  obtain ⟨cell, hcell, rfl⟩ := Finset.mem_image.1 hc2
  have hle : b.colHeight (pl.col + cell.1) - cell.2 ≤ pl.dropOffset b :=
    le_sup_sub pl.shapeUp b.colHeight pl.col hcell
  have hlt : pl.dropOffset b + cell.2 < b.colHeight (pl.col + cell.1) := Board.lt_colHeight hcb
  generalize b.colHeight (pl.col + cell.1) = H at hle hlt
  generalize pl.dropOffset b = D at hle hlt
  omega

/-- **Placing a piece adds exactly 4 filled cells** (before any line clears). -/
theorem count_place (b : Board) (pl : Placement) :
    (pl.place b).count = b.count + 4 := by
  unfold place Board.count
  rw [Finset.card_union_of_disjoint (pl.dropped_disjoint b).symm, card_dropped]

/-- The cells of `cellsAt d` form a Finset of size 4. -/
theorem card_cellsAt (pl : Placement) (d : ℕ) :
    (pl.cellsAt d).card = 4 := by
  unfold cellsAt shapeUp
  rw [Finset.card_image_of_injective _ (cellsAt_injective pl _)]
  exact Piece.shapeUp_card pl.piece pl.rot

/-- `cellsAt d` has positive cardinality. -/
theorem card_cellsAt_pos (pl : Placement) (d : ℕ) :
    0 < (pl.cellsAt d).card := by
  rw [card_cellsAt]; decide

/-- `cellsAt d` is nonempty. -/
theorem cellsAt_nonempty (pl : Placement) (d : ℕ) :
    (pl.cellsAt d).Nonempty :=
  Finset.card_pos.mp (card_cellsAt_pos pl d)

/-- `cellsAt d` is not empty. -/
theorem cellsAt_ne_empty (pl : Placement) (d : ℕ) :
    pl.cellsAt d ≠ ∅ :=
  (cellsAt_nonempty pl d).ne_empty

/-- `Board.count` form: `(cellsAt d).count = 4`. -/
theorem count_cellsAt (pl : Placement) (d : ℕ) :
    (pl.cellsAt d).count = 4 :=
  card_cellsAt pl d

/-- `cellsAt d` has count `≥ 1`. -/
theorem one_le_count_cellsAt (pl : Placement) (d : ℕ) :
    1 ≤ (pl.cellsAt d).count := by
  rw [count_cellsAt]; decide

/-- `dropped b` is definitionally `cellsAt (dropOffset b)`. -/
theorem dropped_eq_cellsAt (b : Board) (pl : Placement) :
    pl.dropped b = pl.cellsAt (pl.dropOffset b) := rfl

/-- `(dropped b).count = 4`. -/
theorem count_dropped' (b : Board) (pl : Placement) :
    (pl.dropped b).count = 4 :=
  card_dropped b pl

/-- `place b = b ∪ dropped b`: the placement is the union with the dropped piece. -/
theorem place_eq_union_dropped (b : Board) (pl : Placement) :
    pl.place b = b ∪ pl.dropped b := rfl

/-- `applyStep cfg b pl = clearLines cfg (place b pl)` (definitional). -/
theorem applyStep_eq_clearLines_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    pl.applyStep cfg b = Board.clearLines cfg (pl.place b) := rfl

/-- `dropOffset b = shapeUp.sup (...)` (definitional). -/
theorem dropOffset_eq_sup (b : Board) (pl : Placement) :
    pl.dropOffset b =
      pl.shapeUp.sup (fun cell => b.colHeight (pl.col + cell.1) - cell.2) := rfl

/-- The dropped piece is the image of shapeUp under the translation. -/
theorem dropped_eq_image (b : Board) (pl : Placement) :
    pl.dropped b =
      pl.shapeUp.image
        (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) := rfl

/-- After a placement, the board has more cells than before. -/
theorem count_lt_count_place (b : Board) (pl : Placement) :
    b.count < (pl.place b).count := by
  rw [count_place]
  omega

/-- `b ⊆ pl.place b`: the original board sits inside the post-placement board. -/
theorem subset_place (b : Board) (pl : Placement) : b ⊆ pl.place b := by
  unfold place
  exact Finset.subset_union_left

/-- The dropped cells sit inside the post-placement board. -/
theorem dropped_subset_place (b : Board) (pl : Placement) :
    pl.dropped b ⊆ pl.place b := by
  unfold place
  exact Finset.subset_union_right

/-- The post-placement board's count is `b.count + 4`, so it's a strict superset. -/
theorem count_place_eq_succ_succ_succ_succ (b : Board) (pl : Placement) :
    (pl.place b).count = b.count + 1 + 1 + 1 + 1 := by
  rw [count_place]

/-- The dropped piece is always nonempty. -/
theorem dropped_nonempty (b : Board) (pl : Placement) :
    (pl.dropped b).Nonempty := by
  have h : (pl.dropped b).card = 4 := card_dropped b pl
  exact Finset.card_pos.mp (by rw [h]; decide)

/-- The post-placement board is always nonempty. -/
theorem place_nonempty (b : Board) (pl : Placement) :
    (pl.place b).Nonempty :=
  (pl.dropped_nonempty b).mono (dropped_subset_place b pl)

/-- The post-placement board is never empty. -/
theorem place_ne_empty (b : Board) (pl : Placement) :
    pl.place b ≠ ∅ :=
  (place_nonempty b pl).ne_empty

/-- The dropped piece has positive cell count. -/
theorem dropped_card_pos (b : Board) (pl : Placement) :
    0 < (pl.dropped b).card := by
  rw [card_dropped]; decide

/-- The post-placement board has positive count. -/
theorem place_count_pos (b : Board) (pl : Placement) :
    0 < (pl.place b).count :=
  Finset.card_pos.mpr (place_nonempty b pl)

/-- The post-placement count is at least 4. -/
theorem four_le_count_place (b : Board) (pl : Placement) :
    4 ≤ (pl.place b).count := by
  rw [count_place]; omega

/-- The dropped piece's count equals the shapeUp's card (both 4). -/
theorem dropped_count_eq_shapeUp_card (b : Board) (pl : Placement) :
    (pl.dropped b).count = pl.shapeUp.card := by
  unfold Board.count shapeUp
  rw [card_dropped, Piece.shapeUp_card]

/-- The dropped piece's count equals 4 (`Board.count` form of `card_dropped`). -/
theorem count_dropped (b : Board) (pl : Placement) :
    (pl.dropped b).count = 4 :=
  card_dropped b pl

/-- The placement's shapeUp has exactly 4 cells. -/
theorem shapeUp_card (pl : Placement) : pl.shapeUp.card = 4 :=
  Piece.shapeUp_card pl.piece pl.rot

/-- The placement's shapeUp is always nonempty. -/
theorem shapeUp_nonempty (pl : Placement) : pl.shapeUp.Nonempty :=
  Piece.shapeUp_nonempty pl.piece pl.rot

/-- The placement's shapeUp has positive cardinality. -/
theorem shapeUp_card_pos (pl : Placement) : 0 < pl.shapeUp.card := by
  rw [shapeUp_card]; decide

/-- The placement's shapeUp has cardinality ≠ 0. -/
theorem shapeUp_card_ne_zero (pl : Placement) : pl.shapeUp.card ≠ 0 :=
  Nat.ne_of_gt (shapeUp_card_pos pl)

/-- The placement's shapeUp is not empty. -/
theorem shapeUp_ne_empty (pl : Placement) : pl.shapeUp ≠ ∅ :=
  (shapeUp_nonempty pl).ne_empty

/-- The placement's shapeUp has cardinality `< 5`. -/
theorem shapeUp_card_lt_five (pl : Placement) : pl.shapeUp.card < 5 := by
  rw [shapeUp_card]; decide

/-- The placement's shapeUp has cardinality `≤ 4`. -/
theorem shapeUp_card_le_four (pl : Placement) : pl.shapeUp.card ≤ 4 := by
  rw [shapeUp_card]

/-- The placement's shapeUp has cardinality `≥ 4`. -/
theorem shapeUp_card_ge_four (pl : Placement) : 4 ≤ pl.shapeUp.card := by
  rw [shapeUp_card]

/-- The placement's shapeUp has cardinality in `[4, 4]` (degenerate). -/
theorem shapeUp_card_interval (pl : Placement) :
    4 ≤ pl.shapeUp.card ∧ pl.shapeUp.card ≤ 4 :=
  ⟨shapeUp_card_ge_four pl, shapeUp_card_le_four pl⟩


end Placement
end Tetris
