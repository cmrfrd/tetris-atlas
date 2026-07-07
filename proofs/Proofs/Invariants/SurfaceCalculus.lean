import Mathlib
import Proofs.Model.Placement
import Proofs.Invariants.Wqo

/-!
# The surface calculus — placement as closed-form skyline algebra

The symbolic engine for the constructive (non-enumerative) route: the
post-placement skyline in **closed form**. For any board and placement,

* `dropOffset` is already a finite `sup` of height differences
  (`Placement.dropOffset_eq_sup`);
* `colHeight_dropped_eq` computes the dropped piece's contribution to each
  column: the `sup`, over the piece cells landing in that column, of
  `dropOffset + cellRow + 1` (zero for untouched columns);
* `colHeight_place_eq` assembles the update:
  `height_after j = max (height_before j) (contribution j)`.

Together with `SurfaceFiber` (placement reads only the skyline) and
`Confluence.dropOffset_skyline_sub` (uniform shifts commute), any claim
"piece `p` placed at column `c` on a surface of shape `σ(params)` yields
shape `σ'(params)`" reduces to finite `ℕ` arithmetic over the piece's four
cells — dischargeable by `omega` after instantiation, with **no state
enumeration anywhere**. This is the lemma layer on which symbolic closed
families (valley–pocket designs and their per-piece closure laws) are
built and proven.
-/

namespace Tetris
namespace SurfaceCalculus

open Tetris

/-- The piece cells of `pl` that land in absolute column `j`. -/
def cellsInCol (pl : Placement) (j : ℕ) : Finset PieceCell :=
  pl.shapeUp.filter (fun cell => pl.col + cell.1 = j)

/-- **The dropped piece's skyline contribution, in closed form**: in each
column, the `sup` over its cells there of `dropOffset + row + 1`; zero for
columns the piece does not touch. -/
theorem colHeight_dropped_eq (b : Board) (pl : Placement) (j : ℕ) :
    (pl.dropped b).colHeight j
      = (cellsInCol pl j).sup (fun cell => pl.dropOffset b + cell.2 + 1) := by
  unfold Board.colHeight Board.colRows cellsInCol
  rw [Placement.dropped_eq_image]
  rw [Finset.filter_image, Finset.image_image]
  rw [Finset.sup_image]
  apply Finset.sup_congr
  · congr 1
  · intro cell _
    simp

/-- **The surface update formula**: placing `pl` on `b` raises each column
to the max of its old height and the piece's contribution there. With
`dropOffset_eq_sup` this is the complete closed-form skyline calculus. -/
theorem colHeight_place_eq (b : Board) (pl : Placement) (j : ℕ) :
    (pl.place b).colHeight j
      = max (b.colHeight j)
          ((cellsInCol pl j).sup (fun cell => pl.dropOffset b + cell.2 + 1)) := by
  rw [Placement.place_eq_union_dropped, WqoCarrier.colHeight_union,
    colHeight_dropped_eq]

/-- Columns the piece does not touch keep their height. -/
theorem colHeight_place_of_cellsInCol_empty (b : Board) (pl : Placement)
    {j : ℕ} (h : cellsInCol pl j = ∅) :
    (pl.place b).colHeight j = b.colHeight j := by
  rw [colHeight_place_eq, h, Finset.sup_empty]
  simp

end SurfaceCalculus
end Tetris
