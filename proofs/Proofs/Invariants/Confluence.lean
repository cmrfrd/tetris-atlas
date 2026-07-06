import Mathlib
import Proofs.Combinatorics.BoardCount
import Proofs.Invariants.HoledSkyline

/-!
# Order-confluence: column-disjoint placements commute

The keystone of the confluence route to `TetrisSolvable`. In the adversarial
game the opponent's only power is the ORDER in which the bag's seven pieces
arrive. A strategy that answers each piece inside its own column zone makes
that order irrelevant, because hard-drop placements with disjoint column
footprints COMMUTE as board transformations: neither placement changes the
column heights the other reads (`Board.colHeight_place_of_notMem_cols`), so
each piece hard-drops to the same cells whether it lands before or after the
other, and the final board is the same union either way.

This file proves the pairwise commutation spine:

* `Placement.footCols` — the absolute columns a placement occupies;
* `Placement.ColsDisjoint` — two placements' column footprints do not meet;
* `Placement.dropOffset_place_of_colsDisjoint` — a column-disjoint placement
  never changes a piece's hard-drop offset;
* `Placement.dropped_place_of_colsDisjoint` — nor its landing cells;
* `Placement.place_comm_of_colsDisjoint` — `place` commutes.

Downstream (later files in the route): all `7!` within-bag arrival orders of a
zone-local response collapse to one canonical trajectory, so a cooperative
witness loop certifies the adversarial bag — the reduction that turns the
cooperative lasso into an ingredient of the adversarial proof.
-/

namespace Tetris
namespace Placement

/-- The absolute columns a placement's cells occupy (its column footprint).
The hard-drop offset and landing cells of a placement depend on the board
only through `colHeight` on `footCols` — so boards agreeing there are
indistinguishable to the placement. -/
def footCols (pl : Placement) : Finset ℕ :=
  pl.shapeUp.image (fun cell => pl.col + cell.1)

/-- **Column-disjointness of two placements**: no absolute column is occupied
by both. The zone discipline of a confluent strategy: each bag piece answers
in its own zone, so any two same-bag placements satisfy this. -/
def ColsDisjoint (pl1 pl2 : Placement) : Prop :=
  ∀ c1 ∈ pl1.shapeUp, ∀ c2 ∈ pl2.shapeUp, pl1.col + c1.1 ≠ pl2.col + c2.1

instance (pl1 pl2 : Placement) : Decidable (ColsDisjoint pl1 pl2) := by
  unfold ColsDisjoint; infer_instance

/-- Column-disjointness is symmetric. -/
theorem ColsDisjoint.symm {pl1 pl2 : Placement} (h : ColsDisjoint pl1 pl2) :
    ColsDisjoint pl2 pl1 :=
  fun c2 hc2 c1 hc1 => (h c1 hc1 c2 hc2).symm

/-- `ColsDisjoint` is exactly disjointness of the column footprints. -/
theorem colsDisjoint_iff_disjoint_footCols (pl1 pl2 : Placement) :
    ColsDisjoint pl1 pl2 ↔ Disjoint pl1.footCols pl2.footCols := by
  constructor
  · intro h
    refine Finset.disjoint_left.2 ?_
    intro j hj1 hj2
    obtain ⟨c1, hc1, rfl⟩ := Finset.mem_image.1 hj1
    obtain ⟨c2, hc2, he⟩ := Finset.mem_image.1 hj2
    exact h c1 hc1 c2 hc2 he.symm
  · intro h c1 hc1 c2 hc2 he
    exact Finset.disjoint_left.1 h
      (Finset.mem_image_of_mem _ hc1)
      (he ▸ Finset.mem_image_of_mem _ hc2)

/-- **A column-disjoint placement never changes a piece's hard-drop offset.**
`dropOffset` reads the board only through `colHeight` on the piece's own
columns, and a disjoint placement preserves those heights (locality). -/
theorem dropOffset_place_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.dropOffset (pl1.place b) = pl2.dropOffset b := by
  rw [dropOffset_eq_sup, dropOffset_eq_sup]
  refine Finset.sup_congr rfl fun c2 hc2 => ?_
  simp only [Board.colHeight_place_of_notMem_cols b pl1 (pl2.col + c2.1)
    (fun c1 hc1 => (hd c1 hc1 c2 hc2).symm)]

/-- **A column-disjoint placement never changes a piece's landing cells.** -/
theorem dropped_place_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.dropped (pl1.place b) = pl2.dropped b := by
  rw [dropped_eq_cellsAt, dropped_eq_cellsAt, dropOffset_place_of_colsDisjoint b hd]

/-- **Column-disjoint placements commute.** Placing `pl1` then `pl2` produces
the same board as placing `pl2` then `pl1`: each piece drops to the same cells
in either order (landing-cell invariance), and the two unions agree. This is
the pairwise engine of order-confluence — the algebraic fact that deletes the
adversary's order power over a zone-local strategy. -/
theorem place_comm_of_colsDisjoint (b : Board) {pl1 pl2 : Placement}
    (hd : ColsDisjoint pl1 pl2) :
    pl2.place (pl1.place b) = pl1.place (pl2.place b) := by
  have h2 : pl2.dropped (pl1.place b) = pl2.dropped b :=
    dropped_place_of_colsDisjoint b hd
  have h1 : pl1.dropped (pl2.place b) = pl1.dropped b :=
    dropped_place_of_colsDisjoint b hd.symm
  show pl1.place b ∪ pl2.dropped (pl1.place b) = pl2.place b ∪ pl1.dropped (pl2.place b)
  rw [h1, h2]
  show (b ∪ pl1.dropped b) ∪ pl2.dropped b = (b ∪ pl2.dropped b) ∪ pl1.dropped b
  exact Finset.union_right_comm b (pl1.dropped b) (pl2.dropped b)

/-- **A full move without completed rows is a bare placement.** When the merge
completes no row, `clearLines` is the identity (`clearLines_eq_self_of_no_fullRows`)
and `applyStep` collapses to `place`. Mid-bag moves of a confluent design are of
this shape: rows are completed only at designated drain points, so between
drains the game evolves by pure (commuting) placements. -/
theorem applyStep_eq_place_of_no_fullRows (cfg : GameConfig) (b : Board)
    (pl : Placement) (h : Board.fullRows cfg (pl.place b) = ∅) :
    pl.applyStep cfg b = pl.place b := by
  rw [applyStep_eq_clearLines_place, Board.clearLines_eq_self_of_no_fullRows cfg h]

/-- **Column-disjoint full moves commute when neither completes a row on its
own.** The intermediate clears are no-ops (`applyStep = place`), the two bare
placements commute, and the final `clearLines` — which MAY fire — is applied to
the *same* merged board on both sides. So the adversary's choice of order
between two zone-local responses is irrelevant even when the second move
triggers a clear: order-confluence survives row completion at the joint
board, only *intermediate* completions are excluded. -/
theorem applyStep_comm_of_colsDisjoint (cfg : GameConfig) (b : Board)
    {pl1 pl2 : Placement} (hd : ColsDisjoint pl1 pl2)
    (h1 : Board.fullRows cfg (pl1.place b) = ∅)
    (h2 : Board.fullRows cfg (pl2.place b) = ∅) :
    pl2.applyStep cfg (pl1.applyStep cfg b) = pl1.applyStep cfg (pl2.applyStep cfg b) := by
  rw [applyStep_eq_place_of_no_fullRows cfg b pl1 h1,
      applyStep_eq_place_of_no_fullRows cfg b pl2 h2,
      applyStep_eq_clearLines_place, applyStep_eq_clearLines_place,
      place_comm_of_colsDisjoint b hd]

end Placement
end Tetris
