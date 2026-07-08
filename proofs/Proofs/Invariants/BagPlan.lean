import Mathlib
import Proofs.Model.Placement
import Proofs.Invariants.Confluence
import Proofs.Invariants.SurfaceCalculus

/-!
# Bag plans — order-freeness is exactly column-disjointness

Reduction 1 of the seam program asked: can the adversary's within-bag order
quantifier be killed by algebra — a **bag plan** (designated placements for
the seven pieces) whose execution is order-free? This file settles the
question, negatively and exactly.

**The degeneracy theorem** (`mutualDropInvariant_iff_colsDisjoint`): two
placements leave each other's landing untouched — in either order — *iff*
they share no column. Gravity admits no cleverer commutation: in a shared
column, mutual non-interference forces each piece's cells strictly below
the other's, a cycle. So the order-free fragment of Tetris is precisely the
column-disjoint fragment already characterized in `Confluence`, and the
width budget (`no_pairwise_colsDisjoint_bag`: the seven pieces' minimal
widths sum to 13 > 10) yields the corollary
(`no_orderFree_bag_plan`): **no full-bag plan is order-free — within-bag
adaptivity is forced by gravity**, upgrading the measured "adaptivity
replaces hold" from observation to theorem.

**The logical conclusion of reduction 1.** The order quantifier cannot be
eliminated at the cell level beyond Cartier–Foata layers of disjoint
placements; cross-layer interleaving is genuine game content. Two residues
remain, now provably necessary rather than suspected: (i) adaptive
responses whose *cell sets* depend on arrival order, with reconvergence
sought at the skyline-plus-debt level (`SurfaceFiber`: the future reads
only the skyline), or (ii) certificates that never mention orders at all —
potential/sublevel-set invariants with one-bag decrease obligations. A
sharper impossibility is conjectured but not proven here: even
*adaptive* plans reaching an order-independent final board should fail
(the canonical role-swap mechanism — L and J jointly filling a 2×4 box —
is already gravity-obstructed), which would make skyline-level
reconvergence not just sufficient but the only possible target.

The one-sided invariance lemma (`dropped_place_of_dropInvariant`) survives
as the positive tool: it is not symmetric, and layered designs use exactly
the one-way direction.
-/

namespace Tetris
namespace BagPlan

open Tetris Tetris.Placement Tetris.SurfaceCalculus

/-- `pl`'s placement does not move `q`'s landing: `q` hard-drops to the
same offset whether or not `pl` has been placed. -/
def DropInvariant (b : Board) (pl q : Placement) : Prop :=
  q.dropOffset (pl.place b) = q.dropOffset b

/-- Neither placement moves the other's landing — the order-freeness of
the pair on base `b`. -/
def MutualDropInvariant (b : Board) (pl q : Placement) : Prop :=
  DropInvariant b pl q ∧ DropInvariant b q pl

/-- One-sided invariance transfers to the dropped cells. -/
theorem dropped_place_of_dropInvariant {b : Board} {pl q : Placement}
    (h : DropInvariant b pl q) :
    q.dropped (pl.place b) = q.dropped b := by
  rw [Placement.dropped_eq_cellsAt, Placement.dropped_eq_cellsAt, h]

/-- Every cell of a hard-dropped piece rests at or above its column's
pre-placement height (the per-cell content of the drop supremum). -/
theorem colHeight_le_drop_cell (b : Board) (q : Placement)
    {cell : PieceCell} (hcell : cell ∈ q.shapeUp) :
    b.colHeight (q.col + cell.1) ≤ q.dropOffset b + cell.2 := by
  have h := Placement.le_sup_sub q.shapeUp b.colHeight q.col hcell
  rw [← Placement.dropOffset_eq_sup] at h
  omega

/-- **In a shared column, invariance forces strict verticality**: if `q`'s
landing is untouched by `pl`, then in any column both occupy, `pl`'s cell
tops out strictly below `q`'s cell. -/
theorem cell_below_of_dropInvariant {b : Board} {pl q : Placement}
    (h : DropInvariant b pl q)
    {c1 c2 : PieceCell} (h1 : c1 ∈ pl.shapeUp) (h2 : c2 ∈ q.shapeUp)
    (hcol : pl.col + c1.1 = q.col + c2.1) :
    pl.dropOffset b + c1.2 + 1 ≤ q.dropOffset b + c2.2 := by
  -- The placed `pl` raises the shared column to at least its cell's top.
  have hraise : pl.dropOffset b + c1.2 + 1
      ≤ (pl.place b).colHeight (pl.col + c1.1) := by
    rw [colHeight_place_eq]
    refine le_trans ?_ (le_max_right _ _)
    have hmem : c1 ∈ cellsInCol pl (pl.col + c1.1) := by
      unfold cellsInCol
      exact Finset.mem_filter.mpr ⟨h1, rfl⟩
    exact Finset.le_sup (f := fun cell => pl.dropOffset b + cell.2 + 1) hmem
  -- `q`'s unchanged supremum bounds the raised height through `q`'s cell.
  have hbound : (pl.place b).colHeight (q.col + c2.1) - c2.2
      ≤ q.dropOffset b := by
    have hle := Placement.le_sup_sub q.shapeUp (pl.place b).colHeight q.col h2
    rw [← Placement.dropOffset_eq_sup] at hle
    rw [← h]
    exact hle
  rw [hcol] at hraise
  omega

/-- **The degeneracy theorem: order-freeness is exactly
column-disjointness.** Two placements are mutually drop-invariant on a base
iff they share no column. The forward direction is the impossibility of
mutual support: in a shared column each piece would have to sit strictly
below the other. Gravity admits no commutation beyond disjointness. -/
theorem mutualDropInvariant_iff_colsDisjoint (b : Board) (pl q : Placement) :
    MutualDropInvariant b pl q ↔ ColsDisjoint pl q := by
  constructor
  · rintro ⟨hpq, hqp⟩
    intro c1 h1 c2 h2 hcol
    have hbelow1 := cell_below_of_dropInvariant hpq h1 h2 hcol
    have hbelow2 := cell_below_of_dropInvariant hqp h2 h1 hcol.symm
    omega
  · intro hd
    exact ⟨dropOffset_place_of_colsDisjoint b hd,
      dropOffset_place_of_colsDisjoint b hd.symm⟩

/-- Order-free pairs commute (restatement through the degeneracy — the
commutation content lives in `Confluence.place_comm_of_colsDisjoint`). -/
theorem place_comm_of_mutualDropInvariant {b : Board} {pl q : Placement}
    (h : MutualDropInvariant b pl q) :
    q.place (pl.place b) = pl.place (q.place b) :=
  place_comm_of_colsDisjoint b ((mutualDropInvariant_iff_colsDisjoint b pl q).mp h)

/-- **No full-bag plan is order-free.** For any assignment of a valid
placement to each of the seven pieces (any board, any width up to 12), some
pair interferes: within-bag adaptivity is forced by gravity. The measured
fact behind the whole adaptive design space, now a theorem. -/
theorem no_orderFree_bag_plan (cfg : GameConfig) (hcols : cfg.cols ≤ 12)
    (b : Board) (f : Piece → Placement)
    (hpiece : ∀ p, (f p).piece = p)
    (hvalid : ∀ p, (f p).Valid cfg) :
    ¬ (∀ p q, p ≠ q → MutualDropInvariant b (f p) (f q)) := by
  intro h
  exact no_pairwise_colsDisjoint_bag cfg hcols f hpiece hvalid
    (fun p q hpq =>
      (mutualDropInvariant_iff_colsDisjoint b (f p) (f q)).mp (h p q hpq))

/-- Standard-config headline form. -/
theorem no_orderFree_bag_plan_standard (b : Board) (f : Piece → Placement)
    (hpiece : ∀ p, (f p).piece = p)
    (hvalid : ∀ p, (f p).Valid GameConfig.standard) :
    ¬ (∀ p q, p ≠ q → MutualDropInvariant b (f p) (f q)) :=
  no_orderFree_bag_plan GameConfig.standard (by decide) b f hpiece hvalid

end BagPlan
end Tetris
