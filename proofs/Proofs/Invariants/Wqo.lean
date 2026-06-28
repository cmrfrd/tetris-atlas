import Mathlib
import Proofs.Board
import Proofs.Placement

/-!
# WQO / height-domination monotonicity primitives

The structural foundation of the finite-basis route: the no-clear hard drop is monotone
under column-height domination, and line clears only lower heights. These are pure facts
about `Board`/`Placement` (no safe-set machinery), so they live in the `Invariants` layer
and are reused by `SurfaceFiber`, `HoleDebt`, `HoleyCarrier`, and `EnergyGame`.

The empirical `monotone` probe established (32.4M checks, 0 non-clearing violations) that
`applyStep` preserves column-height domination *except across line clears*. The keystone
`place_domLE_mono` proves the no-clear half; `clearLines_domLE` tames the clear case.

The `TetrisSolvableValid` reduction built on these (`tetrisSolvableValid_of_wqo`) lives in
the research target (`Proofs.Experiments.WqoCarrier`), which imports this module.
-/

namespace Tetris.WqoCarrier

open Tetris

/-- **Height-domination** — Dickson's well-quasi-order on column-height vectors. -/
def domLE (b β : Board) : Prop := ∀ j, b.colHeight j ≤ β.colHeight j

/-- **The drop offset is monotone under domination**: on the lower board the piece falls
at least as far. Each `dropOffset` summand `colHeight (col+c₁) − c₂` is monotone in the
column height, so the `Finset.sup` is too. -/
theorem dropOffset_mono {b β : Board} (pl : Placement) (h : domLE b β) :
    pl.dropOffset b ≤ pl.dropOffset β := by
  unfold Placement.dropOffset
  apply Finset.sup_mono_fun
  intro cell _
  exact Nat.sub_le_sub_right (h (pl.col + cell.1)) cell.2

/-- **`colHeight` distributes over `∪` as a max.** `colRows` is `filter`-then-`image`,
both of which distribute over union, and `Finset.sup` of a union is the join. -/
theorem colHeight_union (a b : Board) (j : ℕ) :
    (a ∪ b).colHeight j = max (a.colHeight j) (b.colHeight j) := by
  unfold Board.colHeight Board.colRows
  rw [Finset.filter_union, Finset.image_union, Finset.sup_union]

/-- **`colHeight` of the dropped piece is monotone in the drop offset.** The dropped cells
are `cellsAt d`, whose column heights shift up exactly with `d`; a larger offset gives
larger (or equal) per-column height. Combined with `dropOffset_mono` this controls the
piece's contribution on the two boards. -/
theorem colHeight_dropped_mono {b β : Board} (pl : Placement) (j : ℕ)
    (hd : pl.dropOffset b ≤ pl.dropOffset β) :
    (pl.dropped b).colHeight j ≤ (pl.dropped β).colHeight j := by
  unfold Placement.dropped
  set d1 := pl.dropOffset b
  set d2 := pl.dropOffset β
  unfold Board.colHeight
  apply Finset.sup_le
  intro r hr
  rw [Board.colRows, Finset.mem_image] at hr
  obtain ⟨x, hxf, hxr⟩ := hr
  rw [Finset.mem_filter] at hxf
  obtain ⟨hxmem, hxj⟩ := hxf
  rw [Placement.cellsAt, Finset.mem_image] at hxmem
  obtain ⟨c, hc, hcx⟩ := hxmem
  -- hcx : (pl.col + c.1, d1 + c.2) = x ; hxj : x.1 = j ; hxr : x.2 = r
  have hx1 : x.1 = pl.col + c.1 := by rw [← hcx]
  have hx2 : x.2 = d1 + c.2 := by rw [← hcx]
  have hmem : d2 + c.2 ∈ Board.colRows (pl.cellsAt d2) j := by
    rw [Board.colRows, Finset.mem_image]
    refine ⟨(pl.col + c.1, d2 + c.2), ?_, rfl⟩
    rw [Finset.mem_filter]
    refine ⟨?_, by rw [← hxj, hx1]⟩
    rw [Placement.cellsAt, Finset.mem_image]
    exact ⟨c, hc, rfl⟩
  have hle : d2 + c.2 + 1 ≤ ((pl.cellsAt d2).colRows j).sup (· + 1) :=
    Finset.le_sup hmem
  have hr_eq : r = d1 + c.2 := by rw [← hxr, hx2]
  show r + 1 ≤ ((pl.cellsAt d2).colRows j).sup (· + 1)
  omega

/-- **THE no-clear monotonicity keystone.** Domination is preserved by the hard-drop
`place` (pre-clear): `place b pl = b ∪ dropped b`, `colHeight` of the union is the max of
the two parts' heights, and each part is monotone (`h` for `b`; `colHeight_dropped_mono`
+ `dropOffset_mono` for the piece). The max of monotone maps is monotone. -/
theorem place_domLE_mono {b β : Board} (pl : Placement) (h : domLE b β) :
    domLE (pl.place b) (pl.place β) := by
  intro j
  rw [Placement.place_eq_union_dropped, Placement.place_eq_union_dropped,
    colHeight_union, colHeight_union]
  exact max_le_max (h j) (colHeight_dropped_mono pl j (dropOffset_mono pl h))

/-- **Line clears only lower column heights**: `clearLines b ≼ b`. Clearing filters out
full-row cells and shifts every survivor *down* by the number of cleared rows below it,
so each surviving cell ends at a row ≤ its original — hence no column gets taller. This is
the lemma that tames the clear case (the one phenomenon that breaks the no-clear keystone):
`applyStep b pl = clearLines (place b pl) ≼ place b pl`. -/
theorem clearLines_domLE (cfg : GameConfig) (b : Board) :
    domLE (Board.clearLines cfg b) b := by
  intro j
  unfold Board.colHeight
  apply Finset.sup_le
  intro r' hr'
  rw [Board.colRows, Finset.mem_image] at hr'
  obtain ⟨x, hxf, hxr⟩ := hr'
  rw [Finset.mem_filter] at hxf
  obtain ⟨hxmem, hxj⟩ := hxf
  unfold Board.clearLines at hxmem
  rw [Finset.mem_image] at hxmem
  obtain ⟨q, hqf, hqx⟩ := hxmem
  rw [Finset.mem_filter] at hqf
  obtain ⟨hqb, _⟩ := hqf
  -- hqx : (q.1, q.2 - clearedBelow cfg b q.2) = x ; hxj : x.1 = j ; hxr : x.2 = r'
  have hq1 : q.1 = j := by rw [← hxj, ← hqx]
  have hr'le : r' ≤ q.2 := by rw [← hxr, ← hqx]; exact Nat.sub_le _ _
  have hqjb : (j, q.2) ∈ b := by
    have : (q.1, q.2) ∈ b := by simpa using hqb
    rwa [hq1] at this
  have hlt : q.2 < b.colHeight j := Board.lt_colHeight hqjb
  show r' + 1 ≤ b.colHeight j
  omega

/-- Domination is transitive. -/
theorem domLE_trans {a b c : Board} (h1 : domLE a b) (h2 : domLE b c) : domLE a c :=
  fun j => le_trans (h1 j) (h2 j)

end Tetris.WqoCarrier
