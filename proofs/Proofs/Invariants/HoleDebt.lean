import Mathlib
import Proofs.Model.Board
import Proofs.Model.Placement
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Wqo
import Proofs.Invariants.HoleyCarrier
import Proofs.Invariants.SurfaceFiber

/-!
# Hole-debt and board energy — wiring clears into the debt counter

`SurfaceFiber.lean` showed the surface evolves as a hole-independent automaton, justifying
a **surface-automaton × hole-debt** decomposition. This file builds the debt side and wires
the line-clear dynamics into it.

## The debt and the energy identity

Define the per-column hole count `colHoles b j := colHeight b j − |colRows b j|` (stack
height minus the number of filled cells in the column — exactly the buried empties), and
the total **debt** `debt cfg b := Σ_{j<cols} colHoles b j`. Define the **board energy**
`surfaceArea cfg b := Σ_{j<cols} colHeight b j`.

The keystone (`debt_add_card_eq_sum_colHeight`, fully proven):

  **`debt cfg b + b.card = surfaceArea cfg b`**     (for well-formed `b`)

i.e. *energy = debt + mass*. The surface area splits cleanly into buried holes and filled
cells. This is the bridge between the surface representation (`Σ colHeight`) and the debt
counter, and it turns the otherwise-combinatorial debt dynamics into monotonicity of a
single sum.

## The wired dynamics (all fully proven)

* `surfaceArea_le_place` — placement *raises* energy (`colHeight` only grows under `∪`).
* `surfaceArea_clearLines_le` — line clears *lower* energy (`clearLines_domLE`).
* `clearLines_card_le` — clears never increase mass.
* `debt_clearLines_add_card_le` — the wired conservation: after a clear,
  `debt' + card' ≤ debt + card`. Energy is a Lyapunov function: it strictly rises by ≥ 4
  per piece (a tetromino) and can only fall on clears, so unbounded play forces a matching
  clear rate. (The piece-rise lower bound and the *pure* debt monotonicity
  `debt(clearLines b) ≤ debt b` are the next targets — see the closing note.)
-/

namespace Tetris.HoleDebt

open Tetris Tetris.WqoCarrier Tetris.HoleyCarrier Tetris.SurfaceFiber

/-! ## Definitions -/

/-- Buried empties in column `j`: stack height minus filled-cell count. -/
def colHoles (b : Board) (j : ℕ) : ℕ := b.colHeight j - (b.colRows j).card

/-- Total hole debt across the configured columns. -/
def debt (cfg : GameConfig) (b : Board) : ℕ :=
  ∑ j ∈ Finset.range cfg.cols, colHoles b j

/-- Board energy: total column height across the configured columns. -/
def surfaceArea (cfg : GameConfig) (b : Board) : ℕ :=
  ∑ j ∈ Finset.range cfg.cols, b.colHeight j

/-! ## The energy identity: energy = debt + mass -/

/-- A column's filled-cell count never exceeds its stack height (filled rows all sit below
the height, so they embed into `range (colHeight)`). -/
theorem card_colRows_le_colHeight (b : Board) (j : ℕ) :
    (b.colRows j).card ≤ b.colHeight j := by
  have hsub : b.colRows j ⊆ Finset.range (b.colHeight j) := by
    intro r hr
    rw [Finset.mem_range]
    have h1 : r + 1 ≤ b.colHeight j := by
      unfold Board.colHeight; exact Finset.le_sup hr
    omega
  calc (b.colRows j).card ≤ (Finset.range (b.colHeight j)).card := Finset.card_le_card hsub
    _ = b.colHeight j := Finset.card_range _

/-- `colRows` (filled rows, via `image (·.2)`) has the same cardinality as the column's
cell slice, since within a fixed column the row coordinate is injective. -/
theorem card_colRows_eq_card_filter (b : Board) (j : ℕ) :
    (b.colRows j).card = (b.filter (·.1 = j)).card := by
  unfold Board.colRows
  apply Finset.card_image_of_injOn
  intro x hx y hy hxy
  simp only [Finset.mem_coe, Finset.mem_filter] at hx hy
  exact Prod.ext (hx.2.trans hy.2.symm) hxy

/-- Mass splits over columns: `|b| = Σ_j |colRows b j|` for a well-formed board. -/
theorem card_eq_sum_colRows {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b) :
    b.card = ∑ j ∈ Finset.range cfg.cols, (b.colRows j).card := by
  rw [Finset.card_eq_sum_card_fiberwise (f := (·.1)) (t := Finset.range cfg.cols)
    (fun x hx => Finset.mem_range.mpr (hwf x hx))]
  exact Finset.sum_congr rfl (fun j _ => (card_colRows_eq_card_filter b j).symm)

/-- **Energy = debt + mass.** `Σ colHeight = debt + |b|`. The total surface area of the
stack decomposes exactly into buried holes (`debt`) and filled cells (`b.card`). -/
theorem debt_add_card_eq_sum_colHeight {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b) :
    debt cfg b + b.card = surfaceArea cfg b := by
  unfold debt surfaceArea colHoles
  rw [card_eq_sum_colRows hwf, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl
    (fun j _ => Nat.sub_add_cancel (card_colRows_le_colHeight b j))

/-! ## Bridge: the columnwise debt equals the geometric hole count `|holes|`

`debt` is defined columnwise (`Σ colHoles`); `HoleyCarrier.holes` is the geometric set of
buried empties. They coincide (for an in-field board), unifying the two debt notions used
across these experiments. -/

/-- The filled rows of column `j`, characterised inside `range (colHeight)`: a row `r` is
filled iff `r < colHeight b j` and `(j, r) ∈ b`. -/
theorem colRows_eq_filter_range (b : Board) (j : ℕ) :
    b.colRows j = (Finset.range (b.colHeight j)).filter (fun r => (j, r) ∈ b) := by
  ext r
  simp only [Board.colRows, Finset.mem_image, Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨x, ⟨hxb, hxj⟩, hxr⟩
    have hxe : x = (j, r) := Prod.ext hxj hxr
    rw [hxe] at hxb
    exact ⟨Board.lt_colHeight hxb, hxb⟩
  · rintro ⟨_, hmem⟩
    exact ⟨(j, r), ⟨hmem, rfl⟩, rfl⟩

/-- Per-column buried empties as a row set: rows below the stack height that are unfilled. -/
def rowHoles (b : Board) (j : ℕ) : Finset ℕ :=
  (Finset.range (b.colHeight j)).filter (fun r => (j, r) ∉ b)

/-- The row-hole set has cardinality `colHoles b j` (`colHeight − filled`). -/
theorem rowHoles_card (b : Board) (j : ℕ) :
    (rowHoles b j).card = colHoles b j := by
  unfold rowHoles colHoles
  rw [colRows_eq_filter_range]
  have h := Finset.card_filter_add_card_filter_not
    (s := Finset.range (b.colHeight j)) (p := fun r => (j, r) ∈ b)
  rw [Finset.card_range] at h
  omega

/-- **Bridge: `|holes cfg b| = debt cfg b`** for an in-field board (every column height at
most `cfg.rows`). Counts `holes` column-by-column: the fiber over column `j` is exactly
`rowHoles b j` carried up by `r ↦ (j, r)`, whose card is `colHoles b j`. -/
theorem holes_card_eq_debt {cfg : GameConfig} {b : Board}
    (hcol : ∀ j, b.colHeight j ≤ cfg.rows) :
    (holes cfg b).card = debt cfg b := by
  have hf : ∀ p ∈ holes cfg b, p.1 ∈ Finset.range cfg.cols := by
    intro p hp
    rw [mem_holes_iff] at hp
    exact (Finset.mem_product.mp hp.1).1
  rw [Finset.card_eq_sum_card_fiberwise hf]
  unfold debt
  apply Finset.sum_congr rfl
  intro j hj
  rw [← rowHoles_card b j]
  have himg : (holes cfg b).filter (fun p => p.1 = j)
      = (rowHoles b j).image (fun r => (j, r)) := by
    ext p
    simp only [Finset.mem_filter, Finset.mem_image, rowHoles, Finset.mem_range,
      mem_holes_iff, Finset.mem_product]
    constructor
    · rintro ⟨⟨⟨_, _⟩, hpnb, hplt⟩, hpj⟩
      have hpe : (j, p.2) = p := Prod.ext hpj.symm rfl
      exact ⟨p.2, ⟨by rw [← hpj]; exact hplt, by rw [hpe]; exact hpnb⟩, hpe⟩
    · rintro ⟨r, ⟨hrlt, hrnb⟩, hrp⟩
      subst hrp
      exact ⟨⟨⟨Finset.mem_range.mp hj, lt_of_lt_of_le hrlt (hcol j)⟩, hrnb, hrlt⟩, rfl⟩
  rw [himg, Finset.card_image_of_injective _ (fun a c h => by simpa using h)]

/-! ## Wiring the dynamics into the energy -/

/-- **Placement raises energy.** `place b = b ∪ dropped`, and `colHeight` of a union is the
max of the parts, so every column's height can only grow. -/
theorem surfaceArea_le_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    surfaceArea cfg b ≤ surfaceArea cfg (pl.place b) := by
  unfold surfaceArea
  apply Finset.sum_le_sum
  intro j _
  rw [Placement.place_eq_union_dropped, colHeight_union]
  exact le_max_left _ _

/-- **Line clears lower energy.** Immediate from `clearLines_domLE` (clears only lower
column heights), summed over columns. -/
theorem surfaceArea_clearLines_le (cfg : GameConfig) (b : Board) :
    surfaceArea cfg (Board.clearLines cfg b) ≤ surfaceArea cfg b := by
  unfold surfaceArea
  exact Finset.sum_le_sum (fun j _ => clearLines_domLE cfg b j)

/-- **Clears never increase mass.** `clearLines` is a filter followed by an image. -/
theorem clearLines_card_le (cfg : GameConfig) (b : Board) :
    (Board.clearLines cfg b).card ≤ b.card := by
  unfold Board.clearLines
  exact le_trans (Finset.card_image_le) (Finset.card_filter_le _ _)

/-- **The wired conservation law for clears:** `debt' + card' ≤ debt + card`. Energy is a
Lyapunov function — it can only fall under a clear — and energy is exactly `debt + mass`,
so the debt-plus-mass total is non-increasing across a line clear. -/
theorem debt_clearLines_add_card_le {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b) :
    debt cfg (Board.clearLines cfg b) + (Board.clearLines cfg b).card
      ≤ debt cfg b + b.card := by
  rw [debt_add_card_eq_sum_colHeight (Board.clearLines_wf hwf),
    debt_add_card_eq_sum_colHeight hwf]
  exact surfaceArea_clearLines_le cfg b

/-! ## The placement side of the debt counter: holes only grow

The clear side lowers `debt + mass`; the placement side handles `debt` directly. A hard
drop adds cells *on top of* the stack (`dropped` cells sit at or above their column's
height), so it can never fill an existing buried empty — every hole of `b` survives into
`place b`. Hence debt (as the geometric hole count `|holes|`) is non-decreasing under
placement. -/

/-- A dropped cell sits at or above its column's stack height: if `p ∈ dropped b` then
`colHeight b p.1 ≤ p.2`. (The drop offset is the `sup` of the per-cell clearances, so each
landed cell clears the stack.) -/
theorem le_row_of_mem_dropped {b : Board} {pl : Placement} {p : Coord}
    (hp : p ∈ pl.dropped b) : b.colHeight p.1 ≤ p.2 := by
  rw [Placement.dropped_eq_image, Finset.mem_image] at hp
  obtain ⟨cell, hcell, rfl⟩ := hp
  have hle := Placement.le_sup_sub pl.shapeUp b.colHeight pl.col hcell
  rw [← Placement.dropOffset_eq_sup] at hle
  change b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + cell.2
  omega

/-- **No-clear placement never removes a hole.** Every buried empty of `b` is still a
buried empty of `place b`: it is not among the dropped cells (those land at/above the
surface, while a hole sits strictly below it), and the column height only grows. -/
theorem place_holes_subset (cfg : GameConfig) (b : Board) (pl : Placement) :
    holes cfg b ⊆ holes cfg (pl.place b) := by
  intro p hp
  rw [mem_holes_iff] at hp ⊢
  obtain ⟨hgrid, hnotin, hlt⟩ := hp
  refine ⟨hgrid, ?_, ?_⟩
  · rw [Placement.place_eq_union_dropped, Finset.mem_union]
    rintro (hb | hd)
    · exact hnotin hb
    · have := le_row_of_mem_dropped hd
      omega
  · have hmono : b.colHeight p.1 ≤ (pl.place b).colHeight p.1 := by
      rw [Placement.place_eq_union_dropped, colHeight_union]; exact le_max_left _ _
    omega

/-- **Debt is non-decreasing under placement** (debt as the geometric hole count). Direct
from `place_holes_subset`. This is the placement-side counterpart of
`debt_clearLines_add_card_le`: holes accumulate as you place and are only discharged by
clears — exactly the hole-debt counter dynamics. -/
theorem holes_card_le_place (cfg : GameConfig) (b : Board) (pl : Placement) :
    (holes cfg b).card ≤ (holes cfg (pl.place b)).card :=
  Finset.card_le_card (place_holes_subset cfg b pl)


/-! ## Pure debt monotonicity under clears

The deep target: `debt(clearLines b) ≤ debt b`. The codebase already provides the hard
injectivity half — `Board.colCount_clearLines_add`: each clear removes exactly one cell per
column (`colCount b j = colCount(clearLines b) j + |fullRows|`). What remains is the matching
*height* drop: each column loses at least `|fullRows|` from its height. -/

/-- Every full row is a filled row of every valid column (it is filled in all columns). -/
theorem fullRows_subset_colRows {cfg : GameConfig} {b : Board} {j : ℕ} (hj : j < cfg.cols) :
    Board.fullRows cfg b ⊆ b.colRows j := by
  intro r hr
  simp only [Board.fullRows, Finset.mem_filter] at hr
  have hmem : (j, r) ∈ b := hr.2 j (Finset.mem_range.mpr hj)
  simp only [Board.colRows, Finset.mem_image, Finset.mem_filter]
  exact ⟨(j, r), ⟨hmem, rfl⟩, rfl⟩

/-- **The height-drop bound (P3):** a clear lowers every valid column's height by at least
the number of cleared rows. A surviving row `s = r − clearedBelow r` (with `r` a non-full
filled row `< colHeight`) satisfies `s + |fullRows| < colHeight`: the full rows at or above
`r` are distinct integers in `(r, colHeight)`, so they leave exactly that much headroom. -/
theorem clearLines_colHeight_add_le {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) :
    (Board.clearLines cfg b).colHeight j + (Board.fullRows cfg b).card ≤ b.colHeight j := by
  have hk : (Board.fullRows cfg b).card ≤ b.colHeight j :=
    le_trans (Finset.card_le_card (fullRows_subset_colRows hj)) (card_colRows_le_colHeight b j)
  have key : ∀ s ∈ (Board.clearLines cfg b).colRows j,
      s + (Board.fullRows cfg b).card < b.colHeight j := by
    intro s hs
    simp only [Board.colRows, Finset.mem_image, Finset.mem_filter] at hs
    obtain ⟨cell, ⟨hcl, hcj⟩, hcs⟩ := hs
    rw [Board.mem_clearLines_iff] at hcl
    obtain ⟨p, hpb, hpnf, hpq⟩ := hcl
    have hcell1 : cell.1 = p.1 := by rw [← hpq]
    have hcell2 : cell.2 = p.2 - Board.clearedBelow cfg b p.2 := by rw [← hpq]
    have hp1 : p.1 = j := by rw [← hcell1, hcj]
    have hpe : p = (j, p.2) := Prod.ext hp1 rfl
    have hpjb : (j, p.2) ∈ b := hpe ▸ hpb
    have hpd : p.2 < b.colHeight j := Board.lt_colHeight hpjb
    have hseq : s = p.2 - Board.clearedBelow cfg b p.2 := by rw [← hcs, hcell2]
    have ha := Board.clearedBelow_le_row cfg b p.2
    have hb := Finset.card_filter_add_card_filter_not
      (s := Board.fullRows cfg b) (p := fun f => f < p.2)
    have hcbeq : Board.clearedBelow cfg b p.2
        = ((Board.fullRows cfg b).filter (fun f => f < p.2)).card := rfl
    have hsub : (Board.fullRows cfg b).filter (fun f => ¬ f < p.2)
        ⊆ Finset.Ioo p.2 (b.colHeight j) := by
      intro f hf
      rw [Finset.mem_filter] at hf
      rw [Finset.mem_Ioo]
      have hffull : Board.isFull cfg b f := by
        have hf1 := hf.1
        simp only [Board.fullRows, Finset.mem_filter] at hf1
        exact hf1.2
      have hflt : f < b.colHeight j := by
        have hfc : f ∈ b.colRows j := fullRows_subset_colRows hj hf.1
        have hf1 : f + 1 ≤ (b.colRows j).sup (· + 1) := Finset.le_sup hfc
        unfold Board.colHeight
        omega
      have hfne : f ≠ p.2 := fun h => hpnf (h ▸ hffull)
      exact ⟨by omega, hflt⟩
    have hc : ((Board.fullRows cfg b).filter (fun f => ¬ f < p.2)).card
        ≤ b.colHeight j - p.2 - 1 := by
      calc ((Board.fullRows cfg b).filter (fun f => ¬ f < p.2)).card
          ≤ (Finset.Ioo p.2 (b.colHeight j)).card := Finset.card_le_card hsub
        _ = b.colHeight j - p.2 - 1 := Nat.card_Ioo _ _
    omega
  have hsup : (Board.clearLines cfg b).colHeight j
      ≤ b.colHeight j - (Board.fullRows cfg b).card := by
    have hcl : (Board.clearLines cfg b).colHeight j
        = ((Board.clearLines cfg b).colRows j).sup (· + 1) := rfl
    rw [hcl]
    apply Finset.sup_le
    intro s hs
    have := key s hs
    omega
  omega

/-- **Pure debt monotonicity under clears:** `debt(clearLines b) ≤ debt b`. Per column,
`colCount` drops by exactly `|fullRows|` (`colCount_clearLines_add`) while `colHeight` drops
by at least `|fullRows|` (`clearLines_colHeight_add_le`); their difference `colHoles` can
only fall. Summed over columns, the debt counter is non-increasing under a line clear — the
Lyapunov fact the WSTS survival argument needs. -/
theorem clearLines_debt_le {cfg : GameConfig} {b : Board} (_hwf : Board.WF cfg b) :
    debt cfg (Board.clearLines cfg b) ≤ debt cfg b := by
  unfold debt
  apply Finset.sum_le_sum
  intro j hj
  rw [Finset.mem_range] at hj
  have hcount := Board.colCount_clearLines_add cfg b hj
  have hcb : (b.colRows j).card
      = ((Board.clearLines cfg b).colRows j).card + (Board.fullRows cfg b).card := by
    rw [card_colRows_eq_card_filter b j,
      card_colRows_eq_card_filter (Board.clearLines cfg b) j]
    simpa [Board.colCount] using hcount
  have hheight := @clearLines_colHeight_add_le cfg b j hj
  unfold colHoles
  omega


/-! ## Status and next targets

The debt counter is now fully characterised under both transitions:
* **placement** — `debt` is non-decreasing (`holes_card_le_place`, via `place_holes_subset`);
* **line clear** — `debt` is non-increasing (`clearLines_debt_le`), proved per-column from
  `colCount_clearLines_add` (each clear removes one cell per column) and
  `clearLines_colHeight_add_le` (each column drops by ≥ `|fullRows|`).

So `debt` rises only on placement and falls only on clears — a genuine Lyapunov counter.

Remaining for the WSTS survival argument:
* a per-piece *upper* bound `debt (place b pl) ≤ debt b + 3` (a tetromino buries ≤ 3 cells),
  bounding debt growth per move; combined with `clearLines_debt_le` this makes `debt` a
  bounded counter whenever the clear rate keeps pace;
* lifting the bounded-debt + bounded-height invariant into a `Carrier`/`TetrisSolvableValid`
  reduction (the explicit, atlas-style closed set the non-congruence result demands). -/

end Tetris.HoleDebt
