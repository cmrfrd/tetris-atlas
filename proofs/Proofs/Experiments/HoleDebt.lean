import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.SafeSet
import Proofs.Experiments.WqoCarrier
import Proofs.Experiments.HoleyCarrier

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

The keystone (`debt_add_card_eq_sum_colHeight`, sorry-free):

  **`debt cfg b + b.card = surfaceArea cfg b`**     (for well-formed `b`)

i.e. *energy = debt + mass*. The surface area splits cleanly into buried holes and filled
cells. This is the bridge between the surface representation (`Σ colHeight`) and the debt
counter, and it turns the otherwise-combinatorial debt dynamics into monotonicity of a
single sum.

## The wired dynamics (all sorry-free)

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

open Tetris Tetris.WqoCarrier Tetris.HoleyCarrier

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

#print axioms debt_add_card_eq_sum_colHeight
#print axioms debt_clearLines_add_card_le
#print axioms surfaceArea_le_place

/-! ## Next targets (analysis)

The energy law gives `debt' + card' ≤ debt + card` but not yet the *pure* debt
monotonicity `debt(clearLines b) ≤ debt b`. By the energy identity that is equivalent to

  `(surfaceArea b − surfaceArea (clearLines b)) ≥ (b.card − (clearLines b).card)`,

i.e. **the total height dropped by a clear is at least the number of cells removed.** Both
sides equal `cfg.cols * (fullRows cfg b).card`:

* `b.card − (clearLines b).card = cfg.cols * |fullRows|` — each cleared full row removes
  exactly `cols` cells (needs: the shift map is injective on survivors, and each full row
  is `cols` cells for an in-field board).
* `surfaceArea b − surfaceArea (clearLines b) ≥ cfg.cols * |fullRows|` — every column loses
  at least `|fullRows|` from its height (all full rows lie below every column's top).

Proving those two yields `clearLines_debt_le : debt (clearLines b) ≤ debt b` — the debt
counter is non-increasing under clears, the Lyapunov fact the WSTS survival argument needs.
The companion `place` bound `debt b ≤ debt (place b pl) ≤ debt b + 3` (a tetromino buries
at most 3 cells) bounds debt growth per piece; together they make `debt` a bounded counter
whenever the clear rate matches the ≥4-energy-per-piece placement rate. -/

end Tetris.HoleDebt
