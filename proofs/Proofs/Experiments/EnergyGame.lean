import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.SafeSet
import Proofs.Experiments.WqoCarrier
import Proofs.Experiments.HoleyCarrier
import Proofs.Experiments.HoleDebt

/-!
# The energy-game framing — headroom, capacity conservation, and its monotonicity

Reframes controlled Tetris as a **lower-bounded energy game**. Define **headroom**
`headroom = cols·rows − surfaceArea` (the unused capacity above the stack). Survival ⟺ the
player can keep `headroom ≥ 0` forever against the adversary's pieces; energy games are
determined with *memoryless* strategies, so a winning strategy is a lookup table on
`(surface, bag)` — i.e. the Atlas is, without loss of generality, history-free.

This file lays the energy foundation on top of `HoleDebt`:

* **Capacity conservation** (`capacity_conservation`): for an in-field board,
  `headroom + debt + mass = cols·rows`. The board's total capacity splits into three
  ledgers — empty space above the stack (`headroom`), buried empties (`debt`), and filled
  cells (`mass = card`). This is the energy-game's master identity.
* **headroom is the energy**: placement *spends* it (`headroom_place_le`) and line clears
  *refund* it (`headroom_le_clearLines`) — exactly the energy-game weight structure
  (placement = cost edges, clears = reward edges, topping out = energy `< 0`).

Backlog note (recorded): the per-piece bound `debt(place) ≤ debt + 3` is FALSE — a single
piece on a jagged surface buries many cells (e.g. a horizontal I on heights `[10,0,0,0]`
leaves a deep overhang). Debt growth per move is roughness-dependent, which is precisely why
the bounded controller must control roughness. The clean invariant is the conservation law
below, not a constant per-piece bound.
-/

namespace Tetris.EnergyGame

open Tetris Tetris.WqoCarrier Tetris.HoleyCarrier Tetris.HoleDebt

/-- **Headroom**: unused capacity above the stack, `cols·rows − surfaceArea`. The energy of
the energy game — the player keeps it non-negative (never tops out). -/
def headroom (cfg : GameConfig) (b : Board) : ℕ :=
  cfg.cols * cfg.rows - surfaceArea cfg b

/-- Board energy is at most total capacity for an in-field board: every column height is at
most `rows`, so `surfaceArea = Σ colHeight ≤ cols·rows`. -/
theorem surfaceArea_le_capacity {cfg : GameConfig} {b : Board}
    (h : ∀ j, b.colHeight j ≤ cfg.rows) :
    surfaceArea cfg b ≤ cfg.cols * cfg.rows := by
  unfold surfaceArea
  calc ∑ j ∈ Finset.range cfg.cols, b.colHeight j
      ≤ ∑ _j ∈ Finset.range cfg.cols, cfg.rows := Finset.sum_le_sum (fun j _ => h j)
    _ = cfg.cols * cfg.rows := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

/-- **Capacity conservation (the master energy identity):** for an in-field, well-formed
board, `headroom + debt + mass = cols·rows`. Total capacity is partitioned into empty space
above the stack, buried holes, and filled cells. -/
theorem capacity_conservation {cfg : GameConfig} {b : Board} (hwf : Board.WF cfg b)
    (h : ∀ j, b.colHeight j ≤ cfg.rows) :
    headroom cfg b + debt cfg b + b.card = cfg.cols * cfg.rows := by
  unfold headroom
  have hsa := debt_add_card_eq_sum_colHeight hwf
  have hle := surfaceArea_le_capacity h
  omega

/-- **Placement spends headroom.** A hard drop only raises column heights, so `surfaceArea`
grows and headroom shrinks — a cost edge in the energy game. -/
theorem headroom_place_le (cfg : GameConfig) (b : Board) (pl : Placement) :
    headroom cfg (pl.place b) ≤ headroom cfg b := by
  unfold headroom
  have := surfaceArea_le_place cfg b pl
  omega

/-- **Line clears refund headroom.** Clears only lower column heights, so `surfaceArea`
falls and headroom grows — a reward edge in the energy game. -/
theorem headroom_le_clearLines (cfg : GameConfig) (b : Board) :
    headroom cfg b ≤ headroom cfg (Board.clearLines cfg b) := by
  unfold headroom
  have := surfaceArea_clearLines_le cfg b
  omega

#print axioms capacity_conservation
#print axioms headroom_place_le
#print axioms headroom_le_clearLines

/-! ## Next (energy-game backlog)

With headroom established as the energy, the next iterations build toward "survival ⟺ player
wins the lower-bounded energy game on the `(surface, bag)` arena":

1. Hole-free fiber: the full step `applyStep` is surface-determined when `debt = 0`
   (clears become surface-determined), giving a finite congruent sub-game.
2. The I-drain lemma: from every surface in a candidate `Σ`, some I-placement nets
   headroom change `≥ 0` (refills via a clear) — the localized form of crux #66/#72.
3. Exhibit-and-verify a concrete controller on a small `Σ` (`native_decide`), lifting
   soundness through `SurfaceFiber`/`HoleDebt`.
4. Compose into `tetrisSolvableValid_of_height_bounded_invariant`. -/

end Tetris.EnergyGame
