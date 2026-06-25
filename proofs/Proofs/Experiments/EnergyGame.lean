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

/-! ## The hole-free fiber is the surface arena

The energy game lives on surfaces. The bridge: when `debt = 0` every column is a *gapless*
stack, so a hole-free board is completely determined by its column-height vector. Hence the
hole-free fiber is in bijection with the surface space — the finite arena on which the
surface automaton (`SurfaceFiber`) and the energy (`headroom`) operate. -/

/-- **Hole-free columns are gapless:** for a board with `debt = 0`, a valid column is filled
in exactly the rows below its height. (`debt = 0` ⇒ each `colHoles = 0` ⇒ filled-count =
height ⇒ the filled rows are all of `[0, colHeight)`.) -/
theorem holeFree_filled_iff {cfg : GameConfig} {b : Board} (hd : debt cfg b = 0)
    {j : ℕ} (hj : j < cfg.cols) (r : ℕ) :
    (j, r) ∈ b ↔ r < b.colHeight j := by
  have hcolHoles : colHoles b j = 0 := by
    unfold debt at hd
    exact (Finset.sum_eq_zero_iff.mp hd) j (Finset.mem_range.mpr hj)
  have hcard : (b.colRows j).card = b.colHeight j := by
    unfold colHoles at hcolHoles
    have hle := card_colRows_le_colHeight b j
    omega
  have hsub : b.colRows j ⊆ Finset.range (b.colHeight j) := by
    intro x hx
    rw [Finset.mem_range]
    have hx1 : x + 1 ≤ b.colHeight j := by unfold Board.colHeight; exact Finset.le_sup hx
    omega
  have heq : b.colRows j = Finset.range (b.colHeight j) :=
    Finset.eq_of_subset_of_card_le hsub (by rw [Finset.card_range]; omega)
  have hmem : (j, r) ∈ b ↔ r ∈ b.colRows j := by
    simp only [Board.colRows, Finset.mem_image, Finset.mem_filter]
    constructor
    · intro h; exact ⟨(j, r), ⟨h, rfl⟩, rfl⟩
    · rintro ⟨x, ⟨hxb, hxj⟩, hxr⟩
      have hxe : x = (j, r) := Prod.ext hxj hxr
      exact hxe ▸ hxb
  rw [hmem, heq, Finset.mem_range]

/-- **Hole-free boards are determined by their surface:** two hole-free, well-formed boards
with equal column heights are equal. So the hole-free fiber injects into the surface space
(the energy-game arena), one board per height vector. -/
theorem holeFree_ext {cfg : GameConfig} {b β : Board}
    (hb : debt cfg b = 0) (hβ : debt cfg β = 0)
    (hwfb : Board.WF cfg b) (hwfβ : Board.WF cfg β)
    (h : ∀ j, b.colHeight j = β.colHeight j) : b = β := by
  ext ⟨j, r⟩
  by_cases hj : j < cfg.cols
  · rw [holeFree_filled_iff hb hj, holeFree_filled_iff hβ hj, h]
  · constructor
    · intro hmem; exact absurd (hwfb (j, r) hmem) hj
    · intro hmem; exact absurd (hwfβ (j, r) hmem) hj

#print axioms capacity_conservation
#print axioms headroom_place_le
#print axioms headroom_le_clearLines
#print axioms holeFree_filled_iff
#print axioms holeFree_ext

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
