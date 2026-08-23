import Mathlib
import Proofs.Safety.AdversarialRate

/-!
# Proof by contradiction: the available route, and the counting barrier

Can we assume *no* solver exists and derive a contradiction from the ledger
corpus? This file answers both halves.

## The contradiction route exists, and the library already has its skeleton

Solvability is equivalent to a *negative finitary* statement, so "a solver must
exist" is exactly "no finite refutation exists":

* `TetrisSolvableValid ↔ GameState.init ∈ safe` — solvability is membership in
  the greatest fixed point (`tetrisSolvableValidFor_iff_init_safe`).
* `init ∈ safe ↔ init ∈ safeIterFinite …` — the GFP is reached by finitely many
  iterations of a computable operator on the finite state space
  (`init_safe_iff_init_mem_safeIterFinite`,
  `tetrisSolvableValidFor_iff_init_mem_safeIterFinite`).
* Therefore **¬solvable is equivalent to a finite object**: a stage `N` at
  which `init` falls out of the iterate — concretely, an adversarial kill-tree
  of depth `N` that defeats *every* solver. Proof by contradiction is thus
  available in principle: *assume the kill-tree, refute it.* It is even a
  decidable proposition; the obstruction is only that the decision procedure is
  astronomically large.

What the counting corpus contributes to that refutation is real but partial: by
`adversary_cannot_force_gt` / `clearedAdv_ledger`, no kill-tree can operate by
starving the clearing budget — the arithmetic balances identically for every
piece order. Any kill certificate must therefore kill through **geometry**
(holes, unreachable surfaces, forced roughness), never through the rate.

## The barrier: counting alone can never finish the job

Could some cleverer combination of ledger facts close the contradiction
outright? **Provably not**, and this file supplies the witness:

* `GameConfig.flat` — the 10 × **1** board. Every counting theorem in this
  corpus is config-generic and holds there verbatim: mass conservation, the
  clearing rate `4n/cols`, the capacity floor, the bag clocks, the window
  bounds. Nothing in the ledger distinguishes `⟨10, 1⟩` from `⟨10, 20⟩`.
* `not_tetrisSolvableValidFor_flat` — yet `⟨10, 1⟩` is **unsolvable**: the
  adversary opens with O, every placement of O occupies two rows, and a
  one-row field cannot hold it (`flat_O_step_lost`, `init_not_safe_flat`).

So any proof scheme that would derive `TetrisSolvableValid` from
config-generic counting facts alone would prove `⟨10, 1⟩` solvable too —
contradiction. The missing ingredient is *irreducibly geometric*: it must use
the specific fit between the seven tetromino shapes and twenty rows of
headroom, which is precisely the content the safe-set / carrier program is
built to capture. This is the first machine-checked unsolvability instance in
the library, and it doubles as the precise statement of *why* the crux of the
Atlas is geometry.
-/

namespace Tetris
namespace CountingBarrier

/-- The degenerate one-row configuration: ten columns, a single row. All the
ledger theorems hold here; the game does not last one move. -/
def GameConfig.flat : GameConfig where
  cols := 10
  rows := 1
  cols_pos := by decide
  rows_pos := by decide

@[simp] theorem flat_cols : GameConfig.flat.cols = 10 := rfl
@[simp] theorem flat_rows : GameConfig.flat.rows = 1 := rfl

/-- On the empty board every piece rests at drop offset 0. -/
theorem dropOffset_empty (pl : Placement) : pl.dropOffset (∅ : Board) = 0 := by
  rw [Placement.dropOffset_eq_sup]
  refine Nat.le_zero.mp (Finset.sup_le fun cell _ => ?_)
  simp [Board.colHeight_empty]

/-- The O profile reaches height 1 in every rotation. -/
theorem O_shapeUp_mem (r : Rotation) : ((0, 1) : Coord) ∈ Piece.O.shapeUp r := by
  revert r; decide

/-- The O profile spans at most two columns in every rotation. -/
theorem O_shapeUp_col_le (r : Rotation) :
    ∀ cell ∈ Piece.O.shapeUp r, cell.1 ≤ 1 := by
  revert r; decide

/-- **The one-row board cannot absorb an O.** Whatever column and rotation the
solver answers with, the dropped O occupies rows 0 and 1, no row completes
(only two of ten columns are touched), and the surviving cell at row 1 spills
past the single-row field. -/
theorem flat_O_step_lost (pl : Placement) (hO : pl.piece = Piece.O) :
    Board.isLost GameConfig.flat (Placement.applyStep GameConfig.flat (∅ : Board) pl) := by
  -- the placed board is exactly the dropped piece
  have hplace : pl.place (∅ : Board) = pl.dropped (∅ : Board) := by
    unfold Placement.place
    exact Finset.empty_union _
  -- membership description of the dropped cells
  have hmem_dropped : ∀ q : Coord, q ∈ pl.dropped (∅ : Board) ↔
      ∃ cell ∈ pl.shapeUp, (pl.col + cell.1, cell.2) = q := by
    intro q
    unfold Placement.dropped Placement.cellsAt
    rw [dropOffset_empty, Finset.mem_image]
    constructor
    · rintro ⟨cell, hcell, rfl⟩
      exact ⟨cell, hcell, by simp⟩
    · rintro ⟨cell, hcell, rfl⟩
      exact ⟨cell, hcell, by simp⟩
  -- the top cell of the O is on the board
  have htop : (pl.col, 1) ∈ pl.place (∅ : Board) := by
    rw [hplace, hmem_dropped]
    refine ⟨(0, 1), ?_, by simp⟩
    unfold Placement.shapeUp
    rw [hO]
    exact O_shapeUp_mem pl.rot
  -- no row of the placed board is full
  have hnf : ∀ r, ¬ Board.isFull GameConfig.flat (pl.place (∅ : Board)) r := by
    intro r hfull
    -- pick a column the O does not touch
    set c₀ : ℕ := if pl.col = 0 then 2 else 0 with hc₀
    have hc₀_range : c₀ ∈ Finset.range GameConfig.flat.cols := by
      rw [Finset.mem_range, flat_cols, hc₀]
      split <;> omega
    have hc₀_mem := hfull c₀ hc₀_range
    rw [hplace, hmem_dropped] at hc₀_mem
    obtain ⟨cell, hcell, hEq⟩ := hc₀_mem
    have hcol : pl.col + cell.1 = c₀ := congrArg Prod.fst hEq
    have hle : cell.1 ≤ 1 := by
      have := O_shapeUp_col_le pl.rot cell
      unfold Placement.shapeUp at hcell
      rw [hO] at hcell
      exact this hcell
    rw [hc₀] at hcol
    split at hcol <;> omega
  -- clearing does nothing, and the top cell overflows the one-row field
  unfold Placement.applyStep
  rw [Board.clearLines_id_of_no_full hnf]
  exact ⟨(pl.col, 1), htop, by rw [flat_rows]⟩

/-- **`init` is not safe on the one-row board.** If it were, the safe set would
answer the adversary's opening O with a placement whose successor is again
safe — but every O placement tops out, and safe states are never lost. -/
theorem init_not_safe_flat : GameState.init ∉ safe GameConfig.flat := by
  intro hsafe
  obtain ⟨pl, hpiece, _, hstep⟩ := safe_forall_step hsafe Piece.O
    (by rw [GameState.init_bag]; exact Bag.mem_full Piece.O)
  refine safe_not_lost hstep (GameState.lost_of_board_isLost ?_)
  rw [adversarialStep_board, GameState.init_board_eq_emptyset]
  exact flat_O_step_lost _ rfl

/-- **The counting barrier, made concrete.** Ten-columns-one-row Tetris is not
solvable — although every ledger theorem in this library holds for it verbatim.
Any argument deriving solvability from config-generic counting alone would
prove this configuration solvable too; the distinguishing ingredient must be
the geometry of the pieces against the height of the field. -/
theorem not_tetrisSolvableValidFor_flat : ¬ TetrisSolvableValidFor GameConfig.flat := by
  rintro ⟨σ, hσ⟩
  exact init_not_safe_flat (init_safe_of_solvesTetrisValid hσ)

end CountingBarrier
end Tetris
