import Mathlib
import Proofs.Safety.HorizonCompactness

/-!
# Headroom-graded safety: no kill certificate of depth five exists

The compactness program (`solvable_iff_forall_horizon`) reduces solvability to
"for every `N`, no adversarial kill-tree of depth `N` defeats the player". This
file climbs the first five rungs of that ladder, with a theorem that grades
safety by *headroom*:

* **Four rows of headroom buy one guaranteed step, unconditionally**
  (`mem_safeIterate_of_headroom`): a state whose every cell has `4k` rows of
  clearance above it lies in `safeIterate k` — any valid placement works, no
  strategy required, because a hard drop raises the stack by at most four rows
  and line clears only lower it.
* The empty board has twenty rows of headroom, so
  (`init_mem_safeIterate_five`): **`init ∈ safeIterate 5` — no adversarial kill
  certificate of depth ≤ 5 exists**, and this is certified with zero search.

The grading is tight for strategy-free play: five vertical pieces stacked in
one column reach height 20, and a sixth tops out — so depth six is exactly
where survival first *requires* a decision. Every rung beyond five must engage
strategy, and the kill-certificate refutation program starts there.
-/

namespace Tetris

/-- On a wide-enough board every piece admits a valid placement: rotation `0`
at column `0` fits inside the four-column bounding box. -/
theorem exists_valid_placement_of_cols {cfg : GameConfig} (hcols : 4 ≤ cfg.cols)
    (p : Piece) : ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg := by
  refine ⟨⟨p, 0, 0⟩, rfl, ?_⟩
  intro cell hcell
  have h4 := Piece.shapeUp_col_lt_four p 0 cell hcell
  dsimp only
  omega

/-- Column heights are bounded by any uniform strict row bound on the cells. -/
theorem colHeight_le_of_forall_row_lt {b : Board} {H : ℕ}
    (hb : ∀ q ∈ b, q.2 < H) (j : ℕ) : b.colHeight j ≤ H := by
  refine Finset.sup_le fun r hr => ?_
  unfold Board.colRows at hr
  rw [Finset.mem_image] at hr
  obtain ⟨q, hq, rfl⟩ := hr
  rw [Finset.mem_filter] at hq
  exact hb q hq.1

/-- A hard drop lands within four rows of the stack top: every dropped cell
sits strictly below `H + 4` whenever the board's cells sit strictly below `H`. -/
theorem dropped_row_lt {b : Board} {H : ℕ} (hb : ∀ q ∈ b, q.2 < H)
    (pl : Placement) {q : Coord} (hq : q ∈ pl.dropped b) : q.2 < H + 4 := by
  unfold Placement.dropped Placement.cellsAt at hq
  rw [Finset.mem_image] at hq
  obtain ⟨cell, hcell, rfl⟩ := hq
  have hcell4 : cell.2 < 4 := Piece.shapeUp_row_lt_four _ _ cell hcell
  have hoff : pl.dropOffset b ≤ H := by
    rw [Placement.dropOffset_eq_sup]
    refine Finset.sup_le fun c _ => ?_
    have := colHeight_le_of_forall_row_lt hb (pl.col + c.1)
    omega
  dsimp only
  omega

/-- Line clears never raise a cell: every surviving cell sits at or below the
row of some pre-clear cell. -/
theorem clearLines_row_le {cfg : GameConfig} {b : Board} {q : Coord}
    (hq : q ∈ Board.clearLines cfg b) : ∃ q' ∈ b, q.2 ≤ q'.2 := by
  unfold Board.clearLines at hq
  rw [Finset.mem_image] at hq
  obtain ⟨p, hp, rfl⟩ := hq
  rw [Finset.mem_filter] at hp
  exact ⟨p, hp.1, by dsimp only; omega⟩

/-- **One full move raises the stack by at most four rows**: cells of
`applyStep` sit strictly below `H + 4` whenever the board's cells sit strictly
below `H` — for every placement, valid or not. -/
theorem applyStep_row_lt {cfg : GameConfig} {b : Board} {H : ℕ}
    (hb : ∀ q ∈ b, q.2 < H) (pl : Placement) {q : Coord}
    (hq : q ∈ Placement.applyStep cfg b pl) : q.2 < H + 4 := by
  unfold Placement.applyStep at hq
  obtain ⟨q', hq', hle⟩ := clearLines_row_le hq
  unfold Placement.place at hq'
  rcases Finset.mem_union.mp hq' with h | h
  · have := hb q' h
    omega
  · have := dropped_row_lt hb pl h
    omega

/-- **Headroom-graded safety.** A state whose every cell has `4k` rows of
clearance lies in the `k`-th safety iterate: with four rows of headroom per
step, *any* valid placement survives — the adversary's piece choice is
irrelevant and no strategy is needed. -/
theorem mem_safeIterate_of_headroom {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) :
    ∀ k, 4 * k ≤ cfg.rows → ∀ g : GameState,
      (∀ q ∈ g.board, q.2 + 4 * k ≤ cfg.rows) → g ∈ safeIterate cfg k := by
  intro k
  induction k with
  | zero =>
    intro _ g _
    rw [safeIterate_zero]
    trivial
  | succ k ih =>
    intro hR g hg
    rw [safeIterate_succ]
    refine ⟨?_, ?_⟩
    · rw [GameState.not_lost_iff_forall_row_lt]
      intro q hq
      have := hg q hq
      omega
    · intro p hp
      obtain ⟨pl, hpiece, hvalid⟩ := exists_valid_placement_of_cols hcols p
      refine ⟨pl, hpiece, hvalid, ih (by omega) _ ?_⟩
      intro q hq
      rw [adversarialStep_board] at hq
      have hb : ∀ q' ∈ g.board, q'.2 < cfg.rows - (4 * k + 3) := by
        intro q' hq'
        have := hg q' hq'
        omega
      have hstep := applyStep_row_lt hb _ hq
      omega

/-- The empty board's twenty rows of headroom certify five safe steps. -/
theorem init_mem_safeIterate_of_le {k : ℕ} (hk : k ≤ 5) :
    GameState.init ∈ safeIterate GameConfig.standard k := by
  refine mem_safeIterate_of_headroom (by norm_num) k ?_ GameState.init ?_
  · rw [GameConfig.standard_rows]
    omega
  · intro q hq
    exact absurd hq (GameState.init_board_no_mem q)

/-- **No adversarial kill certificate of depth five exists.** The first five
rungs of the compactness ladder (`solvable_iff_forall_horizon`), certified with
zero search. Tight for strategy-free play: five vertical pieces in one column
reach height 20 and a sixth tops out, so depth six is exactly where survival
first requires a decision. -/
theorem init_mem_safeIterate_five :
    GameState.init ∈ safeIterate GameConfig.standard 5 :=
  init_mem_safeIterate_of_le le_rfl

end Tetris
