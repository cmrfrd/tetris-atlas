import Mathlib
import Proofs.Model.Game
import Proofs.Combinatorics.BoardCount

/-!
# Step invariants

Facts about the loss condition, well-formedness, and the cell-count dynamics of
a move. The headline result is `reachable_even_count`: on a board with an even
number of columns (e.g. the standard 10×10... 10-wide board), every reachable
board has an even number of filled cells — because a placement adds exactly 4
cells and a line clear removes a multiple of `cols`.
-/

namespace Tetris
namespace Board

/-! ## Loss basics -/

/-- The empty board is never lost. -/
theorem not_isLost_empty (cfg : GameConfig) : ¬ isLost cfg (empty) := by
  rintro ⟨p, hp, _⟩
  simp [empty] at hp

/-- Loss is monotone: adding cells can only keep a board lost. -/
theorem isLost_mono {cfg : GameConfig} {b b' : Board} (h : b ⊆ b') :
    isLost cfg b → isLost cfg b' := by
  rintro ⟨p, hp, hr⟩
  exact ⟨p, h hp, hr⟩

/-! ## Well-formedness -/

/-- The empty board is well-formed. -/
theorem WF_empty (cfg : GameConfig) : WF cfg (empty) := by
  intro p hp; simp [empty] at hp

/-- A valid placement preserves well-formedness. -/
theorem WF_place {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : WF cfg b) (hv : pl.Valid cfg) : WF cfg (pl.place b) := by
  intro p hp
  simp only [Placement.place, Finset.mem_union] at hp
  rcases hp with h | h
  · exact hb p h
  · have h2 : p ∈ Finset.image
        (fun cell => (pl.col + cell.1, pl.dropOffset b + cell.2)) pl.shapeUp := h
    obtain ⟨cell, hcell, rfl⟩ := Finset.mem_image.1 h2
    exact hv cell hcell

/-- Line clearing preserves well-formedness. -/
theorem WF_clearLines {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    WF cfg (clearLines cfg b) := by
  intro p hp
  simp only [clearLines, Finset.mem_image, Finset.mem_filter] at hp
  obtain ⟨q, ⟨hqb, _⟩, rfl⟩ := hp
  exact hb q hqb

/-! ## `clearedBelow` arithmetic -/

/-- At most `r` rows can have been cleared below row `r`. -/
theorem clearedBelow_le (cfg : GameConfig) (b : Board) (r : ℕ) :
    clearedBelow cfg b r ≤ r := by
  unfold clearedBelow
  calc ((fullRows cfg b).filter (· < r)).card
      ≤ (Finset.range r).card := by
        apply Finset.card_le_card
        intro x hx
        rw [Finset.mem_filter] at hx
        exact Finset.mem_range.2 hx.2
    _ = r := Finset.card_range r

/-- Cleared-row counts grow no faster than the rows themselves. -/
theorem clearedBelow_le_add (cfg : GameConfig) (b : Board) {r s : ℕ} (_h : r ≤ s) :
    clearedBelow cfg b s ≤ clearedBelow cfg b r + (s - r) := by
  unfold clearedBelow
  have hsub : (fullRows cfg b).filter (· < s)
      ⊆ (fullRows cfg b).filter (· < r) ∪ Finset.Ico r s := by
    intro x hx
    rw [Finset.mem_filter] at hx
    by_cases hxr : x < r
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hx.1, hxr⟩)
    · exact Finset.mem_union_right _ (Finset.mem_Ico.2 ⟨not_lt.1 hxr, hx.2⟩)
  calc ((fullRows cfg b).filter (· < s)).card
      ≤ ((fullRows cfg b).filter (· < r) ∪ Finset.Ico r s).card :=
        Finset.card_le_card hsub
    _ ≤ ((fullRows cfg b).filter (· < r)).card + (Finset.Ico r s).card :=
        Finset.card_union_le _ _
    _ = ((fullRows cfg b).filter (· < r)).card + (s - r) := by rw [Nat.card_Ico]

/-- A non-full row contributes nothing to the cleared count above it. -/
theorem clearedBelow_succ_of_not_full (cfg : GameConfig) (b : Board) {r : ℕ}
    (h : ¬ isFull cfg b r) :
    clearedBelow cfg b (r + 1) = clearedBelow cfg b r := by
  unfold clearedBelow
  congr 1
  apply Finset.filter_congr
  intro x hx
  have hxr : x ≠ r := by
    rintro rfl
    rw [fullRows, Finset.mem_filter] at hx
    exact h hx.2
  omega

/-- The post-clear row map `r ↦ r - clearedBelow r` is strictly increasing as it
crosses a non-full row: this is what makes line-clear gravity injective. -/
theorem row_map_lt (cfg : GameConfig) (b : Board) {r1 r2 : ℕ}
    (hlt : r1 < r2) (h1 : ¬ isFull cfg b r1) :
    r1 - clearedBelow cfg b r1 < r2 - clearedBelow cfg b r2 := by
  have hsucc : clearedBelow cfg b (r1 + 1) = clearedBelow cfg b r1 :=
    clearedBelow_succ_of_not_full cfg b h1
  have hle1 : clearedBelow cfg b r1 ≤ r1 := clearedBelow_le cfg b r1
  have hle2 : clearedBelow cfg b r2 ≤ r2 := clearedBelow_le cfg b r2
  have hmono : clearedBelow cfg b r2 ≤ clearedBelow cfg b (r1 + 1) + (r2 - (r1 + 1)) :=
    clearedBelow_le_add cfg b (by omega)
  omega

/-- Two non-full rows with the same post-clear position are the same row: the gravity
row-map is injective on non-full rows. The shared injectivity core consumed by
`clearLines_card`, `colCount_clearLines_add`, and `clearLines_no_full`. -/
theorem row_map_eq_of_not_full {cfg : GameConfig} {b : Board} {r1 r2 : ℕ}
    (h1 : ¬ isFull cfg b r1) (h2 : ¬ isFull cfg b r2)
    (he : r1 - clearedBelow cfg b r1 = r2 - clearedBelow cfg b r2) : r1 = r2 := by
  by_contra hne
  rcases Nat.lt_or_ge r1 r2 with hlt | hge
  · have := row_map_lt cfg b hlt h1; omega
  · have hgt : r2 < r1 := by omega
    have := row_map_lt cfg b hgt h2; omega

/-! ## Line-clear cell counts -/

/-- After clearing lines, the surviving cells are exactly the non-full-row
cells (gravity only relocates them, never merges them). -/
theorem clearLines_card (cfg : GameConfig) (b : Board) :
    (clearLines cfg b).count = (b.filter (fun p => ¬ isFull cfg b p.2)).card := by
  unfold clearLines count
  apply Finset.card_image_of_injOn
  intro p hp q hq hpq
  simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
  simp only [Prod.mk.injEq] at hpq
  obtain ⟨h1, h2⟩ := hpq
  have heq2 : p.2 = q.2 := row_map_eq_of_not_full hp.2 hq.2 h2
  exact Prod.ext_iff.mpr ⟨h1, heq2⟩

/-- The full-row cells are the union of the complete rows. -/
theorem filter_full_eq_biUnion {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    b.filter (fun p => isFull cfg b p.2)
      = (fullRows cfg b).biUnion (fun r => row cfg.cols r) := by
  ext c
  simp only [Finset.mem_filter, Finset.mem_biUnion]
  constructor
  · rintro ⟨hcb, hfull⟩
    refine ⟨c.2, ?_, ?_⟩
    · rw [fullRows, Finset.mem_filter, Finset.mem_image]
      exact ⟨⟨c, hcb, rfl⟩, hfull⟩
    · rw [row, Finset.mem_image]
      exact ⟨c.1, Finset.mem_range.2 (hb c hcb), by simp⟩
  · rintro ⟨r, hr, hcr⟩
    rw [row, Finset.mem_image] at hcr
    obtain ⟨c', hc', rfl⟩ := hcr
    rw [fullRows, Finset.mem_filter] at hr
    exact ⟨hr.2 c' hc', hr.2⟩

/-- A completely filled row has exactly `cols` cells; so the full-row cells
total `cols * (number of full rows)`. -/
theorem fullCells_card {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    (b.filter (fun p => isFull cfg b p.2)).card
      = cfg.cols * (fullRows cfg b).card := by
  rw [filter_full_eq_biUnion hb]
  rw [Finset.card_biUnion]
  · have hrow : ∀ r ∈ fullRows cfg b, (row cfg.cols r).card = cfg.cols := by
      intro r _
      have := Board.count_row cfg.cols r
      simpa [Board.count] using this
    rw [Finset.sum_congr rfl hrow, Finset.sum_const, smul_eq_mul, Nat.mul_comm]
  · intro r1 _ r2 _ hr12
    simp only [Function.onFun]
    rw [Finset.disjoint_left]
    intro c hc1 hc2
    rw [row, Finset.mem_image] at hc1 hc2
    obtain ⟨a, _, ha⟩ := hc1
    obtain ⟨a2, _, ha2⟩ := hc2
    have : (a, r1) = (a2, r2) := ha.trans ha2.symm
    rw [Prod.ext_iff] at this
    exact hr12 this.2

/-- **Line clears remove a multiple of `cols` cells.** -/
theorem clearLines_count_add {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    b.count = (clearLines cfg b).count + cfg.cols * (fullRows cfg b).card := by
  have h1 := Finset.card_filter_add_card_filter_not
    (s := b) (p := fun p => isFull cfg b p.2)
  have h2 := fullCells_card hb
  have h3 := clearLines_card cfg b
  unfold count at *
  omega

end Board

/-! ## Reachability invariants -/

/-- Every reachable board is well-formed. -/
theorem reachable_WF {cfg : GameConfig} {g : GameState} (h : Reachable cfg g) :
    Board.WF cfg g.board := by
  induction h with
  | init => exact Board.WF_empty cfg
  | @step g pl hr _ hvalid _ ih =>
      simp only [GameState.step]
      exact Board.WF_clearLines (Board.WF_place ih hvalid)

/-- **Every reachable board has an even number of filled cells**, whenever the
board has an even width. A placement adds exactly 4 cells (even) and each line
clear removes a multiple of `cols` (even when `cols` is even), so parity is
preserved from the empty board. -/
theorem reachable_even_count {cfg : GameConfig} (hcol : Even cfg.cols)
    {g : GameState} (h : Reachable cfg g) : Even g.board.count := by
  induction h with
  | init => simp [GameState.init, Board.count, Board.empty]
  | @step g pl hr _ _ _ ih =>
      have hWF : Board.WF cfg g.board := reachable_WF hr
      have hWFplace : Board.WF cfg (pl.place g.board) := Board.WF_place hWF (by assumption)
      simp only [GameState.step, Placement.applyStep]
      have key := Board.clearLines_count_add hWFplace
      have hplace := Placement.count_place g.board pl
      have hmul : Even (cfg.cols * (Board.fullRows cfg (pl.place g.board)).card) :=
        hcol.mul_right _
      have hsum := key.symm.trans hplace
      rw [Nat.even_iff] at ih hmul ⊢
      omega

/-- The standard 10-wide board: every reachable board has an even cell count. -/
theorem reachable_even_count_standard {g : GameState}
    (h : Reachable GameConfig.standard g) : Even g.board.count :=
  reachable_even_count (by decide) h

/-- The 4-wide tiny board: every reachable board has an even cell count. -/
theorem reachable_even_count_tiny {g : GameState}
    (h : Reachable GameConfig.tiny g) : Even g.board.count :=
  reachable_even_count (by decide) h

/-- Dot accessor for parity (any config with even cols). -/
theorem Reachable.even_count {cfg : GameConfig} (hcol : Even cfg.cols)
    {g : GameState} (h : Reachable cfg g) : Even g.board.count :=
  reachable_even_count hcol h

/-- Dot accessor for parity on the standard 10-wide board. -/
theorem Reachable.even_count_standard {g : GameState}
    (h : Reachable GameConfig.standard g) : Even g.board.count :=
  reachable_even_count_standard h

/-- Dot accessor for parity on the tiny 4-wide board. -/
theorem Reachable.even_count_tiny {g : GameState}
    (h : Reachable GameConfig.tiny g) : Even g.board.count :=
  reachable_even_count_tiny h

end Tetris
