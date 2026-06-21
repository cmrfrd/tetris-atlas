import Mathlib
import Proofs.Theorems.Gameplay

/-!
# Further gameplay theorems

Line-clearing algebra, hard-drop properties (including maximality), 7-bag
invariants, reachability invariants, and piece geometry. All proven.
-/

namespace Tetris

/-! ## A "top of the stack" lemma (used for hard-drop maximality) -/

/-- The highest filled cell of a nonempty column is at row `colHeight - 1`. -/
theorem top_mem {b : Board} {j : ℕ} (h : 0 < Board.colHeight b j) :
    (j, Board.colHeight b j - 1) ∈ b := by
  have hne : (Board.colRows b j).Nonempty := by
    rcases Finset.eq_empty_or_nonempty (Board.colRows b j) with he | hne
    · exfalso; rw [Board.colHeight, he] at h; simp at h
    · exact hne
  obtain ⟨x, hx, hxe⟩ := Finset.exists_mem_eq_sup _ hne (· + 1)
  have hcol : Board.colHeight b j = x + 1 := hxe
  simp only [Board.colRows, Finset.mem_image, Finset.mem_filter] at hx
  obtain ⟨cell, ⟨hcb, hcj⟩, hcx⟩ := hx
  have heq : (j, Board.colHeight b j - 1) = cell := by
    rw [hcol]; simp only [Nat.add_sub_cancel]
    exact Prod.ext_iff.mpr ⟨hcj.symm, hcx.symm⟩
  rw [heq]; exact hcb

/-! ## Line clearing -/

/-- Clearing is idempotent. -/
theorem clearLines_idem (cfg : GameConfig) (b : Board) (hcol : 0 < cfg.cols) :
    Board.clearLines cfg (Board.clearLines cfg b) = Board.clearLines cfg b :=
  Board.clearLines_id_of_no_full (fun r => Board.clearLines_no_full b hcol r)

/-- Cleared lines are bounded by the available cells. -/
theorem linesCleared_le (cfg : GameConfig) (b : Board) (hb : Board.WF cfg b) :
    cfg.cols * Board.linesCleared cfg b ≤ b.count := by
  have := Board.cells_cleared hb
  omega

/-- Clearing never raises any column's height. -/
theorem colHeight_clearLines_le (cfg : GameConfig) (b : Board) (j : ℕ) :
    Board.colHeight (Board.clearLines cfg b) j ≤ Board.colHeight b j := by
  have key : ∀ y, (j, y) ∈ Board.clearLines cfg b → y < Board.colHeight b j := by
    intro y hy
    simp only [Board.clearLines, Finset.mem_image, Finset.mem_filter] at hy
    obtain ⟨p, ⟨hpb, _⟩, hpe⟩ := hy
    rw [Prod.mk.injEq] at hpe
    have hple : p.2 < Board.colHeight b p.1 := Board.lt_colHeight hpb
    rw [hpe.1] at hple
    omega
  change (Board.colRows (Board.clearLines cfg b) j).sup (· + 1) ≤ Board.colHeight b j
  apply Finset.sup_le
  intro x hx
  simp only [Board.colRows, Finset.mem_image, Finset.mem_filter] at hx
  obtain ⟨cell, ⟨hcl, hcj⟩, hcx⟩ := hx
  have hjx : (j, x) ∈ Board.clearLines cfg b := by
    have : cell = (j, x) := Prod.ext_iff.mpr ⟨hcj, hcx⟩
    rw [← this]; exact hcl
  have := key x hjx
  omega

/-- A single hard drop onto a board with no pending full rows clears ≤ 4 lines. -/
theorem linesCleared_place_le_four (cfg : GameConfig) (b : Board) (pl : Placement)
    (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    Board.linesCleared cfg (pl.place b) ≤ 4 := by
  have hsub : Board.fullRows cfg (pl.place b) ⊆ (pl.dropped b).image Prod.snd := by
    intro r hr
    simp only [Board.fullRows, Finset.mem_filter] at hr
    obtain ⟨c, hc, hcb⟩ : ∃ c ∈ Finset.range cfg.cols, (c, r) ∉ b := by
      by_contra hcon
      push Not at hcon
      exact hnf r hcon
    have hcplace : (c, r) ∈ pl.place b := hr.2 c hc
    have hcdrop : (c, r) ∈ pl.dropped b := by
      simp only [Placement.place, Finset.mem_union] at hcplace
      rcases hcplace with h | h
      · exact absurd h hcb
      · exact h
    rw [Finset.mem_image]
    exact ⟨(c, r), hcdrop, rfl⟩
  calc Board.linesCleared cfg (pl.place b)
      = (Board.fullRows cfg (pl.place b)).card := rfl
    _ ≤ ((pl.dropped b).image Prod.snd).card := Finset.card_le_card hsub
    _ ≤ (pl.dropped b).card := Finset.card_image_le
    _ = 4 := Placement.card_dropped b pl

/-! ## Hard drop & placement -/

/-- Column height is monotone under adding cells. -/
theorem colHeight_mono {b b' : Board} (h : b ⊆ b') (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight b' j := by
  unfold Board.colHeight Board.colRows
  apply Finset.sup_mono
  apply Finset.image_subset_image
  exact Finset.filter_subset_filter _ h

/-- A placement only adds cells. -/
theorem subset_place (b : Board) (pl : Placement) : b ⊆ pl.place b := by
  unfold Placement.place
  exact Finset.subset_union_left

/-- A placement never lowers a column's height. -/
theorem colHeight_le_place (b : Board) (pl : Placement) (j : ℕ) :
    Board.colHeight b j ≤ Board.colHeight (pl.place b) j :=
  colHeight_mono (subset_place b pl) j

/-- Dropping onto the empty board lands on the floor (no shift). -/
theorem dropOffset_empty (pl : Placement) : pl.dropOffset Board.empty = 0 := by
  have hz : ∀ j, Board.colHeight Board.empty j = 0 := by
    intro j; unfold Board.colHeight Board.colRows Board.empty; simp
  apply Nat.le_zero.1
  apply Finset.sup_le
  intro cell _
  rw [hz]; simp

/-- Hard-drop maximality: if the piece did not land on the floor, dropping it one
row further would overlap the stack. -/
theorem dropped_resting (b : Board) (pl : Placement) (h : 0 < pl.dropOffset b) :
    ¬ Disjoint (pl.cellsAt (pl.dropOffset b - 1)) b := by
  have hne : (pl.shapeUp).Nonempty := by
    rw [← Finset.card_pos]
    unfold Placement.shapeUp
    rw [Piece.shapeUp_card]; norm_num
  obtain ⟨cell, hcell, hsup⟩ :=
    Finset.exists_mem_eq_sup pl.shapeUp hne
      (fun c => Board.colHeight b (pl.col + c.1) - c.2)
  have hd : pl.dropOffset b = Board.colHeight b (pl.col + cell.1) - cell.2 := hsup
  have hch : Board.colHeight b (pl.col + cell.1) = pl.dropOffset b + cell.2 := by omega
  have hcolpos : 0 < Board.colHeight b (pl.col + cell.1) := by omega
  have htop := top_mem hcolpos
  rw [Finset.not_disjoint_iff]
  refine ⟨(pl.col + cell.1, (pl.dropOffset b - 1) + cell.2), ?_, ?_⟩
  · simp only [Placement.cellsAt, Finset.mem_image]
    exact ⟨cell, hcell, rfl⟩
  · have hrow : (pl.dropOffset b - 1) + cell.2 = Board.colHeight b (pl.col + cell.1) - 1 := by
      omega
    rw [hrow]; exact htop

/-- Cell-count law for a full move. -/
theorem applyStep_count (cfg : GameConfig) (b : Board) (pl : Placement)
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    (Placement.applyStep cfg b pl).count
      + cfg.cols * Board.linesCleared cfg (pl.place b) = b.count + 4 := by
  have hWF : Board.WF cfg (pl.place b) := Board.WF_place hb hv
  have h1 := Board.cells_cleared hWF
  have h2 := Placement.count_place b pl
  unfold Placement.applyStep
  omega

/-! ## Loss -/

/-- Either the board is lost or every column fits within the field. -/
theorem isLost_or_bounded (cfg : GameConfig) (b : Board) :
    Board.isLost cfg b ∨ ∀ j, Board.colHeight b j ≤ cfg.rows := by
  by_cases h : ∃ j, cfg.rows < Board.colHeight b j
  · exact Or.inl ((Board.isLost_iff cfg b).2 h)
  · push Not at h; exact Or.inr h

/-- A placement whose columns all fit within the field is not lost. -/
theorem not_isLost_place_of_heights (cfg : GameConfig) (b : Board) (pl : Placement)
    (h : ∀ j, Board.colHeight (pl.place b) j ≤ cfg.rows) :
    ¬ Board.isLost cfg (pl.place b) := by
  rw [Board.isLost_iff]; push Not; exact h

/-! ## 7-bag randomizer -/

/-- Every piece is drawable from a full bag. -/
theorem canDraw_full (p : Piece) : Bag.full.canDraw p := Finset.mem_univ p

/-- A draw either removes one piece or refills to a full bag. -/
theorem draw_card (bag : Bag) (p : Piece) (hp : p ∈ bag) :
    (bag.draw p).card = bag.card - 1 ∨ bag.draw p = Bag.full := by
  unfold Bag.draw
  split_ifs with h
  · exact Or.inr rfl
  · exact Or.inl (Finset.card_erase_of_mem hp)

/-- No immediate repeats within a bag. -/
theorem not_canDraw_after_draw (bag : Bag) (p : Piece) (_hp : p ∈ bag)
    (hne : bag.draw p ≠ Bag.full) : ¬ (bag.draw p).canDraw p := by
  unfold Bag.canDraw
  unfold Bag.draw at hne ⊢
  split_ifs at hne ⊢ with h
  · exact absurd rfl hne
  · exact Finset.notMem_erase p bag

/-! ## Reachability invariants -/

/-- The bag is always nonempty during reachable play. -/
theorem reachable_bag_nonempty {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag.Nonempty := by
  induction h with
  | init => exact Bag.full_nonempty
  | @step g pl _ _ _ _ _ => exact Bag.draw_nonempty g.bag pl.piece

/-- The bag is always a sub-bag of the full 7-piece bag. -/
theorem reachable_bag_subset {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.bag ⊆ Bag.full := Finset.subset_univ g.bag

/-- Reachable boards have no pending full rows. -/
theorem reachable_no_full {cfg : GameConfig} (hcol : 0 < cfg.cols) {g : GameState}
    (h : Reachable cfg g) : ∀ r, ¬ Board.isFull cfg g.board r := by
  induction h with
  | init =>
      simp only [GameState.init]
      intro r hf
      have := hf 0 (Finset.mem_range.2 hcol)
      simp [Board.empty] at this
  | @step g pl _ _ _ _ _ =>
      simp only [GameState.step, Placement.applyStep]
      intro r
      exact Board.clearLines_no_full _ hcol r

/-- Generalized parity/conservation: a reachable board's cell count is a multiple
of `gcd(4, cols)`. -/
theorem reachable_count_mod {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.board.count % Nat.gcd 4 cfg.cols = 0 := by
  induction h with
  | init => simp [GameState.init, Board.empty, Board.count]
  | @step g pl hr _ hvalid _ ih =>
      have hWF := reachable_WF hr
      have hstep := applyStep_count cfg g.board pl hWF hvalid
      simp only [GameState.step]
      have hdvdN : Nat.gcd 4 cfg.cols ∣ (Placement.applyStep cfg g.board pl).count := by
        have hN : (Placement.applyStep cfg g.board pl).count
            = (g.board.count + 4)
                - cfg.cols * Board.linesCleared cfg (pl.place g.board) := by omega
        rw [hN]
        apply Nat.dvd_sub
        · exact dvd_add (Nat.dvd_of_mod_eq_zero ih) (Nat.gcd_dvd_left 4 cfg.cols)
        · exact (Nat.gcd_dvd_right 4 cfg.cols).mul_right _
      obtain ⟨c, hc⟩ := hdvdN
      rw [hc]; exact Nat.mul_mod_right _ c

/-! ## Piece geometry -/

/-- Every piece cell lies in the 4×4 bounding box. -/
theorem shape_lt_four (p : Piece) (r : Rotation) :
    ∀ cell ∈ p.shape r, cell.1 < 4 ∧ cell.2 < 4 := by
  fin_cases r <;> cases p <;> decide

/-- The bottom-up profile occupies the same columns as the raw shape. -/
theorem shapeUp_cols (p : Piece) (r : Rotation) :
    (p.shapeUp r).image Prod.fst = (p.shape r).image Prod.fst := by
  fin_cases r <;> cases p <;> decide

/-- The O piece is rotation-invariant. -/
theorem shape_O_rotation_invariant (r : Rotation) :
    Piece.O.shape r = Piece.O.shape 0 := by
  fin_cases r <;> rfl

end Tetris
