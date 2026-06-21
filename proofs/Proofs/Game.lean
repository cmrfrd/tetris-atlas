import Mathlib
import Proofs.Board
import Proofs.Placement
import Proofs.Bag
import Proofs.Theorems.PieceGeometry

/-!
# Game states and reachability

A game state pairs the board with the current 7-bag. Play starts from the empty
board with a full bag; each move draws a legal piece, hard-drops a valid
placement, and clears completed lines. `Reachable` is the inductive set of
states attainable from the start by legal, in-bounds, non-losing moves.
-/

namespace Tetris

namespace Placement

/-- A placement is in-bounds when every cell of its drop profile lands within
the configured columns. -/
def Valid (cfg : GameConfig) (pl : Placement) : Prop :=
  ∀ cell ∈ pl.shapeUp, pl.col + cell.1 < cfg.cols

instance (cfg : GameConfig) (pl : Placement) : Decidable (pl.Valid cfg) := by
  unfold Valid; infer_instance

/-- A valid placement has `pl.col < cfg.cols`, witnessed by the column-0 cell
of `shapeUp`. -/
theorem Valid.col_lt_cols {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) : pl.col < cfg.cols := by
  obtain ⟨cell, hcell, hc⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
  have := h cell hcell
  omega

/-- Reformulation of `Valid` as a cell-translation column bound. -/
theorem Valid.col_add_lt {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) {cell : Coord} (hcell : cell ∈ pl.shapeUp) :
    pl.col + cell.1 < cfg.cols :=
  h cell hcell

/-- A valid placement keeps `cfg.cols > 0` (witnessed by `pl.col < cfg.cols`). -/
theorem Valid.cols_pos {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) : 0 < cfg.cols :=
  Nat.lt_of_le_of_lt (Nat.zero_le _) h.col_lt_cols

/-- A valid placement implies `cfg.cols ≠ 0`. -/
theorem Valid.cols_ne_zero {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) : cfg.cols ≠ 0 :=
  Nat.ne_of_gt h.cols_pos

/-- A valid placement implies `1 ≤ cfg.cols`. -/
theorem Valid.one_le_cols {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) : 1 ≤ cfg.cols :=
  h.cols_pos

/-- For a valid placement, `pl.col ≤ cfg.cols - 1`. -/
theorem Valid.col_le_pred_cols {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) : pl.col ≤ cfg.cols - 1 := by
  have := h.col_lt_cols
  omega

/-- For a valid placement, `pl.col < cfg.cols + n` for any `n`. -/
theorem Valid.col_lt_cols_add {cfg : GameConfig} {pl : Placement}
    (h : pl.Valid cfg) (n : ℕ) : pl.col < cfg.cols + n :=
  Nat.lt_of_lt_of_le h.col_lt_cols (Nat.le_add_right _ _)

/-- `place` preserves well-formedness for valid placements: every cell of
the dropped piece lands at `pl.col + cell.1 < cfg.cols` by `Valid`. -/
theorem place_wf {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    Board.WF cfg (pl.place b) := by
  intro p hp
  unfold place at hp
  rcases Finset.mem_union.mp hp with hb' | hd
  · exact hb p hb'
  · unfold dropped cellsAt shapeUp at hd
    rw [Finset.mem_image] at hd
    obtain ⟨cell, hcell, rfl⟩ := hd
    exact hv cell hcell

/-- `applyStep` preserves well-formedness for valid placements. Chain:
`place_wf` then `Board.clearLines_wf` (vertical-only mutation). -/
theorem applyStep_wf {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    Board.WF cfg (applyStep cfg b pl) :=
  Board.clearLines_wf (place_wf hb hv)

/-- All valid placements playing piece `p` on a board of width `cfg.cols`. The
enumeration covers `(rotation, column) ∈ Fin 4 × range cfg.cols` and then
filters by `Valid`. Any genuinely valid placement satisfies `col < cfg.cols`
(since piece shapes start at column offset 0), so this range is exhaustive. -/
def allValidFor (cfg : GameConfig) (p : Piece) : Finset Placement :=
  (((Finset.range cfg.cols) ×ˢ (Finset.univ : Finset Rotation)).image
      (fun rc : ℕ × Rotation => (⟨p, rc.2, rc.1⟩ : Placement)))
    |>.filter (fun pl => pl.Valid cfg)

/-- Spec: a placement is in `allValidFor cfg p` iff it plays piece `p` and is
in-bounds. -/
theorem mem_allValidFor (cfg : GameConfig) (p : Piece) (pl : Placement) :
    pl ∈ allValidFor cfg p ↔ pl.piece = p ∧ pl.Valid cfg := by
  unfold allValidFor
  rw [Finset.mem_filter, Finset.mem_image]
  refine ⟨?_, ?_⟩
  · rintro ⟨⟨rc, _, hrc⟩, hvalid⟩
    refine ⟨?_, hvalid⟩
    rw [← hrc]
  · rintro ⟨hpiece, hvalid⟩
    refine ⟨⟨(pl.col, pl.rot), ?_, ?_⟩, hvalid⟩
    · rw [Finset.mem_product]
      refine ⟨?_, Finset.mem_univ _⟩
      rw [Finset.mem_range]
      obtain ⟨cell, hcell, hc1⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
      have hh := hvalid cell hcell
      rw [hc1, Nat.add_zero] at hh
      exact hh
    · rw [← hpiece]

end Placement

/-- A game state: the current board together with the current 7-bag. -/
structure GameState where
  board : Board
  bag : Bag
deriving DecidableEq

namespace GameState

/-- Every game state's bag has at most 7 pieces (it's a subset of `Bag.full`). -/
theorem bag_card_le_seven (g : GameState) : g.bag.card ≤ 7 := Bag.card_le_seven _

/-- Every game state's bag has fewer than 8 pieces. -/
theorem bag_card_lt_eight (g : GameState) : g.bag.card < 8 :=
  Nat.lt_succ_of_le (bag_card_le_seven g)

/-- Every game state's bag card is `≤ 7`. Bundled with `Bag.subset_full`. -/
theorem bag_card_le_seven_and_subset_full (g : GameState) :
    g.bag.card ≤ 7 ∧ g.bag ⊆ Bag.full :=
  ⟨bag_card_le_seven g, Bag.subset_full _⟩

/-- A state's bag has card 7 iff it equals the full bag. -/
theorem bag_card_eq_seven_iff_eq_full (g : GameState) :
    g.bag.card = 7 ↔ g.bag = Bag.full :=
  Bag.card_eq_seven_iff_eq_full g.bag

/-- Every state's bag is a subset of `Bag.full`. -/
theorem bag_subset_full (g : GameState) : g.bag ⊆ Bag.full :=
  Finset.subset_univ _

/-- The initial state: empty board, full bag. -/
def init : GameState := ⟨Board.empty, Bag.full⟩

@[simp] theorem init_bag : init.bag = Bag.full := rfl

@[simp] theorem init_board : init.board = Board.empty := rfl

@[simp] theorem init_board_count : init.board.count = 0 := by
  rw [init_board]
  rfl

theorem init_bag_card : init.bag.card = 7 := by rw [init_bag, Bag.full_card]

theorem init_bag_nonempty : init.bag.Nonempty := init_bag.symm ▸ Bag.full_nonempty

/-- The init state's bag is not empty. -/
theorem init_bag_ne_empty : init.bag ≠ ∅ := init_bag_nonempty.ne_empty

/-- The init state's board equals `∅`. -/
@[simp] theorem init_board_eq_emptyset : init.board = (∅ : Board) := by
  rw [init_board]; rfl

/-- The init state's board has card 0. -/
@[simp] theorem init_board_card : init.board.card = 0 := init_board_count

/-- The init state's board is empty (not nonempty). -/
theorem init_board_not_nonempty : ¬ init.board.Nonempty := by
  rw [init_board_eq_emptyset]
  exact Finset.not_nonempty_empty

/-- The init state's board has no cells. -/
theorem init_board_no_mem (p : Coord) : p ∉ init.board := by
  rw [init_board_eq_emptyset]
  exact Finset.notMem_empty p

/-- The init state's board has zero column height in every column. -/
@[simp] theorem init_board_colHeight (j : ℕ) : init.board.colHeight j = 0 := by
  rw [init_board_eq_emptyset]; exact Board.colHeight_empty j

/-- The init state trivially satisfies any `colHeight ≤ N` bound. -/
theorem init_board_colHeight_le (N j : ℕ) :
    init.board.colHeight j ≤ N := by
  rw [init_board_colHeight]; exact Nat.zero_le _

/-- The init state's stack is "low" for any rows ≥ 4 (used by `low_stack_safe`). -/
theorem init_board_low_stack (cfg : GameConfig) :
    ∀ j, init.board.colHeight j ≤ cfg.rows - 4 :=
  fun j => init_board_colHeight_le (cfg.rows - 4) j

/-- Every piece is drawable from `init`'s bag (the full bag). -/
theorem init_canDraw (p : Piece) : init.bag.canDraw p :=
  init_bag.symm ▸ Bag.canDraw_full p


/-- The init state's board is well-formed (vacuously — it's empty). -/
theorem init_board_wf (cfg : GameConfig) : Board.WF cfg init.board := by
  rw [init_board]; exact Board.empty_wf cfg

/-- The init state's board has no cells above the field (vacuously). -/
theorem init_board_in_field (cfg : GameConfig) :
    ∀ p ∈ init.board, p.2 < cfg.rows := by
  intro p hp
  rw [init_board] at hp
  exact absurd hp (Finset.notMem_empty _)


/-- The game is lost when the board has spilled above the visible field. -/
def lost (cfg : GameConfig) (g : GameState) : Prop := Board.isLost cfg g.board

instance (cfg : GameConfig) (g : GameState) : Decidable (g.lost cfg) := by
  unfold lost; infer_instance

/-- `g.lost cfg ↔ Board.isLost cfg g.board` (definitional unfold). -/
theorem lost_iff_board_isLost (cfg : GameConfig) (g : GameState) :
    g.lost cfg ↔ Board.isLost cfg g.board :=
  Iff.rfl

/-- `¬ g.lost cfg ↔ ¬ Board.isLost cfg g.board`. -/
theorem not_lost_iff_not_board_isLost (cfg : GameConfig) (g : GameState) :
    ¬ g.lost cfg ↔ ¬ Board.isLost cfg g.board :=
  Iff.rfl

/-- Lift `Board.isLost` to `GameState.lost`. -/
theorem lost_of_board_isLost {cfg : GameConfig} {g : GameState}
    (h : Board.isLost cfg g.board) : g.lost cfg :=
  h

/-- Extract `Board.isLost` from `GameState.lost`. -/
theorem board_isLost_of_lost {cfg : GameConfig} {g : GameState}
    (h : g.lost cfg) : Board.isLost cfg g.board :=
  h

/-- **A state is non-lost iff its board is in-field.** Game-state lift of
`Board.not_isLost_iff_forall_row_lt`: characterises `¬ g.lost cfg` as "every
filled cell of `g.board` is strictly below `cfg.rows`" — the exact second
component of the `InFieldBoard` predicate. -/
theorem not_lost_iff_forall_row_lt (cfg : GameConfig) (g : GameState) :
    ¬ g.lost cfg ↔ ∀ p ∈ g.board, p.2 < cfg.rows := by
  rw [not_lost_iff_not_board_isLost]
  exact Board.not_isLost_iff_forall_row_lt cfg g.board


/-- The init state is not lost (the empty board cannot have an overflowing cell). -/
theorem init_not_lost (cfg : GameConfig) : ¬ init.lost cfg := by
  unfold lost Board.isLost
  rw [init_board]
  push Not
  intro p hp
  exact absurd hp (Finset.notMem_empty _)

/-- Apply one move: hard-drop `pl` (with line clears) and draw its piece. -/
def step (cfg : GameConfig) (g : GameState) (pl : Placement) : GameState :=
  ⟨Placement.applyStep cfg g.board pl, g.bag.draw pl.piece⟩

@[simp] theorem step_bag (cfg : GameConfig) (g : GameState) (pl : Placement) :
    (g.step cfg pl).bag = g.bag.draw pl.piece := rfl

@[simp] theorem step_board (cfg : GameConfig) (g : GameState) (pl : Placement) :
    (g.step cfg pl).board = Placement.applyStep cfg g.board pl := rfl

/-- Post-step `lost` predicate unfolds via the board. -/
theorem step_lost_iff (cfg : GameConfig) (g : GameState) (pl : Placement) :
    (g.step cfg pl).lost cfg ↔ Board.isLost cfg (Placement.applyStep cfg g.board pl) :=
  Iff.rfl

/-- Post-step `¬ lost` predicate unfolds via the board. -/
theorem not_step_lost_iff (cfg : GameConfig) (g : GameState) (pl : Placement) :
    ¬ (g.step cfg pl).lost cfg ↔
      ¬ Board.isLost cfg (Placement.applyStep cfg g.board pl) :=
  Iff.rfl

/-- Lift `¬ Board.isLost` of the post-applyStep board to `¬ step.lost`. -/
theorem not_step_lost_of_not_board_isLost {cfg : GameConfig}
    {g : GameState} {pl : Placement}
    (h : ¬ Board.isLost cfg (Placement.applyStep cfg g.board pl)) :
    ¬ (g.step cfg pl).lost cfg :=
  h

/-- Extract `¬ Board.isLost` from `¬ step.lost`. -/
theorem not_board_isLost_of_not_step_lost {cfg : GameConfig}
    {g : GameState} {pl : Placement} (h : ¬ (g.step cfg pl).lost cfg) :
    ¬ Board.isLost cfg (Placement.applyStep cfg g.board pl) :=
  h

/-- Post-init-step `lost` predicate unfolds via the empty-board applyStep. -/
theorem init_step_lost_iff (cfg : GameConfig) (pl : Placement) :
    (init.step cfg pl).lost cfg ↔ Board.isLost cfg (Placement.applyStep cfg ∅ pl) := by
  rw [step_lost_iff, init_board_eq_emptyset]

/-- Post-init-step `¬ lost` predicate unfolds via the empty-board applyStep. -/
theorem not_init_step_lost_iff (cfg : GameConfig) (pl : Placement) :
    ¬ (init.step cfg pl).lost cfg ↔
      ¬ Board.isLost cfg (Placement.applyStep cfg ∅ pl) := by
  rw [init_step_lost_iff]

/-- The post-init-step state's board is the applyStep of empty. -/
@[simp] theorem init_step_board (cfg : GameConfig) (pl : Placement) :
    (init.step cfg pl).board = Placement.applyStep cfg ∅ pl := by
  rw [step_board, init_board_eq_emptyset]

/-- Post-step bag is the full bag iff the pre-step `bag.erase` is empty
(refill condition). -/
theorem step_bag_eq_full_iff (cfg : GameConfig) (g : GameState) (pl : Placement) :
    (g.step cfg pl).bag = Bag.full ↔ g.bag.erase pl.piece = ∅ := by
  rw [step_bag]; exact Bag.draw_eq_full_iff_erase_eq_empty g.bag pl.piece

/-- Post-step bag is either the erased pre-step bag or refills to full. -/
theorem step_bag_eq_erase_or_full (cfg : GameConfig) (g : GameState)
    (pl : Placement) :
    (g.step cfg pl).bag = g.bag.erase pl.piece ∨
      (g.step cfg pl).bag = Bag.full := by
  rw [step_bag]; exact Bag.draw_eq_erase_or_full g.bag pl.piece

/-- Post-step bag is a subset of pre-step or refills to full. -/
theorem step_bag_subset_or_eq_full (cfg : GameConfig) (g : GameState)
    (pl : Placement) :
    (g.step cfg pl).bag ⊆ g.bag ∨ (g.step cfg pl).bag = Bag.full := by
  rw [step_bag]; exact Bag.draw_subset_or_eq_full g.bag pl.piece

/-- Post-step bag card is ≤ pre-step or refills to 7. -/
theorem step_bag_card_le_or_eq_seven (cfg : GameConfig) (g : GameState)
    (pl : Placement) :
    (g.step cfg pl).bag.card ≤ g.bag.card ∨ (g.step cfg pl).bag.card = 7 := by
  rw [step_bag]; exact Bag.draw_card_le_card_or_eq_seven g.bag pl.piece

/-- Post-step bag is never empty. -/
theorem step_bag_ne_empty (cfg : GameConfig) (g : GameState)
    (pl : Placement) : (g.step cfg pl).bag ≠ ∅ := by
  rw [step_bag]; exact Bag.draw_ne_empty g.bag pl.piece

/-- Post-step bag is nonempty. -/
theorem step_bag_nonempty (cfg : GameConfig) (g : GameState)
    (pl : Placement) : (g.step cfg pl).bag.Nonempty := by
  rw [step_bag]; exact Bag.draw_nonempty g.bag pl.piece

/-- Post-step bag has card ≥ 1. -/
theorem step_bag_card_pos (cfg : GameConfig) (g : GameState)
    (pl : Placement) : 0 < (g.step cfg pl).bag.card :=
  Finset.card_pos.mpr (step_bag_nonempty cfg g pl)

/-- Post-step bag has card ≤ 7. -/
theorem step_bag_card_le_seven (cfg : GameConfig) (g : GameState)
    (pl : Placement) : (g.step cfg pl).bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- Bundled `[1, 7]` interval for post-step bag-card. -/
theorem step_bag_card_interval (cfg : GameConfig) (g : GameState)
    (pl : Placement) :
    1 ≤ (g.step cfg pl).bag.card ∧ (g.step cfg pl).bag.card ≤ 7 :=
  ⟨step_bag_card_pos cfg g pl, step_bag_card_le_seven cfg g pl⟩

/-- Post-step bag card is nonzero. -/
theorem step_bag_card_ne_zero (cfg : GameConfig) (g : GameState)
    (pl : Placement) : (g.step cfg pl).bag.card ≠ 0 :=
  Nat.ne_of_gt (step_bag_card_pos cfg g pl)

/-- Post-step bag card is strictly less than 8. -/
theorem step_bag_card_lt_eight (cfg : GameConfig) (g : GameState)
    (pl : Placement) : (g.step cfg pl).bag.card < 8 :=
  Nat.lt_succ_of_le (step_bag_card_le_seven cfg g pl)

/-- Post-init-step bag is never `Bag.full` (the drawn piece is now missing). -/
theorem init_step_bag_ne_full (cfg : GameConfig) (pl : Placement) :
    (init.step cfg pl).bag ≠ Bag.full := by
  rw [step_bag, init_bag]; exact Bag.draw_full_ne_full pl.piece


/-- **One step from `init` is never `init`.** Post-step bag is
`Bag.full.draw pl.piece`, which by `Bag.draw_ne_self` differs from
`Bag.full = init.bag`. -/
theorem init_step_ne_init {cfg : GameConfig} (pl : Placement) :
    init.step cfg pl ≠ init := by
  intro heq
  have hbag : (init.step cfg pl).bag = init.bag := congrArg GameState.bag heq
  change Bag.full.draw pl.piece = init.bag at hbag
  rw [init_bag] at hbag
  exact Bag.draw_ne_self (Bag.mem_full pl.piece) hbag

/-- **Concrete bag after one step from `init`**: equals
`Bag.full.erase pl.piece`. The 6-piece remainder after the first
draw — no refill since `Bag.full` has 7 pieces. -/
@[simp] theorem init_step_bag {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag = Bag.full.erase pl.piece := by
  change Bag.full.draw pl.piece = Bag.full.erase pl.piece
  exact Bag.draw_full_eq_erase pl.piece

/-- The bag after one step from `init` has exactly 6 pieces. -/
@[simp] theorem init_step_bag_card {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card = 6 := by
  rw [init_step_bag]; exact Bag.full_erase_card pl.piece

/-- The piece used in the one-step transition from `init` is no longer
in the post-step bag. -/
theorem init_step_bag_notMem {cfg : GameConfig} (pl : Placement) :
    pl.piece ∉ (init.step cfg pl).bag := by
  rw [init_step_bag]; exact Finset.notMem_erase _ _

/-- Post-init-step bag card is `< 7`. -/
theorem init_step_bag_card_lt_seven {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card < 7 := by
  rw [init_step_bag_card]; decide

/-- Post-init-step bag card is `≤ 6`. -/
theorem init_step_bag_card_le_six {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card ≤ 6 := by
  rw [init_step_bag_card]

/-- Post-init-step bag card is not 7. -/
theorem init_step_bag_card_ne_seven {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card ≠ 7 := by
  rw [init_step_bag_card]; decide

/-- Post-init-step bag card is strictly less than init's bag card. -/
theorem init_step_bag_card_lt_init_bag_card {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card < init.bag.card := by
  rw [init_step_bag_card, init_bag_card]; decide

/-- Post-init-step bag card is at most init's bag card. -/
theorem init_step_bag_card_le_init_bag_card {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.card ≤ init.bag.card :=
  le_of_lt (init_step_bag_card_lt_init_bag_card pl)

/-- Post-init-step bag card is positive. -/
theorem init_step_bag_card_pos {cfg : GameConfig} (pl : Placement) :
    0 < (init.step cfg pl).bag.card := by
  rw [init_step_bag_card]; decide

/-- Post-init-step bag is nonempty. -/
theorem init_step_bag_nonempty {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag.Nonempty :=
  Finset.card_pos.mp (init_step_bag_card_pos pl)

/-- Post-init-step bag is a subset of `Bag.full`. -/
theorem init_step_bag_subset_full {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag ⊆ Bag.full :=
  Bag.subset_full _

/-- Post-init-step bag is a strict subset of `Bag.full`. -/
theorem init_step_bag_ssubset_full {cfg : GameConfig} (pl : Placement) :
    (init.step cfg pl).bag ⊂ Bag.full :=
  Finset.ssubset_iff_subset_ne.mpr
    ⟨init_step_bag_subset_full pl, init_step_bag_ne_full cfg pl⟩

/-- **Distinct pieces yield distinct init successors**: if `pl1.piece
≠ pl2.piece`, the corresponding one-step states differ (their bags
differ by the erased piece). -/
theorem init_step_inj_on_piece {cfg : GameConfig} {pl1 pl2 : Placement}
    (h : pl1.piece ≠ pl2.piece) :
    init.step cfg pl1 ≠ init.step cfg pl2 := by
  intro heq
  have hbag : (init.step cfg pl1).bag = (init.step cfg pl2).bag :=
    congrArg GameState.bag heq
  rw [init_step_bag, init_step_bag] at hbag
  -- pl1.piece ∈ erase pl2.piece (since pl1.piece ≠ pl2.piece and ∈ full)
  have hmem : pl1.piece ∈ Bag.full.erase pl2.piece :=
    Finset.mem_erase.mpr ⟨h, Bag.mem_full pl1.piece⟩
  rw [← hbag] at hmem
  exact (Finset.notMem_erase _ _) hmem

/-- **Step from any state with a drawable piece is never identity**.
The post-step bag is `g.bag.draw pl.piece`, which by
`Bag.draw_ne_self` differs from `g.bag`. -/
theorem step_ne_self {cfg : GameConfig} {g : GameState} {pl : Placement}
    (hp : pl.piece ∈ g.bag) : g.step cfg pl ≠ g := by
  intro heq
  have hbag : (g.step cfg pl).bag = g.bag := congrArg GameState.bag heq
  change g.bag.draw pl.piece = g.bag at hbag
  exact Bag.draw_ne_self hp hbag

/-- **General step-injectivity over the bag**: from any state `g`,
two placements playing distinct pieces both in `g.bag` yield distinct
one-step states (their bags differ by `Bag.draw_injOn`). -/
theorem step_inj_on_bag {cfg : GameConfig} {g : GameState}
    {pl1 pl2 : Placement} (hp1 : pl1.piece ∈ g.bag)
    (hp2 : pl2.piece ∈ g.bag) (h : pl1.piece ≠ pl2.piece) :
    g.step cfg pl1 ≠ g.step cfg pl2 := by
  intro heq
  have hbag : (g.step cfg pl1).bag = (g.step cfg pl2).bag :=
    congrArg GameState.bag heq
  change g.bag.draw pl1.piece = g.bag.draw pl2.piece at hbag
  exact h (Bag.draw_injOn g.bag hp1 hp2 hbag)

/-- **Generalised step-image cardinality**: from any state, for any
piece-indexed placement family `f` with `(f p).piece = p`, the image
of `g.step cfg ∘ f` over `g.bag` has cardinality `|g.bag|`. -/
theorem step_image_card {cfg : GameConfig} (g : GameState)
    (f : Piece → Placement) (hf : ∀ p, (f p).piece = p) :
    (g.bag.image (fun p => g.step cfg (f p))).card = g.bag.card := by
  apply Finset.card_image_of_injOn
  intro p hp q hq heq
  by_contra hne
  refine step_inj_on_bag (cfg := cfg) (pl1 := f p) (pl2 := f q) ?_ ?_ ?_ heq
  · rw [hf p]; exact hp
  · rw [hf q]; exact hq
  · rw [hf p, hf q]; exact hne

/-- **7 distinct one-step successors of init**: for any
piece-indexed placement family `f` whose result's piece matches its
input piece (`(f p).piece = p`), the image of `init.step cfg ∘ f`
over all 7 pieces has cardinality 7. -/
theorem init_step_image_card {cfg : GameConfig}
    (f : Piece → Placement) (hf : ∀ p, (f p).piece = p) :
    ((Finset.univ : Finset Piece).image (fun p => init.step cfg (f p))).card
    = 7 := by
  rw [Finset.card_image_of_injOn]
  · exact Piece.card_eq_seven
  · intro p _ q _ heq
    by_contra hne
    apply init_step_inj_on_piece (cfg := cfg) (pl1 := f p) (pl2 := f q) ?_ heq
    rw [hf p, hf q]; exact hne

end GameState

/-- States reachable from `init` by legal moves: the drawn piece must be in the
bag, the placement must be in-bounds, and the prior state must not be lost. -/
inductive Reachable (cfg : GameConfig) : GameState → Prop
  | init : Reachable cfg GameState.init
  | step {g : GameState} {pl : Placement} :
      Reachable cfg g → g.bag.canDraw pl.piece → pl.Valid cfg → ¬ g.lost cfg →
      Reachable cfg (g.step cfg pl)

/-- **Reachability preserves WF.** Every state reachable from `init` has a
well-formed board: `init.board = ∅` is WF (vacuously), and each `step`
applies a `Valid` placement, which preserves WF by `applyStep_wf`. -/
theorem Reachable.wf {cfg : GameConfig} {g : GameState} (h : Reachable cfg g) :
    Board.WF cfg g.board := by
  induction h with
  | init => exact GameState.init_board_wf cfg
  | step _hreach _hpb hvalid _hlost ih =>
    exact Placement.applyStep_wf ih hvalid

/-- Every reachable state has a nonempty bag. -/
theorem Reachable.bag_nonempty {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag.Nonempty := by
  induction h with
  | init => exact Bag.full_nonempty
  | step _ _ _ _ _ => exact Bag.draw_nonempty _ _

/-- Every reachable state has a drawable piece. -/
theorem Reachable.exists_canDraw {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : ∃ p : Piece, g.bag.canDraw p :=
  Bag.exists_canDraw_of_nonempty h.bag_nonempty

/-- The `someElement` of a reachable state's bag is drawable. -/
noncomputable def Reachable.someBagElement {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : Piece := g.bag.someElement h.bag_nonempty

theorem Reachable.someBagElement_canDraw {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag.canDraw h.someBagElement :=
  Bag.someElement_canDraw _ h.bag_nonempty

/-- `someBagElement` is in the bag (∈ form). -/
theorem Reachable.someBagElement_mem {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : h.someBagElement ∈ g.bag :=
  Bag.someElement_mem _ h.bag_nonempty

/-- Every reachable state's bag has at least 1 element and at most 7. -/
theorem Reachable.bag_card_pos {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : 0 < g.bag.card :=
  Finset.card_pos.mpr h.bag_nonempty

theorem Reachable.bag_card_le_seven {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.bag.card ≤ 7 := Bag.card_le_seven _

/-- Bundled `[1, 7]` bag-card interval for any reachable state. -/
theorem Reachable.bag_card_interval {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : 1 ≤ g.bag.card ∧ g.bag.card ≤ 7 :=
  ⟨h.bag_card_pos, h.bag_card_le_seven⟩

/-- Reachable state's bag-card is never zero. -/
theorem Reachable.bag_card_ne_zero {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag.card ≠ 0 :=
  Nat.pos_iff_ne_zero.mp h.bag_card_pos

/-- Reachable state's bag-card is ≥ 1. -/
theorem Reachable.bag_card_ge_one {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : 1 ≤ g.bag.card := h.bag_card_pos

/-- Reachable state's bag is never empty. -/
theorem Reachable.bag_ne_empty {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag ≠ ∅ :=
  Finset.Nonempty.ne_empty h.bag_nonempty

/-- Reachable state's bag-empty proposition is `False`. -/
theorem Reachable.bag_eq_empty_iff_false {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag = ∅ ↔ False :=
  ⟨fun hempty => h.bag_ne_empty hempty, fun f => f.elim⟩

/-- Reachable state's bag does not equal the empty-bag of an
empty-bag-empty-board state — pairs with Tick 267 for non-membership. -/
theorem Reachable.ne_empty_bag_empty_board {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g ≠ ⟨Board.empty, (∅ : Bag)⟩ := by
  intro heq
  have : g.bag = ∅ := congrArg GameState.bag heq
  exact h.bag_ne_empty this

/-- Reachable state's bag is a subset of `Bag.full`. -/
theorem Reachable.bag_subset_full {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.bag ⊆ Bag.full :=
  GameState.bag_subset_full g


/-- Any state's bag is either the full bag or has card strictly < 7. -/
theorem GameState.bag_eq_full_or_card_lt_seven (g : GameState) :
    g.bag = Bag.full ∨ g.bag.card < 7 := by
  rcases (Bag.card_le_seven g.bag).lt_or_eq with h | h
  · right; exact h
  · left; exact (g.bag_card_eq_seven_iff_eq_full).mp h

/-- Any state's bag is either the full bag or has card `≤ 6`. -/
theorem GameState.bag_eq_full_or_card_le_six (g : GameState) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 :=
  Bag.eq_full_or_card_le_six g.bag

/-- `g = init ⇒ g.bag = Bag.full`. -/
theorem GameState.bag_eq_full_of_eq_init {g : GameState} (h : g = init) :
    g.bag = Bag.full := h ▸ init_bag

/-- `g = init ⇒ g.board = Board.empty`. -/
theorem GameState.board_eq_empty_of_eq_init {g : GameState} (h : g = init) :
    g.board = Board.empty := h ▸ init_board

/-- A state equals `init` iff its board is empty and bag is full. -/
theorem GameState.eq_init_iff_board_empty_and_bag_full (g : GameState) :
    g = init ↔ g.board = Board.empty ∧ g.bag = Bag.full := by
  refine ⟨fun h => ⟨board_eq_empty_of_eq_init h, bag_eq_full_of_eq_init h⟩, ?_⟩
  rintro ⟨hb, hbag⟩
  rcases g with ⟨board, bag⟩
  simp only at hb hbag
  rw [hb, hbag]
  rfl

/-- Implication form of Tick 466 (mpr direction). -/
theorem GameState.eq_init_of_board_empty_of_bag_full (g : GameState)
    (hb : g.board = Board.empty) (hbag : g.bag = Bag.full) : g = init :=
  (eq_init_iff_board_empty_and_bag_full g).mpr ⟨hb, hbag⟩

/-- A state equals `init` iff its board count is zero and bag is full. -/
theorem GameState.eq_init_iff_count_zero_and_bag_full (g : GameState) :
    g = init ↔ g.board.count = 0 ∧ g.bag = Bag.full := by
  rw [eq_init_iff_board_empty_and_bag_full]
  refine ⟨fun ⟨hb, hbag⟩ => ⟨by rw [hb]; rfl, hbag⟩, ?_⟩
  rintro ⟨hc, hbag⟩
  refine ⟨?_, hbag⟩
  change g.board = (∅ : Finset _)
  exact Finset.card_eq_zero.mp hc

/-- Contrapositive: `g.bag ≠ full ⇒ g ≠ init`. -/
theorem GameState.ne_init_of_bag_ne_full {g : GameState}
    (h : g.bag ≠ Bag.full) : g ≠ init := fun heq =>
  h (bag_eq_full_of_eq_init heq)

/-- Contrapositive: `g.board ≠ empty ⇒ g ≠ init`. -/
theorem GameState.ne_init_of_board_ne_empty {g : GameState}
    (h : g.board ≠ Board.empty) : g ≠ init := fun heq =>
  h (board_eq_empty_of_eq_init heq)

/-- `g.bag.card < 7 ⇒ g ≠ init` (since init has 7-card bag). -/
theorem GameState.ne_init_of_bag_card_lt_seven {g : GameState}
    (h : g.bag.card < 7) : g ≠ init := by
  intro heq
  rw [heq, init_bag_card] at h
  omega

/-- `g ≠ init ↔ board differs OR bag differs`. -/
theorem GameState.ne_init_iff (g : GameState) :
    g ≠ init ↔ g.board ≠ Board.empty ∨ g.bag ≠ Bag.full := by
  rw [Ne, eq_init_iff_board_empty_and_bag_full, not_and_or]

/-- Dot-notation accessor of the bag dichotomy for `Reachable`. -/
theorem Reachable.bag_eq_full_or_card_lt_seven {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g.bag = Bag.full ∨ g.bag.card < 7 :=
  g.bag_eq_full_or_card_lt_seven

/-- Dot-notation accessor of the bag dichotomy for `Reachable` in
`≤ 6` form. -/
theorem Reachable.bag_eq_full_or_card_le_six {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 :=
  g.bag_eq_full_or_card_le_six

/-- Reachable state's bag is either strictly inside `Bag.full` or
equals it. -/
theorem Reachable.bag_ssubset_full_or_eq_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag ⊂ Bag.full ∨ g.bag = Bag.full := by
  rcases h.bag_eq_full_or_card_lt_seven with hf | hlt
  · right; exact hf
  · left
    rw [Finset.ssubset_iff_subset_ne]
    refine ⟨GameState.bag_subset_full _, ?_⟩
    intro heq
    rw [heq, Bag.full_card] at hlt
    omega

/-- Dot-notation accessor: `bag.card = 7 ↔ bag = full`. -/
theorem Reachable.bag_card_eq_seven_iff_bag_eq_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g.bag.card = 7 ↔ g.bag = Bag.full :=
  Bag.card_eq_seven_iff_eq_full g.bag

/-- For reachable states with non-full bag, some piece is missing. -/
theorem Reachable.exists_notMem_of_bag_ne_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (hne : g.bag ≠ Bag.full) :
    ∃ p : Piece, p ∉ g.bag :=
  Bag.exists_notMem_of_ne_full hne


/-- For reachable states, `bag.card ≠ 7 ↔ bag ≠ Bag.full`. -/
theorem Reachable.bag_card_ne_seven_iff_bag_ne_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag.card ≠ 7 ↔ g.bag ≠ Bag.full :=
  not_congr (Reachable.bag_card_eq_seven_iff_bag_eq_full h)

/-- For reachable states, `bag ⊂ Bag.full ↔ bag ≠ Bag.full`. -/
theorem Reachable.bag_ssubset_full_iff_bag_ne_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag ⊂ Bag.full ↔ g.bag ≠ Bag.full := by
  rw [Finset.ssubset_iff_subset_ne]
  exact ⟨fun ⟨_, hne⟩ => hne, fun hne => ⟨h.bag_subset_full, hne⟩⟩

/-- Trichotomy-free split for reachable bags: full or strictly smaller. -/
theorem Reachable.bag_eq_full_or_ssubset {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag = Bag.full ∨ g.bag ⊂ Bag.full := by
  by_cases heq : g.bag = Bag.full
  · exact Or.inl heq
  · exact Or.inr ((Reachable.bag_ssubset_full_iff_bag_ne_full h).mpr heq)

/-- For reachable states, the bag card is exactly 7 or strictly less. -/
theorem Reachable.bag_card_eq_seven_or_lt {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag.card = 7 ∨ g.bag.card < 7 := by
  have : g.bag.card ≤ 7 := h.bag_card_le_seven
  omega

/-- For reachable states, the bag card is 7 or `≤ 6`. -/
theorem Reachable.bag_card_eq_seven_or_le_six {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag.card = 7 ∨ g.bag.card ≤ 6 := by
  have : g.bag.card ≤ 7 := h.bag_card_le_seven
  omega

/-- For reachable states, `bag.card < 7 ↔ bag ≠ Bag.full`. -/
theorem Reachable.bag_card_lt_seven_iff_bag_ne_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag.card < 7 ↔ g.bag ≠ Bag.full := by
  have hle : g.bag.card ≤ 7 := h.bag_card_le_seven
  constructor
  · intro hlt heq
    have : g.bag.card = 7 := by rw [heq]; exact Bag.full_card
    omega
  · intro hne
    have : g.bag.card ≠ 7 :=
      (Reachable.bag_card_ne_seven_iff_bag_ne_full h).mpr hne
    omega

/-- For reachable states, `bag.card ≤ 6 ↔ bag ≠ Bag.full`. -/
theorem Reachable.bag_card_le_six_iff_bag_ne_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag.card ≤ 6 ↔ g.bag ≠ Bag.full := by
  constructor
  · intro hle heq
    have : g.bag.card = 7 := by rw [heq]; exact Bag.full_card
    omega
  · intro hne
    have hle7 : g.bag.card ≤ 7 := h.bag_card_le_seven
    have : g.bag.card ≠ 7 := fun heq =>
      hne ((Reachable.bag_card_eq_seven_iff_bag_eq_full h).mp heq)
    omega

/-- Reachable state's bag-draw is either smaller or refills to 7. -/
theorem Reachable.bag_draw_card_le_or_eq_seven {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (p : Piece) :
    (g.bag.draw p).card ≤ g.bag.card ∨ (g.bag.draw p).card = 7 :=
  Bag.draw_card_le_card_or_eq_seven g.bag p

/-- `Reachable` accessor: step's bag card ≤ pre-step or refills to 7. -/
theorem Reachable.step_bag_card_le_or_eq_seven {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag.card ≤ g.bag.card ∨
      (g.step cfg pl).bag.card = 7 :=
  GameState.step_bag_card_le_or_eq_seven cfg g pl

/-- `Reachable` accessor: step's bag card interval `[1, 7]`. -/
theorem Reachable.step_bag_card_interval {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (pl : Placement) :
    1 ≤ (g.step cfg pl).bag.card ∧ (g.step cfg pl).bag.card ≤ 7 :=
  ⟨Bag.draw_card_ge_one g.bag pl.piece, Bag.draw_card_le_seven g.bag pl.piece⟩

/-- `Reachable` accessor: step's bag is a subset of `Bag.full`. -/
theorem Reachable.step_bag_subset_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag ⊆ Bag.full :=
  Bag.draw_subset_full g.bag pl.piece

/-- Strict `< 8` form for any reachable state's bag card. -/
theorem Reachable.bag_card_lt_eight {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : g.bag.card < 8 := by
  have := h.bag_card_le_seven
  omega

/-- Successor's bag card is also bounded by 7 (true for any
state, not just reachable). -/
theorem Reachable.step_bag_card_le_seven {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag.card ≤ 7 :=
  Bag.card_le_seven _

/-- Successor's bag card strictly less than 8. -/
theorem Reachable.step_bag_card_lt_eight {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag.card < 8 := by
  have := Bag.card_le_seven (g.step cfg pl).bag
  omega

/-- Successor's bag is nonempty (draw refills). -/
theorem Reachable.step_bag_nonempty {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag.Nonempty := by
  rw [GameState.step_bag]; exact Bag.draw_nonempty g.bag pl.piece


/-- Successor's bag card is positive. -/
theorem Reachable.step_bag_card_pos {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (pl : Placement) :
    0 < (g.step cfg pl).bag.card :=
  Finset.card_pos.mpr (h.step_bag_nonempty pl)

/-- Successor's bag card is at least 1. -/
theorem Reachable.step_bag_card_ge_one {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (pl : Placement) :
    1 ≤ (g.step cfg pl).bag.card :=
  h.step_bag_card_pos pl

/-- Successor's bag is nonempty (≠ ∅ form). -/
theorem Reachable.step_bag_ne_empty {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag ≠ ∅ :=
  (h.step_bag_nonempty pl).ne_empty

/-- Successor's bag.card is nonzero. -/
theorem Reachable.step_bag_card_ne_zero {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag.card ≠ 0 :=
  Nat.ne_of_gt (h.step_bag_card_pos pl)

/-- **Bundled `Reachable` invariants**: at any reachable state,
the board is WF, the bag is non-empty, the bag is a subset of
`Bag.full`, and `1 ≤ bag.card ≤ 7`. -/
theorem Reachable.full_invariants {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    Board.WF cfg g.board ∧ g.bag.Nonempty ∧
    g.bag ⊆ Bag.full ∧ 1 ≤ g.bag.card ∧ g.bag.card ≤ 7 :=
  ⟨h.wf, h.bag_nonempty, h.bag_subset_full,
   h.bag_card_pos, h.bag_card_le_seven⟩

/-- **One-step reachability from init**: any valid placement at
init gives a reachable state. Both canDraw (init has full bag)
and ¬ lost (empty board is never lost) are auto-discharged. -/
theorem Reachable.init_step {cfg : GameConfig} {pl : Placement}
    (hv : pl.Valid cfg) : Reachable cfg (GameState.init.step cfg pl) :=
  Reachable.init.step (GameState.init_canDraw pl.piece) hv
    (GameState.init_not_lost cfg)

/-- `init` invariants at the trivial state: board empty, bag full,
not lost, bag.card = 7. -/
theorem GameState.init_invariants (cfg : GameConfig) :
    GameState.init.board = ∅ ∧ GameState.init.bag = Bag.full ∧
    ¬ GameState.init.lost cfg ∧ GameState.init.bag.card = 7 :=
  ⟨GameState.init_board, GameState.init_bag,
   GameState.init_not_lost cfg, by
     rw [GameState.init_bag]; exact Bag.full_card⟩

/-- `init` is never `lost` for any config. -/
@[simp] theorem GameState.init_lost_iff_false (cfg : GameConfig) :
    GameState.init.lost cfg ↔ False :=
  ⟨GameState.init_not_lost cfg, False.elim⟩

/-- A lost state is never the initial state. -/
theorem GameState.ne_init_of_lost {cfg : GameConfig} {g : GameState}
    (h : g.lost cfg) : g ≠ GameState.init := by
  intro heq
  exact GameState.init_not_lost cfg (heq ▸ h)

/-- A reachable lost state cannot be the initial state. -/
theorem Reachable.ne_init_of_lost {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) (hl : g.lost cfg) : g ≠ GameState.init :=
  GameState.ne_init_of_lost hl

/-- `init.bag.card` is positive (= 7 > 0). -/
theorem GameState.init_bag_card_pos : 0 < GameState.init.bag.card := by
  rw [GameState.init_bag_card]; decide

/-- `init.bag.card ≠ 0`. -/
theorem GameState.init_bag_card_ne_zero : GameState.init.bag.card ≠ 0 :=
  Nat.ne_of_gt GameState.init_bag_card_pos

/-- `init.bag.card ≥ 1`. -/
theorem GameState.init_bag_card_ge_one : 1 ≤ GameState.init.bag.card :=
  GameState.init_bag_card_pos

/-- Every column of init's board has height 0. -/
@[simp] theorem GameState.init_colHeight (j : ℕ) :
    GameState.init.board.colHeight j = 0 := by
  rw [GameState.init_board]; exact Board.colHeight_empty j

/-- The init state is `low_stack` for any config — every column has
height 0, which is `≤ rows - 4` whenever `rows ≥ 4` and trivially
satisfies any nonneg threshold. -/
theorem GameState.init_low_stack_zero (j : ℕ) :
    GameState.init.board.colHeight j ≤ 0 := by
  rw [GameState.init_colHeight]



/-- **Init self-reachability**: `Reachable cfg init` is trivially
true. Convenience accessor avoiding `Reachable.init` indirection. -/
theorem Reachable.of_init {cfg : GameConfig} : Reachable cfg GameState.init :=
  Reachable.init

/-- `g` equals `init` iff its bag is full and board is empty
(reachable-agnostic, but useful for atlas search). -/
theorem Reachable.eq_init_of_bag_full_of_count_zero {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g)
    (hc : g.board.count = 0) (hb : g.bag = Bag.full) : g = GameState.init :=
  (GameState.eq_init_iff_count_zero_and_bag_full g).mpr ⟨hc, hb⟩

/-- Reachable-side iff: `g = init ↔ count = 0 ∧ bag = full`. -/
theorem Reachable.eq_init_iff_count_zero_and_bag_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g = GameState.init ↔ g.board.count = 0 ∧ g.bag = Bag.full :=
  GameState.eq_init_iff_count_zero_and_bag_full g

/-- Reachable-side iff: `g = init ↔ board = empty ∧ bag = full`. -/
theorem Reachable.eq_init_iff_board_empty_and_bag_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g = GameState.init ↔ g.board = Board.empty ∧ g.bag = Bag.full :=
  GameState.eq_init_iff_board_empty_and_bag_full g

/-- Reachable forward direction: `board = empty ∧ bag = full → g = init`. -/
theorem Reachable.eq_init_of_board_empty_of_bag_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g)
    (hb : g.board = Board.empty) (hbag : g.bag = Bag.full) :
    g = GameState.init :=
  GameState.eq_init_of_board_empty_of_bag_full g hb hbag

/-- For a reachable state, if the bag is full but `g ≠ init`, the board
must have positive count. -/
theorem Reachable.board_count_pos_of_bag_full_of_ne_init {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g)
    (hbag : g.bag = Bag.full) (hne : g ≠ GameState.init) :
    0 < g.board.count := by
  by_contra hpos
  push Not at hpos
  have hzero : g.board.count = 0 := Nat.le_zero.mp hpos
  exact hne (h.eq_init_iff_count_zero_and_bag_full.mpr ⟨hzero, hbag⟩)

/-- For a reachable state, if the bag is full but `g ≠ init`, the board
is nonempty. -/
theorem Reachable.board_nonempty_of_bag_full_of_ne_init {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g)
    (hbag : g.bag = Bag.full) (hne : g ≠ GameState.init) :
    g.board.Nonempty :=
  Finset.card_pos.mp (h.board_count_pos_of_bag_full_of_ne_init hbag hne)

/-- For a reachable state, if the bag is full but `g ≠ init`, the board
is not empty. -/
theorem Reachable.board_ne_empty_of_bag_full_of_ne_init {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g)
    (hbag : g.bag = Bag.full) (hne : g ≠ GameState.init) :
    g.board ≠ Board.empty :=
  (h.board_nonempty_of_bag_full_of_ne_init hbag hne).ne_empty

/-- Every reachable state is either `init` or comes from a valid step
applied to another reachable, non-lost, drawable state. -/
theorem Reachable.eq_init_or_exists_step {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    g = GameState.init ∨
    ∃ g' : GameState, ∃ pl : Placement,
      Reachable cfg g' ∧ g'.bag.canDraw pl.piece ∧ pl.Valid cfg ∧ ¬ g'.lost cfg ∧
        g = g'.step cfg pl := by
  cases h with
  | init => exact Or.inl rfl
  | @step g' pl hr hpb hvalid hnl => exact Or.inr ⟨g', pl, hr, hpb, hvalid, hnl, rfl⟩

/-- A reachable non-init state has a valid-step predecessor. -/
theorem Reachable.exists_step_of_ne_init {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) (hne : g ≠ GameState.init) :
    ∃ g' : GameState, ∃ pl : Placement,
      Reachable cfg g' ∧ g'.bag.canDraw pl.piece ∧ pl.Valid cfg ∧ ¬ g'.lost cfg ∧
        g = g'.step cfg pl :=
  (h.eq_init_or_exists_step).resolve_left hne

/-- A reachable non-init state has a reachable predecessor (forgetting the
placement). -/
theorem Reachable.exists_reachable_predecessor_of_ne_init {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g ≠ GameState.init) :
    ∃ g' : GameState, Reachable cfg g' ∧ ¬ g'.lost cfg := by
  obtain ⟨g', _, hr, _, _, hnl, _⟩ := h.exists_step_of_ne_init hne
  exact ⟨g', hr, hnl⟩

/-- Minimal form: a reachable non-init state has a reachable predecessor. -/
theorem Reachable.exists_reachable_pred_of_ne_init {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g ≠ GameState.init) :
    ∃ g' : GameState, Reachable cfg g' := by
  obtain ⟨g', hr, _⟩ := h.exists_reachable_predecessor_of_ne_init hne
  exact ⟨g', hr⟩

/-- For any reachable `g`, either `g` is lost or it is not (decidable LEM). -/
theorem Reachable.lost_or_not_lost {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.lost cfg ∨ ¬ g.lost cfg :=
  Decidable.em _

/-- For any reachable `g`, either `g = init` or `g ≠ init`. -/
theorem Reachable.eq_init_or_ne_init {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g = GameState.init ∨ g ≠ GameState.init :=
  Decidable.em _

/-- For any reachable `g`, either `bag = full` or `bag ≠ full`. -/
theorem Reachable.bag_eq_full_or_ne_full {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.bag = Bag.full ∨ g.bag ≠ Bag.full :=
  Decidable.em _

/-- For any reachable `g`, either `board = empty` or `board ≠ empty`. -/
theorem Reachable.board_eq_empty_or_ne_empty {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.board = Board.empty ∨ g.board ≠ Board.empty :=
  Decidable.em _

/-- For any reachable `g`, the board count is either zero or positive. -/
theorem Reachable.board_count_zero_or_pos {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) : g.board.count = 0 ∨ 0 < g.board.count :=
  Nat.eq_zero_or_pos _

/-- Reachable-side iff: `g ≠ init ↔ board ≠ empty ∨ bag ≠ full`. -/
theorem Reachable.ne_init_iff {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) :
    g ≠ GameState.init ↔ g.board ≠ Board.empty ∨ g.bag ≠ Bag.full :=
  GameState.ne_init_iff g

/-- Reachable-side iff: `g ≠ init ↔ 0 < count ∨ bag ≠ full` (numeric form). -/
theorem Reachable.ne_init_iff_count_pos_or_bag_ne_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g ≠ GameState.init ↔ 0 < g.board.count ∨ g.bag ≠ Bag.full := by
  rw [Ne, h.eq_init_iff_count_zero_and_bag_full, not_and_or]
  constructor
  · rintro (hl | hr)
    · exact Or.inl (Nat.pos_of_ne_zero hl)
    · exact Or.inr hr
  · rintro (hl | hr)
    · exact Or.inl (Nat.ne_of_gt hl)
    · exact Or.inr hr

/-- Reachable-side iff: `g ≠ init ↔ 0 < count ∨ bag.card < 7`. -/
theorem Reachable.ne_init_iff_count_pos_or_bag_card_lt_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g ≠ GameState.init ↔ 0 < g.board.count ∨ g.bag.card < 7 := by
  rw [h.ne_init_iff_count_pos_or_bag_ne_full,
    ← h.bag_card_lt_seven_iff_bag_ne_full]

/-- Reachable-side iff: `g = init ↔ count = 0 ∧ bag.card = 7`. -/
theorem Reachable.eq_init_iff_count_zero_and_bag_card_eq_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g = GameState.init ↔ g.board.count = 0 ∧ g.bag.card = 7 := by
  rw [h.eq_init_iff_count_zero_and_bag_full,
    ← h.bag_card_eq_seven_iff_bag_eq_full]

/-- Reachable-side iff: `g ≠ init ↔ count ≠ 0 ∨ bag.card ≠ 7`. -/
theorem Reachable.ne_init_iff_count_ne_zero_or_bag_card_ne_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g ≠ GameState.init ↔ g.board.count ≠ 0 ∨ g.bag.card ≠ 7 := by
  rw [Ne, h.eq_init_iff_count_zero_and_bag_card_eq_seven, not_and_or]

/-- A reachable state with `bag.card = 7` has `bag = Bag.full`. -/
theorem Reachable.bag_eq_full_of_bag_card_eq_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (h7 : g.bag.card = 7) :
    g.bag = Bag.full :=
  (h.bag_card_eq_seven_iff_bag_eq_full).mp h7

/-- A reachable state's bag-card is exactly 7 (full) or strictly less. -/
theorem Reachable.bag_eq_full_or_bag_card_lt_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag = Bag.full ∨ g.bag.card < 7 := by
  rcases h.bag_card_eq_seven_or_lt with h7 | hlt
  · exact Or.inl (h.bag_eq_full_of_bag_card_eq_seven h7)
  · exact Or.inr hlt

/-- A reachable state's bag is full or has card `≤ 6`. -/
theorem Reachable.bag_eq_full_or_bag_card_le_six {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) :
    g.bag = Bag.full ∨ g.bag.card ≤ 6 := by
  rcases h.bag_eq_full_or_bag_card_lt_seven with hf | hlt
  · exact Or.inl hf
  · exact Or.inr (by omega)

/-- Reachable-side dot form of `step_bag_eq_erase_or_full`. -/
theorem Reachable.step_bag_eq_erase_or_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag = g.bag.erase pl.piece ∨
      (g.step cfg pl).bag = Bag.full :=
  GameState.step_bag_eq_erase_or_full cfg g pl

/-- Reachable-side dot form of `step_bag_subset_or_eq_full`. -/
theorem Reachable.step_bag_subset_or_eq_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).bag ⊆ g.bag ∨ (g.step cfg pl).bag = Bag.full :=
  GameState.step_bag_subset_or_eq_full cfg g pl

/-- A non-full bag implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_bag_ne_full {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (hne : g.bag ≠ Bag.full) :
    g ≠ GameState.init :=
  GameState.ne_init_of_bag_ne_full hne

/-- A non-empty board implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_board_ne_empty {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (hne : g.board ≠ Board.empty) :
    g ≠ GameState.init :=
  GameState.ne_init_of_board_ne_empty hne

/-- `bag.card < 7` implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_bag_card_lt_seven {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (hlt : g.bag.card < 7) :
    g ≠ GameState.init :=
  GameState.ne_init_of_bag_card_lt_seven hlt

/-- A positive board count implies `g ≠ init`. -/
theorem Reachable.ne_init_of_board_count_pos {cfg : GameConfig}
    {g : GameState} (_h : Reachable cfg g) (hpos : 0 < g.board.count) :
    g ≠ GameState.init := by
  intro heq
  rw [heq, GameState.init_board_count] at hpos
  exact Nat.lt_irrefl 0 hpos

/-- A nonzero board count implies `g ≠ init`. -/
theorem Reachable.ne_init_of_board_count_ne_zero {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g.board.count ≠ 0) :
    g ≠ GameState.init :=
  h.ne_init_of_board_count_pos (Nat.pos_of_ne_zero hne)

/-- A nonempty board implies `g ≠ init`. -/
theorem Reachable.ne_init_of_board_nonempty {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g.board.Nonempty) :
    g ≠ GameState.init :=
  h.ne_init_of_board_count_pos (Finset.card_pos.mpr hne)

/-- `bag.card ≤ 6` implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_bag_card_le_six {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hle : g.bag.card ≤ 6) :
    g ≠ GameState.init :=
  h.ne_init_of_bag_card_lt_seven (Nat.lt_of_le_of_lt hle (by decide))

/-- `bag ⊂ Bag.full` implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_bag_ssubset_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hss : g.bag ⊂ Bag.full) :
    g ≠ GameState.init :=
  h.ne_init_of_bag_ne_full (ne_of_lt hss)

/-- `bag.card ≠ 7` implies `g ≠ init` (dot form). -/
theorem Reachable.ne_init_of_bag_card_ne_seven {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g.bag.card ≠ 7) :
    g ≠ GameState.init :=
  h.ne_init_of_bag_ne_full
    ((Reachable.bag_card_ne_seven_iff_bag_ne_full h).mp hne)


end Tetris
