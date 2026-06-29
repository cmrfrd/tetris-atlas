import Mathlib
import Proofs.Invariants.GameplayExtra
import Proofs.Model.PieceGeometry

/-!
# Universal safety / loss obstructions

* `isLost_of_count_gt` — pigeonhole: more cells than the in-field capacity ⇒ lost.
* `low_stack_safe` — universal local safety: low stack + any valid placement ⇒
  the resulting board is not lost.
* `exists_safe_placement` — universal progress: from a low stack, every piece
  admits a valid placement keeping the board not lost.

These are the *universal* safety facts every survivability argument relies on.
-/

namespace Tetris

/-! ## Pigeonhole loss obstruction -/

/-- **Pigeonhole obstruction:** a well-formed board with more cells than the
visible field can hold must be lost. -/
theorem isLost_of_count_gt {cfg : GameConfig} {b : Board}
    (hWF : Board.WF cfg b) (hgt : cfg.cols * cfg.rows < b.count) :
    Board.isLost cfg b := by
  by_contra hnot
  have hbounded : ∀ p ∈ b, p.2 < cfg.rows := by
    intro p hp
    by_contra hge
    exact hnot ⟨p, hp, Nat.not_lt.mp hge⟩
  have hsub : b ⊆ Finset.range cfg.cols ×ˢ Finset.range cfg.rows := by
    intro p hp
    rw [Finset.mem_product]
    exact ⟨Finset.mem_range.2 (hWF p hp), Finset.mem_range.2 (hbounded p hp)⟩
  have h1 : b.card ≤ (Finset.range cfg.cols ×ˢ Finset.range cfg.rows).card :=
    Finset.card_le_card hsub
  simp only [Finset.card_product, Finset.card_range] at h1
  -- h1 : b.card ≤ cfg.cols * cfg.rows
  have : b.count ≤ cfg.cols * cfg.rows := h1
  omega

/-! ## Safety from a low stack -/

/-- **Universal local safety:** on a well-formed board with every column at
height ≤ `rows - 4`, any valid placement keeps the board not lost. The piece
(≤ 4 tall) fits within the field and line clearing only lowers heights. -/
theorem low_stack_safe {cfg : GameConfig} (hrows : 4 ≤ cfg.rows) {b : Board}
    (_hWF : Board.WF cfg b)
    (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4)
    (pl : Placement) (_hv : pl.Valid cfg) :
    ¬ Board.isLost cfg (Placement.applyStep cfg b pl) := by
  -- Step 1: dropOffset is bounded by `rows - 4`.
  have hd : pl.dropOffset b ≤ cfg.rows - 4 := by
    change pl.shapeUp.sup (fun cell => Board.colHeight b (pl.col + cell.1) - cell.2)
        ≤ cfg.rows - 4
    apply Finset.sup_le
    intro cell _
    have := hlow (pl.col + cell.1)
    omega
  -- Step 2: every cell of `pl.place b` is below `rows`.
  have hplace : ∀ q ∈ pl.place b, q.2 < cfg.rows := by
    intro q hq
    simp only [Placement.place, Finset.mem_union] at hq
    rcases hq with h | h
    · have hlt := Board.lt_colHeight h
      have hh := hlow q.1
      omega
    · simp only [Placement.dropped, Placement.cellsAt, Finset.mem_image] at h
      obtain ⟨cell, hcell, rfl⟩ := h
      have hdr := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
      -- (pl.col + cell.1, pl.dropOffset b + cell.2).2 = pl.dropOffset b + cell.2
      change pl.dropOffset b + cell.2 < cfg.rows
      omega
  -- Step 3: line clearing only lowers rows, so every cell of `applyStep` is below `rows`.
  have happly : ∀ q ∈ Placement.applyStep cfg b pl, q.2 < cfg.rows := by
    intro q hq
    unfold Placement.applyStep at hq
    simp only [Board.clearLines, Finset.mem_image, Finset.mem_filter] at hq
    obtain ⟨p, ⟨hpb, _⟩, hpe⟩ := hq
    rw [Prod.mk.injEq] at hpe
    have hpr := hplace p hpb
    have hq2 : q.2 = p.2 - Board.clearedBelow cfg (pl.place b) p.2 := hpe.2.symm
    omega
  exact Board.not_isLost_of_forall_lt happly

/-! ## Progress: every piece has a safe placement -/

/-- **Universal progress:** on a board with all column heights ≤ `rows - 4`,
every piece admits a valid placement keeping the board not lost. The column-0,
rotation-0 placement always fits (the piece is at most 4 wide and the board is
at least 4 wide), and `low_stack_safe` then handles safety. -/
theorem exists_safe_placement {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {b : Board} (hWF : Board.WF cfg b)
    (hlow : ∀ j, Board.colHeight b j ≤ cfg.rows - 4)
    (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg
      ∧ ¬ Board.isLost cfg (Placement.applyStep cfg b pl) := by
  have hvalid : (Placement.mk p 0 0).Valid cfg := by
    intro cell hcell
    have hc := Piece.shapeUp_col_lt_four p 0 cell hcell
    change 0 + cell.1 < cfg.cols
    omega
  exact ⟨⟨p, 0, 0⟩, rfl, hvalid, low_stack_safe hrows hWF hlow _ hvalid⟩

/-- `GameState.init` is low-stack (heights = 0 everywhere). -/
theorem init_low_stack {cfg : GameConfig} (_hrows : 4 ≤ cfg.rows) :
    ∀ j, Board.colHeight GameState.init.board j ≤ cfg.rows - 4 := by
  intro j
  rw [GameState.init_board]
  change Board.colHeight (∅ : Board) j ≤ cfg.rows - 4
  simp

/-- Standard specialization. -/
theorem init_low_stack_standard :
    ∀ j, Board.colHeight GameState.init.board j ≤ GameConfig.standard.rows - 4 :=
  init_low_stack (by decide)

/-- Tiny specialization. -/
theorem init_low_stack_tiny :
    ∀ j, Board.colHeight GameState.init.board j ≤ GameConfig.tiny.rows - 4 :=
  init_low_stack (by decide)

/-- `init` has count 0 ≤ cfg.cols * cfg.rows (vacuous bound). -/
theorem init_count_le_capacity (cfg : GameConfig) :
    GameState.init.board.count ≤ cfg.cols * cfg.rows := by
  rw [GameState.init_board_count]; exact Nat.zero_le _

/-- Specialisation of `init_count_le_capacity` to `standard`. -/
theorem init_count_le_capacity_standard :
    GameState.init.board.count ≤
      GameConfig.standard.cols * GameConfig.standard.rows :=
  init_count_le_capacity GameConfig.standard

/-- Specialisation of `init_count_le_capacity` to `tiny`. -/
theorem init_count_le_capacity_tiny :
    GameState.init.board.count ≤
      GameConfig.tiny.cols * GameConfig.tiny.rows :=
  init_count_le_capacity GameConfig.tiny

/-- For positive-dimension configs, `init.board.count < capacity`. -/
theorem init_count_lt_capacity {cfg : GameConfig}
    (hc : 0 < cfg.cols) (hr : 0 < cfg.rows) :
    GameState.init.board.count < cfg.cols * cfg.rows := by
  rw [GameState.init_board_count]
  exact Nat.mul_pos hc hr

/-- Specialisation of `init_count_lt_capacity` to `standard`. -/
theorem init_count_lt_capacity_standard :
    GameState.init.board.count <
      GameConfig.standard.cols * GameConfig.standard.rows :=
  init_count_lt_capacity (by decide) (by decide)

/-- Specialisation of `init_count_lt_capacity` to `tiny`. -/
theorem init_count_lt_capacity_tiny :
    GameState.init.board.count <
      GameConfig.tiny.cols * GameConfig.tiny.rows :=
  init_count_lt_capacity (by decide) (by decide)

/-- **Init has a safe placement for every piece.** Direct corollary of
`exists_safe_placement` + `init_low_stack`. From the empty starting board,
any piece can be placed (somewhere) without losing the game. -/
theorem init_exists_safe_placement {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg
      ∧ ¬ Board.isLost cfg (Placement.applyStep cfg GameState.init.board pl) := by
  apply exists_safe_placement hcols hrows
  · rw [GameState.init_board]; exact Board.empty_wf cfg
  · exact init_low_stack hrows

/-- Specialisation of `init_exists_safe_placement` to `standard`. -/
theorem standard_init_exists_safe_placement (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard
      ∧ ¬ Board.isLost GameConfig.standard
          (Placement.applyStep GameConfig.standard GameState.init.board pl) :=
  init_exists_safe_placement (by decide) (by decide) p

/-- Specialisation of `init_exists_safe_placement` to `tiny`. -/
theorem tiny_init_exists_safe_placement (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.tiny
      ∧ ¬ Board.isLost GameConfig.tiny
          (Placement.applyStep GameConfig.tiny GameState.init.board pl) :=
  init_exists_safe_placement (by decide) (by decide) p

/-- **One-step reachability from init.** For any valid placement whose
piece is in the (full) init bag, the post-step state is `Reachable cfg`.
Direct application of `Reachable.step` from `Reachable.init`. -/
theorem init_step_reachable {cfg : GameConfig} {pl : Placement}
    (hbag : GameState.init.bag.canDraw pl.piece) (hvalid : pl.Valid cfg) :
    Reachable cfg (GameState.init.step cfg pl) :=
  Reachable.step Reachable.init hbag hvalid (GameState.init_not_lost cfg)

/-- **Every piece admits a `Reachable` non-lost one-step successor from
init.** Combines `init_exists_safe_placement` (piece + valid + not-lost)
with `init_step_reachable`. The placement's piece comes from the full
init bag, so `Bag.canDraw` is automatic. -/
theorem init_exists_reachable_step {cfg : GameConfig}
    (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows) (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      Reachable cfg (GameState.init.step cfg pl) ∧
      ¬ Board.isLost cfg
        (Placement.applyStep cfg GameState.init.board pl) := by
  obtain ⟨pl, hpiece, hvalid, hnot_lost⟩ :=
    init_exists_safe_placement hcols hrows p
  refine ⟨pl, hpiece, hvalid, ?_, hnot_lost⟩
  refine init_step_reachable ?_ hvalid
  rw [GameState.init_bag, hpiece]
  exact Bag.mem_full p

/-- **Generalised reachable one-step extension.** For any reachable
non-lost low-stack state, every piece in the bag admits a valid
placement extending reachability *and* keeping the next state alive.
Generalisation of `init_exists_reachable_step` to arbitrary reachable
states. -/
theorem reachable_low_stack_exists_reachable_step
    {cfg : GameConfig} (hcols : 4 ≤ cfg.cols) (hrows : 4 ≤ cfg.rows)
    {g : GameState} (hr : Reachable cfg g) (hnot_lost : ¬ g.lost cfg)
    (hlow : ∀ j, Board.colHeight g.board j ≤ cfg.rows - 4)
    {p : Piece} (hp : p ∈ g.bag) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid cfg ∧
      Reachable cfg (g.step cfg pl) ∧
      ¬ Board.isLost cfg (Placement.applyStep cfg g.board pl) := by
  obtain ⟨pl, hpiece, hvalid, hnl⟩ :=
    exists_safe_placement hcols hrows hr.wf hlow p
  refine ⟨pl, hpiece, hvalid, ?_, hnl⟩
  refine Reachable.step hr ?_ hvalid hnot_lost
  rw [hpiece]; exact hp

/-- Specialisation of `init_exists_reachable_step` to `standard`. -/
theorem standard_init_exists_reachable_step (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard
      ∧ Reachable GameConfig.standard
          (GameState.init.step GameConfig.standard pl)
      ∧ ¬ Board.isLost GameConfig.standard
          (Placement.applyStep GameConfig.standard GameState.init.board pl) :=
  init_exists_reachable_step (by decide) (by decide) p

/-- Specialisation of `init_exists_reachable_step` to `tiny`. -/
theorem tiny_init_exists_reachable_step (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.tiny
      ∧ Reachable GameConfig.tiny
          (GameState.init.step GameConfig.tiny pl)
      ∧ ¬ Board.isLost GameConfig.tiny
          (Placement.applyStep GameConfig.tiny GameState.init.board pl) :=
  init_exists_reachable_step (by decide) (by decide) p

/-- Any valid placement on `init.board` (with `4 ≤ cfg.rows`) keeps the board
not lost. Direct application of `low_stack_safe` via Tick 1189's
`init_board_low_stack`. -/
theorem init_applyStep_not_lost_of_valid {cfg : GameConfig}
    (hrows : 4 ≤ cfg.rows) (pl : Placement) (hv : pl.Valid cfg) :
    ¬ Board.isLost cfg (Placement.applyStep cfg GameState.init.board pl) :=
  low_stack_safe hrows (GameState.init_board_wf cfg)
    (GameState.init_board_low_stack cfg) pl hv

/-- Standard config: any valid placement on init keeps the board safe. -/
theorem standard_init_applyStep_not_lost_of_valid (pl : Placement)
    (hv : pl.Valid GameConfig.standard) :
    ¬ Board.isLost GameConfig.standard
      (Placement.applyStep GameConfig.standard GameState.init.board pl) :=
  init_applyStep_not_lost_of_valid (by decide) pl hv

/-- Tiny config: any valid placement on init keeps the board safe. -/
theorem tiny_init_applyStep_not_lost_of_valid (pl : Placement)
    (hv : pl.Valid GameConfig.tiny) :
    ¬ Board.isLost GameConfig.tiny
      (Placement.applyStep GameConfig.tiny GameState.init.board pl) :=
  init_applyStep_not_lost_of_valid (by decide) pl hv

/-- The post-step `GameState` from `init` with a valid placement is not lost. -/
theorem init_step_not_lost {cfg : GameConfig} (hrows : 4 ≤ cfg.rows)
    (pl : Placement) (hv : pl.Valid cfg) :
    ¬ (GameState.init.step cfg pl).lost cfg := by
  unfold GameState.lost GameState.step
  exact init_applyStep_not_lost_of_valid hrows pl hv

/-- Standard config: stepping init with a valid placement gives a not-lost state. -/
theorem standard_init_step_not_lost (pl : Placement)
    (hv : pl.Valid GameConfig.standard) :
    ¬ (GameState.init.step GameConfig.standard pl).lost GameConfig.standard :=
  init_step_not_lost (by decide) pl hv

/-- Tiny config: stepping init with a valid placement gives a not-lost state. -/
theorem tiny_init_step_not_lost (pl : Placement)
    (hv : pl.Valid GameConfig.tiny) :
    ¬ (GameState.init.step GameConfig.tiny pl).lost GameConfig.tiny :=
  init_step_not_lost (by decide) pl hv

/-- Init can step with any valid placement (drawability is trivial since
`init.bag = Bag.full`). -/
theorem init_step_canDraw_piece (pl : Placement) :
    GameState.init.bag.canDraw pl.piece :=
  GameState.init_canDraw pl.piece

/-- Init is trivially reachable (alias of the `Reachable.init` constructor). -/
theorem init_reachable (cfg : GameConfig) :
    Reachable cfg GameState.init :=
  Reachable.init

/-- Init has zero board count, so the count-mod invariants hold trivially. -/
theorem init_board_count_eq_zero : GameState.init.board.count = 0 :=
  GameState.init_board_count

/-- Every reachable state has positive bag card. -/
theorem Reachable.bag_card_pos' {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : 0 < g.bag.card :=
  Finset.card_pos.mpr h.bag_nonempty

/-- Every reachable state has bag card `≥ 1`. -/
theorem Reachable.one_le_bag_card {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : 1 ≤ g.bag.card :=
  h.bag_card_pos'

/-- Every reachable state's `init`-form drawable witness (any piece). -/
theorem Reachable.exists_piece_in_bag {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : ∃ p : Piece, p ∈ g.bag :=
  h.bag_nonempty

/-- The init state's `lost` predicate decides to `False`. -/
theorem init_lost_decide (cfg : GameConfig) :
    decide (GameState.init.lost cfg) = false :=
  decide_eq_false (GameState.init_not_lost cfg)

/-- The init state's `lost` predicate is equivalent to `False`. -/
theorem init_lost_iff_false (cfg : GameConfig) :
    GameState.init.lost cfg ↔ False :=
  ⟨fun h => GameState.init_not_lost cfg h, False.elim⟩

/-- The init state's `¬ lost cfg` is decidably `True`. -/
theorem not_init_lost_iff_true (cfg : GameConfig) :
    ¬ GameState.init.lost cfg ↔ True :=
  ⟨fun _ => trivial, fun _ => GameState.init_not_lost cfg⟩

/-- The init state is reachable AND not lost. Direct conjunction. -/
theorem init_reachable_and_not_lost (cfg : GameConfig) :
    Reachable cfg GameState.init ∧ ¬ GameState.init.lost cfg :=
  ⟨init_reachable cfg, GameState.init_not_lost cfg⟩

/-- The init state is reachable AND has a nonempty bag. -/
theorem init_reachable_and_bag_nonempty (cfg : GameConfig) :
    Reachable cfg GameState.init ∧ GameState.init.bag.Nonempty :=
  ⟨init_reachable cfg, GameState.init_bag_nonempty⟩

/-- Full init invariant bundle: Reachable, bag nonempty, not lost. -/
theorem init_invariants (cfg : GameConfig) :
    Reachable cfg GameState.init ∧ GameState.init.bag.Nonempty ∧
      ¬ GameState.init.lost cfg :=
  ⟨init_reachable cfg, GameState.init_bag_nonempty,
    GameState.init_not_lost cfg⟩

/-- Standard config: init invariants pre-bundled. -/
theorem standard_init_invariants :
    Reachable GameConfig.standard GameState.init ∧
    GameState.init.bag.Nonempty ∧
    ¬ GameState.init.lost GameConfig.standard :=
  init_invariants GameConfig.standard

/-- Tiny config: init invariants pre-bundled. -/
theorem tiny_init_invariants :
    Reachable GameConfig.tiny GameState.init ∧
    GameState.init.bag.Nonempty ∧
    ¬ GameState.init.lost GameConfig.tiny :=
  init_invariants GameConfig.tiny

/-- Extended init bundle: Reachable, bag nonempty, not lost, WF. -/
theorem init_invariants_with_wf (cfg : GameConfig) :
    Reachable cfg GameState.init ∧ GameState.init.bag.Nonempty ∧
      ¬ GameState.init.lost cfg ∧ Board.WF cfg GameState.init.board :=
  ⟨init_reachable cfg, GameState.init_bag_nonempty,
    GameState.init_not_lost cfg, GameState.init_board_wf cfg⟩

/-- Standard config: extended init bundle. -/
theorem standard_init_invariants_with_wf :
    Reachable GameConfig.standard GameState.init ∧
    GameState.init.bag.Nonempty ∧
    ¬ GameState.init.lost GameConfig.standard ∧
    Board.WF GameConfig.standard GameState.init.board :=
  init_invariants_with_wf GameConfig.standard

/-- Tiny config: extended init bundle. -/
theorem tiny_init_invariants_with_wf :
    Reachable GameConfig.tiny GameState.init ∧
    GameState.init.bag.Nonempty ∧
    ¬ GameState.init.lost GameConfig.tiny ∧
    Board.WF GameConfig.tiny GameState.init.board :=
  init_invariants_with_wf GameConfig.tiny

/-- The bag after one step from init has positive cardinality. -/
theorem init_step_bag_card_pos {cfg : GameConfig} (pl : Placement) :
    0 < (GameState.init.step cfg pl).bag.card :=
  GameState.init_step_bag_card_pos pl

/-- The bag after one step from init is nonempty. -/
theorem init_step_bag_nonempty {cfg : GameConfig} (pl : Placement) :
    (GameState.init.step cfg pl).bag.Nonempty :=
  GameState.init_step_bag_nonempty pl

/-- The bag after one step from init has nonzero card. -/
theorem init_step_bag_card_ne_zero {cfg : GameConfig} (pl : Placement) :
    (GameState.init.step cfg pl).bag.card ≠ 0 :=
  Nat.ne_of_gt (init_step_bag_card_pos pl)

/-- The bag after one step from init is not empty. -/
theorem init_step_bag_ne_empty {cfg : GameConfig} (pl : Placement) :
    (GameState.init.step cfg pl).bag ≠ ∅ :=
  (init_step_bag_nonempty pl).ne_empty

/-- The bag after one step from init has fewer than 7 pieces. -/
theorem init_step_bag_card_lt_seven {cfg : GameConfig} (pl : Placement) :
    (GameState.init.step cfg pl).bag.card < 7 :=
  GameState.init_step_bag_card_lt_seven pl

/-- The bag after one step from init has at most 6 pieces. -/
theorem init_step_bag_card_le_six {cfg : GameConfig} (pl : Placement) :
    (GameState.init.step cfg pl).bag.card ≤ 6 :=
  GameState.init_step_bag_card_le_six pl

/-- The bag after one step from init has card in `[1, 6]`. -/
theorem init_step_bag_card_interval {cfg : GameConfig} (pl : Placement) :
    1 ≤ (GameState.init.step cfg pl).bag.card ∧
    (GameState.init.step cfg pl).bag.card ≤ 6 :=
  ⟨init_step_bag_card_pos pl, init_step_bag_card_le_six pl⟩

/-- Full one-step-from-init bundle (given valid placement & low cfg): the
post-step state is Reachable, has nonempty bag, and is not lost. -/
theorem init_step_invariants {cfg : GameConfig} (hrows : 4 ≤ cfg.rows)
    (pl : Placement) (hv : pl.Valid cfg) :
    Reachable cfg (GameState.init.step cfg pl) ∧
    (GameState.init.step cfg pl).bag.Nonempty ∧
    ¬ (GameState.init.step cfg pl).lost cfg :=
  ⟨Reachable.init_step hv, init_step_bag_nonempty pl,
    init_step_not_lost hrows pl hv⟩

/-- Standard config: one-step-from-init invariants. -/
theorem standard_init_step_invariants (pl : Placement)
    (hv : pl.Valid GameConfig.standard) :
    Reachable GameConfig.standard (GameState.init.step GameConfig.standard pl) ∧
    (GameState.init.step GameConfig.standard pl).bag.Nonempty ∧
    ¬ (GameState.init.step GameConfig.standard pl).lost GameConfig.standard :=
  init_step_invariants (by decide) pl hv

/-- Tiny config: one-step-from-init invariants. -/
theorem tiny_init_step_invariants (pl : Placement)
    (hv : pl.Valid GameConfig.tiny) :
    Reachable GameConfig.tiny (GameState.init.step GameConfig.tiny pl) ∧
    (GameState.init.step GameConfig.tiny pl).bag.Nonempty ∧
    ¬ (GameState.init.step GameConfig.tiny pl).lost GameConfig.tiny :=
  init_step_invariants (by decide) pl hv

/-- Every reachable state is either `init` or a single step from another
reachable state. Direct case analysis on `Reachable`. -/
theorem Reachable.eq_init_or_step {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    g = GameState.init ∨ ∃ g' : GameState, ∃ pl : Placement,
      Reachable cfg g' ∧ g'.bag.canDraw pl.piece ∧
      pl.Valid cfg ∧ ¬ g'.lost cfg ∧ g = g'.step cfg pl := by
  cases h with
  | init => exact Or.inl rfl
  | @step g' pl hr hpb hv hnl =>
    exact Or.inr ⟨g', pl, hr, hpb, hv, hnl, rfl⟩

/-- A reachable state that's not `init` arose from a single step
(bundled with all four step preconditions). -/
theorem Reachable.exists_step_of_ne_init_bundled {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hne : g ≠ GameState.init) :
    ∃ g' : GameState, ∃ pl : Placement,
      Reachable cfg g' ∧ g'.bag.canDraw pl.piece ∧
      pl.Valid cfg ∧ ¬ g'.lost cfg ∧ g = g'.step cfg pl :=
  h.eq_init_or_step.resolve_left hne

/-- Sugar: `Reachable.init_step` with `cfg` made explicit. -/
theorem Reachable.init_step_of_valid (cfg : GameConfig) {pl : Placement}
    (hv : pl.Valid cfg) : Reachable cfg (GameState.init.step cfg pl) :=
  Reachable.init_step hv

/-- Standard-config specialisation of `init_step_of_valid`. -/
theorem Reachable.standard_init_step_of_valid {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    Reachable GameConfig.standard (GameState.init.step GameConfig.standard pl) :=
  Reachable.init_step_of_valid GameConfig.standard hv

/-- Tiny-config specialisation of `init_step_of_valid`. -/
theorem Reachable.tiny_init_step_of_valid {pl : Placement}
    (hv : pl.Valid GameConfig.tiny) :
    Reachable GameConfig.tiny (GameState.init.step GameConfig.tiny pl) :=
  Reachable.init_step_of_valid GameConfig.tiny hv



end Tetris
