import Mathlib
import Proofs.Invariants.StepInvariants

/-!
# Gameplay theorems

Correctness facts about the two gameplay mechanics in this model:

* **Line clearing** (`Board.clearLines`): clearing the empty board is a no-op;
  clearing never increases the cell count; if no row is full it does nothing;
  and — the key correctness property — after clearing, *no* row is full
  (`clearLines_no_full`): every complete line really is removed.

* **Loss** (`Board.isLost`): a board is lost exactly when some column's stack
  height exceeds the playing field; a board confined to the field is not lost.

Note on the move model: every piece is **hard dropped** — a move is a
`Placement` (piece, rotation, column) and the piece falls straight down to its
resting position. There is no soft drop, lock delay, or post-drop movement.
-/

namespace Tetris
namespace Board

/-! ## Line clearing -/

/-- Clearing the empty board yields the empty board. -/
@[simp] theorem clearLines_empty (cfg : GameConfig) :
    clearLines cfg (empty) = empty := by
  simp [clearLines, empty]

/-- Clearing never increases the number of filled cells. -/
theorem clearLines_count_le (cfg : GameConfig) (b : Board) :
    (clearLines cfg b).count ≤ b.count := by
  rw [clearLines_card]
  exact Finset.card_filter_le _ _

/-- If no row is full, clearing does nothing. -/
theorem clearLines_id_of_no_full {cfg : GameConfig} {b : Board}
    (h : ∀ r, ¬ isFull cfg b r) : clearLines cfg b = b := by
  unfold clearLines
  rw [Finset.filter_true_of_mem (fun p _ => h p.2)]
  have hcb : ∀ r, clearedBelow cfg b r = 0 := by
    intro r
    have hfr : fullRows cfg b = ∅ := by
      unfold fullRows
      rw [Finset.filter_eq_empty_iff]
      intro x _; exact h x
    unfold clearedBelow; rw [hfr]; simp
  have hf : (fun p : Coord => (p.1, p.2 - clearedBelow cfg b p.2)) = id := by
    funext p; rw [hcb]; simp
  rw [hf, Finset.image_id]

/-- The number of lines cleared from a board: the count of complete rows. -/
def linesCleared (cfg : GameConfig) (b : Board) : ℕ := (fullRows cfg b).card

/-- Clearing removes exactly `cols * linesCleared` cells. -/
theorem cells_cleared {cfg : GameConfig} {b : Board} (hb : WF cfg b) :
    b.count = (clearLines cfg b).count + cfg.cols * linesCleared cfg b :=
  clearLines_count_add hb

/-- **Parity invariant under `applyStep`**: the cell count increases by `4`
modulo `cfg.cols`. Combines `Placement.count_place` (drops 4 cells) with
`cells_cleared` (clearing removes a multiple of `cfg.cols`). Atlas-relevant:
every reachable state's cell count satisfies `count ≡ 4n (mod cfg.cols)`
where `n` is the number of pieces placed since `init`. -/
theorem applyStep_count_mod_cols {cfg : GameConfig} {b : Board} {pl : Placement}
    (hb : Board.WF cfg b) (hv : pl.Valid cfg) :
    (Placement.applyStep cfg b pl).count % cfg.cols = (b.count + 4) % cfg.cols := by
  unfold Placement.applyStep
  have hpwf := Placement.place_wf hb hv
  have h := cells_cleared hpwf
  rw [Placement.count_place] at h
  rw [h, Nat.add_mul_mod_self_left]

/-- A single step adds at most 4 cells (and may clear lines). -/
theorem applyStep_count_le {cfg : GameConfig} (b : Board) (pl : Placement) :
    (Placement.applyStep cfg b pl).count ≤ b.count + 4 := by
  unfold Placement.applyStep
  have h1 : (Board.clearLines cfg (pl.place b)).count ≤ (pl.place b).count :=
    Board.clearLines_count_le cfg _
  rw [Placement.count_place] at h1
  exact h1



/-- **Clearing actually clears every full line:** after `clearLines`, no row is
full. (Gravity only relocates surviving rows, and being injective it never
recombines two partial rows into a full one.) -/
theorem clearLines_no_full {cfg : GameConfig} (b : Board) (hcol : 0 < cfg.cols)
    (r' : ℕ) : ¬ isFull cfg (clearLines cfg b) r' := by
  intro hfull
  have m0 := hfull 0 (Finset.mem_range.2 hcol)
  simp only [clearLines, Finset.mem_image, Finset.mem_filter] at m0
  obtain ⟨p0, ⟨_, hp0nf⟩, hp0e⟩ := m0
  rw [Prod.mk.injEq] at hp0e
  apply hp0nf
  intro c hc
  have mc := hfull c hc
  simp only [clearLines, Finset.mem_image, Finset.mem_filter] at mc
  obtain ⟨q, ⟨hqb, hqnf⟩, hqe⟩ := mc
  rw [Prod.mk.injEq] at hqe
  have hqeq : q.2 = p0.2 :=
    row_map_eq_of_not_full hqnf hp0nf (by rw [hqe.2, hp0e.2])
  have : q = (c, p0.2) := Prod.ext_iff.mpr ⟨hqe.1, hqeq⟩
  rw [← this]; exact hqb

/-! ## Loss -/

/-- A board confined to the playing field (every cell strictly below `rows`) is
not lost. -/
theorem not_isLost_of_forall_lt {cfg : GameConfig} {b : Board}
    (h : ∀ p ∈ b, p.2 < cfg.rows) : ¬ isLost cfg b := by
  rintro ⟨p, hp, hr⟩
  have := h p hp
  omega

/-- **A board is lost exactly when some column overflows the field**, i.e. some
column's stack height exceeds `rows`. -/
theorem isLost_iff (cfg : GameConfig) (b : Board) :
    isLost cfg b ↔ ∃ j, cfg.rows < colHeight b j := by
  constructor
  · rintro ⟨p, hp, hr⟩
    refine ⟨p.1, ?_⟩
    unfold colHeight
    rw [Finset.lt_sup_iff]
    refine ⟨p.2, ?_, by omega⟩
    simp only [colRows, Finset.mem_image, Finset.mem_filter]
    exact ⟨p, ⟨hp, rfl⟩, rfl⟩
  · rintro ⟨j, hj⟩
    unfold colHeight at hj
    rw [Finset.lt_sup_iff] at hj
    obtain ⟨x, hx, hxlt⟩ := hj
    simp only [colRows, Finset.mem_image, Finset.mem_filter] at hx
    obtain ⟨p, ⟨hpb, _⟩, hpe⟩ := hx
    exact ⟨p, hpb, by omega⟩

end Board

/-- Lift of `Board.applyStep_count_le` to `GameState.step`. -/
theorem GameState.step_count_le {cfg : GameConfig} (g : GameState) (pl : Placement) :
    (g.step cfg pl).board.count ≤ g.board.count + 4 := by
  rw [GameState.step_board]
  exact Board.applyStep_count_le g.board pl

/-- Reachable lift: any reachable state's successor has count ≤
prior count + 4. Doesn't need any hypothesis beyond `Reachable`. -/
theorem Reachable.step_count_le {cfg : GameConfig} {g : GameState}
    (_h : Reachable cfg g) (pl : Placement) :
    (g.step cfg pl).board.count ≤ g.board.count + 4 :=
  GameState.step_count_le g pl

/-- First-step bound from init: count ≤ 4 (starting from count = 0). -/
theorem GameState.init_step_count_le_four (cfg : GameConfig) (pl : Placement) :
    (GameState.init.step cfg pl).board.count ≤ 4 := by
  have h := GameState.step_count_le (cfg := cfg) GameState.init pl
  rw [GameState.init_board] at h
  simpa using h

/-- **Parity invariant on reachable states**: there exists `n : ℕ` (the
number of pieces placed) such that `g.board.count ≡ 4n (mod cfg.cols)`.
A strong atlas-construction obstruction: reachable cell counts are
constrained to a coset of `4·ℕ` modulo `cfg.cols`. -/
theorem Reachable.exists_count_mod_cols {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    ∃ n : ℕ, g.board.count % cfg.cols = (4 * n) % cfg.cols := by
  induction h with
  | init =>
    refine ⟨0, ?_⟩
    rw [GameState.init_board]
    change (∅ : Board).card % cfg.cols = 0
    simp
  | @step g pl hreach _hpb hvalid _hlost ih =>
    obtain ⟨n, hn⟩ := ih
    refine ⟨n + 1, ?_⟩
    change (Placement.applyStep cfg g.board pl).count % cfg.cols = _
    rw [Board.applyStep_count_mod_cols hreach.wf hvalid]
    rw [show 4 * (n + 1) = 4 * n + 4 from by ring,
        Nat.add_mod g.board.count 4, hn, ← Nat.add_mod]

/-- **Tiny-config corollary**: reachable cell counts are always divisible
by 4 (since `tiny.cols = 4` and pieces have 4 cells). -/
theorem Reachable.tiny_count_mod_four {g : GameState}
    (h : Reachable GameConfig.tiny g) : g.board.count % 4 = 0 := by
  obtain ⟨n, hn⟩ := h.exists_count_mod_cols
  rw [GameConfig.tiny_cols] at hn
  rw [hn]
  exact Nat.mul_mod_right _ _

/-- **Standard-config corollary**: reachable cell counts are always *even*.
Since `4·n` cycles through `{0, 4, 8, 2, 6}` mod 10 — all even — and any
multiple-of-10 line-clear contribution is even, the total cell count
inherits this even-parity invariant. -/
theorem Reachable.standard_count_mod_two {g : GameState}
    (h : Reachable GameConfig.standard g) : g.board.count % 2 = 0 := by
  obtain ⟨n, hn⟩ := h.exists_count_mod_cols
  rw [GameConfig.standard_cols] at hn
  have hdvd : (2 : ℕ) ∣ 10 := by decide
  have h2 : g.board.count % 2 = g.board.count % 10 % 2 :=
    (Nat.mod_mod_of_dvd _ hdvd).symm
  rw [h2, hn, Nat.mod_mod_of_dvd _ hdvd]
  omega

/-- **Contrapositive — odd cell counts are unreachable in standard**. Any
game state with an odd cell count cannot be reached from `init` under
standard rules. Atlas: instant pruning of any candidate state with
odd `board.count`. -/
theorem not_reachable_standard_of_count_odd {g : GameState}
    (hodd : g.board.count % 2 = 1) : ¬ Reachable GameConfig.standard g := by
  intro h
  have := Reachable.standard_count_mod_two h
  omega

/-- `Even`-form of `Reachable.standard_count_mod_two`. -/
theorem Reachable.standard_count_even {g : GameState}
    (h : Reachable GameConfig.standard g) : Even g.board.count :=
  Nat.even_iff.mpr h.standard_count_mod_two

/-- `Even`-form of `Reachable.tiny_count_mod_four` (`% 4 = 0`
implies `% 2 = 0`). -/
theorem Reachable.tiny_count_even {g : GameState}
    (h : Reachable GameConfig.tiny g) : Even g.board.count := by
  have h4 := h.tiny_count_mod_four
  refine Nat.even_iff.mpr ?_
  omega

/-- Standard-config: odd count rules out reachability (`Even`
form). -/
theorem not_reachable_standard_of_not_even {g : GameState}
    (hne : ¬ Even g.board.count) : ¬ Reachable GameConfig.standard g :=
  fun h => hne h.standard_count_even

/-- Tiny-config: odd count rules out reachability (mod-2 form). -/
theorem not_reachable_tiny_of_count_odd {g : GameState}
    (hodd : g.board.count % 2 = 1) : ¬ Reachable GameConfig.tiny g := by
  intro h
  have he : Even g.board.count := h.tiny_count_even
  rw [Nat.even_iff] at he
  omega

/-- Tiny-config: odd count rules out reachability (`Even`
form). -/
theorem not_reachable_tiny_of_not_even {g : GameState}
    (hne : ¬ Even g.board.count) : ¬ Reachable GameConfig.tiny g :=
  fun h => hne h.tiny_count_even


/-- **Contrapositive — non-multiples of 4 are unreachable in tiny**. -/
theorem not_reachable_tiny_of_count_not_mod_four {g : GameState}
    (h4 : g.board.count % 4 ≠ 0) : ¬ Reachable GameConfig.tiny g := fun h =>
  h4 (Reachable.tiny_count_mod_four h)

/-- **Exact count formula**: for every reachable state, there exist `n`
(pieces placed) and `k` (lines cleared so far) such that
`g.board.count + cfg.cols * k = 4 * n`. Strictly stronger than the modular
invariant — gives the precise relationship between pieces, clears, and
current cells. Atlas: classify reachable states by `(n, k)` pair. -/
theorem Reachable.exists_count_eq {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    ∃ n k : ℕ, g.board.count + cfg.cols * k = 4 * n := by
  induction h with
  | init =>
    refine ⟨0, 0, ?_⟩
    rw [GameState.init_board]
    change (∅ : Board).card + cfg.cols * 0 = 0
    simp
  | @step g pl hreach _hpb hvalid _hlost ih =>
    obtain ⟨n, k, hnk⟩ := ih
    have hpwf := Placement.place_wf hreach.wf hvalid
    have h_cells := Board.cells_cleared hpwf
    rw [Placement.count_place] at h_cells
    set lc := Board.linesCleared cfg (pl.place g.board)
    refine ⟨n + 1, k + lc, ?_⟩
    change (Board.clearLines cfg (pl.place g.board)).count +
           cfg.cols * (k + lc) = 4 * (n + 1)
    have hdist : cfg.cols * (k + lc) = cfg.cols * k + cfg.cols * lc := by ring
    rw [hdist]
    omega

/-- **Bound on cleared lines**: at most `4n` total clears after `n` pieces.
Since each piece adds 4 cells and `cfg.cols * k` cells have been cleared,
`k ≤ 4n / cfg.cols ≤ 4n` (using `cfg.cols ≥ 1`). -/
theorem Reachable.exists_count_eq_bounded {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) :
    ∃ n k : ℕ, g.board.count + cfg.cols * k = 4 * n ∧ k ≤ 4 * n := by
  obtain ⟨n, k, hnk⟩ := h.exists_count_eq
  refine ⟨n, k, hnk, ?_⟩
  have h1 : k ≤ cfg.cols * k := Nat.le_mul_of_pos_left k cfg.cols_pos
  omega

/-- **Count is bounded by 4 × pieces placed**: every reachable state has
`count ≤ 4*n` for some `n` (the number of pieces placed). Trivial from
the exact count formula. -/
theorem Reachable.exists_count_le {cfg : GameConfig} {g : GameState}
    (h : Reachable cfg g) : ∃ n, g.board.count ≤ 4 * n := by
  obtain ⟨n, _, hnk⟩ := h.exists_count_eq
  exact ⟨n, by omega⟩

/-- **Tighter bound for wide boards**: when `cfg.cols ≥ 4`, total cleared
lines `k ≤ n` (each piece can clear at most 1 line on average across
wide boards). For standard and tiny configs. -/
theorem Reachable.exists_count_eq_le_n {cfg : GameConfig} {g : GameState}
    (hcols : 4 ≤ cfg.cols) (h : Reachable cfg g) :
    ∃ n k : ℕ, g.board.count + cfg.cols * k = 4 * n ∧ k ≤ n := by
  obtain ⟨n, k, hnk⟩ := h.exists_count_eq
  refine ⟨n, k, hnk, ?_⟩
  have h1 : 4 * k ≤ cfg.cols * k := Nat.mul_le_mul_right k hcols
  omega

/-- Standard specialization of `Reachable.exists_count_eq_le_n`. -/
theorem Reachable.standard_exists_count_eq_le_n {g : GameState}
    (h : Reachable GameConfig.standard g) :
    ∃ n k : ℕ, g.board.count + GameConfig.standard.cols * k = 4 * n ∧ k ≤ n :=
  h.exists_count_eq_le_n (by decide)

/-- Tiny specialization of `Reachable.exists_count_eq_le_n`. -/
theorem Reachable.tiny_exists_count_eq_le_n {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    ∃ n k : ℕ, g.board.count + GameConfig.tiny.cols * k = 4 * n ∧ k ≤ n :=
  h.exists_count_eq_le_n (by decide)

/-- Standard specialization of `Reachable.exists_count_eq`. -/
theorem Reachable.standard_exists_count_eq {g : GameState}
    (h : Reachable GameConfig.standard g) :
    ∃ n k : ℕ, g.board.count + GameConfig.standard.cols * k = 4 * n :=
  h.exists_count_eq

/-- Tiny specialization of `Reachable.exists_count_eq`. -/
theorem Reachable.tiny_exists_count_eq {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    ∃ n k : ℕ, g.board.count + GameConfig.tiny.cols * k = 4 * n :=
  h.exists_count_eq

/-- Any standard-reachable board with count = 1 is unreachable
(1 is odd). Concrete pruning fact. -/
theorem not_reachable_standard_of_count_eq_one {g : GameState}
    (h1 : g.board.count = 1) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 3 is unreachable
(3 is odd). -/
theorem not_reachable_standard_of_count_eq_three {g : GameState}
    (h3 : g.board.count = 3) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 5 is unreachable
(5 is odd). -/
theorem not_reachable_standard_of_count_eq_five {g : GameState}
    (h5 : g.board.count = 5) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 7 is unreachable
(7 is odd). -/
theorem not_reachable_standard_of_count_eq_seven {g : GameState}
    (h7 : g.board.count = 7) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 9 is unreachable
(9 is odd). -/
theorem not_reachable_standard_of_count_eq_nine {g : GameState}
    (h9 : g.board.count = 9) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 11 is unreachable
(11 is odd). -/
theorem not_reachable_standard_of_count_eq_eleven {g : GameState}
    (h11 : g.board.count = 11) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Any standard-reachable board with count = 13 is unreachable
(13 is odd). -/
theorem not_reachable_standard_of_count_eq_thirteen {g : GameState}
    (h13 : g.board.count = 13) : ¬ Reachable GameConfig.standard g := by
  apply not_reachable_standard_of_count_odd
  omega

/-- Standard specialization of `Reachable.exists_count_mod_cols`. -/
theorem Reachable.standard_exists_count_mod_ten {g : GameState}
    (h : Reachable GameConfig.standard g) :
    ∃ n : ℕ, g.board.count % 10 = (4 * n) % 10 := by
  have := h.exists_count_mod_cols
  rw [GameConfig.standard_cols] at this
  exact this

/-- Standard: `count % 10` is even (since `count` is even and `2 ∣ 10`). -/
theorem Reachable.standard_count_mod_ten_even {g : GameState}
    (h : Reachable GameConfig.standard g) : g.board.count % 10 % 2 = 0 := by
  have hdvd : (2 : ℕ) ∣ 10 := by decide
  rw [Nat.mod_mod_of_dvd _ hdvd]
  exact h.standard_count_mod_two

/-- Standard: if `count % 10` is odd, the state is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_odd {g : GameState}
    (hodd : g.board.count % 10 % 2 = 1) : ¬ Reachable GameConfig.standard g := by
  intro h
  have := h.standard_count_mod_ten_even
  omega

/-- Standard: any `count % 10 = 5` is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_eq_five {g : GameState}
    (h5 : g.board.count % 10 = 5) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_odd (by rw [h5])

/-- Standard: any `count % 10 = 1` is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_eq_one {g : GameState}
    (h1 : g.board.count % 10 = 1) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_odd (by rw [h1])

/-- Standard: any `count % 10 = 3` is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_eq_three {g : GameState}
    (h3 : g.board.count % 10 = 3) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_odd (by rw [h3])

/-- Standard: any `count % 10 = 7` is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_eq_seven {g : GameState}
    (h7 : g.board.count % 10 = 7) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_odd (by rw [h7])

/-- Standard: any `count % 10 = 9` is unreachable. -/
theorem not_reachable_standard_of_count_mod_ten_eq_nine {g : GameState}
    (h9 : g.board.count % 10 = 9) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_odd (by rw [h9])

/-- Standard: `count % 10 ∈ {0, 2, 4, 6, 8}` (explicit enumeration). -/
theorem Reachable.standard_count_mod_ten_mem_evens {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g.board.count % 10 = 0 ∨ g.board.count % 10 = 2 ∨
    g.board.count % 10 = 4 ∨ g.board.count % 10 = 6 ∨
    g.board.count % 10 = 8 := by
  have h10 : g.board.count % 10 < 10 := Nat.mod_lt _ (by decide)
  have heven := h.standard_count_mod_ten_even
  omega

/-- Tiny: `count % 4 = 1` is unreachable. -/
theorem not_reachable_tiny_of_count_mod_four_eq_one {g : GameState}
    (h1 : g.board.count % 4 = 1) : ¬ Reachable GameConfig.tiny g :=
  not_reachable_tiny_of_count_not_mod_four (by rw [h1]; decide)

/-- Tiny: `count % 4 = 2` is unreachable. -/
theorem not_reachable_tiny_of_count_mod_four_eq_two {g : GameState}
    (h2 : g.board.count % 4 = 2) : ¬ Reachable GameConfig.tiny g :=
  not_reachable_tiny_of_count_not_mod_four (by rw [h2]; decide)

/-- Tiny: `count % 4 = 3` is unreachable. -/
theorem not_reachable_tiny_of_count_mod_four_eq_three {g : GameState}
    (h3 : g.board.count % 4 = 3) : ¬ Reachable GameConfig.tiny g :=
  not_reachable_tiny_of_count_not_mod_four (by rw [h3]; decide)

/-- Standard: count = 15 is unreachable (15 % 10 = 5, covered by Tick 835). -/
theorem not_reachable_standard_of_count_eq_fifteen {g : GameState}
    (h15 : g.board.count = 15) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_eq_five (by rw [h15])

/-- Standard: count = 25 is unreachable (25 % 10 = 5). -/
theorem not_reachable_standard_of_count_eq_twentyfive {g : GameState}
    (h25 : g.board.count = 25) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_eq_five (by rw [h25])

/-- Standard: count = 35 is unreachable (35 % 10 = 5). -/
theorem not_reachable_standard_of_count_eq_thirtyfive {g : GameState}
    (h35 : g.board.count = 35) : ¬ Reachable GameConfig.standard g :=
  not_reachable_standard_of_count_mod_ten_eq_five (by rw [h35])

/-- Standard: a reachable state with full bag but `≠ init` has count `≥ 2`
(positive even count). -/
theorem Reachable.standard_count_ge_two_of_bag_full_of_ne_init {g : GameState}
    (h : Reachable GameConfig.standard g)
    (hbag : g.bag = Bag.full) (hne : g ≠ GameState.init) :
    2 ≤ g.board.count := by
  have hpos : 0 < g.board.count := h.board_count_pos_of_bag_full_of_ne_init hbag hne
  have hmod : g.board.count % 2 = 0 := h.standard_count_mod_two
  omega

/-- Tiny: a reachable state with full bag but `≠ init` has count `≥ 4`
(positive `% 4 = 0`). -/
theorem Reachable.tiny_count_ge_four_of_bag_full_of_ne_init {g : GameState}
    (h : Reachable GameConfig.tiny g)
    (hbag : g.bag = Bag.full) (hne : g ≠ GameState.init) :
    4 ≤ g.board.count := by
  have hpos : 0 < g.board.count := h.board_count_pos_of_bag_full_of_ne_init hbag hne
  have hmod : g.board.count % 4 = 0 := h.tiny_count_mod_four
  omega

/-- Standard: a reachable state's board count is `0` or `≥ 2`
(no positive odd counts; combines parity with the trichotomy). -/
theorem Reachable.standard_count_zero_or_ge_two {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g.board.count = 0 ∨ 2 ≤ g.board.count := by
  have hmod : g.board.count % 2 = 0 := h.standard_count_mod_two
  rcases Nat.eq_zero_or_pos g.board.count with h0 | hpos
  · exact Or.inl h0
  · exact Or.inr (by omega)

/-- Tiny: a reachable state's board count is `0` or `≥ 4`
(via `% 4 = 0` and trichotomy). -/
theorem Reachable.tiny_count_zero_or_ge_four {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g.board.count = 0 ∨ 4 ≤ g.board.count := by
  have hmod : g.board.count % 4 = 0 := h.tiny_count_mod_four
  rcases Nat.eq_zero_or_pos g.board.count with h0 | hpos
  · exact Or.inl h0
  · exact Or.inr (by omega)

/-- Standard: `count ≤ 1 ↔ count = 0` (since `count` is even). -/
theorem Reachable.standard_count_le_one_iff_eq_zero {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g.board.count ≤ 1 ↔ g.board.count = 0 := by
  rcases h.standard_count_zero_or_ge_two with h0 | h2
  · simp [h0]
  · constructor
    · intro hle; omega
    · intro hzero; omega

/-- Tiny: `count ≤ 3 ↔ count = 0` (since `count % 4 = 0`). -/
theorem Reachable.tiny_count_le_three_iff_eq_zero {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g.board.count ≤ 3 ↔ g.board.count = 0 := by
  rcases h.tiny_count_zero_or_ge_four with h0 | h4
  · simp [h0]
  · constructor
    · intro hle; omega
    · intro hzero; omega

/-- Standard: `count < 2 ↔ count = 0` (strict-`<` form). -/
theorem Reachable.standard_count_lt_two_iff_eq_zero {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g.board.count < 2 ↔ g.board.count = 0 := by
  rcases h.standard_count_zero_or_ge_two with h0 | h2
  · simp [h0]
  · constructor
    · intro hlt; omega
    · intro hzero; omega

/-- Tiny: `count < 4 ↔ count = 0` (strict-`<` form). -/
theorem Reachable.tiny_count_lt_four_iff_eq_zero {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g.board.count < 4 ↔ g.board.count = 0 := by
  rcases h.tiny_count_zero_or_ge_four with h0 | h4
  · simp [h0]
  · constructor
    · intro hlt; omega
    · intro hzero; omega

/-- Standard: `0 < count ↔ 2 ≤ count`. -/
theorem Reachable.standard_count_pos_iff_ge_two {g : GameState}
    (h : Reachable GameConfig.standard g) :
    0 < g.board.count ↔ 2 ≤ g.board.count := by
  rcases h.standard_count_zero_or_ge_two with h0 | h2
  · simp [h0]
  · constructor
    · intro _; omega
    · intro _; omega

/-- Tiny: `0 < count ↔ 4 ≤ count`. -/
theorem Reachable.tiny_count_pos_iff_ge_four {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    0 < g.board.count ↔ 4 ≤ g.board.count := by
  rcases h.tiny_count_zero_or_ge_four with h0 | h4
  · simp [h0]
  · constructor
    · intro _; omega
    · intro _; omega

/-- Standard: `g = init ↔ count < 2 ∧ bag = full`. -/
theorem Reachable.standard_eq_init_iff_count_lt_two_and_bag_full {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g = GameState.init ↔ g.board.count < 2 ∧ g.bag = Bag.full := by
  rw [h.eq_init_iff_count_zero_and_bag_full]
  rw [show (g.board.count = 0) ↔ (g.board.count < 2) from
        (h.standard_count_lt_two_iff_eq_zero).symm]

/-- Tiny: `g = init ↔ count < 4 ∧ bag = full`. -/
theorem Reachable.tiny_eq_init_iff_count_lt_four_and_bag_full {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g = GameState.init ↔ g.board.count < 4 ∧ g.bag = Bag.full := by
  rw [h.eq_init_iff_count_zero_and_bag_full]
  rw [show (g.board.count = 0) ↔ (g.board.count < 4) from
        (h.tiny_count_lt_four_iff_eq_zero).symm]

/-- Standard: `g = init ∨ count ≥ 2 ∨ (count = 0 ∧ bag ≠ full)` —
the third disjunct is the "fully-cleared board with partial bag" case,
which is genuinely reachable (e.g. 5 pieces placed + 2 lines cleared). -/
theorem Reachable.standard_eq_init_or_count_ge_two_or_cleared {g : GameState}
    (h : Reachable GameConfig.standard g) :
    g = GameState.init ∨ 2 ≤ g.board.count ∨
      (g.board.count = 0 ∧ g.bag ≠ Bag.full) := by
  rcases h.standard_count_zero_or_ge_two with h0 | h2
  · by_cases hbag : g.bag = Bag.full
    · exact Or.inl (h.eq_init_of_bag_full_of_count_zero h0 hbag)
    · exact Or.inr (Or.inr ⟨h0, hbag⟩)
  · exact Or.inr (Or.inl h2)

/-- Tiny: `g = init ∨ count ≥ 4 ∨ (count = 0 ∧ bag ≠ full)`. -/
theorem Reachable.tiny_eq_init_or_count_ge_four_or_cleared {g : GameState}
    (h : Reachable GameConfig.tiny g) :
    g = GameState.init ∨ 4 ≤ g.board.count ∨
      (g.board.count = 0 ∧ g.bag ≠ Bag.full) := by
  rcases h.tiny_count_zero_or_ge_four with h0 | h4
  · by_cases hbag : g.bag = Bag.full
    · exact Or.inl (h.eq_init_of_bag_full_of_count_zero h0 hbag)
    · exact Or.inr (Or.inr ⟨h0, hbag⟩)
  · exact Or.inr (Or.inl h4)

/-- Standard: `count = 0 → g = init ∨ bag ≠ full`. Forward direction
splitting `count = 0` into the two valid cases for reachable standard. -/
theorem Reachable.standard_eq_init_or_bag_ne_full_of_count_zero {g : GameState}
    (h : Reachable GameConfig.standard g) (h0 : g.board.count = 0) :
    g = GameState.init ∨ g.bag ≠ Bag.full := by
  by_cases hbag : g.bag = Bag.full
  · exact Or.inl (h.eq_init_of_bag_full_of_count_zero h0 hbag)
  · exact Or.inr hbag

/-- Tiny analogue of Tick 898: `count = 0 → g = init ∨ bag ≠ full`. Note
the same statement holds parametrically — this is just a cfg-renamed alias. -/
theorem Reachable.tiny_eq_init_or_bag_ne_full_of_count_zero {g : GameState}
    (h : Reachable GameConfig.tiny g) (h0 : g.board.count = 0) :
    g = GameState.init ∨ g.bag ≠ Bag.full := by
  by_cases hbag : g.bag = Bag.full
  · exact Or.inl (h.eq_init_of_bag_full_of_count_zero h0 hbag)
  · exact Or.inr hbag

/-- Parametric form: any reachable state with count = 0 is either `init`
or has a non-full bag. -/
theorem Reachable.eq_init_or_bag_ne_full_of_count_zero {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (h0 : g.board.count = 0) :
    g = GameState.init ∨ g.bag ≠ Bag.full := by
  by_cases hbag : g.bag = Bag.full
  · exact Or.inl (h.eq_init_of_bag_full_of_count_zero h0 hbag)
  · exact Or.inr hbag

/-- Parametric: a reachable state with full bag is either `init` or has
positive count. -/
theorem Reachable.eq_init_or_count_pos_of_bag_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hbag : g.bag = Bag.full) :
    g = GameState.init ∨ 0 < g.board.count := by
  by_cases hi : g = GameState.init
  · exact Or.inl hi
  · exact Or.inr (h.board_count_pos_of_bag_full_of_ne_init hbag hi)

/-- Parametric: a reachable state with full bag is either `init` or has
a nonempty board. -/
theorem Reachable.eq_init_or_board_nonempty_of_bag_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hbag : g.bag = Bag.full) :
    g = GameState.init ∨ g.board.Nonempty := by
  by_cases hi : g = GameState.init
  · exact Or.inl hi
  · exact Or.inr (h.board_nonempty_of_bag_full_of_ne_init hbag hi)

/-- Parametric: a reachable state with full bag is either `init` or has
a non-empty board. -/
theorem Reachable.eq_init_or_board_ne_empty_of_bag_full {cfg : GameConfig}
    {g : GameState} (h : Reachable cfg g) (hbag : g.bag = Bag.full) :
    g = GameState.init ∨ g.board ≠ Board.empty := by
  by_cases hi : g = GameState.init
  · exact Or.inl hi
  · exact Or.inr (h.board_ne_empty_of_bag_full_of_ne_init hbag hi)

/-- Standard: state with `count = 2 ∧ bag = full ∧ board ≠ empty` —
this is a coherent description of a reachable state shape. -/
theorem Reachable.standard_bag_full_count_ge_two_iff_board_nonempty
    {g : GameState} (h : Reachable GameConfig.standard g)
    (hbag : g.bag = Bag.full) :
    2 ≤ g.board.count ↔ g.board.Nonempty := by
  constructor
  · intro h2
    have hpos : 0 < g.board.count := by omega
    exact Finset.card_pos.mp hpos
  · intro hne
    have hpos : 0 < g.board.count := Finset.card_pos.mpr hne
    have hi : g ≠ GameState.init := by
      intro heq
      have : g.board = Board.empty := heq ▸ GameState.init_board
      exact (Finset.Nonempty.ne_empty hne) this
    exact h.standard_count_ge_two_of_bag_full_of_ne_init hbag hi

/-- Tiny: under `bag = full`, `count ≥ 4 ↔ board.Nonempty`. -/
theorem Reachable.tiny_bag_full_count_ge_four_iff_board_nonempty
    {g : GameState} (h : Reachable GameConfig.tiny g)
    (hbag : g.bag = Bag.full) :
    4 ≤ g.board.count ↔ g.board.Nonempty := by
  constructor
  · intro h4
    have hpos : 0 < g.board.count := by omega
    exact Finset.card_pos.mp hpos
  · intro hne
    have hpos : 0 < g.board.count := Finset.card_pos.mpr hne
    have hi : g ≠ GameState.init := by
      intro heq
      have : g.board = Board.empty := heq ▸ GameState.init_board
      exact (Finset.Nonempty.ne_empty hne) this
    exact h.tiny_count_ge_four_of_bag_full_of_ne_init hbag hi

/-- Any tiny-reachable board with count = 1 is unreachable
(1 isn't a multiple of 4). -/
theorem not_reachable_tiny_of_count_eq_one {g : GameState}
    (h1 : g.board.count = 1) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: count = 2 is unreachable. -/
theorem not_reachable_tiny_of_count_eq_two {g : GameState}
    (h2 : g.board.count = 2) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: count = 3 is unreachable. -/
theorem not_reachable_tiny_of_count_eq_three {g : GameState}
    (h3 : g.board.count = 3) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: count = 5 is unreachable (5 mod 4 = 1). -/
theorem not_reachable_tiny_of_count_eq_five {g : GameState}
    (h5 : g.board.count = 5) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: count = 6 is unreachable (6 mod 4 = 2). -/
theorem not_reachable_tiny_of_count_eq_six {g : GameState}
    (h6 : g.board.count = 6) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: count = 7 is unreachable (7 mod 4 = 3). -/
theorem not_reachable_tiny_of_count_eq_seven {g : GameState}
    (h7 : g.board.count = 7) : ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: any count in `[5, 7]` (inclusive both sides) is unreachable. -/
theorem not_reachable_tiny_of_count_in_five_to_seven {g : GameState}
    (h₁ : 5 ≤ g.board.count) (h₂ : g.board.count ≤ 7) :
    ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

/-- Tiny: any positive count below 4 is unreachable
(generalizes the three Tick 637/638 enumerated cases). -/
theorem not_reachable_tiny_of_count_pos_lt_four {g : GameState}
    (hpos : 0 < g.board.count) (hlt : g.board.count < 4) :
    ¬ Reachable GameConfig.tiny g := by
  apply not_reachable_tiny_of_count_not_mod_four
  omega

end Tetris
