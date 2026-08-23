import Mathlib
import Proofs.Survival.ClearDeviation

/-!
# Recurrence and period: the arithmetic skeleton of an immortal solver

`ClearRate` pins the asymptotic clearing rate at `2.8`; `ClearDeviation` bounds
the drift around it. This file extracts the remaining *structural* statistics —
facts about which occupancies are reachable at which times, how often the
solver must return to an exact balance, and what lengths a closed cycle can
have.

Everything again reads off the ledger `board.count n + 10 · cleared n = 4n`.

## Results

* **A congruence** (`count_mod_ten`): after `n` placements the board's cell
  count satisfies `count ≡ 4n (mod 10)`. Occupancy is not free to be anything
  in `[0, 200]` — at each time only 21 of the 201 values are arithmetically
  available, and which 21 is fixed by the piece count alone.
* **Exact balance is forced** (`exact_balance_of_count_eq`): whenever the board
  returns to a cell count it has held before, the window between the two visits
  cleared at *exactly* `2.8` rows per bag — no error at all — and its length is
  **a multiple of 5 placements** (`five_dvd_of_count_eq`).
* **And it happens often** (`exists_count_eq_le`): occupancy takes at most 201
  values, so among any 202 consecutive checkpoints two must agree. Every 201
  placements therefore contains an exact-balance window. The `2.8` rate is not
  merely an asymptotic average — it is *attained exactly*, on windows of
  bounded length, over and over.
* **And forever** (`exists_recurrent_count`): some occupancy value is revisited
  infinitely often. The occupancy series of an immortal solver is a recurrent
  walk on a finite set, never a drift.
* **Cycle lengths are quantised** (`five_dvd_of_trace_eq`,
  `thirtyfive_dvd_of_trace_eq`): any closed cycle in the state graph — the M2
  artifact — has length divisible by 5 from the mass ledger, and divisible by 7
  from the bag counter (`bag_card_trace`), hence **divisible by 35 placements =
  5 bags**. No closed cycle of any other length exists, whatever the board
  geometry.

That last item is the payoff: a purely arithmetic lower bound on the size of
any M2 certificate, derived without touching board geometry at all.

## The cell-count distribution: what is and is not determined

**Determined.** The count is always even (`count_even`); its residue mod 10 is
`4n`, hence 5-periodic (`count_mod_ten_add_five`) and **exactly uniform** over
`{0,2,4,6,8}` — two times share a residue iff they agree mod 5
(`count_mod_ten_ne`); the board can be empty only at multiples of 5 placements
(`five_dvd_of_count_eq_zero`); a drop clearing `k` rows needs `10k ≤ count + 4`
beforehand (`clear_step_le`), so a **tetris demands 36 cells already on the
stack** (`thirtysix_le_count_of_tetris`).

Also determined: the **frequency of clearing events**. Every clearing drop
removes between 1 and 4 rows (`fullRows_card_le_four`), and the total must hold
the `0.4`-rows-per-piece pace, so the fraction of line-clearing pieces is
trapped between `1/10` and `2/5` (`clearingSteps_le`, `le_clearingSteps`) —
`1/10` for a tetris-only strategy, `2/5` for a singles-only one.

**Not determined.** Where in `[0, 200]` the count actually *sits*. The ledger is
one linear equation: it fixes the residue and the mean of the increments, and
says nothing about the level. A perfect-clear-heavy strategy would hover near
`0`, a downstacking one near the ceiling, and no counting argument separates
them. Any theorem about the *shape* of the occupancy distribution has to come
from board geometry, not from this file.
-/

namespace Tetris
namespace ClearRate

open Filter Topology

/-! ## The occupancy congruence -/

/-- **Occupancy mod 10 is a clock.** After `n` placements the board holds
`≡ 4n (mod 10)` cells: clears remove exactly 10 at a time, so they cannot change
this residue. Only 21 of the 201 conceivable cell counts are available at any
given moment, and which 21 is determined by the piece count alone. -/
theorem count_mod_ten {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).board.count % 10 = (4 * n) % 10 := by
  have h := init_ledger hv n
  rw [GameConfig.standard_cols] at h
  omega

/-! ## Equal occupancy forces exact balance -/

/-- **Returning to a cell count means the window balanced exactly.** If the
board holds the same number of cells at two times, then between them the clears
consumed precisely the delivered mass: `10 · Δcleared = 4 · Δn`, zero error.
Moreover the gap is a multiple of 5 placements — `2 Δn = 5 Δcleared` and
`gcd(2,5) = 1`. -/
theorem five_dvd_of_count_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (hc : (trace GameConfig.standard π GameState.init n₁).board.count
        = (trace GameConfig.standard π GameState.init n₂).board.count) :
    5 ∣ (n₂ - n₁)
      ∧ 10 * (cleared GameConfig.standard π GameState.init n₂
            - cleared GameConfig.standard π GameState.init n₁)
        = 4 * (n₂ - n₁) := by
  have h1 := init_ledger hv n₁
  have h2 := init_ledger hv n₂
  rw [GameConfig.standard_cols] at h1 h2
  have hm := cleared_mono GameConfig.standard π GameState.init h12
  omega

/-- The exact-balance half, named on its own. -/
theorem exact_balance_of_count_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (hc : (trace GameConfig.standard π GameState.init n₁).board.count
        = (trace GameConfig.standard π GameState.init n₂).board.count) :
    10 * (cleared GameConfig.standard π GameState.init n₂
          - cleared GameConfig.standard π GameState.init n₁)
      = 4 * (n₂ - n₁) :=
  (five_dvd_of_count_eq hv h12 hc).2

/-! ## Exact-balance windows occur at bounded gaps, and forever -/

/-- Occupancy is confined to `[0, 200]` while alive. -/
theorem count_lt_two_hundred_one {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost GameConfig.standard) :
    (trace GameConfig.standard π GameState.init n).board.count < 201 := by
  have h := BagGrowth.count_le_capacity
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) n)
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  omega

/-- **Exact balance recurs at bounded gaps.** Occupancy takes at most 201
values, so among any 202 consecutive checkpoints two must coincide: every window
of 201 placements contains a sub-window over which the solver cleared at exactly
the `2.8`-per-bag pace, with no error whatsoever. The rate law is not only an
asymptotic average — it is attained on the nose, repeatedly, at bounded
intervals. -/
theorem exists_count_eq_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) (a : ℕ) :
    ∃ i j, i < j ∧ j ≤ 201 ∧
      (trace GameConfig.standard π GameState.init (a + i)).board.count
        = (trace GameConfig.standard π GameState.init (a + j)).board.count := by
  have hmaps : ∀ i ∈ Finset.range 202,
      (trace GameConfig.standard π GameState.init (a + i)).board.count
        ∈ Finset.range 201 := by
    intro i _
    exact Finset.mem_range.mpr (count_lt_two_hundred_one hv (hsurv (a + i)))
  have hcard : (Finset.range 201).card < (Finset.range 202).card := by
    simp
  obtain ⟨x, hx, y, hy, hxy, hfxy⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmaps
  rcases lt_or_gt_of_ne hxy with h | h
  · exact ⟨x, y, h, by have := Finset.mem_range.mp hy; omega, hfxy⟩
  · exact ⟨y, x, h, by have := Finset.mem_range.mp hx; omega, hfxy.symm⟩

/-- **Some occupancy recurs forever.** The occupancy of an immortal solver is a
recurrent walk on a finite set: at least one cell count is revisited infinitely
often, and every pair of those visits brackets an exactly-balanced window. -/
theorem exists_recurrent_count {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) :
    ∃ v : ℕ,
      {n : ℕ | (trace GameConfig.standard π GameState.init n).board.count = v}.Infinite := by
  classical
  set f : ℕ → Fin 201 := fun n =>
    ⟨(trace GameConfig.standard π GameState.init n).board.count,
      count_lt_two_hundred_one hv (hsurv n)⟩ with hf
  obtain ⟨b, hb⟩ := Finite.exists_infinite_fiber f
  refine ⟨(b : ℕ), ?_⟩
  rw [Set.infinite_coe_iff] at hb
  have hset : {n : ℕ | (trace GameConfig.standard π GameState.init n).board.count = (b : ℕ)}
      = f ⁻¹' {b} := by
    ext n
    simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_singleton_iff, hf]
    exact ⟨fun h => Fin.ext h, fun h => congrArg Fin.val h⟩
  rw [hset]
  exact hb

/-! ## The bag counter and the length of a closed cycle -/

/-- Drawing from a nonempty bag drops the count by one, or refills to 7 when it
would empty. -/
theorem card_draw {bag : Bag} {p : Piece} (hp : p ∈ bag) :
    (bag.draw p).card = if bag.card = 1 then 7 else bag.card - 1 := by
  have hpos : 0 < bag.card := Finset.card_pos.mpr ⟨p, hp⟩
  have herase := Finset.card_erase_of_mem hp
  unfold Bag.draw
  by_cases hz : bag.erase p = ∅
  · rw [if_pos hz, Bag.full_card]
    rw [hz, Finset.card_empty] at herase
    rw [if_pos (by omega)]
  · rw [if_neg hz, herase]
    have hne : bag.card ≠ 1 := by
      intro h
      refine hz (Finset.card_eq_zero.mp ?_)
      omega
    rw [if_neg hne]

/-- **The bag is a mod-7 clock.** Under legal draws the bag holds `7 − (n mod 7)`
pieces after `n` placements, so the bag state alone counts placements modulo
7. -/
theorem bag_card_trace {π : Policy GameConfig.standard}
    (hdraw : ∀ n, (π (trace GameConfig.standard π GameState.init n)).piece
      ∈ (trace GameConfig.standard π GameState.init n).bag) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).bag.card = 7 - n % 7 := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [trace_succ, GameState.step_bag, card_draw (hdraw k), ih]
    by_cases hk : k % 7 = 6
    · rw [if_pos (by omega)]
      omega
    · rw [if_neg (by omega)]
      omega

/-- **Any closed cycle has length divisible by 5.** A trace that returns to a
previous *board* has, by the mass ledger, cleared exactly `2/5` of a row per
placement over the loop — which forces the loop length to be a multiple of 5. -/
theorem five_dvd_of_trace_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    5 ∣ (n₂ - n₁) :=
  (five_dvd_of_count_eq hv h12 (by rw [h])).1

/-- **Any closed cycle has length divisible by 7.** The bag counter is a mod-7
clock, so returning to the same *bag* costs a whole number of bags. -/
theorem seven_dvd_of_trace_eq {π : Policy GameConfig.standard}
    (hdraw : ∀ n, (π (trace GameConfig.standard π GameState.init n)).piece
      ∈ (trace GameConfig.standard π GameState.init n).bag)
    {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    7 ∣ (n₂ - n₁) := by
  have h1 := bag_card_trace hdraw n₁
  have h2 := bag_card_trace hdraw n₂
  rw [h] at h1
  omega

/-- **Every closed cycle is a whole number of five-bag blocks.** Combining the
mass clock (mod 5) with the bag clock (mod 7): any legal loop in the Tetris
state graph has length divisible by `35` placements — exactly **5 bags**. This
is a lower bound on the size of any M2 cycle certificate, and it is derived from
counting alone: no board geometry, no reachability, no search. -/
theorem thirtyfive_dvd_of_trace_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ n, (π (trace GameConfig.standard π GameState.init n)).piece
      ∈ (trace GameConfig.standard π GameState.init n).bag)
    {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    35 ∣ (n₂ - n₁) := by
  have h5 := five_dvd_of_trace_eq hv h12 h
  have h7 := seven_dvd_of_trace_eq hdraw h12 h
  omega

/-- A nontrivial closed cycle is at least 5 bags long. -/
theorem thirtyfive_le_of_trace_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ n, (π (trace GameConfig.standard π GameState.init n)).piece
      ∈ (trace GameConfig.standard π GameState.init n).bag)
    {n₁ n₂ : ℕ} (h12 : n₁ < n₂)
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    n₁ + 35 ≤ n₂ := by
  have h35 := thirtyfive_dvd_of_trace_eq hv hdraw (le_of_lt h12) h
  omega

/-! ## What the ledger does and does not say about the cell-count distribution

The ledger determines the *arithmetic* of the cell count completely — its
parity, its residue, which times admit an empty board — and it determines the
*frequency* of clearing events. It says nothing whatever about the **level**:
where in `[0, 200]` an immortal solver's board actually sits is not a counting
question, and no theorem here decides it. -/

/-- The board's cell count is always even. -/
theorem count_even {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).board.count % 2 = 0 := by
  have h := init_ledger hv n
  rw [GameConfig.standard_cols] at h
  omega

/-- The residue is 5-periodic in the piece count. -/
theorem count_mod_ten_add_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    (trace GameConfig.standard π GameState.init (n + 5)).board.count % 10
      = (trace GameConfig.standard π GameState.init n).board.count % 10 := by
  rw [count_mod_ten hv, count_mod_ten hv]
  omega

/-- **The residue distribution is exactly uniform.** Two times carry the same
cell-count residue iff they agree mod 5, so across any five consecutive
placements the board visits each of the five even residues `{0,2,4,6,8}`
exactly once. This is the one distributional statement about occupancy that is
fully determined — and it is determined deterministically, not just on
average. -/
theorem count_mod_ten_ne {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {i j : ℕ} (hij : i % 5 ≠ j % 5) :
    (trace GameConfig.standard π GameState.init i).board.count % 10
      ≠ (trace GameConfig.standard π GameState.init j).board.count % 10 := by
  rw [count_mod_ten hv, count_mod_ten hv]
  omega

/-- **The board can only be empty at multiples of 5 placements.** A perfect
clear at piece `n` forces `5 ∣ n`. -/
theorem five_dvd_of_count_eq_zero {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hz : (trace GameConfig.standard π GameState.init n).board.count = 0) :
    5 ∣ n := by
  have h := init_ledger hv n
  rw [GameConfig.standard_cols, hz] at h
  omega

/-- **Clearing costs mass you must already have.** A drop clearing `k` rows
needs `10k` cells present after the drop, of which the piece supplies only 4:
`10k ≤ count + 4`. -/
theorem clear_step_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    10 * (cleared GameConfig.standard π GameState.init (n + 1)
          - cleared GameConfig.standard π GameState.init n)
      ≤ (trace GameConfig.standard π GameState.init n).board.count + 4 := by
  have hwf : Board.WF GameConfig.standard
      (trace GameConfig.standard π GameState.init n).board :=
    trace_board_wf hv (GameState.init_board_wf GameConfig.standard) n
  have h := BagGrowth.count_clearLines_add_cols
    (Placement.place_wf hwf (hv (trace GameConfig.standard π GameState.init n)))
  rw [Placement.count_place, GameConfig.standard_cols] at h
  rw [cleared_succ]
  omega

/-- **A tetris needs 36 cells on the board first.** Clearing four rows at once
consumes 40 cells and the piece brings 4, so the stack must already hold 36 —
nearly four full rows. Deep-clearing strategies are therefore committed to
running the board high. -/
theorem thirtysix_le_count_of_tetris {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (h4 : 4 ≤ cleared GameConfig.standard π GameState.init (n + 1)
              - cleared GameConfig.standard π GameState.init n) :
    36 ≤ (trace GameConfig.standard π GameState.init n).board.count := by
  have h := clear_step_le hv n
  omega

/-! ## How often a piece clears -/

/-- Number of the first `n` placements that cleared at least one row. -/
def clearingSteps (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      clearingSteps cfg π g0 n
        + (if 0 < (Board.fullRows cfg
              ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card then 1 else 0)

@[simp] theorem clearingSteps_zero (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) :
    clearingSteps cfg π g0 0 = 0 := rfl

theorem clearingSteps_succ (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (n : ℕ) :
    clearingSteps cfg π g0 (n + 1)
      = clearingSteps cfg π g0 n
        + (if 0 < (Board.fullRows cfg
              ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card then 1 else 0) :=
  rfl

/-- No row of a trace board is full: `init` is empty and every later board is a
`clearLines` image. -/
theorem trace_board_no_full {cfg : GameConfig} {π : Policy cfg} (n : ℕ) (r : ℕ) :
    ¬ Board.isFull cfg (trace cfg π GameState.init n).board r := by
  cases n with
  | zero =>
    intro hfull
    have h0 := hfull 0 (Finset.mem_range.2 cfg.cols_pos)
    rw [trace_zero] at h0
    exact GameState.init_board_no_mem _ h0
  | succ k =>
    rw [trace_succ, GameState.step_board, Placement.applyStep_eq_clearLines_place]
    exact Board.clearLines_no_full _ cfg.cols_pos r

/-- A single drop clears at most 4 rows. -/
theorem fullRows_card_le_four {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    (Board.fullRows cfg ((π (trace cfg π GameState.init n)).place
      (trace cfg π GameState.init n).board)).card ≤ 4 := by
  have h := linesCleared_place_le_four cfg
    (trace cfg π GameState.init n).board (π (trace cfg π GameState.init n))
    (trace_board_no_full n)
  rwa [Board.linesCleared] at h

/-- Restated on the cumulative counter. -/
theorem cleared_succ_le {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    cleared cfg π GameState.init (n + 1) ≤ cleared cfg π GameState.init n + 4 := by
  have h := fullRows_card_le_four (cfg := cfg) (π := π) n
  rw [cleared_succ]
  omega

/-- Every clearing step contributes at least one row. -/
theorem clearingSteps_le_cleared {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    clearingSteps cfg π GameState.init n ≤ cleared cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [clearingSteps_succ, cleared_succ]
    split <;> omega

/-- Every clearing step contributes at most four rows. -/
theorem cleared_le_four_mul_clearingSteps {cfg : GameConfig} {π : Policy cfg} (n : ℕ) :
    cleared cfg π GameState.init n ≤ 4 * clearingSteps cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have h4 := fullRows_card_le_four (cfg := cfg) (π := π) k
    rw [clearingSteps_succ, cleared_succ]
    split <;> omega

/-- **At most two pieces in five clear a row.** Each clearing piece removes at
least one row, and the total rows cleared can never exceed `0.4` per piece. -/
theorem clearingSteps_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    10 * clearingSteps GameConfig.standard π GameState.init n ≤ 4 * n := by
  have h1 := clearingSteps_le_cleared (cfg := GameConfig.standard) (π := π) n
  have h2 := cols_mul_cleared_le hv n
  rw [GameConfig.standard_cols] at h2
  omega

/-- **At least one piece in ten clears a row.** Each clearing piece removes at
most four rows, and the total rows cleared must keep pace at `0.4` per piece.
Together with `clearingSteps_le`: **the fraction of line-clearing pieces of any
immortal solver lies between `1/10` and `2/5`** — `1/10` exactly for a
tetris-only strategy, `2/5` exactly for a singles-only strategy, and nothing
outside that band is possible. -/
theorem le_clearingSteps {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost GameConfig.standard) :
    4 * n ≤ 40 * clearingSteps GameConfig.standard π GameState.init n + 200 := by
  have h1 := cleared_le_four_mul_clearingSteps (cfg := GameConfig.standard) (π := π) n
  have h2 := le_cols_mul_cleared hv hlive
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h2
  omega

/-! ## Which piece can clear four rows -/

/-- A dropped piece touches the same number of distinct rows as its profile:
hard-dropping only translates it. -/
theorem dropped_rows_card (b : Board) (pl : Placement) :
    ((pl.dropped b).image Prod.snd).card = (pl.shapeUp.image Prod.snd).card := by
  have h : (pl.dropped b).image Prod.snd
      = (pl.shapeUp.image Prod.snd).image (fun r => pl.dropOffset b + r) := by
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.image_image, Finset.image_image]
    rfl
  rw [h, Finset.card_image_of_injective _ (add_right_injective (pl.dropOffset b))]

/-- **Only the I tetromino spans four rows.** A 28-case check over every piece
and rotation. -/
theorem four_rows_only_I :
    ∀ (p : Piece) (r : Rotation),
      4 ≤ ((p.shapeUp r).image Prod.snd).card → p = Piece.I := by
  decide

/-- **A tetris requires an I piece.** On a board with no pending full rows,
every row cleared by a drop must contain a cell of that drop — so clearing four
rows forces the piece to span four rows, and only I does.

Operationally this is the sharpest constraint on a tetris-oriented solver: a bag
contains exactly one I, so a bag admits **at most one tetris**. Sustaining the
`2.8`-rows-per-bag pace on tetrises alone therefore demands converting `0.7` of
every bag's single I into a four-row clear — a 70% conversion rate on the one
piece the adversary controls the timing of. -/
theorem tetris_requires_I {cfg : GameConfig} {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (h4 : 4 ≤ (Board.fullRows cfg (pl.place b)).card) :
    pl.piece = Piece.I := by
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
      rcases hcplace with hb' | hd
      · exact absurd hb' hcb
      · exact hd
    rw [Finset.mem_image]
    exact ⟨(c, r), hcdrop, rfl⟩
  have hcard := Finset.card_le_card hsub
  rw [dropped_rows_card] at hcard
  exact four_rows_only_I pl.piece pl.rot (le_trans h4 hcard)

/-- Trace form: a four-row clear along a policy trace is played with an I. -/
theorem tetris_requires_I_trace {cfg : GameConfig} {π : Policy cfg} {n : ℕ}
    (h4 : 4 ≤ (Board.fullRows cfg
      ((π (trace cfg π GameState.init n)).place
        (trace cfg π GameState.init n).board)).card) :
    (π (trace cfg π GameState.init n)).piece = Piece.I :=
  tetris_requires_I (trace_board_no_full n) h4

/-! ## The trade-off the rate *does* impose

The limiting rate is a single number — the mean of the increments — and a
single number cannot determine a distribution. What it does determine is a
**relation** between two features of the distribution: how high the board is
allowed to run, and how often the solver must clear.

Keeping the board low caps the clear size (a `k`-row clear needs `10k − 4` cells
banked), and capping the clear size forces the clearing *frequency* up, because
the `0.4`-rows-per-piece pace has to be met either way. A solver may buy a tight
occupancy distribution or a low clearing frequency — never both. -/

/-- If no drop ever clears more than `K` rows, the cumulative clears are at most
`K` per clearing piece. -/
theorem cleared_le_mul_clearingSteps {cfg : GameConfig} {π : Policy cfg} {K : ℕ}
    (hK : ∀ n, (Board.fullRows cfg ((π (trace cfg π GameState.init n)).place
      (trace cfg π GameState.init n).board)).card ≤ K) (n : ℕ) :
    cleared cfg π GameState.init n ≤ K * clearingSteps cfg π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [cleared_succ, clearingSteps_succ]
    by_cases hc : 0 < (Board.fullRows cfg ((π (trace cfg π GameState.init k)).place
        (trace cfg π GameState.init k).board)).card
    · rw [if_pos hc, Nat.mul_add, Nat.mul_one]
      exact Nat.add_le_add ih (hK k)
    · have hc0 : (Board.fullRows cfg ((π (trace cfg π GameState.init k)).place
          (trace cfg π GameState.init k).board)).card = 0 := by omega
      rw [if_neg hc, hc0, Nat.add_zero, Nat.add_zero]
      exact ih

/-- **An occupancy ceiling caps the clear size.** If the board never holds more
than `10K − 4` cells, no drop can ever clear more than `K` rows: the mass simply
is not there. -/
theorem fullRows_card_le_of_count_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {K : ℕ}
    (hM : ∀ n, (trace GameConfig.standard π GameState.init n).board.count + 4 ≤ 10 * K)
    (n : ℕ) :
    (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init n)).place
        (trace GameConfig.standard π GameState.init n).board)).card ≤ K := by
  have h := clear_step_le hv n
  rw [cleared_succ] at h
  have hm := hM n
  omega

/-- **The tightness/frequency trade-off.** A solver whose drops never clear more
than `K` rows must clear on at least a `4/(10K)` fraction of its pieces. With
`K = 4` this is the familiar `1/10` (tetris-only); with `K = 1` it is `2/5`
(singles-only). Combined with `fullRows_card_le_of_count_le`: **holding the
occupancy distribution tight forces the clearing frequency up, and buying a low
clearing frequency forces the board to run high.** That relation — not the shape
of either distribution on its own — is what the `2.8` rate actually
determines. -/
theorem le_clearingSteps_of_max_clear {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {K : ℕ}
    (hK : ∀ n, (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init n)).place
        (trace GameConfig.standard π GameState.init n).board)).card ≤ K)
    {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost GameConfig.standard) :
    4 * n ≤ 10 * K * clearingSteps GameConfig.standard π GameState.init n + 200 := by
  have h1 := cleared_le_mul_clearingSteps (cfg := GameConfig.standard) (π := π) hK n
  have h2 := le_cols_mul_cleared hv hlive
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h2
  calc 4 * n
      ≤ 10 * cleared GameConfig.standard π GameState.init n + 200 := h2
    _ ≤ 10 * (K * clearingSteps GameConfig.standard π GameState.init n) + 200 :=
        Nat.add_le_add_right (Nat.mul_le_mul_left 10 h1) 200
    _ = 10 * K * clearingSteps GameConfig.standard π GameState.init n + 200 := by
        ring

/-! ## Design laws: what a solver can compute about itself

The results above are stated for analysis — given a run, what must be true. The
same ledger read forwards gives a solver **exact, checkable obligations from its
own current state**, with no lookahead and no heuristic. Occupancy is the whole
sufficient statistic: nothing else about the clearing constraint is visible to
the future. -/

/-- **The runway.** From a board holding `D` cells, at most `(200 − D)/4` more
placements can be made before a row *must* be cleared: dropping `w` pieces
without a clear leaves `D + 4w` cells, and the board holds 200. An exact,
O(1)-computable deadline — not an estimate. -/
theorem dry_runway_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
          = cleared GameConfig.standard π GameState.init n)
    (hlive : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    (trace GameConfig.standard π GameState.init n).board.count + 4 * w ≤ 200 := by
  have h1 := init_ledger hv n
  have h2 := init_ledger hv (n + w)
  rw [GameConfig.standard_cols] at h1 h2
  have hcap := count_lt_two_hundred_one hv hlive
  omega

/-- Contrapositive: overrunning the runway is death. -/
theorem lost_of_runway_overrun {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
          = cleared GameConfig.standard π GameState.init n)
    (hover : 200 < (trace GameConfig.standard π GameState.init n).board.count + 4 * w) :
    (trace GameConfig.standard π GameState.init (n + w)).lost GameConfig.standard := by
  by_contra hlive
  have := dry_runway_le hv hdry hlive
  omega

/-- **The clearing obligation.** Surviving the next `w` placements from a board
holding `D` cells requires clearing at least `(4w + D − 200)/10` rows in that
window. A solver can evaluate this for any horizon from its current state alone
and prune every search branch that cannot meet it — a sound pruning rule, not a
heuristic one. -/
theorem window_clears_ge_of_count {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    4 * w + (trace GameConfig.standard π GameState.init n).board.count
      ≤ 10 * (cleared GameConfig.standard π GameState.init (n + w)
              - cleared GameConfig.standard π GameState.init n) + 200 := by
  have h1 := init_ledger hv n
  have h2 := init_ledger hv (n + w)
  rw [GameConfig.standard_cols] at h1 h2
  have hcap := count_lt_two_hundred_one hv hlive
  have hm := cleared_mono GameConfig.standard π GameState.init (Nat.le_add_right n w)
  omega

/-! ## The state carries its own phase -/

/-- **A state knows where it is in the 35-piece cycle.** The board's cell count
fixes the piece count mod 5 and the bag's size fixes it mod 7, so `(board, bag)`
determines the phase mod 35 — the solver never has to store it, and two states
at different phases can never be equal.

For cycle search this is the operational form: **only states `35k` placements
apart can possibly coincide.** Comparing states at any other separation is
wasted work. -/
theorem phase_mod_thirtyfive_of_trace_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ n, (π (trace GameConfig.standard π GameState.init n)).piece
      ∈ (trace GameConfig.standard π GameState.init n).bag)
    {n₁ n₂ : ℕ}
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    n₁ % 35 = n₂ % 35 := by
  rcases le_total n₁ n₂ with h12 | h21
  · exact (Nat.modEq_iff_dvd' h12).mpr (thirtyfive_dvd_of_trace_eq hv hdraw h12 h)
  · exact ((Nat.modEq_iff_dvd' h21).mpr
      (thirtyfive_dvd_of_trace_eq hv hdraw h21 h.symm)).symm

/-! ## From arbitrary starts: the quantum applies to any closed cycle

The results above start from `init`. A cycle certificate need not: it is entered
somewhere in the middle of a game. These variants take an arbitrary start and
only assume the placements actually played are legal, which is exactly what a
`ClosedCycle` supplies. -/

/-- Trace-local well-formedness: only the placements actually played need to be
in-bounds. -/
theorem trace_board_wf_of_trace {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hv : ∀ n, (π (trace cfg π g0 n)).Valid cfg) (hwf : Board.WF cfg g0.board) (n : ℕ) :
    Board.WF cfg (trace cfg π g0 n).board := by
  induction n with
  | zero => simpa using hwf
  | succ k ih =>
    rw [trace_succ, GameState.step_board]
    exact Placement.applyStep_wf ih (hv k)

/-- The mass ledger from an arbitrary start, with trace-local legality. -/
theorem mass_ledger_of_trace {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hv : ∀ n, (π (trace cfg π g0 n)).Valid cfg) (hwf : Board.WF cfg g0.board) (n : ℕ) :
    (trace cfg π g0 n).board.count + cfg.cols * cleared cfg π g0 n
      = g0.board.count + 4 * n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := BagGrowth.count_applyStep_add
      (trace_board_wf_of_trace hv hwf k) (hv k)
    rw [trace_succ, GameState.step_board, cleared_succ, Nat.mul_add]
    omega

/-- Equal occupancy still forces `5 ∣ Δn` from an arbitrary start. -/
theorem five_dvd_of_count_eq_from {π : Policy GameConfig.standard} {g0 : GameState}
    (hv : ∀ n, (π (trace GameConfig.standard π g0 n)).Valid GameConfig.standard)
    (hwf : Board.WF GameConfig.standard g0.board) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (hc : (trace GameConfig.standard π g0 n₁).board.count
        = (trace GameConfig.standard π g0 n₂).board.count) :
    5 ∣ (n₂ - n₁) := by
  have h1 := mass_ledger_of_trace hv hwf n₁
  have h2 := mass_ledger_of_trace hv hwf n₂
  rw [GameConfig.standard_cols] at h1 h2
  have hm := cleared_mono GameConfig.standard π g0 h12
  omega

/-- **The bag clock from an arbitrary start.** Legal draws cycle the bag size
through `7, 6, …, 1, 7, …`, so from a start holding `c₀` pieces the size after
`n` placements is `7 − (7 − c₀ + n) mod 7`. -/
theorem bag_card_trace_from {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hdraw : ∀ n, (π (trace cfg π g0 n)).piece ∈ (trace cfg π g0 n).bag) (n : ℕ) :
    (trace cfg π g0 n).bag.card = 7 - ((7 - g0.bag.card) + n) % 7 := by
  have hle : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  induction n with
  | zero =>
    simp only [trace_zero, Nat.add_zero]
    omega
  | succ k ih =>
    have hmod : ((7 - g0.bag.card) + k) % 7 < 7 := Nat.mod_lt _ (by omega)
    rw [trace_succ, GameState.step_bag, card_draw (hdraw k), ih]
    split <;> omega

/-- Equal bag sizes force equal piece counts mod 7. -/
theorem seven_mod_eq_of_bag_card_eq {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hdraw : ∀ n, (π (trace cfg π g0 n)).piece ∈ (trace cfg π g0 n).bag)
    {n₁ n₂ : ℕ}
    (h : (trace cfg π g0 n₁).bag.card = (trace cfg π g0 n₂).bag.card) :
    n₁ % 7 = n₂ % 7 := by
  have hle : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  have h1 := bag_card_trace_from hdraw n₁
  have h2 := bag_card_trace_from hdraw n₂
  have hm1 : ((7 - g0.bag.card) + n₁) % 7 < 7 := Nat.mod_lt _ (by omega)
  have hm2 : ((7 - g0.bag.card) + n₂) % 7 < 7 := Nat.mod_lt _ (by omega)
  rw [h1, h2] at h
  omega

/-- **The 5-bag quantum, from any start.** Any legal trace that revisits a state
does so after a multiple of 35 placements — regardless of where it began. -/
theorem thirtyfive_dvd_of_trace_eq_from {π : Policy GameConfig.standard}
    {g0 : GameState}
    (hv : ∀ n, (π (trace GameConfig.standard π g0 n)).Valid GameConfig.standard)
    (hwf : Board.WF GameConfig.standard g0.board)
    (hdraw : ∀ n, (π (trace GameConfig.standard π g0 n)).piece
      ∈ (trace GameConfig.standard π g0 n).bag)
    {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard π g0 n₁ = trace GameConfig.standard π g0 n₂) :
    35 ∣ (n₂ - n₁) := by
  have h5 : 5 ∣ (n₂ - n₁) := five_dvd_of_count_eq_from hv hwf h12 (by rw [h])
  have h7 : n₁ % 7 = n₂ % 7 := seven_mod_eq_of_bag_card_eq hdraw (by rw [h])
  omega

/-- **The M2 certificate is quantised.** A closed cycle's trace can only return
to a state it has already visited after a multiple of **35 placements = 5
bags**. Cycle search over a `ClosedCycle` therefore never needs to test
separations that are not multiples of 35, and no certificate shorter than 35
placements exists. -/
theorem closedCycle_thirtyfive_dvd
    (C : ClosedCycle GameConfig.standard) {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard C.policy g0 n₁
        = trace GameConfig.standard C.policy g0 n₂) :
    35 ∣ (n₂ - n₁) :=
  thirtyfive_dvd_of_trace_eq_from
    (fun n => C.valid _ (C.trace_mem_states h0 n)) hwf
    (fun n => C.legal_draw _ (C.trace_mem_states h0 n)) h12 h

/-- A closed cycle's nontrivial period is at least 5 bags. -/
theorem closedCycle_thirtyfive_le
    (C : ClosedCycle GameConfig.standard) {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) {n₁ n₂ : ℕ} (h12 : n₁ < n₂)
    (h : trace GameConfig.standard C.policy g0 n₁
        = trace GameConfig.standard C.policy g0 n₂) :
    n₁ + 35 ≤ n₂ := by
  have hd := closedCycle_thirtyfive_dvd C h0 hwf (le_of_lt h12) h
  omega

/-! ## The sharpened recurrence gap -/

/-- **Exact balance recurs within 105 placements.** Sharpening
`exists_count_eq_le`: checkpoints spaced 5 apart all carry the *same* occupancy
residue (`count_mod_ten`), so only 21 cell counts are available to them and 22
such checkpoints must repeat. The guaranteed gap drops from 201 placements to
`5 · 21 = 105`. -/
theorem exists_count_eq_le_of_step_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) (a : ℕ) :
    ∃ i j, i < j ∧ j ≤ 21 ∧
      (trace GameConfig.standard π GameState.init (a + 5 * i)).board.count
        = (trace GameConfig.standard π GameState.init (a + 5 * j)).board.count := by
  have hres : ∀ i : ℕ,
      (trace GameConfig.standard π GameState.init (a + 5 * i)).board.count % 10
        = (4 * a) % 10 := by
    intro i
    rw [count_mod_ten hv]
    omega
  have hmaps : ∀ i ∈ Finset.range 22,
      (trace GameConfig.standard π GameState.init (a + 5 * i)).board.count / 10
        ∈ Finset.range 21 := by
    intro i _
    have hb := count_lt_two_hundred_one hv (hsurv (a + 5 * i))
    exact Finset.mem_range.mpr (by omega)
  have hcard : (Finset.range 21).card < (Finset.range 22).card := by simp
  obtain ⟨x, hx, y, hy, hxy, hfxy⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmaps
  have heq : (trace GameConfig.standard π GameState.init (a + 5 * x)).board.count
      = (trace GameConfig.standard π GameState.init (a + 5 * y)).board.count := by
    have h1 := hres x
    have h2 := hres y
    omega
  rcases lt_or_gt_of_ne hxy with h | h
  · exact ⟨x, y, h, by have := Finset.mem_range.mp hy; omega, heq⟩
  · exact ⟨y, x, h, by have := Finset.mem_range.mp hx; omega, heq.symm⟩

/-- **Every closed cycle holds at least 35 states.** The first 35 trace states
from any cycle member are pairwise distinct — a coincidence at distance below
35 would violate the cycle quantum — and all of them lie in the cycle. The
counting lower bound on the M2 artifact: no certificate smaller than five bags
of states exists. -/
theorem closedCycle_card_ge_thirtyfive (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) :
    35 ≤ C.states.card := by
  have hcalc : (Finset.range 35).card ≤ C.states.card := by
    refine Finset.card_le_card_of_injOn
      (fun i => trace GameConfig.standard C.policy g0 i) ?_ ?_
    · intro i _
      exact C.trace_mem_states h0 i
    · intro i hi j hj hEq
      simp only [Finset.coe_range, Set.mem_Iio] at hi hj
      dsimp only at hEq
      rcases le_total i j with h | h
      · have := closedCycle_thirtyfive_dvd C h0 hwf h hEq
        omega
      · have := closedCycle_thirtyfive_dvd C h0 hwf h hEq.symm
        omega
  rwa [Finset.card_range] at hcalc

/-- Any closed cycle through the initial state holds at least 35 states: the
M3 artifact (a reachable cycle seeded at `init`) is never smaller than five
bags of states. -/
theorem init_closedCycle_card_ge_thirtyfive (C : ClosedCycle GameConfig.standard)
    (h0 : GameState.init ∈ C.states) : 35 ≤ C.states.card :=
  closedCycle_card_ge_thirtyfive C h0 (GameState.init_board_wf GameConfig.standard)

/-- **A cycle window clears exactly 2.8 per bag — no error.** Between two
visits to the same state the ledger balances exactly: `10·Δcleared = 4·Δn`.
With the quantum `Δn = 35k`, every closed-cycle period clears **exactly `14k`
rows** — the minimal five-bag cycle clears exactly 14. -/
theorem trace_eq_clears {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂) :
    10 * (cleared GameConfig.standard π GameState.init n₂
          - cleared GameConfig.standard π GameState.init n₁)
      = 4 * (n₂ - n₁) :=
  exact_balance_of_count_eq hv h12 (by rw [h])

/-- The minimal-period form: a 35-placement return clears exactly 14 rows. -/
theorem trace_eq_thirtyfive_clears_fourteen {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (h : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    cleared GameConfig.standard π GameState.init (n + 35)
      - cleared GameConfig.standard π GameState.init n = 14 := by
  have hbal := trace_eq_clears hv (Nat.le_add_right n 35) h
  omega

/-- **The linear clearing law on cycles**: one 35-return pins every horizon —
`j` periods clear exactly `14·j` rows. Periodicity iterates
(`trace_period_multiples`) and the ledger balances exactly on each return. -/
theorem multi_period_clears {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    cleared GameConfig.standard π GameState.init (n + 35 * j)
      - cleared GameConfig.standard π GameState.init n = 14 * j := by
  have hiter := trace_period_multiples π GameState.init hcyc j
  rw [show n + j * 35 = n + 35 * j by ring] at hiter
  have hbal := trace_eq_clears hv (Nat.le_add_right n (35 * j)) hiter
  omega

/-- **The clearing bracket at every horizon**: on a cycle, the cleared count
never strays more than fourteen rows from the linear 2.8-per-bag law —
`14·⌊(m−n)/35⌋ ≤ Δcleared ≤ 14·⌊(m−n)/35⌋ + 14` for every `m ≥ n`. Squeeze
between the two enclosing period boundaries; sharper on cycles than the
general 20-row deviation budget. -/
theorem cycle_clears_bracket {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m : ℕ}
    (hnm : n ≤ m) :
    14 * ((m - n) / 35)
        ≤ cleared GameConfig.standard π GameState.init m
          - cleared GameConfig.standard π GameState.init n
      ∧ cleared GameConfig.standard π GameState.init m
          - cleared GameConfig.standard π GameState.init n
        ≤ 14 * ((m - n) / 35) + 14 := by
  set j := (m - n) / 35 with hj
  have hlo : n + 35 * j ≤ m := by omega
  have hhi : m ≤ n + 35 * (j + 1) := by omega
  have hjlaw := multi_period_clears hv hcyc j
  have hjlaw' := multi_period_clears hv hcyc (j + 1)
  have hm1 := cleared_mono GameConfig.standard π GameState.init hlo
  have hm2 := cleared_mono GameConfig.standard π GameState.init hhi
  have hm0 := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n (35 * j))
  constructor
  · omega
  · omega

/-- **The mass band**: on a cycle the board occupancy is trapped within a
fourteen-row band of its boundary value at *every* horizon —
`count(n) − 140 ≤ count(m) ≤ count(n) + 136`. The ledger converts the
clearing bracket into a mass bracket: between period boundaries at most
34 placements (136 cells) can accumulate, and at most one period's worth of
clearing (140 cells) can drain. -/
theorem cycle_mass_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m : ℕ}
    (hnm : n ≤ m) :
    (trace GameConfig.standard π GameState.init m).board.count
        ≤ (trace GameConfig.standard π GameState.init n).board.count + 136
      ∧ (trace GameConfig.standard π GameState.init n).board.count
        ≤ (trace GameConfig.standard π GameState.init m).board.count + 140 := by
  have hln := init_ledger hv n
  have hlm := init_ledger hv m
  rw [GameConfig.standard_cols] at hln hlm
  obtain ⟨hlo, hhi⟩ := cycle_clears_bracket hv hcyc hnm
  have hclm := cleared_mono GameConfig.standard π GameState.init hnm
  have hdiv : 35 * ((m - n) / 35) ≤ m - n ∧ m - n < 35 * ((m - n) / 35) + 35 := by
    omega
  omega

/-- Board occupancy is exactly periodic at cycle boundaries: `j` periods
return the mass to its boundary value. -/
theorem cycle_mass_periodic {π : Policy GameConfig.standard} {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    (trace GameConfig.standard π GameState.init (n + 35 * j)).board.count
      = (trace GameConfig.standard π GameState.init n).board.count := by
  have hiter := trace_period_multiples π GameState.init hcyc j
  rw [show n + 35 * j = n + j * 35 by ring, ← hiter]

/-- **Every 69-window on a cycle clears (at least fourteen rows)**: any such
window contains a complete aligned period, and every period clears exactly
14. The window bound is tight in form: an unaligned start can waste up to 34
placements before the next period boundary. -/
theorem cycle_window_clears_fourteen {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) :
    14 ≤ cleared GameConfig.standard π GameState.init (m₀ + 69)
      - cleared GameConfig.standard π GameState.init m₀ := by
  set i := (m₀ - n + 34) / 35 with hi
  have hb1 : m₀ ≤ n + 35 * i := by omega
  have hb2 : n + 35 * i ≤ m₀ + 34 := by omega
  have hlaw1 := multi_period_clears hv hcyc i
  have hlaw2 := multi_period_clears hv hcyc (i + 1)
  have hmono0 := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n (35 * i))
  have hmono1 := cleared_mono GameConfig.standard π GameState.init hb1
  have hmono2 := cleared_mono GameConfig.standard π GameState.init
    (show n + 35 * (i + 1) ≤ m₀ + 69 by omega)
  omega

/-- **Dry spells on a cycle last at most 68 placements**: a clear-free
stretch of 69 would contain a full period, which must clear. Far tighter
than the general 50-placement clear-free horizon from capacity — and it
applies at every point of the cycle forever, not just from the empty
board. -/
theorem cycle_dry_spell_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ L : ℕ}
    (hm : n ≤ m₀)
    (hdry : cleared GameConfig.standard π GameState.init (m₀ + L)
      = cleared GameConfig.standard π GameState.init m₀) :
    L ≤ 68 := by
  by_contra hcon
  have h69 := cycle_window_clears_fourteen hv hcyc hm
  have hmono := cleared_mono GameConfig.standard π GameState.init
    (show m₀ + 69 ≤ m₀ + L by omega)
  have hmono0 := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right m₀ 69)
  omega

/-- **A heavy cycle keeps a tall column forever**: if the cycle's boundary
board carries more than `140 + 10·H` cells, then at *every* horizon some
column rises strictly above `H` — the mass band caps the drainage at 140
cells, and mass needs volume. A cycle cannot alternately flatten and
rebuild past this floor. -/
theorem cycle_height_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {H m : ℕ}
    (hnm : n ≤ m)
    (hheavy : 140 + 10 * H
      < (trace GameConfig.standard π GameState.init n).board.count) :
    H < Board.maxHeight GameConfig.standard
      (trace GameConfig.standard π GameState.init m).board := by
  obtain ⟨-, hlo⟩ := cycle_mass_band hv hcyc hnm
  have hwf := trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m
  apply Board.lt_maxHeight_of_cols_mul_lt_count hwf
  rw [GameConfig.standard_cols]
  omega

/-! ## The clear-free horizon is fifty placements -/

/-- **Clear-free survival ends by placement fifty.** With no rows cleared the
delivered mass sits on the board in full, and the board holds 200 cells: a
live clear-free trace has `4n ≤ 200`. Any safety certificate of depth 51 or
more must therefore include line clears — the exact point where the
headroom/packing schedule family (`HeadroomIterate`) provably cannot reach. -/
theorem clear_free_le_fifty {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard)
    (hnc : cleared GameConfig.standard π GameState.init n = 0) :
    n ≤ 50 := by
  have h := init_ledger hv n
  rw [GameConfig.standard_cols, hnc] at h
  have hcap := count_lt_two_hundred_one hv hlive
  omega

/-- Config-generic form: clear-free survival is bounded by a quarter of the
playfield capacity, whatever the board dimensions. -/
theorem clear_free_le_capacity {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {n : ℕ}
    (hlive : ¬ (trace cfg π GameState.init n).lost cfg)
    (hnc : cleared cfg π GameState.init n = 0) :
    4 * n ≤ cfg.cols * cfg.rows := by
  have h := init_ledger hv n
  rw [hnc] at h
  have hcap := BagGrowth.count_le_capacity
    (trace_board_wf hv (GameState.init_board_wf cfg) n)
    ((GameState.not_lost_iff_forall_row_lt cfg _).mp hlive)
  omega

/-- **The first clear arrives by placement fifty-one.** Any surviving policy
has cleared at least one row within its first 51 placements. -/
theorem first_clear_by_fiftyone {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ} (hn : 51 ≤ n)
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    0 < cleared GameConfig.standard π GameState.init n := by
  by_contra hc
  have hz : cleared GameConfig.standard π GameState.init n = 0 := by omega
  have := clear_free_le_fifty hv hlive hz
  omega

end ClearRate
end Tetris
