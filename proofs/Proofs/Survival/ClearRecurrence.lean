import Mathlib
import Proofs.Survival.ClearDeviation
import Proofs.Invariants.ColumnCount

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

/-- **The stationary clearing bracket**: at *every* position of a cycle and
*every* window length, the rows cleared track the 2.8-per-bag line —
`14·⌊(w−34)/35⌋ ≤ Δcleared ≤ 14·⌊w/35⌋ + 28`. The lower bound counts the
full aligned periods the window must contain; the upper bound subtracts two
boundary brackets. Shift-invariant: no alignment with the cycle's entry
point is assumed. -/
theorem cycle_clears_stationary_bracket {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) (w : ℕ) :
    14 * ((w - 34) / 35)
        ≤ cleared GameConfig.standard π GameState.init (m₀ + w)
          - cleared GameConfig.standard π GameState.init m₀
      ∧ cleared GameConfig.standard π GameState.init (m₀ + w)
          - cleared GameConfig.standard π GameState.init m₀
        ≤ 14 * (w / 35) + 28 := by
  constructor
  · -- lower: the window contains ⌊(w−34)/35⌋ full aligned periods
    rcases Nat.lt_or_ge w 34 with hw | hw
    · have hz : (w - 34) / 35 = 0 := by omega
      omega
    set i := (m₀ - n + 34) / 35 with hi
    set k := (w - 34) / 35 with hk
    have hb1 : m₀ ≤ n + 35 * i := by omega
    have hb2 : n + 35 * i ≤ m₀ + 34 := by omega
    have hlaw1 := multi_period_clears hv hcyc i
    have hlaw2 := multi_period_clears hv hcyc (i + k)
    have hmono1 := cleared_mono GameConfig.standard π GameState.init hb1
    have hmono2 := cleared_mono GameConfig.standard π GameState.init
      (show n + 35 * (i + k) ≤ m₀ + w by omega)
    have hmono0 := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right n (35 * i))
    omega
  · -- upper: subtract the two boundary brackets
    obtain ⟨hlo1, hhi1⟩ := cycle_clears_bracket hv hcyc
      (show n ≤ m₀ + w by omega)
    obtain ⟨hlo2, hhi2⟩ := cycle_clears_bracket hv hcyc hm
    have hmono := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right m₀ w)
    have hmono' := cleared_mono GameConfig.standard π GameState.init hm
    omega

/-- **The mass diameter**: any two states on a cycle differ by at most 276
cells of occupancy — both sit inside the `[count(n) − 140, count(n) + 136]`
band. The whole cycle lives in a 276-cell (≈ 28-row) occupancy corridor. -/
theorem cycle_mass_diameter {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₁ m₂ : ℕ}
    (h1 : n ≤ m₁) (h2 : n ≤ m₂) :
    (trace GameConfig.standard π GameState.init m₁).board.count
      ≤ (trace GameConfig.standard π GameState.init m₂).board.count + 276 := by
  obtain ⟨hup1, hlo1⟩ := cycle_mass_band hv hcyc h1
  obtain ⟨hup2, hlo2⟩ := cycle_mass_band hv hcyc h2
  omega

/-- **Per-step mass conservation**: one placement adds four cells and a
`k`-row clear removes `10k` — `count(m+1) + 10·size(m) = count(m) + 4`
exactly, at every step. The occupancy trajectory moves by
`+4, −6, −16, −26, −36` according to the clear size. -/
theorem count_step_eq {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    (trace GameConfig.standard π GameState.init (m + 1)).board.count
        + 10 * (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init m)).place
            (trace GameConfig.standard π GameState.init m).board)).card
      = (trace GameConfig.standard π GameState.init m).board.count + 4 := by
  have h1 := init_ledger hv m
  have h2 := init_ledger hv (m + 1)
  have hc := cleared_succ GameConfig.standard π GameState.init m
  rw [GameConfig.standard_cols] at h1 h2
  omega

/-- **Clearing needs standing mass**: a `k`-row clear at step `m` requires at
least `10k − 4` cells already on the board — the piece brings only four. In
particular a tetris needs **36 standing cells**: no board lighter than 36
cells can host a four-clear. -/
theorem clear_requires_mass {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    10 * (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card
      ≤ (trace GameConfig.standard π GameState.init m).board.count + 4 := by
  have h := count_step_eq hv m
  omega

/-- A tetris step stands on at least 36 cells. -/
theorem tetris_requires_thirtysix {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    36 ≤ (trace GameConfig.standard π GameState.init m).board.count := by
  have h := clear_requires_mass hv m
  omega

/-- Cells delivered to column `j` over the first `n` placements. -/
def colDelivered (π : Policy GameConfig.standard) (j : ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => colDelivered π j n
      + (π (trace GameConfig.standard π GameState.init n)).colProfile j

@[simp] theorem colDelivered_zero (π : Policy GameConfig.standard) (j : ℕ) :
    colDelivered π j 0 = 0 := rfl

theorem colDelivered_succ (π : Policy GameConfig.standard) (j n : ℕ) :
    colDelivered π j (n + 1) = colDelivered π j n
      + (π (trace GameConfig.standard π GameState.init n)).colProfile j := rfl

/-- **The per-column ledger**: what column `j` holds plus what it lost to
clears equals what it received — clears bill every column exactly one cell
per row. -/
theorem colDelivered_ledger {π : Policy GameConfig.standard} {j : ℕ}
    (hj : j < 10) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).board.colCount j
        + cleared GameConfig.standard π GameState.init n
      = colDelivered π j n := by
  induction n with
  | zero => simp [cleared, Board.colCount, GameState.init, Board.empty]
  | succ k ih =>
    have hstep := applyStep_colCount GameConfig.standard
      (trace GameConfig.standard π GameState.init k).board
      (π (trace GameConfig.standard π GameState.init k))
      (j := j) (by rw [GameConfig.standard_cols]; omega)
    rw [trace_succ, GameState.step_board, cleared_succ, colDelivered_succ]
    unfold Board.linesCleared at hstep
    omega

/-- A column of an in-field board holds at most `rows` cells (its cells sit
at distinct rows below the ceiling). -/
theorem colCount_le_rows {cfg : GameConfig} {b : Board}
    (hif : ∀ p ∈ b, p.2 < cfg.rows) (j : ℕ) :
    b.colCount j ≤ cfg.rows := by
  classical
  unfold Board.colCount
  calc (b.filter (fun p => p.1 = j)).card
      ≤ (Finset.range cfg.rows).card := by
        refine Finset.card_le_card_of_injOn (fun p => p.2) ?_ ?_
        · intro p hp
          simp only [Finset.mem_coe, Finset.mem_filter] at hp
          exact Finset.mem_range.mpr (hif p hp.1)
        · intro p hp q hq hpq
          simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
          apply Prod.ext
          · rw [hp.2, hq.2]
          · exact hpq
    _ = cfg.rows := Finset.card_range _

/-- **The load-distribution law**: on a live trace, every column has received
between `cleared` and `cleared + 20` cells — the clearing duty is billed to
all ten columns equally, up to one board-height of slack. -/
theorem column_load_bracket {π : Policy GameConfig.standard}
    {j : ℕ} (hj : j < 10) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    cleared GameConfig.standard π GameState.init n ≤ colDelivered π j n
      ∧ colDelivered π j n
        ≤ cleared GameConfig.standard π GameState.init n + 20 := by
  have hled := colDelivered_ledger (π := π) hj n
  have hif : ∀ p ∈ (trace GameConfig.standard π GameState.init n).board,
      p.2 < GameConfig.standard.rows :=
    (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive
  have hcc := colCount_le_rows hif j
  rw [GameConfig.standard_rows] at hcc
  exact ⟨by omega, by omega⟩

/-- **Column-pair balance**: on a live trace, any two columns' cumulative
deliveries differ by at most twenty cells. -/
theorem column_pair_balance {π : Policy GameConfig.standard}
    {j j' : ℕ} (hj : j < 10) (hj' : j' < 10) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    colDelivered π j n ≤ colDelivered π j' n + 20 := by
  obtain ⟨h1, h2⟩ := column_load_bracket (π := π) hj hlive
  obtain ⟨h1', h2'⟩ := column_load_bracket (π := π) hj' hlive
  omega

/-- **Every column receives exactly `14·k` cells per `k` cycle periods**:
the column's holdings return with the state, so its intake equals its
clearing bill exactly — the 140-cells-per-period total splits as
`10 × 14`, column by column. -/
theorem cycle_column_load_exact {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (k : ℕ) :
    colDelivered π j (n + 35 * k) - colDelivered π j n = 14 * k := by
  have hled1 := colDelivered_ledger (π := π) hj n
  have hled2 := colDelivered_ledger (π := π) hj (n + 35 * k)
  have hiter := trace_period_multiples π GameState.init hcyc k
  rw [show n + k * 35 = n + 35 * k by ring] at hiter
  have hcol : (trace GameConfig.standard π GameState.init
        (n + 35 * k)).board.colCount j
      = (trace GameConfig.standard π GameState.init n).board.colCount j := by
    rw [← hiter]
  have hcl := multi_period_clears hv hcyc k
  have hclm := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n (35 * k))
  omega

/-- Column deliveries never decrease. -/
theorem colDelivered_mono (π : Policy GameConfig.standard) (j : ℕ) :
    Monotone (colDelivered π j) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [colDelivered_succ]
  exact Nat.le_add_right _ _

/-- **The per-column frequency law on cycles**: every window of a cycle
delivers between `14⌊w/35⌋` and `14⌊w/35⌋ + 14` cells to every column —
each column's intake runs at exactly `0.4` cells per placement with at most
one period of slack, at every position and scale. -/
theorem cycle_column_window_bracket {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) (w : ℕ) :
    14 * (w / 35) ≤ colDelivered π j (m₀ + w) - colDelivered π j m₀
      ∧ colDelivered π j (m₀ + w) - colDelivered π j m₀
        ≤ 14 * (w / 35) + 14 := by
  have hanchor : trace GameConfig.standard π GameState.init m₀
      = trace GameConfig.standard π GameState.init (m₀ + 35) := by
    have h := trace_eq_of_state_eq π GameState.init hcyc (m₀ - n)
    rw [show n + (m₀ - n) = m₀ by omega] at h
    rw [show n + 35 + (m₀ - n) = m₀ + 35 by omega] at h
    exact h
  set q := w / 35 with hq
  have hexq := cycle_column_load_exact hv hj hanchor q
  have hexq1 := cycle_column_load_exact hv hj hanchor (q + 1)
  have hm1 := colDelivered_mono π j (Nat.le_add_right m₀ (35 * q))
  have hm2 := colDelivered_mono π j
    (show m₀ + 35 * q ≤ m₀ + w by omega)
  have hm3 := colDelivered_mono π j
    (show m₀ + w ≤ m₀ + 35 * (q + 1) by omega)
  exact ⟨by omega, by omega⟩

/-- **The column ledgers sum to the global ledger**: total deliveries across
the ten columns equal the four cells of every placement — the ten column
brackets are a decomposition of mass conservation, not new information
piled on top. -/
theorem sum_colDelivered {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    ∑ j ∈ Finset.range 10, colDelivered π j n = 4 * n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hsum : ∑ j ∈ Finset.range 10,
        (π (trace GameConfig.standard π GameState.init k)).colProfile j = 4 := by
      have h := Placement.sum_colProfile
        (hv (trace GameConfig.standard π GameState.init k))
      rwa [GameConfig.standard_cols] at h
    calc ∑ j ∈ Finset.range 10, colDelivered π j (k + 1)
        = ∑ j ∈ Finset.range 10, (colDelivered π j k
            + (π (trace GameConfig.standard π GameState.init k)).colProfile j)
          := by
          apply Finset.sum_congr rfl
          intro j _
          rw [colDelivered_succ]
      _ = (∑ j ∈ Finset.range 10, colDelivered π j k)
            + ∑ j ∈ Finset.range 10,
              (π (trace GameConfig.standard π GameState.init k)).colProfile j
          := Finset.sum_add_distrib
      _ = 4 * k + 4 := by rw [ih, hsum]
      _ = 4 * (k + 1) := by ring

/-- The windowed column intake is the sum of the window's profiles. -/
theorem colDelivered_window (π : Policy GameConfig.standard) (j n : ℕ) :
    ∀ w, colDelivered π j (n + w) - colDelivered π j n
      = ∑ k ∈ Finset.range w,
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j := by
  intro w
  induction w with
  | zero => simp
  | succ w ih =>
    have hmono := colDelivered_mono π j (Nat.le_add_right n w)
    rw [show n + (w + 1) = (n + w) + 1 by omega, colDelivered_succ,
      Finset.sum_range_succ, ← ih]
    omega

/-- **The tall-drop cap**: per cycle period, at most three placements pour
their full four cells into any one fixed column — a column's 14-cell period
budget cannot absorb a fourth vertical I. -/
theorem cycle_tall_drop_column_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j
          = 4)).card ≤ 3 := by
  classical
  have hload := cycle_column_load_exact hv hj hcyc 1
  rw [mul_one] at hload
  have hwin := colDelivered_window π j n 35
  have hsum : 4 * ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j
          = 4)).card
      ≤ ∑ k ∈ Finset.range 35,
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j := by
    calc 4 * ((Finset.range 35).filter (fun k =>
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j
            = 4)).card
        = ∑ k ∈ (Finset.range 35).filter (fun k =>
            (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
              j = 4),
            (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
              j := by
          rw [Finset.sum_congr rfl (fun k hk =>
            (Finset.mem_filter.mp hk).2)]
          rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ _ := Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  omega

/-- **The general column-profile cap**: per cycle period, placements
delivering at least `p ≥ 1` cells into a fixed column number at most
`⌊14/p⌋` — heavy feeders of any one column are rationed by its exact
period budget (`p = 4`: ≤ 3 tall drops; `p = 3`: ≤ 4; `p = 2`: ≤ 7). -/
theorem cycle_column_profile_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35))
    {p : ℕ} (hp : 1 ≤ p) :
    ((Finset.range 35).filter (fun k =>
        p ≤ (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
          j)).card ≤ 14 / p := by
  classical
  have hload := cycle_column_load_exact hv hj hcyc 1
  rw [mul_one] at hload
  have hwin := colDelivered_window π j n 35
  have hsum : p * ((Finset.range 35).filter (fun k =>
        p ≤ (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
          j)).card
      ≤ ∑ k ∈ Finset.range 35,
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
            j := by
    calc p * ((Finset.range 35).filter (fun k =>
          p ≤ (π (trace GameConfig.standard π GameState.init
            (n + k))).colProfile j)).card
        = ∑ _k ∈ (Finset.range 35).filter (fun k =>
            p ≤ (π (trace GameConfig.standard π GameState.init
              (n + k))).colProfile j), p := by
          rw [Finset.sum_const, smul_eq_mul, mul_comm]
      _ ≤ ∑ k ∈ (Finset.range 35).filter (fun k =>
            p ≤ (π (trace GameConfig.standard π GameState.init
              (n + k))).colProfile j),
            (π (trace GameConfig.standard π GameState.init
              (n + k))).colProfile j :=
          Finset.sum_le_sum (fun k hk => (Finset.mem_filter.mp hk).2)
      _ ≤ _ := Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  have hS : (∑ k ∈ Finset.range 35,
      (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j)
      = 14 := by omega
  have hmul := hsum.trans (le_of_eq hS)
  exact (Nat.le_div_iff_mul_le hp).mpr (by rw [mul_comm]; exact hmul)

/-- **A cleared row was already nearly full**: any row completed by a
placement held at least `cols − 4` cells before the piece arrived — the
piece brings four cells at most to one row. -/
theorem cleared_row_pre_count_ge {cfg : GameConfig} {b : Board}
    {pl : Placement} {r : ℕ}
    (hr : r ∈ Board.fullRows cfg (pl.place b)) :
    cfg.cols ≤ b.rowCount r + 4 := by
  by_contra hcon
  push Not at hcon
  have hlt : Board.rowCount (pl.place b) r < cfg.cols := by
    have := Board.rowCount_place_le b pl r
    omega
  exact Board.not_isFull_of_rowCount_lt cfg (pl.place b) r hlt
    (Board.isFull_of_mem_fullRows hr)

/-- At standard width: **every cleared row was at least six-tenths full**
before the finishing piece — clears must be prepared, never improvised. -/
theorem cleared_row_was_six_tenths {π : Policy GameConfig.standard} {m r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)) :
    6 ≤ (trace GameConfig.standard π GameState.init m).board.rowCount r := by
  have h := cleared_row_pre_count_ge (cfg := GameConfig.standard) hr
  rw [GameConfig.standard_cols] at h
  omega

/-- A full row of a well-formed board holds exactly `cols` cells. -/
theorem rowCount_of_isFull {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) {r : ℕ} (hfull : Board.isFull cfg b r) :
    b.rowCount r = cfg.cols := by
  classical
  apply le_antisymm
  · unfold Board.rowCount
    refine le_trans (Finset.card_le_card_of_injOn (fun p => p.1) ?_ ?_)
      (le_of_eq (Finset.card_range cfg.cols))
    · intro p hp
      simp only [Finset.mem_coe, Finset.mem_filter] at hp
      exact Finset.mem_range.mpr (hwf p hp.1)
    · intro p hp q hq hpq
      simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
      apply Prod.ext
      · exact hpq
      · rw [hp.2, hq.2]
  · unfold Board.rowCount
    have hsub : (Finset.range cfg.cols).image (fun c => ((c, r) : ℕ × ℕ))
        ⊆ b.filter (fun p => p.2 = r) := by
      intro p hp
      rw [Finset.mem_image] at hp
      obtain ⟨c, hc, rfl⟩ := hp
      rw [Finset.mem_range] at hc
      exact Finset.mem_filter.mpr ⟨Board.mem_of_isFull cfg hc hfull, rfl⟩
    have := Finset.card_le_card hsub
    rwa [Finset.card_image_of_injective _
      (fun a b hab => by simpa using hab), Finset.card_range] at this

/-- The piece's total contribution across any set of rows is at most four. -/
theorem sum_row_added_le_four (b : Board) (pl : Placement) (R : Finset ℕ) :
    ∑ r ∈ R, ((pl.dropped b).filter (fun p => p.2 = r)).card ≤ 4 := by
  classical
  have hdisj : ∀ r ∈ R, ∀ r' ∈ R, r ≠ r' →
      Disjoint ((pl.dropped b).filter (fun p => p.2 = r))
        ((pl.dropped b).filter (fun p => p.2 = r')) := by
    intro r _ r' _ hne
    rw [Finset.disjoint_left]
    intro p hp hp'
    rw [Finset.mem_filter] at hp hp'
    exact hne (by rw [← hp.2, hp'.2])
  calc ∑ r ∈ R, ((pl.dropped b).filter (fun p => p.2 = r)).card
      = (R.biUnion (fun r => (pl.dropped b).filter (fun p => p.2 = r))).card :=
        (Finset.card_biUnion hdisj).symm
    _ ≤ (pl.dropped b).card := by
        apply Finset.card_le_card
        intro p hp
        rw [Finset.mem_biUnion] at hp
        obtain ⟨r, -, hpr⟩ := hp
        exact (Finset.mem_filter.mp hpr).1
    _ = 4 := Placement.card_dropped b pl

/-- **The localized clear mass**: the rows a `k`-clear completes held at
least `10k − 4` cells between them before the piece landed — the clearing
mass must be standing in the *cleared rows themselves*, not merely
somewhere on the board. Localizes the 36-cell tetris floor. -/
theorem cleared_rows_pre_mass {cfg : GameConfig} {b : Board} {pl : Placement}
    (hwf : Board.WF cfg b) (hv : pl.Valid cfg) :
    cfg.cols * (Board.fullRows cfg (pl.place b)).card
      ≤ (∑ r ∈ Board.fullRows cfg (pl.place b), b.rowCount r) + 4 := by
  classical
  have hplacewf : Board.WF cfg (pl.place b) := Placement.place_wf hwf hv
  have hfullcnt : ∀ r ∈ Board.fullRows cfg (pl.place b),
      Board.rowCount (pl.place b) r = cfg.cols := by
    intro r hr
    exact rowCount_of_isFull hplacewf (Board.isFull_of_mem_fullRows hr)
  have hsplit : ∀ r, Board.rowCount (pl.place b) r
      ≤ b.rowCount r + ((pl.dropped b).filter (fun p => p.2 = r)).card := by
    intro r
    unfold Board.rowCount
    rw [Placement.place_eq_union_dropped, Finset.filter_union]
    exact Finset.card_union_le _ _
  have hsum : ∑ r ∈ Board.fullRows cfg (pl.place b),
      Board.rowCount (pl.place b) r
      ≤ (∑ r ∈ Board.fullRows cfg (pl.place b), b.rowCount r)
        + ∑ r ∈ Board.fullRows cfg (pl.place b),
          ((pl.dropped b).filter (fun p => p.2 = r)).card := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_le_sum (fun r _ => hsplit r)
  have hconst : ∑ r ∈ Board.fullRows cfg (pl.place b),
      Board.rowCount (pl.place b) r
      = cfg.cols * (Board.fullRows cfg (pl.place b)).card := by
    rw [Finset.sum_congr rfl hfullcnt, Finset.sum_const, smul_eq_mul, mul_comm]
  have hadd := sum_row_added_le_four b pl (Board.fullRows cfg (pl.place b))
  omega

/-- On a board with no pre-existing full rows, every row a placement
completes contains a cell of the piece itself. -/
theorem mem_fullRows_place_has_piece_cell {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r) {r : ℕ}
    (hr : r ∈ Board.fullRows cfg (pl.place b)) :
    ∃ q ∈ pl.dropped b, q.2 = r := by
  classical
  by_contra hcon
  push Not at hcon
  apply hnf r
  intro c hc
  have hfull := Board.isFull_of_mem_fullRows hr
  have hmem := hfull c hc
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
  rcases hmem with h | h
  · exact h
  · exact absurd rfl (hcon (c, r) h)

/-- **Clears are vertically local**: on a board with no pre-existing full
rows, any two rows completed by one placement lie within three of each
other — a single piece spans at most four rows, and every completed row
touches the piece. -/
theorem fullRows_place_span_le_three {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r) {r r' : ℕ}
    (hr : r ∈ Board.fullRows cfg (pl.place b))
    (hr' : r' ∈ Board.fullRows cfg (pl.place b)) (hle : r ≤ r') :
    r' - r ≤ 3 := by
  obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr
  obtain ⟨q', hq', hqr'⟩ := mem_fullRows_place_has_piece_cell hnf hr'
  rw [Placement.dropped_eq_image, Finset.mem_image] at hq hq'
  obtain ⟨cell, hcell, rfl⟩ := hq
  obtain ⟨cell', hcell', rfl⟩ := hq'
  have h1 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  have h2 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell' hcell'
  dsimp only at hqr hqr'
  omega

/-- **A tetris clears four consecutive rows**: when a placement completes
four rows of a no-full-rows board, they form exactly the interval
`[r₀, r₀ + 3]` — four distinct rows within a span of three have no other
shape. -/
theorem four_clear_rows_eq_Icc {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (h4 : (Board.fullRows cfg (pl.place b)).card = 4) :
    ∃ r₀, Board.fullRows cfg (pl.place b) = Finset.Icc r₀ (r₀ + 3) := by
  classical
  have hne : (Board.fullRows cfg (pl.place b)).Nonempty :=
    Finset.card_pos.mp (by omega)
  set r₀ := (Board.fullRows cfg (pl.place b)).min' hne with hr₀
  refine ⟨r₀, ?_⟩
  have hsub : Board.fullRows cfg (pl.place b) ⊆ Finset.Icc r₀ (r₀ + 3) := by
    intro r hr
    have hmin := Finset.min'_le _ r hr
    have hspan := fullRows_place_span_le_three hnf
      ((Board.fullRows cfg (pl.place b)).min'_mem hne) hr hmin
    rw [Finset.mem_Icc]
    omega
  apply Finset.eq_of_subset_of_card_le hsub
  rw [h4, Nat.card_Icc]
  omega

/-- **A tetris's finishing piece stands vertical**: a placement completing
four rows must have its four cells in four distinct rows — the shape spans
exactly four rows. With `tetris_requires_I`, the finisher is an I in its
vertical orientation. -/
theorem four_clear_piece_rows_card {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (h4 : (Board.fullRows cfg (pl.place b)).card = 4) :
    (pl.shapeUp.image (fun c => c.2)).card = 4 := by
  classical
  have hsub : Board.fullRows cfg (pl.place b)
      ⊆ (pl.dropped b).image (fun q => q.2) := by
    intro r hr
    obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr
    exact Finset.mem_image.mpr ⟨q, hq, hqr⟩
  have h1 : 4 ≤ ((pl.dropped b).image (fun q => q.2)).card := by
    have := Finset.card_le_card hsub
    omega
  have himg : (pl.dropped b).image (fun q => q.2)
      = (pl.shapeUp.image (fun c => c.2)).image
          (fun t => pl.dropOffset b + t) := by
    rw [Placement.dropped_eq_image, Finset.image_image, Finset.image_image]
    rfl
  have h3 : ((pl.dropped b).image (fun q => q.2)).card
      = ((pl.shapeUp).image (fun c => c.2)).card := by
    rw [himg]
    exact Finset.card_image_of_injective _ (fun a b hab => by omega)
  have h2 : ((pl.shapeUp).image (fun c => c.2)).card ≤ 4 := by
    refine le_trans Finset.card_image_le ?_
    exact le_of_eq pl.shapeUp_card
  omega

/-- Placing splits row counts exactly (the union is disjoint). -/
theorem rowCount_place_eq (b : Board) (pl : Placement) (r : ℕ) :
    Board.rowCount (pl.place b) r
      = b.rowCount r + ((pl.dropped b).filter (fun p => p.2 = r)).card := by
  classical
  unfold Board.rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union,
    Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (pl.dropped_disjoint b).symm)]

/-- **The tetris anatomy completes**: at a four-clear, each of the four
rows held *exactly nine* cells before the vertical I supplied its tenth —
the four fibers of the piece split one cell per row, forced by counting. -/
theorem tetris_rows_pre_nine {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4)
    {r : ℕ} (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    b.rowCount r = 9 := by
  classical
  have hplacewf := Placement.place_wf hwf hv
  have hten : Board.rowCount (pl.place b) r = 10 := by
    have h := rowCount_of_isFull hplacewf (Board.isFull_of_mem_fullRows hr)
    rwa [GameConfig.standard_cols] at h
  have hsplit := rowCount_place_eq b pl r
  -- each of the four completed rows takes at least one piece cell
  have hone : ∀ r' ∈ Board.fullRows GameConfig.standard (pl.place b),
      1 ≤ ((pl.dropped b).filter (fun p => p.2 = r')).card := by
    intro r' hr'
    obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr'
    exact Finset.card_pos.mpr ⟨q, Finset.mem_filter.mpr ⟨hq, hqr⟩⟩
  -- the row r takes at most one: the other three each take one of the four
  have herase : 3 ≤ ∑ r' ∈ (Board.fullRows GameConfig.standard
      (pl.place b)).erase r,
      ((pl.dropped b).filter (fun p => p.2 = r')).card := by
    have hcard : ((Board.fullRows GameConfig.standard (pl.place b)).erase
        r).card = 3 := by
      rw [Finset.card_erase_of_mem hr, h4]
    calc (3 : ℕ) = ∑ _r' ∈ (Board.fullRows GameConfig.standard
          (pl.place b)).erase r, 1 := by
          rw [Finset.sum_const, hcard, smul_eq_mul, mul_one]
      _ ≤ _ := Finset.sum_le_sum (fun r' hr' =>
          hone r' (Finset.mem_of_mem_erase hr'))
  have hsum := sum_row_added_le_four b pl
    (Board.fullRows GameConfig.standard (pl.place b))
  have hchain : ((pl.dropped b).filter (fun p => p.2 = r)).card + 3 ≤ 4 := by
    calc ((pl.dropped b).filter (fun p => p.2 = r)).card + 3
        ≤ ((pl.dropped b).filter (fun p => p.2 = r)).card
          + ∑ r' ∈ (Board.fullRows GameConfig.standard (pl.place b)).erase r,
            ((pl.dropped b).filter (fun p => p.2 = r')).card := by omega
      _ = ∑ r' ∈ Board.fullRows GameConfig.standard (pl.place b),
            ((pl.dropped b).filter (fun p => p.2 = r')).card :=
          Finset.add_sum_erase _
            (fun r' => ((pl.dropped b).filter (fun p => p.2 = r')).card) hr
      _ ≤ 4 := hsum
  have honer := hone r hr
  omega

/-- A row holding `cols − 1` cells of a well-formed board misses exactly one
column. -/
theorem row_missing_unique {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) {r : ℕ}
    (h9 : b.rowCount r + 1 = cfg.cols) :
    ∃! c, c < cfg.cols ∧ (c, r) ∉ b := by
  classical
  set filled := (b.filter (fun p => p.2 = r)).image (fun p => p.1) with hfil
  have hfsub : filled ⊆ Finset.range cfg.cols := by
    intro c hc
    rw [hfil, Finset.mem_image] at hc
    obtain ⟨p, hp, rfl⟩ := hc
    exact Finset.mem_range.mpr (hwf p (Finset.mem_filter.mp hp).1)
  have hfcard : filled.card = cfg.cols - 1 := by
    rw [hfil, Finset.card_image_of_injOn]
    · unfold Board.rowCount at h9
      omega
    · intro p hp q hq hpq
      simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
      apply Prod.ext
      · exact hpq
      · rw [hp.2, hq.2]
  have hmem_iff : ∀ c, c ∈ filled ↔ (c, r) ∈ b := by
    intro c
    rw [hfil, Finset.mem_image]
    constructor
    · rintro ⟨p, hp, rfl⟩
      rw [Finset.mem_filter] at hp
      have : p = (p.1, r) := Prod.ext rfl hp.2
      rw [← this]
      exact hp.1
    · intro h
      exact ⟨(c, r), Finset.mem_filter.mpr ⟨h, rfl⟩, rfl⟩
  have hcomp : ((Finset.range cfg.cols) \ filled).card = 1 := by
    rw [Finset.card_sdiff, Finset.inter_eq_left.mpr hfsub,
      Finset.card_range, hfcard]
    omega
  obtain ⟨c₀, hc₀⟩ := Finset.card_eq_one.mp hcomp
  refine ⟨c₀, ?_, ?_⟩
  · have : c₀ ∈ (Finset.range cfg.cols) \ filled := by
      rw [hc₀]
      exact Finset.mem_singleton_self c₀
    rw [Finset.mem_sdiff, Finset.mem_range] at this
    exact ⟨this.1, fun hmem => this.2 ((hmem_iff c₀).mpr hmem)⟩
  · rintro c ⟨hclt, hcnot⟩
    have : c ∈ (Finset.range cfg.cols) \ filled := by
      rw [Finset.mem_sdiff, Finset.mem_range]
      exact ⟨hclt, fun hmem => hcnot ((hmem_iff c).mp hmem)⟩
    rw [hc₀, Finset.mem_singleton] at this
    exact this

/-- **The tetris well**: each of a four-clear's rows misses exactly one
column before the piece — the four rows each present a single one-cell gap
for the vertical I to fill. -/
theorem tetris_row_missing_unique {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4)
    {r : ℕ} (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    ∃! c, c < 10 ∧ (c, r) ∉ b := by
  have h9 := tetris_rows_pre_nine hwf hv hnf h4 hr
  have := row_missing_unique (cfg := GameConfig.standard) hwf
    (r := r) (by rw [h9, GameConfig.standard_cols])
  rwa [GameConfig.standard_cols] at this

/-- A row's gap is filled by the piece: the missing cell of a completed row
belongs to the dropped piece. -/
theorem gap_filled_by_piece {cfg : GameConfig} {b : Board} {pl : Placement}
    {r c : ℕ} (hc : c < cfg.cols) (hnotb : (c, r) ∉ b)
    (hr : r ∈ Board.fullRows cfg (pl.place b)) :
    (c, r) ∈ pl.dropped b := by
  have hfull := Board.isFull_of_mem_fullRows hr
  have hmem := hfull c (Finset.mem_range.mpr hc)
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
  rcases hmem with h | h
  · exact absurd h hnotb
  · exact h

/-- An I whose shape spans four rows occupies a single column offset. -/
theorem I_four_rows_single_col :
    ∀ rot : Rotation,
      ((Piece.I.shapeUp rot).image (fun c => c.2)).card = 4 →
      ∀ cell ∈ Piece.I.shapeUp rot, ∀ cell' ∈ Piece.I.shapeUp rot,
        cell.1 = cell'.1 := by
  decide

/-- **The well is straight**: at a four-clear, the four one-cell gaps of the
completed rows all sit in the *same column* — the vertical I's column. The
tetris demands a clean 1-wide, 4-deep well, and nothing else can be true of
the pre-board. -/
theorem tetris_gaps_share_column {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀, ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      (c₀, r) ∉ b ∧ (c₀, r) ∈ pl.dropped b := by
  classical
  -- the finishing piece is an I spanning four rows
  have hI : pl.piece = Piece.I :=
    tetris_requires_I (b := b) (pl := pl) hnf (by omega)
  have hrows := four_clear_piece_rows_card hnf h4
  -- all shape cells share one column offset
  have hsingle : ∀ cell ∈ pl.shapeUp, ∀ cell' ∈ pl.shapeUp,
      cell.1 = cell'.1 := by
    have h := I_four_rows_single_col pl.rot
    unfold Placement.shapeUp at hrows ⊢
    rw [hI] at hrows ⊢
    exact h hrows
  -- pick any shape cell to name the column
  have hne : pl.shapeUp.Nonempty := by
    have : pl.shapeUp.card = 4 := pl.shapeUp_card
    exact Finset.card_pos.mp (by omega)
  obtain ⟨cell₀, hcell₀⟩ := hne
  refine ⟨pl.col + cell₀.1, ?_⟩
  intro r hr
  -- the row's unique gap is a dropped cell, whose column is the shared one
  obtain ⟨c, ⟨hclt, hcnot⟩, -⟩ := tetris_row_missing_unique hwf hv hnf h4 hr
  have hdrop := gap_filled_by_piece
    (by rw [GameConfig.standard_cols]; omega) hcnot hr
  have hcol : c = pl.col + cell₀.1 := by
    rw [Placement.dropped_eq_image, Finset.mem_image] at hdrop
    obtain ⟨cell, hcell, hEq⟩ := hdrop
    have h1 : pl.col + cell.1 = c := congrArg Prod.fst hEq
    have h2 := hsingle cell hcell cell₀ hcell₀
    omega
  rw [← hcol]
  exact ⟨hcnot, hdrop⟩

/-- **The unified per-row floor**: each row of a `k`-clear held at least
`5 + k` cells beforehand — the other `k − 1` completed rows each claim a
piece cell, leaving this row at most `5 − k` of the four. Recovers the
6-cell single-clear floor (`k = 1`) and the exact-9 tetris case (`k = 4`). -/
theorem cleared_row_pre_ge {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    {r : ℕ} (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    5 + (Board.fullRows GameConfig.standard (pl.place b)).card
      ≤ b.rowCount r := by
  classical
  set k := (Board.fullRows GameConfig.standard (pl.place b)).card with hk
  have hplacewf := Placement.place_wf hwf hv
  have hten : Board.rowCount (pl.place b) r = 10 := by
    have h := rowCount_of_isFull hplacewf (Board.isFull_of_mem_fullRows hr)
    rwa [GameConfig.standard_cols] at h
  have hsplit := rowCount_place_eq b pl r
  have hone : ∀ r' ∈ Board.fullRows GameConfig.standard (pl.place b),
      1 ≤ ((pl.dropped b).filter (fun p => p.2 = r')).card := by
    intro r' hr'
    obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr'
    exact Finset.card_pos.mpr ⟨q, Finset.mem_filter.mpr ⟨hq, hqr⟩⟩
  have herase : k - 1 ≤ ∑ r' ∈ (Board.fullRows GameConfig.standard
      (pl.place b)).erase r,
      ((pl.dropped b).filter (fun p => p.2 = r')).card := by
    have hcard : ((Board.fullRows GameConfig.standard (pl.place b)).erase
        r).card = k - 1 := by
      rw [Finset.card_erase_of_mem hr]
    calc k - 1 = ∑ _r' ∈ (Board.fullRows GameConfig.standard
          (pl.place b)).erase r, 1 := by
          rw [Finset.sum_const, hcard, smul_eq_mul, mul_one]
      _ ≤ _ := Finset.sum_le_sum (fun r' hr' =>
          hone r' (Finset.mem_of_mem_erase hr'))
  have hsum := sum_row_added_le_four b pl
    (Board.fullRows GameConfig.standard (pl.place b))
  have hkpos : 1 ≤ k := by
    rw [hk]
    exact Finset.card_pos.mpr ⟨r, hr⟩
  have hchain : ((pl.dropped b).filter (fun p => p.2 = r)).card + (k - 1)
      ≤ 4 := by
    calc ((pl.dropped b).filter (fun p => p.2 = r)).card + (k - 1)
        ≤ ((pl.dropped b).filter (fun p => p.2 = r)).card
          + ∑ r' ∈ (Board.fullRows GameConfig.standard (pl.place b)).erase r,
            ((pl.dropped b).filter (fun p => p.2 = r')).card := by omega
      _ = ∑ r' ∈ Board.fullRows GameConfig.standard (pl.place b),
            ((pl.dropped b).filter (fun p => p.2 = r')).card :=
          Finset.add_sum_erase _
            (fun r' => ((pl.dropped b).filter (fun p => p.2 = r')).card) hr
      _ ≤ 4 := hsum
  omega

/-- **Gaps are horizontally local**: any two cells missing from a row a
placement completes lie within three columns of each other — both gaps are
filled by the piece, and a piece spans at most four columns. The horizontal
dual of the vertical span law. -/
theorem cleared_row_gaps_within_four_cols {cfg : GameConfig} {b : Board}
    {pl : Placement} {r c c' : ℕ} (hc : c < cfg.cols) (hc' : c' < cfg.cols)
    (hnotb : (c, r) ∉ b) (hnotb' : (c', r) ∉ b)
    (hr : r ∈ Board.fullRows cfg (pl.place b)) (hle : c ≤ c') :
    c' - c ≤ 3 := by
  have h1 := gap_filled_by_piece hc hnotb hr
  have h2 := gap_filled_by_piece hc' hnotb' hr
  rw [Placement.dropped_eq_image, Finset.mem_image] at h1 h2
  obtain ⟨cell, hcell, hEq⟩ := h1
  obtain ⟨cell', hcell', hEq'⟩ := h2
  have hcol : pl.col + cell.1 = c := congrArg Prod.fst hEq
  have hcol' : pl.col + cell'.1 = c' := congrArg Prod.fst hEq'
  have hb1 := Piece.shapeUp_col_lt_four pl.piece pl.rot cell hcell
  have hb2 := Piece.shapeUp_col_lt_four pl.piece pl.rot cell' hcell'
  omega

/-- **The clearing box**: every pair of gaps a single placement closes —
across all its completed rows — lies within a `4 × 4` box: both
coordinates differ by at most three. One move's entire clearing action is
confined to one tetromino-sized window of the board. -/
theorem clearing_gaps_in_four_box {cfg : GameConfig} {b : Board}
    {pl : Placement} {r r' c c' : ℕ} (hc : c < cfg.cols) (hc' : c' < cfg.cols)
    (hnotb : (c, r) ∉ b) (hnotb' : (c', r') ∉ b)
    (hr : r ∈ Board.fullRows cfg (pl.place b))
    (hr' : r' ∈ Board.fullRows cfg (pl.place b)) :
    (c - c' ≤ 3 ∧ c' - c ≤ 3) ∧ (r - r' ≤ 3 ∧ r' - r ≤ 3) := by
  have h1 := gap_filled_by_piece hc hnotb hr
  have h2 := gap_filled_by_piece hc' hnotb' hr'
  rw [Placement.dropped_eq_image, Finset.mem_image] at h1 h2
  obtain ⟨cell, hcell, hEq⟩ := h1
  obtain ⟨cell', hcell', hEq'⟩ := h2
  have hcol : pl.col + cell.1 = c := congrArg Prod.fst hEq
  have hcol' : pl.col + cell'.1 = c' := congrArg Prod.fst hEq'
  have hrow : pl.dropOffset b + cell.2 = r := congrArg Prod.snd hEq
  have hrow' : pl.dropOffset b + cell'.2 = r' := congrArg Prod.snd hEq'
  have hb1 := Piece.shapeUp_col_lt_four pl.piece pl.rot cell hcell
  have hb2 := Piece.shapeUp_col_lt_four pl.piece pl.rot cell' hcell'
  have hb3 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  have hb4 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell' hcell'
  omega

/-- Any row of a well-formed board holds at most `cols` cells. -/
theorem rowCount_le_cols {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) (r : ℕ) : b.rowCount r ≤ cfg.cols := by
  classical
  unfold Board.rowCount
  refine le_trans (Finset.card_le_card_of_injOn (fun p => p.1) ?_ ?_)
    (le_of_eq (Finset.card_range cfg.cols))
  · intro p hp
    simp only [Finset.mem_coe, Finset.mem_filter] at hp
    exact Finset.mem_range.mpr (hwf p hp.1)
  · intro p hp q hq hpq
    simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
    apply Prod.ext
    · exact hpq
    · rw [hp.2, hq.2]

/-- A row of a well-formed board holding `cols` cells is full — its columns
exhaust the range. -/
theorem isFull_of_rowCount_eq_cols {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) {r : ℕ} (hfullcnt : b.rowCount r = cfg.cols) :
    Board.isFull cfg b r := by
  classical
  set filled := (b.filter (fun p => p.2 = r)).image (fun p => p.1) with hfil
  have hfsub : filled ⊆ Finset.range cfg.cols := by
    intro c hc
    rw [hfil, Finset.mem_image] at hc
    obtain ⟨p, hp, rfl⟩ := hc
    exact Finset.mem_range.mpr (hwf p (Finset.mem_filter.mp hp).1)
  have hfcard : filled.card = cfg.cols := by
    rw [hfil, Finset.card_image_of_injOn]
    · unfold Board.rowCount at hfullcnt
      exact hfullcnt
    · intro p hp q hq hpq
      simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
      apply Prod.ext
      · exact hpq
      · rw [hp.2, hq.2]
  have heq : filled = Finset.range cfg.cols :=
    Finset.eq_of_subset_of_card_le hfsub (by rw [hfcard, Finset.card_range])
  intro c hc
  have : c ∈ filled := heq ▸ hc
  rw [hfil, Finset.mem_image] at this
  obtain ⟨p, hp, rfl⟩ := this
  rw [Finset.mem_filter] at hp
  have : p = (p.1, r) := Prod.ext rfl hp.2
  rw [← this]
  exact hp.1

/-- **The total-gap bracket**: a `k`-clear closes between `k` and four gaps
in total — each completed row was missing at least one cell, and the piece
carries only four. -/
theorem clearing_total_gaps_bracket {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    (Board.fullRows GameConfig.standard (pl.place b)).card
        ≤ ∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
          (10 - b.rowCount r)
      ∧ ∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
          (10 - b.rowCount r) ≤ 4 := by
  classical
  have hgap1 : ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      1 ≤ 10 - b.rowCount r := by
    intro r hr
    have hle := rowCount_le_cols hwf r
    rw [GameConfig.standard_cols] at hle
    have hne : b.rowCount r ≠ 10 := by
      intro heq
      exact hnf r (isFull_of_rowCount_eq_cols hwf
        (by rw [heq, GameConfig.standard_cols]))
    omega
  constructor
  · calc (Board.fullRows GameConfig.standard (pl.place b)).card
        = ∑ _r ∈ Board.fullRows GameConfig.standard (pl.place b), 1 := by
          rw [Finset.sum_const, smul_eq_mul, mul_one]
      _ ≤ _ := Finset.sum_le_sum hgap1
  · have hplacewf := Placement.place_wf hwf hv
    have hpoint : ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
        10 - b.rowCount r
          ≤ ((pl.dropped b).filter (fun p => p.2 = r)).card := by
      intro r hr
      have hten : Board.rowCount (pl.place b) r = 10 := by
        have h := rowCount_of_isFull hplacewf
          (Board.isFull_of_mem_fullRows hr)
        rwa [GameConfig.standard_cols] at h
      have hsplit := rowCount_place_eq b pl r
      omega
    calc ∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
        (10 - b.rowCount r)
        ≤ ∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
          ((pl.dropped b).filter (fun p => p.2 = r)).card :=
        Finset.sum_le_sum hpoint
      _ ≤ 4 := sum_row_added_le_four b pl _

/-- **The I vanishes**: every cell of a tetris's finishing piece lies in a
cleared row — the vertical I is consumed whole, leaving no trace of itself
on the post-clear board. -/
theorem tetris_piece_vanishes {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∀ q ∈ pl.dropped b, q.2 ∈ Board.fullRows GameConfig.standard
      (pl.place b) := by
  classical
  -- each of the four rows takes exactly one piece cell
  have hone : ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      ((pl.dropped b).filter (fun p => p.2 = r)).card = 1 := by
    intro r hr
    have h9 := tetris_rows_pre_nine hwf hv hnf h4 hr
    have hten : Board.rowCount (pl.place b) r = 10 := by
      have h := rowCount_of_isFull (Placement.place_wf hwf hv)
        (Board.isFull_of_mem_fullRows hr)
      rwa [GameConfig.standard_cols] at h
    have hsplit := rowCount_place_eq b pl r
    omega
  -- the four one-cell fibers exhaust the four-cell piece
  have hdisj : ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      ∀ r' ∈ Board.fullRows GameConfig.standard (pl.place b), r ≠ r' →
      Disjoint ((pl.dropped b).filter (fun p => p.2 = r))
        ((pl.dropped b).filter (fun p => p.2 = r')) := by
    intro r _ r' _ hne
    rw [Finset.disjoint_left]
    intro p hp hp'
    rw [Finset.mem_filter] at hp hp'
    exact hne (by rw [← hp.2, hp'.2])
  have hbicard : ((Board.fullRows GameConfig.standard (pl.place b)).biUnion
      (fun r => (pl.dropped b).filter (fun p => p.2 = r))).card = 4 := by
    rw [Finset.card_biUnion hdisj, Finset.sum_congr rfl hone,
      Finset.sum_const, smul_eq_mul, mul_one, h4]
  have hbisub : (Board.fullRows GameConfig.standard (pl.place b)).biUnion
      (fun r => (pl.dropped b).filter (fun p => p.2 = r)) ⊆ pl.dropped b := by
    intro q hq
    rw [Finset.mem_biUnion] at hq
    obtain ⟨r, -, hqr⟩ := hq
    exact (Finset.mem_filter.mp hqr).1
  have heq : (Board.fullRows GameConfig.standard (pl.place b)).biUnion
      (fun r => (pl.dropped b).filter (fun p => p.2 = r)) = pl.dropped b :=
    Finset.eq_of_subset_of_card_le hbisub
      (by rw [hbicard, Placement.card_dropped])
  intro q hq
  have : q ∈ (Board.fullRows GameConfig.standard (pl.place b)).biUnion
      (fun r => (pl.dropped b).filter (fun p => p.2 = r)) := by
    rw [heq]
    exact hq
  rw [Finset.mem_biUnion] at this
  obtain ⟨r, hr, hqr⟩ := this
  rw [Finset.mem_filter] at hqr
  rw [hqr.2]
  exact hr

/-- Shapes of S, Z and O put at most two cells in any one column, and no
piece but I puts four — a shape table check. -/
theorem shape_col_fiber_not_big_of_SZO :
    ∀ p : Piece, p = Piece.S ∨ p = Piece.Z ∨ p = Piece.O →
    ∀ r : Rotation, ∀ t < 4,
      ¬ (3 ≤ ((p.shapeUp r).filter (fun cell => cell.1 = t)).card) := by
  decide

theorem shape_col_fiber_le_three_of_ne_I :
    ∀ p : Piece, p ≠ Piece.I → ∀ r : Rotation, ∀ t < 4,
      ((p.shapeUp r).filter (fun cell => cell.1 = t)).card ≤ 3 := by
  decide

theorem shape_col_fiber_not_four_of_ne_I :
    ∀ p : Piece, p ≠ Piece.I → ∀ r : Rotation, ∀ t < 4,
      ¬ (4 ≤ ((p.shapeUp r).filter (fun cell => cell.1 = t)).card) := by
  decide

/-- **A full-column feed pins the I**: only the I piece can pour all four
of its cells into a single column. -/
theorem full_feed_requires_I {pl : Placement} {j : ℕ}
    (h4 : 4 ≤ pl.colProfile j) : pl.piece = Piece.I := by
  by_contra hI
  unfold Placement.colProfile at h4
  rcases Nat.lt_or_ge j pl.col with hj | hj
  · have hempty : (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j)) = ∅ := by
      rw [Finset.filter_eq_empty_iff]
      intro cell _
      omega
    rw [hempty] at h4
    simp at h4
  · set t := j - pl.col with ht
    have hsame : (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j))
        = (pl.shapeUp.filter (fun cell => cell.1 = t)) := by
      apply Finset.filter_congr
      intro cell _
      constructor
      · intro h
        omega
      · intro h
        omega
    rw [hsame] at h4
    rcases Nat.lt_or_ge t 4 with ht4 | ht4
    · unfold Placement.shapeUp at h4
      exact shape_col_fiber_not_four_of_ne_I pl.piece hI pl.rot t ht4 h4
    · have hempty : (pl.shapeUp.filter (fun cell => cell.1 = t)) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro cell hcell
        have := Piece.shapeUp_col_lt_four pl.piece pl.rot cell hcell
        omega
      rw [hempty] at h4
      simp at h4

/-- **Heavy column feeds require a tall piece**: a placement delivering
three or more cells into one column plays I, L, J or T (vertical T carries
a 3-cell column too) — S, Z and O cannot feed any column past two. -/
theorem heavy_feed_requires_tall {pl : Placement} {j : ℕ}
    (h3 : 3 ≤ pl.colProfile j) :
    pl.piece = Piece.I ∨ pl.piece = Piece.L ∨ pl.piece = Piece.J
      ∨ pl.piece = Piece.T := by
  by_contra hcon
  push Not at hcon
  obtain ⟨hI, hL, hJ, hT⟩ := hcon
  have hSZO : pl.piece = Piece.S ∨ pl.piece = Piece.Z ∨ pl.piece = Piece.O := by
    cases hp : pl.piece <;> simp_all
  unfold Placement.colProfile at h3
  rcases Nat.lt_or_ge j pl.col with hj | hj
  · -- the column is left of the piece: the fiber is empty
    have hempty : (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j)) = ∅ := by
      rw [Finset.filter_eq_empty_iff]
      intro cell _
      omega
    rw [hempty] at h3
    simp at h3
  · -- shift to shape coordinates
    set t := j - pl.col with ht
    have hsame : (pl.shapeUp.filter (fun cell => pl.col + cell.1 = j))
        = (pl.shapeUp.filter (fun cell => cell.1 = t)) := by
      apply Finset.filter_congr
      intro cell _
      constructor
      · intro h
        omega
      · intro h
        omega
    rw [hsame] at h3
    rcases Nat.lt_or_ge t 4 with ht4 | ht4
    · unfold Placement.shapeUp at h3
      exact shape_col_fiber_not_big_of_SZO pl.piece hSZO pl.rot t ht4 h3
    · have hempty : (pl.shapeUp.filter (fun cell => cell.1 = t)) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro cell hcell
        have := Piece.shapeUp_col_lt_four pl.piece pl.rot cell hcell
        omega
      rw [hempty] at h3
      simp at h3

/-- **The tetris feeds one column**: at a four-clear the finishing I pours
all four of its cells into the well column and delivers nothing anywhere
else — the column profile is `4` at one column and `0` at the other nine.
The quantitative closure of the anatomy: combined with the column ledger,
every tetris spends its entire four-cell feed budget on a single column. -/
theorem tetris_feeds_single_column {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10, pl.colProfile c₀ = 4 ∧
      ∀ j < 10, j ≠ c₀ → pl.colProfile j = 0 := by
  classical
  obtain ⟨c₀, hc₀⟩ := tetris_gaps_share_column hwf hv hnf h4
  -- the well column is on the board
  have hne : (Board.fullRows GameConfig.standard (pl.place b)).Nonempty :=
    Finset.card_pos.mp (by omega)
  obtain ⟨r₀, hr₀⟩ := hne
  have hc₀lt : c₀ < 10 := by
    have hdropmem := (hc₀ r₀ hr₀).2
    rw [Placement.dropped_eq_image, Finset.mem_image] at hdropmem
    obtain ⟨cell, hcell, hEq⟩ := hdropmem
    have h1 : pl.col + cell.1 = c₀ := congrArg Prod.fst hEq
    have h2 := hv cell hcell
    rw [GameConfig.standard_cols] at h2
    omega
  -- the four gap cells sit in the dropped piece's column-c₀ fiber
  have hfour : 4 ≤ ((pl.dropped b).filter (fun p => p.1 = c₀)).card := by
    have hinj : (Board.fullRows GameConfig.standard (pl.place b)).card
        ≤ ((pl.dropped b).filter (fun p => p.1 = c₀)).card := by
      refine Finset.card_le_card_of_injOn (fun r => (c₀, r)) ?_ ?_
      · intro r hr
        exact Finset.mem_filter.mpr ⟨(hc₀ r hr).2, rfl⟩
      · intro r _ r' _ h
        exact congrArg Prod.snd h
    omega
  -- and the fiber is the column profile, capped at four
  have hprof : ((pl.dropped b).filter (fun p => p.1 = c₀)).card
      = pl.colProfile c₀ := by
    have h := Placement.colCount_cellsAt pl (pl.dropOffset b) c₀
    unfold Board.colCount at h
    unfold Placement.dropped
    exact h
  have hcap : ((pl.dropped b).filter (fun p => p.1 = c₀)).card ≤ 4 := by
    calc ((pl.dropped b).filter (fun p => p.1 = c₀)).card
        ≤ (pl.dropped b).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
      _ = 4 := Placement.card_dropped b pl
  have hc4 : pl.colProfile c₀ = 4 := by omega
  -- the profile sums to four, so every other column gets nothing
  have hsum : ∑ j ∈ Finset.range 10, pl.colProfile j = 4 := by
    have h := Placement.sum_colProfile hv
    rwa [GameConfig.standard_cols] at h
  have herase : ∑ j ∈ (Finset.range 10).erase c₀, pl.colProfile j = 0 := by
    have h := Finset.add_sum_erase (Finset.range 10) pl.colProfile
      (Finset.mem_range.mpr hc₀lt)
    omega
  refine ⟨c₀, hc₀lt, hc4, ?_⟩
  intro j hj hne
  exact Finset.sum_eq_zero_iff.mp herase j
    (Finset.mem_erase.mpr ⟨hne, Finset.mem_range.mpr hj⟩)

/-- A four-clear whose completed rows all take a piece cell in column `j`
feeds column `j` its entire four-cell budget — the four rows inject into
the dropped piece's `j`-fiber, which the piece caps at four. -/
theorem well_feed_four {b : Board} {pl : Placement} {j : ℕ}
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4)
    (hwell : ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      (j, r) ∈ pl.dropped b) :
    pl.colProfile j = 4 := by
  classical
  have hfour : 4 ≤ ((pl.dropped b).filter (fun p => p.1 = j)).card := by
    have hinj : (Board.fullRows GameConfig.standard (pl.place b)).card
        ≤ ((pl.dropped b).filter (fun p => p.1 = j)).card := by
      refine Finset.card_le_card_of_injOn (fun r => (j, r)) ?_ ?_
      · intro r hr
        exact Finset.mem_filter.mpr ⟨hwell r hr, rfl⟩
      · intro r _ r' _ h
        exact congrArg Prod.snd h
    omega
  have hprof : ((pl.dropped b).filter (fun p => p.1 = j)).card
      = pl.colProfile j := by
    have h := Placement.colCount_cellsAt pl (pl.dropOffset b) j
    unfold Board.colCount at h
    unfold Placement.dropped
    exact h
  have hcap : ((pl.dropped b).filter (fun p => p.1 = j)).card ≤ 4 := by
    calc ((pl.dropped b).filter (fun p => p.1 = j)).card
        ≤ (pl.dropped b).card :=
          Finset.card_le_card (Finset.filter_subset _ _)
      _ = 4 := Placement.card_dropped b pl
  omega

/-- **The tetris-well rationing law**: per cycle period, at most three
tetrises may sink their well into any one fixed column — a fourth would
demand sixteen of the column's exact fourteen-cell period budget. The
solver-design consequence: tetris wells must rotate across the board. -/
theorem cycle_tetris_well_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ((Finset.range 35).filter (fun k =>
        (Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init (n + k))).place
              (trace GameConfig.standard π GameState.init (n + k)).board)).card
          = 4
        ∧ ∀ r ∈ Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init (n + k))).place
              (trace GameConfig.standard π GameState.init (n + k)).board),
          (j, r) ∈ (π (trace GameConfig.standard π GameState.init
            (n + k))).dropped
              (trace GameConfig.standard π GameState.init (n + k)).board)).card
      ≤ 3 := by
  classical
  refine le_trans (Finset.card_le_card ?_)
    (cycle_tall_drop_column_cap hv hj hcyc)
  intro k hk
  rw [Finset.mem_filter] at hk ⊢
  exact ⟨hk.1, well_feed_four hk.2.1 hk.2.2⟩

/-- Every non-I shape occupies at most three distinct rows (28 cases). -/
theorem shape_rows_le_three_of_ne_I :
    ∀ p : Piece, p ≠ Piece.I → ∀ r : Rotation,
      ((p.shapeUp r).image (fun cell => cell.2)).card ≤ 3 := by
  decide

/-- The O shape occupies at most two distinct rows in every rotation. -/
theorem shape_rows_le_two_of_O :
    ∀ r : Rotation,
      ((Piece.O.shapeUp r).image (fun cell => cell.2)).card ≤ 2 := by
  decide

/-- **Clears are bounded by the piece's row span**: a placement completes
at most as many rows as its shape occupies — each cleared row must take a
piece cell, and the cleared rows inject into the piece's occupied rows. -/
theorem clear_count_le_shape_rows {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    (Board.fullRows cfg (pl.place b)).card
      ≤ (pl.shapeUp.image (fun cell => cell.2)).card := by
  classical
  have hsub : Board.fullRows cfg (pl.place b)
      ⊆ (pl.dropped b).image (fun q => q.2) := by
    intro r hr
    obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr
    exact Finset.mem_image.mpr ⟨q, hq, hqr⟩
  calc (Board.fullRows cfg (pl.place b)).card
      ≤ ((pl.dropped b).image (fun q => q.2)).card :=
        Finset.card_le_card hsub
    _ = (pl.shapeUp.image (fun cell => pl.dropOffset b + cell.2)).card := by
        rw [Placement.dropped_eq_image, Finset.image_image]
        rfl
    _ = ((pl.shapeUp.image (fun cell => cell.2)).image
          (fun x => pl.dropOffset b + x)).card := by
        rw [Finset.image_image]
        rfl
    _ ≤ (pl.shapeUp.image (fun cell => cell.2)).card :=
        Finset.card_image_le

/-- **The O never clears more than two rows** — its square spans two rows
in every rotation. -/
theorem clears_le_two_of_O {cfg : GameConfig} {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull cfg b r) (hO : pl.piece = Piece.O) :
    (Board.fullRows cfg (pl.place b)).card ≤ 2 := by
  have h := clear_count_le_shape_rows (pl := pl) hnf
  unfold Placement.shapeUp at h
  rw [hO] at h
  exact le_trans h (shape_rows_le_two_of_O pl.rot)

/-- **Only the I can clear more than three rows**: every other piece spans
at most three rows — the graded companion of `tetris_requires_I`, and with
it the full per-piece clear-cap ladder: O ≤ 2, non-I ≤ 3, four ⇒ I. -/
theorem clears_le_three_of_ne_I {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (hI : pl.piece ≠ Piece.I) :
    (Board.fullRows cfg (pl.place b)).card ≤ 3 := by
  have h := clear_count_le_shape_rows (pl := pl) hnf
  unfold Placement.shapeUp at h
  exact le_trans h (shape_rows_le_three_of_ne_I pl.piece hI pl.rot)

/-- **A starving column caps the game's clears**: over any window in which
column `j` receives no cells, the whole game clears at most as many rows as
column `j` held at the window's start — every clear bills the starving
column one cell it never restocks. -/
theorem starving_column_caps_clears {π : Policy GameConfig.standard}
    {j : ℕ} (hj : j < 10) {n w : ℕ}
    (hstarve : colDelivered π j (n + w) = colDelivered π j n) :
    cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n
      ≤ (trace GameConfig.standard π GameState.init n).board.colCount j := by
  have hled1 := colDelivered_ledger (π := π) hj n
  have hled2 := colDelivered_ledger (π := π) hj (n + w)
  have hclm := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right n w)
  omega

/-- **No column starves on a cycle**: on a cycle, a window in which some
column receives no cells lasts at most 34 placements — the exact
14-cells-per-period intake forbids a starving period. -/
theorem cycle_column_starvation_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {j : ℕ} (hj : j < 10) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) {w : ℕ}
    (hstarve : colDelivered π j (m₀ + w) = colDelivered π j m₀) :
    w ≤ 34 := by
  have hbr := (cycle_column_window_bracket hv hj hcyc hm w).1
  omega

/-- **Clear-free intake is pure stacking**: over a window with no clears,
what a column receives is exactly what it gains — the ledger has no
clearing outflow to hide behind. -/
theorem clear_free_column_feed_eq {π : Policy GameConfig.standard}
    {j : ℕ} (hj : j < 10) {n w : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
      = cleared GameConfig.standard π GameState.init n) :
    colDelivered π j (n + w) - colDelivered π j n
      = (trace GameConfig.standard π GameState.init (n + w)).board.colCount j
        - (trace GameConfig.standard π GameState.init n).board.colCount j := by
  have hled1 := colDelivered_ledger (π := π) hj n
  have hled2 := colDelivered_ledger (π := π) hj (n + w)
  have hmono := colDelivered_mono π j (Nat.le_add_right n w)
  omega

/-- **The clear-free feed cap**: over a clear-free window that ends alive,
every column receives at most twenty cells — a starved outflow turns the
board's height ceiling into an intake ceiling, column by column. (Summed
over the ten columns this recovers the fifty-placement clear-free horizon:
`4w ≤ 200`.) -/
theorem clear_free_column_feed_le {π : Policy GameConfig.standard}
    {j : ℕ} (hj : j < 10) {n w : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
      = cleared GameConfig.standard π GameState.init n)
    (hlive : ¬ (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    colDelivered π j (n + w) - colDelivered π j n ≤ 20 := by
  have heq := clear_free_column_feed_eq (π := π) hj hdry
  have hif : ∀ p ∈ (trace GameConfig.standard π GameState.init (n + w)).board,
      p.2 < GameConfig.standard.rows :=
    (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive
  have hcc := colCount_le_rows hif j
  rw [GameConfig.standard_rows] at hcc
  omega

/-- **The tetris column flow**: a four-clear is a no-op on its well column
and a pure four-cell drain on each of the other nine — the well gains four
from the I and loses four to the clears, while every other column is
billed four rows it never restocked. In particular every non-well column
held at least four cells going in. -/
theorem tetris_step_column_flow {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10,
      (Placement.applyStep GameConfig.standard b pl).colCount c₀
        = b.colCount c₀
      ∧ ∀ j < 10, j ≠ c₀ →
        (Placement.applyStep GameConfig.standard b pl).colCount j + 4
          = b.colCount j := by
  obtain ⟨c₀, hc₀lt, hc4, hz⟩ := tetris_feeds_single_column hwf hv hnf h4
  have hlc : Board.linesCleared GameConfig.standard (pl.place b) = 4 := by
    unfold Board.linesCleared
    exact h4
  refine ⟨c₀, hc₀lt, ?_, ?_⟩
  · have h := applyStep_colCount GameConfig.standard b pl
      (j := c₀) (by rw [GameConfig.standard_cols]; omega)
    rw [hlc, hc4] at h
    omega
  · intro j hj hne
    have h := applyStep_colCount GameConfig.standard b pl
      (j := j) (by rw [GameConfig.standard_cols]; omega)
    rw [hlc, hz j hj hne] at h
    omega

/-- **Untouched columns pay the full bill**: at a `k`-clear, every column
the piece does not feed drops by exactly `k` — and so held at least `k`
cells going in. The general-`k` form of the tetris drain. -/
theorem clear_step_column_drain {cfg : GameConfig} {b : Board}
    {pl : Placement} {j : ℕ} (hj : j < cfg.cols)
    (hz : pl.colProfile j = 0) :
    (Placement.applyStep cfg b pl).colCount j
        + (Board.fullRows cfg (pl.place b)).card
      = b.colCount j := by
  have h := applyStep_colCount cfg b pl (j := j) hj
  unfold Board.linesCleared at h
  rw [hz] at h
  omega

/-- **A placement touches at most four columns**: the four cells cannot
spread wider. -/
theorem placement_touched_columns_le_four {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    ((Finset.range 10).filter (fun j => ¬ (pl.colProfile j = 0))).card
      ≤ 4 := by
  classical
  have hsum : ∑ j ∈ Finset.range 10, pl.colProfile j = 4 := by
    have h := Placement.sum_colProfile hv
    rwa [GameConfig.standard_cols] at h
  have hbound : ((Finset.range 10).filter
        (fun j => ¬ (pl.colProfile j = 0))).card
      ≤ ∑ j ∈ (Finset.range 10).filter (fun j => ¬ (pl.colProfile j = 0)),
          pl.colProfile j := by
    calc ((Finset.range 10).filter (fun j => ¬ (pl.colProfile j = 0))).card
        = ∑ _j ∈ (Finset.range 10).filter
            (fun j => ¬ (pl.colProfile j = 0)), 1 := by
          rw [Finset.sum_const, smul_eq_mul, mul_one]
      _ ≤ _ := Finset.sum_le_sum (fun j hj => by
          have := (Finset.mem_filter.mp hj).2
          omega)
  have hle : ∑ j ∈ (Finset.range 10).filter
        (fun j => ¬ (pl.colProfile j = 0)), pl.colProfile j
      ≤ ∑ j ∈ Finset.range 10, pl.colProfile j :=
    Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  omega

/-- **Six columns always go unfed**: every placement leaves at least six of
the ten columns without a single cell — with `clear_step_column_drain`, a
`k`-clear drains `k` cells from each of at least six columns in one move. -/
theorem placement_untouched_columns_ge_six {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    6 ≤ ((Finset.range 10).filter (fun j => pl.colProfile j = 0)).card := by
  classical
  have htouch := placement_touched_columns_le_four hv
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := Finset.range 10) (fun j => pl.colProfile j = 0)
  rw [Finset.card_range] at hsplit
  omega

/-- **The mass clock**: the board's cell count is congruent to `4n` modulo
ten at every step — deliveries add four, clears remove exact tens. -/
theorem trace_board_count_mod_ten {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).board.count % 10
      = (4 * n) % 10 := by
  have h := init_ledger (cfg := GameConfig.standard) hv n
  rw [GameConfig.standard_cols] at h
  omega

/-- **Reachable boards have even mass**: every board a game can visit holds
an even number of cells — an odd-count board is unreachable from the empty
board, a blanket obstruction that prunes half of all board configurations
from the Atlas before any search begins. -/
theorem trace_board_count_even {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    2 ∣ (trace GameConfig.standard π GameState.init n).board.count := by
  have h := init_ledger (cfg := GameConfig.standard) hv n
  rw [GameConfig.standard_cols] at h
  omega

/-- **Board mass reveals the step count mod five** — even across different
policies: two reachable boards with the same mass residue mod ten sit at
the same step index mod five. The mass clock is an observable shared by
every game ever played. -/
theorem trace_board_count_determines_step_mod_five
    {π π' : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hv' : ∀ g, (π' g).Valid GameConfig.standard) {m n : ℕ}
    (hcount : (trace GameConfig.standard π GameState.init m).board.count % 10
      = (trace GameConfig.standard π' GameState.init n).board.count % 10) :
    m % 5 = n % 5 := by
  have h1 := trace_board_count_mod_ten hv m
  have h2 := trace_board_count_mod_ten hv' n
  omega

set_option maxRecDepth 40000 in
/-- **The action alphabet has exactly 240 letters**: the standard board
admits precisely 240 valid placements — piece × rotation × column,
in-bounds. A kernel-checked count of the game's entire move vocabulary. -/
theorem card_valid_placements :
    (((Finset.univ : Finset Piece) ×ˢ (Finset.univ : Finset Rotation)
        ×ˢ Finset.range 10).filter (fun t =>
      (⟨t.1, t.2.1, t.2.2⟩ : Placement).Valid GameConfig.standard)).card
      = 240 := by
  decide

/-- The enumeration is faithful: a placement is valid iff its triple lands
in the 240-element table — validity forces `col < 10`, so the `range 10`
column window misses nothing. -/
theorem valid_iff_mem_enum (pl : Placement) :
    pl.Valid GameConfig.standard
      ↔ (pl.piece, pl.rot, pl.col)
        ∈ (((Finset.univ : Finset Piece) ×ˢ (Finset.univ : Finset Rotation)
            ×ˢ Finset.range 10).filter (fun t =>
          (⟨t.1, t.2.1, t.2.2⟩ : Placement).Valid GameConfig.standard)) := by
  constructor
  · intro hv
    rw [Finset.mem_filter]
    refine ⟨?_, hv⟩
    rw [Finset.mem_product]
    refine ⟨Finset.mem_univ _, ?_⟩
    rw [Finset.mem_product]
    refine ⟨Finset.mem_univ _, ?_⟩
    rw [Finset.mem_range]
    have h := hv.col_lt_cols
    rwa [GameConfig.standard_cols] at h
  · intro hm
    exact (Finset.mem_filter.mp hm).2

set_option maxRecDepth 40000 in
/-- **The square is the most placeable piece**: the O admits 36 valid
placements — its 2-wide footprint fits nine columns in each of the four
(identical) rotations. -/
theorem card_valid_placements_O :
    (((Finset.univ : Finset Rotation) ×ˢ Finset.range 10).filter (fun t =>
      (⟨Piece.O, t.1, t.2⟩ : Placement).Valid GameConfig.standard)).card
      = 36 := by
  decide

set_option maxRecDepth 40000 in
/-- Every piece other than the O admits exactly 34 valid placements: the
240-letter alphabet splits as `36 + 6 × 34`, and the O — the piece that
can never clear more than two rows — is, ironically, the most placeable. -/
theorem card_valid_placements_of_ne_O :
    ∀ p : Piece, p ≠ Piece.O →
      (((Finset.univ : Finset Rotation) ×ˢ Finset.range 10).filter (fun t =>
        (⟨p, t.1, t.2⟩ : Placement).Valid GameConfig.standard)).card
        = 34 := by
  decide

/-- **The tetris pre-board is determined on its window**: before a
four-clear, the four rows about to clear are *exactly* "everything but the
well" — cell `(c, r)` is present iff `c ≠ c₀`. The anatomy is not merely
constrained; on the cleared window it is unique up to the well's column. -/
theorem tetris_rows_pre_shape {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10, ∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      ∀ c < 10, ((c, r) ∈ b ↔ c ≠ c₀) := by
  obtain ⟨c₀, hc₀⟩ := tetris_gaps_share_column hwf hv hnf h4
  have hne : (Board.fullRows GameConfig.standard (pl.place b)).Nonempty :=
    Finset.card_pos.mp (by omega)
  obtain ⟨r₀, hr₀⟩ := hne
  have hc₀lt : c₀ < 10 := by
    have hdropmem := (hc₀ r₀ hr₀).2
    rw [Placement.dropped_eq_image, Finset.mem_image] at hdropmem
    obtain ⟨cell, hcell, hEq⟩ := hdropmem
    have h1 : pl.col + cell.1 = c₀ := congrArg Prod.fst hEq
    have h2 := hv cell hcell
    rw [GameConfig.standard_cols] at h2
    omega
  refine ⟨c₀, hc₀lt, ?_⟩
  intro r hr c hc
  obtain ⟨c', hc', huniq⟩ := tetris_row_missing_unique hwf hv hnf h4 hr
  have hc₀c' : c₀ = c' := huniq c₀ ⟨hc₀lt, (hc₀ r hr).1⟩
  constructor
  · intro hmem heq
    rw [heq, hc₀c'] at hmem
    exact hc'.2 hmem
  · intro hne'
    by_contra hnotb
    have : c = c' := huniq c ⟨hc, hnotb⟩
    rw [← hc₀c'] at this
    exact hne' this

/-- A shape occupying four distinct rows occupies exactly rows 0–3. -/
theorem shape_rows_eq_of_card_four :
    ∀ p : Piece, ∀ r : Rotation,
      ((p.shapeUp r).image (fun cell => cell.2)).card = 4 →
      (p.shapeUp r).image (fun cell => cell.2)
        = ({0, 1, 2, 3} : Finset ℕ) := by
  decide

/-- **The tetris window sits exactly at the drop offset**: the four rows a
four-clear completes are precisely `[dropOffset, dropOffset + 3]` — the
vertical I lands on the well stack and the clearing window starts exactly
where the piece comes to rest. Pins `four_clear_rows_eq_Icc`'s abstract
base to the geometry of the drop. -/
theorem tetris_window_base {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    Board.fullRows GameConfig.standard (pl.place b)
      = Finset.Icc (pl.dropOffset b) (pl.dropOffset b + 3) := by
  classical
  have hrows := four_clear_piece_rows_card hnf h4
  have hshape : pl.shapeUp.image (fun c => c.2)
      = ({0, 1, 2, 3} : Finset ℕ) := by
    unfold Placement.shapeUp at hrows ⊢
    exact shape_rows_eq_of_card_four pl.piece pl.rot hrows
  have himg : (pl.dropped b).image (fun q => q.2)
      = (pl.shapeUp.image (fun c => c.2)).image
          (fun t => pl.dropOffset b + t) := by
    rw [Placement.dropped_eq_image, Finset.image_image, Finset.image_image]
    rfl
  have hIcc : (pl.dropped b).image (fun q => q.2)
      = Finset.Icc (pl.dropOffset b) (pl.dropOffset b + 3) := by
    rw [himg, hshape]
    ext x
    simp only [Finset.mem_image, Finset.mem_insert, Finset.mem_singleton,
      Finset.mem_Icc]
    constructor
    · rintro ⟨y, hy, rfl⟩
      omega
    · intro hx
      exact ⟨x - pl.dropOffset b, by omega, by omega⟩
  have hsub : Board.fullRows GameConfig.standard (pl.place b)
      ⊆ Finset.Icc (pl.dropOffset b) (pl.dropOffset b + 3) := by
    intro r hr
    obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr
    rw [← hIcc]
    exact Finset.mem_image.mpr ⟨q, hq, hqr⟩
  apply Finset.eq_of_subset_of_card_le hsub
  rw [h4, Nat.card_Icc]
  omega

/-- An I occupying four distinct rows is the vertical I: its shape is a
single column of four cells. -/
theorem I_shape_vertical_eq :
    ∀ rot : Rotation,
      ((Piece.I.shapeUp rot).image (fun c => c.2)).card = 4 →
      ∃ t < 4, Piece.I.shapeUp rot
        = ({(t, 0), (t, 1), (t, 2), (t, 3)} : Finset Coord) := by
  decide

/-- **The window is the well-stack's crown**: at a four-clear, the drop
offset equals the well column's height, and the four clearing rows are
exactly `[colHeight c₀, colHeight c₀ + 3]` — the tetris clears the four
rows immediately atop the well stack, no more geometry left unpinned. -/
theorem tetris_window_at_well_height {b : Board} {pl : Placement}
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10, pl.dropOffset b = b.colHeight c₀
      ∧ Board.fullRows GameConfig.standard (pl.place b)
        = Finset.Icc (b.colHeight c₀) (b.colHeight c₀ + 3) := by
  classical
  have hI : pl.piece = Piece.I :=
    tetris_requires_I (b := b) (pl := pl) hnf (by omega)
  have hrows := four_clear_piece_rows_card hnf h4
  have hrows' : ((Piece.I.shapeUp pl.rot).image (fun c => c.2)).card = 4 := by
    unfold Placement.shapeUp at hrows
    rw [hI] at hrows
    exact hrows
  obtain ⟨t, ht4, hshape⟩ := I_shape_vertical_eq pl.rot hrows'
  have hshapeUp : pl.shapeUp
      = ({(t, 0), (t, 1), (t, 2), (t, 3)} : Finset Coord) := by
    unfold Placement.shapeUp
    rw [hI]
    exact hshape
  have hc₀lt : pl.col + t < 10 := by
    have hmem : (t, 0) ∈ pl.shapeUp := by
      rw [hshapeUp]
      simp
    have h := hv (t, 0) hmem
    rwa [GameConfig.standard_cols] at h
  have hoff : pl.dropOffset b = b.colHeight (pl.col + t) := by
    unfold Placement.dropOffset
    rw [hshapeUp]
    simp only [Finset.sup_insert, Finset.sup_singleton]
    omega
  refine ⟨pl.col + t, hc₀lt, hoff, ?_⟩
  rw [← hoff]
  exact tetris_window_base hnf h4

/-- **The well outruns the skyline by four**: a tetris demands a well
column at least four rows deeper than *every* other column — each of the
nine must reach through the entire clearing window while the well stops
below it. The steepest possible local relief, forced at every four-clear. -/
theorem tetris_well_depth {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10, ∀ c < 10, c ≠ c₀ →
      b.colHeight c₀ + 4 ≤ b.colHeight c := by
  obtain ⟨c₀, hc₀lt, hoff, hwin⟩ := tetris_window_at_well_height hv hnf h4
  obtain ⟨c₀', hc₀'lt, hshape⟩ := tetris_rows_pre_shape hwf hv hnf h4
  have hr3 : b.colHeight c₀ + 3
      ∈ Board.fullRows GameConfig.standard (pl.place b) := by
    rw [hwin, Finset.mem_Icc]
    omega
  have hc0eq : c₀' = c₀ := by
    by_contra hne
    have hmem : (c₀, b.colHeight c₀ + 3) ∈ b :=
      (hshape _ hr3 c₀ hc₀lt).mpr (fun heq => hne heq.symm)
    have := Board.lt_colHeight hmem
    omega
  refine ⟨c₀, hc₀lt, ?_⟩
  intro c hc hne
  have hmem : (c, b.colHeight c₀ + 3) ∈ b :=
    (hshape _ hr3 c hc).mpr (by rw [hc0eq]; exact hne)
  have := Board.lt_colHeight hmem
  omega

/-- **The well height cap**: on an in-field board, a tetris can only fire
with its well at height sixteen or lower — the other nine columns must
still fit their four extra rows under the twenty-row ceiling. A tetris is
a mid-board event; there are no last-gasp tetrises off a full-height
stack. -/
theorem tetris_well_height_cap {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4)
    (hif : ∀ p ∈ b, p.2 < GameConfig.standard.rows) :
    ∃ c₀ < 10,
      (∀ c < 10, c ≠ c₀ → b.colHeight c₀ + 4 ≤ b.colHeight c)
      ∧ b.colHeight c₀ ≤ 16 := by
  obtain ⟨c₀, hc₀lt, hdepth⟩ := tetris_well_depth hwf hv hnf h4
  refine ⟨c₀, hc₀lt, hdepth, ?_⟩
  set c := (c₀ + 1) % 10 with hc
  have hclt : c < 10 := Nat.mod_lt _ (by omega)
  have hcne : c ≠ c₀ := by omega
  have h1 := hdepth c hclt hcne
  have h2 := Board.colHeight_le_rows_of_in_field hif c
  rw [GameConfig.standard_rows] at h2
  omega

/-- **All clears happen in the landing window**: every row a placement
completes lies within `[dropOffset, dropOffset + 3]` — a piece can only
finish rows it physically occupies, and it occupies four rows above where
it lands. Config-general. -/
theorem clear_rows_in_drop_window {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r) {r : ℕ}
    (hr : r ∈ Board.fullRows cfg (pl.place b)) :
    pl.dropOffset b ≤ r ∧ r ≤ pl.dropOffset b + 3 := by
  obtain ⟨q, hq, hqr⟩ := mem_fullRows_place_has_piece_cell hnf hr
  rw [Placement.dropped_eq_image, Finset.mem_image] at hq
  obtain ⟨cell, hcell, hEq⟩ := hq
  have hrow : pl.dropOffset b + cell.2 = q.2 := congrArg Prod.snd hEq
  have hb := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  omega

/-- **Untouched columns overtop the landing site**: whenever a placement
clears anything, every column it does not feed rises strictly above the
drop offset — the stack the piece lands beside must already reach through
the row it completes. Config-general. -/
theorem clear_untouched_column_height {cfg : GameConfig} {b : Board}
    {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (hcl : (Board.fullRows cfg (pl.place b)).Nonempty) {c : ℕ}
    (hc : c < cfg.cols) (hz : pl.colProfile c = 0) :
    pl.dropOffset b < b.colHeight c := by
  obtain ⟨r, hr⟩ := hcl
  have hwin := clear_rows_in_drop_window hnf hr
  have hfull := Board.isFull_of_mem_fullRows hr
  have hmem := hfull c (Finset.mem_range.mpr hc)
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
  rcases hmem with h | h
  · have := Board.lt_colHeight h
    omega
  · exfalso
    have hfib : ((pl.dropped b).filter (fun p => p.1 = c)).Nonempty :=
      ⟨(c, r), Finset.mem_filter.mpr ⟨h, rfl⟩⟩
    have hprof : ((pl.dropped b).filter (fun p => p.1 = c)).card
        = pl.colProfile c := by
      have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) c
      unfold Board.colCount at hcc
      unfold Placement.dropped
      exact hcc
    have := Finset.card_pos.mpr hfib
    omega

/-- **No placement clears on the empty board**: four cells cannot complete
a ten-cell row from nothing. -/
theorem no_clear_on_empty {pl : Placement} :
    Board.fullRows GameConfig.standard (pl.place Board.empty) = ∅ := by
  classical
  rw [Finset.eq_empty_iff_forall_notMem]
  intro r hr
  have h := cleared_row_pre_count_ge (cfg := GameConfig.standard)
    (b := Board.empty) hr
  rw [GameConfig.standard_cols] at h
  have hz : Board.rowCount Board.empty r = 0 := by
    simp [Board.rowCount, Board.empty]
  omega

/-- **The first move never clears**: every game opens with a clear-free
placement — `cleared 1 = 0` for every policy. The base case of every
clearing-rate induction, pinned. -/
theorem cleared_one_eq_zero (π : Policy GameConfig.standard) :
    cleared GameConfig.standard π GameState.init 1 = 0 := by
  have h : (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init 0)).place
        (trace GameConfig.standard π GameState.init 0).board)).card = 0 :=
    congrArg Finset.card
      (no_clear_on_empty
        (pl := π (trace GameConfig.standard π GameState.init 0)))
  rw [show (1 : ℕ) = 0 + 1 from rfl, cleared_succ]
  simp only [cleared_zero]
  omega

/-- A row never holds more cells than the whole board. -/
theorem rowCount_le_count (b : Board) (r : ℕ) :
    b.rowCount r ≤ b.count :=
  Finset.card_le_card (Finset.filter_subset _ _)

/-- **No clears in the opening two moves**: a cleared row needs six prior
cells in one row, but after one placement the whole board holds only four —
`cleared 2 = 0` for every valid policy. With `cleared_one_eq_zero`, the
earliest a game can possibly clear is its third placement. -/
theorem cleared_two_eq_zero {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) :
    cleared GameConfig.standard π GameState.init 2 = 0 := by
  classical
  have h1 := cleared_one_eq_zero π
  have hcount : (trace GameConfig.standard π GameState.init 1).board.count
      = 4 := by
    have h := init_ledger (cfg := GameConfig.standard) hv 1
    rw [GameConfig.standard_cols, h1] at h
    omega
  have hnone : (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init 1)).place
        (trace GameConfig.standard π GameState.init 1).board)) = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro r hr
    have hpre := cleared_row_pre_count_ge (cfg := GameConfig.standard) hr
    rw [GameConfig.standard_cols] at hpre
    have hle := rowCount_le_count
      (trace GameConfig.standard π GameState.init 1).board r
    omega
  rw [show (2 : ℕ) = 1 + 1 from rfl, cleared_succ, hnone, Finset.card_empty,
    h1]

/-- **The opening clear schedule**: a `k`-clear at step `m` obeys
`10k ≤ 4m + 4` — clearing needs banked mass and mass accrues four cells a
move. Singles from move 2, doubles from move 4, triples from move 6. -/
theorem earliest_clear_law {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    10 * (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card
      ≤ 4 * m + 4 := by
  have hmass := clear_requires_mass hv m
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- **The earliest tetris is the tenth placement**: a four-clear stands on
thirty-six banked cells, and nine moves is the soonest the board can hold
them — no game tetrises before step nine. -/
theorem earliest_tetris_step {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    9 ≤ m := by
  have h36 := tetris_requires_thirtysix hv h4
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- **A tetris taxes the whole opening**: a four-clear at step `m` forces
`10·cleared(m) + 36 ≤ 4m` — the thirty-six-cell well bill and every prior
clear's ten-cell bill are both financed by the same four-cells-a-move
income. In particular a tetris at steps nine through eleven demands a
perfectly clear-free opening. -/
theorem tetris_dry_opening {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    10 * cleared GameConfig.standard π GameState.init m + 36 ≤ 4 * m := by
  have h36 := tetris_requires_thirtysix hv h4
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- **The earliest tetris caps the earliest opening**: a tetris at step
nine, ten or eleven can only follow a game that has cleared nothing at
all. -/
theorem earliest_tetris_needs_dry_opening {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : m ≤ 11)
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    cleared GameConfig.standard π GameState.init m = 0 := by
  have h := tetris_dry_opening hv h4
  omega

/-- **Live games must clear**: at any live step, `4m ≤ 10·cleared + 200` —
the delivered mass has nowhere to stand but the 200-cell board and the
clearing ledger. The floor to `cleared ≤ 2m/5`'s ceiling, valid at every
step (not only bag boundaries). -/
theorem live_clear_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) :
    4 * m ≤ 10 * cleared GameConfig.standard π GameState.init m + 200 := by
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  have hcap := count_lt_two_hundred_one hv hlive
  omega

/-- **The clearing pinch**: every live game's cleared total is trapped in
the band `(4m − 200)/10 ≤ cleared ≤ 4m/10` — a window of exactly twenty
rows around the exact `0.4`-per-move line, at every horizon. -/
theorem cleared_pinch {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) :
    4 * m ≤ 10 * cleared GameConfig.standard π GameState.init m + 200
      ∧ 10 * cleared GameConfig.standard π GameState.init m ≤ 4 * m := by
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  have hcap := count_lt_two_hundred_one hv hlive
  exact ⟨by omega, by omega⟩

/-- **Perfect clears keep the mass clock's beat**: the board can only be
empty at step indices divisible by five — `count = 0` forces
`4n ≡ 0 (mod 10)`. -/
theorem perfect_clear_step_mod_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hempty : (trace GameConfig.standard π GameState.init n).board.count
      = 0) :
    5 ∣ n := by
  have h := trace_board_count_mod_ten hv n
  rw [hempty] at h
  omega

/-- **Returns to the very start are quantised**: the trace revisits the
exact initial state — empty board, full bag — only at multiples of 35.
The empty-board reset, if a policy ever achieves one with a full bag, is
locked to the five-bag grid. -/
theorem init_revisit_thirtyfive_dvd {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hret : trace GameConfig.standard π GameState.init n = GameState.init) :
    35 ∣ n := by
  have h := thirtyfive_dvd_of_trace_eq_from (π := π)
    (g0 := GameState.init) (fun k => hv _)
    (GameState.init_board_wf GameConfig.standard) hdraw
    (Nat.zero_le n)
    (by rw [trace_zero]; exact hret.symm)
  simpa using h

/-- **A perfect clear settles the ledger exactly**: with the board empty,
`10·cleared = 4n` on the nose — a perfectly cleared game has run at
*exactly* the 0.4-rows-per-move rate, no slack term at all. -/
theorem perfect_clear_exact_rate {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hempty : (trace GameConfig.standard π GameState.init n).board.count
      = 0) :
    10 * cleared GameConfig.standard π GameState.init n = 4 * n := by
  have hled := init_ledger (cfg := GameConfig.standard) hv n
  rw [GameConfig.standard_cols] at hled
  omega

/-- The first possible perfect clear is move five: five pieces, two full
rows, nothing left. -/
theorem perfect_clear_ge_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ} (hn : 0 < n)
    (hempty : (trace GameConfig.standard π GameState.init n).board.count
      = 0) :
    5 ≤ n := by
  have h := perfect_clear_step_mod_five hv hempty
  omega

/-- A column holds at most as many cells as its height: its cells sit at
distinct rows strictly below the height. -/
theorem colCount_le_colHeight (b : Board) (j : ℕ) :
    b.colCount j ≤ b.colHeight j := by
  classical
  unfold Board.colCount
  calc (b.filter (fun p => p.1 = j)).card
      ≤ (Finset.range (b.colHeight j)).card := by
        refine Finset.card_le_card_of_injOn (fun p => p.2) ?_ ?_
        · intro p hp
          simp only [Finset.mem_coe, Finset.mem_filter] at hp
          have hmem : (j, p.2) ∈ b := by
            have hpe : p = (j, p.2) := Prod.ext hp.2 rfl
            rw [← hpe]
            exact hp.1
          exact Finset.mem_range.mpr (Board.lt_colHeight hmem)
        · intro p hp q hq hpq
          simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
          apply Prod.ext
          · rw [hp.2, hq.2]
          · exact hpq
      _ = b.colHeight j := Finset.card_range _

/-- **Mass forces height**: some column of a well-formed board reaches at
least a tenth of the total mass — `D` cells cannot lie flatter than
`D/10`. With the mass band, cycle boards carry a *standing* skyline. -/
theorem exists_tall_column {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    ∃ j < 10, b.count ≤ 10 * b.colHeight j := by
  classical
  obtain ⟨j, hj, hmax⟩ := Finset.exists_max_image (Finset.range 10)
    b.colHeight ⟨0, by simp⟩
  refine ⟨j, Finset.mem_range.mp hj, ?_⟩
  have hsum := Board.sum_colCount (cfg := GameConfig.standard) hwf
  rw [GameConfig.standard_cols] at hsum
  have h1 : b.count ≤ ∑ j' ∈ Finset.range 10, b.colHeight j' := by
    rw [← hsum]
    exact Finset.sum_le_sum (fun j' _ => colCount_le_colHeight b j')
  have h2 : ∑ j' ∈ Finset.range 10, b.colHeight j'
      ≤ (Finset.range 10).card * b.colHeight j := by
    rw [← smul_eq_mul]
    exact Finset.sum_le_card_nsmul _ _ _ (fun j' hj' => hmax j' hj')
  rw [Finset.card_range] at h2
  omega

/-- **A tetris needs four rows of relief**: some pair of columns differs in
height by at least four at every four-clear. -/
theorem tetris_relief_ge_four {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ j < 10, ∃ j' < 10, b.colHeight j + 4 ≤ b.colHeight j' := by
  obtain ⟨c₀, hc₀, hdepth⟩ := tetris_well_depth hwf hv hnf h4
  set c := (c₀ + 1) % 10 with hc
  have hclt : c < 10 := Nat.mod_lt _ (by omega)
  have hcne : c ≠ c₀ := by omega
  exact ⟨c₀, hc₀, c, hclt, hdepth c hclt hcne⟩

/-- **Flat boards cannot tetris**: if all ten columns stand at one height,
no placement clears four — the skyline must be broken before it can be
harvested. Flat-stacking strategies structurally forfeit the tetris. -/
theorem no_tetris_on_flat {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hflat : ∀ j < 10, ∀ j' < 10, b.colHeight j = b.colHeight j') :
    (Board.fullRows GameConfig.standard (pl.place b)).card ≠ 4 := by
  intro h4
  obtain ⟨j, hj, j', hj', hrel⟩ := tetris_relief_ge_four hwf hv hnf h4
  have := hflat j hj j' hj'
  omega

/-- **Unfed columns overtop the landing site by the clear count**: a
`k`-clear's untouched columns rise at least `k` rows above the drop
offset — they reach through all `k` completed rows, whose span pushes the
topmost at least `k − 1` above the base. Sharpens
`clear_untouched_column_height` from `+1` to `+k`; at `k = 4` it recovers
the tetris well-depth geometry for the nine unfed columns. -/
theorem clear_untouched_column_height_ge {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (hpos : 0 < (Board.fullRows cfg (pl.place b)).card) {c : ℕ}
    (hc : c < cfg.cols) (hz : pl.colProfile c = 0) :
    pl.dropOffset b + (Board.fullRows cfg (pl.place b)).card
      ≤ b.colHeight c := by
  classical
  have hne : (Board.fullRows cfg (pl.place b)).Nonempty :=
    Finset.card_pos.mp hpos
  set rtop := (Board.fullRows cfg (pl.place b)).max' hne with hrtop
  have hrmem : rtop ∈ Board.fullRows cfg (pl.place b) :=
    Finset.max'_mem _ hne
  have hdle : pl.dropOffset b ≤ rtop :=
    (clear_rows_in_drop_window hnf hrmem).1
  have hsub : Board.fullRows cfg (pl.place b)
      ⊆ Finset.Icc (pl.dropOffset b) rtop := by
    intro r hr
    rw [Finset.mem_Icc]
    exact ⟨(clear_rows_in_drop_window hnf hr).1, Finset.le_max' _ _ hr⟩
  have hcard := Finset.card_le_card hsub
  rw [Nat.card_Icc] at hcard
  have hfull := Board.isFull_of_mem_fullRows hrmem
  have hmem := hfull c (Finset.mem_range.mpr hc)
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
  rcases hmem with h | h
  · have := Board.lt_colHeight h
    omega
  · exfalso
    have hprof : ((pl.dropped b).filter (fun p => p.1 = c)).card
        = pl.colProfile c := by
      have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) c
      unfold Board.colCount at hcc
      unfold Placement.dropped
      exact hcc
    have hfib : ((pl.dropped b).filter (fun p => p.1 = c)).Nonempty :=
      ⟨(c, rtop), Finset.mem_filter.mpr ⟨h, rfl⟩⟩
    have := Finset.card_pos.mpr hfib
    omega

/-- **Fed columns sit at or below the landing**: every column the piece
occupies has height at most `dropOffset + cell-row` — the drop offset is
the supremum of the per-cell falls, so no supporting stack can poke past
where its cell comes to rest. -/
theorem fed_column_height_le {b : Board} {pl : Placement} :
    ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + cell.2 := by
  intro cell hcell
  have h := Finset.le_sup (f := fun cell =>
    b.colHeight (pl.col + cell.1) - cell.2) hcell
  have : b.colHeight (pl.col + cell.1) - cell.2 ≤ pl.dropOffset b := h
  omega

/-- Every fed column stops within three rows of the drop offset — the dual
of `clear_untouched_column_height_ge`: at a `k`-clear the board splits
into fed columns capped at `dropOffset + 3` and unfed columns reaching
`dropOffset + k`. -/
theorem fed_column_height_le_three {b : Board} {pl : Placement} :
    ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + 3 := by
  intro cell hcell
  have h := fed_column_height_le (b := b) cell hcell
  have hb := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  omega

/-- On the empty board every piece falls to the floor: the drop offset is
zero. -/
theorem dropOffset_empty (pl : Placement) :
    pl.dropOffset Board.empty = 0 := by
  unfold Placement.dropOffset
  refine Nat.le_antisymm ?_ (Nat.zero_le _)
  apply Finset.sup_le
  intro cell _
  simp [Board.empty, Board.colHeight_empty]

/-- **The first piece lies in the bottom four rows**: placing on the empty
board leaves every cell strictly below row four. -/
theorem place_empty_low {pl : Placement} :
    ∀ q ∈ pl.place Board.empty, q.2 < 4 := by
  intro q hq
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hq
  rcases hq with h | h
  · exact absurd h (by simp [Board.empty])
  · rw [Placement.dropped_eq_image, Finset.mem_image] at h
    obtain ⟨cell, hcell, hEq⟩ := h
    have hrow : pl.dropOffset Board.empty + cell.2 = q.2 :=
      congrArg Prod.snd hEq
    have hb := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
    have hz := dropOffset_empty pl
    omega

/-- **One move lifts the skyline by at most four**: after placing, every
column's height is within four of some column's height *before* the move —
a piece can bridge from a tall stack, but it cannot levitate more than its
own four rows above what already stood. -/
theorem place_colHeight_le {b : Board} {pl : Placement} (j : ℕ) :
    ∃ j', (pl.place b).colHeight j ≤ b.colHeight j' + 4 := by
  classical
  have hup : (pl.place b).colHeight j
      ≤ max (b.colHeight j) (pl.dropOffset b + 4) := by
    unfold Board.colHeight
    apply Finset.sup_le
    intro r hr
    unfold Board.colRows at hr
    rw [Finset.mem_image] at hr
    obtain ⟨q, hq, rfl⟩ := hr
    rw [Finset.mem_filter] at hq
    obtain ⟨hqb, hqj⟩ := hq
    rw [Placement.place_eq_union_dropped, Finset.mem_union] at hqb
    rcases hqb with h | h
    · have hmem : (j, q.2) ∈ b := by
        have hqe : q = (j, q.2) := Prod.ext hqj rfl
        rw [← hqe]
        exact h
      have := Board.lt_colHeight hmem
      refine le_trans ?_ (le_max_left _ _)
      exact this
    · rw [Placement.dropped_eq_image, Finset.mem_image] at h
      obtain ⟨cell, hcell, hEq⟩ := h
      have hrow : pl.dropOffset b + cell.2 = q.2 := congrArg Prod.snd hEq
      have hb4 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
      refine le_trans ?_ (le_max_right _ _)
      change q.2 + 1 ≤ pl.dropOffset b + 4
      omega
  have hne : pl.shapeUp.Nonempty := by
    apply Finset.card_pos.mp
    rw [pl.shapeUp_card]
    omega
  obtain ⟨cell₀, hcell₀, hsup⟩ := Finset.exists_mem_eq_sup pl.shapeUp hne
    (fun cell => b.colHeight (pl.col + cell.1) - cell.2)
  have hd : pl.dropOffset b
      = b.colHeight (pl.col + cell₀.1) - cell₀.2 := by
    unfold Placement.dropOffset
    exact hsup
  by_cases hc : (pl.place b).colHeight j ≤ b.colHeight j + 4
  · exact ⟨j, hc⟩
  · refine ⟨pl.col + cell₀.1, ?_⟩
    omega

/-- The full move (place + clear) also lifts the skyline by at most four:
clearing only lowers columns. -/
theorem applyStep_colHeight_le {cfg : GameConfig} {b : Board}
    {pl : Placement} (j : ℕ) :
    ∃ j', (Placement.applyStep cfg b pl).colHeight j
      ≤ b.colHeight j' + 4 := by
  obtain ⟨j', h⟩ := place_colHeight_le (b := b) (pl := pl) j
  refine ⟨j', ?_⟩
  have hcl := colHeight_clearLines_le cfg (pl.place b) j
  unfold Placement.applyStep
  omega

/-- **The skyline climbs at most four per step along any trace**: each
successor board's columns stay within four rows of some column of the
predecessor — height spikes are rate-limited by the geometry of a single
piece. -/
theorem trace_succ_colHeight_le {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} (n j : ℕ) :
    ∃ j', (trace cfg π g0 (n + 1)).board.colHeight j
      ≤ (trace cfg π g0 n).board.colHeight j' + 4 := by
  rw [trace_succ, GameState.step_board]
  exact applyStep_colHeight_le j

/-- **The tetris pair law**: two four-clears at steps `m < m'` satisfy
`76 ≤ count(m) + 4·(m' − m)` — the first tetris burns thirty-six banked
cells plus its forty-cell bill, and the second needs its own thirty-six
back. A tetris fired from a lean 36-cell board pushes the next at least
ten moves out; only a rich board can fire twice in quick succession. -/
theorem tetris_pair_mass_law {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m m' : ℕ} (hmm : m < m')
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4)
    (h4' : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m')).place
          (trace GameConfig.standard π GameState.init m').board)).card
      = 4) :
    76 ≤ (trace GameConfig.standard π GameState.init m).board.count
      + 4 * (m' - m) := by
  have h36' := tetris_requires_thirtysix hv h4'
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  have hled' := init_ledger (cfg := GameConfig.standard) hv m'
  rw [GameConfig.standard_cols] at hled hled'
  have hsucc := cleared_succ GameConfig.standard π GameState.init m
  rw [h4] at hsucc
  have hmono := cleared_mono GameConfig.standard π GameState.init
    (show m + 1 ≤ m' by omega)
  omega

/-- **The window clearing band**: across any `w`-move window with live
endpoints, the rows cleared sit within one boardful of the exact
0.4-per-move line — `4w − 200 ≤ 10·Δcleared ≤ 4w + 200`. The per-window
form of the pinch, at every position and scale. -/
theorem cleared_window_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m w : ℕ}
    (hlive_m : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard)
    (hlive_mw : ¬ (trace GameConfig.standard π GameState.init (m + w)).lost
      GameConfig.standard) :
    4 * w ≤ 10 * (cleared GameConfig.standard π GameState.init (m + w)
        - cleared GameConfig.standard π GameState.init m) + 200
      ∧ 10 * (cleared GameConfig.standard π GameState.init (m + w)
        - cleared GameConfig.standard π GameState.init m)
        ≤ 4 * w + 200 := by
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  have hled' := init_ledger (cfg := GameConfig.standard) hv (m + w)
  rw [GameConfig.standard_cols] at hled hled'
  have hcap := count_lt_two_hundred_one hv hlive_m
  have hcap' := count_lt_two_hundred_one hv hlive_mw
  have hclm := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right m w)
  exact ⟨by omega, by omega⟩

/-- **Hard drops never burrow**: every cell of the dropped piece lands at
or above its own column's stack top — the drop offset is the supremum of
the per-cell falls, so no cell slides in beneath what already stands.
The fundamental no-interleaving property of the hard-drop model. -/
theorem dropped_above_own_column {b : Board} {pl : Placement} :
    ∀ q ∈ pl.dropped b, b.colHeight q.1 ≤ q.2 := by
  intro q hq
  rw [Placement.dropped_eq_image, Finset.mem_image] at hq
  obtain ⟨cell, hcell, rfl⟩ := hq
  have h := Finset.le_sup (f := fun cell =>
    b.colHeight (pl.col + cell.1) - cell.2) hcell
  have hle : b.colHeight (pl.col + cell.1) - cell.2 ≤ pl.dropOffset b := h
  change b.colHeight (pl.col + cell.1) ≤ pl.dropOffset b + cell.2
  omega

/-- **The landing is exact**: after placing, a fed column's height equals
`dropOffset + (its top piece cell) + 1` — old stack capped below the cell
(`fed_column_height_le`), other piece cells capped by topness, the top
cell itself delivers. The complete post-move height formula for every
column the piece touches. -/
theorem place_fed_colHeight_eq {b : Board} {pl : Placement} {cell : Coord}
    (hcell : cell ∈ pl.shapeUp)
    (htop : ∀ cell' ∈ pl.shapeUp, cell'.1 = cell.1 → cell'.2 ≤ cell.2) :
    (pl.place b).colHeight (pl.col + cell.1)
      = pl.dropOffset b + cell.2 + 1 := by
  classical
  refine Nat.le_antisymm ?_ ?_
  · unfold Board.colHeight
    apply Finset.sup_le
    intro r hr
    unfold Board.colRows at hr
    rw [Finset.mem_image] at hr
    obtain ⟨q, hq, rfl⟩ := hr
    rw [Finset.mem_filter] at hq
    obtain ⟨hqb, hqj⟩ := hq
    rw [Placement.place_eq_union_dropped, Finset.mem_union] at hqb
    rcases hqb with h | h
    · have hmem : (pl.col + cell.1, q.2) ∈ b := by
        have hqe : q = (pl.col + cell.1, q.2) := Prod.ext hqj rfl
        rw [← hqe]
        exact h
      have h1 := Board.lt_colHeight hmem
      have h2 := fed_column_height_le (b := b) cell hcell
      change q.2 + 1 ≤ pl.dropOffset b + cell.2 + 1
      omega
    · rw [Placement.dropped_eq_image, Finset.mem_image] at h
      obtain ⟨cell', hcell', hEq⟩ := h
      have hcol : pl.col + cell'.1 = q.1 := congrArg Prod.fst hEq
      have hrow : pl.dropOffset b + cell'.2 = q.2 := congrArg Prod.snd hEq
      have hc1 : cell'.1 = cell.1 := by omega
      have := htop cell' hcell' hc1
      change q.2 + 1 ≤ pl.dropOffset b + cell.2 + 1
      omega
  · have hmem : (pl.col + cell.1, pl.dropOffset b + cell.2)
        ∈ pl.place b := by
      rw [Placement.place_eq_union_dropped, Finset.mem_union]
      right
      rw [Placement.dropped_eq_image, Finset.mem_image]
      exact ⟨cell, hcell, rfl⟩
    exact Board.lt_colHeight hmem

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
