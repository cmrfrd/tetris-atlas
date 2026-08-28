import Mathlib
import Proofs.Survival.ClearDeviation
import Proofs.Invariants.ColumnCount
import Proofs.Invariants.Holes
import Proofs.Invariants.HoleDebt
import Proofs.Invariants.SurfaceCalculus

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

/-- **Placing leaves unfed columns untouched**: a column the piece does
not feed keeps its exact height through the merge — with
`place_fed_colHeight_eq`, the complete skyline update of a placement is
now determined column by column. -/
theorem place_unfed_colHeight_eq {b : Board} {pl : Placement} {c : ℕ}
    (hz : pl.colProfile c = 0) :
    (pl.place b).colHeight c = b.colHeight c := by
  classical
  have hprof : ((pl.dropped b).filter (fun p => p.1 = c)).card
      = pl.colProfile c := by
    have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) c
    unfold Board.colCount at hcc
    unfold Placement.dropped
    exact hcc
  have hempty : (pl.dropped b).filter (fun p => p.1 = c) = ∅ :=
    Finset.card_eq_zero.mp (by omega)
  unfold Board.colHeight Board.colRows
  rw [Placement.place_eq_union_dropped, Finset.filter_union, hempty,
    Finset.union_empty]

/-- **The drop reads only the surface**: two boards agreeing on the
heights of the piece's columns give the same drop offset — holes, and
everything below the skyline, are invisible to the falling piece. -/
theorem dropOffset_eq_of_colHeight_eq {b b' : Board} {pl : Placement}
    (h : ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) = b'.colHeight (pl.col + cell.1)) :
    pl.dropOffset b = pl.dropOffset b' := by
  unfold Placement.dropOffset
  apply Finset.sup_congr rfl
  intro cell hcell
  rw [h cell hcell]

/-- **Placement is surface-determined**: the dropped cells themselves
depend only on the columns' heights — the same move on two boards with
the same skyline lands identically, whatever lies buried beneath. The
hole-independence of placement, certified in the main bank. -/
theorem dropped_eq_of_colHeight_eq {b b' : Board} {pl : Placement}
    (h : ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) = b'.colHeight (pl.col + cell.1)) :
    pl.dropped b = pl.dropped b' := by
  unfold Placement.dropped
  rw [dropOffset_eq_of_colHeight_eq h]

/-- **Clearing is NOT surface-determined**: two boards with identical
skylines and identical landed cells can clear *different* rows — a
kernel-checked witness (a solid 2-stack versus the same stack with a
buried hole; the vertical I completes two rows on one and only one on the
other). The exact complement of `dropped_eq_of_colHeight_eq`: the piece
reads only the surface, but the clears read the holes. Any faithful
surface-only abstraction of Tetris must therefore lose the clear ledger. -/
theorem clears_not_surface_determined :
    ∃ (cfg : GameConfig) (b b' : Board) (pl : Placement),
      (∀ j, j < cfg.cols → b.colHeight j = b'.colHeight j)
      ∧ pl.dropped b = pl.dropped b'
      ∧ Board.fullRows cfg (pl.place b)
        ≠ Board.fullRows cfg (pl.place b') := by
  refine ⟨⟨2, 4, by omega, by omega⟩,
    ({(0, 0), (0, 1)} : Finset Coord), ({(0, 1)} : Finset Coord),
    ⟨Piece.I, 1, 1⟩, ?_, ?_, ?_⟩
  · decide
  · decide
  · decide

/-- **More material can mean fewer clears**: adding one cell to a board
can destroy a clear the same placement would have made — the extra cell
lifts the landing above the row it would have completed. Clearing is not
monotone in the board; kernel-checked witness. Any domination argument
of the form "a fuller board clears at least as much" is dead on arrival. -/
theorem clears_not_monotone :
    ∃ (cfg : GameConfig) (b b' : Board) (pl : Placement),
      b ⊆ b'
      ∧ ¬ (Board.fullRows cfg (pl.place b)
        ⊆ Board.fullRows cfg (pl.place b')) := by
  refine ⟨⟨2, 8, by omega, by omega⟩,
    ({(0, 0)} : Finset Coord), ({(0, 0), (1, 1)} : Finset Coord),
    ⟨Piece.I, 1, 1⟩, ?_, ?_⟩
  · decide
  · decide

/-- Column heights are monotone in the board. -/
theorem colHeight_mono {b b' : Board} (h : b ⊆ b') (j : ℕ) :
    b.colHeight j ≤ b'.colHeight j := by
  unfold Board.colHeight
  apply Finset.sup_mono
  unfold Board.colRows
  exact Finset.image_subset_image (Finset.filter_subset_filter _ h)

/-- **Pieces land higher on fuller boards**: the drop offset is monotone
in the board — the one clearing-adjacent quantity that *is* monotone,
in contrast to the clears themselves (`clears_not_monotone`). -/
theorem dropOffset_mono {b b' : Board} {pl : Placement} (h : b ⊆ b') :
    pl.dropOffset b ≤ pl.dropOffset b' := by
  unfold Placement.dropOffset
  apply Finset.sup_mono_fun
  intro cell _
  have := colHeight_mono h (pl.col + cell.1)
  omega

/-- The drop offset is bounded by any bound on the fed columns' heights. -/
theorem dropOffset_le_of_heights {b : Board} {pl : Placement} {H : ℕ}
    (h : ∀ cell ∈ pl.shapeUp, b.colHeight (pl.col + cell.1) ≤ H) :
    pl.dropOffset b ≤ H := by
  unfold Placement.dropOffset
  apply Finset.sup_le
  intro cell hcell
  have := h cell hcell
  omega

/-- A low landing keeps the merge in the field: if the piece comes to rest
with headroom, every cell of the placed board stays below the ceiling. -/
theorem place_in_field_of_low_drop {cfg : GameConfig} {b : Board}
    {pl : Placement} (hif : ∀ p ∈ b, p.2 < cfg.rows)
    (hd : pl.dropOffset b + 3 < cfg.rows) :
    ∀ p ∈ pl.place b, p.2 < cfg.rows := by
  intro p hp
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at hp
  rcases hp with h | h
  · exact hif p h
  · rw [Placement.dropped_eq_image, Finset.mem_image] at h
    obtain ⟨cell, hcell, hEq⟩ := h
    have hrow : pl.dropOffset b + cell.2 = p.2 := congrArg Prod.snd hEq
    have hb := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
    omega

/-- **Four rows of headroom make a move top-out-free**: if every column
the piece touches stands at least four rows below the ceiling, the merge
stays entirely in the field — the exact safety margin a solver must
preserve to guarantee its next placement cannot lose. -/
theorem place_safe_of_low_skyline {cfg : GameConfig} {b : Board}
    {pl : Placement} (hif : ∀ p ∈ b, p.2 < cfg.rows)
    (hsky : ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) + 4 ≤ cfg.rows) :
    ∀ p ∈ pl.place b, p.2 < cfg.rows := by
  apply place_in_field_of_low_drop hif
  have hd : pl.dropOffset b ≤ cfg.rows - 4 := by
    apply dropOffset_le_of_heights
    intro cell hcell
    have := hsky cell hcell
    omega
  have hne : pl.shapeUp.Nonempty := by
    apply Finset.card_pos.mp
    rw [pl.shapeUp_card]
    omega
  obtain ⟨cell₀, hcell₀⟩ := hne
  have h4 := hsky cell₀ hcell₀
  omega

/-- Clearing keeps the board in the field: cleared cells only move down. -/
theorem clearLines_in_field {cfg : GameConfig} {b : Board}
    (hif : ∀ p ∈ b, p.2 < cfg.rows) :
    ∀ p ∈ Board.clearLines cfg b, p.2 < cfg.rows := by
  intro p hp
  have h1 : p.2 < (Board.clearLines cfg b).colHeight p.1 :=
    Board.lt_colHeight hp
  have h2 := colHeight_clearLines_le cfg b p.1
  have h3 := Board.colHeight_le_rows_of_in_field hif p.1
  omega

/-- **The full move is safe under four rows of headroom**: place and clear
together keep every cell below the ceiling whenever the touched columns
stand at least four rows short of it — the per-move safety certificate a
solver can check in O(4) height reads. -/
theorem applyStep_safe_of_low_skyline {cfg : GameConfig} {b : Board}
    {pl : Placement} (hif : ∀ p ∈ b, p.2 < cfg.rows)
    (hsky : ∀ cell ∈ pl.shapeUp,
      b.colHeight (pl.col + cell.1) + 4 ≤ cfg.rows) :
    ∀ p ∈ Placement.applyStep cfg b pl, p.2 < cfg.rows := by
  intro p hp
  unfold Placement.applyStep at hp
  exact clearLines_in_field (place_safe_of_low_skyline hif hsky) p hp

/-- **Perpetual headroom is perpetual survival**: a policy that, at every
state it reaches, drops only onto columns standing at least four rows
below the ceiling, never loses — the per-move safety certificate chains
into a full `SurvivesForever` proof by induction. Reduces the M1/M4 goal
to maintaining one O(4)-checkable invariant. -/
theorem survivesForever_of_headroom {cfg : GameConfig} {π : Policy cfg}
    (hsky : ∀ n, ∀ cell ∈ (π (trace cfg π GameState.init n)).shapeUp,
      (trace cfg π GameState.init n).board.colHeight
          ((π (trace cfg π GameState.init n)).col + cell.1) + 4
        ≤ cfg.rows) :
    SurvivesForever cfg π GameState.init := by
  have hif : ∀ n, ∀ p ∈ (trace cfg π GameState.init n).board,
      p.2 < cfg.rows := by
    intro n
    induction n with
    | zero =>
      intro p hp
      exact absurd hp (Finset.notMem_empty p)
    | succ k ih =>
      rw [trace_succ, GameState.step_board]
      exact applyStep_safe_of_low_skyline ih (hsky k)
  intro n
  rw [GameState.not_lost_iff_forall_row_lt]
  exact hif n

/-- One live step under headroom stays live. -/
theorem step_live_of_headroom {cfg : GameConfig} {g : GameState}
    {pl : Placement} (hlive : ¬ g.lost cfg)
    (hsky : ∀ cell ∈ pl.shapeUp,
      g.board.colHeight (pl.col + cell.1) + 4 ≤ cfg.rows) :
    ¬ (g.step cfg pl).lost cfg := by
  rw [GameState.not_lost_iff_forall_row_lt] at hlive ⊢
  rw [GameState.step_board]
  exact applyStep_safe_of_low_skyline hlive hsky

/-- **Death requires a high column**: a live state that loses on its next
move must have dropped onto a column within four rows of the ceiling —
every top-out is a drop into a nearly-full column, never an accident of a
low board. The complete diagnosis of the loss event. -/
theorem lost_step_requires_high_column {cfg : GameConfig} {g : GameState}
    {pl : Placement} (hlive : ¬ g.lost cfg)
    (hlost : (g.step cfg pl).lost cfg) :
    ∃ cell ∈ pl.shapeUp,
      cfg.rows < g.board.colHeight (pl.col + cell.1) + 4 := by
  by_contra hcon
  push Not at hcon
  exact step_live_of_headroom hlive
    (fun cell hcell => hcon cell hcell) hlost

/-- Every piece has a rotation at most two columns wide (the vertical
orientations). -/
theorem exists_narrow_rotation :
    ∀ p : Piece, ∃ r : Rotation, ∀ cell ∈ p.shapeUp r, cell.1 < 2 := by
  decide

/-- **Two low neighbours guarantee a headroom move for every piece**: if
some adjacent column pair stands at least four rows below the ceiling,
then whatever piece is dealt admits a valid placement touching only those
columns — the availability half of the headroom survival reduction. A
solver never runs out of safe moves while it keeps one low two-column
window anywhere on the board. -/
theorem headroom_move_exists {b : Board} {j : ℕ} (hj : j + 1 < 10)
    (h1 : b.colHeight j + 4 ≤ 20) (h2 : b.colHeight (j + 1) + 4 ≤ 20)
    (p : Piece) :
    ∃ pl : Placement, pl.piece = p
      ∧ pl.Valid GameConfig.standard
      ∧ ∀ cell ∈ pl.shapeUp,
        b.colHeight (pl.col + cell.1) + 4 ≤ GameConfig.standard.rows := by
  obtain ⟨r, hr⟩ := exists_narrow_rotation p
  refine ⟨⟨p, r, j⟩, rfl, ?_, ?_⟩
  · intro cell hcell
    have hw := hr cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · intro cell hcell
    have hw := hr cell hcell
    change b.colHeight (j + cell.1) + 4 ≤ GameConfig.standard.rows
    rw [GameConfig.standard_rows]
    rcases (show cell.1 = 0 ∨ cell.1 = 1 by omega) with h | h <;> rw [h]
    · simpa using h1
    · exact h2

/-- **No safe window means half the board is high**: a standard board with
no adjacent pair of columns four below the ceiling has at least five
columns of height seventeen or more — each of the five disjoint pairs
must contribute one. The dichotomy behind headroom availability: either
`headroom_move_exists` applies somewhere, or the board is already half
towers. -/
theorem no_low_pair_five_high {b : Board}
    (h : ∀ j, j + 1 < 10 →
      ¬ (b.colHeight j + 4 ≤ 20 ∧ b.colHeight (j + 1) + 4 ≤ 20)) :
    5 ≤ ((Finset.range 10).filter
      (fun j => 17 ≤ b.colHeight j)).card := by
  classical
  have hpair : ∀ i, i < 5 →
      17 ≤ b.colHeight (2 * i) ∨ 17 ≤ b.colHeight (2 * i + 1) := by
    intro i hi
    have := h (2 * i) (by omega)
    omega
  have hmem : ∀ i ∈ Finset.range 5,
      (if 17 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1)
        ∈ (Finset.range 10).filter (fun j => 17 ≤ b.colHeight j) := by
    intro i hi
    rw [Finset.mem_range] at hi
    rw [Finset.mem_filter, Finset.mem_range]
    by_cases hc : 17 ≤ b.colHeight (2 * i)
    · rw [if_pos hc]
      exact ⟨by omega, hc⟩
    · have h1 : 17 ≤ b.colHeight (2 * i + 1) := by
        rcases hpair i hi with h' | h'
        · exact absurd h' hc
        · exact h'
      rw [if_neg hc]
      exact ⟨by omega, h1⟩
  have hinj : Set.InjOn
      (fun i => if 17 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1)
      ↑(Finset.range 5) := by
    intro i hi j hj hij
    simp only [Finset.mem_coe, Finset.mem_range] at hi hj
    simp only [] at hij
    have h1 : (if 17 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1) / 2
        = i := by
      split_ifs <;> omega
    have h2 : (if 17 ≤ b.colHeight (2 * j) then 2 * j else 2 * j + 1) / 2
        = j := by
      split_ifs <;> omega
    rw [hij] at h1
    omega
  calc (5 : ℕ) = (Finset.range 5).card := (Finset.card_range 5).symm
    _ ≤ _ := Finset.card_le_card_of_injOn _ hmem hinj

/-- Every tetromino's column fiber is vertically contiguous — no piece has
a gap inside one of its own columns (28-case check). -/
theorem shape_col_fiber_contiguous :
    ∀ p : Piece, ∀ r : Rotation, ∀ t y₁ y₂ y : Fin 4,
      ¬ (((t : ℕ), (y₁ : ℕ)) ∈ p.shapeUp r
        ∧ ((t : ℕ), (y₂ : ℕ)) ∈ p.shapeUp r
        ∧ (y₁ : ℕ) ≤ (y : ℕ) ∧ (y : ℕ) ≤ (y₂ : ℕ)
        ∧ ((t : ℕ), (y : ℕ)) ∉ p.shapeUp r) := by
  decide

/-- Implication form of `shape_col_fiber_contiguous`. -/
theorem shape_col_fiber_contiguous' (p : Piece) (r : Rotation)
    {t y₁ y₂ y : ℕ} (ht : t < 4) (h₁ : y₁ < 4) (h₂ : y₂ < 4) (hy : y < 4)
    (m1 : (t, y₁) ∈ p.shapeUp r) (m2 : (t, y₂) ∈ p.shapeUp r)
    (l1 : y₁ ≤ y) (l2 : y ≤ y₂) : (t, y) ∈ p.shapeUp r := by
  by_contra hnot
  exact shape_col_fiber_contiguous p r ⟨t, ht⟩ ⟨y₁, h₁⟩ ⟨y₂, h₂⟩ ⟨y, hy⟩
    ⟨m1, m2, l1, l2, hnot⟩

/-- **The dropped piece is solid in every column**: between two landed
cells of the same column, every row is landed too — a placement never
sandwiches a fresh gap between its own cells; holes are born only in the
space *under* the piece, never inside it. -/
theorem dropped_fiber_contiguous {b : Board} {pl : Placement}
    {q q' : Coord} (hq : q ∈ pl.dropped b) (hq' : q' ∈ pl.dropped b)
    (hcol : q.1 = q'.1) {r : ℕ} (h1 : q.2 ≤ r) (h2 : r ≤ q'.2) :
    (q.1, r) ∈ pl.dropped b := by
  rw [Placement.dropped_eq_image, Finset.mem_image] at hq hq' ⊢
  obtain ⟨cell, hcell, hEq⟩ := hq
  obtain ⟨cell', hcell', hEq'⟩ := hq'
  have hc1 : pl.col + cell.1 = q.1 := congrArg Prod.fst hEq
  have hr1 : pl.dropOffset b + cell.2 = q.2 := congrArg Prod.snd hEq
  have hc1' : pl.col + cell'.1 = q'.1 := congrArg Prod.fst hEq'
  have hr1' : pl.dropOffset b + cell'.2 = q'.2 := congrArg Prod.snd hEq'
  have ht : cell.1 = cell'.1 := by omega
  have hb1 := Piece.shapeUp_col_lt_four pl.piece pl.rot cell hcell
  have hb2 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  have hb2' := Piece.shapeUp_row_lt_four pl.piece pl.rot cell' hcell'
  refine ⟨(cell.1, r - pl.dropOffset b), ?_, ?_⟩
  · exact shape_col_fiber_contiguous' pl.piece pl.rot hb1 hb2 hb2'
      (by omega) hcell (by rw [ht]; exact hcell') (by omega) (by omega)
  · change (pl.col + cell.1, pl.dropOffset b + (r - pl.dropOffset b))
      = (q.1, r)
    rw [hc1]
    have h2' : pl.dropOffset b + (r - pl.dropOffset b) = r := by omega
    rw [h2']

/-- **The S makes a hole on virgin ground**: dropped flat on the empty
board, the S leaves a covered empty cell — its staggered bottom hangs one
cell over nothing. Kernel-checked witness. -/
theorem S_creates_hole_on_empty :
    ∃ pl : Placement, pl.piece = Piece.S
      ∧ pl.Valid GameConfig.standard
      ∧ ∃ c r r', r < r'
        ∧ (c, r) ∉ pl.place Board.empty
        ∧ (c, r') ∈ pl.place Board.empty := by
  refine ⟨⟨Piece.S, 0, 0⟩, rfl, by decide, 2, 0, 1, by omega,
    by decide, by decide⟩

/-- The Z too, mirrored: even the empty board cannot receive a flat Z
without burying a cell. With `S_creates_hole_on_empty`: the two skew
pieces are hole factories on any flat ground — the geometric seed of the
S/Z pressure every survival argument must absorb. -/
theorem Z_creates_hole_on_empty :
    ∃ pl : Placement, pl.piece = Piece.Z
      ∧ pl.Valid GameConfig.standard
      ∧ ∃ c r r', r < r'
        ∧ (c, r) ∉ pl.place Board.empty
        ∧ (c, r') ∈ pl.place Board.empty := by
  refine ⟨⟨Piece.Z, 0, 0⟩, rfl, by decide, 0, 0, 1, by omega,
    by decide, by decide⟩

/-- **The I never buries a cell on virgin ground**: in every rotation and
column, an I dropped on the empty board is fully grounded — the column
below each placed cell is solid. The clean complement to the S/Z hole
factories. -/
theorem I_grounded_on_empty :
    ∀ rot : Rotation, ∀ col : Fin 10,
      ∀ q ∈ (⟨Piece.I, rot, (col : ℕ)⟩ : Placement).place Board.empty,
        ∀ r < q.2,
          (q.1, r) ∈ (⟨Piece.I, rot, (col : ℕ)⟩ : Placement).place
            Board.empty := by
  decide

/-- The O too: every flat-bottomed drop on empty ground is grounded. Of
the seven pieces, only the skew pair is forced to bury cells on a flat
floor. -/
theorem O_grounded_on_empty :
    ∀ rot : Rotation, ∀ col : Fin 10,
      ∀ q ∈ (⟨Piece.O, rot, (col : ℕ)⟩ : Placement).place Board.empty,
        ∀ r < q.2,
          (q.1, r) ∈ (⟨Piece.O, rot, (col : ℕ)⟩ : Placement).place
            Board.empty := by
  decide

/-- **Clean flat landings exist exactly for the non-skew pieces**: a piece
has a rotation that lands fully grounded at *every* column of the empty
board if and only if it is not S and not Z — the complete classification
of who can keep virgin ground hole-free, as one kernel-checked
equivalence. -/
theorem grounded_rotation_iff_not_skew :
    ∀ p : Piece, (p ≠ Piece.S ∧ p ≠ Piece.Z) ↔
      ∃ rot : Rotation, ∀ col : Fin 10,
        ∀ q ∈ (⟨p, rot, (col : ℕ)⟩ : Placement).place Board.empty,
          ∀ r < q.2,
            (q.1, r) ∈ (⟨p, rot, (col : ℕ)⟩ : Placement).place
              Board.empty := by
  decide

/-- **A hole can never be filled by a drop**: no piece cell ever lands in
a row below its column's height — the space under the skyline is
unreachable to every future placement. -/
theorem hole_never_filled_by_drop {b : Board} {pl : Placement} {c r : ℕ}
    (hr : r < b.colHeight c) : (c, r) ∉ pl.dropped b := by
  intro h
  have h2 : b.colHeight c ≤ r := dropped_above_own_column (c, r) h
  omega

/-- **Holes persist through every placement**: a covered empty cell stays
empty across any merge — only line clearing can ever repair it. The
board-level core of the hole-debt principle: debt rises on placement and
falls only on clears. -/
theorem hole_persists_place {b : Board} {pl : Placement} {c r : ℕ}
    (hempty : (c, r) ∉ b) (hr : r < b.colHeight c) :
    (c, r) ∉ pl.place b := by
  intro h
  rw [Placement.place_eq_union_dropped, Finset.mem_union] at h
  rcases h with h | h
  · exact hempty h
  · exact hole_never_filled_by_drop hr h

/-- **A hole blocks its row**: a row containing a covered empty cell can
never be completed by any placement — the hole is unfillable, so the row
is uncloseable while the cover stands. Holes don't merely cost cells;
they freeze their row out of the clearing economy entirely, and with
`clear_rows_in_drop_window` only clears above can ever release them. -/
theorem hole_blocks_row {cfg : GameConfig} {b : Board} {pl : Placement}
    {c r : ℕ} (hc : c < cfg.cols) (hempty : (c, r) ∉ b)
    (hr : r < b.colHeight c) :
    r ∉ Board.fullRows cfg (pl.place b) := by
  intro hmem
  have hfull := Board.isFull_of_mem_fullRows hmem
  exact hole_persists_place hempty hr (hfull c (Finset.mem_range.mpr hc))

/-- **A clear-free move carries every hole forward as a hole**: if the
step completes no rows, a covered empty cell is still empty and still
covered afterwards — hole-count is non-decreasing along clear-free play,
the step form of the hole-debt monotone. -/
theorem hole_persists_step {cfg : GameConfig} {b : Board} {pl : Placement}
    {c r : ℕ} (hempty : (c, r) ∉ b) (hr : r < b.colHeight c)
    (hnc : Board.fullRows cfg (pl.place b) = ∅) :
    (c, r) ∉ Placement.applyStep cfg b pl
      ∧ r < (Placement.applyStep cfg b pl).colHeight c := by
  have hid : Placement.applyStep cfg b pl = pl.place b := by
    unfold Placement.applyStep
    exact Board.clearLines_eq_self_of_no_fullRows cfg hnc
  rw [hid]
  refine ⟨hole_persists_place hempty hr, ?_⟩
  have hsub : b ⊆ pl.place b := by
    rw [Placement.place_eq_union_dropped]
    exact Finset.subset_union_left
  have hmono := colHeight_mono hsub c
  omega

/-- **`lost` is not absorbing** (model documentation, kernel witness): a
state with a cell above the ceiling can *return to life* — piling the
overflow into full rows clears them away. This is exactly why the
survival predicate quantifies over *all* times (`SurvivesForever`:
`∀ n, ¬ lost`) rather than asking for a single final verdict: in this
model losing is an event you must avoid at every step, not a trap you
fall into once. -/
theorem lost_not_absorbing :
    ∃ (cfg : GameConfig) (g : GameState) (pl : Placement),
      g.lost cfg ∧ ¬ (g.step cfg pl).lost cfg := by
  refine ⟨⟨1, 1, by omega, by omega⟩,
    ⟨({(0, 1)} : Finset Coord), Bag.full⟩, ⟨Piece.I, 1, 0⟩, ?_, ?_⟩
  · decide
  · decide

/-- **The complete tetris anatomy, one theorem**: at any four-clear the
finisher is the I, and there is a single well column `c₀` carrying every
law at once — the drop lands at its height, the window is exactly the
four rows above it, the piece feeds it all four cells and no other
column, the window rows are exactly "everything but the well", and every
other column overtops it by at least four. The whole anatomical suite,
with all its well columns identified. -/
theorem tetris_anatomy {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    pl.piece = Piece.I
    ∧ ∃ c₀ < 10,
        pl.dropOffset b = b.colHeight c₀
        ∧ Board.fullRows GameConfig.standard (pl.place b)
          = Finset.Icc (b.colHeight c₀) (b.colHeight c₀ + 3)
        ∧ pl.colProfile c₀ = 4
        ∧ (∀ j < 10, j ≠ c₀ → pl.colProfile j = 0)
        ∧ (∀ r ∈ Board.fullRows GameConfig.standard (pl.place b),
            ∀ c < 10, ((c, r) ∈ b ↔ c ≠ c₀))
        ∧ (∀ c < 10, c ≠ c₀ → b.colHeight c₀ + 4 ≤ b.colHeight c) := by
  refine ⟨tetris_requires_I (b := b) (pl := pl) hnf (by omega), ?_⟩
  obtain ⟨c₀, hlt, hoff, hwin⟩ := tetris_window_at_well_height hv hnf h4
  obtain ⟨c₁, hlt₁, hprof4, hz⟩ := tetris_feeds_single_column hwf hv hnf h4
  obtain ⟨c₂, hlt₂, hshape⟩ := tetris_rows_pre_shape hwf hv hnf h4
  have h10 : c₁ = c₀ := by
    by_contra hne
    have hz0 := hz c₀ hlt (fun h => hne h.symm)
    have hpos : 0 < (Board.fullRows GameConfig.standard (pl.place b)).card :=
      by omega
    have hgt := clear_untouched_column_height (cfg := GameConfig.standard)
      hnf (Finset.card_pos.mp hpos)
      (by rw [GameConfig.standard_cols]; omega) hz0
    omega
  have hr3 : b.colHeight c₀ + 3
      ∈ Board.fullRows GameConfig.standard (pl.place b) := by
    rw [hwin, Finset.mem_Icc]
    omega
  have h20 : c₂ = c₀ := by
    by_contra hne
    have hmem : (c₀, b.colHeight c₀ + 3) ∈ b :=
      (hshape _ hr3 c₀ hlt).mpr (fun h => hne h.symm)
    have := Board.lt_colHeight hmem
    omega
  refine ⟨c₀, hlt, hoff, hwin, h10 ▸ hprof4, ?_, ?_, ?_⟩
  · intro j hj hne
    exact hz j hj (by rw [h10]; exact hne)
  · intro r hr c hc
    have h := hshape r hr c hc
    rw [h20] at h
    exact h
  · intro c hc hne
    have hmem : (c, b.colHeight c₀ + 3) ∈ b :=
      (hshape _ hr3 c hc).mpr (by rw [h20]; exact hne)
    have := Board.lt_colHeight hmem
    omega

/-- **The anatomy is sufficient — the constructive tetris**: on a board
whose rows `[h, h+3]` are complete except for column `c₀` (standing at
height exactly `h`), dropping the vertical I into `c₀` clears exactly
those four rows. Together with `tetris_anatomy`, the four-clear is fully
characterized: it happens *iff* the board presents this well and the I
takes it. -/
theorem tetris_of_well {b : Board} {c₀ h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hh : b.colHeight c₀ = h)
    (hfull : ∀ r, h ≤ r → r ≤ h + 3 → ∀ c < 10, c ≠ c₀ → (c, r) ∈ b) :
    Board.fullRows GameConfig.standard
      ((⟨Piece.I, 1, c₀⟩ : Placement).place b)
      = Finset.Icc h (h + 3) := by
  classical
  have hshapeI : Piece.I.shapeUp 1
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset Coord) := by
    decide
  have hshape : (⟨Piece.I, 1, c₀⟩ : Placement).shapeUp
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset Coord) := hshapeI
  have hoff : (⟨Piece.I, 1, c₀⟩ : Placement).dropOffset b = h := by
    refine Nat.le_antisymm ?_ ?_
    · apply dropOffset_le_of_heights
      intro cell hcell
      rw [hshape] at hcell
      simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
      rcases hcell with h1 | h1 | h1 | h1 <;> rw [h1] <;> simp [hh]
    · have hcell : ((0, 0) : Coord)
          ∈ (⟨Piece.I, 1, c₀⟩ : Placement).shapeUp := by
        rw [hshape]
        simp
      have hle := Finset.le_sup (f := fun cell =>
        b.colHeight ((⟨Piece.I, 1, c₀⟩ : Placement).col + cell.1) - cell.2)
        hcell
      have h3 : b.colHeight (c₀ + 0) - 0
          ≤ (⟨Piece.I, 1, c₀⟩ : Placement).dropOffset b := hle
      rw [Nat.add_zero, hh] at h3
      omega
  have hdropmem : ∀ k, k ≤ 3 →
      (c₀, h + k) ∈ (⟨Piece.I, 1, c₀⟩ : Placement).dropped b := by
    intro k hk
    rw [Placement.dropped_eq_image, Finset.mem_image]
    refine ⟨(0, k), ?_, ?_⟩
    · rw [hshape]
      interval_cases k <;> simp
    · rw [hoff]
      change (c₀ + 0, h + k) = (c₀, h + k)
      rw [Nat.add_zero]
  have hdroprows : ∀ q ∈ (⟨Piece.I, 1, c₀⟩ : Placement).dropped b,
      h ≤ q.2 ∧ q.2 ≤ h + 3 := by
    intro q hq
    rw [Placement.dropped_eq_image, Finset.mem_image] at hq
    obtain ⟨cell, hcell, hEq⟩ := hq
    have hrow : (⟨Piece.I, 1, c₀⟩ : Placement).dropOffset b + cell.2 = q.2 :=
      congrArg Prod.snd hEq
    have hb := Piece.shapeUp_row_lt_four Piece.I 1 cell hcell
    rw [hoff] at hrow
    omega
  ext r
  rw [Finset.mem_Icc]
  constructor
  · intro hr
    by_contra hout
    apply hnf r
    intro c hc
    have hfullr := Board.isFull_of_mem_fullRows hr
    have hmem := hfullr c hc
    rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmem
    rcases hmem with hmem | hmem
    · exact hmem
    · have h5 : h ≤ r ∧ r ≤ h + 3 := hdroprows _ hmem
      exact absurd h5 hout
  · intro hr
    have hcell : (c₀, r) ∈ (⟨Piece.I, 1, c₀⟩ : Placement).place b := by
      rw [Placement.place_eq_union_dropped, Finset.mem_union]
      right
      have hk := hdropmem (r - h) (by omega)
      rw [show h + (r - h) = r by omega] at hk
      exact hk
    have hisfull : Board.isFull GameConfig.standard
        ((⟨Piece.I, 1, c₀⟩ : Placement).place b) r := by
      intro c hcr
      rw [Finset.mem_range, GameConfig.standard_cols] at hcr
      by_cases hcc : c = c₀
      · rw [hcc]
        exact hcell
      · have hb := hfull r hr.1 hr.2 c hcr hcc
        rw [Placement.place_eq_union_dropped, Finset.mem_union]
        left
        exact hb
    simp only [Board.fullRows, Finset.mem_filter, Finset.mem_image]
    exact ⟨⟨(c₀, r), hcell, rfl⟩, hisfull⟩

/-- **Placing never repairs a hole, counted**: each column's hole count is
non-decreasing through the merge — the fed columns gain exactly their
landing gap and the rest are untouched. The quantitative face of
`hole_persists_place`. -/
theorem colHoles_place_ge {b : Board} {pl : Placement} (j : ℕ) :
    Board.colHoles b j ≤ Board.colHoles (pl.place b) j := by
  classical
  by_cases hfib : ((pl.dropped b).filter (fun p => p.1 = j)).Nonempty
  · set F := ((pl.dropped b).filter (fun p => p.1 = j)).image
      (fun p => p.2) with hF
    have hFne : F.Nonempty := hfib.image _
    have hmmem := Finset.min'_mem F hFne
    have hMmem := Finset.max'_mem F hFne
    have hget : ∀ r ∈ F, (j, r) ∈ pl.dropped b := by
      intro r hr
      rw [hF, Finset.mem_image] at hr
      obtain ⟨q, hq, rfl⟩ := hr
      rw [Finset.mem_filter] at hq
      have hqe : q = (j, q.2) := Prod.ext hq.2 rfl
      rw [← hqe]
      exact hq.1
    have h_hm : b.colHeight j ≤ F.min' hFne :=
      dropped_above_own_column (j, F.min' hFne) (hget _ hmmem)
    have h_hM : F.max' hFne < (pl.place b).colHeight j := by
      have hmem : (j, F.max' hFne) ∈ pl.place b := by
        rw [Placement.place_eq_union_dropped, Finset.mem_union]
        exact Or.inr (hget _ hMmem)
      exact Board.lt_colHeight hmem
    have hsub : (pl.place b).colRows j ⊆ b.colRows j ∪ F := by
      unfold Board.colRows
      rw [Placement.place_eq_union_dropped, Finset.filter_union,
        Finset.image_union]
    have hcard' : ((pl.place b).colRows j).card
        ≤ (b.colRows j).card + F.card :=
      le_trans (Finset.card_le_card hsub) (Finset.card_union_le _ _)
    have hFcard : F.card ≤ F.max' hFne - F.min' hFne + 1 := by
      calc F.card ≤ (Finset.Icc (F.min' hFne) (F.max' hFne)).card :=
          Finset.card_le_card (fun x hx => Finset.mem_Icc.mpr
            ⟨Finset.min'_le _ _ hx, Finset.le_max' _ _ hx⟩)
        _ ≤ F.max' hFne - F.min' hFne + 1 := by
          rw [Nat.card_Icc]
          omega
    have hcards := Board.colRows_card_le_colHeight b j
    have hcards' := Board.colRows_card_le_colHeight (pl.place b) j
    have hmM : F.min' hFne ≤ F.max' hFne := Finset.min'_le _ _ hMmem
    unfold Board.colHoles
    omega
  · have hprof : ((pl.dropped b).filter (fun p => p.1 = j)).card
        = pl.colProfile j := by
      have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) j
      unfold Board.colCount at hcc
      unfold Placement.dropped
      exact hcc
    rw [Finset.not_nonempty_iff_eq_empty] at hfib
    have hz : pl.colProfile j = 0 := by
      rw [← hprof, hfib, Finset.card_empty]
    have hh := place_unfed_colHeight_eq (b := b) (pl := pl) hz
    have hr : (pl.place b).colRows j = b.colRows j := by
      unfold Board.colRows
      rw [Placement.place_eq_union_dropped, Finset.filter_union, hfib,
        Finset.union_empty]
    unfold Board.colHoles
    rw [hh, hr]

/-- Total holes are non-decreasing through the merge: only clearing can
lower the debt. -/
theorem holes_place_ge {cfg : GameConfig} {b : Board} {pl : Placement} :
    Board.holes cfg b ≤ Board.holes cfg (pl.place b) := by
  unfold Board.holes
  exact Finset.sum_le_sum (fun j _ => colHoles_place_ge j)

/-- A clear-free full move keeps the hole count non-decreasing. -/
theorem holes_step_ge_of_no_clear {cfg : GameConfig} {b : Board}
    {pl : Placement}
    (hnc : Board.fullRows cfg (pl.place b) = ∅) :
    Board.holes cfg b ≤ Board.holes cfg (Placement.applyStep cfg b pl) := by
  have hid : Placement.applyStep cfg b pl = pl.place b := by
    unfold Placement.applyStep
    exact Board.clearLines_eq_self_of_no_fullRows cfg hnc
  rw [hid]
  exact holes_place_ge

/-- **The hole debt is monotone along dry play**: over any clear-free
window of a trace, the board's total hole count never decreases — the
trace form of the hole-debt Lyapunov's rising half. Debt accumulated in
a drought is debt owed until a clear. -/
theorem trace_holes_mono_of_dry {π : Policy GameConfig.standard} {n w : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
      = cleared GameConfig.standard π GameState.init n) :
    Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init n).board
      ≤ Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init (n + w)).board := by
  induction w with
  | zero => exact le_refl _
  | succ k ih =>
    have hm1 := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right n k)
    have hm2 := cleared_mono GameConfig.standard π GameState.init
      (show n + k ≤ n + (k + 1) by omega)
    have hs := cleared_succ GameConfig.standard π GameState.init (n + k)
    have h1 : cleared GameConfig.standard π GameState.init (n + k)
        = cleared GameConfig.standard π GameState.init n := by
      rw [show n + (k + 1) = (n + k) + 1 by omega] at hdry
      omega
    have hcard0 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init (n + k))).place
          (trace GameConfig.standard π GameState.init (n + k)).board)).card
        = 0 := by
      rw [show n + (k + 1) = (n + k) + 1 by omega] at hdry
      omega
    have hnc := Finset.card_eq_zero.mp hcard0
    have hstep := holes_step_ge_of_no_clear (cfg := GameConfig.standard)
      hnc
    rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
      GameState.step_board]
    exact le_trans (ih h1) hstep

/-- **Debt plus mass fits the board**: on an in-field well-formed board,
total holes plus total cells never exceed the 200-cell volume — holes
live inside the stack envelope, so hole debt is capped by the free
volume the mass leaves behind. -/
theorem holes_add_count_le_two_hundred {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hif : ∀ p ∈ b, p.2 < GameConfig.standard.rows) :
    Board.holes GameConfig.standard b + b.count ≤ 200 := by
  classical
  have hsum := Board.sum_colRows_card_add_holes GameConfig.standard b
  have hcc : ∀ j, (b.colRows j).card = b.colCount j := by
    intro j
    unfold Board.colRows Board.colCount
    apply Finset.card_image_of_injOn
    intro p hp q hq hpq
    simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
    exact Prod.ext (by rw [hp.2, hq.2]) hpq
  have hcnt := Board.sum_colCount (cfg := GameConfig.standard) hwf
  have hheights : ∑ j ∈ Finset.range 10, b.colHeight j ≤ 200 := by
    calc ∑ j ∈ Finset.range 10, b.colHeight j
        ≤ ∑ _j ∈ Finset.range 10, 20 :=
          Finset.sum_le_sum (fun j _ => by
            have h := Board.colHeight_le_rows_of_in_field hif j
            rwa [GameConfig.standard_rows] at h)
      _ = 200 := by simp
  rw [GameConfig.standard_cols] at hsum hcnt
  have hrw : ∑ j ∈ Finset.range 10, (b.colRows j).card
      = ∑ j ∈ Finset.range 10, b.colCount j :=
    Finset.sum_congr rfl (fun j _ => hcc j)
  omega

/-- The debt-plus-mass cap, on live traces. -/
theorem trace_holes_add_count_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) :
    Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init m).board
      + (trace GameConfig.standard π GameState.init m).board.count
      ≤ 200 :=
  holes_add_count_le_two_hundred
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m)
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)

/-- **The hole-debt ledger cap**: at any live step,
`holes + 4m ≤ 200 + 10·cleared` — the game can only afford holes out of
whatever clearing has already freed. A slow-clearing game is forced
nearly hole-free; a hole-heavy board is a debt against future clears,
priced exactly. -/
theorem trace_holes_ledger_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard) :
    Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init m).board
      + 4 * m
      ≤ 200 + 10 * cleared GameConfig.standard π GameState.init m := by
  have hcap := trace_holes_add_count_le hv hlive
  have hled := init_ledger (cfg := GameConfig.standard) hv m
  rw [GameConfig.standard_cols] at hled
  omega

/-- A column's visible rows count exactly its cells. -/
theorem colRows_card_eq_colCount (b : Board) (j : ℕ) :
    (b.colRows j).card = b.colCount j := by
  classical
  unfold Board.colRows Board.colCount
  apply Finset.card_image_of_injOn
  intro p hp q hq hpq
  simp only [Finset.mem_coe, Finset.mem_filter] at hp hq
  exact Prod.ext (by rw [hp.2, hq.2]) hpq

/-- **Clearing drops a column's height by at least the clear count**: with
`k` full rows cleared, every valid column ends at least `k` lower — each
surviving cell slides down by its cleared-below count, and the cleared
rows above it push the old height even higher. -/
theorem colHeight_clearLines_add_le {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) (hk : 0 < (Board.fullRows cfg b).card) :
    (Board.clearLines cfg b).colHeight j + (Board.fullRows cfg b).card
      ≤ b.colHeight j := by
  classical
  set k := (Board.fullRows cfg b).card with hkdef
  have hkey : ∀ r' ∈ (Board.clearLines cfg b).colRows j,
      r' + 1 + k ≤ b.colHeight j := by
    intro r' hr'
    unfold Board.colRows at hr'
    rw [Finset.mem_image] at hr'
    obtain ⟨q, hq, rfl⟩ := hr'
    rw [Finset.mem_filter] at hq
    obtain ⟨hqmem, hqj⟩ := hq
    unfold Board.clearLines at hqmem
    rw [Finset.mem_image] at hqmem
    obtain ⟨c, hc, hEq⟩ := hqmem
    rw [Finset.mem_filter] at hc
    obtain ⟨hcb, hcnf⟩ := hc
    have hcol : c.1 = q.1 := by
      have h := congrArg Prod.fst hEq
      exact h
    have hrow : c.2 - Board.clearedBelow cfg b c.2 = q.2 := by
      have h := congrArg Prod.snd hEq
      exact h
    set r := c.2 with hrdef
    -- (j, r) ∈ b
    have hjr : (j, r) ∈ b := by
      have hce : c = (j, r) := Prod.ext (by rw [hcol, hqj]) rfl
      rw [← hce]
      exact hcb
    have hrh := Board.lt_colHeight hjr
    -- clearedBelow ≤ k and ≤ r
    have hcble : Board.clearedBelow cfg b r ≤ k := by
      unfold Board.clearedBelow
      exact Finset.card_le_card (Finset.filter_subset _ _)
    have hcbler : Board.clearedBelow cfg b r ≤ r := by
      unfold Board.clearedBelow
      calc ((Board.fullRows cfg b).filter (· < r)).card
          ≤ (Finset.range r).card := Finset.card_le_card (by
            intro x hx
            rw [Finset.mem_filter] at hx
            exact Finset.mem_range.mpr hx.2)
        _ = r := Finset.card_range r
    -- r is not a full row
    have hrnotC : r ∉ Board.fullRows cfg b := by
      intro hmem
      exact hcnf (Board.isFull_of_mem_fullRows hmem)
    -- the cleared rows not below r
    set A := (Board.fullRows cfg b).filter (fun t => ¬ t < r) with hA
    have hAcard : A.card + Board.clearedBelow cfg b r = k := by
      unfold Board.clearedBelow
      rw [hA, hkdef]
      have := Finset.card_filter_add_card_filter_not
        (s := Board.fullRows cfg b) (fun t => t < r)
      omega
    by_cases hAne : A.Nonempty
    · set t := A.max' hAne with ht
      have htmem : t ∈ A := Finset.max'_mem _ _
      have htC : t ∈ Board.fullRows cfg b := (Finset.mem_filter.mp htmem).1
      have htgt : r < t := by
        have h1 : ¬ t < r := (Finset.mem_filter.mp htmem).2
        rcases Nat.lt_or_ge r t with h | h
        · exact h
        · exfalso
          have : t = r := by omega
          rw [this] at htC
          exact hrnotC htC
      -- A ⊆ Icc (r+1) t so card A ≤ t − r
      have hAsub : A ⊆ Finset.Icc (r + 1) t := by
        intro x hx
        rw [Finset.mem_Icc]
        have h1 : ¬ x < r := (Finset.mem_filter.mp hx).2
        have h2 : x ≤ t := Finset.le_max' _ _ hx
        have h3 : x ≠ r := by
          intro hxr
          rw [hxr] at hx
          exact hrnotC (Finset.mem_filter.mp hx).1
        omega
      have hAle : A.card ≤ t - r := by
        calc A.card ≤ (Finset.Icc (r + 1) t).card :=
            Finset.card_le_card hAsub
          _ ≤ t - r := by
            rw [Nat.card_Icc]
            omega
      -- t < height
      have hth : t < b.colHeight j := by
        have hjt : (j, t) ∈ b := Board.isFull_of_mem_fullRows htC j
          (Finset.mem_range.mpr hj)
        exact Board.lt_colHeight hjt
      omega
    · rw [Finset.not_nonempty_iff_eq_empty] at hAne
      have hA0 : A.card = 0 := by rw [hAne]; exact Finset.card_empty
      omega
  -- conclude via sup
  have hcnt : k ≤ b.colHeight j := by
    -- column j holds a cell of some full row
    obtain ⟨r₀, hr₀⟩ := Finset.card_pos.mp hk
    have hjr₀ : (j, r₀) ∈ b := Board.isFull_of_mem_fullRows hr₀ j
      (Finset.mem_range.mpr hj)
    -- every full row contributes a distinct cell to column j
    have hsub : Board.fullRows cfg b ⊆ b.colRows j := by
      intro r hr
      unfold Board.colRows
      rw [Finset.mem_image]
      exact ⟨(j, r), Finset.mem_filter.mpr
        ⟨Board.isFull_of_mem_fullRows hr j (Finset.mem_range.mpr hj), rfl⟩,
        rfl⟩
    calc k ≤ (b.colRows j).card := Finset.card_le_card hsub
      _ ≤ b.colHeight j := Board.colRows_card_le_colHeight b j
  have hsup : (Board.clearLines cfg b).colHeight j
      ≤ b.colHeight j - k := by
    unfold Board.colHeight
    apply Finset.sup_le
    intro r' hr'
    change r' + 1 ≤ b.colHeight j - k
    have h2 := hkey r' hr'
    omega
  omega

/-- **Clearing never increases a column's holes**: line clears are the
sole repair mechanism, and they never backfire — the falling half of the
hole-debt Lyapunov. -/
theorem colHoles_clearLines_le {cfg : GameConfig} {b : Board} {j : ℕ}
    (hj : j < cfg.cols) :
    Board.colHoles (Board.clearLines cfg b) j ≤ Board.colHoles b j := by
  classical
  by_cases hk : (Board.fullRows cfg b).card = 0
  · have hempty := Finset.card_eq_zero.mp hk
    rw [Board.clearLines_eq_self_of_no_fullRows cfg hempty]
  · have h1 := colHeight_clearLines_add_le (b := b) hj (by omega)
    have h2 := Board.colCount_clearLines_add cfg b hj
    have h3 := colRows_card_eq_colCount b j
    have h4 := colRows_card_eq_colCount (Board.clearLines cfg b) j
    have h5 := Board.colRows_card_le_colHeight b j
    have h6 := Board.colRows_card_le_colHeight (Board.clearLines cfg b) j
    unfold Board.colHoles
    omega

/-- Total holes never increase under clearing. -/
theorem holes_clearLines_le (cfg : GameConfig) (b : Board) :
    Board.holes cfg (Board.clearLines cfg b) ≤ Board.holes cfg b := by
  unfold Board.holes
  exact Finset.sum_le_sum (fun j hj =>
    colHoles_clearLines_le (Finset.mem_range.mp hj))

/-- **Unfed columns sink by the clear count in height too**: a `k`-clear
lowers every valid column the piece did not feed by at least `k` rows of
height — the skyline drain matching the cell drain of
`clear_step_column_drain`. Clearing planes the board down wherever the
piece didn't build. -/
theorem clear_step_unfed_colHeight_le {cfg : GameConfig} {b : Board}
    {pl : Placement} {j : ℕ} (hj : j < cfg.cols)
    (hz : pl.colProfile j = 0)
    (hk : 0 < (Board.fullRows cfg (pl.place b)).card) :
    (Placement.applyStep cfg b pl).colHeight j
      + (Board.fullRows cfg (pl.place b)).card ≤ b.colHeight j := by
  have h1 := colHeight_clearLines_add_le (b := pl.place b) hj hk
  have h2 := place_unfed_colHeight_eq (b := b) (pl := pl) hz
  unfold Placement.applyStep
  omega

/-- **The well's height survives the tetris unchanged**: through the full
move — vertical I in, four rows out — the well column ends at exactly the
height it started. The I fills the window, the window clears, and the
stack beneath is untouched: a tetris is a pure harvest of the other nine
columns, invisible in the well's own skyline. -/
theorem tetris_well_height_preserved {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10, (Placement.applyStep GameConfig.standard b pl).colHeight c₀
      = b.colHeight c₀ := by
  classical
  obtain ⟨hI, c₀, hlt, hoff, hwin, hprof4, hz, hshape, hdepth⟩ :=
    tetris_anatomy hwf hv hnf h4
  refine ⟨c₀, hlt, ?_⟩
  have hub : (Placement.applyStep GameConfig.standard b pl).colHeight c₀ + 4
      ≤ (pl.place b).colHeight c₀ := by
    have hcl := colHeight_clearLines_add_le (b := pl.place b) (j := c₀)
      (by rw [GameConfig.standard_cols]; omega) (by omega)
    unfold Placement.applyStep
    omega
  have hpl_ub : (pl.place b).colHeight c₀ ≤ b.colHeight c₀ + 4 := by
    unfold Board.colHeight
    apply Finset.sup_le
    intro r hr
    unfold Board.colRows at hr
    rw [Finset.mem_image] at hr
    obtain ⟨q, hq, rfl⟩ := hr
    rw [Finset.mem_filter] at hq
    obtain ⟨hqb, hqj⟩ := hq
    rw [Placement.place_eq_union_dropped, Finset.mem_union] at hqb
    rcases hqb with hmem | hmem
    · have hcell : (c₀, q.2) ∈ b := by
        have hqe : q = (c₀, q.2) := Prod.ext hqj rfl
        rw [← hqe]
        exact hmem
      have := Board.lt_colHeight hcell
      change q.2 + 1 ≤ b.colHeight c₀ + 4
      omega
    · rw [Placement.dropped_eq_image, Finset.mem_image] at hmem
      obtain ⟨cell, hcell, hEq⟩ := hmem
      have hrow : pl.dropOffset b + cell.2 = q.2 := by
        have hh2 := congrArg Prod.snd hEq
        exact hh2
      have hb4 := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
      change q.2 + 1 ≤ b.colHeight c₀ + 4
      rw [hoff] at hrow
      omega
  rcases Nat.eq_zero_or_pos (b.colHeight c₀) with h0 | hpos
  · omega
  · have hne : (b.colRows c₀).Nonempty := by
      by_contra hcon
      rw [Finset.not_nonempty_iff_eq_empty] at hcon
      have hz0 : b.colHeight c₀ = 0 := by
        unfold Board.colHeight
        rw [hcon]
        simp
      omega
    obtain ⟨rtop, hrtopmem, hsup⟩ := Finset.exists_mem_eq_sup _ hne
      (fun r => r + 1)
    have hrtop : rtop + 1 = b.colHeight c₀ := by
      have h := hsup
      exact h.symm
    have hcelltop : (c₀, rtop) ∈ b := by
      unfold Board.colRows at hrtopmem
      rw [Finset.mem_image] at hrtopmem
      obtain ⟨q, hq, rfl⟩ := hrtopmem
      rw [Finset.mem_filter] at hq
      have hqe : q = (c₀, q.2) := Prod.ext hq.2 rfl
      rw [← hqe]
      exact hq.1
    have hplacetop : (c₀, rtop) ∈ pl.place b := by
      rw [Placement.place_eq_union_dropped, Finset.mem_union]
      exact Or.inl hcelltop
    have hnotfull : ¬ Board.isFull GameConfig.standard (pl.place b) rtop := by
      intro hfull
      have hmem : rtop ∈ Board.fullRows GameConfig.standard (pl.place b) := by
        simp only [Board.fullRows, Finset.mem_filter, Finset.mem_image]
        exact ⟨⟨(c₀, rtop), hplacetop, rfl⟩, hfull⟩
      rw [hwin, Finset.mem_Icc] at hmem
      omega
    have hcb0 : Board.clearedBelow GameConfig.standard (pl.place b) rtop
        = 0 := by
      unfold Board.clearedBelow
      rw [hwin, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro x hx
      rw [Finset.mem_Icc] at hx
      omega
    have hsurv : (c₀, rtop)
        ∈ Placement.applyStep GameConfig.standard b pl := by
      unfold Placement.applyStep Board.clearLines
      rw [Finset.mem_image]
      refine ⟨(c₀, rtop), Finset.mem_filter.mpr ⟨hplacetop, hnotfull⟩, ?_⟩
      rw [hcb0]
      change (c₀, rtop - 0) = (c₀, rtop)
      rw [Nat.sub_zero]
    have hlb := Board.lt_colHeight hsurv
    omega

/-- **The tetris skyline law**: through the full four-clear move, the well
column's height is exactly preserved while every other column sinks by at
least four — the complete before/after skyline of the tetris, with the
well identified across both clauses. -/
theorem tetris_step_skyline {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    ∃ c₀ < 10,
      (Placement.applyStep GameConfig.standard b pl).colHeight c₀
        = b.colHeight c₀
      ∧ ∀ j < 10, j ≠ c₀ →
        (Placement.applyStep GameConfig.standard b pl).colHeight j + 4
          ≤ b.colHeight j := by
  obtain ⟨hI, c₀, hlt, hoff, hwin, hprof4, hz, hshape, hdepth⟩ :=
    tetris_anatomy hwf hv hnf h4
  have hothers : ∀ j < 10, j ≠ c₀ →
      (Placement.applyStep GameConfig.standard b pl).colHeight j + 4
        ≤ b.colHeight j := by
    intro j hj hne
    have hdrop := clear_step_unfed_colHeight_le (b := b) (pl := pl)
      (j := j) (by rw [GameConfig.standard_cols]; omega)
      (hz j hj hne) (by omega)
    omega
  refine ⟨c₀, hlt, ?_, hothers⟩
  obtain ⟨c₁, hlt₁, hpres⟩ := tetris_well_height_preserved hwf hv hnf h4
  by_cases he : c₁ = c₀
  · rw [← he]
    exact hpres
  · exfalso
    have hdrop := hothers c₁ hlt₁ he
    omega

/-- **A tetris shaves thirty-six rows of skyline**: the total column
height drops by at least 36 through the move — nine columns lose four
each and the well loses nothing. The skyline-mass bill matching the
36-cell mass bill of the column-flow law. -/
theorem tetris_skyline_mass {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    (∑ j ∈ Finset.range 10,
      (Placement.applyStep GameConfig.standard b pl).colHeight j) + 36
      ≤ ∑ j ∈ Finset.range 10, b.colHeight j := by
  classical
  obtain ⟨c₀, hlt, hpres, hothers⟩ := tetris_step_skyline hwf hv hnf h4
  have hmem : c₀ ∈ Finset.range 10 := Finset.mem_range.mpr hlt
  have h1 := Finset.add_sum_erase (Finset.range 10)
    (fun j => (Placement.applyStep GameConfig.standard b pl).colHeight j)
    hmem
  have h2 := Finset.add_sum_erase (Finset.range 10)
    (fun j => b.colHeight j) hmem
  have hcard : ((Finset.range 10).erase c₀).card = 9 := by
    rw [Finset.card_erase_of_mem hmem, Finset.card_range]
  have herase : (∑ j ∈ (Finset.range 10).erase c₀,
      (Placement.applyStep GameConfig.standard b pl).colHeight j) + 36
      ≤ ∑ j ∈ (Finset.range 10).erase c₀, b.colHeight j := by
    calc (∑ j ∈ (Finset.range 10).erase c₀,
          (Placement.applyStep GameConfig.standard b pl).colHeight j) + 36
        = ∑ j ∈ (Finset.range 10).erase c₀,
            ((Placement.applyStep GameConfig.standard b pl).colHeight j
              + 4) := by
          rw [Finset.sum_add_distrib, Finset.sum_const, hcard, smul_eq_mul]
      _ ≤ ∑ j ∈ (Finset.range 10).erase c₀, b.colHeight j :=
          Finset.sum_le_sum (fun j hj => by
            rw [Finset.mem_erase] at hj
            exact hothers j (Finset.mem_range.mp hj.2) hj.1)
  simp only [] at h1 h2
  omega

/-- **A tetris creates no holes**: the vertical I lands with zero gap on
the well stack, so the merge leaves every column's hole count unchanged,
and the clear can only lower it — `holes(after) ≤ holes(before)`. The
tetris is a debt-free harvest: unlike every gap-landing placement, it
never adds to the hole ledger it may even repay. -/
theorem tetris_no_new_holes {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    Board.holes GameConfig.standard
      (Placement.applyStep GameConfig.standard b pl)
      ≤ Board.holes GameConfig.standard b := by
  classical
  obtain ⟨hI, c₀, hlt, hoff, hwin, hprof4, hz, hshape, hdepth⟩ :=
    tetris_anatomy hwf hv hnf h4
  have hcols : ∀ j < 10,
      Board.colHoles (pl.place b) j = Board.colHoles b j := by
    intro j hj
    by_cases hjc : j = c₀
    · subst hjc
      have hrows := four_clear_piece_rows_card hnf h4
      have hrows' : ((Piece.I.shapeUp pl.rot).image (fun c => c.2)).card
          = 4 := by
        unfold Placement.shapeUp at hrows
        rw [hI] at hrows
        exact hrows
      obtain ⟨t, ht4, hshapeI⟩ := I_shape_vertical_eq pl.rot hrows'
      have hshapeUp : pl.shapeUp
          = ({(t, 0), (t, 1), (t, 2), (t, 3)} : Finset Coord) := by
        unfold Placement.shapeUp
        rw [hI]
        exact hshapeI
      have hct : pl.col + t = j := by
        by_contra hne
        have hclt : pl.col + t < 10 := by
          have hmem : ((t, 0) : Coord) ∈ pl.shapeUp := by
            rw [hshapeUp]
            simp
          have h := hv (t, 0) hmem
          rwa [GameConfig.standard_cols] at h
        have h0 := hz (pl.col + t) hclt hne
        have h1 : 1 ≤ pl.colProfile (pl.col + t) := by
          unfold Placement.colProfile
          apply Finset.card_pos.mpr
          exact ⟨(t, 0), Finset.mem_filter.mpr
            ⟨by rw [hshapeUp]; simp, rfl⟩⟩
        omega
      have hfed := place_fed_colHeight_eq (b := b)
        (cell := ((t, 3) : Coord))
        (by rw [hshapeUp]; simp)
        (by
          intro cell' hc' hcc
          rw [hshapeUp] at hc'
          simp only [Finset.mem_insert, Finset.mem_singleton] at hc'
          rcases hc' with h | h | h | h <;> rw [h] <;> simp)
      have hfed' : (pl.place b).colHeight j = b.colHeight j + 4 := by
        have h := hfed
        rw [show pl.col + ((t, 3) : Coord).1 = j from hct] at h
        rw [h, hoff]
      have hcc := Placement.colCount_place b pl j
      unfold Board.colHoles
      rw [colRows_card_eq_colCount, colRows_card_eq_colCount, hfed', hcc,
        hprof4]
      have hle := Board.colCount_le_count b j
      have hhle := colCount_le_colHeight b j
      omega
    · have hz0 : pl.colProfile j = 0 := hz j hj hjc
      have hprof : ((pl.dropped b).filter (fun p => p.1 = j)).card
          = pl.colProfile j := by
        have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) j
        unfold Board.colCount at hcc
        unfold Placement.dropped
        exact hcc
      have hempty : (pl.dropped b).filter (fun p => p.1 = j) = ∅ :=
        Finset.card_eq_zero.mp (by omega)
      have hh := place_unfed_colHeight_eq (b := b) (pl := pl) hz0
      have hr : (pl.place b).colRows j = b.colRows j := by
        unfold Board.colRows
        rw [Placement.place_eq_union_dropped, Finset.filter_union, hempty,
          Finset.union_empty]
      unfold Board.colHoles
      rw [hh, hr]
  have hplace : Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b := by
    unfold Board.holes
    rw [GameConfig.standard_cols]
    exact Finset.sum_congr rfl (fun j hj =>
      hcols j (Finset.mem_range.mp hj))
  have hfinal : Board.holes GameConfig.standard
      (Placement.applyStep GameConfig.standard b pl)
      ≤ Board.holes GameConfig.standard (pl.place b) := by
    unfold Placement.applyStep
    exact holes_clearLines_le GameConfig.standard (pl.place b)
  omega

/-- **The exact hole-genesis formula**: a fed column's hole count grows by
precisely its landing gap — the space between the old stack top and the
lowest landed cell. Holes are not merely bounded; every placement's hole
bill is computed to the cell: `Δholes = (bottom of fiber) − (old height)`
per fed column, zero for the rest. -/
theorem colHoles_place_eq {b : Board} {pl : Placement} {j m : ℕ}
    (hmem : (j, m) ∈ pl.dropped b)
    (hmin : ∀ r, (j, r) ∈ pl.dropped b → m ≤ r) :
    Board.colHoles (pl.place b) j
      = Board.colHoles b j + (m - b.colHeight j) := by
  classical
  set F := ((pl.dropped b).filter (fun p => p.1 = j)).image (fun p => p.2)
    with hF
  have hput : ∀ r, (j, r) ∈ pl.dropped b → r ∈ F := by
    intro r hr
    rw [hF, Finset.mem_image]
    exact ⟨(j, r), Finset.mem_filter.mpr ⟨hr, rfl⟩, rfl⟩
  have hFne : F.Nonempty := ⟨m, hput m hmem⟩
  set M := F.max' hFne with hM
  have hget : ∀ r ∈ F, (j, r) ∈ pl.dropped b := by
    intro r hr
    rw [hF, Finset.mem_image] at hr
    obtain ⟨q, hq, rfl⟩ := hr
    rw [Finset.mem_filter] at hq
    have hqe : q = (j, q.2) := Prod.ext hq.2 rfl
    rw [← hqe]
    exact hq.1
  have h_hm : b.colHeight j ≤ m :=
    dropped_above_own_column (j, m) hmem
  have hmM : m ≤ M := hmin M (hget _ (Finset.max'_mem _ _))
  -- F is the interval [m, M]
  have hFIcc : F = Finset.Icc m M := by
    apply Finset.Subset.antisymm
    · intro r hr
      rw [Finset.mem_Icc]
      exact ⟨hmin r (hget _ hr), Finset.le_max' _ _ hr⟩
    · intro r hr
      rw [Finset.mem_Icc] at hr
      apply hput
      exact dropped_fiber_contiguous (b := b) (pl := pl)
        hmem (hget _ (Finset.max'_mem F hFne)) rfl hr.1 hr.2
  have hFcard : F.card = M - m + 1 := by
    rw [hFIcc, Nat.card_Icc]
    omega
  -- colRows of place = colRows b ∪ F
  have hunion : (pl.place b).colRows j = b.colRows j ∪ F := by
    unfold Board.colRows
    rw [Placement.place_eq_union_dropped, Finset.filter_union,
      Finset.image_union]
  have hdisj : Disjoint (b.colRows j) F := by
    rw [Finset.disjoint_left]
    intro r hr hrF
    have h1 : r < b.colHeight j := by
      unfold Board.colRows at hr
      rw [Finset.mem_image] at hr
      obtain ⟨q, hq, rfl⟩ := hr
      rw [Finset.mem_filter] at hq
      have hqe : q = (j, q.2) := Prod.ext hq.2 rfl
      exact Board.lt_colHeight (by rw [← hqe] at *; exact hq.1)
    have h2 : b.colHeight j ≤ r :=
      dropped_above_own_column (j, r) (hget _ hrF)
    omega
  have hcard' : ((pl.place b).colRows j).card
      = (b.colRows j).card + F.card := by
    rw [hunion, Finset.card_union_of_disjoint hdisj]
  -- height of place = M + 1
  have hub : (pl.place b).colHeight j ≤ M + 1 := by
    unfold Board.colHeight
    apply Finset.sup_le
    intro r hr
    rw [hunion, Finset.mem_union] at hr
    rcases hr with hr | hr
    · have h1 : r < b.colHeight j := by
        unfold Board.colRows at hr
        rw [Finset.mem_image] at hr
        obtain ⟨q, hq, rfl⟩ := hr
        rw [Finset.mem_filter] at hq
        have hqe : q = (j, q.2) := Prod.ext hq.2 rfl
        exact Board.lt_colHeight (by rw [← hqe] at *; exact hq.1)
      change r + 1 ≤ M + 1
      omega
    · have := Finset.le_max' _ _ hr
      change r + 1 ≤ M + 1
      omega
  have hlb : M < (pl.place b).colHeight j := by
    have hmem : (j, M) ∈ pl.place b := by
      rw [Placement.place_eq_union_dropped, Finset.mem_union]
      exact Or.inr (hget _ (Finset.max'_mem _ _))
    exact Board.lt_colHeight hmem
  have hcards := Board.colRows_card_le_colHeight b j
  unfold Board.colHoles
  omega

/-- **Flush landings are hole-neutral**: if the piece drops a cell exactly
onto the old stack top of a column, that column gains no holes — by the
no-burrow law every landed cell sits at or above the height, so the flush
cell is automatically the fiber's bottom and the landing gap is zero. -/
theorem colHoles_place_eq_of_flush {b : Board} {pl : Placement} {j : ℕ}
    (hmem : (j, b.colHeight j) ∈ pl.dropped b) :
    Board.colHoles (pl.place b) j = Board.colHoles b j := by
  have hmin : ∀ r, (j, r) ∈ pl.dropped b → b.colHeight j ≤ r :=
    fun r hr => dropped_above_own_column (j, r) hr
  have h := colHoles_place_eq hmem hmin
  omega

/-- Every shape touches row zero (28-case check). -/
theorem shape_has_bottom_cell :
    ∀ p : Piece, ∀ r : Rotation, ∃ cell ∈ p.shapeUp r, cell.2 = 0 := by
  decide

/-- **Every placement lands flush somewhere**: some cell of every drop
comes to rest exactly on its column's old stack top (or the floor) — the
piece must touch down. With `colHoles_place_eq_of_flush`: every move has
at least one fed column that gains no holes; gap damage is never total. -/
theorem exists_flush_cell {b : Board} {pl : Placement} :
    ∃ cell ∈ pl.shapeUp,
      pl.dropOffset b + cell.2 = b.colHeight (pl.col + cell.1) := by
  classical
  rcases Nat.eq_zero_or_pos (pl.dropOffset b) with h0 | hpos
  · obtain ⟨cell, hcell, hc2⟩ := shape_has_bottom_cell pl.piece pl.rot
    refine ⟨cell, hcell, ?_⟩
    have hle := fed_column_height_le (b := b) cell hcell
    omega
  · have hne : pl.shapeUp.Nonempty := by
      apply Finset.card_pos.mp
      rw [pl.shapeUp_card]
      omega
    obtain ⟨cell, hcell, hsup⟩ := Finset.exists_mem_eq_sup pl.shapeUp hne
      (fun cell => b.colHeight (pl.col + cell.1) - cell.2)
    refine ⟨cell, hcell, ?_⟩
    have hd : pl.dropOffset b
        = b.colHeight (pl.col + cell.1) - cell.2 := by
      unfold Placement.dropOffset
      exact hsup
    omega

/-- An unfed column's hole count is unchanged by the merge. -/
theorem colHoles_place_eq_of_unfed {b : Board} {pl : Placement} {j : ℕ}
    (hz : pl.colProfile j = 0) :
    Board.colHoles (pl.place b) j = Board.colHoles b j := by
  classical
  have hprof : ((pl.dropped b).filter (fun p => p.1 = j)).card
      = pl.colProfile j := by
    have hcc := Placement.colCount_cellsAt pl (pl.dropOffset b) j
    unfold Board.colCount at hcc
    unfold Placement.dropped
    exact hcc
  have hempty : (pl.dropped b).filter (fun p => p.1 = j) = ∅ :=
    Finset.card_eq_zero.mp (by omega)
  have hh := place_unfed_colHeight_eq (b := b) (pl := pl) hz
  have hr : (pl.place b).colRows j = b.colRows j := by
    unfold Board.colRows
    rw [Placement.place_eq_union_dropped, Finset.filter_union, hempty,
      Finset.union_empty]
  unfold Board.colHoles
  rw [hh, hr]

/-- **At most three columns per move can gain holes**: a placement touches
at most four columns and always lands flush in one of them, so the
hole-gaining columns number at most three. Hole damage is narrow as well
as priced. -/
theorem place_hole_columns_le_three {b : Board} {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    ((Finset.range 10).filter (fun j =>
      Board.colHoles b j < Board.colHoles (pl.place b) j)).card ≤ 3 := by
  classical
  obtain ⟨cell₀, hcell₀, hflush⟩ := exists_flush_cell (b := b) (pl := pl)
  set c := pl.col + cell₀.1 with hc
  have hclt : c < 10 := by
    have h := hv cell₀ hcell₀
    rwa [GameConfig.standard_cols] at h
  have hcprof : 1 ≤ pl.colProfile c := by
    unfold Placement.colProfile
    apply Finset.card_pos.mpr
    exact ⟨cell₀, Finset.mem_filter.mpr ⟨hcell₀, rfl⟩⟩
  have hcflushmem : (c, b.colHeight c) ∈ pl.dropped b := by
    rw [Placement.dropped_eq_image, Finset.mem_image]
    refine ⟨cell₀, hcell₀, ?_⟩
    rw [hflush]
  have hcnoinc : ¬ (Board.colHoles b c < Board.colHoles (pl.place b) c) := by
    rw [colHoles_place_eq_of_flush hcflushmem]
    omega
  have hsub : (Finset.range 10).filter (fun j =>
      Board.colHoles b j < Board.colHoles (pl.place b) j)
      ⊆ ((Finset.range 10).filter
        (fun j => ¬ (pl.colProfile j = 0))).erase c := by
    intro j hj
    rw [Finset.mem_filter] at hj
    rw [Finset.mem_erase, Finset.mem_filter]
    refine ⟨?_, hj.1, ?_⟩
    · intro hjc
      rw [hjc] at hj
      exact hcnoinc hj.2
    · intro hz
      rw [colHoles_place_eq_of_unfed (b := b) hz] at hj
      omega
  have htouch := placement_touched_columns_le_four hv
  have hcmem : c ∈ (Finset.range 10).filter
      (fun j => ¬ (pl.colProfile j = 0)) := by
    rw [Finset.mem_filter, Finset.mem_range]
    exact ⟨hclt, by omega⟩
  calc ((Finset.range 10).filter (fun j =>
        Board.colHoles b j < Board.colHoles (pl.place b) j)).card
      ≤ (((Finset.range 10).filter
          (fun j => ¬ (pl.colProfile j = 0))).erase c).card :=
        Finset.card_le_card hsub
    _ = ((Finset.range 10).filter
          (fun j => ¬ (pl.colProfile j = 0))).card - 1 :=
        Finset.card_erase_of_mem hcmem
    _ ≤ 3 := by omega

/-- **The skew tax is exactly one buried cell**: an S dropped on virgin
ground pays exactly one hole in *every* rotation — there is no
orientation that avoids the bill, and none that doubles it. -/
theorem S_flat_hole_bill :
    ∀ r : Rotation,
      Board.holes GameConfig.standard
        ((⟨Piece.S, r, 0⟩ : Placement).place Board.empty) = 1 := by
  decide

/-- The Z pays the same one-hole tax in every rotation. Together with the
grounded-rotation classification: on flat ground the skew pair costs
exactly one hole each, and every other piece has a free orientation. -/
theorem Z_flat_hole_bill :
    ∀ r : Rotation,
      Board.holes GameConfig.standard
        ((⟨Piece.Z, r, 0⟩ : Placement).place Board.empty) = 1 := by
  decide

/-- **One step of stagger absorbs the S entirely**: on a board with a
single cell forming a one-step rise to the right, the S drops with zero
holes — the skew tax is a property of *flat* ground, not of the piece.
Roughness is the currency that pays for skew pieces (kernel witness). -/
theorem S_fits_stagger :
    Board.holes GameConfig.standard
      ((⟨Piece.S, 0, 0⟩ : Placement).place
        ({(2, 0)} : Finset Coord)) = 0 := by
  decide

/-- The mirrored stagger absorbs the Z. Together with the exact flat
bills: the skew pair each demand precisely one step of relief, in
opposite directions — the formal seed of why survival needs a rough,
alternating surface. -/
theorem Z_fits_stagger :
    Board.holes GameConfig.standard
      ((⟨Piece.Z, 0, 0⟩ : Placement).place
        ({(0, 0)} : Finset Coord)) = 0 := by
  decide

/-- **Total hole-neutrality is pointwise**: a merge preserves the total
hole count iff it preserves every column's — since no column can lose,
one column's gain shows in the sum. Reduces global hole-neutrality
questions to per-column landing-gap checks. -/
theorem holes_place_eq_iff {b : Board} {pl : Placement} :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
    ↔ ∀ j < 10,
        Board.colHoles (pl.place b) j = Board.colHoles b j := by
  constructor
  · intro h j hj
    by_contra hne
    have hge := colHoles_place_ge (b := b) (pl := pl) j
    have hgt : Board.colHoles b j < Board.colHoles (pl.place b) j := by
      omega
    have hlt := Finset.sum_lt_sum (s := Finset.range 10)
      (f := fun j => Board.colHoles b j)
      (g := fun j => Board.colHoles (pl.place b) j)
      (fun i _ => colHoles_place_ge i)
      ⟨j, Finset.mem_range.mpr hj, hgt⟩
    unfold Board.holes at h
    rw [GameConfig.standard_cols] at h
    simp only [] at hlt
    omega
  · intro h
    unfold Board.holes
    rw [GameConfig.standard_cols]
    exact Finset.sum_congr rfl (fun j hj => h j (Finset.mem_range.mp hj))

/-- The debt-free harvest, on traces: a four-clear step never raises the
board's hole count. -/
theorem trace_tetris_no_new_holes {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (h4 : (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init (m + 1)).board
      ≤ Board.holes GameConfig.standard
        (trace GameConfig.standard π GameState.init m).board := by
  rw [trace_succ, GameState.step_board]
  exact tetris_no_new_holes
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m)
    (hv _) (fun r => trace_board_no_full m r) h4

/-- **The low-window survival scaffold**: a policy that at every step
plays entirely inside some adjacent column pair standing at least four
rows below the ceiling survives forever. Together with
`headroom_move_exists` (such a move exists for every piece whenever such
a pair exists), solving Tetris reduces to one question: *can a low
two-column window always be maintained under the 7-bag?* Everything else
about survival is settled. -/
theorem survivesForever_of_low_pair_play {π : Policy GameConfig.standard}
    (h : ∀ n, ∃ j, j + 1 < 10
      ∧ (trace GameConfig.standard π GameState.init n).board.colHeight j
          + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init n).board.colHeight
          (j + 1) + 4 ≤ 20
      ∧ ∀ cell ∈ (π (trace GameConfig.standard π GameState.init n)).shapeUp,
          (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = j
          ∨ (π (trace GameConfig.standard π GameState.init n)).col + cell.1
            = j + 1) :
    SurvivesForever GameConfig.standard π GameState.init := by
  apply survivesForever_of_headroom
  intro n cell hcell
  obtain ⟨j, hj, h1, h2, hcells⟩ := h n
  rw [GameConfig.standard_rows]
  rcases hcells cell hcell with hc | hc <;> rw [hc]
  · exact h1
  · exact h2

/-- A placement confined to two adjacent columns splits its four cells
between them. -/
theorem colProfile_pair_of_confined {pl : Placement} {j : ℕ}
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    pl.colProfile j + pl.colProfile (j + 1) = 4 := by
  classical
  unfold Placement.colProfile
  have hcong : pl.shapeUp.filter (fun cell => pl.col + cell.1 = j + 1)
      = pl.shapeUp.filter (fun cell => ¬ (pl.col + cell.1 = j)) := by
    apply Finset.filter_congr
    intro cell hmem
    constructor
    · intro h
      omega
    · intro h
      rcases hcells cell hmem with h1 | h1
      · exact absurd h1 h
      · exact h1
  rw [hcong, Finset.card_filter_add_card_filter_not
    (s := pl.shapeUp) (fun cell => pl.col + cell.1 = j), pl.shapeUp_card]

/-- **A window burns out in eight**: confining nine consecutive clear-free
drops to one adjacent column pair pushes some column of the pair past
height sixteen — a low two-column window absorbs at most eight pieces
before it must clear or move. The quantitative wall of the low-window
reduction: window maintenance forces clears (or migration) at least once
per eight drops. -/
theorem window_burnout {π : Policy GameConfig.standard} {n w j : ℕ}
    (hdry : cleared GameConfig.standard π GameState.init (n + w)
      = cleared GameConfig.standard π GameState.init n)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hw : 9 ≤ w) :
    ¬ ((trace GameConfig.standard π GameState.init (n + w)).board.colHeight j
          + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init (n + w)).board.colHeight
          (j + 1) + 4 ≤ 20) := by
  classical
  -- pair cell-count grows by exactly 4 per confined dry step
  have hgrow : ∀ v, v ≤ w →
      (trace GameConfig.standard π GameState.init n).board.colCount j
        + (trace GameConfig.standard π GameState.init n).board.colCount
          (j + 1) + 4 * v
      ≤ (trace GameConfig.standard π GameState.init (n + v)).board.colCount j
        + (trace GameConfig.standard π GameState.init
            (n + v)).board.colCount (j + 1) := by
    intro v
    induction v with
    | zero =>
      intro _
      simp
    | succ k ih =>
      intro hv
      have hm1 := cleared_mono GameConfig.standard π GameState.init
        (Nat.le_add_right n k)
      have hm2 := cleared_mono GameConfig.standard π GameState.init
        (show n + k ≤ (n + k) + 1 by omega)
      have hm3 := cleared_mono GameConfig.standard π GameState.init
        (show (n + k) + 1 ≤ n + w by omega)
      have hs := cleared_succ GameConfig.standard π GameState.init (n + k)
      have hcard0 : (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init (n + k))).place
            (trace GameConfig.standard π GameState.init (n + k)).board)).card
          = 0 := by
        omega
      have hnc := Finset.card_eq_zero.mp hcard0
      have hid : Placement.applyStep GameConfig.standard
          (trace GameConfig.standard π GameState.init (n + k)).board
          (π (trace GameConfig.standard π GameState.init (n + k)))
          = (π (trace GameConfig.standard π GameState.init (n + k))).place
            (trace GameConfig.standard π GameState.init (n + k)).board := by
        unfold Placement.applyStep
        exact Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hnc
      have hcp1 := Placement.colCount_place
        (trace GameConfig.standard π GameState.init (n + k)).board
        (π (trace GameConfig.standard π GameState.init (n + k))) j
      have hcp2 := Placement.colCount_place
        (trace GameConfig.standard π GameState.init (n + k)).board
        (π (trace GameConfig.standard π GameState.init (n + k))) (j + 1)
      have hsplit := colProfile_pair_of_confined
        (pl := π (trace GameConfig.standard π GameState.init (n + k)))
        (hcells k (by omega))
      have hihk := ih (by omega)
      rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
        GameState.step_board, hid]
      omega
  have hg := hgrow w (le_refl w)
  intro hcon
  have hc1 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board j
  have hc2 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board (j + 1)
  omega

/-- **The window ledger**: over `w` drops confined to the adjacent pair
`(j, j+1)`, the pair's cell count plus twice the rows cleared equals its
start plus `4w` — all four cells of every drop land in the pair, and each
cleared row bills the pair two. -/
theorem window_feed_ledger {π : Policy GameConfig.standard} {n j : ℕ}
    (hj : j + 1 < 10) :
    ∀ w, (∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1) →
      (trace GameConfig.standard π GameState.init (n + w)).board.colCount j
        + (trace GameConfig.standard π GameState.init
            (n + w)).board.colCount (j + 1)
        + 2 * (cleared GameConfig.standard π GameState.init (n + w)
            - cleared GameConfig.standard π GameState.init n)
      = (trace GameConfig.standard π GameState.init n).board.colCount j
        + (trace GameConfig.standard π GameState.init n).board.colCount
            (j + 1)
        + 4 * w := by
  intro w
  induction w with
  | zero =>
    intro _
    simp
  | succ k ih =>
    intro hcells
    have hihk := ih (fun i hi => hcells i (by omega))
    have hm1 := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right n k)
    have hm2 := cleared_mono GameConfig.standard π GameState.init
      (show n + k ≤ (n + k) + 1 by omega)
    have hs := cleared_succ GameConfig.standard π GameState.init (n + k)
    have ha1 := applyStep_colCount GameConfig.standard
      (trace GameConfig.standard π GameState.init (n + k)).board
      (π (trace GameConfig.standard π GameState.init (n + k)))
      (j := j) (by rw [GameConfig.standard_cols]; omega)
    have ha2 := applyStep_colCount GameConfig.standard
      (trace GameConfig.standard π GameState.init (n + k)).board
      (π (trace GameConfig.standard π GameState.init (n + k)))
      (j := j + 1) (by rw [GameConfig.standard_cols]; omega)
    have hsplit := colProfile_pair_of_confined
      (pl := π (trace GameConfig.standard π GameState.init (n + k)))
      (hcells k (by omega))
    unfold Board.linesCleared at ha1 ha2
    rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
      GameState.step_board]
    omega

/-- **A sustained window clears two rows per drop**: keeping the pair low
through `w` confined drops forces at least `2w − 16` cleared rows — five
times the global 0.4-per-move speed limit. Fixed-window play is possible
only in short bursts; a survivor's window must migrate, and this is the
exact price of staying. -/
theorem window_sustain_clear_rate {π : Policy GameConfig.standard}
    {n j w : ℕ} (hj : j + 1 < 10)
    (hcells : ∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          = j
        ∨ (π (trace GameConfig.standard π GameState.init (n + k))).col
            + cell.1 = j + 1)
    (hlow : (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight j + 4 ≤ 20
      ∧ (trace GameConfig.standard π GameState.init
          (n + w)).board.colHeight (j + 1) + 4 ≤ 20) :
    2 * w ≤ 2 * (cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n) + 16 := by
  have hled := window_feed_ledger (n := n) hj w hcells
  have hc1 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board j
  have hc2 := colCount_le_colHeight
    (trace GameConfig.standard π GameState.init (n + w)).board (j + 1)
  omega

/-- A placement confined to a column set splits its four cells among the
set's columns. Generalizes `colProfile_pair_of_confined` to any width. -/
theorem colProfile_sum_of_confined {pl : Placement} {S : Finset ℕ}
    (hcells : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ∈ S) :
    ∑ j ∈ S, pl.colProfile j = 4 := by
  classical
  have h := Finset.card_eq_sum_card_fiberwise
    (f := fun cell => pl.col + cell.1) (s := pl.shapeUp) (t := S)
    (fun cell hcell => hcells cell (Finset.mem_coe.mp hcell))
  rw [pl.shapeUp_card] at h
  unfold Placement.colProfile
  exact h.symm

/-- **The width-`k` confinement ledger**: over `w` drops confined to any
column set `S`, the set's cell count plus `|S|` per cleared row equals its
start plus `4w` — every drop feeds the set four, every cleared row bills
it `|S|`. The pair ledger at every width. -/
theorem window_feed_ledger_set {π : Policy GameConfig.standard} {n : ℕ}
    {S : Finset ℕ} (hS : ∀ j ∈ S, j < 10) :
    ∀ w, (∀ k < w,
      ∀ cell ∈ (π (trace GameConfig.standard π GameState.init (n + k))).shapeUp,
        (π (trace GameConfig.standard π GameState.init (n + k))).col + cell.1
          ∈ S) →
      (∑ j ∈ S, (trace GameConfig.standard π GameState.init
          (n + w)).board.colCount j)
        + S.card * (cleared GameConfig.standard π GameState.init (n + w)
            - cleared GameConfig.standard π GameState.init n)
      = (∑ j ∈ S, (trace GameConfig.standard π GameState.init
          n).board.colCount j)
        + 4 * w := by
  intro w
  induction w with
  | zero =>
    intro _
    simp
  | succ k ih =>
    intro hcells
    have hihk := ih (fun i hi => hcells i (by omega))
    have hm1 := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right n k)
    have hm2 := cleared_mono GameConfig.standard π GameState.init
      (show n + k ≤ (n + k) + 1 by omega)
    have hs := cleared_succ GameConfig.standard π GameState.init (n + k)
    have hstep : ∀ j ∈ S,
        (Placement.applyStep GameConfig.standard
            (trace GameConfig.standard π GameState.init (n + k)).board
            (π (trace GameConfig.standard π GameState.init (n + k)))).colCount
            j
          + Board.linesCleared GameConfig.standard
              ((π (trace GameConfig.standard π GameState.init (n + k))).place
                (trace GameConfig.standard π GameState.init (n + k)).board)
        = (trace GameConfig.standard π GameState.init (n + k)).board.colCount
            j
          + (π (trace GameConfig.standard π GameState.init
              (n + k))).colProfile j :=
      fun j hj => applyStep_colCount GameConfig.standard _ _
        (by rw [GameConfig.standard_cols]; exact hS j hj)
    have hsum := Finset.sum_congr rfl hstep
    simp only [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul]
      at hsum
    have hprof := colProfile_sum_of_confined
      (pl := π (trace GameConfig.standard π GameState.init (n + k)))
      (hcells k (by omega))
    have hdiff : cleared GameConfig.standard π GameState.init ((n + k) + 1)
        - cleared GameConfig.standard π GameState.init n
      = (cleared GameConfig.standard π GameState.init (n + k)
          - cleared GameConfig.standard π GameState.init n)
        + (Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init (n + k))).place
              (trace GameConfig.standard π GameState.init
                (n + k)).board)).card := by
      omega
    have hdist : S.card
          * (cleared GameConfig.standard π GameState.init ((n + k) + 1)
            - cleared GameConfig.standard π GameState.init n)
        = S.card * (cleared GameConfig.standard π GameState.init (n + k)
            - cleared GameConfig.standard π GameState.init n)
          + S.card * (Board.fullRows GameConfig.standard
              ((π (trace GameConfig.standard π GameState.init (n + k))).place
                (trace GameConfig.standard π GameState.init
                  (n + k)).board)).card := by
      rw [hdiff, Nat.mul_add]
    unfold Board.linesCleared at hsum
    rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
      GameState.step_board]
    simp only [] at hihk hsum hprof hdist ⊢
    omega

/-- A board with a full row occupies every column. -/
theorem colCount_pos_of_fullRow {cfg : GameConfig} {b : Board} {r : ℕ}
    (hr : r ∈ Board.fullRows cfg b) {j : ℕ} (hj : j < cfg.cols) :
    1 ≤ b.colCount j := by
  have hfull := (Finset.mem_filter.mp hr).2
  have hmem : (j, r) ∈ b := hfull j (Finset.mem_range.mpr hj)
  unfold Board.colCount
  rw [Nat.succ_le_iff, Finset.card_pos]
  exact ⟨(j, r), Finset.mem_filter.mpr ⟨hmem, rfl⟩⟩

/-- **A clearing move spans the board**: at any step where the cleared
count jumps, the merged board occupies all ten columns — a cleared row is
a certificate of full-width play. -/
theorem clearing_move_spans_board {π : Policy GameConfig.standard} {m : ℕ}
    (hjump : cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1)) :
    ∀ j < 10, 1 ≤ ((π (trace GameConfig.standard π GameState.init m)).place
      (trace GameConfig.standard π GameState.init m).board).colCount j := by
  intro j hj
  have hs := cleared_succ GameConfig.standard π GameState.init m
  have hpos : 0 < (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card := by
    omega
  obtain ⟨r, hr⟩ := Finset.card_pos.mp hpos
  exact colCount_pos_of_fullRow hr
    (by rw [GameConfig.standard_cols]; omega)

/-- **Full rows are mass**: each full row banks `cols` cells, so a board
about to clear `k` rows holds at least `cols · k` cells. -/
theorem mass_floor_of_fullRows {cfg : GameConfig} (b : Board) :
    cfg.cols * (Board.fullRows cfg b).card ≤ b.count := by
  classical
  have hsub : (Board.fullRows cfg b).biUnion
      (fun r => (Finset.range cfg.cols).image (fun c => (c, r))) ⊆ b := by
    intro p hp
    obtain ⟨r, hr, hpmem⟩ := Finset.mem_biUnion.mp hp
    obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hpmem
    exact (Finset.mem_filter.mp hr).2 c hc
  have hdisj : ∀ r₁ ∈ Board.fullRows cfg b, ∀ r₂ ∈ Board.fullRows cfg b,
      r₁ ≠ r₂ →
        Disjoint ((Finset.range cfg.cols).image (fun c => (c, r₁)))
          ((Finset.range cfg.cols).image (fun c => (c, r₂))) := by
    intro r₁ _ r₂ _ hne
    rw [Finset.disjoint_left]
    intro p hp1 hp2
    obtain ⟨c1, _, rfl⟩ := Finset.mem_image.mp hp1
    obtain ⟨c2, _, heq⟩ := Finset.mem_image.mp hp2
    have hsnd := congrArg Prod.snd heq
    exact hne hsnd.symm
  have hcard := Finset.card_biUnion hdisj
  have himg : ∀ r : ℕ,
      ((Finset.range cfg.cols).image (fun c => (c, r))).card = cfg.cols := by
    intro r
    rw [Finset.card_image_of_injective _
      (fun a b h => (Prod.ext_iff.mp h).1), Finset.card_range]
  have hsum : ((Board.fullRows cfg b).biUnion
      (fun r => (Finset.range cfg.cols).image (fun c => (c, r)))).card
      = cfg.cols * (Board.fullRows cfg b).card := by
    rw [hcard, Finset.sum_congr rfl (fun r _ => himg r), Finset.sum_const,
      smul_eq_mul, Nat.mul_comm]
  unfold Board.count
  rw [← hsum]
  exact Finset.card_le_card hsub

/-- **The clearing moment banks its clears**: a step that clears `k` rows
does so from a merged board holding at least `10k` cells — the mass being
harvested is visible on the board at the moment of harvest. -/
theorem clearing_move_mass_floor {π : Policy GameConfig.standard} {m : ℕ} :
    10 * (cleared GameConfig.standard π GameState.init (m + 1)
        - cleared GameConfig.standard π GameState.init m)
      ≤ ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board).count := by
  have hs := cleared_succ GameConfig.standard π GameState.init m
  have h := mass_floor_of_fullRows (cfg := GameConfig.standard)
    ((π (trace GameConfig.standard π GameState.init m)).place
      (trace GameConfig.standard π GameState.init m).board)
  rw [GameConfig.standard_cols] at h
  omega

/-- **Growth localizes to a step**: any function of the step counter that
grows over a window grows at some single step of it. The bridge from
window-level clear counts to clearing *moments*, for any counter. -/
theorem exists_jump_of_lt {f : ℕ → ℕ} {N w : ℕ} (h : f N < f (N + w)) :
    ∃ k < w, f (N + k) < f ((N + k) + 1) := by
  by_contra hnone
  push Not at hnone
  have hflat : ∀ v, v ≤ w → f (N + v) ≤ f N := by
    intro v
    induction v with
    | zero =>
      intro _
      simp
    | succ k ih =>
      intro hvw
      have hle := hnone k (by omega)
      have hik := ih (by omega)
      rw [show N + (k + 1) = (N + k) + 1 by omega]
      omega
  have hw := hflat w (le_refl w)
  omega

/-- Clearing steps never decrease. -/
theorem clearingSteps_mono {cfg : GameConfig} {π : Policy cfg} :
    Monotone (clearingSteps cfg π GameState.init) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [clearingSteps_succ]
  exact Nat.le_add_right _ _

/-- Windowed pricing of clearing steps, upper side: over any window the
rows cleared are at most four per clearing step of the window. -/
theorem cleared_window_le_four_mul_clearingSteps {cfg : GameConfig}
    {π : Policy cfg} (n : ℕ) :
    ∀ w, cleared cfg π GameState.init (n + w)
        - cleared cfg π GameState.init n
      ≤ 4 * (clearingSteps cfg π GameState.init (n + w)
          - clearingSteps cfg π GameState.init n) := by
  intro w
  induction w with
  | zero => simp
  | succ k ih =>
    have h4 := fullRows_card_le_four (cfg := cfg) (π := π) (n + k)
    have hc := cleared_succ cfg π GameState.init (n + k)
    have hcs := clearingSteps_succ cfg π GameState.init (n + k)
    have hm1 := cleared_mono cfg π GameState.init (Nat.le_add_right n k)
    have hms := clearingSteps_mono (cfg := cfg) (π := π)
      (Nat.le_add_right n k)
    rw [show n + (k + 1) = (n + k) + 1 by omega]
    by_cases hpos : 0 < (Board.fullRows cfg
        ((π (trace cfg π GameState.init (n + k))).place
          (trace cfg π GameState.init (n + k)).board)).card
    · rw [if_pos hpos] at hcs
      omega
    · rw [if_neg hpos] at hcs
      omega

/-- Windowed pricing of clearing steps, lower side: over any window each
clearing step clears at least one row. -/
theorem clearingSteps_window_le_cleared {cfg : GameConfig}
    {π : Policy cfg} (n : ℕ) :
    ∀ w, clearingSteps cfg π GameState.init (n + w)
        - clearingSteps cfg π GameState.init n
      ≤ cleared cfg π GameState.init (n + w)
        - cleared cfg π GameState.init n := by
  intro w
  induction w with
  | zero => simp
  | succ k ih =>
    have hc := cleared_succ cfg π GameState.init (n + k)
    have hcs := clearingSteps_succ cfg π GameState.init (n + k)
    have hm1 := cleared_mono cfg π GameState.init (Nat.le_add_right n k)
    have hms := clearingSteps_mono (cfg := cfg) (π := π)
      (Nat.le_add_right n k)
    rw [show n + (k + 1) = (n + k) + 1 by omega]
    by_cases hpos : 0 < (Board.fullRows cfg
        ((π (trace cfg π GameState.init (n + k))).place
          (trace cfg π GameState.init (n + k)).board)).card
    · rw [if_pos hpos] at hcs
      omega
    · rw [if_neg hpos] at hcs
      omega

/-- The clearing-step counter over a window is exactly the number of
clearing moments in it. -/
theorem clearingSteps_window_card {cfg : GameConfig} {π : Policy cfg}
    (n : ℕ) :
    ∀ w, clearingSteps cfg π GameState.init (n + w)
        - clearingSteps cfg π GameState.init n
      = ((Finset.range w).filter (fun k =>
          0 < (Board.fullRows cfg
            ((π (trace cfg π GameState.init (n + k))).place
              (trace cfg π GameState.init (n + k)).board)).card)).card := by
  classical
  intro w
  induction w with
  | zero => simp
  | succ k ih =>
    rw [show n + (k + 1) = (n + k) + 1 by omega, clearingSteps_succ,
      Finset.range_add_one, Finset.filter_insert]
    have hms := clearingSteps_mono (cfg := cfg) (π := π)
      (Nat.le_add_right n k)
    by_cases hpos : 0 < (Board.fullRows cfg
        ((π (trace cfg π GameState.init (n + k))).place
          (trace cfg π GameState.init (n + k)).board)).card
    · rw [if_pos hpos, if_pos hpos,
        Finset.card_insert_of_notMem (by simp)]
      omega
    · rw [if_neg hpos, if_neg hpos]
      omega

/-- **The clearing-piece band holds on every window**: across any live
`w`-move window, the clearing moments number at least `w/10 − 5` and at
most `2w/5 + 20` — the lifetime 1/10–2/5 law, localized to every position
and scale. No stretch of play, anywhere, can clear much rarer than one
piece in ten or much oftener than two in five. -/
theorem clearingSteps_window_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m w : ℕ}
    (hlive_m : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard)
    (hlive_mw : ¬ (trace GameConfig.standard π GameState.init (m + w)).lost
      GameConfig.standard) :
    4 * w ≤ 40 * (clearingSteps GameConfig.standard π GameState.init (m + w)
        - clearingSteps GameConfig.standard π GameState.init m) + 200
      ∧ 10 * (clearingSteps GameConfig.standard π GameState.init (m + w)
        - clearingSteps GameConfig.standard π GameState.init m)
        ≤ 4 * w + 200 := by
  have hband := cleared_window_band hv hlive_m hlive_mw
  have hup := cleared_window_le_four_mul_clearingSteps
    (cfg := GameConfig.standard) (π := π) m w
  have hlo := clearingSteps_window_le_cleared
    (cfg := GameConfig.standard) (π := π) m w
  exact ⟨by omega, by omega⟩

/-- A survivor's clearing-piece fraction obeys the 1/10–2/5 band on every
window of its play. -/
theorem survivor_clearingSteps_window_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    (m w : ℕ) :
    4 * w ≤ 40 * (clearingSteps GameConfig.standard π GameState.init (m + w)
        - clearingSteps GameConfig.standard π GameState.init m) + 200
      ∧ 10 * (clearingSteps GameConfig.standard π GameState.init (m + w)
        - clearingSteps GameConfig.standard π GameState.init m)
        ≤ 4 * w + 200 :=
  clearingSteps_window_band hv (hsurv m) (hsurv (m + w))

/-- **The board-level inventory price**: a drop that completes `k` rows
lands on a board holding at least `cols·k − 4` cells — full rows carry
`cols` cells each and the piece brings four. -/
theorem fullRows_place_card_le_count {cfg : GameConfig} (b : Board)
    (pl : Placement) :
    cfg.cols * (Board.fullRows cfg (pl.place b)).card ≤ b.count + 4 := by
  have h := mass_floor_of_fullRows (cfg := cfg) (pl.place b)
  rw [Placement.count_place] at h
  exact h

/-- An unfed column never rises through a full move: unchanged by the
merge, lowered (or untouched) by the clear. -/
theorem applyStep_unfed_colHeight_le {cfg : GameConfig} {b : Board}
    {pl : Placement} {j : ℕ} (hj : j < cfg.cols)
    (hz : pl.colProfile j = 0) :
    (Placement.applyStep cfg b pl).colHeight j ≤ b.colHeight j := by
  by_cases hk : 0 < (Board.fullRows cfg (pl.place b)).card
  · have h := clear_step_unfed_colHeight_le hj hz hk
    omega
  · have hcard0 : (Board.fullRows cfg (pl.place b)).card = 0 := by omega
    have hnc := Finset.card_eq_zero.mp hcard0
    have hid : Placement.applyStep cfg b pl = pl.place b := by
      unfold Placement.applyStep
      exact Board.clearLines_eq_self_of_no_fullRows cfg hnc
    rw [hid]
    exact le_of_eq (place_unfed_colHeight_eq hz)

/-- A clear-free move never lowers any column: the merge only adds
cells, and without a full row the clear phase is the identity. -/
theorem dry_step_colHeight_ge {cfg : GameConfig} {b : Board}
    {pl : Placement} (j : ℕ)
    (hnc : (Board.fullRows cfg (pl.place b)).card = 0) :
    b.colHeight j ≤ (Placement.applyStep cfg b pl).colHeight j := by
  have hid : Placement.applyStep cfg b pl = pl.place b := by
    unfold Placement.applyStep
    exact Board.clearLines_eq_self_of_no_fullRows cfg
      (Finset.card_eq_zero.mp hnc)
  rw [hid, Placement.place_eq_union_dropped]
  exact colHeight_mono Finset.subset_union_left j

/-- **The skyline sinks only at clearing moments**: any drop in any
column's height across a step certifies that the step cleared. Height
relief is never free — it is always bought with a full row. -/
theorem height_drop_certifies_clear {π : Policy GameConfig.standard}
    {m j : ℕ}
    (hdrop : (trace GameConfig.standard π GameState.init (m + 1)).board.colHeight j
      < (trace GameConfig.standard π GameState.init m).board.colHeight j) :
    cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1) := by
  have hs := cleared_succ GameConfig.standard π GameState.init m
  by_contra hnone
  have hcard0 : (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card = 0 := by
    omega
  have hge := dry_step_colHeight_ge (cfg := GameConfig.standard)
    (b := (trace GameConfig.standard π GameState.init m).board)
    (pl := π (trace GameConfig.standard π GameState.init m)) j hcard0
  rw [trace_succ, GameState.step_board] at hdrop
  omega

/-- On a well-formed board every column — in range or not — is bounded
by the board max-height: out-of-range columns are empty. -/
theorem colHeight_le_maxHeight_of_wf {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) (j : ℕ) :
    b.colHeight j ≤ Board.maxHeight cfg b := by
  by_cases hj : j < cfg.cols
  · exact Board.colHeight_le_maxHeight hj
  · have hfilter : b.filter (fun p => p.1 = j) = ∅ := by
      rw [Finset.eq_empty_iff_forall_notMem]
      intro p hp
      rw [Finset.mem_filter] at hp
      have := hwf p hp.1
      omega
    unfold Board.colHeight Board.colRows
    rw [hfilter]
    simp

/-- **The board top climbs at most four per move**: the merge can raise
the skyline's peak by at most one piece height (the drop rests on some
existing column) and the clear phase only lowers it. -/
theorem applyStep_maxHeight_le {cfg : GameConfig} {b : Board}
    {pl : Placement} (hwf : Board.WF cfg b) :
    Board.maxHeight cfg (Placement.applyStep cfg b pl)
      ≤ Board.maxHeight cfg b + 4 := by
  unfold Board.maxHeight
  apply Finset.sup_le
  intro j _
  have hcl := colHeight_clearLines_le cfg (pl.place b) j
  obtain ⟨j', h⟩ := place_colHeight_le (b := b) (pl := pl) j
  have hb := colHeight_le_maxHeight_of_wf hwf j'
  unfold Placement.applyStep
  unfold Board.maxHeight at hb
  omega

/-- **The window climb budget for the whole board**: over any `w`-move
window the skyline's peak rises at most `4w`. -/
theorem trace_maxHeight_window_climb {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    ∀ w, Board.maxHeight GameConfig.standard
        (trace GameConfig.standard π GameState.init (n + w)).board
      ≤ Board.maxHeight GameConfig.standard
          (trace GameConfig.standard π GameState.init n).board + 4 * w := by
  intro w
  induction w with
  | zero => simp
  | succ k ih =>
    have hwf := trace_board_wf hv
      (GameState.init_board_wf GameConfig.standard) (n + k)
    have hstep := applyStep_maxHeight_le (cfg := GameConfig.standard)
      (b := (trace GameConfig.standard π GameState.init (n + k)).board)
      (pl := π (trace GameConfig.standard π GameState.init (n + k))) hwf
    rw [show n + (k + 1) = (n + k) + 1 by omega, trace_succ,
      GameState.step_board]
    omega

/-- A lost well-formed board's peak reaches at least twenty-one: some
cell sits at or above row twenty. -/
theorem lost_maxHeight_ge {cfg : GameConfig} {b : Board}
    (hwf : Board.WF cfg b) (hlost : Board.isLost cfg b)
    (hrows : cfg.rows = 20) :
    21 ≤ Board.maxHeight cfg b := by
  obtain ⟨p, hp, hr⟩ := hlost
  have hlt := Board.lt_colHeight hp
  have hcol := hwf p hp
  have hle := Board.colHeight_le_maxHeight (b := b) hcol
  omega

/-- **Death needs time**: a game live with peak height `h` at step `n`
cannot be lost before `4w` moves have raised the peak past twenty — the
climb budget prices the survival horizon. -/
theorem death_needs_time {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n w h : ℕ}
    (hh : Board.maxHeight GameConfig.standard
      (trace GameConfig.standard π GameState.init n).board ≤ h)
    (hlost : (trace GameConfig.standard π GameState.init (n + w)).lost
      GameConfig.standard) :
    21 ≤ h + 4 * w := by
  have hclimb := trace_maxHeight_window_climb hv n w
  have hwf := trace_board_wf hv
    (GameState.init_board_wf GameConfig.standard) (n + w)
  have hge := lost_maxHeight_ge hwf
    ((GameState.lost_iff_board_isLost GameConfig.standard _).mp hlost)
    GameConfig.standard_rows
  omega

/-- **No game dies before its sixth move**: from the empty board the peak
climbs at most four per move, and death needs a peak past twenty. -/
theorem no_death_before_six {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {w : ℕ}
    (hlost : (trace GameConfig.standard π GameState.init w).lost
      GameConfig.standard) :
    6 ≤ w := by
  have h := death_needs_time hv (n := 0) (w := w) (h := 0)
    (by
      rw [trace_zero, GameState.init_board_eq_emptyset,
        Board.maxHeight_empty])
    (by rw [Nat.zero_add]; exact hlost)
  omega

/-- A clear-free move never lowers the skyline's peak. -/
theorem dry_step_maxHeight_ge {cfg : GameConfig} {b : Board}
    {pl : Placement}
    (hnc : (Board.fullRows cfg (pl.place b)).card = 0) :
    Board.maxHeight cfg b
      ≤ Board.maxHeight cfg (Placement.applyStep cfg b pl) := by
  unfold Board.maxHeight
  exact Finset.sup_mono_fun (fun j _ => dry_step_colHeight_ge j hnc)

/-- **The peak falls only at clearing moments**: any drop in the board's
maximum height across a step certifies a clear — the tallest column can
only be relieved by a full row. -/
theorem peak_drop_certifies_clear {π : Policy GameConfig.standard} {m : ℕ}
    (hdrop : Board.maxHeight GameConfig.standard
        (trace GameConfig.standard π GameState.init (m + 1)).board
      < Board.maxHeight GameConfig.standard
          (trace GameConfig.standard π GameState.init m).board) :
    cleared GameConfig.standard π GameState.init m
      < cleared GameConfig.standard π GameState.init (m + 1) := by
  classical
  obtain ⟨j, hj, hmax⟩ := Finset.exists_max_image (Finset.range 10)
    (trace GameConfig.standard π GameState.init m).board.colHeight
    ⟨0, by simp⟩
  have hsup : Board.maxHeight GameConfig.standard
      (trace GameConfig.standard π GameState.init m).board
      ≤ (trace GameConfig.standard π GameState.init m).board.colHeight j := by
    unfold Board.maxHeight
    rw [GameConfig.standard_cols]
    exact Finset.sup_le (fun j' hj' => hmax j' hj')
  have hle : (trace GameConfig.standard π
      GameState.init (m + 1)).board.colHeight j
      ≤ Board.maxHeight GameConfig.standard
        (trace GameConfig.standard π GameState.init (m + 1)).board :=
    Board.colHeight_le_maxHeight
      (by rw [GameConfig.standard_cols]; exact Finset.mem_range.mp hj)
  apply height_drop_certifies_clear (j := j)
  omega

/-- An unfed column's hole debt never grows through a full move: the
merge leaves its cells untouched and the clear phase only repairs. -/
theorem applyStep_unfed_colHoles_le {cfg : GameConfig} {b : Board}
    {pl : Placement} {j : ℕ} (hj : j < cfg.cols)
    (hz : pl.colProfile j = 0) :
    Board.colHoles (Placement.applyStep cfg b pl) j
      ≤ Board.colHoles b j := by
  have hcl := colHoles_clearLines_le (cfg := cfg) (b := pl.place b) hj
  have hpl := colHoles_place_eq_of_unfed (b := b) (pl := pl) hz
  unfold Placement.applyStep
  omega

/-- Every I rotation is single-column or four wide — nothing between. -/
theorem I_shape_single_or_wide :
    ∀ r : Rotation,
      (∀ cell ∈ Piece.I.shapeUp r, ∀ cell' ∈ Piece.I.shapeUp r,
        cell.1 = cell'.1)
      ∨ (∃ cell ∈ Piece.I.shapeUp r, ∃ cell' ∈ Piece.I.shapeUp r,
          cell'.1 + 3 ≤ cell.1) := by
  decide

/-- **A pair-confined I is vertical**: an I whose cells all land in two
adjacent columns stands in a single column — its horizontal rotations
span four. -/
theorem I_pair_confined_single_column {pl : Placement} {j : ℕ}
    (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    ∀ cell ∈ pl.shapeUp, ∀ cell' ∈ pl.shapeUp, cell.1 = cell'.1 := by
  rcases I_shape_single_or_wide pl.rot with h | h
  · intro cell hcell cell' hcell'
    unfold Placement.shapeUp at hcell hcell'
    rw [hI] at hcell hcell'
    exact h cell hcell cell' hcell'
  · exfalso
    obtain ⟨cell, hcell, cell', hcell', hw⟩ := h
    have h1 := hcells cell (by unfold Placement.shapeUp; rw [hI]; exact hcell)
    have h2 := hcells cell'
      (by unfold Placement.shapeUp; rw [hI]; exact hcell')
    omega

/-- **A pair-confined I is a full feed**: it pours all four cells into
one of the window's two columns — inside a low window, the I can only
arrive as a tower brick. -/
theorem I_pair_confined_full_feed {pl : Placement} {j : ℕ}
    (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    ∃ c, (c = j ∨ c = j + 1) ∧ pl.colProfile c = 4 := by
  classical
  have hsingle := I_pair_confined_single_column hI hcells
  have hne : pl.shapeUp.Nonempty := by
    rw [← Finset.card_pos, pl.shapeUp_card]
    omega
  obtain ⟨cell0, hcell0⟩ := hne
  refine ⟨pl.col + cell0.1, hcells cell0 hcell0, ?_⟩
  unfold Placement.colProfile
  have hall : pl.shapeUp.filter
      (fun cell => pl.col + cell.1 = pl.col + cell0.1) = pl.shapeUp := by
    rw [Finset.filter_eq_self]
    intro cell hcell
    have := hsingle cell hcell cell0 hcell0
    omega
  rw [hall, pl.shapeUp_card]

/-- A placement's column profile at offset `t` is its shape's column-`t`
fiber. -/
theorem colProfile_eq_fiber (pl : Placement) (t : ℕ) :
    pl.colProfile (pl.col + t)
      = (pl.shapeUp.filter (fun c => c.1 = t)).card := by
  unfold Placement.colProfile
  congr 1
  apply Finset.filter_congr
  intro c _
  constructor
  · intro h
    omega
  · intro h
    omega

/-- The O occupies columns 0 and 1 with two cells each, in every
rotation. -/
theorem O_shape_columns : ∀ r : Rotation,
    ((Piece.O.shapeUp r).filter (fun c => c.1 = 0)).card = 2
    ∧ ((Piece.O.shapeUp r).filter (fun c => c.1 = 1)).card = 2
    ∧ ∀ cell ∈ Piece.O.shapeUp r, cell.1 ≤ 1 := by
  decide

/-- **A window O splits evenly**: an O confined to an adjacent pair
feeds each of the two columns exactly two cells — the square is the
window's only perfectly balanced brick. -/
theorem O_pair_confined_even_split {pl : Placement} {j : ℕ}
    (hO : pl.piece = Piece.O)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    pl.colProfile j = 2 ∧ pl.colProfile (j + 1) = 2 := by
  classical
  obtain ⟨h0, h1, hnarrow⟩ := O_shape_columns pl.rot
  have hsh : pl.shapeUp = Piece.O.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hO]
  obtain ⟨c0, hc0⟩ := Finset.card_pos.mp
    (show 0 < ((Piece.O.shapeUp pl.rot).filter (fun c => c.1 = 0)).card by
      rw [h0]
      omega)
  obtain ⟨c1, hc1⟩ := Finset.card_pos.mp
    (show 0 < ((Piece.O.shapeUp pl.rot).filter (fun c => c.1 = 1)).card by
      rw [h1]
      omega)
  rw [Finset.mem_filter] at hc0 hc1
  have hj0 := hcells c0 (by rw [hsh]; exact hc0.1)
  have hj1 := hcells c1 (by rw [hsh]; exact hc1.1)
  have hcol : pl.col = j := by omega
  constructor
  · rw [show j = pl.col + 0 by omega, colProfile_eq_fiber pl 0, hsh]
    exact h0
  · rw [show j + 1 = pl.col + 1 by omega, colProfile_eq_fiber pl 1, hsh]
    exact h1

/-- S and Z rotations either sit in columns 0–1 with a 2 + 2 split or
reach column 2. -/
theorem SZ_shape_window_split :
    ∀ p : Piece, p = Piece.S ∨ p = Piece.Z → ∀ r : Rotation,
      (((p.shapeUp r).filter (fun c => c.1 = 0)).card = 2
        ∧ ((p.shapeUp r).filter (fun c => c.1 = 1)).card = 2
        ∧ ∀ cell ∈ p.shapeUp r, cell.1 ≤ 1)
      ∨ (∃ cell ∈ p.shapeUp r, cell.1 = 2) := by
  decide

/-- L, J and T rotations either sit in columns 0–1 with a 3 + 1 split
(one way or the other) or reach column 2. -/
theorem LJT_shape_window_split :
    ∀ p : Piece, p = Piece.L ∨ p = Piece.J ∨ p = Piece.T →
    ∀ r : Rotation,
      (((((p.shapeUp r).filter (fun c => c.1 = 0)).card = 3
          ∧ ((p.shapeUp r).filter (fun c => c.1 = 1)).card = 1)
        ∨ (((p.shapeUp r).filter (fun c => c.1 = 0)).card = 1
          ∧ ((p.shapeUp r).filter (fun c => c.1 = 1)).card = 3))
        ∧ ∀ cell ∈ p.shapeUp r, cell.1 ≤ 1)
      ∨ (∃ cell ∈ p.shapeUp r, cell.1 = 2) := by
  decide

/-- **Window S and Z split evenly**: an S or Z confined to an adjacent
pair stands vertical and feeds each column exactly two cells. -/
theorem SZ_pair_confined_even_split {pl : Placement} {j : ℕ}
    (hSZ : pl.piece = Piece.S ∨ pl.piece = Piece.Z)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    pl.colProfile j = 2 ∧ pl.colProfile (j + 1) = 2 := by
  classical
  rcases SZ_shape_window_split pl.piece hSZ pl.rot with
    ⟨h0, h1, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · obtain ⟨c0, hc0⟩ := Finset.card_pos.mp
      (show 0 < ((pl.piece.shapeUp pl.rot).filter
          (fun c => c.1 = 0)).card by rw [h0]; omega)
    obtain ⟨c1, hc1⟩ := Finset.card_pos.mp
      (show 0 < ((pl.piece.shapeUp pl.rot).filter
          (fun c => c.1 = 1)).card by rw [h1]; omega)
    rw [Finset.mem_filter] at hc0 hc1
    have hj0 := hcells c0 (by unfold Placement.shapeUp; exact hc0.1)
    have hj1 := hcells c1 (by unfold Placement.shapeUp; exact hc1.1)
    have hcol : pl.col = j := by omega
    constructor
    · rw [show j = pl.col + 0 by omega, colProfile_eq_fiber pl 0]
      unfold Placement.shapeUp
      exact h0
    · rw [show j + 1 = pl.col + 1 by omega, colProfile_eq_fiber pl 1]
      unfold Placement.shapeUp
      exact h1
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- **Window L, J and T split three-and-one**: confined to an adjacent
pair they stand vertical, pouring three cells into one column and one
into the other. -/
theorem LJT_pair_confined_split {pl : Placement} {j : ℕ}
    (hLJT : pl.piece = Piece.L ∨ pl.piece = Piece.J ∨ pl.piece = Piece.T)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    (pl.colProfile j = 3 ∧ pl.colProfile (j + 1) = 1)
    ∨ (pl.colProfile j = 1 ∧ pl.colProfile (j + 1) = 3) := by
  classical
  rcases LJT_shape_window_split pl.piece hLJT pl.rot with
    ⟨hsplit, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hget : ∀ a b : ℕ,
        ((pl.piece.shapeUp pl.rot).filter (fun c => c.1 = 0)).card = a →
        ((pl.piece.shapeUp pl.rot).filter (fun c => c.1 = 1)).card = b →
        0 < a → 0 < b →
        pl.colProfile j = a ∧ pl.colProfile (j + 1) = b := by
      intro a b h0 h1 ha hb
      obtain ⟨c0, hc0⟩ := Finset.card_pos.mp
        (show 0 < ((pl.piece.shapeUp pl.rot).filter
            (fun c => c.1 = 0)).card by rw [h0]; omega)
      obtain ⟨c1, hc1⟩ := Finset.card_pos.mp
        (show 0 < ((pl.piece.shapeUp pl.rot).filter
            (fun c => c.1 = 1)).card by rw [h1]; omega)
      rw [Finset.mem_filter] at hc0 hc1
      have hj0 := hcells c0 (by unfold Placement.shapeUp; exact hc0.1)
      have hj1 := hcells c1 (by unfold Placement.shapeUp; exact hc1.1)
      have hcol : pl.col = j := by omega
      constructor
      · rw [show j = pl.col + 0 by omega, colProfile_eq_fiber pl 0]
        unfold Placement.shapeUp
        exact h0
      · rw [show j + 1 = pl.col + 1 by omega, colProfile_eq_fiber pl 1]
        unfold Placement.shapeUp
        exact h1
    rcases hsplit with ⟨h0, h1⟩ | ⟨h0, h1⟩
    · exact Or.inl (hget 3 1 h0 h1 (by omega) (by omega))
    · exact Or.inr (hget 1 3 h0 h1 (by omega) (by omega))
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- Every value a sequence takes on a window is its initial value or the
value just after a change point. -/
theorem seq_image_subset_changes {f : ℕ → ℕ} {N w : ℕ} :
    (Finset.range w).image (fun k => f (N + k))
      ⊆ insert (f N) (((Finset.range (w - 1)).filter
          (fun k => f (N + k + 1) ≠ f (N + k))).image
            (fun k => f (N + k + 1))) := by
  classical
  intro v hv
  obtain ⟨k₀, hk₀, rfl⟩ := Finset.mem_image.mp hv
  have hne : ((Finset.range w).filter
      (fun k => f (N + k) = f (N + k₀))).Nonempty :=
    ⟨k₀, Finset.mem_filter.mpr ⟨hk₀, rfl⟩⟩
  set k := ((Finset.range w).filter
    (fun k => f (N + k) = f (N + k₀))).min' hne with hkdef
  have hkmem := Finset.min'_mem _ hne
  rw [Finset.mem_filter] at hkmem
  obtain ⟨hkr, hkv⟩ := hkmem
  rw [← hkdef] at hkr hkv
  have hkw := Finset.mem_range.mp hkr
  by_cases hk0 : k = 0
  · rw [Finset.mem_insert]
    left
    rw [← hkv, hk0, Nat.add_zero]
  · rw [Finset.mem_insert]
    right
    have hkpos : 0 < k := Nat.pos_of_ne_zero hk0
    have hprev : f (N + (k - 1)) ≠ f (N + k₀) := by
      intro heq
      have hmem : (k - 1) ∈ (Finset.range w).filter
          (fun k => f (N + k) = f (N + k₀)) :=
        Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), heq⟩
      have hmle := Finset.min'_le _ _ hmem
      rw [← hkdef] at hmle
      omega
    apply Finset.mem_image.mpr
    refine ⟨k - 1, ?_, ?_⟩
    · rw [Finset.mem_filter]
      refine ⟨Finset.mem_range.mpr (by omega), ?_⟩
      rw [show N + (k - 1) + 1 = N + k by omega]
      intro heq
      exact hprev (heq ▸ hkv)
    · rw [show N + (k - 1) + 1 = N + k by omega]
      exact hkv

/-- A sequence shows at most one more distinct value on a window than it
has change points. -/
theorem seq_image_card_le_changes {f : ℕ → ℕ} {N w : ℕ} :
    ((Finset.range w).image (fun k => f (N + k))).card
      ≤ ((Finset.range (w - 1)).filter
          (fun k => f (N + k + 1) ≠ f (N + k))).card + 1 := by
  classical
  have hle := Finset.card_le_card
    (seq_image_subset_changes (f := f) (N := N) (w := w))
  have hins := Finset.card_insert_le (f N)
    (((Finset.range (w - 1)).filter
      (fun k => f (N + k + 1) ≠ f (N + k))).image (fun k => f (N + k + 1)))
  have him := Finset.card_image_le
    (s := (Finset.range (w - 1)).filter
      (fun k => f (N + k + 1) ≠ f (N + k)))
    (f := fun k => f (N + k + 1))
  omega

/-- A column no cell lands in has zero profile. -/
theorem colProfile_eq_zero_of_not_touched {pl : Placement} {c : ℕ}
    (h : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ≠ c) :
    pl.colProfile c = 0 := by
  classical
  unfold Placement.colProfile
  rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro cell hmem
  rw [Finset.mem_filter] at hmem
  exact h cell hmem.1 hmem.2

/-- Deliveries over a window are the sum of the window's profiles. -/
theorem colDelivered_window_sum (π : Policy GameConfig.standard)
    (j n : ℕ) :
    ∀ w, colDelivered π j (n + w)
      = colDelivered π j n
        + ∑ k ∈ Finset.range w,
            (π (trace GameConfig.standard π
              GameState.init (n + k))).colProfile j := by
  intro w
  induction w with
  | zero => simp
  | succ k ih =>
    rw [show n + (k + 1) = (n + k) + 1 by omega, colDelivered_succ,
      Finset.sum_range_succ, ih]
    omega

/-- **Every column eats at the global rate**: across any live window,
each column's deliveries sit within two boardfuls of the exact
0.4-cells-per-move line — `4w − 400 ≤ 10·Δdelivered ≤ 4w + 400`. The
global clearing pinch descends to every single column: no column can be
starved or force-fed for long, anywhere in the game. -/
theorem column_delivery_window_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m w j : ℕ} (hj : j < 10)
    (hlive_m : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard)
    (hlive_mw : ¬ (trace GameConfig.standard π GameState.init (m + w)).lost
      GameConfig.standard) :
    4 * w ≤ 10 * (colDelivered π j (m + w) - colDelivered π j m) + 400
    ∧ 10 * (colDelivered π j (m + w) - colDelivered π j m)
      ≤ 4 * w + 400 := by
  have hled1 := colDelivered_ledger (π := π) hj m
  have hled2 := colDelivered_ledger (π := π) hj (m + w)
  have hband := cleared_window_band hv hlive_m hlive_mw
  have hif1 := (GameState.not_lost_iff_forall_row_lt
    GameConfig.standard _).mp hlive_m
  have hif2 := (GameState.not_lost_iff_forall_row_lt
    GameConfig.standard _).mp hlive_mw
  have hc1 : (trace GameConfig.standard π GameState.init m).board.colCount j
      ≤ 20 := by
    have h1 := colCount_le_colHeight
      (trace GameConfig.standard π GameState.init m).board j
    have h2 := Board.colHeight_le_rows_of_in_field
      (cfg := GameConfig.standard) hif1 j
    rw [GameConfig.standard_rows] at h2
    omega
  have hc2 : (trace GameConfig.standard π
      GameState.init (m + w)).board.colCount j ≤ 20 := by
    have h1 := colCount_le_colHeight
      (trace GameConfig.standard π GameState.init (m + w)).board j
    have h2 := Board.colHeight_le_rows_of_in_field
      (cfg := GameConfig.standard) hif2 j
    rw [GameConfig.standard_rows] at h2
    omega
  have hdm := colDelivered_mono π j (Nat.le_add_right m w)
  have hcm := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right m w)
  exact ⟨by omega, by omega⟩

/-- A survivor feeds every column at the 0.4 rate on every window of its
play. -/
theorem survivor_column_delivery_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init)
    {j : ℕ} (hj : j < 10) (m w : ℕ) :
    4 * w ≤ 10 * (colDelivered π j (m + w) - colDelivered π j m) + 400
    ∧ 10 * (colDelivered π j (m + w) - colDelivered π j m)
      ≤ 4 * w + 400 :=
  column_delivery_window_band hv hj (hsurv m) (hsurv (m + w))

/-- A placement puts at most four cells in any one column. -/
theorem colProfile_le_four (pl : Placement) (c : ℕ) :
    pl.colProfile c ≤ 4 := by
  unfold Placement.colProfile
  calc (pl.shapeUp.filter (fun cell => pl.col + cell.1 = c)).card
      ≤ pl.shapeUp.card := Finset.card_filter_le _ _
    _ = 4 := pl.shapeUp_card

/-- **The per-column feeding-event band**: on any live window, the
number of moves that feed column `j` lies between `w/10 − 10` and
`2w/5 + 40` — deliveries are pinned at 0.4 per move and each feeding
carries one to four cells. Every column is not merely fed on schedule:
it is fed at a pinned *frequency*, everywhere in the game. -/
theorem column_fed_events_band {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m w j : ℕ} (hj : j < 10)
    (hlive_m : ¬ (trace GameConfig.standard π GameState.init m).lost
      GameConfig.standard)
    (hlive_mw : ¬ (trace GameConfig.standard π GameState.init (m + w)).lost
      GameConfig.standard) :
    4 * w ≤ 40 * ((Finset.range w).filter (fun k =>
        0 < (π (trace GameConfig.standard π
          GameState.init (m + k))).colProfile j)).card + 400
    ∧ 10 * ((Finset.range w).filter (fun k =>
        0 < (π (trace GameConfig.standard π
          GameState.init (m + k))).colProfile j)).card ≤ 4 * w + 400 := by
  classical
  have hband := column_delivery_window_band hv hj hlive_m hlive_mw
  have hsum := colDelivered_window_sum π j m w
  have hsplit := Finset.sum_filter_add_sum_filter_not (Finset.range w)
    (fun k => 0 < (π (trace GameConfig.standard π
      GameState.init (m + k))).colProfile j)
    (fun k => (π (trace GameConfig.standard π
      GameState.init (m + k))).colProfile j)
  have hoff : (∑ k ∈ (Finset.range w).filter (fun k =>
      ¬ 0 < (π (trace GameConfig.standard π
        GameState.init (m + k))).colProfile j),
      (π (trace GameConfig.standard π
        GameState.init (m + k))).colProfile j) = 0 := by
    apply Finset.sum_eq_zero
    intro k hk
    rw [Finset.mem_filter] at hk
    omega
  have hup : (∑ k ∈ (Finset.range w).filter (fun k =>
      0 < (π (trace GameConfig.standard π
        GameState.init (m + k))).colProfile j),
      (π (trace GameConfig.standard π
        GameState.init (m + k))).colProfile j)
      ≤ ((Finset.range w).filter (fun k =>
        0 < (π (trace GameConfig.standard π
          GameState.init (m + k))).colProfile j)).card • 4 :=
    Finset.sum_le_card_nsmul _ _ 4
      (fun k _ => colProfile_le_four _ j)
  have hlow : ((Finset.range w).filter (fun k =>
      0 < (π (trace GameConfig.standard π
        GameState.init (m + k))).colProfile j)).card • 1
      ≤ (∑ k ∈ (Finset.range w).filter (fun k =>
        0 < (π (trace GameConfig.standard π
          GameState.init (m + k))).colProfile j),
        (π (trace GameConfig.standard π
          GameState.init (m + k))).colProfile j) :=
    Finset.card_nsmul_le_sum _ _ 1 (fun k hk => by
      rw [Finset.mem_filter] at hk
      omega)
  simp only [smul_eq_mul] at hup hlow
  simp only [] at hsplit hoff hup hlow hsum ⊢
  omega

/-- **Height is cells plus holes, exactly**: every column's height splits
into its filled cells and its buried gaps. -/
theorem colHeight_eq_colCount_add_colHoles (b : Board) (j : ℕ) :
    b.colHeight j = b.colCount j + Board.colHoles b j := by
  unfold Board.colHoles
  have hcc := colRows_card_eq_colCount b j
  have hle := colCount_le_colHeight b j
  omega

/-- **The skyline mass identity**: the sum of the ten column heights is
exactly the board's cell count plus its hole debt — the skyline is mass
plus rot, nothing else. -/
theorem skyline_eq_count_add_holes {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    (∑ j ∈ Finset.range 10, b.colHeight j)
      = b.count + Board.holes GameConfig.standard b := by
  have hsum := Board.sum_colCount (cfg := GameConfig.standard) hwf
  rw [GameConfig.standard_cols] at hsum
  unfold Board.holes
  rw [GameConfig.standard_cols, ← hsum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl
    (fun j _ => colHeight_eq_colCount_add_colHoles b j)

/-- **The skyline sums to at most two hundred**: on any live board the
ten column heights total no more than one boardful — mass plus debt
never exceeds capacity, so neither does the skyline. -/
theorem skyline_sum_le_two_hundred {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hif : ∀ p ∈ b, p.2 < GameConfig.standard.rows) :
    (∑ j ∈ Finset.range 10, b.colHeight j) ≤ 200 := by
  have hid := skyline_eq_count_add_holes hwf
  have hcap := holes_add_count_le_two_hundred hwf hif
  omega

/-- **A light board always offers a window**: whenever mass plus hole
debt is at most eighty-four, some adjacent low pair exists — refusing
every window requires five seventeen-high towers, and five towers cost
eighty-five skyline. The EXISTENCE side of the moving-window crux, under
a weight condition: the driver can always find a window as long as the
board is kept light. -/
theorem low_pair_exists_of_light {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hlight : b.count + Board.holes GameConfig.standard b ≤ 84) :
    ∃ j, j + 1 < 10 ∧ b.colHeight j + 4 ≤ 20
      ∧ b.colHeight (j + 1) + 4 ≤ 20 := by
  classical
  by_contra hnone
  have h : ∀ j, j + 1 < 10 →
      ¬ (b.colHeight j + 4 ≤ 20 ∧ b.colHeight (j + 1) + 4 ≤ 20) := by
    intro j hj hcon
    exact hnone ⟨j, hj, hcon.1, hcon.2⟩
  have h5 := no_low_pair_five_high h
  have hsum1 : (∑ j ∈ (Finset.range 10).filter
        (fun j => 17 ≤ b.colHeight j), b.colHeight j)
      ≤ ∑ j ∈ Finset.range 10, b.colHeight j :=
    Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  have hsum2 : ((Finset.range 10).filter
        (fun j => 17 ≤ b.colHeight j)).card • 17
      ≤ (∑ j ∈ (Finset.range 10).filter
          (fun j => 17 ≤ b.colHeight j), b.colHeight j) :=
    Finset.card_nsmul_le_sum _ _ 17
      (fun j hj => (Finset.mem_filter.mp hj).2)
  have hid := skyline_eq_count_add_holes hwf
  simp only [smul_eq_mul] at hsum2
  omega

/-- **The one-step driver exists on light boards**: whenever mass plus
debt is at most eighty-four, every piece admits a valid placement
confined to some low adjacent pair — the window is there
(`low_pair_exists_of_light`) and every piece has a two-wide rotation to
enter it. The moving-window crux is now PURELY the lightness invariant:
a policy that keeps `count + holes ≤ 84` forever can always make a
capstone move, and capstone play survives. -/
theorem light_board_window_move_exists {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hlight : b.count + Board.holes GameConfig.standard b ≤ 84)
    (p : Piece) :
    ∃ j, j + 1 < 10
      ∧ b.colHeight j + 4 ≤ 20 ∧ b.colHeight (j + 1) + 4 ≤ 20
      ∧ ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard
        ∧ ∀ cell ∈ pl.shapeUp,
            pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1 := by
  obtain ⟨j, hj, h1, h2⟩ := low_pair_exists_of_light hwf hlight
  obtain ⟨r, hr⟩ := exists_narrow_rotation p
  refine ⟨j, hj, h1, h2, ⟨p, r, j⟩, rfl, ?_, ?_⟩
  · intro cell hcell
    have hw := hr cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · intro cell hcell
    have hw := hr cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega

/-- A single-column I rotation is the four-cell tower `{(t,0)…(t,3)}`. -/
theorem I_vertical_shape :
    ∀ r : Rotation,
      (∀ cell ∈ Piece.I.shapeUp r, ∀ cell' ∈ Piece.I.shapeUp r,
        cell.1 = cell'.1) →
      ∃ t < 4, Piece.I.shapeUp r
        = ({(t, 0), (t, 1), (t, 2), (t, 3)} : Finset PieceCell) := by
  decide

/-- **The tower brick costs exactly four**: an I confined to an adjacent
pair lands vertically on one of the two columns and raises that column's
height by exactly four — no more, no less, whatever stood there. The
window's forced I-event (`confined_run_tower_event`) has an exact
price in height as well as in mass. -/
theorem vertical_I_raises_exactly_four {b : Board} {pl : Placement}
    {j : ℕ} (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    ∃ c, (c = j ∨ c = j + 1)
      ∧ (pl.place b).colHeight c = b.colHeight c + 4 := by
  classical
  have hsingle := I_pair_confined_single_column hI hcells
  have hsh : pl.shapeUp = Piece.I.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hI]
  obtain ⟨t, _, ht⟩ := I_vertical_shape pl.rot (by
    intro cell hcell cell' hcell'
    exact hsingle cell (by rw [hsh]; exact hcell) cell'
      (by rw [hsh]; exact hcell'))
  have htsh : pl.shapeUp = ({(t, 0), (t, 1), (t, 2), (t, 3)}
      : Finset PieceCell) := by
    rw [hsh, ht]
  have hmem0 : ((t, 0) : PieceCell) ∈ pl.shapeUp := by
    rw [htsh]
    simp
  have hmem3 : ((t, 3) : PieceCell) ∈ pl.shapeUp := by
    rw [htsh]
    simp
  have hdrop : pl.dropOffset b = b.colHeight (pl.col + t) := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      rw [htsh] at hcell
      simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
      rcases hcell with h | h | h | h <;> subst h <;> simp
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hmem0
      unfold Placement.dropOffset
      simpa using hle
  have htop : ∀ cell' ∈ pl.shapeUp, cell'.1 = ((t, 3) : PieceCell).1 →
      cell'.2 ≤ ((t, 3) : PieceCell).2 := by
    intro cell' hcell' _
    rw [htsh] at hcell'
    simp only [Finset.mem_insert, Finset.mem_singleton] at hcell'
    rcases hcell' with h | h | h | h <;> subst h <;> simp
  have hfed := place_fed_colHeight_eq (b := b) hmem3 htop
  refine ⟨pl.col + t, hcells (t, 0) hmem0, ?_⟩
  rw [show pl.col + ((t, 3) : PieceCell).1 = pl.col + t from rfl] at hfed
  rw [hfed, hdrop]

/-- **A twelve-low window absorbs the tower**: if both window columns
stand at most twelve, the forced vertical I leaves the pair still low —
the fed column ends at sixteen at worst and the other is untouched. -/
theorem tower_event_absorbed_of_low {b : Board} {pl : Placement} {j : ℕ}
    (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (h1 : b.colHeight j ≤ 12) (h2 : b.colHeight (j + 1) ≤ 12) :
    (pl.place b).colHeight j + 4 ≤ 20
      ∧ (pl.place b).colHeight (j + 1) + 4 ≤ 20 := by
  classical
  obtain ⟨c, hc, hraise⟩ := vertical_I_raises_exactly_four (b := b) hI hcells
  have hsingle := I_pair_confined_single_column hI hcells
  have hcell0 : ∃ cell₀ ∈ pl.shapeUp, pl.col + cell₀.1 = c := by
    by_contra hno
    push Not at hno
    have hz := colProfile_eq_zero_of_not_touched (pl := pl) (c := c) hno
    have := place_unfed_colHeight_eq (b := b) (pl := pl) hz
    omega
  obtain ⟨cell₀, hc0m, hc0⟩ := hcell0
  have hunfed : ∀ d, d ≠ c →
      Board.colHeight (pl.place b) d = b.colHeight d := by
    intro d hd
    apply place_unfed_colHeight_eq
    apply colProfile_eq_zero_of_not_touched
    intro cell hcell heq
    have := hsingle cell hcell cell₀ hc0m
    omega
  constructor
  · rcases eq_or_ne j c with h | h
    · rw [h]
      rw [h] at h1
      omega
    · have := hunfed j h
      omega
  · rcases eq_or_ne (j + 1) c with h | h
    · rw [h]
      rw [h] at h2
      omega
    · have := hunfed (j + 1) h
      omega

/-- **A thirteen-high window breaks on the tower**: if both window
columns stand at least thirteen, the forced vertical I ends the window —
whichever column it feeds passes sixteen. The I-ready window is
twelve-low, not sixteen-low: the bag's thirteen-clock effectively
tightens the working height by four rows. -/
theorem tower_event_breaks_high_window {b : Board} {pl : Placement}
    {j : ℕ} (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (h1 : 13 ≤ b.colHeight j) (h2 : 13 ≤ b.colHeight (j + 1)) :
    ¬ ((pl.place b).colHeight j + 4 ≤ 20
      ∧ (pl.place b).colHeight (j + 1) + 4 ≤ 20) := by
  obtain ⟨c, hc, hraise⟩ := vertical_I_raises_exactly_four (b := b) hI hcells
  intro hcon
  rcases hc with h | h
  · rw [h] at hraise
    omega
  · rw [h] at hraise
    omega

/-- Refusing every `h`-low adjacent pair costs five columns above `h`:
the threshold-general tower count. -/
theorem no_h_low_pair_five_high {b : Board} {h : ℕ}
    (hno : ∀ j, j + 1 < 10 →
      ¬ (b.colHeight j ≤ h ∧ b.colHeight (j + 1) ≤ h)) :
    5 ≤ ((Finset.range 10).filter
      (fun j => h + 1 ≤ b.colHeight j)).card := by
  classical
  have hpair : ∀ i, i < 5 →
      h + 1 ≤ b.colHeight (2 * i) ∨ h + 1 ≤ b.colHeight (2 * i + 1) := by
    intro i hi
    have := hno (2 * i) (by omega)
    omega
  have hmem : ∀ i ∈ Finset.range 5,
      (if h + 1 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1)
        ∈ (Finset.range 10).filter (fun j => h + 1 ≤ b.colHeight j) := by
    intro i hi
    rw [Finset.mem_range] at hi
    rw [Finset.mem_filter, Finset.mem_range]
    by_cases hc : h + 1 ≤ b.colHeight (2 * i)
    · rw [if_pos hc]
      exact ⟨by omega, hc⟩
    · have h1 : h + 1 ≤ b.colHeight (2 * i + 1) := by
        rcases hpair i hi with h' | h'
        · exact absurd h' hc
        · exact h'
      rw [if_neg hc]
      exact ⟨by omega, h1⟩
  have hinj : Set.InjOn
      (fun i => if h + 1 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1)
      ↑(Finset.range 5) := by
    intro i hi j hj hij
    simp only [Finset.mem_coe, Finset.mem_range] at hi hj
    simp only [] at hij
    have h1 : (if h + 1 ≤ b.colHeight (2 * i) then 2 * i else 2 * i + 1) / 2
        = i := by
      split_ifs <;> omega
    have h2 : (if h + 1 ≤ b.colHeight (2 * j) then 2 * j else 2 * j + 1) / 2
        = j := by
      split_ifs <;> omega
    rw [hij] at h1
    omega
  calc (5 : ℕ) = (Finset.range 5).card := (Finset.card_range 5).symm
    _ ≤ _ := Finset.card_le_card_of_injOn _ hmem hinj

/-- **A sixty-four-light board offers an I-ready window**: whenever mass
plus debt is at most sixty-four, some adjacent pair stands twelve-low —
low enough to absorb the bag's forced vertical I and remain a window.
Refusing every twelve-low pair costs five thirteen-columns, and five
thirteens are sixty-five skyline. -/
theorem twelve_low_pair_exists_of_light {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hlight : b.count + Board.holes GameConfig.standard b ≤ 64) :
    ∃ j, j + 1 < 10 ∧ b.colHeight j ≤ 12 ∧ b.colHeight (j + 1) ≤ 12 := by
  classical
  by_contra hnone
  have hno : ∀ j, j + 1 < 10 →
      ¬ (b.colHeight j ≤ 12 ∧ b.colHeight (j + 1) ≤ 12) := by
    intro j hj hcon
    exact hnone ⟨j, hj, hcon.1, hcon.2⟩
  have h5 := no_h_low_pair_five_high (h := 12) hno
  simp only [show (12 : ℕ) + 1 = 13 from rfl] at h5
  have hsum1 : (∑ j ∈ (Finset.range 10).filter
        (fun j => 13 ≤ b.colHeight j), b.colHeight j)
      ≤ ∑ j ∈ Finset.range 10, b.colHeight j :=
    Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)
  have hsum2 : ((Finset.range 10).filter
        (fun j => 13 ≤ b.colHeight j)).card • 13
      ≤ (∑ j ∈ (Finset.range 10).filter
          (fun j => 13 ≤ b.colHeight j), b.colHeight j) :=
    Finset.card_nsmul_le_sum _ _ 13
      (fun j hj => (Finset.mem_filter.mp hj).2)
  have hid := skyline_eq_count_add_holes hwf
  simp only [smul_eq_mul] at hsum2
  omega

/-- **The I-ready one-step driver**: on a sixty-four-light board every
piece admits a valid placement confined to some twelve-low pair — a
window that will absorb even the forced vertical I and remain a window
(`tower_event_absorbed_of_low`). The lightness ladder closes: keeping
`count + holes ≤ 64` maintains not just a window but an I-proof one,
move after move. -/
theorem I_ready_window_move_exists {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hlight : b.count + Board.holes GameConfig.standard b ≤ 64)
    (p : Piece) :
    ∃ j, j + 1 < 10 ∧ b.colHeight j ≤ 12 ∧ b.colHeight (j + 1) ≤ 12
      ∧ ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard
        ∧ ∀ cell ∈ pl.shapeUp,
            pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1 := by
  obtain ⟨j, hj, h1, h2⟩ := twelve_low_pair_exists_of_light hwf hlight
  obtain ⟨r, hr⟩ := exists_narrow_rotation p
  refine ⟨j, hj, h1, h2, ⟨p, r, j⟩, rfl, ?_, ?_⟩
  · intro cell hcell
    have hw := hr cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · intro cell hcell
    have hw := hr cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega

/-- **The tower brick is debt-free**: an I confined to an adjacent pair
lands flush on its column's stack and creates no holes anywhere — its
foot sits at offset zero, so the drop seats exactly on the surface. The
window's forced I-event costs four height but zero debt. -/
theorem vertical_I_hole_free {b : Board} {pl : Placement} {j : ℕ}
    (hI : pl.piece = Piece.I)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b := by
  classical
  have hsingle := I_pair_confined_single_column hI hcells
  have hsh : pl.shapeUp = Piece.I.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hI]
  obtain ⟨t, _, ht⟩ := I_vertical_shape pl.rot (by
    intro cell hcell cell' hcell'
    exact hsingle cell (by rw [hsh]; exact hcell) cell'
      (by rw [hsh]; exact hcell'))
  have htsh : pl.shapeUp = ({(t, 0), (t, 1), (t, 2), (t, 3)}
      : Finset PieceCell) := by
    rw [hsh, ht]
  have hmem0 : ((t, 0) : PieceCell) ∈ pl.shapeUp := by
    rw [htsh]
    simp
  have hdrop : pl.dropOffset b = b.colHeight (pl.col + t) := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      rw [htsh] at hcell
      simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
      rcases hcell with h | h | h | h <;> subst h <;> simp
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hmem0
      unfold Placement.dropOffset
      simpa using hle
  apply holes_place_eq_iff.mpr
  intro c hc
  by_cases hfed : c = pl.col + t
  · subst hfed
    have hbot : (pl.col + t, pl.dropOffset b) ∈ pl.dropped b := by
      unfold Placement.dropped Placement.cellsAt
      rw [Finset.mem_image]
      exact ⟨(t, 0), hmem0, by simp⟩
    have hmin : ∀ r, (pl.col + t, r) ∈ pl.dropped b →
        pl.dropOffset b ≤ r := by
      intro r hr
      unfold Placement.dropped Placement.cellsAt at hr
      rw [Finset.mem_image] at hr
      obtain ⟨cell, hcell, heq⟩ := hr
      have hrow := congrArg Prod.snd heq
      simp only [] at hrow
      omega
    have heq := colHoles_place_eq (b := b) hbot hmin
    rw [heq, hdrop]
    omega
  · apply colHoles_place_eq_of_unfed
    apply colProfile_eq_zero_of_not_touched
    intro cell hcell hceq
    apply hfed
    have := hsingle cell hcell (t, 0) hmem0
    omega

/-- The O's feet sit at offset zero in both its columns, every
rotation. -/
theorem O_shape_feet : ∀ r : Rotation,
    ((0 : ℕ), (0 : ℕ)) ∈ Piece.O.shapeUp r
    ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.O.shapeUp r := by
  decide

/-- **The window O's exact hole bill**: an O confined to an adjacent
pair bridges the pair at its taller column's height and buries exactly
the height difference — `Δholes = (max − h_j) + (max − h_{j+1})`, which
is `|h_j − h_{j+1}|`. On a flat window the square is debt-free; on a
staggered one it pays the stagger, exactly. -/
theorem window_O_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hO : pl.piece = Piece.O)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (max (b.colHeight j) (b.colHeight (j + 1)) - b.colHeight j)
        + (max (b.colHeight j) (b.colHeight (j + 1))
            - b.colHeight (j + 1)) := by
  classical
  obtain ⟨hf0, hf1⟩ := O_shape_feet pl.rot
  have hsh : pl.shapeUp = Piece.O.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hO]
  have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf0
  have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf1
  have hj0 := hcells _ hm0
  have hj1 := hcells _ hm1
  have hcol : pl.col = j := by omega
  have hnarrow := (O_shape_columns pl.rot).2.2
  have hdrop : pl.dropOffset b
      = max (b.colHeight j) (b.colHeight (j + 1)) := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hw : cell.1 ≤ 1 := by
        have := hnarrow cell (by rw [← hsh]; exact hcell)
        omega
      rcases hcells cell hcell with h | h
      · rw [h]
        have : b.colHeight j ≤ max (b.colHeight j) (b.colHeight (j + 1)) :=
          le_max_left _ _
        omega
      · rw [h]
        have : b.colHeight (j + 1)
            ≤ max (b.colHeight j) (b.colHeight (j + 1)) := le_max_right _ _
        omega
    · apply max_le
      · have hle := Finset.le_sup
          (f := fun cell : PieceCell =>
            b.colHeight (pl.col + cell.1) - cell.2) hm0
        unfold Placement.dropOffset
        rw [hcol]
        rw [hcol] at hle
        simpa using hle
      · have hle := Finset.le_sup
          (f := fun cell : PieceCell =>
            b.colHeight (pl.col + cell.1) - cell.2) hm1
        unfold Placement.dropOffset
        rw [hcol]
        rw [hcol] at hle
        simpa using hle
  have hbot : ∀ c, c = 0 ∨ c = 1 →
      (j + c, pl.dropOffset b) ∈ pl.dropped b
      ∧ ∀ r, (j + c, r) ∈ pl.dropped b → pl.dropOffset b ≤ r := by
    intro c hc
    constructor
    · unfold Placement.dropped Placement.cellsAt
      rw [Finset.mem_image]
      rcases hc with h | h
      · exact ⟨(0, 0), hm0, by subst h; rw [hcol]; simp⟩
      · exact ⟨(1, 0), hm1, by subst h; rw [hcol]; simp⟩
    · intro r hr
      unfold Placement.dropped Placement.cellsAt at hr
      rw [Finset.mem_image] at hr
      obtain ⟨cell, hcell, heq⟩ := hr
      have hrow := congrArg Prod.snd heq
      simp only [] at hrow
      omega
  obtain ⟨hbot0, hmin0⟩ := hbot 0 (Or.inl rfl)
  obtain ⟨hbot1, hmin1⟩ := hbot 1 (Or.inr rfl)
  rw [show j + 0 = j by omega] at hbot0 hmin0
  have hg0 := colHoles_place_eq (b := b) hbot0 hmin0
  have hg1 := colHoles_place_eq (b := b) hbot1 hmin1
  have hpoint : ∀ c ∈ Finset.range 10,
      Board.colHoles (pl.place b) c
      = Board.colHoles b c
        + (if c = j then pl.dropOffset b - b.colHeight j else 0)
        + (if c = j + 1 then pl.dropOffset b - b.colHeight (j + 1)
            else 0) := by
    intro c _
    by_cases h0 : c = j
    · subst h0
      rw [if_pos rfl, if_neg (by omega)]
      omega
    · by_cases h1 : c = j + 1
      · subst h1
        rw [if_neg h0, if_pos rfl]
        omega
      · rw [if_neg h0, if_neg h1]
        have hz : pl.colProfile c = 0 := by
          apply colProfile_eq_zero_of_not_touched
          intro cell hcell hceq
          rcases hcells cell hcell with h | h
          · exact h0 (by omega)
          · exact h1 (by omega)
        have := colHoles_place_eq_of_unfed (b := b) (pl := pl) hz
        omega
  unfold Board.holes
  rw [GameConfig.standard_cols]
  rw [Finset.sum_congr rfl hpoint]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  rw [Finset.sum_ite_eq' (Finset.range 10) j
      (fun _ => pl.dropOffset b - b.colHeight j),
    Finset.sum_ite_eq' (Finset.range 10) (j + 1)
      (fun _ => pl.dropOffset b - b.colHeight (j + 1))]
  have hjr : j ∈ Finset.range 10 := Finset.mem_range.mpr (by omega)
  have hj1r : j + 1 ∈ Finset.range 10 := Finset.mem_range.mpr (by omega)
  rw [if_pos hjr, if_pos hj1r, hdrop]

/-- **The master two-column hole bill**: for any placement confined to
an adjacent pair with a cell in each column, whose per-column feet sit
at offsets `f₀` and `f₁`, the holes created are exactly the two landing
gaps `(dropOffset + f₀ − h_j) + (dropOffset + f₁ − h_{j+1})`. Every
window piece's debt price is an instance. -/
theorem window_two_col_hole_bill {b : Board} {pl : Placement}
    {j f₀ f₁ : ℕ} (hj : j + 1 < 10)
    (hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1)
    (hcol : pl.col = j)
    (hm0 : ((0 : ℕ), f₀) ∈ pl.shapeUp)
    (hmin0 : ∀ cell ∈ pl.shapeUp, cell.1 = 0 → f₀ ≤ cell.2)
    (hm1 : ((1 : ℕ), f₁) ∈ pl.shapeUp)
    (hmin1 : ∀ cell ∈ pl.shapeUp, cell.1 = 1 → f₁ ≤ cell.2) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b + f₀ - b.colHeight j)
        + (pl.dropOffset b + f₁ - b.colHeight (j + 1)) := by
  classical
  have hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    rw [hcol]
    omega
  have hbot0 : (j, pl.dropOffset b + f₀) ∈ pl.dropped b := by
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.mem_image]
    exact ⟨(0, f₀), hm0, by rw [hcol]; simp⟩
  have hminb0 : ∀ r, (j, r) ∈ pl.dropped b →
      pl.dropOffset b + f₀ ≤ r := by
    intro r hr
    unfold Placement.dropped Placement.cellsAt at hr
    rw [Finset.mem_image] at hr
    obtain ⟨cell, hcell, heq⟩ := hr
    have hrow := congrArg Prod.snd heq
    have hcc := congrArg Prod.fst heq
    simp only [] at hrow hcc
    have hc0 : cell.1 = 0 := by omega
    have := hmin0 cell hcell hc0
    omega
  have hbot1 : (j + 1, pl.dropOffset b + f₁) ∈ pl.dropped b := by
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.mem_image]
    exact ⟨(1, f₁), hm1, by rw [hcol]⟩
  have hminb1 : ∀ r, (j + 1, r) ∈ pl.dropped b →
      pl.dropOffset b + f₁ ≤ r := by
    intro r hr
    unfold Placement.dropped Placement.cellsAt at hr
    rw [Finset.mem_image] at hr
    obtain ⟨cell, hcell, heq⟩ := hr
    have hrow := congrArg Prod.snd heq
    have hcc := congrArg Prod.fst heq
    simp only [] at hrow hcc
    have hc1 : cell.1 = 1 := by
      have := hn cell hcell
      omega
    have := hmin1 cell hcell hc1
    omega
  have hg0 := colHoles_place_eq (b := b) hbot0 hminb0
  have hg1 := colHoles_place_eq (b := b) hbot1 hminb1
  have hpoint : ∀ c ∈ Finset.range 10,
      Board.colHoles (pl.place b) c
      = Board.colHoles b c
        + (if c = j then pl.dropOffset b + f₀ - b.colHeight j else 0)
        + (if c = j + 1 then pl.dropOffset b + f₁ - b.colHeight (j + 1)
            else 0) := by
    intro c _
    by_cases h0 : c = j
    · subst h0
      rw [if_pos rfl, if_neg (by omega)]
      omega
    · by_cases h1 : c = j + 1
      · subst h1
        rw [if_neg h0, if_pos rfl]
        omega
      · rw [if_neg h0, if_neg h1]
        have hz : pl.colProfile c = 0 := by
          apply colProfile_eq_zero_of_not_touched
          intro cell hcell hceq
          rcases hcells cell hcell with h | h
          · exact h0 (by omega)
          · exact h1 (by omega)
        have := colHoles_place_eq_of_unfed (b := b) (pl := pl) hz
        omega
  unfold Board.holes
  rw [GameConfig.standard_cols]
  rw [Finset.sum_congr rfl hpoint]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  rw [Finset.sum_ite_eq' (Finset.range 10) j
      (fun _ => pl.dropOffset b + f₀ - b.colHeight j),
    Finset.sum_ite_eq' (Finset.range 10) (j + 1)
      (fun _ => pl.dropOffset b + f₁ - b.colHeight (j + 1))]
  rw [if_pos (Finset.mem_range.mpr (by omega : j < 10)),
    if_pos (Finset.mem_range.mpr (by omega : j + 1 < 10))]

/-- A narrow S stands with its right foot down: fiber-0 foot at offset
1, fiber-1 foot at offset 0. -/
theorem S_narrow_feet : ∀ r : Rotation,
    (∀ cell ∈ Piece.S.shapeUp r, cell.1 ≤ 1) →
    (((0 : ℕ), (1 : ℕ)) ∈ Piece.S.shapeUp r
      ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.S.shapeUp r
      ∧ ∀ cell ∈ Piece.S.shapeUp r, cell.1 = 0 → 1 ≤ cell.2) := by
  decide

/-- A narrow Z stands with its left foot down: fiber-0 foot at offset 0,
fiber-1 foot at offset 1. -/
theorem Z_narrow_feet : ∀ r : Rotation,
    (∀ cell ∈ Piece.Z.shapeUp r, cell.1 ≤ 1) →
    (((0 : ℕ), (0 : ℕ)) ∈ Piece.Z.shapeUp r
      ∧ ((1 : ℕ), (1 : ℕ)) ∈ Piece.Z.shapeUp r
      ∧ ∀ cell ∈ Piece.Z.shapeUp r, cell.1 = 1 → 1 ≤ cell.2) := by
  decide

/-- **The window S's exact hole bill**: a pair-confined S pays
`(D + 1 − h_j) + (D − h_{j+1})` where `D` is its drop offset — free
exactly when the pair steps down by one under it. -/
theorem window_S_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hS : pl.piece = Piece.S)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b + 1 - b.colHeight j)
        + (pl.dropOffset b - b.colHeight (j + 1)) := by
  have hsh : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hS]
  rcases SZ_shape_window_split pl.piece (Or.inl hS) pl.rot with
    ⟨h0, h1, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
      intro cell hcell
      apply hnarrow
      exact hcell
    obtain ⟨hf0, hf1, hfmin⟩ := S_narrow_feet pl.rot (by
      intro cell hcell
      apply hn
      rw [hsh]
      exact hcell)
    have hm0 : ((0 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
      rw [hsh]
      exact hf0
    have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
      rw [hsh]
      exact hf1
    have hj0 := hcells _ hm0
    have hj1 := hcells _ hm1
    have hcol : pl.col = j := by
      have e0 : pl.col + ((0 : ℕ), (1 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      omega
    have hbill := window_two_col_hole_bill (b := b) (f₀ := 1) (f₁ := 0)
      hj hn hcol hm0
      (fun cell hcell hc0 => hfmin cell (by rw [← hsh]; exact hcell) hc0)
      hm1 (fun cell _ _ => Nat.zero_le _)
    simpa using hbill
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- **The window Z's exact hole bill**: mirror of the S — free exactly
when the pair steps up by one under it. -/
theorem window_Z_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hZ : pl.piece = Piece.Z)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b - b.colHeight j)
        + (pl.dropOffset b + 1 - b.colHeight (j + 1)) := by
  have hsh : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hZ]
  rcases SZ_shape_window_split pl.piece (Or.inr hZ) pl.rot with
    ⟨h0, h1, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
      intro cell hcell
      apply hnarrow
      exact hcell
    obtain ⟨hf0, hf1, hfmin⟩ := Z_narrow_feet pl.rot (by
      intro cell hcell
      apply hn
      rw [hsh]
      exact hcell)
    have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
      rw [hsh]
      exact hf0
    have hm1 : ((1 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
      rw [hsh]
      exact hf1
    have hj0 := hcells _ hm0
    have hj1 := hcells _ hm1
    have hcol : pl.col = j := by
      have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (1 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      omega
    have hbill := window_two_col_hole_bill (b := b) (f₀ := 0) (f₁ := 1)
      hj hn hcol hm0 (fun cell _ _ => Nat.zero_le _) hm1
      (fun cell hcell hc1 => hfmin cell (by rw [← hsh]; exact hcell) hc1)
    simpa using hbill
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- A narrow L stands arm-down (feet 0, 0) or arm-up (feet 2, 0). -/
theorem L_narrow_feet : ∀ r : Rotation,
    (∀ cell ∈ Piece.L.shapeUp r, cell.1 ≤ 1) →
    ((((0 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp r
        ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp r)
      ∨ (((0 : ℕ), (2 : ℕ)) ∈ Piece.L.shapeUp r
        ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp r
        ∧ ∀ cell ∈ Piece.L.shapeUp r, cell.1 = 0 → 2 ≤ cell.2)) := by
  decide

/-- A narrow J stands arm-up (feet 0, 2) or arm-down (feet 0, 0). -/
theorem J_narrow_feet : ∀ r : Rotation,
    (∀ cell ∈ Piece.J.shapeUp r, cell.1 ≤ 1) →
    ((((0 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp r
        ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.J.shapeUp r
        ∧ ∀ cell ∈ Piece.J.shapeUp r, cell.1 = 1 → 2 ≤ cell.2)
      ∨ (((0 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp r
        ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp r)) := by
  decide

/-- A narrow T points left (feet 1, 0) or right (feet 0, 1). -/
theorem T_narrow_feet : ∀ r : Rotation,
    (∀ cell ∈ Piece.T.shapeUp r, cell.1 ≤ 1) →
    ((((0 : ℕ), (1 : ℕ)) ∈ Piece.T.shapeUp r
        ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.T.shapeUp r
        ∧ ∀ cell ∈ Piece.T.shapeUp r, cell.1 = 0 → 1 ≤ cell.2)
      ∨ (((0 : ℕ), (0 : ℕ)) ∈ Piece.T.shapeUp r
        ∧ ((1 : ℕ), (1 : ℕ)) ∈ Piece.T.shapeUp r
        ∧ ∀ cell ∈ Piece.T.shapeUp r, cell.1 = 1 → 1 ≤ cell.2)) := by
  decide

/-- **The window L's exact hole bill**: arm-down it seats like a flat
pair (`(D − h_j) + (D − h_{j+1})`); arm-up it pays a two-deep overhang
on the left (`(D + 2 − h_j) + (D − h_{j+1})`). -/
theorem window_L_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hL : pl.piece = Piece.L)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b - b.colHeight j)
        + (pl.dropOffset b - b.colHeight (j + 1)))
    ∨ (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b + 2 - b.colHeight j)
        + (pl.dropOffset b - b.colHeight (j + 1))) := by
  have hsh : pl.shapeUp = Piece.L.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hL]
  rcases LJT_shape_window_split pl.piece (Or.inl hL) pl.rot with
    ⟨_, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
      intro cell hcell
      apply hnarrow
      exact hcell
    rcases L_narrow_feet pl.rot (by
        intro cell hcell
        apply hn
        rw [hsh]
        exact hcell) with ⟨hf0, hf1⟩ | ⟨hf0, hf1, hfmin⟩
    · left
      have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 0) (f₁ := 0)
        hj hn hcol hm0 (fun cell _ _ => Nat.zero_le _) hm1
        (fun cell _ _ => Nat.zero_le _)
      simpa using hbill
    · right
      have hm0 : ((0 : ℕ), (2 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (2 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 2) (f₁ := 0)
        hj hn hcol hm0
        (fun cell hcell hc0 => hfmin cell (by rw [← hsh]; exact hcell) hc0)
        hm1 (fun cell _ _ => Nat.zero_le _)
      simpa using hbill
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- **The window J's exact hole bill**: mirror of the L — arm-up pays a
two-deep overhang on the right. -/
theorem window_J_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hJ : pl.piece = Piece.J)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b - b.colHeight j)
        + (pl.dropOffset b + 2 - b.colHeight (j + 1)))
    ∨ (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b - b.colHeight j)
        + (pl.dropOffset b - b.colHeight (j + 1))) := by
  have hsh : pl.shapeUp = Piece.J.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hJ]
  rcases LJT_shape_window_split pl.piece (Or.inr (Or.inl hJ)) pl.rot with
    ⟨_, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
      intro cell hcell
      apply hnarrow
      exact hcell
    rcases J_narrow_feet pl.rot (by
        intro cell hcell
        apply hn
        rw [hsh]
        exact hcell) with ⟨hf0, hf1, hfmin⟩ | ⟨hf0, hf1⟩
    · left
      have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (2 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (2 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 0) (f₁ := 2)
        hj hn hcol hm0 (fun cell _ _ => Nat.zero_le _) hm1
        (fun cell hcell hc1 => hfmin cell (by rw [← hsh]; exact hcell) hc1)
      simpa using hbill
    · right
      have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 0) (f₁ := 0)
        hj hn hcol hm0 (fun cell _ _ => Nat.zero_le _) hm1
        (fun cell _ _ => Nat.zero_le _)
      simpa using hbill
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- **The window T's exact hole bill**: pointing left it pays a one-deep
notch on the left, pointing right on the right — never more than one. -/
theorem window_T_hole_bill {b : Board} {pl : Placement} {j : ℕ}
    (hj : j + 1 < 10) (hT : pl.piece = Piece.T)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1) :
    (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b + 1 - b.colHeight j)
        + (pl.dropOffset b - b.colHeight (j + 1)))
    ∨ (Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
        + (pl.dropOffset b - b.colHeight j)
        + (pl.dropOffset b + 1 - b.colHeight (j + 1))) := by
  have hsh : pl.shapeUp = Piece.T.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hT]
  rcases LJT_shape_window_split pl.piece (Or.inr (Or.inr hT)) pl.rot with
    ⟨_, hnarrow⟩ | ⟨cw, hcw, hc2⟩
  · have hn : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
      intro cell hcell
      apply hnarrow
      exact hcell
    rcases T_narrow_feet pl.rot (by
        intro cell hcell
        apply hn
        rw [hsh]
        exact hcell) with ⟨hf0, hf1, hfmin⟩ | ⟨hf0, hf1, hfmin⟩
    · left
      have hm0 : ((0 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (1 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 1) (f₁ := 0)
        hj hn hcol hm0
        (fun cell hcell hc0 => hfmin cell (by rw [← hsh]; exact hcell) hc0)
        hm1 (fun cell _ _ => Nat.zero_le _)
      simpa using hbill
    · right
      have hm0 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf0
      have hm1 : ((1 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]; exact hf1
      have hj0 := hcells _ hm0
      have hj1 := hcells _ hm1
      have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
      have e1 : pl.col + ((1 : ℕ), (1 : ℕ)).1 = pl.col + 1 := by simp
      rw [e0] at hj0
      rw [e1] at hj1
      have hcol : pl.col = j := by omega
      have hbill := window_two_col_hole_bill (b := b) (f₀ := 0) (f₁ := 1)
        hj hn hcol hm0 (fun cell _ _ => Nat.zero_le _) hm1
        (fun cell hcell hc1 => hfmin cell (by rw [← hsh]; exact hcell) hc1)
      simpa using hbill
  · exfalso
    obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
    have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
    have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
    omega

/-- On a flat pair every confined drop's offset is at most the pair's
height. -/
theorem confined_dropOffset_le_of_flat {b : Board} {pl : Placement}
    {j h : ℕ}
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (hflat0 : b.colHeight j = h) (hflat1 : b.colHeight (j + 1) = h) :
    pl.dropOffset b ≤ h := by
  unfold Placement.dropOffset
  apply Finset.sup_le
  intro cell hcell
  rcases hcells cell hcell with hc | hc <;> rw [hc]
  · omega
  · omega

/-- **A flat window never pays more than two**: whatever piece arrives,
any placement confined to a flat pair creates at most two holes — zero
for I, O and seated L/J, one for S, Z and T, two only for an
arm-up L/J. The flat window is a universal cheap landing zone. -/
theorem flat_window_bill_le_two {b : Board} {pl : Placement} {j h : ℕ}
    (hj : j + 1 < 10)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (hflat0 : b.colHeight j = h) (hflat1 : b.colHeight (j + 1) = h) :
    Board.holes GameConfig.standard (pl.place b)
      ≤ Board.holes GameConfig.standard b + 2 := by
  have hD := confined_dropOffset_le_of_flat hcells hflat0 hflat1
  cases hp : pl.piece with
  | I =>
    have := vertical_I_hole_free (b := b) hp hcells
    omega
  | O =>
    have := window_O_hole_bill (b := b) hj hp hcells
    rw [hflat0, hflat1] at this
    simp only [max_self] at this
    omega
  | S =>
    have := window_S_hole_bill (b := b) hj hp hcells
    rw [hflat0, hflat1] at this
    omega
  | Z =>
    have := window_Z_hole_bill (b := b) hj hp hcells
    rw [hflat0, hflat1] at this
    omega
  | T =>
    rcases window_T_hole_bill (b := b) hj hp hcells with hbill | hbill <;>
      rw [hflat0, hflat1] at hbill <;> omega
  | L =>
    rcases window_L_hole_bill (b := b) hj hp hcells with hbill | hbill <;>
      rw [hflat0, hflat1] at hbill <;> omega
  | J =>
    rcases window_J_hole_bill (b := b) hj hp hcells with hbill | hbill <;>
      rw [hflat0, hflat1] at hbill <;> omega

/-- The cheap standing rotation of each piece: narrow, and for L and J
seated with both feet down. -/
theorem cheap_rotations :
    (∀ cell ∈ Piece.I.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ (∀ cell ∈ Piece.O.shapeUp (0 : Rotation), cell.1 ≤ 1)
    ∧ (∀ cell ∈ Piece.S.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ (∀ cell ∈ Piece.Z.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ (∀ cell ∈ Piece.T.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ ((∀ cell ∈ Piece.L.shapeUp (1 : Rotation), cell.1 ≤ 1)
      ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp (1 : Rotation)
      ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp (1 : Rotation))
    ∧ ((∀ cell ∈ Piece.J.shapeUp (3 : Rotation), cell.1 ≤ 1)
      ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp (3 : Rotation)
      ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp (3 : Rotation)) := by
  decide

/-- **The flat window's best-response bill is one**: on a flat pair,
every piece admits a valid confined placement creating at most ONE hole
— I, O and seated L/J land free; S, Z and T pay their unavoidable
single notch. Combined with the clearing machinery this is the
lightness ledger's income statement: flat-window play accrues debt at
most one hole per move, and only on three pieces in seven. -/
theorem flat_window_cheap_move_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (hflat0 : b.colHeight j = h) (hflat1 : b.colHeight (j + 1) = h)
    (p : Piece) :
    ∃ pl : Placement, pl.piece = p ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          ≤ Board.holes GameConfig.standard b + 1 := by
  obtain ⟨hI, hO, hS, hZ, hT, ⟨hL, hL0, hL1⟩, ⟨hJ, hJ0, hJ1⟩⟩ :=
    cheap_rotations
  cases p with
  | I =>
    refine ⟨⟨Piece.I, 1, j⟩, rfl, ?_, ?_, ?_⟩
    · intro cell hcell
      have := hI cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · intro cell hcell
      have := hI cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    · have hfree := vertical_I_hole_free (b := b)
        (pl := ⟨Piece.I, 1, j⟩) rfl (by
          intro cell hcell
          have := hI cell hcell
          change j + cell.1 = j ∨ j + cell.1 = j + 1
          omega)
      omega
  | O =>
    have hcells : ∀ cell ∈ (⟨Piece.O, 0, j⟩ : Placement).shapeUp,
        (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hO cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.O, 0, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hO cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hbill := window_O_hole_bill (b := b) hj rfl hcells
      rw [hflat0, hflat1] at hbill
      simp only [max_self] at hbill
      omega
  | S =>
    have hcells : ∀ cell ∈ (⟨Piece.S, 1, j⟩ : Placement).shapeUp,
        (⟨Piece.S, 1, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.S, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hS cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.S, 1, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hS cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hbill := window_S_hole_bill (b := b) hj rfl hcells
      have hD := confined_dropOffset_le_of_flat (b := b) hcells hflat0 hflat1
      rw [hflat0, hflat1] at hbill
      omega
  | Z =>
    have hcells : ∀ cell ∈ (⟨Piece.Z, 1, j⟩ : Placement).shapeUp,
        (⟨Piece.Z, 1, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.Z, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hZ cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.Z, 1, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hZ cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hbill := window_Z_hole_bill (b := b) hj rfl hcells
      have hD := confined_dropOffset_le_of_flat (b := b) hcells hflat0 hflat1
      rw [hflat0, hflat1] at hbill
      omega
  | T =>
    have hcells : ∀ cell ∈ (⟨Piece.T, 1, j⟩ : Placement).shapeUp,
        (⟨Piece.T, 1, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.T, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hT cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.T, 1, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hT cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hD := confined_dropOffset_le_of_flat (b := b) hcells hflat0 hflat1
      rcases window_T_hole_bill (b := b) hj rfl hcells with hbill | hbill <;>
        rw [hflat0, hflat1] at hbill <;> omega
  | L =>
    have hn : ∀ cell ∈ (⟨Piece.L, 1, j⟩ : Placement).shapeUp,
        cell.1 ≤ 1 := hL
    have hcells : ∀ cell ∈ (⟨Piece.L, 1, j⟩ : Placement).shapeUp,
        (⟨Piece.L, 1, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.L, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hn cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.L, 1, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hn cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hD := confined_dropOffset_le_of_flat (b := b) hcells hflat0 hflat1
      have hbill := window_two_col_hole_bill (b := b)
        (pl := ⟨Piece.L, 1, j⟩) (f₀ := 0) (f₁ := 0) hj hn rfl hL0
        (fun cell _ _ => Nat.zero_le _) hL1
        (fun cell _ _ => Nat.zero_le _)
      rw [hflat0, hflat1] at hbill
      omega
  | J =>
    have hn : ∀ cell ∈ (⟨Piece.J, 3, j⟩ : Placement).shapeUp,
        cell.1 ≤ 1 := hJ
    have hcells : ∀ cell ∈ (⟨Piece.J, 3, j⟩ : Placement).shapeUp,
        (⟨Piece.J, 3, j⟩ : Placement).col + cell.1 = j
        ∨ (⟨Piece.J, 3, j⟩ : Placement).col + cell.1 = j + 1 := by
      intro cell hcell
      have := hn cell hcell
      change j + cell.1 = j ∨ j + cell.1 = j + 1
      omega
    refine ⟨⟨Piece.J, 3, j⟩, rfl, ?_, hcells, ?_⟩
    · intro cell hcell
      have := hn cell hcell
      change j + cell.1 < GameConfig.standard.cols
      rw [GameConfig.standard_cols]
      omega
    · have hD := confined_dropOffset_le_of_flat (b := b) hcells hflat0 hflat1
      have hbill := window_two_col_hole_bill (b := b)
        (pl := ⟨Piece.J, 3, j⟩) (f₀ := 0) (f₁ := 0) hj hn rfl hJ0
        (fun cell _ _ => Nat.zero_le _) hJ1
        (fun cell _ _ => Nat.zero_le _)
      rw [hflat0, hflat1] at hbill
      omega

/-- The O's tops sit at offset one in both its columns, every
rotation. -/
theorem O_shape_tops : ∀ r : Rotation,
    ((0 : ℕ), (1 : ℕ)) ∈ Piece.O.shapeUp r
    ∧ ((1 : ℕ), (1 : ℕ)) ∈ Piece.O.shapeUp r
    ∧ ∀ cell ∈ Piece.O.shapeUp r, cell.2 ≤ 1 := by
  decide

/-- **The square preserves flatness**: an O confined to a flat pair
leaves the pair flat, two rows higher — the one window piece whose
cheap move needs no repair at all. Flat windows are a fixed point of
O-play. -/
theorem window_O_keeps_flat {b : Board} {pl : Placement} {j h : ℕ}
    (hO : pl.piece = Piece.O)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (hflat0 : b.colHeight j = h) (hflat1 : b.colHeight (j + 1) = h) :
    (pl.place b).colHeight j = h + 2
      ∧ (pl.place b).colHeight (j + 1) = h + 2 := by
  classical
  obtain ⟨ht0, ht1, htop⟩ := O_shape_tops pl.rot
  obtain ⟨hf0, hf1, _⟩ := O_shape_columns pl.rot
  have hsh : pl.shapeUp = Piece.O.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hO]
  obtain ⟨c0, hc0⟩ := Finset.card_pos.mp
    (show 0 < ((Piece.O.shapeUp pl.rot).filter (fun c => c.1 = 0)).card by
      rw [hf0]
      omega)
  obtain ⟨c1, hc1⟩ := Finset.card_pos.mp
    (show 0 < ((Piece.O.shapeUp pl.rot).filter (fun c => c.1 = 1)).card by
      rw [hf1]
      omega)
  rw [Finset.mem_filter] at hc0 hc1
  have hj0 := hcells c0 (by rw [hsh]; exact hc0.1)
  have hj1 := hcells c1 (by rw [hsh]; exact hc1.1)
  have hcol : pl.col = j := by omega
  have hD : pl.dropOffset b = h := by
    apply Nat.le_antisymm
    · exact confined_dropOffset_le_of_flat hcells hflat0 hflat1
    · have hm00 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
        rw [hsh]
        exact (O_shape_feet pl.rot).1
      have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hm00
      unfold Placement.dropOffset
      rw [hcol] at hle
      rw [hcol]
      simpa [hflat0] using hle
  have hm01 : ((0 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht0
  have hm11 : ((1 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht1
  have htop0 : ∀ cell' ∈ pl.shapeUp,
      cell'.1 = ((0 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (1 : ℕ)).2 := by
    intro cell' hcell' _
    exact htop cell' (by rw [← hsh]; exact hcell')
  have htop1 : ∀ cell' ∈ pl.shapeUp,
      cell'.1 = ((1 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (1 : ℕ)).2 := by
    intro cell' hcell' _
    exact htop cell' (by rw [← hsh]; exact hcell')
  have hfed0 := place_fed_colHeight_eq (b := b) hm01 htop0
  have hfed1 := place_fed_colHeight_eq (b := b) hm11 htop1
  rw [hcol] at hfed0 hfed1
  constructor
  · rw [show j = j + ((0 : ℕ), (1 : ℕ)).1 by simp] at hflat0 ⊢
    rw [hfed0, hD]
  · rw [show j + 1 = j + ((1 : ℕ), (1 : ℕ)).1 from rfl] at hflat1 ⊢
    rw [hfed1, hD]

/-- A narrow S's tops: offset 2 in its left column, 1 in its right. -/
theorem S_narrow_tops : ∀ r : Rotation,
    (∀ cell ∈ Piece.S.shapeUp r, cell.1 ≤ 1) →
    (((0 : ℕ), (2 : ℕ)) ∈ Piece.S.shapeUp r
      ∧ ((1 : ℕ), (1 : ℕ)) ∈ Piece.S.shapeUp r
      ∧ ∀ cell ∈ Piece.S.shapeUp r,
          (cell.1 = 0 → cell.2 ≤ 2) ∧ (cell.1 = 1 → cell.2 ≤ 1)) := by
  decide

/-- **The S-staircase**: on a pair stepping down by one, a confined S
lands free of debt AND reproduces the one-step-down profile two rows
higher. After a single setup hole on flat ground, S-chains are
debt-free forever — the staircase is a self-sustaining shape. -/
theorem window_S_staircase {b : Board} {pl : Placement} {j h : ℕ}
    (hj : j + 1 < 10) (hS : pl.piece = Piece.S)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (h0 : b.colHeight j = h + 1) (h1 : b.colHeight (j + 1) = h) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
    ∧ (pl.place b).colHeight j = h + 3
    ∧ (pl.place b).colHeight (j + 1) = h + 2 := by
  classical
  have hsh : pl.shapeUp = Piece.S.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hS]
  have hnarrow : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
    rcases SZ_shape_window_split pl.piece (Or.inl hS) pl.rot with
      ⟨_, _, hn⟩ | ⟨cw, hcw, hc2⟩
    · intro cell hcell
      apply hn
      exact hcell
    · exfalso
      obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
      have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
      have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
      omega
  obtain ⟨hf0, hf1, hfmin⟩ := S_narrow_feet pl.rot (by
    intro cell hcell
    apply hnarrow
    rw [hsh]
    exact hcell)
  obtain ⟨ht0, ht1, htops⟩ := S_narrow_tops pl.rot (by
    intro cell hcell
    apply hnarrow
    rw [hsh]
    exact hcell)
  have hm10 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf1
  have hmt0 : ((0 : ℕ), (2 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht0
  have hmt1 : ((1 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht1
  have hj0 := hcells _ hmt0
  have hj1 := hcells _ hm10
  have e0 : pl.col + ((0 : ℕ), (2 : ℕ)).1 = pl.col := by simp
  have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
  rw [e0] at hj0
  rw [e1] at hj1
  have hcol : pl.col = j := by omega
  have hD : pl.dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hnarrow cell hcell
      have hfm := hfmin cell (by rw [← hsh]; exact hcell)
      rcases hcells cell hcell with hc | hc <;> rw [hc]
      · have hc0 : cell.1 = 0 := by omega
        have := hfm hc0
        omega
      · omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hm10
      unfold Placement.dropOffset
      rw [hcol]
      rw [hcol] at hle
      simp only [] at hle
      rw [show j + ((1 : ℕ), (0 : ℕ)).1 = j + 1 from rfl, h1] at hle
      simpa using hle
  refine ⟨?_, ?_, ?_⟩
  · have hbill := window_S_hole_bill (b := b) hj hS hcells
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ pl.shapeUp,
        cell'.1 = ((0 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (htops cell' (by rw [← hsh]; exact hcell')).1 hc
    have hfed := place_fed_colHeight_eq (b := b) hmt0 htop0
    rw [hcol] at hfed
    rw [show j + ((0 : ℕ), (2 : ℕ)).1 = j from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ pl.shapeUp,
        cell'.1 = ((1 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (1 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (htops cell' (by rw [← hsh]; exact hcell')).2 hc
    have hfed := place_fed_colHeight_eq (b := b) hmt1 htop1
    rw [hcol] at hfed
    rw [show j + ((1 : ℕ), (1 : ℕ)).1 = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- A narrow Z's tops: offset 1 in its left column, 2 in its right. -/
theorem Z_narrow_tops : ∀ r : Rotation,
    (∀ cell ∈ Piece.Z.shapeUp r, cell.1 ≤ 1) →
    (((0 : ℕ), (1 : ℕ)) ∈ Piece.Z.shapeUp r
      ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.Z.shapeUp r
      ∧ ∀ cell ∈ Piece.Z.shapeUp r,
          (cell.1 = 0 → cell.2 ≤ 1) ∧ (cell.1 = 1 → cell.2 ≤ 2)) := by
  decide

/-- **The Z-staircase**: mirror of the S — on a pair stepping up by one,
a confined Z lands free of debt and reproduces the one-step-up profile
two rows higher. -/
theorem window_Z_staircase {b : Board} {pl : Placement} {j h : ℕ}
    (hj : j + 1 < 10) (hZ : pl.piece = Piece.Z)
    (hcells : ∀ cell ∈ pl.shapeUp,
      pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h + 1) :
    Board.holes GameConfig.standard (pl.place b)
      = Board.holes GameConfig.standard b
    ∧ (pl.place b).colHeight j = h + 2
    ∧ (pl.place b).colHeight (j + 1) = h + 3 := by
  classical
  have hsh : pl.shapeUp = Piece.Z.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hZ]
  have hnarrow : ∀ cell ∈ pl.shapeUp, cell.1 ≤ 1 := by
    rcases SZ_shape_window_split pl.piece (Or.inr hZ) pl.rot with
      ⟨_, _, hn⟩ | ⟨cw, hcw, hc2⟩
    · intro cell hcell
      apply hn
      exact hcell
    · exfalso
      obtain ⟨cz, hczmem, hcz0⟩ := Piece.shapeUp_zero_mem pl.piece pl.rot
      have hA := hcells cz (by unfold Placement.shapeUp; exact hczmem)
      have hB := hcells cw (by unfold Placement.shapeUp; exact hcw)
      omega
  obtain ⟨hf0, hf1, hfmin⟩ := Z_narrow_feet pl.rot (by
    intro cell hcell
    apply hnarrow
    rw [hsh]
    exact hcell)
  obtain ⟨ht0, ht1, htops⟩ := Z_narrow_tops pl.rot (by
    intro cell hcell
    apply hnarrow
    rw [hsh]
    exact hcell)
  have hm00 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf0
  have hmt0 : ((0 : ℕ), (1 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht0
  have hmt1 : ((1 : ℕ), (2 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact ht1
  have hj0 := hcells _ hm00
  have hj1 := hcells _ hmt1
  have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
  have e1 : pl.col + ((1 : ℕ), (2 : ℕ)).1 = pl.col + 1 := by simp
  rw [e0] at hj0
  rw [e1] at hj1
  have hcol : pl.col = j := by omega
  have hD : pl.dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hnarrow cell hcell
      have hfm := hfmin cell (by rw [← hsh]; exact hcell)
      rcases hcells cell hcell with hc | hc <;> rw [hc]
      · omega
      · have hc1 : cell.1 = 1 := by omega
        have := hfm hc1
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hm00
      unfold Placement.dropOffset
      rw [hcol]
      rw [hcol] at hle
      simp only [] at hle
      rw [show j + ((0 : ℕ), (0 : ℕ)).1 = j from by simp, h0] at hle
      simpa using hle
  refine ⟨?_, ?_, ?_⟩
  · have hbill := window_Z_hole_bill (b := b) hj hZ hcells
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ pl.shapeUp,
        cell'.1 = ((0 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (1 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (htops cell' (by rw [← hsh]; exact hcell')).1 hc
    have hfed := place_fed_colHeight_eq (b := b) hmt0 htop0
    rw [hcol] at hfed
    rw [show j + ((0 : ℕ), (1 : ℕ)).1 = j from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ pl.shapeUp,
        cell'.1 = ((1 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (htops cell' (by rw [← hsh]; exact hcell')).2 hc
    have hfed := place_fed_colHeight_eq (b := b) hmt1 htop1
    rw [hcol] at hfed
    rw [show j + ((1 : ℕ), (2 : ℕ)).1 = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- The left-pointing T (rotation 1): narrow, foot/top structure. -/
theorem T_r1_shape :
    (∀ cell ∈ Piece.T.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (1 : ℕ)) ∈ Piece.T.shapeUp (1 : Rotation)
    ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.T.shapeUp (1 : Rotation)
    ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.T.shapeUp (1 : Rotation)
    ∧ (∀ cell ∈ Piece.T.shapeUp (1 : Rotation),
        (cell.1 = 0 → cell.2 = 1) ∧ (cell.1 = 1 → cell.2 ≤ 2)) := by
  decide

/-- The right-pointing T (rotation 3): narrow, foot/top structure. -/
theorem T_r3_shape :
    (∀ cell ∈ Piece.T.shapeUp (3 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.T.shapeUp (3 : Rotation)
    ∧ ((0 : ℕ), (2 : ℕ)) ∈ Piece.T.shapeUp (3 : Rotation)
    ∧ ((1 : ℕ), (1 : ℕ)) ∈ Piece.T.shapeUp (3 : Rotation)
    ∧ (∀ cell ∈ Piece.T.shapeUp (3 : Rotation),
        (cell.1 = 0 → cell.2 ≤ 2) ∧ (cell.1 = 1 → cell.2 = 1)) := by
  decide

/-- **The T flips the down-stair**: on a pair stepping down by one, a
left-pointing T lands free of debt and turns the profile into a
one-step-up — the T is the stair's direction-reverser, free of
charge. -/
theorem window_T_downstair_flip_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h + 1) (h1 : b.colHeight (j + 1) = h) :
    ∃ pl : Placement, pl.piece = Piece.T ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 2
      ∧ (pl.place b).colHeight (j + 1) = h + 3 := by
  classical
  obtain ⟨hn, hm01, hm10, hm12, hmm⟩ := T_r1_shape
  have hcells : ∀ cell ∈ (⟨Piece.T, 1, j⟩ : Placement).shapeUp,
      (⟨Piece.T, 1, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.T, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.T, 1, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      have hmmc := hmm cell hcell
      by_cases hc0 : cell.1 = 0
      · have := hmmc.1 hc0
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.T, 1, j⟩ : Placement).col + cell.1)
            - cell.2) hm10
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.T, 1, j⟩ : Placement).col + ((1 : ℕ), (0 : ℕ)).1
          = j + 1 from rfl, h1] at hle
      simpa using hle
  refine ⟨⟨Piece.T, 1, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.T, 1, j⟩) (f₀ := 1) (f₁ := 0) hj hn rfl hm01
      (fun cell hcell hc0 => le_of_eq ((hmm cell hcell).1 hc0).symm)
      hm10 (fun cell _ _ => Nat.zero_le _)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.T, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (1 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').1 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.T, 1, j⟩) hm01 htop0
    rw [show (⟨Piece.T, 1, j⟩ : Placement).col + ((0 : ℕ), (1 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ (⟨Piece.T, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').2 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.T, 1, j⟩) hm12 htop1
    rw [show (⟨Piece.T, 1, j⟩ : Placement).col + ((1 : ℕ), (2 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- **The T flips the up-stair**: mirror — on a one-step-up pair a
right-pointing T lands free and turns the profile into a
one-step-down. -/
theorem window_T_upstair_flip_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h + 1) :
    ∃ pl : Placement, pl.piece = Piece.T ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 3
      ∧ (pl.place b).colHeight (j + 1) = h + 2 := by
  classical
  obtain ⟨hn, hm00, hm02, hm11, hmm⟩ := T_r3_shape
  have hcells : ∀ cell ∈ (⟨Piece.T, 3, j⟩ : Placement).shapeUp,
      (⟨Piece.T, 3, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.T, 3, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.T, 3, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      have hmmc := hmm cell hcell
      by_cases hc0 : cell.1 = 0
      · change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        have := hmmc.2 hc1
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.T, 3, j⟩ : Placement).col + cell.1)
            - cell.2) hm00
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.T, 3, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  refine ⟨⟨Piece.T, 3, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.T, 3, j⟩) (f₀ := 0) (f₁ := 1) hj hn rfl hm00
      (fun cell _ _ => Nat.zero_le _) hm11
      (fun cell hcell hc1 => le_of_eq ((hmm cell hcell).2 hc1).symm)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.T, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').1 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.T, 3, j⟩) hm02 htop0
    rw [show (⟨Piece.T, 3, j⟩ : Placement).col + ((0 : ℕ), (2 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ (⟨Piece.T, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (1 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (1 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').2 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.T, 3, j⟩) hm11 htop1
    rw [show (⟨Piece.T, 3, j⟩ : Placement).col + ((1 : ℕ), (1 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- The arm-up L (rotation 3): narrow, single high cell left, full
column right. -/
theorem L_r3_shape :
    (∀ cell ∈ Piece.L.shapeUp (3 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (2 : ℕ)) ∈ Piece.L.shapeUp (3 : Rotation)
    ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp (3 : Rotation)
    ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.L.shapeUp (3 : Rotation)
    ∧ (∀ cell ∈ Piece.L.shapeUp (3 : Rotation),
        (cell.1 = 0 → cell.2 = 2) ∧ (cell.1 = 1 → cell.2 ≤ 2)) := by
  decide

/-- The arm-up J (rotation 1): narrow, full column left, single high
cell right. -/
theorem J_r1_shape :
    (∀ cell ∈ Piece.J.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp (1 : Rotation)
    ∧ ((0 : ℕ), (2 : ℕ)) ∈ Piece.J.shapeUp (1 : Rotation)
    ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.J.shapeUp (1 : Rotation)
    ∧ (∀ cell ∈ Piece.J.shapeUp (1 : Rotation),
        (cell.1 = 0 → cell.2 ≤ 2) ∧ (cell.1 = 1 → cell.2 = 2)) := by
  decide

/-- **The L repairs the two-step**: on a pair dropping two from left to
right, an arm-up L lands free of debt and leaves the pair FLAT — the L
is the deep-step's repairer. -/
theorem window_L_repairs_down2_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h + 2) (h1 : b.colHeight (j + 1) = h) :
    ∃ pl : Placement, pl.piece = Piece.L ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 3
      ∧ (pl.place b).colHeight (j + 1) = h + 3 := by
  classical
  obtain ⟨hn, hm02, hm10, hm12, hmm⟩ := L_r3_shape
  have hcells : ∀ cell ∈ (⟨Piece.L, 3, j⟩ : Placement).shapeUp,
      (⟨Piece.L, 3, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.L, 3, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.L, 3, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      have hmmc := hmm cell hcell
      by_cases hc0 : cell.1 = 0
      · have := hmmc.1 hc0
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.L, 3, j⟩ : Placement).col + cell.1)
            - cell.2) hm10
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.L, 3, j⟩ : Placement).col + ((1 : ℕ), (0 : ℕ)).1
          = j + 1 from rfl, h1] at hle
      simpa using hle
  refine ⟨⟨Piece.L, 3, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.L, 3, j⟩) (f₀ := 2) (f₁ := 0) hj hn rfl hm02
      (fun cell hcell hc0 => le_of_eq ((hmm cell hcell).1 hc0).symm)
      hm10 (fun cell _ _ => Nat.zero_le _)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.L, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').1 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.L, 3, j⟩) hm02 htop0
    rw [show (⟨Piece.L, 3, j⟩ : Placement).col + ((0 : ℕ), (2 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ (⟨Piece.L, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').2 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.L, 3, j⟩) hm12 htop1
    rw [show (⟨Piece.L, 3, j⟩ : Placement).col + ((1 : ℕ), (2 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- **The J repairs the two-step**: mirror — on a pair rising two from
left to right, an arm-up J lands free and leaves the pair flat. -/
theorem window_J_repairs_up2_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h + 2) :
    ∃ pl : Placement, pl.piece = Piece.J ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 3
      ∧ (pl.place b).colHeight (j + 1) = h + 3 := by
  classical
  obtain ⟨hn, hm00, hm02, hm12, hmm⟩ := J_r1_shape
  have hcells : ∀ cell ∈ (⟨Piece.J, 1, j⟩ : Placement).shapeUp,
      (⟨Piece.J, 1, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.J, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.J, 1, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      have hmmc := hmm cell hcell
      by_cases hc0 : cell.1 = 0
      · change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        have := hmmc.2 hc1
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.J, 1, j⟩ : Placement).col + cell.1)
            - cell.2) hm00
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.J, 1, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  refine ⟨⟨Piece.J, 1, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.J, 1, j⟩) (f₀ := 0) (f₁ := 2) hj hn rfl hm00
      (fun cell _ _ => Nat.zero_le _) hm12
      (fun cell hcell hc1 => le_of_eq ((hmm cell hcell).2 hc1).symm)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.J, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').1 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.J, 1, j⟩) hm02 htop0
    rw [show (⟨Piece.J, 1, j⟩ : Placement).col + ((0 : ℕ), (2 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ (⟨Piece.J, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').2 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.J, 1, j⟩) hm12 htop1
    rw [show (⟨Piece.J, 1, j⟩ : Placement).col + ((1 : ℕ), (2 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- The seated L (rotation 1): narrow, full column left, single foot
right. -/
theorem L_r1_shape :
    (∀ cell ∈ Piece.L.shapeUp (1 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp (1 : Rotation)
    ∧ ((0 : ℕ), (2 : ℕ)) ∈ Piece.L.shapeUp (1 : Rotation)
    ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.L.shapeUp (1 : Rotation)
    ∧ (∀ cell ∈ Piece.L.shapeUp (1 : Rotation),
        (cell.1 = 0 → cell.2 ≤ 2) ∧ (cell.1 = 1 → cell.2 = 0)) := by
  decide

/-- The seated J (rotation 3): narrow, single foot left, full column
right. -/
theorem J_r3_shape :
    (∀ cell ∈ Piece.J.shapeUp (3 : Rotation), cell.1 ≤ 1)
    ∧ ((0 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp (3 : Rotation)
    ∧ ((1 : ℕ), (0 : ℕ)) ∈ Piece.J.shapeUp (3 : Rotation)
    ∧ ((1 : ℕ), (2 : ℕ)) ∈ Piece.J.shapeUp (3 : Rotation)
    ∧ (∀ cell ∈ Piece.J.shapeUp (3 : Rotation),
        (cell.1 = 0 → cell.2 = 0) ∧ (cell.1 = 1 → cell.2 ≤ 2)) := by
  decide

/-- **The seated L digs the two-step**: on a flat pair it lands free and
leaves a two-step drop — the inverse of the arm-up repair. -/
theorem window_L_makes_down2_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h) :
    ∃ pl : Placement, pl.piece = Piece.L ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 3
      ∧ (pl.place b).colHeight (j + 1) = h + 1 := by
  classical
  obtain ⟨hn, hm00, hm02, hm10, hmm⟩ := L_r1_shape
  have hcells : ∀ cell ∈ (⟨Piece.L, 1, j⟩ : Placement).shapeUp,
      (⟨Piece.L, 1, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.L, 1, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.L, 1, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      by_cases hc0 : cell.1 = 0
      · change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.L, 1, j⟩ : Placement).col + cell.1)
            - cell.2) hm00
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.L, 1, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  refine ⟨⟨Piece.L, 1, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.L, 1, j⟩) (f₀ := 0) (f₁ := 0) hj hn rfl hm00
      (fun cell _ _ => Nat.zero_le _) hm10
      (fun cell _ _ => Nat.zero_le _)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.L, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').1 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.L, 1, j⟩) hm02 htop0
    rw [show (⟨Piece.L, 1, j⟩ : Placement).col + ((0 : ℕ), (2 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
  · have htop1 : ∀ cell' ∈ (⟨Piece.L, 1, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (0 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (0 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').2 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.L, 1, j⟩) hm10 htop1
    rw [show (⟨Piece.L, 1, j⟩ : Placement).col + ((1 : ℕ), (0 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]
    simp

/-- **The seated J digs the two-step**: mirror — on a flat pair it lands
free and leaves a two-step rise. -/
theorem window_J_makes_up2_exists {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h) :
    ∃ pl : Placement, pl.piece = Piece.J ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.holes GameConfig.standard (pl.place b)
          = Board.holes GameConfig.standard b
      ∧ (pl.place b).colHeight j = h + 1
      ∧ (pl.place b).colHeight (j + 1) = h + 3 := by
  classical
  obtain ⟨hn, hm00, hm10, hm12, hmm⟩ := J_r3_shape
  have hcells : ∀ cell ∈ (⟨Piece.J, 3, j⟩ : Placement).shapeUp,
      (⟨Piece.J, 3, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.J, 3, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hn cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.J, 3, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have hle1 := hn cell hcell
      by_cases hc0 : cell.1 = 0
      · change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc0]
        simp only [Nat.add_zero]
        omega
      · have hc1 : cell.1 = 1 := by omega
        change b.colHeight (j + cell.1) - cell.2 ≤ h
        rw [hc1, h1]
        omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.J, 3, j⟩ : Placement).col + cell.1)
            - cell.2) hm00
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.J, 3, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  refine ⟨⟨Piece.J, 3, j⟩, rfl, ?_, hcells, ?_, ?_, ?_⟩
  · intro cell hcell
    have := hn cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hbill := window_two_col_hole_bill (b := b)
      (pl := ⟨Piece.J, 3, j⟩) (f₀ := 0) (f₁ := 0) hj hn rfl hm00
      (fun cell _ _ => Nat.zero_le _) hm10
      (fun cell _ _ => Nat.zero_le _)
    rw [h0, h1, hD] at hbill
    omega
  · have htop0 : ∀ cell' ∈ (⟨Piece.J, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((0 : ℕ), (0 : ℕ)).1 → cell'.2 ≤ ((0 : ℕ), (0 : ℕ)).2 := by
      intro cell' hcell' hc
      exact le_of_eq ((hmm cell' hcell').1 hc)
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.J, 3, j⟩) hm00 htop0
    rw [show (⟨Piece.J, 3, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1 = j
        from by simp] at hfed
    rw [hfed, hD]
    simp
  · have htop1 : ∀ cell' ∈ (⟨Piece.J, 3, j⟩ : Placement).shapeUp,
        cell'.1 = ((1 : ℕ), (2 : ℕ)).1 → cell'.2 ≤ ((1 : ℕ), (2 : ℕ)).2 := by
      intro cell' hcell' hc
      exact (hmm cell' hcell').2 hc
    have hfed := place_fed_colHeight_eq (b := b)
      (pl := ⟨Piece.J, 3, j⟩) hm12 htop1
    rw [show (⟨Piece.J, 3, j⟩ : Placement).col + ((1 : ℕ), (2 : ℕ)).1
        = j + 1 from rfl] at hfed
    rw [hfed, hD]

/-- A confined move completes any row the other eight columns prepared,
provided its drop covers the pair's two cells of that row. -/
theorem confined_move_completes_row {b : Board} {pl : Placement}
    {j r : ℕ} (hj : j + 1 < 10)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 → (c, r) ∈ b)
    (hd0 : (j, r) ∈ pl.dropped b) (hd1 : (j + 1, r) ∈ pl.dropped b) :
    r ∈ Board.fullRows GameConfig.standard (pl.place b) := by
  classical
  have hmem : ∀ c < 10, (c, r) ∈ pl.place b := by
    intro c hc
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    by_cases h0 : c = j
    · right
      rw [h0]
      exact hd0
    · by_cases h1 : c = j + 1
      · right
        rw [h1]
        exact hd1
      · left
        exact hprep c hc h0 h1
  rw [Board.fullRows, Finset.mem_filter]
  constructor
  · rw [Finset.mem_image]
    exact ⟨(j, r), hmem j (by omega), rfl⟩
  · intro c hc
    rw [GameConfig.standard_cols, Finset.mem_range] at hc
    exact hmem c hc
/-- **The window O harvests a double**: on a flat pair whose next two
rows the other eight columns have prepared, a confined O completes both
at once — the window economy's clearing side: sweeps prepare rows, the
returning window's square reaps them two at a time, debt-free. -/
theorem window_O_completes_two_rows {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 →
      (c, h) ∈ b ∧ (c, h + 1) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.O ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ 2 ≤ (Board.fullRows GameConfig.standard
          (pl.place b)).card := by
  classical
  obtain ⟨hf0, hf1⟩ := O_shape_feet (0 : Rotation)
  obtain ⟨ht0, ht1, _⟩ := O_shape_tops (0 : Rotation)
  have hnarrow := (O_shape_columns (0 : Rotation)).2.2
  have hcells : ∀ cell ∈ (⟨Piece.O, 0, j⟩ : Placement).shapeUp,
      (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hnarrow cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.O, 0, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · exact confined_dropOffset_le_of_flat hcells h0 h1
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.O, 0, j⟩ : Placement).col + cell.1)
            - cell.2) hf0
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.O, 0, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  have hdmem : ∀ cell ∈ (⟨Piece.O, 0, j⟩ : Placement).shapeUp,
      (j + cell.1, h + cell.2) ∈ (⟨Piece.O, 0, j⟩ : Placement).dropped b := by
    intro cell hcell
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.mem_image]
    exact ⟨cell, hcell, by rw [hD]⟩
  have hfull0 : h ∈ Board.fullRows GameConfig.standard
      ((⟨Piece.O, 0, j⟩ : Placement).place b) := by
    apply confined_move_completes_row hj
      (fun c hc hn0 hn1 => (hprep c hc hn0 hn1).1)
    · have := hdmem _ hf0
      simpa using this
    · have := hdmem _ hf1
      simpa using this
  have hfull1 : h + 1 ∈ Board.fullRows GameConfig.standard
      ((⟨Piece.O, 0, j⟩ : Placement).place b) := by
    apply confined_move_completes_row hj
      (fun c hc hn0 hn1 => (hprep c hc hn0 hn1).2)
    · have := hdmem _ ht0
      simpa using this
    · have := hdmem _ ht1
      simpa using this
  refine ⟨⟨Piece.O, 0, j⟩, rfl, ?_, hcells, ?_⟩
  · intro cell hcell
    have := hnarrow cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · have hsub : ({h, h + 1} : Finset ℕ)
        ⊆ Board.fullRows GameConfig.standard
          ((⟨Piece.O, 0, j⟩ : Placement).place b) := by
      intro r hr
      simp only [Finset.mem_insert, Finset.mem_singleton] at hr
      rcases hr with h' | h' <;> subst h'
      · exact hfull0
      · exact hfull1
    calc (2 : ℕ) = ({h, h + 1} : Finset ℕ).card := by
          rw [Finset.card_insert_of_notMem (by simp), Finset.card_singleton]
      _ ≤ _ := Finset.card_le_card hsub

/-- **The O harvest is exact**: on a clear-free board with a flat
prepared pair, the confined O completes precisely the two prepared rows
— no more, no fewer. The double is surgical. -/
theorem window_O_harvest_exact {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 →
      (c, h) ∈ b ∧ (c, h + 1) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.O ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.fullRows GameConfig.standard (pl.place b)
          = ({h, h + 1} : Finset ℕ) := by
  classical
  obtain ⟨hf0, hf1⟩ := O_shape_feet (0 : Rotation)
  obtain ⟨ht0, ht1, htops⟩ := O_shape_tops (0 : Rotation)
  have hnarrow := (O_shape_columns (0 : Rotation)).2.2
  have hcells : ∀ cell ∈ (⟨Piece.O, 0, j⟩ : Placement).shapeUp,
      (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j
      ∨ (⟨Piece.O, 0, j⟩ : Placement).col + cell.1 = j + 1 := by
    intro cell hcell
    have := hnarrow cell hcell
    change j + cell.1 = j ∨ j + cell.1 = j + 1
    omega
  have hD : (⟨Piece.O, 0, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · exact confined_dropOffset_le_of_flat hcells h0 h1
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.O, 0, j⟩ : Placement).col + cell.1)
            - cell.2) hf0
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.O, 0, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  have hdrows : ∀ p ∈ (⟨Piece.O, 0, j⟩ : Placement).dropped b,
      p.2 = h ∨ p.2 = h + 1 := by
    intro p hp
    unfold Placement.dropped Placement.cellsAt at hp
    rw [Finset.mem_image] at hp
    obtain ⟨cell, hcell, heq⟩ := hp
    have hrow := congrArg Prod.snd heq
    simp only [] at hrow
    have := htops cell hcell
    omega
  have hdmem : ∀ cell ∈ (⟨Piece.O, 0, j⟩ : Placement).shapeUp,
      (j + cell.1, h + cell.2)
        ∈ (⟨Piece.O, 0, j⟩ : Placement).dropped b := by
    intro cell hcell
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.mem_image]
    exact ⟨cell, hcell, by rw [hD]⟩
  refine ⟨⟨Piece.O, 0, j⟩, rfl, ?_, hcells, ?_⟩
  · intro cell hcell
    have := hnarrow cell hcell
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · apply Finset.Subset.antisymm
    · intro r hr
      have hfull := (Finset.mem_filter.mp hr).2
      simp only [Finset.mem_insert, Finset.mem_singleton]
      by_contra hne
      push Not at hne
      have hmemj : (j, r) ∈ (⟨Piece.O, 0, j⟩ : Placement).place b :=
        hfull j (by
          rw [GameConfig.standard_cols]
          exact Finset.mem_range.mpr (by omega))
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmemj
      have hrlt : r < h := by
        rcases hmemj with hb | hd
        · have := Board.lt_colHeight hb
          omega
        · have := hdrows _ hd
          simp only [] at this
          omega
      apply hnf r
      intro c hcr
      have hmemc := hfull c hcr
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmemc
      rcases hmemc with hb | hd
      · exact hb
      · exfalso
        have := hdrows _ hd
        simp only [] at this
        omega
    · intro r hr
      simp only [Finset.mem_insert, Finset.mem_singleton] at hr
      rcases hr with h' | h' <;> subst h'
      · apply confined_move_completes_row hj
          (fun c hc hn0 hn1 => (hprep c hc hn0 hn1).1)
        · have := hdmem _ hf0
          simpa using this
        · have := hdmem _ hf1
          simpa using this
      · apply confined_move_completes_row hj
          (fun c hc hn0 hn1 => (hprep c hc hn0 hn1).2)
        · have := hdmem _ ht0
          simpa using this
        · have := hdmem _ ht1
          simpa using this

/-- A column whose drop lands only in the two rows about to clear comes
out of the full move at exactly its original height: the added cells
vanish with their rows and nothing below shifts. -/
theorem applyStep_colHeight_reset {b : Board} {pl : Placement}
    {c h : ℕ} (hcH : b.colHeight c = h)
    (hfr : Board.fullRows GameConfig.standard (pl.place b)
      = ({h, h + 1} : Finset ℕ))
    (hdc : ∀ p ∈ pl.dropped b, p.2 = h ∨ p.2 = h + 1) :
    (Placement.applyStep GameConfig.standard b pl).colHeight c = h := by
  classical
  have hrows : (Placement.applyStep GameConfig.standard b pl).colRows c
      = b.colRows c := by
    unfold Placement.applyStep
    ext r
    unfold Board.colRows
    simp only [Finset.mem_image, Finset.mem_filter]
    constructor
    · rintro ⟨q, ⟨hq, hqc⟩, rfl⟩
      rw [Board.mem_clearLines_iff] at hq
      obtain ⟨p, hpm, hpnf, hpq⟩ := hq
      have hpc := congrArg Prod.fst hpq
      have hpr := congrArg Prod.snd hpq
      simp only [] at hpc hpr
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hpm
      rcases hpm with hb | hd
      · have hp1c : p.1 = c := by
          have := hpc
          omega
        have hlt : p.2 < h := by
          have hl := Board.lt_colHeight (b := b) (j := p.1) (r := p.2)
            (by exact hb)
          rw [hp1c, hcH] at hl
          exact hl
        have hcb : Board.clearedBelow GameConfig.standard (pl.place b) p.2
            = 0 := by
          unfold Board.clearedBelow
          rw [hfr]
          rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
          intro x hx
          rw [Finset.mem_filter] at hx
          simp only [Finset.mem_insert, Finset.mem_singleton] at hx
          omega
        refine ⟨p, ⟨?_, ?_⟩, ?_⟩
        · exact hb
        · omega
        · omega
      · exfalso
        have := hdc p hd
        have hfull : Board.isFull GameConfig.standard (pl.place b) p.2 := by
          have hmem : p.2 ∈ Board.fullRows GameConfig.standard
              (pl.place b) := by
            rw [hfr]
            simp only [Finset.mem_insert, Finset.mem_singleton]
            omega
          exact (Finset.mem_filter.mp hmem).2
        exact hpnf hfull
    · rintro ⟨p, ⟨hpb, hpc⟩, rfl⟩
      have hlt : p.2 < h := by
        have hl := Board.lt_colHeight (b := b) (j := p.1) (r := p.2) hpb
        rw [hpc, hcH] at hl
        exact hl
      have hnf : ¬ Board.isFull GameConfig.standard (pl.place b) p.2 := by
        intro hfull
        have hmem : p.2 ∈ Board.fullRows GameConfig.standard
            (pl.place b) := by
          rw [Board.fullRows, Finset.mem_filter]
          refine ⟨?_, hfull⟩
          rw [Finset.mem_image]
          refine ⟨p, ?_, rfl⟩
          rw [Placement.place_eq_union_dropped, Finset.mem_union]
          left
          exact hpb
        rw [hfr] at hmem
        simp only [Finset.mem_insert, Finset.mem_singleton] at hmem
        omega
      have hcb : Board.clearedBelow GameConfig.standard (pl.place b) p.2
          = 0 := by
        unfold Board.clearedBelow
        rw [hfr]
        rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
        intro x hx
        rw [Finset.mem_filter] at hx
        simp only [Finset.mem_insert, Finset.mem_singleton] at hx
        omega
      refine ⟨p, ⟨?_, hpc⟩, ?_⟩
      · rw [Board.mem_clearLines_iff]
        refine ⟨p, ?_, hnf, ?_⟩
        · rw [Placement.place_eq_union_dropped, Finset.mem_union]
          left
          exact hpb
        · rw [hcb]
          simp
      · rfl
  unfold Board.colHeight
  rw [hrows]
  have := hcH
  unfold Board.colHeight at this
  exact this

/-- **THE PERFECT SERVICE**: on a clear-free board whose flat pair sits
before two prepared rows, one confined O clears exactly two rows and
returns both pair columns to precisely their starting height — a
complete deposit-harvest-reset cycle in a single move, debt-free. The
moving window's ideal visit, realized: arrive, drop the square, clear
the double, leave the window exactly as found. -/
theorem window_O_perfect_service {b : Board} {j h : ℕ}
    (hj : j + 1 < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 →
      (c, h) ∈ b ∧ (c, h + 1) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.O ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.linesCleared GameConfig.standard (pl.place b) = 2
      ∧ (Placement.applyStep GameConfig.standard b pl).colHeight j = h
      ∧ (Placement.applyStep GameConfig.standard b pl).colHeight (j + 1)
          = h := by
  classical
  obtain ⟨pl, hpiece, hvalid, hcells, hfr⟩ :=
    window_O_harvest_exact (b := b) hj hnf h0 h1 hprep
  have hsh : pl.shapeUp = Piece.O.shapeUp pl.rot := by
    unfold Placement.shapeUp
    rw [hpiece]
  obtain ⟨hf0, hf1⟩ := O_shape_feet pl.rot
  obtain ⟨_, _, htops⟩ := O_shape_tops pl.rot
  have hm00 : ((0 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf0
  have hm10 : ((1 : ℕ), (0 : ℕ)) ∈ pl.shapeUp := by
    rw [hsh]
    exact hf1
  have hj0 := hcells _ hm00
  have hj1 := hcells _ hm10
  have e0 : pl.col + ((0 : ℕ), (0 : ℕ)).1 = pl.col := by simp
  have e1 : pl.col + ((1 : ℕ), (0 : ℕ)).1 = pl.col + 1 := by simp
  rw [e0] at hj0
  rw [e1] at hj1
  have hcol : pl.col = j := by omega
  have hD : pl.dropOffset b = h := by
    apply Nat.le_antisymm
    · exact confined_dropOffset_le_of_flat hcells h0 h1
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight (pl.col + cell.1) - cell.2) hm00
      unfold Placement.dropOffset
      rw [hcol]
      rw [hcol] at hle
      simp only [] at hle ⊢
      rw [show j + ((0 : ℕ), (0 : ℕ)).1 = j from by simp, h0] at hle
      simpa using hle
  have hdrows : ∀ p ∈ pl.dropped b, p.2 = h ∨ p.2 = h + 1 := by
    intro p hp
    unfold Placement.dropped Placement.cellsAt at hp
    rw [Finset.mem_image] at hp
    obtain ⟨cell, hcell, heq⟩ := hp
    have hrow := congrArg Prod.snd heq
    simp only [] at hrow
    have := htops cell (by rw [← hsh]; exact hcell)
    omega
  have hlines : Board.linesCleared GameConfig.standard (pl.place b)
      = 2 := by
    unfold Board.linesCleared
    rw [hfr, Finset.card_insert_of_notMem (by simp),
      Finset.card_singleton]
  exact ⟨pl, hpiece, hvalid, hcells, hlines,
    applyStep_colHeight_reset h0 hfr hdrows,
    applyStep_colHeight_reset h1 hfr hdrows⟩

/-- **The general reset lemma**: a column whose drop lands only in the
clearing band — all of it at or above the column's height — exits the
full move at exactly its original height, whatever the band. -/
theorem applyStep_colHeight_reset_general {b : Board} {pl : Placement}
    {c h : ℕ} {F : Finset ℕ} (hcH : b.colHeight c = h)
    (hfr : Board.fullRows GameConfig.standard (pl.place b) = F)
    (hF : ∀ x ∈ F, h ≤ x)
    (hdc : ∀ p ∈ pl.dropped b, p.2 ∈ F) :
    (Placement.applyStep GameConfig.standard b pl).colHeight c = h := by
  classical
  have hrows : (Placement.applyStep GameConfig.standard b pl).colRows c
      = b.colRows c := by
    unfold Placement.applyStep
    ext r
    unfold Board.colRows
    simp only [Finset.mem_image, Finset.mem_filter]
    constructor
    · rintro ⟨q, ⟨hq, hqc⟩, rfl⟩
      rw [Board.mem_clearLines_iff] at hq
      obtain ⟨p, hpm, hpnf, hpq⟩ := hq
      have hpc := congrArg Prod.fst hpq
      have hpr := congrArg Prod.snd hpq
      simp only [] at hpc hpr
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hpm
      rcases hpm with hb | hd
      · have hp1c : p.1 = c := by
          have := hpc
          omega
        have hlt : p.2 < h := by
          have hl := Board.lt_colHeight (b := b) (j := p.1) (r := p.2)
            (by exact hb)
          rw [hp1c, hcH] at hl
          exact hl
        have hcb : Board.clearedBelow GameConfig.standard (pl.place b) p.2
            = 0 := by
          unfold Board.clearedBelow
          rw [hfr]
          rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
          intro x hx
          rw [Finset.mem_filter] at hx
          have := hF x hx.1
          omega
        refine ⟨p, ⟨?_, ?_⟩, ?_⟩
        · exact hb
        · omega
        · omega
      · exfalso
        have hpF := hdc p hd
        have hfull : Board.isFull GameConfig.standard (pl.place b) p.2 := by
          have hmem : p.2 ∈ Board.fullRows GameConfig.standard
              (pl.place b) := by
            rw [hfr]
            exact hpF
          exact (Finset.mem_filter.mp hmem).2
        exact hpnf hfull
    · rintro ⟨p, ⟨hpb, hpc⟩, rfl⟩
      have hlt : p.2 < h := by
        have hl := Board.lt_colHeight (b := b) (j := p.1) (r := p.2) hpb
        rw [hpc, hcH] at hl
        exact hl
      have hnf : ¬ Board.isFull GameConfig.standard (pl.place b) p.2 := by
        intro hfull
        have hmem : p.2 ∈ Board.fullRows GameConfig.standard
            (pl.place b) := by
          rw [Board.fullRows, Finset.mem_filter]
          refine ⟨?_, hfull⟩
          rw [Finset.mem_image]
          refine ⟨p, ?_, rfl⟩
          rw [Placement.place_eq_union_dropped, Finset.mem_union]
          left
          exact hpb
        rw [hfr] at hmem
        have := hF _ hmem
        omega
      have hcb : Board.clearedBelow GameConfig.standard (pl.place b) p.2
          = 0 := by
        unfold Board.clearedBelow
        rw [hfr]
        rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
        intro x hx
        rw [Finset.mem_filter] at hx
        have := hF x hx.1
        omega
      refine ⟨p, ⟨?_, hpc⟩, ?_⟩
      · rw [Board.mem_clearLines_iff]
        refine ⟨p, ?_, hnf, ?_⟩
        · rw [Placement.place_eq_union_dropped, Finset.mem_union]
          left
          exact hpb
        · rw [hcb]
          simp
      · rfl
  unfold Board.colHeight
  rw [hrows]
  have := hcH
  unfold Board.colHeight at this
  exact this

/-- The vertical I (rotation 1): the four-cell tower in column zero. -/
theorem I_r1_shape :
    ((0 : ℕ), (0 : ℕ)) ∈ Piece.I.shapeUp (1 : Rotation)
    ∧ ((0 : ℕ), (1 : ℕ)) ∈ Piece.I.shapeUp (1 : Rotation)
    ∧ ((0 : ℕ), (2 : ℕ)) ∈ Piece.I.shapeUp (1 : Rotation)
    ∧ ((0 : ℕ), (3 : ℕ)) ∈ Piece.I.shapeUp (1 : Rotation)
    ∧ ∀ cell ∈ Piece.I.shapeUp (1 : Rotation),
        cell.1 = 0 ∧ cell.2 ≤ 3 := by
  decide

/-- A single-column drop completes any row the other nine columns
prepared. -/
theorem single_col_move_completes_row {b : Board} {pl : Placement}
    {j r : ℕ} (hj : j < 10)
    (hprep : ∀ c < 10, c ≠ j → (c, r) ∈ b)
    (hd : (j, r) ∈ pl.dropped b) :
    r ∈ Board.fullRows GameConfig.standard (pl.place b) := by
  classical
  have hmem : ∀ c < 10, (c, r) ∈ pl.place b := by
    intro c hc
    rw [Placement.place_eq_union_dropped, Finset.mem_union]
    by_cases h0 : c = j
    · right
      rw [h0]
      exact hd
    · left
      exact hprep c hc h0
  rw [Board.fullRows, Finset.mem_filter]
  constructor
  · rw [Finset.mem_image]
    exact ⟨(j, r), hmem j hj, rfl⟩
  · intro c hc
    rw [GameConfig.standard_cols, Finset.mem_range] at hc
    exact hmem c hc

/-- **THE WINDOW TETRIS**: on a clear-free board where column `j` stands
at height `h` and the four rows above it are prepared in every other
column, the vertical I clears exactly four rows and returns column `j`
to precisely height `h` — the biggest harvest is also a perfect
service. -/
theorem window_I_tetris_service {b : Board} {j h : ℕ} (hj : j < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h0 : b.colHeight j = h)
    (hprep : ∀ c < 10, c ≠ j → ∀ k < 4, (c, h + k) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.I ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp, pl.col + cell.1 = j)
      ∧ Board.linesCleared GameConfig.standard (pl.place b) = 4
      ∧ (Placement.applyStep GameConfig.standard b pl).colHeight j = h := by
  classical
  obtain ⟨hc0, hc1, hc2, hc3, hall⟩ := I_r1_shape
  have hcells : ∀ cell ∈ (⟨Piece.I, 1, j⟩ : Placement).shapeUp,
      (⟨Piece.I, 1, j⟩ : Placement).col + cell.1 = j := by
    intro cell hcell
    have := (hall cell hcell).1
    change j + cell.1 = j
    omega
  have hD : (⟨Piece.I, 1, j⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · unfold Placement.dropOffset
      apply Finset.sup_le
      intro cell hcell
      have := (hall cell hcell).1
      change b.colHeight (j + cell.1) - cell.2 ≤ h
      rw [this]
      simp only [Nat.add_zero]
      omega
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.I, 1, j⟩ : Placement).col + cell.1)
            - cell.2) hc0
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.I, 1, j⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = j from by simp, h0] at hle
      simpa using hle
  have hdrows : ∀ p ∈ (⟨Piece.I, 1, j⟩ : Placement).dropped b,
      p.2 ∈ ({h, h + 1, h + 2, h + 3} : Finset ℕ) := by
    intro p hp
    unfold Placement.dropped Placement.cellsAt at hp
    rw [Finset.mem_image] at hp
    obtain ⟨cell, hcell, heq⟩ := hp
    have hrow := congrArg Prod.snd heq
    simp only [] at hrow
    have := (hall cell hcell).2
    simp only [Finset.mem_insert, Finset.mem_singleton]
    omega
  have hdmem : ∀ cell ∈ (⟨Piece.I, 1, j⟩ : Placement).shapeUp,
      (j, h + cell.2) ∈ (⟨Piece.I, 1, j⟩ : Placement).dropped b := by
    intro cell hcell
    unfold Placement.dropped Placement.cellsAt
    rw [Finset.mem_image]
    refine ⟨cell, hcell, ?_⟩
    rw [hD]
    have := (hall cell hcell).1
    change (j + cell.1, h + cell.2) = (j, h + cell.2)
    rw [this]
    simp
  have hfr : Board.fullRows GameConfig.standard
      ((⟨Piece.I, 1, j⟩ : Placement).place b)
      = ({h, h + 1, h + 2, h + 3} : Finset ℕ) := by
    apply Finset.Subset.antisymm
    · intro r hr
      have hfull := (Finset.mem_filter.mp hr).2
      simp only [Finset.mem_insert, Finset.mem_singleton]
      by_contra hne
      push Not at hne
      have hmemj : (j, r) ∈ (⟨Piece.I, 1, j⟩ : Placement).place b :=
        hfull j (by
          rw [GameConfig.standard_cols]
          exact Finset.mem_range.mpr hj)
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmemj
      have hrlt : r < h := by
        rcases hmemj with hb | hd
        · have := Board.lt_colHeight hb
          omega
        · have := hdrows _ hd
          simp only [Finset.mem_insert, Finset.mem_singleton] at this
          omega
      apply hnf r
      intro c hcr
      have hmemc := hfull c hcr
      rw [Placement.place_eq_union_dropped, Finset.mem_union] at hmemc
      rcases hmemc with hb | hd
      · exact hb
      · exfalso
        have := hdrows _ hd
        simp only [Finset.mem_insert, Finset.mem_singleton] at this
        omega
    · intro r hr
      simp only [Finset.mem_insert, Finset.mem_singleton] at hr
      have hget : ∀ k, k < 4 → h + k ∈ Board.fullRows GameConfig.standard
          ((⟨Piece.I, 1, j⟩ : Placement).place b) := by
        intro k hk
        apply single_col_move_completes_row hj
          (fun c hc hn0 => hprep c hc hn0 k hk)
        rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 by omega) with
          h' | h' | h' | h' <;> subst h'
        · have := hdmem _ hc0
          simpa using this
        · have := hdmem _ hc1
          simpa using this
        · have := hdmem _ hc2
          simpa using this
        · have := hdmem _ hc3
          simpa using this
      rcases hr with h' | h' | h' | h' <;> subst h'
      · simpa using hget 0 (by omega)
      · exact hget 1 (by omega)
      · exact hget 2 (by omega)
      · exact hget 3 (by omega)
  refine ⟨⟨Piece.I, 1, j⟩, rfl, ?_, hcells, ?_, ?_⟩
  · intro cell hcell
    have := (hall cell hcell).1
    change j + cell.1 < GameConfig.standard.cols
    rw [GameConfig.standard_cols]
    omega
  · unfold Board.linesCleared
    rw [hfr]
    rw [Finset.card_insert_of_notMem (by simp),
      Finset.card_insert_of_notMem (by simp),
      Finset.card_insert_of_notMem (by simp), Finset.card_singleton]
  · apply applyStep_colHeight_reset_general h0 hfr
    · intro x hx
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx
      omega
    · exact hdrows

/-- **Each perfect service buys sixteen lightness**: the O double drops
the cell count by exactly sixteen (four in, twenty out) and never adds
a hole — perfect services are the lightness invariant's income. One
double per five moves exactly balances the +4-per-move mass tax. -/
theorem window_O_perfect_service_lightens {b : Board} {j h : ℕ}
    (hj : j + 1 < 10) (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h0 : b.colHeight j = h) (h1 : b.colHeight (j + 1) = h)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 →
      (c, h) ∈ b ∧ (c, h + 1) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.O ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ (Placement.applyStep GameConfig.standard b pl).count + 16
          = b.count
      ∧ Board.holes GameConfig.standard
          (Placement.applyStep GameConfig.standard b pl)
        ≤ Board.holes GameConfig.standard b := by
  obtain ⟨pl, hpiece, hvalid, hcells, hlines, _, _⟩ :=
    window_O_perfect_service (b := b) hj hnf h0 h1 hprep
  refine ⟨pl, hpiece, hvalid, hcells, ?_, ?_⟩
  · have hcnt := applyStep_count GameConfig.standard b pl hwf hvalid
    rw [GameConfig.standard_cols, hlines] at hcnt
    omega
  · have hbill := window_O_hole_bill (b := b) hj hpiece hcells
    rw [h0, h1] at hbill
    simp only [max_self] at hbill
    have hcl := holes_clearLines_le GameConfig.standard (pl.place b)
    unfold Placement.applyStep
    omega

/-- **The tetris service buys thirty-six lightness**: the window tetris
drops the cell count by exactly thirty-six (four in, forty out) and
never adds a hole — the deepest single-move deleveraging the game
allows. -/
theorem window_I_tetris_lightens {b : Board} {j h : ℕ} (hj : j < 10)
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (h0 : b.colHeight j = h)
    (hprep : ∀ c < 10, c ≠ j → ∀ k < 4, (c, h + k) ∈ b) :
    ∃ pl : Placement, pl.piece = Piece.I ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp, pl.col + cell.1 = j)
      ∧ (Placement.applyStep GameConfig.standard b pl).count + 36
          = b.count
      ∧ Board.holes GameConfig.standard
          (Placement.applyStep GameConfig.standard b pl)
        ≤ Board.holes GameConfig.standard b := by
  obtain ⟨pl, hpiece, hvalid, hcells, hlines, _⟩ :=
    window_I_tetris_service (b := b) hj hnf h0 hprep
  refine ⟨pl, hpiece, hvalid, hcells, ?_, ?_⟩
  · have hcnt := applyStep_count GameConfig.standard b pl hwf hvalid
    rw [GameConfig.standard_cols, hlines] at hcnt
    omega
  · have hfree := vertical_I_hole_free (b := b) (pl := pl) (j := j)
      hpiece (fun cell hcell => Or.inl (hcells cell hcell))
    have hcl := holes_clearLines_le GameConfig.standard (pl.place b)
    unfold Placement.applyStep
    omega

/-- **Perfect services are available in-game**: every trace board is
clear-free, so whenever a flat pair sits before two prepared rows at
any step of any game, the O double service — exact two-clear, full
height reset — is a legal move right there. -/
theorem trace_window_O_perfect_service {π : Policy GameConfig.standard}
    (n : ℕ) {j h : ℕ} (hj : j + 1 < 10)
    (h0 : (trace GameConfig.standard π GameState.init n).board.colHeight j
      = h)
    (h1 : (trace GameConfig.standard π
      GameState.init n).board.colHeight (j + 1) = h)
    (hprep : ∀ c < 10, c ≠ j → c ≠ j + 1 →
      (c, h) ∈ (trace GameConfig.standard π GameState.init n).board
      ∧ (c, h + 1) ∈ (trace GameConfig.standard π GameState.init n).board) :
    ∃ pl : Placement, pl.piece = Piece.O ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp,
          pl.col + cell.1 = j ∨ pl.col + cell.1 = j + 1)
      ∧ Board.linesCleared GameConfig.standard
          (pl.place (trace GameConfig.standard π GameState.init n).board)
          = 2
      ∧ (Placement.applyStep GameConfig.standard
          (trace GameConfig.standard π GameState.init n).board pl).colHeight
          j = h
      ∧ (Placement.applyStep GameConfig.standard
          (trace GameConfig.standard π GameState.init n).board pl).colHeight
          (j + 1) = h :=
  window_O_perfect_service hj
    (fun r => trace_board_no_full n r) h0 h1 hprep

/-- **The window tetris is available in-game**: at any step where a
column sits at `h` with its four-row band prepared elsewhere, the
vertical I's exact four-clear with reset is a legal move. -/
theorem trace_window_I_tetris_service {π : Policy GameConfig.standard}
    (n : ℕ) {j h : ℕ} (hj : j < 10)
    (h0 : (trace GameConfig.standard π GameState.init n).board.colHeight j
      = h)
    (hprep : ∀ c < 10, c ≠ j → ∀ k < 4,
      (c, h + k) ∈ (trace GameConfig.standard π GameState.init n).board) :
    ∃ pl : Placement, pl.piece = Piece.I ∧ pl.Valid GameConfig.standard
      ∧ (∀ cell ∈ pl.shapeUp, pl.col + cell.1 = j)
      ∧ Board.linesCleared GameConfig.standard
          (pl.place (trace GameConfig.standard π GameState.init n).board)
          = 4
      ∧ (Placement.applyStep GameConfig.standard
          (trace GameConfig.standard π GameState.init n).board pl).colHeight
          j = h :=
  window_I_tetris_service hj (fun r => trace_board_no_full n r) h0 hprep

/-- Filling two complete rows above a clear-free board and clearing
returns exactly the original board: the band vanishes without a trace.
The board-level germ of a closed cycle. -/
theorem clearLines_two_band_reset {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h) :
    Board.clearLines GameConfig.standard
      (b ∪ (Finset.range 10) ×ˢ ({h, h + 1} : Finset ℕ)) = b := by
  classical
  have hXfull : ∀ r, r = h ∨ r = h + 1 →
      Board.isFull GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ ({h, h + 1} : Finset ℕ)) r := by
    intro r hr
    intro c hc
    rw [GameConfig.standard_cols] at hc
    rw [Finset.mem_union]
    right
    rw [Finset.mem_product]
    simp only [Finset.mem_insert, Finset.mem_singleton]
    exact ⟨hc, hr⟩
  have hlowfull : ∀ r, r ≠ h → r ≠ h + 1 →
      ¬ Board.isFull GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ ({h, h + 1} : Finset ℕ)) r := by
    intro r hr0 hr1 hfull
    apply hnf r
    intro c hc
    have := hfull c hc
    rw [Finset.mem_union] at this
    rcases this with hb | hX
    · exact hb
    · exfalso
      rw [Finset.mem_product] at hX
      simp only [Finset.mem_insert, Finset.mem_singleton] at hX
      omega
  have hcb : ∀ r, r < h →
      Board.clearedBelow GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ ({h, h + 1} : Finset ℕ)) r = 0 := by
    intro r hr
    unfold Board.clearedBelow
    rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro x hx
    rw [Finset.mem_filter] at hx
    obtain ⟨hxf, hxlt⟩ := hx
    have hxfull := (Finset.mem_filter.mp hxf).2
    by_cases hx0 : x = h
    · omega
    · by_cases hx1 : x = h + 1
      · omega
      · exact hlowfull x hx0 hx1 hxfull
  ext p
  rw [Board.mem_clearLines_iff]
  constructor
  · rintro ⟨q, hq, hqnf, hqp⟩
    rw [Finset.mem_union] at hq
    rcases hq with hb | hX
    · have hlt := hlow q hb
      have := hcb q.2 hlt
      rw [this] at hqp
      have : (q.1, q.2 - 0) = q := by simp
      rw [this] at hqp
      rw [← hqp]
      exact hb
    · exfalso
      rw [Finset.mem_product] at hX
      simp only [Finset.mem_insert, Finset.mem_singleton] at hX
      exact hqnf (hXfull q.2 hX.2)
  · intro hp
    have hlt := hlow p hp
    refine ⟨p, ?_, ?_, ?_⟩
    · rw [Finset.mem_union]
      left
      exact hp
    · intro hfull
      exact hlowfull p.2 (by omega) (by omega) hfull
    · rw [hcb p.2 hlt]
      simp

/-- The O's base rotation is exactly the four-cell square. -/
theorem O_r0_shape_eq :
    Piece.O.shapeUp (0 : Rotation)
      = ({(0, 0), (0, 1), (1, 0), (1, 1)} : Finset PieceCell) := by
  decide

/-- **The O's drop is the pair band**: on a board whose columns `c` and
`c + 1` both stand at height `h`, the square placed at `c` occupies
exactly the four cells of rows `h, h+1` in those two columns, and the
merge is the plain union with them. -/
theorem O_pair_dropped_eq {b : Board} {c h : ℕ}
    (h0 : b.colHeight c = h) (h1 : b.colHeight (c + 1) = h) :
    (⟨Piece.O, 0, c⟩ : Placement).dropped b
      = ({(c, h), (c, h + 1), (c + 1, h), (c + 1, h + 1)} : Finset Coord)
    ∧ (⟨Piece.O, 0, c⟩ : Placement).place b
      = b ∪ ({(c, h), (c, h + 1), (c + 1, h), (c + 1, h + 1)}
          : Finset Coord) := by
  classical
  have hnarrow := (O_shape_columns (0 : Rotation)).2.2
  have hcells : ∀ cell ∈ (⟨Piece.O, 0, c⟩ : Placement).shapeUp,
      (⟨Piece.O, 0, c⟩ : Placement).col + cell.1 = c
      ∨ (⟨Piece.O, 0, c⟩ : Placement).col + cell.1 = c + 1 := by
    intro cell hcell
    have := hnarrow cell hcell
    change c + cell.1 = c ∨ c + cell.1 = c + 1
    omega
  have hD : (⟨Piece.O, 0, c⟩ : Placement).dropOffset b = h := by
    apply Nat.le_antisymm
    · exact confined_dropOffset_le_of_flat hcells h0 h1
    · have hle := Finset.le_sup
        (f := fun cell : PieceCell =>
          b.colHeight ((⟨Piece.O, 0, c⟩ : Placement).col + cell.1)
            - cell.2) (O_shape_feet (0 : Rotation)).1
      unfold Placement.dropOffset
      simp only [] at hle ⊢
      rw [show (⟨Piece.O, 0, c⟩ : Placement).col + ((0 : ℕ), (0 : ℕ)).1
          = c from by simp, h0] at hle
      simpa using hle
  have hdrop : (⟨Piece.O, 0, c⟩ : Placement).dropped b
      = ({(c, h), (c, h + 1), (c + 1, h), (c + 1, h + 1)}
          : Finset Coord) := by
    unfold Placement.dropped Placement.cellsAt
    rw [hD]
    change (Piece.O.shapeUp (0 : Rotation)).image
        (fun cell => (c + cell.1, h + cell.2)) = _
    rw [O_r0_shape_eq]
    ext p
    simp only [Finset.mem_image, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨cell, hcell, rfl⟩
      rcases hcell with h' | h' | h' | h' <;> subst h' <;> simp
    · intro hp
      rcases hp with h' | h' | h' | h' <;> subst h'
      · exact ⟨(0, 0), by simp⟩
      · exact ⟨(0, 1), by simp⟩
      · exact ⟨(1, 0), by simp⟩
      · exact ⟨(1, 1), by simp⟩
  refine ⟨hdrop, ?_⟩
  rw [Placement.place_eq_union_dropped, hdrop]

/-- Adding cells confined to low-index columns leaves every higher
column's height unchanged. -/
theorem colHeight_union_high_cols {X Y : Board} {m : ℕ}
    (hY : ∀ p ∈ Y, p.1 < m) {c : ℕ} (hc : m ≤ c) :
    (X ∪ Y).colHeight c = X.colHeight c := by
  classical
  unfold Board.colHeight Board.colRows
  congr 1
  congr 1
  ext p
  simp only [Finset.mem_filter, Finset.mem_union]
  constructor
  · rintro ⟨hp, hpc⟩
    rcases hp with h' | h'
    · exact ⟨h', hpc⟩
    · exfalso
      have := hY p h'
      omega
  · rintro ⟨hp, hpc⟩
    exact ⟨Or.inl hp, hpc⟩

/-- A clear-free below-`h` board stays clear-free after any partial band
over fewer than ten columns. -/
theorem no_full_of_partial_band {b : Board} {h m : ℕ} (hm : m < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h) :
    ∀ r, ¬ Board.isFull GameConfig.standard
      (b ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ)) r := by
  intro r hfull
  have h9 := hfull 9 (by rw [GameConfig.standard_cols]; simp)
  rw [Finset.mem_union] at h9
  rcases h9 with hb | hX
  · have hrlt := hlow _ hb
    apply hnf r
    intro c hc
    have := hfull c hc
    rw [Finset.mem_union] at this
    rcases this with h' | h'
    · exact h'
    · exfalso
      rw [Finset.mem_product] at h'
      simp only [Finset.mem_insert, Finset.mem_singleton] at h'
      omega
  · rw [Finset.mem_product, Finset.mem_range] at hX
    omega

/-- Extending a band by its next pair of columns. -/
theorem band_extend {m h : ℕ} :
    (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ)
      ∪ ({(m, h), (m, h + 1), (m + 1, h), (m + 1, h + 1)} : Finset Coord)
    = (Finset.range (m + 2)) ×ˢ ({h, h + 1} : Finset ℕ) := by
  ext p
  simp only [Finset.mem_union, Finset.mem_product, Finset.mem_range,
    Finset.mem_insert, Finset.mem_singleton, Prod.ext_iff]
  constructor
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩) <;>
      exact ⟨by omega, by omega⟩
  · rintro ⟨h1, h2⟩
    by_cases hlt : p.1 < m
    · exact Or.inl ⟨hlt, h2⟩
    · rcases h2 with h2 | h2
      · by_cases hm0 : p.1 = m
        · right; left
          exact ⟨hm0, h2⟩
        · right; right; right; left
          exact ⟨by omega, h2⟩
      · by_cases hm0 : p.1 = m
        · right; right; left
          exact ⟨hm0, h2⟩
        · right; right; right; right
          exact ⟨by omega, h2⟩

/-- A dry step of the five-O ritual: the square at column `m` extends
the partial band by one pair, clearing nothing. -/
theorem five_O_dry_step {b : Board} {h m : ℕ} (hm : m + 2 < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH0 : b.colHeight m = h) (hH1 : b.colHeight (m + 1) = h) :
    Placement.applyStep GameConfig.standard
      (b ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ))
      ⟨Piece.O, 0, m⟩
    = b ∪ (Finset.range (m + 2)) ×ˢ ({h, h + 1} : Finset ℕ) := by
  classical
  have hband : ∀ p ∈ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ),
      p.1 < m := fun p hp =>
    Finset.mem_range.mp (Finset.mem_product.mp hp).1
  have h0 : (b ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ)).colHeight m
      = h := by
    rw [colHeight_union_high_cols hband (le_refl m)]
    exact hH0
  have h1 : (b ∪ (Finset.range m) ×ˢ
      ({h, h + 1} : Finset ℕ)).colHeight (m + 1) = h := by
    rw [colHeight_union_high_cols hband (by omega)]
    exact hH1
  obtain ⟨_, hplace⟩ := O_pair_dropped_eq
    (b := b ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ)) h0 h1
  have hplace' : (⟨Piece.O, 0, m⟩ : Placement).place
      (b ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ))
      = b ∪ (Finset.range (m + 2)) ×ˢ ({h, h + 1} : Finset ℕ) := by
    rw [hplace, Finset.union_assoc, band_extend]
  have hnofull := no_full_of_partial_band (m := m + 2) (by omega) hnf hlow
  have hempty : Board.fullRows GameConfig.standard
      (b ∪ (Finset.range (m + 2)) ×ˢ ({h, h + 1} : Finset ℕ)) = ∅ :=
    Finset.eq_empty_iff_forall_notMem.mpr (fun r hr =>
      hnofull r (Finset.mem_filter.mp hr).2)
  unfold Placement.applyStep
  rw [hplace']
  exact Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hempty

/-- The final step of the five-O ritual: the fifth square completes the
band, the double clears, and the original board returns. -/
theorem five_O_final_step {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH8 : b.colHeight 8 = h) (hH9 : b.colHeight 9 = h) :
    Placement.applyStep GameConfig.standard
      (b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ))
      ⟨Piece.O, 0, 8⟩ = b := by
  classical
  have hband : ∀ p ∈ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ),
      p.1 < 8 := fun p hp =>
    Finset.mem_range.mp (Finset.mem_product.mp hp).1
  have h0 : (b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ)).colHeight 8
      = h := by
    rw [colHeight_union_high_cols hband (le_refl 8)]
    exact hH8
  have h1 : (b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ)).colHeight 9
      = h := by
    rw [colHeight_union_high_cols hband (by omega)]
    exact hH9
  obtain ⟨_, hplace⟩ := O_pair_dropped_eq
    (b := b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ)) h0 h1
  have hplace' : (⟨Piece.O, 0, 8⟩ : Placement).place
      (b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ))
      = b ∪ (Finset.range 10) ×ˢ ({h, h + 1} : Finset ℕ) := by
    rw [hplace, Finset.union_assoc, band_extend]
  unfold Placement.applyStep
  rw [hplace']
  exact clearLines_two_band_reset hnf hlow

/-- **THE FIVE-O CYCLE**: from any clear-free board standing level at
height `h` across all ten columns, dropping the square on the five even
pairs in turn — five moves, one for each window of the even tiling —
clears a double on the last drop and returns the board to EXACTLY its
starting state, cell for cell. A concrete closed board cycle: the
even-tiling tour's minimal service loop, realized as five explicit
placements. -/
theorem five_O_cycle {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH : ∀ c < 10, b.colHeight c = h) :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard b
              ⟨Piece.O, 0, 0⟩) ⟨Piece.O, 0, 2⟩) ⟨Piece.O, 0, 4⟩)
          ⟨Piece.O, 0, 6⟩) ⟨Piece.O, 0, 8⟩ = b := by
  have e0 : Placement.applyStep GameConfig.standard b ⟨Piece.O, 0, 0⟩
      = b ∪ (Finset.range 2) ×ˢ ({h, h + 1} : Finset ℕ) := by
    have h' := five_O_dry_step (b := b) (m := 0) (by omega) hnf hlow
      (hH 0 (by omega)) (hH 1 (by omega))
    simpa using h'
  have e1 := five_O_dry_step (b := b) (m := 2) (by omega) hnf hlow
    (hH 2 (by omega)) (hH 3 (by omega))
  have e2 := five_O_dry_step (b := b) (m := 4) (by omega) hnf hlow
    (hH 4 (by omega)) (hH 5 (by omega))
  have e3 := five_O_dry_step (b := b) (m := 6) (by omega) hnf hlow
    (hH 6 (by omega)) (hH 7 (by omega))
  have e4 := five_O_final_step (b := b) hnf hlow
    (hH 8 (by omega)) (hH 9 (by omega))
  rw [e0]
  rw [show (2 : ℕ) + 2 = 4 from rfl] at e1
  rw [show (4 : ℕ) + 2 = 6 from rfl] at e2
  rw [show (6 : ℕ) + 2 = 8 from rfl] at e3
  rw [e1, e2, e3, e4]

/-- **The empty board rides a five-move loop**: five squares dropped on
the even pairs take the empty board back to the empty board — the
simplest closed cycle in all of Tetris, fully verified: four dry drops
building the ground floor, one drop completing it, a double clear, and
nothing remains. If the dealer hands five O's, the game is exactly
where it began. -/
theorem five_O_cycle_empty :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard (∅ : Board)
              ⟨Piece.O, 0, 0⟩) ⟨Piece.O, 0, 2⟩) ⟨Piece.O, 0, 4⟩)
          ⟨Piece.O, 0, 6⟩) ⟨Piece.O, 0, 8⟩ = (∅ : Board) := by
  apply five_O_cycle (h := 0)
  · intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  · intro p hp
    exact absurd hp (Finset.notMem_empty _)
  · intro c _
    exact Board.colHeight_empty c

/-- The four intermediate boards of the five-O ritual, in one package:
each drop extends the band by one even pair. -/
theorem five_O_intermediate_boards {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH : ∀ c < 10, b.colHeight c = h) :
    Placement.applyStep GameConfig.standard b ⟨Piece.O, 0, 0⟩
        = b ∪ (Finset.range 2) ×ˢ ({h, h + 1} : Finset ℕ)
    ∧ Placement.applyStep GameConfig.standard
        (b ∪ (Finset.range 2) ×ˢ ({h, h + 1} : Finset ℕ)) ⟨Piece.O, 0, 2⟩
        = b ∪ (Finset.range 4) ×ˢ ({h, h + 1} : Finset ℕ)
    ∧ Placement.applyStep GameConfig.standard
        (b ∪ (Finset.range 4) ×ˢ ({h, h + 1} : Finset ℕ)) ⟨Piece.O, 0, 4⟩
        = b ∪ (Finset.range 6) ×ˢ ({h, h + 1} : Finset ℕ)
    ∧ Placement.applyStep GameConfig.standard
        (b ∪ (Finset.range 6) ×ˢ ({h, h + 1} : Finset ℕ)) ⟨Piece.O, 0, 6⟩
        = b ∪ (Finset.range 8) ×ˢ ({h, h + 1} : Finset ℕ) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h' := five_O_dry_step (b := b) (m := 0) (by omega) hnf hlow
      (hH 0 (by omega)) (hH 1 (by omega))
    simpa using h'
  · have h' := five_O_dry_step (b := b) (m := 2) (by omega) hnf hlow
      (hH 2 (by omega)) (hH 3 (by omega))
    rwa [show (2 : ℕ) + 2 = 4 from rfl] at h'
  · have h' := five_O_dry_step (b := b) (m := 4) (by omega) hnf hlow
      (hH 4 (by omega)) (hH 5 (by omega))
    rwa [show (4 : ℕ) + 2 = 6 from rfl] at h'
  · have h' := five_O_dry_step (b := b) (m := 6) (by omega) hnf hlow
      (hH 6 (by omega)) (hH 7 (by omega))
    rwa [show (6 : ℕ) + 2 = 8 from rfl] at h'

/-- **The five-O cycle at the game-state level**: five O steps from any
state whose board is clear-free and level return the BOARD component
exactly. Only the bag moves on. -/
theorem five_O_state_board_cycle {g : GameState} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard g.board r)
    (hlow : ∀ p ∈ g.board, p.2 < h)
    (hH : ∀ c < 10, g.board.colHeight c = h) :
    (((((g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).step
        GameConfig.standard ⟨Piece.O, 0, 2⟩).step
        GameConfig.standard ⟨Piece.O, 0, 4⟩).step
        GameConfig.standard ⟨Piece.O, 0, 6⟩).step
        GameConfig.standard ⟨Piece.O, 0, 8⟩).board = g.board := by
  simp only [GameState.step_board]
  exact five_O_cycle hnf hlow hH

/-- **The five-O loop never tops out**: with two rows of ceiling above
the level (`h + 2 ≤ 20`), every state along the five-O ritual — the four
partial-band states and the return — is non-lost. The cycle is not only
closed, it is SAFE all the way round. -/
theorem five_O_loop_safe {g : GameState} {h : ℕ} (hcap : h + 2 ≤ 20)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard g.board r)
    (hlow : ∀ p ∈ g.board, p.2 < h)
    (hH : ∀ c < 10, g.board.colHeight c = h) :
    ¬ (g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).lost GameConfig.standard
    ∧ ¬ ((g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).step
        GameConfig.standard ⟨Piece.O, 0, 2⟩).lost GameConfig.standard
    ∧ ¬ (((g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).step
        GameConfig.standard ⟨Piece.O, 0, 2⟩).step
        GameConfig.standard ⟨Piece.O, 0, 4⟩).lost GameConfig.standard
    ∧ ¬ ((((g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).step
        GameConfig.standard ⟨Piece.O, 0, 2⟩).step
        GameConfig.standard ⟨Piece.O, 0, 4⟩).step
        GameConfig.standard ⟨Piece.O, 0, 6⟩).lost GameConfig.standard
    ∧ ¬ (((((g.step GameConfig.standard ⟨Piece.O, 0, 0⟩).step
        GameConfig.standard ⟨Piece.O, 0, 2⟩).step
        GameConfig.standard ⟨Piece.O, 0, 4⟩).step
        GameConfig.standard ⟨Piece.O, 0, 6⟩).step
        GameConfig.standard ⟨Piece.O, 0, 8⟩).lost GameConfig.standard := by
  obtain ⟨e0, e1, e2, e3⟩ := five_O_intermediate_boards hnf hlow hH
  have e4 := five_O_final_step (b := g.board) hnf hlow
    (hH 8 (by omega)) (hH 9 (by omega))
  have hband : ∀ m : ℕ, ∀ p ∈ g.board ∪ (Finset.range m) ×ˢ
      ({h, h + 1} : Finset ℕ), p.2 < 20 := by
    intro m p hp
    rw [Finset.mem_union] at hp
    rcases hp with hb | hX
    · have := hlow p hb
      omega
    · rw [Finset.mem_product] at hX
      obtain ⟨-, h2⟩ := hX
      simp only [Finset.mem_insert, Finset.mem_singleton] at h2
      omega
  have safe : ∀ m : ℕ,
      ¬ Board.isLost GameConfig.standard
        (g.board ∪ (Finset.range m) ×ˢ ({h, h + 1} : Finset ℕ)) := by
    intro m
    rw [Board.not_isLost_iff_forall_row_lt, GameConfig.standard_rows]
    exact hband m
  have safeb : ¬ Board.isLost GameConfig.standard g.board := by
    rw [Board.not_isLost_iff_forall_row_lt, GameConfig.standard_rows]
    intro p hp
    have := hlow p hp
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [GameState.not_lost_iff_not_board_isLost, GameState.step_board, e0]
    exact safe 2
  · rw [GameState.not_lost_iff_not_board_isLost]
    simp only [GameState.step_board]
    rw [e0, e1]
    exact safe 4
  · rw [GameState.not_lost_iff_not_board_isLost]
    simp only [GameState.step_board]
    rw [e0, e1, e2]
    exact safe 6
  · rw [GameState.not_lost_iff_not_board_isLost]
    simp only [GameState.step_board]
    rw [e0, e1, e2, e3]
    exact safe 8
  · rw [GameState.not_lost_iff_not_board_isLost]
    simp only [GameState.step_board]
    rw [e0, e1, e2, e3, e4]
    exact safeb

/-- **The five-O waltz**: the policy that reads the partial band off the
board — via the marker cells `(0,h), (2,h), (4,h), (6,h)` — and plays
the next square of the ritual. A closed-form, five-line strategy. -/
def fiveOPolicy (h : ℕ) : Policy GameConfig.standard := fun g =>
  if (6, h) ∈ g.board then ⟨Piece.O, 0, 8⟩
  else if (4, h) ∈ g.board then ⟨Piece.O, 0, 6⟩
  else if (2, h) ∈ g.board then ⟨Piece.O, 0, 4⟩
  else if (0, h) ∈ g.board then ⟨Piece.O, 0, 2⟩
  else ⟨Piece.O, 0, 0⟩

/-- Evaluation unfolding for the five-O policy. -/
theorem fiveOPolicy_eval (h : ℕ) (g : GameState) :
    fiveOPolicy h g
      = if (6, h) ∈ g.board then ⟨Piece.O, 0, 8⟩
        else if (4, h) ∈ g.board then ⟨Piece.O, 0, 6⟩
        else if (2, h) ∈ g.board then ⟨Piece.O, 0, 4⟩
        else if (0, h) ∈ g.board then ⟨Piece.O, 0, 2⟩
        else ⟨Piece.O, 0, 0⟩ := rfl

/-- **The waltz's exact orbit**: under the five-O policy from any
clear-free level state, the board at step `n` is the base board plus
the partial band of width `2·(n % 5)` — a closed-form formula for the
ENTIRE infinite trace. -/
theorem fiveOPolicy_trace_board {g0 : GameState} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard g0.board r)
    (hlow : ∀ p ∈ g0.board, p.2 < h)
    (hH : ∀ c < 10, g0.board.colHeight c = h) (n : ℕ) :
    (trace GameConfig.standard (fiveOPolicy h) g0 n).board
      = g0.board ∪ (Finset.range (2 * (n % 5))) ×ˢ
          ({h, h + 1} : Finset ℕ) := by
  have hmem : ∀ (m k : ℕ),
      ((k, h) ∈ g0.board ∪ (Finset.range m) ×ˢ
        ({h, h + 1} : Finset ℕ)) ↔ k < m := by
    intro m k
    simp only [Finset.mem_union, Finset.mem_product, Finset.mem_range,
      Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro (hb | ⟨hk, _⟩)
      · have := hlow _ hb
        simp at this
      · exact hk
    · intro hk
      right
      exact ⟨hk, by simp⟩
  obtain ⟨e0, e1, e2, e3⟩ := five_O_intermediate_boards hnf hlow hH
  have e4 := five_O_final_step (b := g0.board) hnf hlow
    (hH 8 (by omega)) (hH 9 (by omega))
  induction n with
  | zero => simp
  | succ k ih =>
    have hr : k % 5 = 0 ∨ k % 5 = 1 ∨ k % 5 = 2 ∨ k % 5 = 3
        ∨ k % 5 = 4 := by omega
    rcases hr with hr | hr | hr | hr | hr
    · have hu : g0.board ∪ (Finset.range (2 * (0 : ℕ))) ×ˢ
          ({h, h + 1} : Finset ℕ) = g0.board := by simp
      have hpol : fiveOPolicy h
          (trace GameConfig.standard (fiveOPolicy h) g0 k)
          = ⟨Piece.O, 0, 0⟩ := by
        rw [fiveOPolicy_eval, ih, hr, hu]
        rw [if_neg, if_neg, if_neg, if_neg]
        all_goals (intro hc; have := hlow _ hc; simp at this)
      rw [trace_succ, GameState.step_board, hpol, ih, hr, hu, e0,
        show (k + 1) % 5 = 1 from by omega,
        show (2 : ℕ) * 1 = 2 from rfl]
    · have hpol : fiveOPolicy h
          (trace GameConfig.standard (fiveOPolicy h) g0 k)
          = ⟨Piece.O, 0, 2⟩ := by
        rw [fiveOPolicy_eval, ih, hr]
        rw [if_neg, if_neg, if_neg, if_pos]
        all_goals (rw [hmem]; omega)
      rw [trace_succ, GameState.step_board, hpol, ih, hr,
        show (2 : ℕ) * 1 = 2 from rfl, e1,
        show (k + 1) % 5 = 2 from by omega,
        show (2 : ℕ) * 2 = 4 from rfl]
    · have hpol : fiveOPolicy h
          (trace GameConfig.standard (fiveOPolicy h) g0 k)
          = ⟨Piece.O, 0, 4⟩ := by
        rw [fiveOPolicy_eval, ih, hr]
        rw [if_neg, if_neg, if_pos]
        all_goals (rw [hmem]; omega)
      rw [trace_succ, GameState.step_board, hpol, ih, hr,
        show (2 : ℕ) * 2 = 4 from rfl, e2,
        show (k + 1) % 5 = 3 from by omega,
        show (2 : ℕ) * 3 = 6 from rfl]
    · have hpol : fiveOPolicy h
          (trace GameConfig.standard (fiveOPolicy h) g0 k)
          = ⟨Piece.O, 0, 6⟩ := by
        rw [fiveOPolicy_eval, ih, hr]
        rw [if_neg, if_pos]
        all_goals (rw [hmem]; omega)
      rw [trace_succ, GameState.step_board, hpol, ih, hr,
        show (2 : ℕ) * 3 = 6 from rfl, e3,
        show (k + 1) % 5 = 4 from by omega,
        show (2 : ℕ) * 4 = 8 from rfl]
    · have hpol : fiveOPolicy h
          (trace GameConfig.standard (fiveOPolicy h) g0 k)
          = ⟨Piece.O, 0, 8⟩ := by
        rw [fiveOPolicy_eval, ih, hr]
        rw [if_pos]
        rw [hmem]; omega
      rw [trace_succ, GameState.step_board, hpol, ih, hr,
        show (2 : ℕ) * 4 = 8 from rfl, e4,
        show (k + 1) % 5 = 0 from by omega]
      simp

/-- **The waltz returns every five steps**: the board component of the
five-O trace is periodic with period five, exactly. -/
theorem fiveOPolicy_board_period {g0 : GameState} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard g0.board r)
    (hlow : ∀ p ∈ g0.board, p.2 < h)
    (hH : ∀ c < 10, g0.board.colHeight c = h) (k : ℕ) :
    (trace GameConfig.standard (fiveOPolicy h) g0 (5 * k)).board
      = g0.board := by
  rw [fiveOPolicy_trace_board hnf hlow hH]
  simp [Nat.mul_mod_right]

/-- **THE FIVE-O WALTZ SURVIVES FOREVER**: from any clear-free level
state with two rows of ceiling, the five-O policy never tops out. A
closed-form strategy with a closed-form orbit — the free-piece world's
simplest possible perpetual game. (Policy world: the piece stream is
chosen by the policy, not dealt by a bag.) -/
theorem fiveOPolicy_survives {g0 : GameState} {h : ℕ} (hcap : h + 2 ≤ 20)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard g0.board r)
    (hlow : ∀ p ∈ g0.board, p.2 < h)
    (hH : ∀ c < 10, g0.board.colHeight c = h) :
    SurvivesForever GameConfig.standard (fiveOPolicy h) g0 := by
  intro n
  rw [GameState.not_lost_iff_not_board_isLost,
    fiveOPolicy_trace_board hnf hlow hH,
    Board.not_isLost_iff_forall_row_lt, GameConfig.standard_rows]
  intro p hp
  rw [Finset.mem_union] at hp
  rcases hp with hb | hX
  · have := hlow p hb
    omega
  · rw [Finset.mem_product] at hX
    obtain ⟨-, h2⟩ := hX
    simp only [Finset.mem_insert, Finset.mem_singleton] at h2
    omega

/-- **The waltz from the empty board**: the five-O policy at ground
level survives forever from `init` — a fully explicit, five-line,
period-five witness of `∃ π, SurvivesForever init` in the free-piece
world. -/
theorem fiveOPolicy_survives_init :
    SurvivesForever GameConfig.standard (fiveOPolicy 0) GameState.init := by
  apply fiveOPolicy_survives (h := 0) (by omega)
  · intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    rw [GameState.init_board_eq_emptyset] at h0
    exact absurd h0 (Finset.notMem_empty _)
  · intro p hp
    rw [GameState.init_board_eq_emptyset] at hp
    exact absurd hp (Finset.notMem_empty _)
  · intro c _
    rw [GameState.init_board_eq_emptyset]
    exact Board.colHeight_empty c

/-- Drawing an absent piece from a nonempty bag changes nothing: the
erase is a no-op and the refill guard never fires. -/
theorem bag_draw_absent_fixed {bag : Bag} {p : Piece}
    (hp : p ∉ bag) (hne : bag ≠ ∅) : bag.draw p = bag := by
  unfold Bag.draw
  rw [Finset.erase_eq_of_notMem hp, if_neg hne]

/-- Every branch of the five-O policy plays the square. -/
theorem fiveOPolicy_piece (h : ℕ) (g : GameState) :
    (fiveOPolicy h g).piece = Piece.O := by
  rw [fiveOPolicy_eval]
  split_ifs <;> rfl

/-- **The waltz's bag freezes after one step**: from `init`, the first
O leaves the six-piece bag `full.erase O`, and every later O-draw is a
no-op — the bag component is CONSTANT from step one on. -/
theorem fiveOPolicy_trace_bag_init {h : ℕ} (n : ℕ) (hn : 1 ≤ n) :
    (trace GameConfig.standard (fiveOPolicy h) GameState.init n).bag
      = Bag.full.erase Piece.O := by
  induction n with
  | zero => omega
  | succ k ih =>
    rw [trace_succ, GameState.step_bag, fiveOPolicy_piece]
    rcases Nat.eq_or_lt_of_le hn with h1 | h1
    · have hk : k = 0 := by omega
      subst hk
      rw [trace_zero, GameState.init_bag, Bag.draw_full_eq_erase]
    · rw [ih (by omega)]
      exact bag_draw_absent_fixed (Finset.notMem_erase _ _)
        (fun hemp => by
          have : (6 : ℕ) = 0 := by
            rw [← Bag.draw_full_card Piece.O, Bag.draw_full_eq_erase,
              hemp, Finset.card_empty]
          omega)

/-- Two game states with equal boards and equal bags are equal. -/
theorem GameState.eq_of_board_bag {g g' : GameState}
    (hb : g.board = g'.board) (hs : g.bag = g'.bag) : g = g' := by
  cases g
  cases g'
  rw [GameState.mk.injEq]
  exact ⟨hb, hs⟩

/-- **THE WALTZ'S STATE CYCLE**: from step one on, the FULL game state
(board and bag together) under the five-O policy from `init` repeats
with period five. Not just the board: the entire state graph orbit is
closed — a genuine five-state cycle reachable from the initial state in
one move. (Free-piece world.) -/
theorem fiveOPolicy_state_cycle_init (n : ℕ) (hn : 1 ≤ n) :
    trace GameConfig.standard (fiveOPolicy 0) GameState.init (n + 5)
      = trace GameConfig.standard (fiveOPolicy 0) GameState.init n := by
  have hnf : ∀ r, ¬ Board.isFull GameConfig.standard
      GameState.init.board r := by
    intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    rw [GameState.init_board_eq_emptyset] at h0
    exact absurd h0 (Finset.notMem_empty _)
  have hlow : ∀ p ∈ GameState.init.board, p.2 < 0 := by
    intro p hp
    rw [GameState.init_board_eq_emptyset] at hp
    exact absurd hp (Finset.notMem_empty _)
  have hH : ∀ c < 10, GameState.init.board.colHeight c = 0 := by
    intro c _
    rw [GameState.init_board_eq_emptyset]
    exact Board.colHeight_empty c
  apply GameState.eq_of_board_bag
  · rw [fiveOPolicy_trace_board hnf hlow hH,
      fiveOPolicy_trace_board hnf hlow hH,
      show (n + 5) % 5 = n % 5 from by omega]
  · rw [fiveOPolicy_trace_bag_init (n + 5) (by omega),
      fiveOPolicy_trace_bag_init n hn]

/-- **The waltz's orbit is FIVE STATES**: every trace state from step
one on equals one of the first five. The five-O policy realizes a
finite closed orbit in the full state graph — the free-piece world's
M2 object, explicitly. -/
theorem fiveOPolicy_orbit_five_states (n : ℕ) (hn : 1 ≤ n) :
    ∃ m, 1 ≤ m ∧ m ≤ 5 ∧
      trace GameConfig.standard (fiveOPolicy 0) GameState.init n
        = trace GameConfig.standard (fiveOPolicy 0) GameState.init m := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases hle : n ≤ 5
    · exact ⟨n, hn, hle, rfl⟩
    · have hrec : trace GameConfig.standard (fiveOPolicy 0)
          GameState.init n
          = trace GameConfig.standard (fiveOPolicy 0)
              GameState.init (n - 5) := by
        have := fiveOPolicy_state_cycle_init (n - 5) (by omega)
        rwa [show n - 5 + 5 = n from by omega] at this
      obtain ⟨m, hm1, hm5, hme⟩ := ih (n - 5) (by omega) (by omega)
      exact ⟨m, hm1, hm5, hrec.trans hme⟩

/-- The vertical I's base shape: four cells stacked in one column. -/
theorem I_r1_shape_eq :
    Piece.I.shapeUp (1 : Rotation)
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset PieceCell) := by
  decide

/-- **The general band reset**: filling `k` complete rows above a
clear-free board and clearing returns exactly the original board,
whatever `k`. The two-row reset generalized to any band height. -/
theorem clearLines_band_reset {b : Board} {h k : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h) :
    Board.clearLines GameConfig.standard
      (b ∪ (Finset.range 10) ×ˢ (Finset.Ico h (h + k))) = b := by
  classical
  have hXfull : ∀ r, h ≤ r → r < h + k →
      Board.isFull GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ (Finset.Ico h (h + k))) r := by
    intro r hr1 hr2 c hc
    rw [GameConfig.standard_cols] at hc
    rw [Finset.mem_union]
    right
    rw [Finset.mem_product, Finset.mem_Ico]
    exact ⟨hc, hr1, hr2⟩
  have hlowfull : ∀ r, ¬ (h ≤ r ∧ r < h + k) →
      ¬ Board.isFull GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ (Finset.Ico h (h + k))) r := by
    intro r hr hfull
    apply hnf r
    intro c hc
    have := hfull c hc
    rw [Finset.mem_union] at this
    rcases this with hb | hX
    · exact hb
    · exfalso
      rw [Finset.mem_product, Finset.mem_Ico] at hX
      exact hr ⟨hX.2.1, hX.2.2⟩
  have hcb : ∀ r, r < h →
      Board.clearedBelow GameConfig.standard
        (b ∪ (Finset.range 10) ×ˢ (Finset.Ico h (h + k))) r = 0 := by
    intro r hr
    unfold Board.clearedBelow
    rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro x hx
    rw [Finset.mem_filter] at hx
    obtain ⟨hxf, hxlt⟩ := hx
    have hxfull := (Finset.mem_filter.mp hxf).2
    exact hlowfull x (by omega) hxfull
  ext p
  rw [Board.mem_clearLines_iff]
  constructor
  · rintro ⟨q, hq, hqnf, hqp⟩
    rw [Finset.mem_union] at hq
    rcases hq with hb | hX
    · have hlt := hlow q hb
      have := hcb q.2 hlt
      rw [this] at hqp
      have : (q.1, q.2 - 0) = q := by simp
      rw [this] at hqp
      rw [← hqp]
      exact hb
    · exfalso
      rw [Finset.mem_product, Finset.mem_Ico] at hX
      exact hqnf (hXfull q.2 hX.2.1 hX.2.2)
  · intro hp
    have hlt := hlow p hp
    refine ⟨p, ?_, ?_, ?_⟩
    · rw [Finset.mem_union]
      left
      exact hp
    · intro hfull
      exact hlowfull p.2 (by omega) hfull
    · rw [hcb p.2 hlt]
      simp

/-- **The vertical I's drop is the column tower**: on a board whose
column `c` stands at height `h`, the vertical I placed at `c` occupies
exactly the four cells of rows `h..h+3` in that column, and the merge
is the plain union with them. -/
theorem I_column_dropped_eq {b : Board} {c h : ℕ}
    (hH : b.colHeight c = h) :
    (⟨Piece.I, 1, c⟩ : Placement).dropped b
      = ({(c, h), (c, h + 1), (c, h + 2), (c, h + 3)} : Finset Coord)
    ∧ (⟨Piece.I, 1, c⟩ : Placement).place b
      = b ∪ ({(c, h), (c, h + 1), (c, h + 2), (c, h + 3)}
          : Finset Coord) := by
  classical
  have hshape : (⟨Piece.I, 1, c⟩ : Placement).shapeUp
      = ({(0, 0), (0, 1), (0, 2), (0, 3)} : Finset Coord) := I_r1_shape_eq
  have hD : (⟨Piece.I, 1, c⟩ : Placement).dropOffset b = h := by
    refine Nat.le_antisymm ?_ ?_
    · apply dropOffset_le_of_heights
      intro cell hcell
      rw [hshape] at hcell
      simp only [Finset.mem_insert, Finset.mem_singleton] at hcell
      rcases hcell with h1 | h1 | h1 | h1 <;> rw [h1] <;> simp [hH]
    · have hcell : ((0, 0) : Coord)
          ∈ (⟨Piece.I, 1, c⟩ : Placement).shapeUp := by
        rw [hshape]
        simp
      have hle := Finset.le_sup (f := fun cell =>
        b.colHeight ((⟨Piece.I, 1, c⟩ : Placement).col + cell.1) - cell.2)
        hcell
      have h3 : b.colHeight (c + 0) - 0
          ≤ (⟨Piece.I, 1, c⟩ : Placement).dropOffset b := hle
      rw [Nat.add_zero, hH] at h3
      omega
  have hdrop : (⟨Piece.I, 1, c⟩ : Placement).dropped b
      = ({(c, h), (c, h + 1), (c, h + 2), (c, h + 3)}
          : Finset Coord) := by
    unfold Placement.dropped Placement.cellsAt
    rw [hD]
    change (Piece.I.shapeUp (1 : Rotation)).image
        (fun cell => (c + cell.1, h + cell.2)) = _
    rw [I_r1_shape_eq]
    ext p
    simp only [Finset.mem_image, Finset.mem_insert, Finset.mem_singleton]
    constructor
    · rintro ⟨cell, hcell, rfl⟩
      rcases hcell with h' | h' | h' | h' <;> subst h' <;> simp
    · intro hp
      rcases hp with h' | h' | h' | h' <;> subst h'
      · exact ⟨(0, 0), by simp⟩
      · exact ⟨(0, 1), by simp⟩
      · exact ⟨(0, 2), by simp⟩
      · exact ⟨(0, 3), by simp⟩
  refine ⟨hdrop, ?_⟩
  rw [Placement.place_eq_union_dropped, hdrop]

/-- Extending a four-row column band by its next column. -/
theorem column_band_extend {m h : ℕ} :
    (Finset.range m) ×ˢ (Finset.Ico h (h + 4))
      ∪ ({(m, h), (m, h + 1), (m, h + 2), (m, h + 3)} : Finset Coord)
    = (Finset.range (m + 1)) ×ˢ (Finset.Ico h (h + 4)) := by
  ext p
  simp only [Finset.mem_union, Finset.mem_product, Finset.mem_range,
    Finset.mem_Ico, Finset.mem_insert, Finset.mem_singleton,
    Prod.ext_iff]
  constructor
  · rintro (⟨h1, h2⟩ | h' | h' | h' | h')
    · exact ⟨by omega, h2⟩
    all_goals (obtain ⟨h1, h2⟩ := h'; exact ⟨by omega, by omega, by omega⟩)
  · rintro ⟨h1, h2⟩
    by_cases hlt : p.1 < m
    · exact Or.inl ⟨hlt, h2⟩
    · have hm : p.1 = m := by omega
      have hp : p.2 = h ∨ p.2 = h + 1 ∨ p.2 = h + 2 ∨ p.2 = h + 3 := by
        omega
      rcases hp with hp | hp | hp | hp
      · exact Or.inr (Or.inl ⟨hm, hp⟩)
      · exact Or.inr (Or.inr (Or.inl ⟨hm, hp⟩))
      · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨hm, hp⟩)))
      · exact Or.inr (Or.inr (Or.inr (Or.inr ⟨hm, hp⟩)))

/-- A clear-free below-`h` board stays clear-free after any partial
`k`-row band over fewer than ten columns. -/
theorem no_full_of_partial_band_Ico {b : Board} {h m k : ℕ}
    (hm : m < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h) :
    ∀ r, ¬ Board.isFull GameConfig.standard
      (b ∪ (Finset.range m) ×ˢ (Finset.Ico h (h + k))) r := by
  intro r hfull
  have h9 := hfull 9 (by rw [GameConfig.standard_cols]; simp)
  rw [Finset.mem_union] at h9
  rcases h9 with hb | hX
  · have hrlt := hlow _ hb
    apply hnf r
    intro c hc
    have := hfull c hc
    rw [Finset.mem_union] at this
    rcases this with h' | h'
    · exact h'
    · exfalso
      rw [Finset.mem_product, Finset.mem_Ico] at h'
      omega
  · rw [Finset.mem_product, Finset.mem_range] at hX
    omega

/-- A dry step of the ten-I ritual: the vertical I at column `m`
extends the four-row band by one column, clearing nothing. -/
theorem ten_I_dry_step {b : Board} {h m : ℕ} (hm : m + 1 < 10)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hHm : b.colHeight m = h) :
    Placement.applyStep GameConfig.standard
      (b ∪ (Finset.range m) ×ˢ (Finset.Ico h (h + 4)))
      ⟨Piece.I, 1, m⟩
    = b ∪ (Finset.range (m + 1)) ×ˢ (Finset.Ico h (h + 4)) := by
  classical
  have hband : ∀ p ∈ (Finset.range m) ×ˢ (Finset.Ico h (h + 4)),
      p.1 < m := fun p hp =>
    Finset.mem_range.mp (Finset.mem_product.mp hp).1
  have h0 : (b ∪ (Finset.range m) ×ˢ
      (Finset.Ico h (h + 4))).colHeight m = h := by
    rw [colHeight_union_high_cols hband (le_refl m)]
    exact hHm
  obtain ⟨_, hplace⟩ := I_column_dropped_eq
    (b := b ∪ (Finset.range m) ×ˢ (Finset.Ico h (h + 4))) h0
  have hplace' : (⟨Piece.I, 1, m⟩ : Placement).place
      (b ∪ (Finset.range m) ×ˢ (Finset.Ico h (h + 4)))
      = b ∪ (Finset.range (m + 1)) ×ˢ (Finset.Ico h (h + 4)) := by
    rw [hplace, Finset.union_assoc, column_band_extend]
  have hnofull := no_full_of_partial_band_Ico (m := m + 1) (k := 4)
    (by omega) hnf hlow
  have hempty : Board.fullRows GameConfig.standard
      (b ∪ (Finset.range (m + 1)) ×ˢ (Finset.Ico h (h + 4))) = ∅ :=
    Finset.eq_empty_iff_forall_notMem.mpr (fun r hr =>
      hnofull r (Finset.mem_filter.mp hr).2)
  unfold Placement.applyStep
  rw [hplace']
  exact Board.clearLines_eq_self_of_no_fullRows GameConfig.standard hempty

/-- The final step of the ten-I ritual: the tenth vertical I completes
the four-row band, the quadruple clears, and the original board
returns. -/
theorem ten_I_final_step {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH9 : b.colHeight 9 = h) :
    Placement.applyStep GameConfig.standard
      (b ∪ (Finset.range 9) ×ˢ (Finset.Ico h (h + 4)))
      ⟨Piece.I, 1, 9⟩ = b := by
  classical
  have hband : ∀ p ∈ (Finset.range 9) ×ˢ (Finset.Ico h (h + 4)),
      p.1 < 9 := fun p hp =>
    Finset.mem_range.mp (Finset.mem_product.mp hp).1
  have h0 : (b ∪ (Finset.range 9) ×ˢ
      (Finset.Ico h (h + 4))).colHeight 9 = h := by
    rw [colHeight_union_high_cols hband (le_refl 9)]
    exact hH9
  obtain ⟨_, hplace⟩ := I_column_dropped_eq
    (b := b ∪ (Finset.range 9) ×ˢ (Finset.Ico h (h + 4))) h0
  have hplace' : (⟨Piece.I, 1, 9⟩ : Placement).place
      (b ∪ (Finset.range 9) ×ˢ (Finset.Ico h (h + 4)))
      = b ∪ (Finset.range 10) ×ˢ (Finset.Ico h (h + 4)) := by
    rw [hplace, Finset.union_assoc, column_band_extend]
  unfold Placement.applyStep
  rw [hplace']
  exact clearLines_band_reset hnf hlow

/-- **THE TEN-I CYCLE**: from any clear-free board standing level at
height `h` across all ten columns, dropping the vertical I into each
column in turn — ten moves, one tower per column — completes a four-row
band, clears a TETRIS on the last drop, and returns the board to
EXACTLY its starting state, cell for cell. The five-O cycle's tall
sibling: same board, same return, but the harvest is one quadruple
instead of five doubles. -/
theorem ten_I_cycle {b : Board} {h : ℕ}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hlow : ∀ p ∈ b, p.2 < h)
    (hH : ∀ c < 10, b.colHeight c = h) :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard
              (Placement.applyStep GameConfig.standard
                (Placement.applyStep GameConfig.standard
                  (Placement.applyStep GameConfig.standard
                    (Placement.applyStep GameConfig.standard
                      (Placement.applyStep GameConfig.standard b
                        ⟨Piece.I, 1, 0⟩) ⟨Piece.I, 1, 1⟩)
                    ⟨Piece.I, 1, 2⟩) ⟨Piece.I, 1, 3⟩)
                ⟨Piece.I, 1, 4⟩) ⟨Piece.I, 1, 5⟩)
            ⟨Piece.I, 1, 6⟩) ⟨Piece.I, 1, 7⟩)
        ⟨Piece.I, 1, 8⟩) ⟨Piece.I, 1, 9⟩ = b := by
  have e0 : Placement.applyStep GameConfig.standard b ⟨Piece.I, 1, 0⟩
      = b ∪ (Finset.range 1) ×ˢ (Finset.Ico h (h + 4)) := by
    have h' := ten_I_dry_step (b := b) (m := 0) (by omega) hnf hlow
      (hH 0 (by omega))
    simpa using h'
  have e1 := ten_I_dry_step (b := b) (m := 1) (by omega) hnf hlow
    (hH 1 (by omega))
  have e2 := ten_I_dry_step (b := b) (m := 2) (by omega) hnf hlow
    (hH 2 (by omega))
  have e3 := ten_I_dry_step (b := b) (m := 3) (by omega) hnf hlow
    (hH 3 (by omega))
  have e4 := ten_I_dry_step (b := b) (m := 4) (by omega) hnf hlow
    (hH 4 (by omega))
  have e5 := ten_I_dry_step (b := b) (m := 5) (by omega) hnf hlow
    (hH 5 (by omega))
  have e6 := ten_I_dry_step (b := b) (m := 6) (by omega) hnf hlow
    (hH 6 (by omega))
  have e7 := ten_I_dry_step (b := b) (m := 7) (by omega) hnf hlow
    (hH 7 (by omega))
  have e8 := ten_I_dry_step (b := b) (m := 8) (by omega) hnf hlow
    (hH 8 (by omega))
  have e9 := ten_I_final_step (b := b) hnf hlow (hH 9 (by omega))
  rw [show (1 : ℕ) + 1 = 2 from rfl] at e1
  rw [show (2 : ℕ) + 1 = 3 from rfl] at e2
  rw [show (3 : ℕ) + 1 = 4 from rfl] at e3
  rw [show (4 : ℕ) + 1 = 5 from rfl] at e4
  rw [show (5 : ℕ) + 1 = 6 from rfl] at e5
  rw [show (6 : ℕ) + 1 = 7 from rfl] at e6
  rw [show (7 : ℕ) + 1 = 8 from rfl] at e7
  rw [show (8 : ℕ) + 1 = 9 from rfl] at e8
  rw [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]

/-- **The empty board rides the ten-I loop too**: ten vertical I's, one
per column, take the empty board back to the empty board — the ground
floor rises four rows and a tetris reaps it whole. The empty board thus
sits on (at least) TWO distinct closed cycles: the five-O double-mill
and the ten-I tetris-mill. -/
theorem ten_I_cycle_empty :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard
              (Placement.applyStep GameConfig.standard
                (Placement.applyStep GameConfig.standard
                  (Placement.applyStep GameConfig.standard
                    (Placement.applyStep GameConfig.standard
                      (Placement.applyStep GameConfig.standard
                        (∅ : Board) ⟨Piece.I, 1, 0⟩) ⟨Piece.I, 1, 1⟩)
                    ⟨Piece.I, 1, 2⟩) ⟨Piece.I, 1, 3⟩)
                ⟨Piece.I, 1, 4⟩) ⟨Piece.I, 1, 5⟩)
            ⟨Piece.I, 1, 6⟩) ⟨Piece.I, 1, 7⟩)
        ⟨Piece.I, 1, 8⟩) ⟨Piece.I, 1, 9⟩ = (∅ : Board) := by
  apply ten_I_cycle (h := 0)
  · intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  · intro p hp
    exact absurd hp (Finset.notMem_empty _)
  · intro c _
    exact Board.colHeight_empty c

/-- On a well-formed board, being level at height `h` already forces
every cell below `h`: the lowness hypothesis of the mill cycles is FREE
for in-field boards. -/
theorem low_of_level_wf {b : Board} {h : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hH : ∀ c < 10, b.colHeight c = h) :
    ∀ p ∈ b, p.2 < h := by
  intro p hp
  have hc : p.1 < 10 := by
    have := hwf p hp
    rwa [GameConfig.standard_cols] at this
  have hmem : (p.1, p.2) ∈ b := hp
  have := Board.lt_colHeight hmem
  rw [hH p.1 hc] at this
  exact this

/-- The five-O cycle for well-formed boards: clear-free + level is
enough — lowness comes free from `lt_colHeight`. -/
theorem five_O_cycle_of_level {b : Board} {h : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hH : ∀ c < 10, b.colHeight c = h) :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard b
              ⟨Piece.O, 0, 0⟩) ⟨Piece.O, 0, 2⟩) ⟨Piece.O, 0, 4⟩)
          ⟨Piece.O, 0, 6⟩) ⟨Piece.O, 0, 8⟩ = b :=
  five_O_cycle hnf (low_of_level_wf hwf hH) hH

/-- The ten-I cycle for well-formed boards: clear-free + level is
enough. -/
theorem ten_I_cycle_of_level {b : Board} {h : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hH : ∀ c < 10, b.colHeight c = h) :
    Placement.applyStep GameConfig.standard
      (Placement.applyStep GameConfig.standard
        (Placement.applyStep GameConfig.standard
          (Placement.applyStep GameConfig.standard
            (Placement.applyStep GameConfig.standard
              (Placement.applyStep GameConfig.standard
                (Placement.applyStep GameConfig.standard
                  (Placement.applyStep GameConfig.standard
                    (Placement.applyStep GameConfig.standard
                      (Placement.applyStep GameConfig.standard b
                        ⟨Piece.I, 1, 0⟩) ⟨Piece.I, 1, 1⟩)
                    ⟨Piece.I, 1, 2⟩) ⟨Piece.I, 1, 3⟩)
                ⟨Piece.I, 1, 4⟩) ⟨Piece.I, 1, 5⟩)
            ⟨Piece.I, 1, 6⟩) ⟨Piece.I, 1, 7⟩)
        ⟨Piece.I, 1, 8⟩) ⟨Piece.I, 1, 9⟩ = b :=
  ten_I_cycle hnf (low_of_level_wf hwf hH) hH

/-- A board sits on a closed cycle of length `n` when some nonempty
placement word of that length folds back to it. -/
def BoardOnCycle (b : Board) (n : ℕ) : Prop :=
  ∃ pls : List Placement, pls.length = n ∧ 0 < n ∧
    (∀ pl ∈ pls, pl.Valid GameConfig.standard) ∧
    pls.foldl (Placement.applyStep GameConfig.standard) b = b

/-- **Every clear-free level board sits on a five-cycle**: the five-O
word witnesses `BoardOnCycle b 5` for every well-formed, clear-free,
level board. Closed cycles are not exotic — they are EVERYWHERE on the
level stratum. -/
theorem level_board_on_five_cycle {b : Board} {h : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hH : ∀ c < 10, b.colHeight c = h) :
    BoardOnCycle b 5 := by
  refine ⟨[⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
    ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩], rfl, by omega, by decide, ?_⟩
  simp only [List.foldl]
  exact five_O_cycle_of_level hwf hnf hH

/-- **…and on a ten-cycle**: the ten-I word witnesses
`BoardOnCycle b 10` on the same stratum. -/
theorem level_board_on_ten_cycle {b : Board} {h : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hH : ∀ c < 10, b.colHeight c = h) :
    BoardOnCycle b 10 := by
  refine ⟨[⟨Piece.I, 1, 0⟩, ⟨Piece.I, 1, 1⟩, ⟨Piece.I, 1, 2⟩,
    ⟨Piece.I, 1, 3⟩, ⟨Piece.I, 1, 4⟩, ⟨Piece.I, 1, 5⟩,
    ⟨Piece.I, 1, 6⟩, ⟨Piece.I, 1, 7⟩, ⟨Piece.I, 1, 8⟩,
    ⟨Piece.I, 1, 9⟩], rfl, by omega, by decide, ?_⟩
  simp only [List.foldl]
  exact ten_I_cycle_of_level hwf hnf hH

/-- **The empty board sits on both mills**: `BoardOnCycle ∅ 5` and
`BoardOnCycle ∅ 10` — two distinct closed cycles through the initial
board, in one statement. -/
theorem empty_board_on_two_cycles :
    BoardOnCycle (∅ : Board) 5 ∧ BoardOnCycle (∅ : Board) 10 := by
  have hwf : Board.WF GameConfig.standard (∅ : Board) :=
    Board.empty_wf GameConfig.standard
  have hnf : ∀ r, ¬ Board.isFull GameConfig.standard (∅ : Board) r := by
    intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  have hH : ∀ c < 10, (∅ : Board).colHeight c = 0 := fun c _ =>
    Board.colHeight_empty c
  exact ⟨level_board_on_five_cycle hwf hnf hH,
    level_board_on_ten_cycle hwf hnf hH⟩

/-- Well-formedness survives any valid placement word. -/
theorem foldl_applyStep_wf {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    Board.WF GameConfig.standard
      (pls.foldl (Placement.applyStep GameConfig.standard) b) := by
  induction pls generalizing b with
  | nil => exact hwf
  | cons pl rest ih =>
    rw [List.foldl_cons]
    exact ih (Placement.applyStep_wf hwf (hv pl (by simp)))
      (fun q hq => hv q (by simp [hq]))

/-- **The word ledger**: along any valid placement word, the final count
plus ten per cleared row equals the initial count plus four per move.
The per-step mass law, folded. -/
theorem foldl_applyStep_count_ledger {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    ∃ C, (pls.foldl (Placement.applyStep GameConfig.standard) b).count
        + 10 * C = b.count + 4 * pls.length := by
  induction pls generalizing b with
  | nil => exact ⟨0, by simp⟩
  | cons pl rest ih =>
    have hstep := BagGrowth.count_applyStep_add
      (cfg := GameConfig.standard) hwf (hv pl (by simp))
    rw [GameConfig.standard_cols] at hstep
    obtain ⟨C, hC⟩ := ih (Placement.applyStep_wf hwf (hv pl (by simp)))
      (fun q hq => hv q (by simp [hq]))
    refine ⟨(Board.fullRows GameConfig.standard (pl.place b)).card
      + C, ?_⟩
    rw [List.foldl_cons]
    simp only [List.length_cons]
    omega

/-- **THE BOARD-CYCLE QUANTUM**: every closed board cycle has length
divisible by five. Four cells in per move, ten out per cleared row, and
a return to the same count forces `10 C = 4 n` — pure mass arithmetic,
no geometry needed. -/
theorem board_cycle_length_quantum {b : Board} {n : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hcyc : BoardOnCycle b n) : 5 ∣ n := by
  obtain ⟨pls, hlen, hpos, hv, hfold⟩ := hcyc
  obtain ⟨C, hC⟩ := foldl_applyStep_count_ledger hwf hv
  rw [hfold, hlen] at hC
  omega

/-- Cycles through the same board concatenate. -/
theorem BoardOnCycle.add {b : Board} {n m : ℕ}
    (hn : BoardOnCycle b n) (hm : BoardOnCycle b m) :
    BoardOnCycle b (n + m) := by
  obtain ⟨p1, hl1, hp1, hv1, hf1⟩ := hn
  obtain ⟨p2, hl2, hp2, hv2, hf2⟩ := hm
  refine ⟨p1 ++ p2, by simp [hl1, hl2], by omega, ?_, ?_⟩
  · intro pl hpl
    rw [List.mem_append] at hpl
    rcases hpl with hh | hh
    · exact hv1 pl hh
    · exact hv2 pl hh
  · rw [List.foldl_append, hf1, hf2]

/-- **THE COMPLETE CYCLE-LENGTH SPECTRUM**: on any well-formed,
clear-free, level board, closed cycles exist in EXACTLY the lengths
`5, 10, 15, 20, …` — the positive multiples of five. Existence from
iterating the five-O mill; exclusion from the mass quantum. The cycle
structure of the level stratum is completely characterized. -/
theorem level_board_cycle_lengths {b : Board} {h n : ℕ}
    (hwf : Board.WF GameConfig.standard b)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r)
    (hH : ∀ c < 10, b.colHeight c = h) :
    BoardOnCycle b n ↔ (0 < n ∧ 5 ∣ n) := by
  have haux : ∀ k : ℕ, BoardOnCycle b (5 * (k + 1)) := by
    intro k
    induction k with
    | zero =>
      rw [show 5 * (0 + 1) = 5 from by norm_num]
      exact level_board_on_five_cycle hwf hnf hH
    | succ j ihj =>
      have hcomp := BoardOnCycle.add ihj
        (level_board_on_five_cycle hwf hnf hH)
      rwa [show 5 * (j + 1) + 5 = 5 * (j + 1 + 1) from by ring] at hcomp
  constructor
  · intro hcyc
    obtain ⟨pls, hlen, hpos, hv, hfold⟩ := hcyc
    exact ⟨hpos, board_cycle_length_quantum hwf
      ⟨pls, hlen, hpos, hv, hfold⟩⟩
  · rintro ⟨hpos, ⟨k, rfl⟩⟩
    rcases k with _ | k'
    · omega
    · exact haux k'

/-- The empty board's cycle spectrum: exactly the positive multiples of
five. -/
theorem empty_board_cycle_lengths (n : ℕ) :
    BoardOnCycle (∅ : Board) n ↔ (0 < n ∧ 5 ∣ n) := by
  apply level_board_cycle_lengths (h := 0)
  · exact Board.empty_wf GameConfig.standard
  · intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  · intro c _
    exact Board.colHeight_empty c

/-- The gravity shift of a row is at most the row itself. -/
theorem clearedBelow_le (cfg : GameConfig) (b : Board) (r : ℕ) :
    Board.clearedBelow cfg b r ≤ r := by
  unfold Board.clearedBelow
  calc ((Board.fullRows cfg b).filter (· < r)).card
      ≤ (Finset.range r).card := Finset.card_le_card (by
        intro x hx
        rw [Finset.mem_filter] at hx
        exact Finset.mem_range.mpr hx.2)
    _ = r := Finset.card_range r

/-- **Shifted rows keep their order**: if `r1 < r2` and `r1` survives
the clear, the post-gravity positions stay strictly ordered — the count
of full rows strictly between them is short of the gap by at least the
surviving `r1` itself. -/
theorem clearedBelow_shift_strictMono {cfg : GameConfig} {b : Board}
    {r1 r2 : ℕ} (h12 : r1 < r2)
    (hnf : ¬ Board.isFull cfg b r1) :
    r1 - Board.clearedBelow cfg b r1
      < r2 - Board.clearedBelow cfg b r2 := by
  have hle1 := clearedBelow_le cfg b r1
  have hsub : (Board.fullRows cfg b).filter (· < r2)
      ⊆ ((Board.fullRows cfg b).filter (· < r1))
        ∪ (Finset.Ico (r1 + 1) r2) := by
    intro x hx
    rw [Finset.mem_filter] at hx
    obtain ⟨hxf, hxlt⟩ := hx
    rw [Finset.mem_union, Finset.mem_filter, Finset.mem_Ico]
    by_cases hlt : x < r1
    · exact Or.inl ⟨hxf, hlt⟩
    · right
      have hxne : x ≠ r1 := by
        intro he
        subst he
        exact hnf ((Finset.mem_filter.mp hxf).2)
      omega
  have hcard := Finset.card_le_card hsub
  have hcard2 := Finset.card_union_le
    ((Board.fullRows cfg b).filter (· < r1)) (Finset.Ico (r1 + 1) r2)
  rw [Nat.card_Ico] at hcard2
  unfold Board.clearedBelow at hle1 ⊢
  omega

/-- **Clearing leaves no full rows**: whatever the board, every row of
`clearLines` is missing a cell. Distinct surviving source rows land on
distinct target rows (`clearedBelow_shift_strictMono`), so a full
target row would force a single full source row — which would have been
cleared. -/
theorem clearLines_no_fullRows {b : Board} :
    ∀ r, ¬ Board.isFull GameConfig.standard
      (Board.clearLines GameConfig.standard b) r := by
  intro r hfull
  have hsrc : ∀ c, c < 10 → ∃ ρ, (((c, ρ) : Coord) ∈ b
      ∧ ¬ Board.isFull GameConfig.standard b ρ)
      ∧ ρ - Board.clearedBelow GameConfig.standard b ρ = r := by
    intro c hc
    have hmem := hfull c
      (by rw [GameConfig.standard_cols]; exact Finset.mem_range.mpr hc)
    rw [Board.mem_clearLines_iff] at hmem
    obtain ⟨q, hq, hqnf, hqp⟩ := hmem
    have h1 : q.1 = c := congrArg Prod.fst hqp
    have h2 : q.2 - Board.clearedBelow GameConfig.standard b q.2 = r :=
      congrArg Prod.snd hqp
    refine ⟨q.2, ⟨?_, hqnf⟩, h2⟩
    rw [← h1]
    exact hq
  obtain ⟨ρ0, ⟨hmem0, hnf0⟩, hsh0⟩ := hsrc 0 (by omega)
  apply hnf0
  intro c hc
  rw [GameConfig.standard_cols] at hc
  obtain ⟨ρc, ⟨hmemc, hnfc⟩, hshc⟩ := hsrc c (Finset.mem_range.mp hc)
  have heq : ρc = ρ0 := by
    rcases lt_trichotomy ρc ρ0 with hlt | heq | hgt
    · have := clearedBelow_shift_strictMono
        (cfg := GameConfig.standard) (b := b) hlt hnfc
      omega
    · exact heq
    · have := clearedBelow_shift_strictMono
        (cfg := GameConfig.standard) (b := b) hgt hnf0
      omega
  rw [← heq]
  exact hmemc

/-- Every post-move board is clear-free: `applyStep` ends in
`clearLines`. -/
theorem applyStep_clear_free (b : Board) (pl : Placement) :
    ∀ r, ¬ Board.isFull GameConfig.standard
      (Placement.applyStep GameConfig.standard b pl) r := by
  unfold Placement.applyStep
  exact clearLines_no_fullRows

/-- **Cycling boards are clear-free**: a board that sits on any closed
cycle carries no full row — it is the image of a move, and moves always
sweep. The mills' clear-freeness hypothesis is NECESSARY, not just
convenient. -/
theorem board_on_cycle_clear_free {b : Board} {n : ℕ}
    (hcyc : BoardOnCycle b n) :
    ∀ r, ¬ Board.isFull GameConfig.standard b r := by
  obtain ⟨pls, hlen, hpos, hv, hfold⟩ := hcyc
  rcases pls.eq_nil_or_concat with rfl | ⟨ys, y, rfl⟩
  · rw [List.length_nil] at hlen
    omega
  · simp only [List.concat_eq_append, List.foldl_append,
      List.foldl] at hfold
    rw [← hfold]
    exact applyStep_clear_free _ _

/-- **Cycles rotate**: every mid-cycle board is itself on a closed
cycle — ride the rest of the word, then the prefix. The set of cycling
boards is closed under cycle steps. -/
theorem board_on_cycle_shift {b : Board} {w1 w2 : List Placement}
    (hpos : 0 < w1.length + w2.length)
    (hv : ∀ pl ∈ w1 ++ w2, pl.Valid GameConfig.standard)
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b) :
    BoardOnCycle
      (w1.foldl (Placement.applyStep GameConfig.standard) b)
      (w1.length + w2.length) := by
  refine ⟨w2 ++ w1, by simp [Nat.add_comm], hpos, ?_, ?_⟩
  · intro pl hpl
    rw [List.mem_append] at hpl
    apply hv
    rw [List.mem_append]
    tauto
  · rw [List.foldl_append]
    rw [List.foldl_append] at hfold
    rw [hfold]

/-- Cycles iterate: riding a closed cycle `k + 1` times is a closed
cycle. -/
theorem BoardOnCycle.iterate {b : Board} {n : ℕ}
    (hn : BoardOnCycle b n) (k : ℕ) :
    BoardOnCycle b ((k + 1) * n) := by
  induction k with
  | zero => simpa using hn
  | succ j ihj =>
    have := ihj.add hn
    rwa [show (j + 1) * n + n = (j + 1 + 1) * n from by ring] at this

/-- **A non-level board on a closed cycle**: the two-column plinth
`{0,1} × {rows 0,1}` — columns 0 and 1 standing at height two, the
rest empty — rides the rotated five-O word back to itself. Closed
cycles are NOT confined to the level stratum. -/
theorem two_col_band_on_cycle :
    BoardOnCycle
      ((Finset.range 2) ×ˢ ({0, 1} : Finset ℕ) : Board) 5 := by
  have hnf : ∀ r, ¬ Board.isFull GameConfig.standard (∅ : Board) r := by
    intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  have hlow : ∀ p ∈ (∅ : Board), p.2 < 0 := by
    intro p hp
    exact absurd hp (Finset.notMem_empty _)
  have hH : ∀ c < 10, (∅ : Board).colHeight c = 0 := fun c _ =>
    Board.colHeight_empty c
  obtain ⟨e0, -, -, -⟩ := five_O_intermediate_boards hnf hlow hH
  have h1 : ([⟨Piece.O, 0, 0⟩] : List Placement).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board)
      = (Finset.range 2) ×ˢ ({0, 1} : Finset ℕ) := by
    simp only [List.foldl]
    rw [e0, Finset.empty_union]
  have hfold5 : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
      ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩] : List Placement).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board) = ∅ := by
    simp only [List.foldl]
    exact five_O_cycle_empty
  have hshift := board_on_cycle_shift
    (b := (∅ : Board))
    (w1 := [⟨Piece.O, 0, 0⟩])
    (w2 := [⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩, ⟨Piece.O, 0, 6⟩,
      ⟨Piece.O, 0, 8⟩])
    (by simp) (by decide)
    (by simp only [List.cons_append, List.nil_append]; exact hfold5)
  rw [h1] at hshift
  simpa using hshift

/-- **The plinth tour**: EVERY intermediate board of the five-O ritual
— the partial bands of width 0, 2, 4, 6, 8 over rows 0–1 — sits on a
closed five-cycle, by riding the rotated word. The ritual's whole orbit
is cyclic, not just its level anchor. -/
theorem five_O_ritual_boards_on_cycle (k : ℕ) (hk : k ≤ 4) :
    BoardOnCycle
      ((Finset.range (2 * k)) ×ˢ ({0, 1} : Finset ℕ) : Board) 5 := by
  have hnf : ∀ r, ¬ Board.isFull GameConfig.standard (∅ : Board) r := by
    intro r hfull
    have h0 := hfull 0 (by rw [GameConfig.standard_cols]; simp)
    exact absurd h0 (Finset.notMem_empty _)
  have hlow : ∀ p ∈ (∅ : Board), p.2 < 0 := by
    intro p hp
    exact absurd hp (Finset.notMem_empty _)
  have hH : ∀ c < 10, (∅ : Board).colHeight c = 0 := fun c _ =>
    Board.colHeight_empty c
  obtain ⟨e0, e1, e2, e3⟩ := five_O_intermediate_boards hnf hlow hH
  have hfold5 : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
      ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩] : List Placement).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board) = ∅ := by
    simp only [List.foldl]
    exact five_O_cycle_empty
  interval_cases k
  · rw [show (Finset.range (2 * 0)) ×ˢ ({0, 1} : Finset ℕ)
      = (∅ : Board) from by simp]
    exact empty_board_on_two_cycles.1
  · have h1 : ([⟨Piece.O, 0, 0⟩] : List Placement).foldl
        (Placement.applyStep GameConfig.standard) (∅ : Board)
        = (Finset.range (2 * 1)) ×ˢ ({0, 1} : Finset ℕ) := by
      simp only [List.foldl]
      rw [e0, Finset.empty_union]
    have hshift := board_on_cycle_shift
      (b := (∅ : Board))
      (w1 := [⟨Piece.O, 0, 0⟩])
      (w2 := [⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩, ⟨Piece.O, 0, 6⟩,
        ⟨Piece.O, 0, 8⟩])
      (by simp) (by decide)
      (by simp only [List.cons_append, List.nil_append]; exact hfold5)
    rw [h1] at hshift
    simpa using hshift
  · have h1 : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩] : List Placement).foldl
        (Placement.applyStep GameConfig.standard) (∅ : Board)
        = (Finset.range (2 * 2)) ×ˢ ({0, 1} : Finset ℕ) := by
      simp only [List.foldl]
      rw [e0, e1, Finset.empty_union]
    have hshift := board_on_cycle_shift
      (b := (∅ : Board))
      (w1 := [⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩])
      (w2 := [⟨Piece.O, 0, 4⟩, ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩])
      (by simp) (by decide)
      (by simp only [List.cons_append, List.nil_append]; exact hfold5)
    rw [h1] at hshift
    simpa using hshift
  · have h1 : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩,
        ⟨Piece.O, 0, 4⟩] : List Placement).foldl
        (Placement.applyStep GameConfig.standard) (∅ : Board)
        = (Finset.range (2 * 3)) ×ˢ ({0, 1} : Finset ℕ) := by
      simp only [List.foldl]
      rw [e0, e1, e2, Finset.empty_union]
    have hshift := board_on_cycle_shift
      (b := (∅ : Board))
      (w1 := [⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩])
      (w2 := [⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩])
      (by simp) (by decide)
      (by simp only [List.cons_append, List.nil_append]; exact hfold5)
    rw [h1] at hshift
    simpa using hshift
  · have h1 : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
        ⟨Piece.O, 0, 6⟩] : List Placement).foldl
        (Placement.applyStep GameConfig.standard) (∅ : Board)
        = (Finset.range (2 * 4)) ×ˢ ({0, 1} : Finset ℕ) := by
      simp only [List.foldl]
      rw [e0, e1, e2, e3, Finset.empty_union]
    have hshift := board_on_cycle_shift
      (b := (∅ : Board))
      (w1 := [⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
        ⟨Piece.O, 0, 6⟩])
      (w2 := [⟨Piece.O, 0, 8⟩])
      (by simp) (by decide)
      (by simp only [List.cons_append, List.nil_append]; exact hfold5)
    rw [h1] at hshift
    simpa using hshift

/-- A piece stream deals full bags: every seven-block contains every
piece. -/
def IsBagStream (s : ℕ → Piece) : Prop :=
  ∀ j : ℕ, ∀ p : Piece, ∃ i < 7, s (7 * j + i) = p

/-- There are exactly seven pieces. -/
theorem piece_univ_card : (Finset.univ : Finset Piece).card = 7 := by
  decide

/-- **Each block deals each piece exactly once**: seven slots, seven
pieces, each present — pigeonhole forces multiplicity one. -/
theorem bag_block_count {s : ℕ → Piece} (hs : IsBagStream s) (j : ℕ)
    (p : Piece) :
    ((Finset.Ico (7 * j) (7 * j + 7)).filter (fun i => s i = p)).card
      = 1 := by
  classical
  have hsum : ∑ q : Piece, ((Finset.Ico (7 * j) (7 * j + 7)).filter
      (fun i => s i = q)).card = 7 := by
    rw [← Finset.card_eq_sum_card_fiberwise (f := s)
      (s := Finset.Ico (7 * j) (7 * j + 7)) (t := Finset.univ)
      (fun x _ => Finset.mem_univ (s x))]
    rw [Nat.card_Ico]
    omega
  have hpos : ∀ q : Piece,
      1 ≤ ((Finset.Ico (7 * j) (7 * j + 7)).filter
        (fun i => s i = q)).card := by
    intro q
    obtain ⟨i, hi, hsi⟩ := hs j q
    apply Finset.card_pos.mpr
    exact ⟨7 * j + i, Finset.mem_filter.mpr
      ⟨Finset.mem_Ico.mpr ⟨by omega, by omega⟩, hsi⟩⟩
  by_contra hne
  have h2 : 2 ≤ ((Finset.Ico (7 * j) (7 * j + 7)).filter
      (fun i => s i = p)).card := by
    have := hpos p
    omega
  have hsplit : ∑ q : Piece, ((Finset.Ico (7 * j) (7 * j + 7)).filter
      (fun i => s i = q)).card
      = (∑ q ∈ Finset.univ.erase p,
          ((Finset.Ico (7 * j) (7 * j + 7)).filter
            (fun i => s i = q)).card)
        + ((Finset.Ico (7 * j) (7 * j + 7)).filter
            (fun i => s i = p)).card :=
    (Finset.sum_eq_sum_diff_singleton_add (Finset.mem_univ p) _).trans
      (by rw [Finset.sdiff_singleton_eq_erase])
  have hrest : 6 ≤ ∑ q ∈ Finset.univ.erase p,
      ((Finset.Ico (7 * j) (7 * j + 7)).filter
        (fun i => s i = q)).card := by
    calc (6 : ℕ)
        = ∑ _q ∈ Finset.univ.erase p, 1 := by
          rw [Finset.sum_const, smul_eq_mul, mul_one,
            Finset.card_erase_of_mem (Finset.mem_univ p),
            piece_univ_card]
      _ ≤ _ := Finset.sum_le_sum (fun q _ => hpos q)
  omega

/-- **A bag stream deals each piece `n` times in `7n` moves.** -/
theorem bag_stream_range_count {s : ℕ → Piece} (hs : IsBagStream s)
    (p : Piece) (n : ℕ) :
    ((Finset.range (7 * n)).filter (fun i => s i = p)).card = n := by
  classical
  induction n with
  | zero => simp
  | succ k ih =>
    have hsplit : Finset.range (7 * (k + 1))
        = Finset.range (7 * k) ∪ Finset.Ico (7 * k) (7 * k + 7) := by
      rw [Finset.range_eq_Ico,
        Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
      congr 1
    have hdisj : Disjoint
        ((Finset.range (7 * k)).filter (fun i => s i = p))
        ((Finset.Ico (7 * k) (7 * k + 7)).filter (fun i => s i = p)) := by
      apply Finset.disjoint_left.mpr
      intro a ha hb
      rw [Finset.mem_filter, Finset.mem_range] at ha
      rw [Finset.mem_filter, Finset.mem_Ico] at hb
      omega
    rw [hsplit, Finset.filter_union,
      Finset.card_union_of_disjoint hdisj, ih, bag_block_count hs]

/-- Iterated periodicity. -/
theorem periodic_add_mul {s : ℕ → Piece} {n : ℕ}
    (hper : ∀ i, s (i + n) = s i) :
    ∀ k i, s (i + k * n) = s i := by
  intro k
  induction k with
  | zero => intro i; simp
  | succ j ih =>
    intro i
    rw [show i + (j + 1) * n = (i + j * n) + n from by ring, hper, ih]

/-- **Periodic streams count by blocks**: over `m` periods, each piece
appears `m` times its per-period count. -/
theorem periodic_count_mul {s : ℕ → Piece} {n : ℕ} (hn : 0 < n)
    (hper : ∀ i, s (i + n) = s i) (p : Piece) (m : ℕ) :
    ((Finset.range (m * n)).filter (fun i => s i = p)).card
      = m * ((Finset.range n).filter (fun i => s i = p)).card := by
  classical
  induction m with
  | zero => simp
  | succ k ih =>
    have hsplit : Finset.range ((k + 1) * n)
        = Finset.range (k * n) ∪ Finset.Ico (k * n) (k * n + n) := by
      rw [Finset.range_eq_Ico,
        Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
      congr 1
      ring
    have hdisj : Disjoint
        ((Finset.range (k * n)).filter (fun i => s i = p))
        ((Finset.Ico (k * n) (k * n + n)).filter (fun i => s i = p)) := by
      apply Finset.disjoint_left.mpr
      intro a ha hb
      rw [Finset.mem_filter, Finset.mem_range] at ha
      rw [Finset.mem_filter, Finset.mem_Ico] at hb
      omega
    have hblock : (Finset.Ico (k * n) (k * n + n)).filter
        (fun i => s i = p)
        = ((Finset.range n).filter (fun i => s i = p)).image
            (· + k * n) := by
      ext x
      simp only [Finset.mem_filter, Finset.mem_Ico, Finset.mem_image,
        Finset.mem_range]
      constructor
      · rintro ⟨⟨h1, h2⟩, h3⟩
        refine ⟨x - k * n, ⟨by omega, ?_⟩, by omega⟩
        have hx : x - k * n + k * n = x := by omega
        rw [← hx] at h3
        rwa [periodic_add_mul hper] at h3
      · rintro ⟨y, ⟨hy, hsy⟩, rfl⟩
        refine ⟨⟨by omega, by omega⟩, ?_⟩
        rwa [periodic_add_mul hper]
    rw [hsplit, Finset.filter_union,
      Finset.card_union_of_disjoint hdisj, ih, hblock,
      Finset.card_image_of_injective _ (add_left_injective (k * n))]
    ring

/-- **THE BAG'S HEARTBEAT CANNOT BE COMPRESSED**: any periodic piece
stream that deals full bags has period divisible by seven. Count one
piece two ways over seven periods: the bags say `n` occurrences, the
periodicity says `7` per-period counts — so `n = 7c`. -/
theorem bag_stream_period_seven_dvd {s : ℕ → Piece} {n : ℕ}
    (hn : 0 < n) (hper : ∀ i, s (i + n) = s i)
    (hs : IsBagStream s) : 7 ∣ n := by
  have hA := bag_stream_range_count hs Piece.O n
  have hB := periodic_count_mul hn hper Piece.O 7
  omega

/-- The piece stream obtained by repeating a placement word forever. -/
def wordStream (w : List Placement) : ℕ → Piece :=
  fun i => (w.getD (i % w.length) ⟨Piece.O, 0, 0⟩).piece

/-- The repeated word's stream has the word's length as a period. -/
theorem wordStream_periodic (w : List Placement) :
    ∀ i, wordStream w (i + w.length) = wordStream w i := by
  intro i
  unfold wordStream
  rw [Nat.add_mod_right]

/-- **THE 35-QUANTUM FOR LEGAL CYCLES**: a placement word that folds a
well-formed board back to itself AND whose infinite repetition deals
full bags has length divisible by thirty-five. Five from the mass
ledger (`10 C = 4 n`), seven from the bag heartbeat — the two clocks
are coprime, so they multiply. The M2 object, if it exists, lives in
lengths 35, 70, 105, … -/
theorem legal_cycle_word_thirty_five_dvd {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    35 ∣ w.length := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h5 : 5 ∣ w.length := board_cycle_length_quantum hwf
    ⟨w, rfl, hpos, hv, hfold⟩
  have h7 : 7 ∣ w.length :=
    bag_stream_period_seven_dvd hpos (wordStream_periodic w) hbag
  omega

/-- **Legal cycles are long**: any bag-legal repeatable cycle word has
at least thirty-five moves. The five-O and ten-I mills (lengths 5 and
10) are thus PROVABLY illegal under the seven-bag — no dealer will
ever hand you their pieces. -/
theorem legal_cycle_word_min_length {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    35 ≤ w.length := by
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  omega

/-- Inside the first period, the repeated stream reads the word
itself. -/
theorem wordStream_eq_of_lt {w : List Placement} {i : ℕ}
    (hi : i < w.length) :
    wordStream w i = (w.getD i ⟨Piece.O, 0, 0⟩).piece := by
  unfold wordStream
  rw [Nat.mod_eq_of_lt hi]

/-- **The bag census of a word**: if a word's infinite repetition deals
full bags and its length is a multiple of seven, then the word contains
each piece EXACTLY one seventh of its length many times. -/
theorem bag_word_piece_census {w : List Placement}
    (hbag : IsBagStream (wordStream w)) (h7 : 7 ∣ w.length)
    (p : Piece) :
    ((Finset.range w.length).filter
      (fun i => (w.getD i ⟨Piece.O, 0, 0⟩).piece = p)).card
      = w.length / 7 := by
  classical
  obtain ⟨k, hk⟩ := h7
  have hcount := bag_stream_range_count hbag p k
  rw [← hk] at hcount
  have hconv : (Finset.range w.length).filter
      (fun i => (w.getD i ⟨Piece.O, 0, 0⟩).piece = p)
      = (Finset.range w.length).filter (fun i => wordStream w i = p) := by
    apply Finset.filter_congr
    intro i hi
    rw [Finset.mem_range] at hi
    rw [wordStream_eq_of_lt hi]
  rw [hconv, hcount, hk]
  omega

/-- **THE LEGAL CYCLE'S PIECE CENSUS**: any bag-legal repeatable cycle
word on a well-formed board contains each of the seven pieces exactly
`length / 7` times — a 35-word deals each piece exactly five times.
The mill structure (five services per period) is forced by arithmetic
before any geometry is considered. -/
theorem legal_cycle_word_piece_census {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (p : Piece) :
    ((Finset.range w.length).filter
      (fun i => (w.getD i ⟨Piece.O, 0, 0⟩).piece = p)).card
      = w.length / 7 := by
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  exact bag_word_piece_census hbag (by omega) p

/-- Total rows cleared while playing a word from a board. -/
def wordClears (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (Board.fullRows GameConfig.standard (pl.place b)).card
        + wordClears (Placement.applyStep GameConfig.standard b pl) rest

@[simp] theorem wordClears_nil (b : Board) : wordClears b [] = 0 := rfl

theorem wordClears_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordClears b (pl :: rest)
      = (Board.fullRows GameConfig.standard (pl.place b)).card
        + wordClears (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- **The exact word ledger**: final count plus ten per cleared row
equals initial count plus four per move — with the clear total NAMED,
not existential. -/
theorem foldl_count_ledger_exact {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    (pls.foldl (Placement.applyStep GameConfig.standard) b).count
      + 10 * wordClears b pls = b.count + 4 * pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hstep := BagGrowth.count_applyStep_add
      (cfg := GameConfig.standard) hwf (hv pl (by simp))
    rw [GameConfig.standard_cols] at hstep
    have hrec := ih (Placement.applyStep_wf hwf (hv pl (by simp)))
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, wordClears_cons]
    simp only [List.length_cons]
    omega

/-- **THE CLEAR CENSUS OF A CYCLE**: any valid word that folds a
well-formed board back to itself clears EXACTLY two fifths of a row
per move: `5 · clears = 2 · length`. A legal 35-word therefore clears
exactly 14 rows per period — the trace-level fourteen-per-period law,
recovered at the level of pure placement words. -/
theorem cycle_word_clear_census {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    5 * wordClears b w = 2 * w.length := by
  have h := foldl_count_ledger_exact hwf hv
  rw [hfold] at h
  omega

/-- A legal 35-cycle clears exactly fourteen rows. -/
theorem legal_cycle_word_clears_fourteen {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordClears b w = 14 := by
  have h := cycle_word_clear_census hwf hv hfold
  rw [hlen] at h
  omega

/-- Clears add along concatenation. -/
theorem wordClears_append (b : Board) (w1 w2 : List Placement) :
    wordClears b (w1 ++ w2)
      = wordClears b w1
        + wordClears (w1.foldl (Placement.applyStep GameConfig.standard) b)
            w2 := by
  induction w1 generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [List.cons_append, wordClears_cons, wordClears_cons,
      List.foldl_cons, ih]
    omega

/-- **The double-mill's harvest, from arithmetic alone**: the five-O
word clears exactly two rows — no board computation, just the census
`5 · clears = 2 · 5`. -/
theorem five_O_word_clears_two :
    wordClears (∅ : Board)
      [⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
       ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩] = 2 := by
  have hfold : ([⟨Piece.O, 0, 0⟩, ⟨Piece.O, 0, 2⟩, ⟨Piece.O, 0, 4⟩,
      ⟨Piece.O, 0, 6⟩, ⟨Piece.O, 0, 8⟩] : List Placement).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board) = ∅ := by
    simp only [List.foldl]
    exact five_O_cycle_empty
  have h := cycle_word_clear_census
    (Board.empty_wf GameConfig.standard) (by decide) hfold
  simp only [List.length_cons, List.length_nil] at h
  omega

/-- **The tetris-mill's harvest, from arithmetic alone**: the ten-I
word clears exactly four rows — the census `5 · clears = 2 · 10`
already knows the tetris. -/
theorem ten_I_word_clears_four :
    wordClears (∅ : Board)
      [⟨Piece.I, 1, 0⟩, ⟨Piece.I, 1, 1⟩, ⟨Piece.I, 1, 2⟩,
       ⟨Piece.I, 1, 3⟩, ⟨Piece.I, 1, 4⟩, ⟨Piece.I, 1, 5⟩,
       ⟨Piece.I, 1, 6⟩, ⟨Piece.I, 1, 7⟩, ⟨Piece.I, 1, 8⟩,
       ⟨Piece.I, 1, 9⟩] = 4 := by
  have hfold : ([⟨Piece.I, 1, 0⟩, ⟨Piece.I, 1, 1⟩, ⟨Piece.I, 1, 2⟩,
      ⟨Piece.I, 1, 3⟩, ⟨Piece.I, 1, 4⟩, ⟨Piece.I, 1, 5⟩,
      ⟨Piece.I, 1, 6⟩, ⟨Piece.I, 1, 7⟩, ⟨Piece.I, 1, 8⟩,
      ⟨Piece.I, 1, 9⟩] : List Placement).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board) = ∅ := by
    simp only [List.foldl]
    exact ten_I_cycle_empty
  have h := cycle_word_clear_census
    (Board.empty_wf GameConfig.standard) (by decide) hfold
  simp only [List.length_cons, List.length_nil] at h
  omega

/-- Number of clearing moves (moves that clear at least one row) while
playing a word from a board. -/
def wordClearMoves (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (if 0 < (Board.fullRows GameConfig.standard (pl.place b)).card
        then 1 else 0)
        + wordClearMoves (Placement.applyStep GameConfig.standard b pl)
            rest

@[simp] theorem wordClearMoves_nil (b : Board) :
    wordClearMoves b [] = 0 := rfl

theorem wordClearMoves_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordClearMoves b (pl :: rest)
      = (if 0 < (Board.fullRows GameConfig.standard (pl.place b)).card
          then 1 else 0)
        + wordClearMoves (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- Each clearing move reaps at least one row. -/
theorem wordClearMoves_le_wordClears (b : Board) (w : List Placement) :
    wordClearMoves b w ≤ wordClears b w := by
  induction w generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordClearMoves_cons, wordClears_cons]
    have := ih (Placement.applyStep GameConfig.standard b pl)
    split <;> omega

/-- Each clearing move reaps at most four rows — provided the start
board is clear-free, which every later board then is automatically. -/
theorem wordClears_le_four_mul_moves {b : Board} {w : List Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    wordClears b w ≤ 4 * wordClearMoves b w := by
  induction w generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordClearMoves_cons, wordClears_cons]
    have hcap : (Board.fullRows GameConfig.standard (pl.place b)).card
        ≤ 4 := linesCleared_place_le_four GameConfig.standard b pl hnf
    have hrec := ih (applyStep_clear_free b pl)
    split <;> omega

/-- **THE CLEARING-MOVES BRACKET**: a legal 35-cycle clears on at least
4 and at most 14 of its moves — fourteen rows at one-to-four rows per
harvest. The trace-level bracket, recovered for pure words. -/
theorem legal_cycle_word_clearing_moves_bracket {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    4 ≤ wordClearMoves b w ∧ wordClearMoves b w ≤ 14 := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h14 := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen
  have hnfb : ∀ r, ¬ Board.isFull GameConfig.standard b r :=
    board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩
  have hlo := wordClears_le_four_mul_moves (w := w) hnfb
  have hhi := wordClearMoves_le_wordClears b w
  omega

/-- **THE LEGAL CYCLE LEDGER** — everything arithmetic knows about a
bag-legal repeatable cycle word, in one statement. For any valid word
`w` folding a well-formed board back to itself whose repetition deals
full bags:

1. the length is a positive multiple of 35;
2. each of the seven pieces appears exactly `length / 7` times;
3. exactly `2 · length / 5` rows are cleared (`5 · clears = 2 · n`);
4. the clearing moves are bracketed: `2n ≤ 20 · moves` and
   `5 · moves ≤ 2n` — between a tenth and two fifths of the moves
   harvest.

The M2 witness, whatever its geometry, must run on these clocks. -/
theorem legal_cycle_ledger {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    (0 < w.length ∧ 35 ∣ w.length)
    ∧ (∀ p : Piece, ((Finset.range w.length).filter
        (fun i => (w.getD i ⟨Piece.O, 0, 0⟩).piece = p)).card
          = w.length / 7)
    ∧ 5 * wordClears b w = 2 * w.length
    ∧ 2 * w.length ≤ 20 * wordClearMoves b w
    ∧ 5 * wordClearMoves b w ≤ 2 * w.length := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have hcensus := legal_cycle_word_piece_census hwf hne hv hbag hfold
  have hclears := cycle_word_clear_census hwf hv hfold
  have hnfb : ∀ r, ¬ Board.isFull GameConfig.standard b r :=
    board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩
  have hlo := wordClears_le_four_mul_moves (w := w) hnfb
  have hhi := wordClearMoves_le_wordClears b w
  exact ⟨⟨hpos, h35⟩, hcensus, hclears, by omega, by omega⟩

/-- **Exhausting a bag refills it**: drawing, in any order, exactly the
pieces a bag holds — each once — ends with the freshly refilled full
bag. The seventh draw of every bag block hits the refill guard. -/
theorem foldl_draw_exhaust : ∀ (l : List Piece) (B : Bag), l ≠ [] →
    l.Nodup → (∀ p ∈ l, p ∈ B) → B.card = l.length →
    l.foldl Bag.draw B = Bag.full := by
  intro l
  induction l with
  | nil => intro B hne; exact absurd rfl hne
  | cons p rest ih =>
    intro B _ hnodup hmem hcard
    have hpB : p ∈ B := hmem p (by simp)
    rcases List.eq_nil_or_concat rest with hrest | _
    · subst hrest
      have hB : B = {p} := by
        rw [List.length_cons, List.length_nil] at hcard
        obtain ⟨a, ha⟩ := Finset.card_eq_one.mp hcard
        rw [ha] at hpB ⊢
        rw [Finset.mem_singleton] at hpB
        rw [hpB]
      rw [List.foldl_cons, List.foldl_nil, hB]
      unfold Bag.draw
      rw [if_pos (by rw [Finset.erase_singleton])]
    · have hrne : rest ≠ [] := by
        rintro rfl
        simp_all
      have hdraw : B.draw p = B.erase p := by
        unfold Bag.draw
        rw [if_neg]
        intro hemp
        have hce := Finset.card_erase_of_mem hpB
        rw [hemp, Finset.card_empty] at hce
        rw [List.length_cons] at hcard
        have hlen := List.length_pos_iff.mpr hrne
        omega
      rw [List.foldl_cons, hdraw]
      apply ih (B.erase p) hrne (List.Nodup.of_cons hnodup)
      · intro q hq
        rw [Finset.mem_erase]
        refine ⟨?_, hmem q (by simp [hq])⟩
        intro hqp
        subst hqp
        exact (List.nodup_cons.mp hnodup).1 hq
      · rw [Finset.card_erase_of_mem hpB, hcard, List.length_cons]
        omega

/-- A seven-piece list containing every piece has no repeats. -/
theorem full_block_nodup {l : List Piece} (hlen : l.length = 7)
    (hall : ∀ p : Piece, p ∈ l) : l.Nodup := by
  classical
  have hsub : (Finset.univ : Finset Piece) ⊆ l.toFinset := fun p _ =>
    List.mem_toFinset.mpr (hall p)
  have hcard7 : 7 ≤ l.toFinset.card := by
    rw [← piece_univ_card]
    exact Finset.card_le_card hsub
  have hle : l.toFinset.card ≤ l.length := l.toFinset_card_le
  have hdlen : l.dedup.length = l.length := by
    have hct := l.card_toFinset
    omega
  have hde : l.dedup = l := (List.dedup_sublist l).eq_of_length hdlen
  exact List.dedup_eq_self.mp hde

/-- **The bag block reset**: playing a seven-piece block that contains
every piece, from the full bag, ends back at the full bag. Bag legality
makes the bag component PERIODIC with period seven. -/
theorem bag_refills_after_full_block {l : List Piece}
    (hlen : l.length = 7) (hall : ∀ p : Piece, p ∈ l) :
    l.foldl Bag.draw Bag.full = Bag.full := by
  apply foldl_draw_exhaust l Bag.full
  · intro hnil
    rw [hnil] at hlen
    simp at hlen
  · exact full_block_nodup hlen hall
  · intro q _
    exact Bag.mem_full q
  · rw [hlen, Bag.full_card]

/-- Play a placement word as game-state steps. -/
def stepWord (g : GameState) (w : List Placement) : GameState :=
  w.foldl (fun g' pl => g'.step GameConfig.standard pl) g

/-- The board component of a played word is the board fold. -/
theorem stepWord_board (g : GameState) (w : List Placement) :
    (stepWord g w).board
      = w.foldl (Placement.applyStep GameConfig.standard) g.board := by
  induction w generalizing g with
  | nil => rfl
  | cons pl rest ih =>
    unfold stepWord at ih ⊢
    rw [List.foldl_cons, List.foldl_cons, ih, GameState.step_board]

/-- The bag component of a played word is the draw fold over the
word's pieces. -/
theorem stepWord_bag (g : GameState) (w : List Placement) :
    (stepWord g w).bag
      = (w.map (·.piece)).foldl Bag.draw g.bag := by
  induction w generalizing g with
  | nil => rfl
  | cons pl rest ih =>
    unfold stepWord at ih ⊢
    rw [List.foldl_cons, List.map_cons, List.foldl_cons, ih,
      GameState.step_bag]

/-- **Block-structured piece lists drain to the full bag**: a list of
`7k` pieces whose every seven-block contains every piece folds the full
bag back to the full bag — block by block via the bag reset. -/
theorem bag_stream_list_foldl : ∀ (k : ℕ) (l : List Piece),
    l.length = 7 * k →
    (∀ j < k, ∀ p : Piece, ∃ i < 7, l.getD (7 * j + i) Piece.O = p) →
    l.foldl Bag.draw Bag.full = Bag.full := by
  intro k
  induction k with
  | zero =>
    intro l hlen _
    rw [Nat.mul_zero] at hlen
    rw [List.length_eq_zero_iff.mp hlen]
    rfl
  | succ m ih =>
    intro l hlen hblock
    have h7le : 7 ≤ l.length := by omega
    have hsplit : l.take 7 ++ l.drop 7 = l := List.take_append_drop 7 l
    have htlen : (l.take 7).length = 7 := by
      rw [List.length_take]
      omega
    have hall : ∀ p : Piece, p ∈ l.take 7 := by
      intro p
      obtain ⟨i, hi, hget⟩ := hblock 0 (by omega) p
      rw [Nat.mul_zero, Nat.zero_add] at hget
      have hilen : i < l.length := by omega
      rw [List.getD_eq_getElem l Piece.O hilen] at hget
      have hit : (l.take 7)[i]'(by omega) = l[i] := by
        rw [List.getElem_take]
      rw [← hget, ← hit]
      exact List.getElem_mem _
    have hdlen : (l.drop 7).length = 7 * m := by
      rw [List.length_drop]
      omega
    have hgd : ∀ n, n < (l.drop 7).length →
        (l.drop 7).getD n Piece.O = l.getD (7 + n) Piece.O := by
      intro n hn
      rw [List.getD_eq_getElem _ _ hn,
        List.getD_eq_getElem _ _ (by rw [List.length_drop] at hn; omega)]
      rw [List.getElem_drop]
    have hdblock : ∀ j < m, ∀ p : Piece,
        ∃ i < 7, (l.drop 7).getD (7 * j + i) Piece.O = p := by
      intro j hj p
      obtain ⟨i, hi, hget⟩ := hblock (j + 1) (by omega) p
      refine ⟨i, hi, ?_⟩
      rw [hgd (7 * j + i) (by omega)]
      rw [show 7 + (7 * j + i) = 7 * (j + 1) + i from by ring]
      exact hget
    rw [← hsplit, List.foldl_append,
      bag_refills_after_full_block htlen hall]
    exact ih (l.drop 7) hdlen hdblock

/-- **A bag-legal word drains to the full bag**: if the repeated word
deals full bags and its length is a multiple of seven, playing its
pieces from the full bag ends at the full bag. -/
theorem legal_word_bag_reset {w : List Placement}
    (hbag : IsBagStream (wordStream w)) (h7 : 7 ∣ w.length) :
    (w.map (·.piece)).foldl Bag.draw Bag.full = Bag.full := by
  obtain ⟨k, hk⟩ := h7
  apply bag_stream_list_foldl k
  · rw [List.length_map, hk]
  · intro j hj p
    obtain ⟨i, hi, hws⟩ := hbag j p
    refine ⟨i, hi, ?_⟩
    have hn : 7 * j + i < w.length := by omega
    unfold wordStream at hws
    rw [Nat.mod_eq_of_lt hn] at hws
    rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hn),
      List.getElem_map]
    rw [List.getD_eq_getElem _ _ hn] at hws
    exact hws

/-- **THE FULL-STATE LEGAL CYCLE**: a bag-legal cycle word, played as
game-state steps from `⟨b, full bag⟩`, returns the ENTIRE game state —
board AND bag — exactly. Every draw along the way is legal (the stream
deals what the bag holds), the board rides its 35-quantum orbit, and
the bag beats its seven-pulse. This is the shape of the M2 object:
all that remains open is exhibiting one such word. -/
theorem legal_cycle_word_state_cycle {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    stepWord ⟨b, Bag.full⟩ w = ⟨b, Bag.full⟩ := by
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  apply GameState.eq_of_board_bag
  · rw [stepWord_board]
    exact hfold
  · rw [stepWord_bag]
    exact legal_word_bag_reset hbag (by omega)

/-- The infinite play obtained by repeating a placement word forever
from a state: move `n` plays the word's letter `n mod length`. -/
def wordPlay (g : GameState) (w : List Placement) : ℕ → GameState
  | 0 => g
  | n + 1 =>
      (wordPlay g w n).step GameConfig.standard
        (w.getD (n % w.length) ⟨Piece.O, 0, 0⟩)

@[simp] theorem wordPlay_zero (g : GameState) (w : List Placement) :
    wordPlay g w 0 = g := rfl

theorem wordPlay_succ (g : GameState) (w : List Placement) (n : ℕ) :
    wordPlay g w (n + 1)
      = (wordPlay g w n).step GameConfig.standard
          (w.getD (n % w.length) ⟨Piece.O, 0, 0⟩) := rfl

/-- Inside the first period, the infinite play is the prefix fold. -/
theorem wordPlay_eq_stepWord_take {g : GameState} {w : List Placement}
    {n : ℕ} (hn : n ≤ w.length) :
    wordPlay g w n = stepWord g (w.take n) := by
  induction n with
  | zero => simp [stepWord]
  | succ m ih =>
    have hm : m < w.length := by omega
    rw [wordPlay_succ, ih (by omega), Nat.mod_eq_of_lt hm]
    rw [List.take_succ]
    unfold stepWord
    rw [List.foldl_append]
    rw [List.getElem?_eq_getElem hm]
    simp only [Option.toList_some, List.foldl_cons, List.foldl_nil]
    rw [List.getD_eq_getElem _ _ hm]

/-- **The mill turns forever**: repeating a legal cycle word makes the
infinite play periodic — one full period returns the state, and every
later step repeats the pattern exactly. -/
theorem wordPlay_periodic {b : Board} {w : List Placement}
    (hne : w ≠ [])
    (hcyc : stepWord ⟨b, Bag.full⟩ w = ⟨b, Bag.full⟩) :
    ∀ m, wordPlay ⟨b, Bag.full⟩ w (w.length + m)
      = wordPlay ⟨b, Bag.full⟩ w m := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  intro m
  induction m with
  | zero =>
    rw [Nat.add_zero, wordPlay_eq_stepWord_take (le_refl w.length),
      List.take_length, hcyc]
    rfl
  | succ k ih =>
    rw [show w.length + (k + 1) = (w.length + k) + 1 from rfl,
      wordPlay_succ, wordPlay_succ, ih,
      Nat.add_mod_left w.length k]

/-- **THE INFINITE MILL**: a bag-legal cycle word that never tops out
within its first period, repeated forever from `⟨b, full bag⟩`, NEVER
tops out. Periodicity reduces every future state to a first-period
state, and those are safe by hypothesis. A safe legal cycle word is a
complete proof of infinite play — M2 implies M1, at the level of pure
words. -/
theorem legal_cycle_word_survives_forever {b : Board}
    {w : List Placement} (hne : w ≠ [])
    (hcyc : stepWord ⟨b, Bag.full⟩ w = ⟨b, Bag.full⟩)
    (hsafe : ∀ n ≤ w.length,
      ¬ (stepWord ⟨b, Bag.full⟩ (w.take n)).lost GameConfig.standard) :
    ∀ n, ¬ (wordPlay ⟨b, Bag.full⟩ w n).lost GameConfig.standard := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases hlt : n < w.length
    · rw [wordPlay_eq_stepWord_take (by omega)]
      exact hsafe n (by omega)
    · have hsplit : n = w.length + (n - w.length) := by omega
      rw [hsplit, wordPlay_periodic hne hcyc]
      exact ih (n - w.length) (by omega)

/-- Number of tetrises (four-clear moves) while playing a word. -/
def wordTetrises (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (if 4 ≤ (Board.fullRows GameConfig.standard (pl.place b)).card
        then 1 else 0)
        + wordTetrises (Placement.applyStep GameConfig.standard b pl)
            rest

@[simp] theorem wordTetrises_nil (b : Board) : wordTetrises b [] = 0 :=
  rfl

theorem wordTetrises_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordTetrises b (pl :: rest)
      = (if 4 ≤ (Board.fullRows GameConfig.standard (pl.place b)).card
          then 1 else 0)
        + wordTetrises (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- **Every tetris is an I**: along any word from a clear-free board,
the four-clear moves are at most the I-moves. -/
theorem wordTetrises_le_I_count {b : Board} {w : List Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    wordTetrises b w ≤ (w.map (·.piece)).count Piece.I := by
  induction w generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordTetrises_cons, List.map_cons, List.count_cons]
    have hrec := ih (applyStep_clear_free b pl)
    by_cases h4 : 4 ≤ (Board.fullRows GameConfig.standard
        (pl.place b)).card
    · have hI : pl.piece = Piece.I := tetris_requires_I hnf h4
      rw [if_pos h4]
      split
      · omega
      · next hcond => simp [hI] at hcond
    · rw [if_neg h4]
      split <;> omega

/-- The index census of a word equals the value count of its pieces. -/
theorem census_eq_count (p : Piece) : ∀ (w : List Placement),
    ((Finset.range w.length).filter
      (fun i => (w.getD i ⟨Piece.O, 0, 0⟩).piece = p)).card
    = (w.map (·.piece)).count p := by
  intro w
  induction w using List.reverseRecOn with
  | nil => simp
  | append_singleton t a ih =>
    classical
    rw [List.length_append, List.length_singleton,
      Finset.range_add_one, Finset.filter_insert]
    have hlast : (t ++ [a]).getD t.length ⟨Piece.O, 0, 0⟩ = a := by
      simp
    have hfeq : (Finset.range t.length).filter
        (fun i => ((t ++ [a]).getD i ⟨Piece.O, 0, 0⟩).piece = p)
        = (Finset.range t.length).filter
            (fun i => (t.getD i ⟨Piece.O, 0, 0⟩).piece = p) := by
      apply Finset.filter_congr
      intro i hi
      rw [Finset.mem_range] at hi
      rw [List.getD_append _ _ _ _ hi]
    have hnotmem : t.length ∉ (Finset.range t.length).filter
        (fun i => ((t ++ [a]).getD i ⟨Piece.O, 0, 0⟩).piece = p) := by
      intro hmem
      have := (Finset.mem_filter.mp hmem).1
      rw [Finset.mem_range] at this
      omega
    rw [List.map_append, List.count_append]
    simp only [List.map_cons, List.map_nil, List.count_cons,
      List.count_nil]
    by_cases hap : a.piece = p
    · rw [if_pos (by rw [hlast]; exact hap),
        Finset.card_insert_of_notMem hnotmem, hfeq, ih]
      simp [hap]
    · rw [if_neg (by rw [hlast]; exact hap), hfeq, ih]
      simp [hap]

/-- **THE TETRIS CAP**: a bag-legal cycle word plays at most
`length / 7` tetrises — one per bag, the bag's single I. A legal
35-cycle holds at most five tetrises against its fourteen rows. -/
theorem legal_cycle_word_tetris_cap {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordTetrises b w ≤ w.length / 7 := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have hnfb : ∀ r, ¬ Board.isFull GameConfig.standard b r :=
    board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩
  have h1 := wordTetrises_le_I_count (w := w) hnfb
  have h2 := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.I
  rw [census_eq_count] at h2
  omega

/-- **Clears fit the piece's row span**: on a clear-free board, a move
clears at most as many rows as its piece's shape occupies — every
cleared row must contain a fresh cell. -/
theorem clears_le_row_span {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r) :
    (Board.fullRows cfg (pl.place b)).card
      ≤ ((pl.shapeUp).image (·.2)).card := by
  classical
  have hsub : Board.fullRows cfg (pl.place b)
      ⊆ (pl.dropped b).image Prod.snd := by
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
  have himg : (pl.dropped b).image Prod.snd
      = ((pl.shapeUp).image (·.2)).image (pl.dropOffset b + ·) := by
    rw [Placement.dropped_eq_image, Finset.image_image,
      Finset.image_image]
    rfl
  calc (Board.fullRows cfg (pl.place b)).card
      ≤ ((pl.dropped b).image Prod.snd).card := Finset.card_le_card hsub
    _ = (((pl.shapeUp).image (·.2)).image (pl.dropOffset b + ·)).card :=
        by rw [himg]
    _ = ((pl.shapeUp).image (·.2)).card :=
        Finset.card_image_of_injective _ (add_right_injective _)

/-- Every non-I shape spans at most three rows, in every rotation. -/
theorem non_I_row_span_le_three :
    ∀ (p : Piece) (r : Rotation), p ≠ Piece.I →
      ((p.shapeUp r).image (·.2)).card ≤ 3 := by
  decide

/-- **Non-I moves clear at most three rows** on a clear-free board:
only the I spans four rows, so only the I can take four. -/
theorem non_I_move_clears_le_three {cfg : GameConfig} {b : Board}
    {pl : Placement} (hnf : ∀ r, ¬ Board.isFull cfg b r)
    (hpI : pl.piece ≠ Piece.I) :
    (Board.fullRows cfg (pl.place b)).card ≤ 3 := by
  have h1 := clears_le_row_span (b := b) (pl := pl) hnf
  have h2 := non_I_row_span_le_three pl.piece pl.rot hpI
  have h3 : (pl.shapeUp).image (·.2)
      = (pl.piece.shapeUp pl.rot).image (·.2) := rfl
  rw [h3] at h1
  omega

/-- **THE CLEAR DICHOTOMY**: on a clear-free board, every move either
clears at most three rows, or clears exactly four and is the vertical
I — there is nothing in between the ordinary harvest and the tetris. -/
theorem move_clear_dichotomy {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    (Board.fullRows GameConfig.standard (pl.place b)).card ≤ 3
    ∨ ((Board.fullRows GameConfig.standard (pl.place b)).card = 4
        ∧ pl.piece = Piece.I) := by
  have hle := linesCleared_place_le_four GameConfig.standard b pl hnf
  rw [Board.linesCleared] at hle
  by_cases h4 : 4 ≤ (Board.fullRows GameConfig.standard
      (pl.place b)).card
  · exact Or.inr ⟨by omega, tetris_requires_I hnf h4⟩
  · exact Or.inl (by omega)

/-- **The word mix bound**: total rows cleared is at most three per
clearing move plus one extra per tetris — the clear dichotomy, folded
along the word. -/
theorem word_clear_mix_bound {b : Board} {w : List Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    wordClears b w ≤ 3 * wordClearMoves b w + wordTetrises b w := by
  induction w generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordClears_cons, wordClearMoves_cons, wordTetrises_cons]
    have hcap : (Board.fullRows GameConfig.standard (pl.place b)).card
        ≤ 4 := by
      have := linesCleared_place_le_four GameConfig.standard b pl hnf
      rwa [Board.linesCleared] at this
    have hrec := ih (applyStep_clear_free b pl)
    by_cases h4 : 4 ≤ (Board.fullRows GameConfig.standard
        (pl.place b)).card
    · rw [if_pos h4, if_pos (by omega)]
      omega
    · rw [if_neg h4]
      by_cases h0 : 0 < (Board.fullRows GameConfig.standard
          (pl.place b)).card
      · rw [if_pos h0]
        omega
      · rw [if_neg h0]
        have hz : (Board.fullRows GameConfig.standard
            (pl.place b)).card = 0 := by omega
        omega

/-- **Tetris-free cycles work harder**: a bag-legal cycle word playing
NO tetrises must clear on at least `2·length/15` of its moves — the
mix bound with the fourth row withheld. A tetris-free legal 35-cycle
harvests on at least five moves. -/
theorem legal_cycle_word_tetris_free_moves {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hT : wordTetrises b w = 0) :
    2 * w.length ≤ 15 * wordClearMoves b w := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have hnfb : ∀ r, ¬ Board.isFull GameConfig.standard b r :=
    board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩
  have hmix := word_clear_mix_bound (w := w) hnfb
  have hcensus := cycle_word_clear_census hwf hv hfold
  omega

/-- **Mid-block draws are exact set differences**: drawing distinct
present pieces, strictly fewer than the bag holds, removes exactly
those pieces — the refill guard never fires. -/
theorem foldl_draw_prefix : ∀ (l : List Piece) (B : Bag), l.Nodup →
    (∀ p ∈ l, p ∈ B) → l.length < B.card →
    l.foldl Bag.draw B = B \ l.toFinset := by
  intro l
  induction l with
  | nil =>
    intro B _ _ _
    simp
  | cons p rest ih =>
    intro B hnodup hmem hlen
    have hpB : p ∈ B := hmem p (by simp)
    have hdraw : B.draw p = B.erase p := by
      unfold Bag.draw
      rw [if_neg]
      intro hemp
      have hce := Finset.card_erase_of_mem hpB
      rw [hemp, Finset.card_empty] at hce
      rw [List.length_cons] at hlen
      omega
    rw [List.foldl_cons, hdraw]
    have hrec := ih (B.erase p) (List.Nodup.of_cons hnodup)
      (fun q hq => Finset.mem_erase.mpr
        ⟨fun hqp => (List.nodup_cons.mp hnodup).1 (hqp ▸ hq),
          hmem q (by simp [hq])⟩)
      (by
        rw [Finset.card_erase_of_mem hpB]
        rw [List.length_cons] at hlen
        omega)
    rw [hrec]
    ext q
    simp only [Finset.mem_sdiff, Finset.mem_erase, List.toFinset_cons,
      Finset.mem_insert]
    tauto

/-- The card of a mid-block draw: the bag loses exactly one per
draw. -/
theorem foldl_draw_prefix_card {l : List Piece} {B : Bag}
    (hnodup : l.Nodup) (hmem : ∀ p ∈ l, p ∈ B)
    (hlen : l.length < B.card) :
    (l.foldl Bag.draw B).card = B.card - l.length := by
  rw [foldl_draw_prefix l B hnodup hmem hlen]
  have hsubset : l.toFinset ⊆ B := fun q hq =>
    hmem q (List.mem_toFinset.mp hq)
  rw [Finset.card_sdiff, Finset.inter_eq_left.mpr hsubset,
    List.card_toFinset, List.dedup_eq_self.mpr hnodup]

/-- **The bag-card signature of block-structured play**: along a piece
list of full seven-blocks, the bag after `n` draws from full holds
exactly `7 - n % 7` pieces — full at every block boundary, draining one
per draw between. -/
theorem bag_stream_take_card : ∀ (n : ℕ) (l : List Piece),
    7 ∣ l.length → n ≤ l.length →
    (∀ j, 7 * j + 7 ≤ l.length → ∀ p : Piece,
      ∃ i < 7, l.getD (7 * j + i) Piece.O = p) →
    ((l.take n).foldl Bag.draw Bag.full).card = 7 - n % 7 := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro l h7 hn hblock
    by_cases hlt : n < 7
    · by_cases hn0 : n = 0
      · subst hn0
        simp [Bag.full_card]
      · have hlen7 : 7 ≤ l.length := by
          obtain ⟨k, hk⟩ := h7
          omega
        have hblk_len : (l.take 7).length = 7 := by
          rw [List.length_take]
          omega
        have hblk_all : ∀ p : Piece, p ∈ l.take 7 := by
          intro p
          obtain ⟨i, hi, hget⟩ := hblock 0 (by omega) p
          rw [Nat.mul_zero, Nat.zero_add] at hget
          have hilen : i < l.length := by omega
          rw [List.getD_eq_getElem l Piece.O hilen] at hget
          have hit : (l.take 7)[i]'(by omega) = l[i] := by
            rw [List.getElem_take]
          rw [← hget, ← hit]
          exact List.getElem_mem _
        have hblk_nodup := full_block_nodup hblk_len hblk_all
        have htn : l.take n = (l.take 7).take n := by
          rw [List.take_take]
          congr 1
          omega
        have hpre_nodup : ((l.take 7).take n).Nodup :=
          hblk_nodup.sublist (List.take_sublist _ _)
        have hlen_take : ((l.take 7).take n).length = n := by
          rw [List.length_take]
          omega
        rw [htn, foldl_draw_prefix_card hpre_nodup
          (fun q _ => Bag.mem_full q)
          (by rw [hlen_take, Bag.full_card]; omega)]
        rw [hlen_take, Bag.full_card, Nat.mod_eq_of_lt hlt]
    · have hlen7 : 7 ≤ l.length := by omega
      have htlen : (l.take 7).length = 7 := by
        rw [List.length_take]
        omega
      have hall : ∀ p : Piece, p ∈ l.take 7 := by
        intro p
        obtain ⟨i, hi, hget⟩ := hblock 0 (by omega) p
        rw [Nat.mul_zero, Nat.zero_add] at hget
        have hilen : i < l.length := by omega
        rw [List.getD_eq_getElem l Piece.O hilen] at hget
        have hit : (l.take 7)[i]'(by omega) = l[i] := by
          rw [List.getElem_take]
        rw [← hget, ← hit]
        exact List.getElem_mem _
      have hsplit : l.take n = l.take 7 ++ (l.drop 7).take (n - 7) := by
        conv_lhs => rw [show n = 7 + (n - 7) from by omega]
        rw [List.take_add]
      have hgd : ∀ m, m < (l.drop 7).length →
          (l.drop 7).getD m Piece.O = l.getD (7 + m) Piece.O := by
        intro m hm
        rw [List.getD_eq_getElem _ _ hm,
          List.getD_eq_getElem _ _
            (by rw [List.length_drop] at hm; omega)]
        rw [List.getElem_drop]
      have hrec := ih (n - 7) (by omega) (l.drop 7)
        (by rw [List.length_drop]; omega)
        (by rw [List.length_drop]; omega)
        (by
          intro j hj p
          rw [List.length_drop] at hj
          obtain ⟨i, hi, hget⟩ := hblock (j + 1) (by omega) p
          refine ⟨i, hi, ?_⟩
          rw [hgd (7 * j + i) (by rw [List.length_drop]; omega)]
          rw [show 7 + (7 * j + i) = 7 * (j + 1) + i from by ring]
          exact hget)
      rw [hsplit, List.foldl_append,
        bag_refills_after_full_block htlen hall, hrec]
      omega

/-- A bag-legal word's piece list has full seven-blocks. -/
theorem map_piece_blocks {w : List Placement}
    (hbag : IsBagStream (wordStream w)) :
    ∀ j, 7 * j + 7 ≤ (w.map (·.piece)).length → ∀ p : Piece,
      ∃ i < 7, (w.map (·.piece)).getD (7 * j + i) Piece.O = p := by
  intro j hj p
  rw [List.length_map] at hj
  obtain ⟨i, hi, hws⟩ := hbag j p
  refine ⟨i, hi, ?_⟩
  have hn : 7 * j + i < w.length := by omega
  unfold wordStream at hws
  rw [Nat.mod_eq_of_lt hn] at hws
  rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hn),
    List.getElem_map]
  rw [List.getD_eq_getElem _ _ hn] at hws
  exact hws

/-- **The bag clock of a legal play**: within the first period, the
bag after `n` moves holds exactly `7 - n % 7` pieces. -/
theorem wordPlay_bag_card {b : Board} {w : List Placement}
    (hbag : IsBagStream (wordStream w)) (h7 : 7 ∣ w.length)
    {n : ℕ} (hn : n ≤ w.length) :
    (wordPlay ⟨b, Bag.full⟩ w n).bag.card = 7 - n % 7 := by
  rw [wordPlay_eq_stepWord_take hn, stepWord_bag]
  have hmt : (w.take n).map (·.piece) = (w.map (·.piece)).take n :=
    List.map_take
  rw [hmt]
  exact bag_stream_take_card n (w.map (·.piece))
    (by rw [List.length_map]; exact h7)
    (by rw [List.length_map]; exact hn)
    (map_piece_blocks hbag)

/-- **The mass clock of a legal play**: within the first period, the
board count after `n` moves is `b.count + 4n` modulo ten. -/
theorem wordPlay_count_mod {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    {n : ℕ} (hn : n ≤ w.length) :
    (wordPlay ⟨b, Bag.full⟩ w n).board.count % 10
      = (b.count + 4 * n) % 10 := by
  rw [wordPlay_eq_stepWord_take hn, stepWord_board,
    show (⟨b, Bag.full⟩ : GameState).board = b from rfl]
  have hvt : ∀ pl ∈ w.take n, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_of_mem_take hpl)
  obtain ⟨C, hC⟩ := foldl_applyStep_count_ledger
    (b := b) (pls := w.take n) hwf hvt
  rw [List.length_take] at hC
  have hmin : min n w.length = n := by omega
  rw [hmin] at hC
  omega

/-- **THE ORBIT SIGNATURE**: along any legal cycle word's first
period, every state wears two clocks — the bag reads `7 - n % 7` and
the board count reads `b.count + 4n` mod ten. Two states at different
first-period times can only coincide if their times agree mod 35. -/
theorem legal_cycle_orbit_signature {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w)) (h7 : 7 ∣ w.length)
    {n : ℕ} (hn : n ≤ w.length) :
    (wordPlay ⟨b, Bag.full⟩ w n).bag.card = 7 - n % 7
    ∧ (wordPlay ⟨b, Bag.full⟩ w n).board.count % 10
        = (b.count + 4 * n) % 10 :=
  ⟨wordPlay_bag_card hbag h7 hn, wordPlay_count_mod hwf hv hn⟩

/-- **No shortcuts on the orbit**: along a legal cycle word, two
first-period states less than 35 steps apart are DISTINCT — the bag
clock separates times differing mod 7, the mass clock separates times
differing mod 5, and the clocks are coprime. -/
theorem legal_cycle_orbit_distinct {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    {i j : ℕ} (hij : i < j) (hj : j ≤ w.length) (hlt : j - i < 35) :
    wordPlay ⟨b, Bag.full⟩ w i ≠ wordPlay ⟨b, Bag.full⟩ w j := by
  intro heq
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have h7 : 7 ∣ w.length := by omega
  have hci := wordPlay_bag_card (b := b) hbag h7
    (show i ≤ w.length by omega)
  have hcj := wordPlay_bag_card (b := b) hbag h7 hj
  have hmi := wordPlay_count_mod (b := b) (w := w) hwf hv
    (show i ≤ w.length by omega)
  have hmj := wordPlay_count_mod (b := b) (w := w) hwf hv hj
  rw [heq] at hci hmi
  omega

/-- **THE ORBIT IS A TRUE 35-CYCLE**: a legal cycle word's first 35
states are pairwise distinct. The M2 object cannot degenerate — its
orbit genuinely visits (at least) thirty-five states, one for each
tick of the joint mass-and-bag clock. -/
theorem legal_cycle_thirty_five_distinct_states {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b) :
    ∀ i < 35, ∀ j < 35, i ≠ j →
      wordPlay ⟨b, Bag.full⟩ w i ≠ wordPlay ⟨b, Bag.full⟩ w j := by
  have hmin := legal_cycle_word_min_length hwf hne hv hbag hfold
  intro i hi j hj hne'
  rcases Nat.lt_or_ge i j with hlt | hge
  · exact legal_cycle_orbit_distinct hwf hne hv hbag hfold hlt
      (by omega) (by omega)
  · have hlt : j < i := by omega
    exact (legal_cycle_orbit_distinct hwf hne hv hbag hfold hlt
      (by omega) (by omega)).symm

/-- **The bag's exact value along block play**: after `n` draws from
full, the bag is the full bag minus precisely the pieces drawn so far
in the CURRENT block — the first `n mod 7` pieces of block `n / 7`. -/
theorem bag_stream_take_val : ∀ (n : ℕ) (l : List Piece),
    7 ∣ l.length → n ≤ l.length →
    (∀ j, 7 * j + 7 ≤ l.length → ∀ p : Piece,
      ∃ i < 7, l.getD (7 * j + i) Piece.O = p) →
    (l.take n).foldl Bag.draw Bag.full
      = Bag.full \ ((l.drop (7 * (n / 7))).take (n % 7)).toFinset := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro l h7 hn hblock
    by_cases hlt : n < 7
    · rw [Nat.div_eq_of_lt hlt, Nat.mod_eq_of_lt hlt, Nat.mul_zero,
        List.drop_zero]
      by_cases hn0 : n = 0
      · subst hn0
        simp
      · have hlen7 : 7 ≤ l.length := by
          obtain ⟨k, hk⟩ := h7
          omega
        have hblk_len : (l.take 7).length = 7 := by
          rw [List.length_take]
          omega
        have hblk_all : ∀ p : Piece, p ∈ l.take 7 := by
          intro p
          obtain ⟨i, hi, hget⟩ := hblock 0 (by omega) p
          rw [Nat.mul_zero, Nat.zero_add] at hget
          have hilen : i < l.length := by omega
          rw [List.getD_eq_getElem l Piece.O hilen] at hget
          have hit : (l.take 7)[i]'(by omega) = l[i] := by
            rw [List.getElem_take]
          rw [← hget, ← hit]
          exact List.getElem_mem _
        have hblk_nodup := full_block_nodup hblk_len hblk_all
        have htn : l.take n = (l.take 7).take n := by
          rw [List.take_take]
          congr 1
          omega
        have hpre_nodup : ((l.take 7).take n).Nodup :=
          hblk_nodup.sublist (List.take_sublist _ _)
        have hlen_take : ((l.take 7).take n).length = n := by
          rw [List.length_take]
          omega
        rw [htn, foldl_draw_prefix _ _ hpre_nodup
          (fun q _ => Bag.mem_full q)
          (by rw [hlen_take, Bag.full_card]; omega)]
    · have hlen7 : 7 ≤ l.length := by omega
      have htlen : (l.take 7).length = 7 := by
        rw [List.length_take]
        omega
      have hall : ∀ p : Piece, p ∈ l.take 7 := by
        intro p
        obtain ⟨i, hi, hget⟩ := hblock 0 (by omega) p
        rw [Nat.mul_zero, Nat.zero_add] at hget
        have hilen : i < l.length := by omega
        rw [List.getD_eq_getElem l Piece.O hilen] at hget
        have hit : (l.take 7)[i]'(by omega) = l[i] := by
          rw [List.getElem_take]
        rw [← hget, ← hit]
        exact List.getElem_mem _
      have hsplit : l.take n = l.take 7 ++ (l.drop 7).take (n - 7) := by
        conv_lhs => rw [show n = 7 + (n - 7) from by omega]
        rw [List.take_add]
      have hgd : ∀ m, m < (l.drop 7).length →
          (l.drop 7).getD m Piece.O = l.getD (7 + m) Piece.O := by
        intro m hm
        rw [List.getD_eq_getElem _ _ hm,
          List.getD_eq_getElem _ _
            (by rw [List.length_drop] at hm; omega)]
        rw [List.getElem_drop]
      have hrec := ih (n - 7) (by omega) (l.drop 7)
        (by rw [List.length_drop]; omega)
        (by rw [List.length_drop]; omega)
        (by
          intro j hj p
          rw [List.length_drop] at hj
          obtain ⟨i, hi, hget⟩ := hblock (j + 1) (by omega) p
          refine ⟨i, hi, ?_⟩
          rw [hgd (7 * j + i) (by rw [List.length_drop]; omega)]
          rw [show 7 + (7 * j + i) = 7 * (j + 1) + i from by ring]
          exact hget)
      rw [hsplit, List.foldl_append,
        bag_refills_after_full_block htlen hall, hrec]
      rw [List.drop_drop]
      rw [show (n - 7) % 7 = n % 7 from by omega]
      first
        | rw [show 7 * ((n - 7) / 7) + 7 = 7 * (n / 7) from by omega]
        | rw [show 7 + 7 * ((n - 7) / 7) = 7 * (n / 7) from by omega]

/-- `getD` through `drop`. -/
theorem getD_drop_add (l : List Piece) (m i : ℕ)
    (h : m + i < l.length) :
    (l.drop m).getD i Piece.O = l.getD (m + i) Piece.O := by
  rw [List.getD_eq_getElem _ _ (by rw [List.length_drop]; omega),
    List.getD_eq_getElem _ _ h]
  rw [List.getElem_drop]

/-- **Every draw of a legal word is legal**: at each step, the piece
the word plays is present in the current bag — it is the next fresh
piece of the current block, and the bag holds exactly the block's
undrawn remainder. -/
theorem legal_word_draw_legal {b : Board} {w : List Placement}
    (hbag : IsBagStream (wordStream w)) (h7 : 7 ∣ w.length)
    {n : ℕ} (hn : n < w.length) :
    (w.getD n ⟨Piece.O, 0, 0⟩).piece
      ∈ (wordPlay ⟨b, Bag.full⟩ w n).bag := by
  classical
  have hlen : (w.map (·.piece)).length = w.length := List.length_map ..
  have hbag_val := bag_stream_take_val n (w.map (·.piece))
    (by rw [hlen]; exact h7) (by rw [hlen]; omega)
    (map_piece_blocks hbag)
  have hbagn : (wordPlay ⟨b, Bag.full⟩ w n).bag
      = Bag.full \ (((w.map (·.piece)).drop (7 * (n / 7))).take
          (n % 7)).toFinset := by
    rw [wordPlay_eq_stepWord_take (by omega), stepWord_bag,
      show (⟨b, Bag.full⟩ : GameState).bag = Bag.full from rfl,
      List.map_take]
    exact hbag_val
  rw [hbagn, Finset.mem_sdiff]
  refine ⟨Bag.mem_full _, ?_⟩
  have hpiece : (w.getD n ⟨Piece.O, 0, 0⟩).piece
      = (w.map (·.piece)).getD n Piece.O := by
    rw [List.getD_eq_getElem w _ hn,
      List.getD_eq_getElem _ _ (by rw [hlen]; exact hn),
      List.getElem_map]
  rw [hpiece]
  intro hmem
  rw [List.mem_toFinset] at hmem
  -- block data
  have hq7 : 7 * (n / 7) + 7 ≤ w.length := by
    obtain ⟨k, hk⟩ := h7
    omega
  have hdlen : ((w.map (·.piece)).drop (7 * (n / 7))).length
      = w.length - 7 * (n / 7) := by
    rw [List.length_drop, hlen]
  have hblk_len : (((w.map (·.piece)).drop (7 * (n / 7))).take
      7).length = 7 := by
    rw [List.length_take, hdlen]
    omega
  have hblk_all : ∀ p : Piece,
      p ∈ ((w.map (·.piece)).drop (7 * (n / 7))).take 7 := by
    intro p
    obtain ⟨i, hi, hget⟩ := map_piece_blocks hbag (n / 7)
      (by rw [hlen]; exact hq7) p
    have hdi : ((w.map (·.piece)).drop (7 * (n / 7))).getD i Piece.O
        = p := by
      rw [getD_drop_add _ _ _ (by rw [hlen]; omega)]
      exact hget
    have hilt : i < ((w.map (·.piece)).drop (7 * (n / 7))).length := by
      rw [hdlen]
      omega
    rw [List.getD_eq_getElem _ _ hilt] at hdi
    have hit : (((w.map (·.piece)).drop (7 * (n / 7))).take 7)[i]'(by
        rw [hblk_len]; omega)
        = ((w.map (·.piece)).drop (7 * (n / 7)))[i] := by
      rw [List.getElem_take]
    rw [← hdi, ← hit]
    exact List.getElem_mem _
  have hblk_nodup := full_block_nodup hblk_len hblk_all
  -- the played piece is the block's element at index n % 7
  have hplayed : (w.map (·.piece)).getD n Piece.O
      = ((w.map (·.piece)).drop (7 * (n / 7)))[n % 7]'(by
          rw [hdlen]; omega) := by
    rw [← List.getD_eq_getElem _ Piece.O]
    rw [getD_drop_add _ _ _ (by rw [hlen]; omega)]
    congr 1
    omega
  -- membership in the prefix yields a smaller index with equal value
  obtain ⟨i, hilt, hgeti⟩ := List.getElem_of_mem hmem
  have hir : i < n % 7 := by
    have := hilt
    rw [List.length_take, hdlen] at this
    omega
  have htake_i : (((w.map (·.piece)).drop (7 * (n / 7))).take
      (n % 7))[i]'hilt
      = ((w.map (·.piece)).drop (7 * (n / 7)))[i]'(by
          rw [hdlen]; omega) := by
    rw [List.getElem_take]
  -- lift both to the seven-block and contradict nodup
  have hbi : (((w.map (·.piece)).drop (7 * (n / 7))).take 7)[i]'(by
      rw [hblk_len]; omega)
      = ((w.map (·.piece)).drop (7 * (n / 7)))[i]'(by
          rw [hdlen]; omega) := by
    rw [List.getElem_take]
  have hbr : (((w.map (·.piece)).drop (7 * (n / 7))).take 7)[n % 7]'(by
      rw [hblk_len]; omega)
      = ((w.map (·.piece)).drop (7 * (n / 7)))[n % 7]'(by
          rw [hdlen]; omega) := by
    rw [List.getElem_take]
  have heq2 : (((w.map (·.piece)).drop (7 * (n / 7))).take 7)[i]'(by
      rw [hblk_len]; omega)
      = (((w.map (·.piece)).drop (7 * (n / 7))).take 7)[n % 7]'(by
          rw [hblk_len]; omega) := by
    rw [hbi, hbr]
    rw [← htake_i, hgeti, hplayed]
  have := (List.Nodup.getElem_inj_iff hblk_nodup).mp heq2
  omega

/-- The orbit of a word play, as a lookup table of states. -/
def wordOrbit (b : Board) (w : List Placement) : List GameState :=
  (List.range w.length).map (wordPlay ⟨b, Bag.full⟩ w)

/-- The table policy that realizes a word: look the state up in the
orbit and play the word's letter at that position. -/
def wordPolicy (b : Board) (w : List Placement) :
    Policy GameConfig.standard :=
  fun g => w.getD ((wordOrbit b w).idxOf g) ⟨Piece.O, 0, 0⟩

/-- The play reduces to its first period. -/
theorem wordPlay_mod {b : Board} {w : List Placement} (hne : w ≠ [])
    (hcyc : stepWord ⟨b, Bag.full⟩ w = ⟨b, Bag.full⟩) :
    ∀ n, wordPlay ⟨b, Bag.full⟩ w n
      = wordPlay ⟨b, Bag.full⟩ w (n % w.length) := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases hlt : n < w.length
    · rw [Nat.mod_eq_of_lt hlt]
    · have hge : w.length ≤ n := by omega
      have hsplit : n = w.length + (n - w.length) := by omega
      rw [hsplit, wordPlay_periodic hne hcyc,
        ih (n - w.length) (by omega), Nat.add_mod_left]

/-- A legal 35-word's orbit table has no repeated states. -/
theorem wordOrbit_nodup {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    (wordOrbit b w).Nodup := by
  have hdis := legal_cycle_thirty_five_distinct_states hwf hne hv
    hbag hfold
  unfold wordOrbit
  apply List.Nodup.map_on
  · intro x hx y hy hxy
    rw [List.mem_range, hlen] at hx hy
    by_contra hne'
    exact hdis x hx y hy hne' hxy
  · exact List.nodup_range ..

/-- **The table policy reads the word**: at the orbit's `i`-th state,
the policy plays exactly the word's `i`-th letter. -/
theorem wordPolicy_eval {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {i : ℕ} (hi : i < 35) :
    wordPolicy b w (wordPlay ⟨b, Bag.full⟩ w i)
      = w.getD i ⟨Piece.O, 0, 0⟩ := by
  have hnodup := wordOrbit_nodup hwf hne hv hbag hfold hlen
  have hdis := legal_cycle_thirty_five_distinct_states hwf hne hv
    hbag hfold
  have hmem : wordPlay ⟨b, Bag.full⟩ w i ∈ wordOrbit b w := by
    unfold wordOrbit
    rw [List.mem_map]
    exact ⟨i, by rw [List.mem_range, hlen]; exact hi, rfl⟩
  have hjlt : (wordOrbit b w).idxOf (wordPlay ⟨b, Bag.full⟩ w i)
      < (wordOrbit b w).length := List.idxOf_lt_length_of_mem hmem
  have hjget : (wordOrbit b w)[(wordOrbit b w).idxOf
      (wordPlay ⟨b, Bag.full⟩ w i)]'hjlt
      = wordPlay ⟨b, Bag.full⟩ w i := List.getElem_idxOf hjlt
  have hlen_orbit : (wordOrbit b w).length = 35 := by
    unfold wordOrbit
    rw [List.length_map, List.length_range, hlen]
  have hjw : ∀ (j : ℕ) (hj : j < (wordOrbit b w).length),
      (wordOrbit b w)[j]'hj = wordPlay ⟨b, Bag.full⟩ w j := by
    intro j hj
    unfold wordOrbit
    simp only [List.getElem_map, List.getElem_range]
  have hji : (wordOrbit b w).idxOf (wordPlay ⟨b, Bag.full⟩ w i)
      = i := by
    by_contra hne'
    apply hdis ((wordOrbit b w).idxOf (wordPlay ⟨b, Bag.full⟩ w i))
      (by have h := hjlt; rw [hlen_orbit] at h; omega) i hi hne'
    rw [← hjw _ hjlt]
    exact hjget
  unfold wordPolicy
  rw [hji]

/-! ### The weighted-column ledger

Every cycle law proved so far weights all ten columns equally. Weighting
column `j` by `w j` instead produces a whole FAMILY of conserved
quantities, because a cleared row takes exactly one cell out of every
column — so a clear always costs exactly `∑ w`, whatever the geometry.
Three choices of `w` give three unrelated laws: the indicator of one
column (exact per-column delivery), the odd-column indicator (a parity
law), and the column index itself (a moment law mod five, the first
cycle law that constrains WHERE pieces are dropped). -/

/-- A weighted count of the board's cells: column `j` contributes its cell
count `w j` times. -/
def colWeight (w : ℕ → ℕ) (b : Board) : ℕ :=
  ∑ j ∈ Finset.range 10, w j * b.colCount j

/-- The same weighting applied to a placement's column profile. -/
def weightedProfile (w : ℕ → ℕ) (pl : Placement) : ℕ :=
  ∑ j ∈ Finset.range 10, w j * pl.colProfile j

/-- **The weighted profile is a cell sum**: weighting columns and summing
against the profile is the same as weighting each of the piece's four
cells by the column it lands in. -/
theorem weightedProfile_eq_cell_sum {w : ℕ → ℕ} {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    weightedProfile w pl = ∑ cell ∈ pl.shapeUp, w (pl.col + cell.1) := by
  classical
  have hmaps : ∀ cell ∈ pl.shapeUp, pl.col + cell.1 ∈ Finset.range 10 := by
    intro cell hcell
    have := hv cell hcell
    rw [GameConfig.standard_cols] at this
    exact Finset.mem_range.mpr this
  unfold weightedProfile
  rw [← Finset.sum_fiberwise_of_maps_to hmaps
    (fun cell => w (pl.col + cell.1))]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  have hval : ∀ cell ∈ pl.shapeUp.filter (fun cell => pl.col + cell.1 = j),
      w (pl.col + cell.1) = w j := by
    intro cell hcell
    rw [(Finset.mem_filter.mp hcell).2]
  rw [Finset.sum_congr rfl hval, Finset.sum_const, smul_eq_mul, mul_comm]
  rfl

/-- **The weighted ledger for one move**: the weighted count grows by the
piece's weighted profile and falls by the total weight per cleared row —
a cleared row surrenders exactly one cell from each column. -/
theorem colWeight_applyStep (w : ℕ → ℕ) (b : Board) (pl : Placement) :
    colWeight w (Placement.applyStep GameConfig.standard b pl)
      + (∑ j ∈ Finset.range 10, w j)
        * (Board.fullRows GameConfig.standard (pl.place b)).card
      = colWeight w b + weightedProfile w pl := by
  unfold colWeight weightedProfile
  rw [Finset.sum_mul, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun j hj => ?_)
  have h := applyStep_colCount GameConfig.standard b pl
    (j := j) (by rw [GameConfig.standard_cols]; exact Finset.mem_range.mp hj)
  rw [Board.linesCleared] at h
  rw [← Nat.mul_add, ← Nat.mul_add, h]

/-- The word's total weighted profile. -/
def wordWeightedProfile (w : ℕ → ℕ) (pls : List Placement) : ℕ :=
  (pls.map (weightedProfile w)).sum

/-- **The weighted ledger along a word.** -/
theorem colWeight_word (w : ℕ → ℕ) {b : Board} {pls : List Placement} :
    colWeight w (pls.foldl (Placement.applyStep GameConfig.standard) b)
      + (∑ j ∈ Finset.range 10, w j) * wordClears b pls
      = colWeight w b + wordWeightedProfile w pls := by
  induction pls generalizing b with
  | nil => simp [wordWeightedProfile]
  | cons pl rest ih =>
    have hstep := colWeight_applyStep w b pl
    have hrec := ih (b := Placement.applyStep GameConfig.standard b pl)
    rw [List.foldl_cons, wordClears_cons]
    unfold wordWeightedProfile at hrec ⊢
    rw [List.map_cons, List.sum_cons, Nat.mul_add]
    omega

/-- **THE WEIGHTED CYCLE LAW**: a word that folds the board back to itself
pays its total column weight exactly once per cleared row. One equation
for every weight function — an infinite family of cycle invariants. -/
theorem cycle_weighted_law (w : ℕ → ℕ) {b : Board} {pls : List Placement}
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    (∑ j ∈ Finset.range 10, w j) * wordClears b pls
      = wordWeightedProfile w pls := by
  have h := colWeight_word w (b := b) (pls := pls)
  rw [hfold] at h
  omega

/-! #### Weight one: a single column — exact delivery -/

/-- The word's total delivery to column `c`. -/
def wordColProfile (c : ℕ) (pls : List Placement) : ℕ :=
  (pls.map (fun pl => pl.colProfile c)).sum

/-- The per-column ledger along a word. -/
theorem colCount_word {c : ℕ} (hc : c < 10) {b : Board}
    {pls : List Placement} :
    (pls.foldl (Placement.applyStep GameConfig.standard) b).colCount c
      + wordClears b pls
      = b.colCount c + wordColProfile c pls := by
  induction pls generalizing b with
  | nil => simp [wordColProfile]
  | cons pl rest ih =>
    have hstep := applyStep_colCount GameConfig.standard b pl
      (j := c) (by rw [GameConfig.standard_cols]; exact hc)
    rw [Board.linesCleared] at hstep
    have hrec := ih (b := Placement.applyStep GameConfig.standard b pl)
    rw [List.foldl_cons, wordClears_cons]
    unfold wordColProfile at hrec ⊢
    rw [List.map_cons, List.sum_cons]
    omega

/-- **EVERY COLUMN IS FED EXACTLY THE CLEAR COUNT**: around any board
cycle each of the ten columns receives exactly as many cells as the cycle
clears rows. The delivery is perfectly even across the board — no column
can be favoured, whatever the geometry. -/
theorem cycle_column_delivery {c : ℕ} (hc : c < 10) {b : Board}
    {pls : List Placement}
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordColProfile c pls = wordClears b pls := by
  have h := colCount_word hc (b := b) (pls := pls)
  rw [hfold] at h
  omega

/-- A legal 35-cycle delivers exactly FOURTEEN cells to every one of the
ten columns — 140 cells split ten ways with no remainder. -/
theorem legal_cycle_column_fourteen {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {c : ℕ} (hc : c < 10) :
    wordColProfile c w = 14 := by
  rw [cycle_column_delivery hc hfold]
  exact legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen

/-- Each column is fed on at least four of a legal 35-cycle's moves: a
single placement can drop at most four cells into one column. -/
theorem legal_cycle_column_feed_moves {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {c : ℕ} (hc : c < 10) :
    14 ≤ 4 * (w.filter (fun pl => 0 < pl.colProfile c)).length := by
  classical
  have h14 := legal_cycle_column_fourteen hwf hne hv hbag hfold hlen hc
  have hbound : ∀ (l : List Placement),
      (l.map (fun pl => pl.colProfile c)).sum
        ≤ 4 * (l.filter (fun pl => 0 < pl.colProfile c)).length := by
    intro l
    induction l with
    | nil => simp
    | cons pl rest ih =>
      have hle : pl.colProfile c ≤ 4 := by
        unfold Placement.colProfile
        calc (pl.shapeUp.filter (fun cell => pl.col + cell.1 = c)).card
            ≤ pl.shapeUp.card := Finset.card_filter_le _ _
          _ = 4 := pl.shapeUp_card
      rw [List.map_cons, List.sum_cons, List.filter_cons]
      by_cases hpos : 0 < pl.colProfile c
      · rw [if_pos (by simpa using hpos)]
        rw [List.length_cons]
        omega
      · rw [if_neg (by simpa using hpos)]
        omega
  unfold wordColProfile at h14
  have := hbound w
  omega

/-! #### Weight two: the odd columns — a parity law -/

/-- The odd-column indicator weight. -/
def oddW : ℕ → ℕ := fun j => if j % 2 = 1 then 1 else 0

theorem sum_oddW : ∑ j ∈ Finset.range 10, oddW j = 5 := by decide

/-- **The odd-column charge is a SHAPE invariant**: modulo two, how many
of a piece's four cells land in odd columns does not depend on where the
piece is dropped. Shifting the drop column by one swaps the odd cells for
the even ones, and a four-cell shape has `4 - k ≡ k`. -/
theorem oddProfile_col_free {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    weightedProfile oddW pl % 2
      = (pl.shapeUp.filter (fun cell => cell.1 % 2 = 1)).card % 2 := by
  classical
  rw [weightedProfile_eq_cell_sum hv]
  unfold oddW
  rw [← Finset.card_filter]
  have hcol : pl.col % 2 = 0 ∨ pl.col % 2 = 1 := by omega
  rcases hcol with hc0 | hc1
  · congr 2
    apply Finset.filter_congr
    intro cell _
    omega
  · have hcompl : pl.shapeUp.filter (fun cell => (pl.col + cell.1) % 2 = 1)
        = pl.shapeUp.filter (fun cell => ¬ (cell.1 % 2 = 1)) := by
      apply Finset.filter_congr
      intro cell _
      omega
    rw [hcompl]
    have hsum := Finset.filter_card_add_filter_neg_card_eq_card
      (s := pl.shapeUp) (p := fun cell => cell.1 % 2 = 1)
    have h4 := pl.shapeUp_card
    omega

/-- **THE ODD-COLUMN PARITY LAW**: around any cycle the pieces' total
odd-column delivery and the cleared-row count have the same parity. Unlike
the checkerboard charge, this one SURVIVES line clears — gravity never
moves a cell between columns. -/
theorem cycle_odd_parity {b : Board} {pls : List Placement}
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordWeightedProfile oddW pls % 2 = wordClears b pls % 2 := by
  have h := cycle_weighted_law oddW hfold
  rw [sum_oddW] at h
  omega

/-- A legal 35-cycle's total odd-column delivery is EVEN, since it clears
fourteen rows. -/
theorem legal_cycle_odd_delivery_even {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordWeightedProfile oddW w % 2 = 0 := by
  rw [cycle_odd_parity hfold,
    legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen]

/-! #### Weight three: the column index — a moment law mod five -/

theorem sum_idW : ∑ j ∈ Finset.range 10, j = 45 := by decide

/-- A piece's own column moment: the sum of its cells' column offsets. -/
def shapeMoment (pl : Placement) : ℕ := ∑ cell ∈ pl.shapeUp, cell.1

/-- **The moment of a drop**: weighting columns by their index, a
placement contributes four times its drop column plus the shape's own
moment. This is the first cycle quantity that MENTIONS the drop column. -/
theorem weightedProfile_id {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    weightedProfile (fun j => j) pl = 4 * pl.col + shapeMoment pl := by
  rw [weightedProfile_eq_cell_sum hv]
  unfold shapeMoment
  rw [Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul,
    pl.shapeUp_card]

/-- **THE MOMENT LAW MOD FIVE**: a full row carries column moment
`0+1+…+9 = 45`, which vanishes mod five, so clears are INVISIBLE to the
column moment mod five. Around any cycle the drops' moments must
therefore vanish mod five — a constraint on where the pieces are placed,
not merely on which pieces are played. -/
theorem cycle_moment_mod_five {b : Board} {pls : List Placement}
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordWeightedProfile (fun j => j) pls % 5 = 0 := by
  have h := cycle_weighted_law (fun j => j) hfold
  rw [sum_idW] at h
  omega

/-- The moment law in drop coordinates: around a cycle, four times the sum
of the drop columns plus the sum of the shape moments is divisible by
five. -/
theorem cycle_drop_column_moment {b : Board} {pls : List Placement}
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    5 ∣ (pls.map (fun pl => 4 * pl.col + shapeMoment pl)).sum := by
  have h := cycle_moment_mod_five hfold
  have hmap : (pls.map (fun pl => 4 * pl.col + shapeMoment pl)).sum
      = wordWeightedProfile (fun j => j) pls := by
    unfold wordWeightedProfile
    congr 1
    apply List.map_congr_left
    intro pl hpl
    exact (weightedProfile_id (hv pl hpl)).symm
  rw [hmap]
  omega

/-! ### How much the weighted family really knows

The weighted ledger produced one equation per weight function, which
looks like an infinite supply of constraints. It is not: because the
per-column delivery is *exactly* the clear count for every column, the
word's weighted profile is forced for every `w` at once. The family
collapses to its first instance.

What does NOT collapse is the DECOMPOSITION of each weighted profile
into drop coordinates. Reading the same equations in terms of where the
pieces were dropped turns two of them from congruences into exact
Diophantine equations on the drop-column multiset — one linear, one
quadratic. -/

/-- **The word's weighted profile is the weighted sum of the per-column
deliveries.** -/
theorem wordWeightedProfile_eq_sum (w : ℕ → ℕ) (pls : List Placement) :
    wordWeightedProfile w pls
      = ∑ j ∈ Finset.range 10, w j * wordColProfile j pls := by
  induction pls with
  | nil => simp [wordWeightedProfile, wordColProfile]
  | cons pl rest ih =>
    unfold wordWeightedProfile at ih ⊢
    rw [List.map_cons, List.sum_cons, ih]
    unfold weightedProfile
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [← Nat.mul_add]
    rfl

/-- **THE WEIGHTED FAMILY COLLAPSES**: every weighted cycle law is a
consequence of the single per-column delivery law. Mining further weight
functions cannot produce new information — the boundary of the method,
stated as a theorem rather than guessed at. -/
theorem cycle_weighted_of_delivery (w : ℕ → ℕ) {b : Board}
    {pls : List Placement}
    (hdel : ∀ c, c < 10 → wordColProfile c pls = wordClears b pls) :
    wordWeightedProfile w pls
      = (∑ j ∈ Finset.range 10, w j) * wordClears b pls := by
  rw [wordWeightedProfile_eq_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl (fun j hj => ?_)
  rw [hdel j (Finset.mem_range.mp hj)]

/-! #### The exact drop-column equations -/

/-- A piece's quadratic column moment: the sum of the squares of its
cells' column offsets. -/
def shapeQuad (pl : Placement) : ℕ := ∑ cell ∈ pl.shapeUp, cell.1 * cell.1

theorem sum_sqW : ∑ j ∈ Finset.range 10, j * j = 285 := by decide

/-- **The quadratic moment of a drop**, in drop coordinates. -/
theorem weightedProfile_sq {pl : Placement}
    (hv : pl.Valid GameConfig.standard) :
    weightedProfile (fun j => j * j) pl
      = 4 * (pl.col * pl.col) + 2 * pl.col * shapeMoment pl
        + shapeQuad pl := by
  rw [weightedProfile_eq_cell_sum hv]
  unfold shapeMoment shapeQuad
  have hexp : ∀ cell ∈ pl.shapeUp,
      (pl.col + cell.1) * (pl.col + cell.1)
        = pl.col * pl.col + 2 * pl.col * cell.1 + cell.1 * cell.1 := by
    intro cell _
    ring
  rw [Finset.sum_congr rfl hexp, Finset.sum_add_distrib,
    Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul,
    pl.shapeUp_card, ← Finset.mul_sum]

/-- **THE EXACT MOMENT EQUATION**: around any cycle, four times the sum
of the drop columns plus the sum of the shape moments equals forty-five
per cleared row — an exact equation, not a congruence. -/
theorem cycle_moment_exact {b : Board} {pls : List Placement}
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    (pls.map (fun pl => 4 * pl.col + shapeMoment pl)).sum
      = 45 * wordClears b pls := by
  have h := cycle_weighted_law (fun j => j) hfold
  rw [sum_idW] at h
  have hmap : (pls.map (fun pl => 4 * pl.col + shapeMoment pl)).sum
      = wordWeightedProfile (fun j => j) pls := by
    unfold wordWeightedProfile
    congr 1
    apply List.map_congr_left
    intro pl hpl
    exact (weightedProfile_id (hv pl hpl)).symm
  rw [hmap, ← h]

/-- A legal 35-cycle's drop columns satisfy an exact linear equation:
`∑ (4·col + shapeMoment) = 630`. -/
theorem legal_cycle_moment_630 {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    (w.map (fun pl => 4 * pl.col + shapeMoment pl)).sum = 630 := by
  rw [cycle_moment_exact hv hfold,
    legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen]

/-- **THE EXACT QUADRATIC MOMENT EQUATION**: the drop columns satisfy a
second, independent equation — quadratic this time — at two hundred
eighty-five per cleared row. -/
theorem cycle_quad_moment_exact {b : Board} {pls : List Placement}
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    (pls.map (fun pl => 4 * (pl.col * pl.col)
        + 2 * pl.col * shapeMoment pl + shapeQuad pl)).sum
      = 285 * wordClears b pls := by
  have h := cycle_weighted_law (fun j => j * j) hfold
  rw [sum_sqW] at h
  have hmap : (pls.map (fun pl => 4 * (pl.col * pl.col)
        + 2 * pl.col * shapeMoment pl + shapeQuad pl)).sum
      = wordWeightedProfile (fun j => j * j) pls := by
    unfold wordWeightedProfile
    congr 1
    apply List.map_congr_left
    intro pl hpl
    exact (weightedProfile_sq (hv pl hpl)).symm
  rw [hmap, ← h]

/-- A legal 35-cycle's drop columns satisfy the exact quadratic equation
`∑ (4·col² + 2·col·shapeMoment + shapeQuad) = 3990`. -/
theorem legal_cycle_quad_3990 {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    (w.map (fun pl => 4 * (pl.col * pl.col)
      + 2 * pl.col * shapeMoment pl + shapeQuad pl)).sum = 3990 := by
  rw [cycle_quad_moment_exact hv hfold,
    legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen]

/-! ### The burial ledger: what a cycle does with its holes

Mass and column-delivery are conserved around a cycle; so is the hole
debt, but far less trivially, because holes enter and leave by two
different mechanisms. Placements BURY cells (a piece bridging a gap
leaves empties underneath) and clears UNBURY them (removing a column's
top cells can lift its holes above the new skyline). Around a cycle the
two must balance exactly.

The missing ingredient is that a placement never *reduces* the debt.
The library proves this via the geometric hole set, at the price of
in-field hypotheses on every board. The route below is hypothesis-free:
a column's height rises by at least the number of cells it receives,
because the dropped cells in that column sit at distinct rows, all at
or above the old skyline. -/

/-- **A sup lower bound from injectivity**: `card` distinct values, all
above `h`, force the supremum up to `h + card`. -/
theorem sup_ge_add_card {α : Type*} {S : Finset α} {f : α → ℕ} {h : ℕ}
    (hne : S.Nonempty) (hinj : Set.InjOn f S)
    (hlow : ∀ x ∈ S, h + 1 ≤ f x) :
    h + S.card ≤ S.sup f := by
  classical
  have hsub : S.image f ⊆ Finset.Icc (h + 1) (S.sup f) := by
    intro y hy
    rw [Finset.mem_image] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    rw [Finset.mem_Icc]
    exact ⟨hlow x hx, Finset.le_sup hx⟩
  have hcard : (S.image f).card = S.card :=
    Finset.card_image_of_injOn hinj
  have hle := Finset.card_le_card hsub
  rw [Nat.card_Icc, hcard] at hle
  obtain ⟨x0, hx0⟩ := hne
  have h1 := hlow x0 hx0
  have h2 : f x0 ≤ S.sup f := Finset.le_sup hx0
  omega

/-- **A column's skyline rises by at least what it receives**: the
dropped cells of column `j` occupy distinct rows, all at or above the
old height, so the new height clears the old by at least the column
profile. Hypothesis-free — no validity, no in-field assumption. -/
theorem colHeight_place_ge_add_colProfile (b : Board) (pl : Placement)
    (j : ℕ) :
    b.colHeight j + pl.colProfile j ≤ (pl.place b).colHeight j := by
  classical
  rw [SurfaceCalculus.colHeight_place_eq]
  have hprof : pl.colProfile j = (SurfaceCalculus.cellsInCol pl j).card :=
    rfl
  by_cases hne : (SurfaceCalculus.cellsInCol pl j).Nonempty
  · have hlow : ∀ cell ∈ SurfaceCalculus.cellsInCol pl j,
        b.colHeight j + 1 ≤ pl.dropOffset b + cell.2 + 1 := by
      intro cell hcell
      rw [SurfaceCalculus.cellsInCol, Finset.mem_filter] at hcell
      have hle := Finset.le_sup
        (f := fun c : PieceCell => b.colHeight (pl.col + c.1) - c.2)
        hcell.1
      have hD : b.colHeight (pl.col + cell.1) - cell.2
          ≤ pl.dropOffset b := by
        rw [Placement.dropOffset_eq_sup]
        simpa using hle
      rw [hcell.2] at hD
      omega
    have hinj : Set.InjOn
        (fun cell : PieceCell => pl.dropOffset b + cell.2 + 1)
        (SurfaceCalculus.cellsInCol pl j) := by
      intro x hx y hy hxy
      have hx2 : pl.col + x.1 = j := by
        have hx' := Finset.mem_coe.mp hx
        rw [SurfaceCalculus.cellsInCol, Finset.mem_filter] at hx'
        exact hx'.2
      have hy2 : pl.col + y.1 = j := by
        have hy' := Finset.mem_coe.mp hy
        rw [SurfaceCalculus.cellsInCol, Finset.mem_filter] at hy'
        exact hy'.2
      have hxy' : pl.dropOffset b + x.2 + 1 = pl.dropOffset b + y.2 + 1 :=
        hxy
      exact Prod.ext_iff.mpr ⟨by omega, by omega⟩
    have hsup := sup_ge_add_card (S := SurfaceCalculus.cellsInCol pl j)
      (f := fun cell => pl.dropOffset b + cell.2 + 1) hne hinj hlow
    rw [hprof]
    exact le_trans hsup (le_max_right _ _)
  · rw [Finset.not_nonempty_iff_eq_empty] at hne
    rw [hprof, hne]
    simp

/-- **Placements never discharge debt** — hypothesis-free. Each column
gains exactly its profile in cells and at least its profile in height,
so no column's hole count can fall. -/
theorem debt_le_debt_place (b : Board) (pl : Placement) :
    HoleDebt.debt GameConfig.standard b
      ≤ HoleDebt.debt GameConfig.standard (pl.place b) := by
  unfold HoleDebt.debt
  apply Finset.sum_le_sum
  intro j _
  unfold HoleDebt.colHoles
  have hh := colHeight_place_ge_add_colProfile b pl j
  have hc : ((pl.place b).colRows j).card
      = (b.colRows j).card + pl.colProfile j := by
    rw [HoleDebt.card_colRows_eq_card_filter,
      HoleDebt.card_colRows_eq_card_filter]
    have h := Placement.colCount_place b pl j
    unfold Board.colCount at h
    exact h
  omega

/-- Clears never raise debt (the mirror fact, from the library). -/
theorem debt_applyStep_le (b : Board) (pl : Placement)
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    HoleDebt.debt GameConfig.standard
        (Placement.applyStep GameConfig.standard b pl)
      ≤ HoleDebt.debt GameConfig.standard (pl.place b) := by
  rw [Placement.applyStep_eq_clearLines_place]
  exact HoleDebt.clearLines_debt_le (Placement.place_wf hwf hv)

/-- Holes buried by the placements of a word. -/
def wordBuried (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (HoleDebt.debt GameConfig.standard (pl.place b)
        - HoleDebt.debt GameConfig.standard b)
      + wordBuried (Placement.applyStep GameConfig.standard b pl) rest

/-- Holes unburied by the clears of a word. -/
def wordUnburied (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (HoleDebt.debt GameConfig.standard (pl.place b)
        - HoleDebt.debt GameConfig.standard
            (Placement.applyStep GameConfig.standard b pl))
      + wordUnburied (Placement.applyStep GameConfig.standard b pl) rest

@[simp] theorem wordBuried_nil (b : Board) : wordBuried b [] = 0 := rfl

@[simp] theorem wordUnburied_nil (b : Board) : wordUnburied b [] = 0 := rfl

theorem wordBuried_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordBuried b (pl :: rest)
      = (HoleDebt.debt GameConfig.standard (pl.place b)
          - HoleDebt.debt GameConfig.standard b)
        + wordBuried (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

theorem wordUnburied_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordUnburied b (pl :: rest)
      = (HoleDebt.debt GameConfig.standard (pl.place b)
          - HoleDebt.debt GameConfig.standard
              (Placement.applyStep GameConfig.standard b pl))
        + wordUnburied (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- **The burial ledger along a word**: starting debt plus everything
buried equals ending debt plus everything unburied. -/
theorem debt_word {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    HoleDebt.debt GameConfig.standard b + wordBuried b pls
      = HoleDebt.debt GameConfig.standard
          (pls.foldl (Placement.applyStep GameConfig.standard) b)
        + wordUnburied b pls := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    have h1 := debt_le_debt_place b pl
    have h2 := debt_applyStep_le b pl hwf hvpl
    have hrec := ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, wordBuried_cons, wordUnburied_cons]
    omega

/-- **THE BURIAL CONSERVATION LAW**: around any cycle, the holes the
placements bury are exactly the holes the clears set free. A cycle
cannot bury on credit — every cell it covers must be uncovered again
before the loop closes. -/
theorem cycle_burial_conservation {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordBuried b pls = wordUnburied b pls := by
  have h := debt_word hwf hv
  rw [hfold] at h
  omega

/-- **Flush play and hole recycling stand or fall together**: a cycle
buries nothing exactly when its clears free nothing. Either the loop is
played perfectly clean, or it both digs and recovers — there is no
half-way cycle that only digs, and none that only recovers. -/
theorem cycle_flush_iff_no_recycling {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordBuried b pls = 0 ↔ wordUnburied b pls = 0 := by
  have h := cycle_burial_conservation hwf hv hfold
  omega

/-! ### Three more angles: schedule, obstruction, coordinates

The laws so far are all TOTALS over the whole cycle. Three orthogonal
refinements: a constraint on every PREFIX (when the clears may happen),
a general principle for REFUTING cycles, and the coordinate structure
the two clocks impose on the orbit. -/

/-! #### Angle one: the clear schedule -/

/-- **You cannot clear faster than you deliver**: at every prefix of a
word, ten times the rows cleared so far is at most the starting mass
plus four per move played. A constraint on the SCHEDULE of clears, not
merely on their total. -/
theorem prefix_clear_bound {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) (i : ℕ) :
    10 * wordClears b (pls.take i) ≤ b.count + 4 * i := by
  have hvt : ∀ pl ∈ pls.take i, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_of_mem_take hpl)
  have h := foldl_count_ledger_exact (b := b) (pls := pls.take i) hwf hvt
  have hlen : (pls.take i).length ≤ i := by
    rw [List.length_take]
    omega
  omega

/-- **…and you cannot fall more than a boardful behind**: while the
board stays inside the field, the mass delivered so far exceeds the mass
cleared by at most one full board. The clears are pinned into a band
around the `0.4`-per-move line at every prefix. -/
theorem prefix_mass_bound {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) {i : ℕ}
    (hi : i ≤ pls.length)
    (hcap : ((pls.take i).foldl
      (Placement.applyStep GameConfig.standard) b).count ≤ 200) :
    b.count + 4 * i ≤ 200 + 10 * wordClears b (pls.take i) := by
  have hvt : ∀ pl ∈ pls.take i, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_of_mem_take hpl)
  have h := foldl_count_ledger_exact (b := b) (pls := pls.take i) hwf hvt
  have hlen : (pls.take i).length = i := by
    rw [List.length_take]
    omega
  rw [hlen] at h
  omega

/-- A legal 35-cycle's opening cannot be clear-heavy: after `i` moves it
has cleared at most `(base mass + 4i)/10` rows. -/
theorem legal_cycle_prefix_clears {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard) (i : ℕ) :
    10 * wordClears b (w.take i) ≤ b.count + 4 * i :=
  prefix_clear_bound hwf hv i

/-! #### Angle two: refuting cycles with a monotone quantity -/

/-- A quantity that never falls under a move never falls along a word. -/
theorem foldl_monotone {Φ : Board → ℕ}
    (hmono : ∀ b pl, Φ b
      ≤ Φ (Placement.applyStep GameConfig.standard b pl))
    (b : Board) (pls : List Placement) :
    Φ b ≤ Φ (pls.foldl (Placement.applyStep GameConfig.standard) b) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [List.foldl_cons]
    exact le_trans (hmono b pl) (ih _)

/-- **Monotone quantities are FROZEN on a cycle**: if `Φ` never falls
under a move, then every board a cycle visits carries the same `Φ`.
Nothing that only ever grows can actually grow inside a loop. -/
theorem cycle_monotone_const {Φ : Board → ℕ}
    (hmono : ∀ b pl, Φ b
      ≤ Φ (Placement.applyStep GameConfig.standard b pl))
    {b : Board} {pls : List Placement}
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b)
    (i : ℕ) :
    Φ ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
      = Φ b := by
  have h1 := foldl_monotone hmono b (pls.take i)
  have h2 := foldl_monotone hmono
    ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
    (pls.drop i)
  have h3 : (pls.drop i).foldl (Placement.applyStep GameConfig.standard)
      ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
      = pls.foldl (Placement.applyStep GameConfig.standard) b := by
    rw [← List.foldl_append, List.take_append_drop]
  rw [h3, hfold] at h2
  omega

/-- **THE MONOTONE OBSTRUCTION**: a single strict increase of a
monotone quantity anywhere along a word refutes the whole cycle. To
prove no cycle passes through a board it therefore suffices to exhibit
one never-falling measure that genuinely rises at one move — a general
recipe for cycle non-existence proofs. -/
theorem no_cycle_of_strict_increase {Φ : Board → ℕ}
    (hmono : ∀ b pl, Φ b
      ≤ Φ (Placement.applyStep GameConfig.standard b pl))
    {b : Board} {pls : List Placement} {i : ℕ}
    (hstrict :
      Φ ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
        < Φ ((pls.take (i + 1)).foldl
              (Placement.applyStep GameConfig.standard) b)) :
    pls.foldl (Placement.applyStep GameConfig.standard) b ≠ b := by
  intro hfold
  have h1 := cycle_monotone_const hmono hfold i
  have h2 := cycle_monotone_const hmono hfold (i + 1)
  omega

/-! #### Angle three: the orbit's coordinates -/

/-- Every residue pair modulo five and seven is realized below 35. -/
theorem crt_five_seven :
    ∀ a < 5, ∀ b < 7, ∃ i < 35, i % 5 = a ∧ i % 7 = b := by decide

/-- …and realized only once. -/
theorem crt_five_seven_unique {i j : ℕ} (hi : i < 35) (hj : j < 35)
    (h5 : i % 5 = j % 5) (h7 : i % 7 = j % 7) : i = j := by omega

/-- **THE ORBIT IS A FIVE-BY-SEVEN TORUS**: the mass clock reads the
position modulo five and the bag clock reads it modulo seven, so the
pair of clocks is a COORDINATE SYSTEM on a legal cycle's orbit — every
combination of a mass residue and a bag size occurs, and occurs exactly
once, among the thirty-five states. -/
theorem legal_cycle_orbit_torus {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    ∀ a < 5, ∀ c < 7, ∃ i < 35,
      (wordPlay ⟨b, Bag.full⟩ w i).bag.card = 7 - c
      ∧ (wordPlay ⟨b, Bag.full⟩ w i).board.count % 10
          = (b.count + 4 * a) % 10 := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have h7 : 7 ∣ w.length := by omega
  intro a ha c hc
  obtain ⟨i, hi, h5i, h7i⟩ := crt_five_seven a ha c hc
  refine ⟨i, hi, ?_, ?_⟩
  · rw [wordPlay_bag_card (b := b) hbag h7 (by omega), h7i]
  · rw [wordPlay_count_mod (b := b) (w := w) hwf hv (by omega)]
    omega

/-- The coordinates separate: two orbit positions agreeing on both
clocks are the same position. -/
theorem legal_cycle_coordinates_faithful {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    {i j : ℕ} (hi : i < 35) (hj : j < 35)
    (hbagcard : (wordPlay ⟨b, Bag.full⟩ w i).bag.card
      = (wordPlay ⟨b, Bag.full⟩ w j).bag.card)
    (hcount : (wordPlay ⟨b, Bag.full⟩ w i).board.count % 10
      = (wordPlay ⟨b, Bag.full⟩ w j).board.count % 10) :
    i = j := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have h7 : 7 ∣ w.length := by omega
  have hbi := wordPlay_bag_card (b := b) hbag h7 (show i ≤ w.length by omega)
  have hbj := wordPlay_bag_card (b := b) hbag h7 (show j ≤ w.length by omega)
  have hci := wordPlay_count_mod (b := b) (w := w) hwf hv
    (show i ≤ w.length by omega)
  have hcj := wordPlay_count_mod (b := b) (w := w) hwf hv
    (show j ≤ w.length by omega)
  rw [hbi, hbj] at hbagcard
  rw [hci, hcj] at hcount
  omega

/-! ### The obstruction sharpened, and two things it catches

The monotone obstruction asked for a quantity monotone on ALL boards,
which is a heavy demand. Only monotonicity ALONG THE WORD is ever used,
and that is a far weaker hypothesis — one that ordinary board measures
actually satisfy. Sharpening it immediately catches two facts. -/

/-- **Word-local monotonicity suffices**: a quantity that never falls
across the moves of a cycle is constant on every board the cycle
visits. No global hypothesis on `Φ` at all. -/
theorem cycle_prefix_monotone_const {Φ : Board → ℕ} {b : Board}
    {pls : List Placement}
    (hmono : ∀ i < pls.length,
      Φ ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
        ≤ Φ ((pls.take (i + 1)).foldl
              (Placement.applyStep GameConfig.standard) b))
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    ∀ i ≤ pls.length,
      Φ ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
        = Φ b := by
  have hchain : ∀ j, j ≤ pls.length → ∀ i, i ≤ j →
      Φ ((pls.take i).foldl (Placement.applyStep GameConfig.standard) b)
        ≤ Φ ((pls.take j).foldl
              (Placement.applyStep GameConfig.standard) b) := by
    intro j
    induction j with
    | zero =>
      intro _ i hi
      have hi0 : i = 0 := by omega
      rw [hi0]
    | succ k ihk =>
      intro hk i hi
      by_cases hcase : i = k + 1
      · rw [hcase]
      · exact le_trans (ihk (by omega) i (by omega)) (hmono k (by omega))
  intro i hi
  have h1 := hchain i hi 0 (by omega)
  have h2 := hchain pls.length (le_refl _) i hi
  rw [List.take_zero, List.foldl_nil] at h1
  rw [List.take_length, hfold] at h2
  omega

/-- **A move never leaves the mass unchanged**: four cells in, ten per
cleared row out, and `10k = 4` has no solution. -/
theorem move_count_ne {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    (Placement.applyStep GameConfig.standard b pl).count ≠ b.count := by
  have h := BagGrowth.count_applyStep_add (cfg := GameConfig.standard)
    hwf hv
  rw [GameConfig.standard_cols] at h
  omega

/-- **EVERY CYCLE MUST SHED MASS SOMEWHERE**: a non-empty cycle has a
move at which the board's cell count strictly falls. Mass cannot merely
drift upward and land back home — the loop has to breathe out. -/
theorem cycle_has_mass_drop {b : Board} {pls : List Placement}
    (hne : pls ≠ [])
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    ∃ i < pls.length,
      ((pls.take (i + 1)).foldl
          (Placement.applyStep GameConfig.standard) b).count
        < ((pls.take i).foldl
            (Placement.applyStep GameConfig.standard) b).count := by
  by_contra hcon
  push Not at hcon
  have hconst := cycle_prefix_monotone_const (Φ := fun x => x.count)
    hcon hfold
  cases pls with
  | nil => exact hne rfl
  | cons pl rest =>
    have h1 := hconst 1 (by simp)
    have htake : ((pl :: rest).take 1).foldl
        (Placement.applyStep GameConfig.standard) b
        = Placement.applyStep GameConfig.standard b pl := by
      simp
    rw [htake] at h1
    exact move_count_ne hwf (hv pl (by simp)) h1

/-- Flushness of a word, unfolded one move at a time: burying nothing
means the first drop creates no hole and the rest of the word buries
nothing either. -/
theorem wordBuried_eq_zero_cons {b : Board} {pl : Placement}
    {rest : List Placement} :
    wordBuried b (pl :: rest) = 0
      ↔ HoleDebt.debt GameConfig.standard (pl.place b)
            = HoleDebt.debt GameConfig.standard b
        ∧ wordBuried (Placement.applyStep GameConfig.standard b pl)
            rest = 0 := by
  have hle := debt_le_debt_place b pl
  rw [wordBuried_cons]
  constructor
  · intro h
    exact ⟨by omega, by omega⟩
  · rintro ⟨h1, h2⟩
    omega

/-- **A cycle that recycles nothing lands its first piece perfectly
flush.** With `board_on_cycle_shift` rotating the word, this pins every
move of such a cycle: no drop may leave a single cell buried. -/
theorem cycle_first_move_flush {b : Board} {pl : Placement}
    {rest : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ q ∈ pl :: rest, q.Valid GameConfig.standard)
    (hfold : (pl :: rest).foldl
      (Placement.applyStep GameConfig.standard) b = b)
    (hnorecycle : wordUnburied b (pl :: rest) = 0) :
    HoleDebt.debt GameConfig.standard (pl.place b)
      = HoleDebt.debt GameConfig.standard b := by
  have hcons := cycle_burial_conservation hwf hv hfold
  rw [hnorecycle] at hcons
  exact (wordBuried_eq_zero_cons.mp hcons).1

/-! ### The bottom row: a geometric application of the obstruction

The bottom row is special. Placements can only ADD to it, and a clear
that does not take row zero leaves it exactly alone — nothing can fall
into the bottom row unless the bottom row itself is removed. So the
bottom row is a monotone quantity along any word that never fills it,
and the obstruction principle freezes it. -/

/-- The cells sitting in the bottom row. -/
def bottomCells (b : Board) : Finset Coord := b.filter (fun p => p.2 = 0)

/-- Placements only ever add to the bottom row. -/
theorem bottomCells_subset_place (b : Board) (pl : Placement) :
    bottomCells b ⊆ bottomCells (pl.place b) := by
  unfold bottomCells
  apply Finset.filter_subset_filter
  rw [Placement.place_eq_union_dropped]
  exact Finset.subset_union_left

/-- **A clear that spares the bottom row leaves it untouched**: nothing
can fall into row zero unless row zero itself is cleared, because a cell
reaching row zero from above would need every row beneath it full. -/
theorem bottomCells_clearLines {b : Board}
    (hnf : ¬ Board.isFull GameConfig.standard b 0) :
    bottomCells (Board.clearLines GameConfig.standard b) = bottomCells b := by
  classical
  have hcb0 : Board.clearedBelow GameConfig.standard b 0 = 0 := by
    unfold Board.clearedBelow
    rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro x hx
    have := (Finset.mem_filter.mp hx).2
    omega
  ext p
  unfold bottomCells
  simp only [Finset.mem_filter]
  constructor
  · rintro ⟨hp, hp0⟩
    rw [Board.mem_clearLines_iff] at hp
    obtain ⟨q, hq, hqnf, hqp⟩ := hp
    have h2 : q.2 - Board.clearedBelow GameConfig.standard b q.2 = p.2 :=
      congrArg Prod.snd hqp
    have hle := clearedBelow_le GameConfig.standard b q.2
    have hq0 : q.2 = 0 := by
      by_contra hne
      have heq : Board.clearedBelow GameConfig.standard b q.2 = q.2 := by
        omega
      have hsub : (Board.fullRows GameConfig.standard b).filter (· < q.2)
          ⊆ Finset.range q.2 := by
        intro x hx
        exact Finset.mem_range.mpr (Finset.mem_filter.mp hx).2
      have hcard : (Finset.range q.2).card
          ≤ ((Board.fullRows GameConfig.standard b).filter (· < q.2)).card := by
        rw [Finset.card_range]
        unfold Board.clearedBelow at heq
        omega
      have hfe := Finset.eq_of_subset_of_card_le hsub hcard
      have h0mem : (0 : ℕ) ∈ (Board.fullRows GameConfig.standard b).filter
          (· < q.2) := by
        rw [hfe]
        exact Finset.mem_range.mpr (by omega)
      have := (Finset.mem_filter.mp h0mem).1
      exact hnf (Board.isFull_of_mem_fullRows this)
    have hpq : p = q := by
      rw [← hqp, hq0, hcb0]
      exact Prod.ext_iff.mpr ⟨rfl, by omega⟩
    rw [hpq]
    exact ⟨hq, by rw [← hq0]⟩
  · rintro ⟨hp, hp0⟩
    refine ⟨?_, hp0⟩
    rw [Board.mem_clearLines_iff]
    refine ⟨p, hp, ?_, ?_⟩
    · rw [hp0]
      exact hnf
    · rw [hp0, hcb0]
      exact Prod.ext_iff.mpr ⟨rfl, by omega⟩

/-- One move never shrinks the bottom row, provided it does not clear
it. -/
theorem bottomCells_card_applyStep_ge {b : Board} {pl : Placement}
    (hnf : ¬ Board.isFull GameConfig.standard (pl.place b) 0) :
    (bottomCells b).card
      ≤ (bottomCells (Placement.applyStep GameConfig.standard b pl)).card := by
  rw [Placement.applyStep_eq_clearLines_place, bottomCells_clearLines hnf]
  exact Finset.card_le_card (bottomCells_subset_place b pl)

/-- The word fills its bottom row at some point. -/
def wordBottomClear (b : Board) : List Placement → Prop
  | [] => False
  | pl :: rest =>
      Board.isFull GameConfig.standard (pl.place b) 0
      ∨ wordBottomClear (Placement.applyStep GameConfig.standard b pl) rest

theorem wordBottomClear_cons {b : Board} {pl : Placement}
    {rest : List Placement} :
    wordBottomClear b (pl :: rest)
      ↔ Board.isFull GameConfig.standard (pl.place b) 0
        ∨ wordBottomClear (Placement.applyStep GameConfig.standard b pl)
            rest := Iff.rfl

/-- A word that never fills the bottom row never shrinks it. -/
theorem bottomCells_card_word_ge {b : Board} {pls : List Placement}
    (hnb : ¬ wordBottomClear b pls) :
    (bottomCells b).card
      ≤ (bottomCells (pls.foldl
          (Placement.applyStep GameConfig.standard) b)).card := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordBottomClear_cons] at hnb
    push Not at hnb
    rw [List.foldl_cons]
    exact le_trans (bottomCells_card_applyStep_ge hnb.1) (ih hnb.2)

/-- **THE BOTTOM-ROW DICHOTOMY**: a cycle either fills its bottom row at
some point, or its pieces never put a single cell there. There is no
middle course — a loop cannot deposit cells in the bottom row and get
them back without ever completing it. (Cycle rotation carries the
statement from the first move to every move.) -/
theorem cycle_bottom_row_frozen {b : Board} {pl : Placement}
    {rest : List Placement}
    (hnb : ¬ wordBottomClear b (pl :: rest))
    (hfold : (pl :: rest).foldl
      (Placement.applyStep GameConfig.standard) b = b) :
    bottomCells (pl.place b) = bottomCells b := by
  rw [wordBottomClear_cons] at hnb
  push Not at hnb
  have hstep := bottomCells_card_applyStep_ge (b := b) (pl := pl) hnb.1
  have htail := bottomCells_card_word_ge hnb.2
  rw [List.foldl_cons] at hfold
  rw [hfold] at htail
  have hplace : bottomCells
      (Placement.applyStep GameConfig.standard b pl)
      = bottomCells (pl.place b) := by
    rw [Placement.applyStep_eq_clearLines_place]
    exact bottomCells_clearLines hnb.1
  rw [hplace] at hstep htail
  exact (Finset.eq_of_subset_of_card_le (bottomCells_subset_place b pl)
    (by omega)).symm

/-! ### The frozen foundation, and what it says about the empty board -/

/-- The cells of the board at or below row `r`. -/
def lowCells (r : ℕ) (b : Board) : Finset Coord :=
  b.filter (fun p => p.2 ≤ r)

/-- **THE FROZEN FOUNDATION**: if no row at or below `r` is full, the
whole band `[0, r]` is fixed by the clear — cells inside it cannot move
(nothing beneath them is removed) and cells above it cannot enter (a
cell dropping to row `≤ r` would need every row beneath it full). The
bottom-row lemma at every depth. -/
theorem lowCells_clearLines {b : Board} {r : ℕ}
    (hnf : ∀ s, s ≤ r → ¬ Board.isFull GameConfig.standard b s) :
    lowCells r (Board.clearLines GameConfig.standard b) = lowCells r b := by
  classical
  have hfull_gt : ∀ t ∈ Board.fullRows GameConfig.standard b, r < t := by
    intro t ht
    by_contra hle
    exact hnf t (by omega) (Board.isFull_of_mem_fullRows ht)
  have hcb_low : ∀ s, s ≤ r →
      Board.clearedBelow GameConfig.standard b s = 0 := by
    intro s hs
    unfold Board.clearedBelow
    rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
    intro x hx
    obtain ⟨hxf, hxlt⟩ := Finset.mem_filter.mp hx
    have := hfull_gt x hxf
    omega
  have hcb_high : ∀ t, r < t →
      Board.clearedBelow GameConfig.standard b t ≤ t - r - 1 := by
    intro t ht
    unfold Board.clearedBelow
    have hsub : (Board.fullRows GameConfig.standard b).filter (· < t)
        ⊆ Finset.Ico (r + 1) t := by
      intro x hx
      obtain ⟨hxf, hxlt⟩ := Finset.mem_filter.mp hx
      rw [Finset.mem_Ico]
      exact ⟨by have := hfull_gt x hxf; omega, hxlt⟩
    have := Finset.card_le_card hsub
    rw [Nat.card_Ico] at this
    omega
  ext p
  unfold lowCells
  simp only [Finset.mem_filter]
  constructor
  · rintro ⟨hp, hp0⟩
    rw [Board.mem_clearLines_iff] at hp
    obtain ⟨q, hq, hqnf, hqp⟩ := hp
    have h2 : q.2 - Board.clearedBelow GameConfig.standard b q.2 = p.2 :=
      congrArg Prod.snd hqp
    have hqlow : q.2 ≤ r := by
      by_contra hgt
      have := hcb_high q.2 (by omega)
      omega
    have hcb := hcb_low q.2 hqlow
    have hpq : p = q := by
      rw [← hqp, hcb]
      exact Prod.ext_iff.mpr ⟨rfl, by omega⟩
    rw [hpq]
    exact ⟨hq, hqlow⟩
  · rintro ⟨hp, hp0⟩
    refine ⟨?_, hp0⟩
    rw [Board.mem_clearLines_iff]
    refine ⟨p, hp, hnf p.2 hp0, ?_⟩
    rw [hcb_low p.2 hp0]
    exact Prod.ext_iff.mpr ⟨rfl, by omega⟩

/-- On the empty board every piece rests on the floor (emptyset form). -/
theorem dropOffset_emptyset (pl : Placement) :
    pl.dropOffset (∅ : Board) = 0 := by
  rw [Placement.dropOffset_eq_sup]
  refine Nat.le_antisymm ?_ (Nat.zero_le _)
  apply Finset.sup_le
  intro cell _
  rw [Board.colHeight_empty]
  omega

/-- **The first piece always lands on the floor**: every shape has a cell
on its own bottom row, and on the empty board the drop offset is zero,
so the opening move necessarily deposits a cell in row zero. -/
theorem bottomCells_place_empty_nonempty (pl : Placement) :
    (bottomCells (pl.place (∅ : Board))).Nonempty := by
  obtain ⟨cell, hcell, hc0⟩ := Piece.shapeUp_row_zero_mem pl.piece pl.rot
  refine ⟨(pl.col + cell.1, 0), ?_⟩
  unfold bottomCells
  rw [Finset.mem_filter]
  refine ⟨?_, rfl⟩
  rw [Placement.place_eq_union_dropped, Finset.mem_union]
  right
  rw [Placement.dropped_eq_image, Finset.mem_image]
  refine ⟨cell, hcell, ?_⟩
  rw [dropOffset_emptyset, hc0]

/-- **ANY CYCLE THROUGH THE EMPTY BOARD MUST CLEAR ITS BOTTOM ROW.** The
opening piece puts a cell on the floor, so by the bottom-row dichotomy
the loop cannot leave row zero alone; it has to complete and clear it.
An unconditional structural fact about every `∅`-cycle, geometry
included. -/
theorem cycle_through_empty_clears_bottom {pl : Placement}
    {rest : List Placement}
    (hfold : (pl :: rest).foldl
      (Placement.applyStep GameConfig.standard) (∅ : Board) = ∅) :
    wordBottomClear (∅ : Board) (pl :: rest) := by
  by_contra hnb
  have h := cycle_bottom_row_frozen hnb hfold
  have hne := bottomCells_place_empty_nonempty pl
  rw [h] at hne
  unfold bottomCells at hne
  simp at hne

/-! ### The checkerboard charge across a clear

The checkerboard charge is the sharpest near-geometric tool in the
library: only the T tetromino is charged, so a clear-free stack's charge
counts its T's. It is normally abandoned the moment a line clears,
because gravity RECOLOURS the board — every cell above a cleared row
changes colour as it falls.

Abandoning it is unnecessary. The recolouring is not arbitrary: a
surviving cell flips exactly when an odd number of rows below it were
cleared. Naming that total — the *gravity work* — turns a destroyed
invariant into an exact law, and one that finally couples the piece
census to WHERE the clears happen. -/

/-- The gravity work of a clear: the cells whose colour it flips, namely
those with an odd number of cleared rows beneath them, counted mod two. -/
def gravityWork (b : Board) : ZMod 2 :=
  ∑ p ∈ b.filter (fun p => ¬ Board.isFull GameConfig.standard b p.2),
    ((Board.clearedBelow GameConfig.standard b p.2 : ℕ) : ZMod 2)

/-- Falling by `d` flips a cell's colour exactly when `d` is odd. -/
theorem cast_sub_two {a d : ℕ} (h : d ≤ a) :
    ((a - d : ℕ) : ZMod 2) = (a : ZMod 2) + (d : ZMod 2) := by
  have hneg : ∀ x : ZMod 2, -x = x := by decide
  rw [Nat.cast_sub h, sub_eq_add_neg, hneg]

/-- The cells a clear removes, taken row by row, are the full rows
themselves. -/
theorem cleared_fiber_eq {b : Board} (hwf : Board.WF GameConfig.standard b)
    {t : ℕ} (ht : t ∈ Board.fullRows GameConfig.standard b) :
    (b.filter (fun p => Board.isFull GameConfig.standard b p.2)).filter
        (fun p => p.2 = t)
      = (Finset.range 10).image (fun c => ((c, t) : Coord)) := by
  have hfull := Board.isFull_of_mem_fullRows ht
  ext p
  simp only [Finset.mem_filter, Finset.mem_image, Finset.mem_range]
  constructor
  · rintro ⟨⟨hpb, _⟩, hp2⟩
    refine ⟨p.1, ?_, ?_⟩
    · have hlt := hwf p hpb
      rwa [GameConfig.standard_cols] at hlt
    · exact Prod.ext_iff.mpr ⟨rfl, hp2.symm⟩
  · rintro ⟨c, hc, rfl⟩
    have hcr : c ∈ Finset.range GameConfig.standard.cols := by
      rw [GameConfig.standard_cols]
      exact Finset.mem_range.mpr hc
    exact ⟨⟨hfull c hcr, hfull⟩, rfl⟩

/-- **The charge a clear removes is its row count**: each full row of the
ten-wide board carries checkerboard charge one, so removing `k` rows
removes charge `k`. -/
theorem charge_cleared_cells {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    BagGrowth.charge
        (b.filter (fun p => Board.isFull GameConfig.standard b p.2))
      = ((Board.fullRows GameConfig.standard b).card : ZMod 2) := by
  classical
  have hmaps : ∀ p ∈ b.filter
      (fun p => Board.isFull GameConfig.standard b p.2),
      p.2 ∈ Board.fullRows GameConfig.standard b := by
    intro p hp
    obtain ⟨hpb, hpf⟩ := Finset.mem_filter.mp hp
    unfold Board.fullRows
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_image.mpr ⟨p, hpb, rfl⟩, hpf⟩
  have hfib := Finset.sum_fiberwise_of_maps_to hmaps
    (fun p : Coord => ((p.1 + p.2 : ℕ) : ZMod 2))
  have hrow : ∀ t ∈ Board.fullRows GameConfig.standard b,
      (∑ p ∈ (b.filter
          (fun p => Board.isFull GameConfig.standard b p.2)).filter
          (fun p => p.2 = t), ((p.1 + p.2 : ℕ) : ZMod 2)) = 1 := by
    intro t ht
    rw [cleared_fiber_eq hwf ht]
    exact BagGrowth.charge_row t
  unfold BagGrowth.charge
  rw [← hfib, Finset.sum_congr rfl hrow, Finset.sum_const, nsmul_eq_mul,
    mul_one]

/-- **THE CHARGE LAW ACROSS A CLEAR**: clearing changes the checkerboard
charge by the number of rows removed plus the gravity work. The
invariant is not destroyed by a clear — it is corrected by one. -/
theorem charge_clearLines {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    BagGrowth.charge (Board.clearLines GameConfig.standard b)
      = BagGrowth.charge b
        + ((Board.fullRows GameConfig.standard b).card : ZMod 2)
        + gravityWork b := by
  classical
  have hinj : ∀ x ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ∀ y ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ((x.1, x.2 - Board.clearedBelow GameConfig.standard b x.2) : Coord)
        = ((y.1, y.2 - Board.clearedBelow GameConfig.standard b y.2)
            : Coord) → x = y := by
    intro x hx y hy hxy
    rw [Finset.mem_filter] at hx hy
    have hpair := Prod.ext_iff.mp hxy
    have h1 : x.1 = y.1 := hpair.1
    have h2 : x.2 - Board.clearedBelow GameConfig.standard b x.2
        = y.2 - Board.clearedBelow GameConfig.standard b y.2 := hpair.2
    have h3 : x.2 = y.2 := by
      rcases lt_trichotomy x.2 y.2 with hlt | heq | hgt
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hlt hx.2
        omega
      · exact heq
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hgt hy.2
        omega
    exact Prod.ext_iff.mpr ⟨h1, h3⟩
  have hcl : Board.clearLines GameConfig.standard b
      = (b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2)).image
        (fun p => ((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
          : Coord)) := rfl
  have hstep :
      (∑ p ∈ b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2),
        ((((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
            : Coord).1
          + ((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
              : Coord).2 : ℕ) : ZMod 2))
      = (∑ p ∈ b.filter
            (fun p => ¬ Board.isFull GameConfig.standard b p.2),
          ((p.1 + p.2 : ℕ) : ZMod 2))
        + ∑ p ∈ b.filter
            (fun p => ¬ Board.isFull GameConfig.standard b p.2),
          ((Board.clearedBelow GameConfig.standard b p.2 : ℕ)
            : ZMod 2) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun p _ => ?_)
    have hle := clearedBelow_le GameConfig.standard b p.2
    show ((p.1 + (p.2 - Board.clearedBelow GameConfig.standard b p.2) : ℕ)
        : ZMod 2) = _
    rw [Nat.cast_add, cast_sub_two hle, Nat.cast_add]
    ring
  have hsplit := Finset.sum_filter_add_sum_filter_not b
    (fun p => Board.isFull GameConfig.standard b p.2)
    (fun p => ((p.1 + p.2 : ℕ) : ZMod 2))
  have hcl2 := charge_cleared_cells hwf
  unfold BagGrowth.charge at hcl2 ⊢
  unfold gravityWork
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  rw [hcl, Finset.sum_image hinj, hstep, ← hsplit, hcl2]
  have key := hchar ((Board.fullRows GameConfig.standard b).card : ZMod 2)
  first
    | linear_combination -key
    | linear_combination key

/-- **The charge law for a whole move**: a drop adds the piece's charge
(one for T, zero otherwise) and the ensuing clear adds its row count plus
its gravity work. Every term is explicit. -/
theorem charge_applyStep {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    BagGrowth.charge (Placement.applyStep GameConfig.standard b pl)
      = BagGrowth.charge b
        + (if pl.piece = Piece.T then 1 else 0)
        + ((Board.fullRows GameConfig.standard (pl.place b)).card : ZMod 2)
        + gravityWork (pl.place b) := by
  rw [Placement.applyStep_eq_clearLines_place,
    charge_clearLines (Placement.place_wf hwf hv),
    BagGrowth.charge_place, BagGrowth.charge_shapeUp_pl]

/-! ### What the charge law says about a cycle

Folding the charge law along a word gives, for a loop, a single
equation in `ZMod 2` tying three things together: how many T's were
played, how many rows were cleared, and how much gravity work the
clears did. On a legal 35-cycle the first two are pinned (five T's,
fourteen rows), so the third is FORCED — and forced to be odd, hence
nonzero. -/

/-- The T-charge of a word: one per T played, mod two. -/
def wordTCharge : List Placement → ZMod 2
  | [] => 0
  | pl :: rest => (if pl.piece = Piece.T then 1 else 0) + wordTCharge rest

@[simp] theorem wordTCharge_nil : wordTCharge [] = 0 := rfl

theorem wordTCharge_cons (pl : Placement) (rest : List Placement) :
    wordTCharge (pl :: rest)
      = (if pl.piece = Piece.T then 1 else 0) + wordTCharge rest := rfl

/-- The total gravity work of a word's clears. -/
def wordGravity (b : Board) : List Placement → ZMod 2
  | [] => 0
  | pl :: rest =>
      gravityWork (pl.place b)
      + wordGravity (Placement.applyStep GameConfig.standard b pl) rest

@[simp] theorem wordGravity_nil (b : Board) : wordGravity b [] = 0 := rfl

theorem wordGravity_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordGravity b (pl :: rest)
      = gravityWork (pl.place b)
        + wordGravity (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- The T-charge counts the T's. -/
theorem wordTCharge_eq_count (pls : List Placement) :
    wordTCharge pls = (((pls.map (·.piece)).count Piece.T : ℕ) : ZMod 2) := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    rw [wordTCharge_cons, ih, List.map_cons, List.count_cons]
    by_cases hp : pl.piece = Piece.T
    · rw [if_pos hp, if_pos (show
        (((fun x : Placement => x.piece) pl) == Piece.T) = true from by
          simp [hp])]
      push_cast
      ring
    · rw [if_neg hp, if_neg (show
        ¬ ((((fun x : Placement => x.piece) pl) == Piece.T) = true) from by
          simp [hp])]
      push_cast
      ring

/-- **The charge ledger along a word**: the final charge is the initial
charge plus one per T, plus the rows cleared, plus the gravity work. -/
theorem charge_word {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    BagGrowth.charge
        (pls.foldl (Placement.applyStep GameConfig.standard) b)
      = BagGrowth.charge b + wordTCharge pls
        + ((wordClears b pls : ℕ) : ZMod 2) + wordGravity b pls := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    have hstep := charge_applyStep hwf hvpl
    have hrec := ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, hrec, hstep, wordClears_cons, wordTCharge_cons,
      wordGravity_cons]
    push_cast
    ring

/-- **THE CYCLE CHARGE EQUATION**: around a loop the charge returns, so
the gravity work is exactly the T-count plus the cleared-row count,
modulo two. Piece census on one side, clear geometry on the other. -/
theorem cycle_charge_law {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordGravity b pls
      = wordTCharge pls + ((wordClears b pls : ℕ) : ZMod 2) := by
  have h := charge_word hwf hv
  rw [hfold] at h
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  have key := hchar (BagGrowth.charge b)
  have key2 := hchar (wordTCharge pls)
  have key3 := hchar (((wordClears b pls : ℕ) : ZMod 2))
  first
    | linear_combination -h - key2 - key3
    | linear_combination h - key2 - key3
    | linear_combination -h - key - key2 - key3

/-- **A LEGAL 35-CYCLE DOES ODD GRAVITY WORK.** Five T's are played
(odd) against fourteen rows cleared (even), so the gravity work must be
one. In particular it is NOT zero: the loop cannot do all its clearing
at the very top of the stack. -/
theorem legal_cycle_gravity_odd {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordGravity b w = 1 := by
  have hcensus := legal_cycle_word_piece_census hwf hne hv hbag hfold
    Piece.T
  rw [census_eq_count, hlen] at hcensus
  have hclears := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold
    hlen
  rw [cycle_charge_law hwf hv hfold, wordTCharge_eq_count, hcensus,
    hclears]
  decide

/-- Odd gravity work means some clear happened with material above it:
there is a surviving cell sitting over a cleared row. -/
theorem exists_cell_above_clear {B : Board} (h : gravityWork B ≠ 0) :
    ∃ p ∈ B, ¬ Board.isFull GameConfig.standard B p.2
      ∧ 0 < Board.clearedBelow GameConfig.standard B p.2 := by
  classical
  by_contra hcon
  push Not at hcon
  apply h
  unfold gravityWork
  apply Finset.sum_eq_zero
  intro p hp
  obtain ⟨hpb, hpnf⟩ := Finset.mem_filter.mp hp
  have := hcon p hpb hpnf
  rw [show Board.clearedBelow GameConfig.standard B p.2 = 0 from by omega]
  simp

/-- **SOME CLEAR OF A LEGAL CYCLE CARRIES A LOAD**: not every clear can
be a clean sweep off the top — at least one of them happens with
surviving cells stacked above it. A structural fact about the geometry
of the loop, derived from the piece census alone. -/
theorem exists_loaded_clear {b : Board} {pls : List Placement}
    (h : wordGravity b pls ≠ 0) :
    ∃ (c : Board) (pl : Placement),
      ∃ p ∈ pl.place c, ¬ Board.isFull GameConfig.standard (pl.place c) p.2
        ∧ 0 < Board.clearedBelow GameConfig.standard (pl.place c) p.2 := by
  induction pls generalizing b with
  | nil => exact absurd rfl h
  | cons pl rest ih =>
    rw [wordGravity_cons] at h
    by_cases hhead : gravityWork (pl.place b) = 0
    · rw [hhead, zero_add] at h
      exact ih h
    · exact ⟨b, pl, exists_cell_above_clear hhead⟩

/-! ### Which clears can do gravity work

Gravity work is done only by a clear that leaves something behind AND
takes something away. A move that clears nothing does none (nothing
falls); a move that clears EVERYTHING does none either (nothing is left
to fall). Since a legal 35-cycle must do odd — hence nonzero — gravity
work, it can be neither all-dry nor all-perfect: somewhere it must make
a partial clear. -/

/-- A move that clears nothing does no gravity work. -/
theorem gravityWork_of_no_clears {B : Board}
    (h : Board.fullRows GameConfig.standard B = ∅) :
    gravityWork B = 0 := by
  classical
  unfold gravityWork
  apply Finset.sum_eq_zero
  intro p _
  have hcb : Board.clearedBelow GameConfig.standard B p.2 = 0 := by
    unfold Board.clearedBelow
    rw [h]
    simp
  rw [hcb]
  simp

/-- A move that clears EVERYTHING does no gravity work either: nothing
survives to fall. -/
theorem gravityWork_of_perfect {B : Board}
    (h : ∀ p ∈ B, Board.isFull GameConfig.standard B p.2) :
    gravityWork B = 0 := by
  classical
  unfold gravityWork
  apply Finset.sum_eq_zero
  intro p hp
  obtain ⟨hpB, hpnf⟩ := Finset.mem_filter.mp hp
  exact absurd (h p hpB) hpnf

/-- **Gravity work means a partial clear**: the move removed at least one
row and left at least one cell standing. -/
theorem partial_of_gravityWork {B : Board} (h : gravityWork B ≠ 0) :
    Board.fullRows GameConfig.standard B ≠ ∅
      ∧ ∃ p ∈ B, ¬ Board.isFull GameConfig.standard B p.2 := by
  classical
  constructor
  · intro hnil
    exact h (gravityWork_of_no_clears hnil)
  · by_contra hcon
    push Not at hcon
    exact h (gravityWork_of_perfect hcon)

/-- **A LEGAL CYCLE MAKES A PARTIAL CLEAR**: some move of it removes at
least one row while leaving at least one cell standing. The loop can be
neither all-dry nor a chain of perfect clears — a concrete exclusion
extracted from the piece census through the charge law. -/
theorem exists_partial_clear {b : Board} {pls : List Placement}
    (h : wordGravity b pls ≠ 0) :
    ∃ (c : Board) (pl : Placement),
      Board.fullRows GameConfig.standard (pl.place c) ≠ ∅
        ∧ ∃ p ∈ pl.place c,
            ¬ Board.isFull GameConfig.standard (pl.place c) p.2 := by
  induction pls generalizing b with
  | nil => exact absurd rfl h
  | cons pl rest ih =>
    rw [wordGravity_cons] at h
    by_cases hhead : gravityWork (pl.place b) = 0
    · rw [hhead, zero_add] at h
      exact ih h
    · exact ⟨b, pl, partial_of_gravityWork hhead⟩

/-- The legal 35-cycle instance: five T's against fourteen rows force
odd gravity work, so a partial clear exists. -/
theorem legal_cycle_has_partial_clear {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    ∃ (c : Board) (pl : Placement),
      Board.fullRows GameConfig.standard (pl.place c) ≠ ∅
        ∧ ∃ p ∈ pl.place c,
            ¬ Board.isFull GameConfig.standard (pl.place c) p.2 := by
  apply exists_partial_clear (b := b) (pls := w)
  rw [legal_cycle_gravity_odd hwf hne hv hbag hfold hlen]
  decide

/-- **NO LEGAL CYCLE IS A PERFECT-CLEAR LOOP**: a cycle whose every
clearing move empties the board outright cannot be bag-legal at length
35. The T-parity that forbids five bags from tiling a rectangle
survives the passage to loops WITH clears. -/
theorem no_legal_perfect_clear_cycle {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35)
    (hperfect : ∀ (c : Board) (pl : Placement),
      Board.fullRows GameConfig.standard (pl.place c) ≠ ∅ →
      ∀ p ∈ pl.place c,
        Board.isFull GameConfig.standard (pl.place c) p.2) : False := by
  obtain ⟨c, pl, hne', p, hp, hpnf⟩ :=
    legal_cycle_has_partial_clear hwf hne hv hbag hfold hlen
  exact hpnf (hperfect c pl hne' p hp)

/-! ### Even contiguous clears are invisible to the charge

If the cleared rows form one unbroken block, every surviving cell is
either entirely below it (nothing falls) or entirely above it (falls by
the block's height). So the gravity work is the block height times the
mass above — and when the block height is EVEN it vanishes.

Doubles and tetrises are exactly the even blocks. A legal 35-cycle owes
odd gravity work, so it cannot be built out of them alone. -/

/-- **A contiguous clear of even height does no gravity work**: every
survivor falls by zero or by the whole block, and the block is even. -/
theorem gravityWork_of_contiguous_even {B : Board} {t k : ℕ}
    (hk : Even k)
    (hf : Board.fullRows GameConfig.standard B = Finset.Ico t (t + k)) :
    gravityWork B = 0 := by
  classical
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  obtain ⟨m, hm⟩ := hk
  unfold gravityWork
  apply Finset.sum_eq_zero
  intro p hp
  obtain ⟨hpB, hpnf⟩ := Finset.mem_filter.mp hp
  have hout : p.2 ∉ Board.fullRows GameConfig.standard B := by
    intro hmem
    exact hpnf (Board.isFull_of_mem_fullRows hmem)
  rw [hf, Finset.mem_Ico] at hout
  push Not at hout
  have hcb : Board.clearedBelow GameConfig.standard B p.2 = 0
      ∨ Board.clearedBelow GameConfig.standard B p.2 = k := by
    unfold Board.clearedBelow
    rw [hf]
    by_cases hlt : p.2 < t
    · left
      rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
      intro x hx
      obtain ⟨hx1, hx2⟩ := Finset.mem_filter.mp hx
      rw [Finset.mem_Ico] at hx1
      omega
    · right
      have hge : t + k ≤ p.2 := hout (by omega)
      rw [Finset.filter_true_of_mem (fun x hx => by
        rw [Finset.mem_Ico] at hx
        omega)]
      rw [Nat.card_Ico]
      omega
  rcases hcb with h0 | hkk
  · rw [h0]
    simp
  · rw [hkk, hm]
    push_cast
    exact hchar (m : ZMod 2)

/-- A tetris does no gravity work: four rows is an even block. -/
theorem gravityWork_tetris {B : Board} {h : ℕ}
    (hf : Board.fullRows GameConfig.standard B = Finset.Ico h (h + 4)) :
    gravityWork B = 0 :=
  gravityWork_of_contiguous_even ⟨2, by omega⟩ hf

/-- A double on adjacent rows does no gravity work either. -/
theorem gravityWork_double {B : Board} {h : ℕ}
    (hf : Board.fullRows GameConfig.standard B = Finset.Ico h (h + 2)) :
    gravityWork B = 0 :=
  gravityWork_of_contiguous_even ⟨1, by omega⟩ hf

/-- **A LEGAL CYCLE CANNOT LIVE ON DOUBLES AND TETRISES**: some move of
it clears a block that is not an even contiguous run — a single, a
triple, or a split clear straddling an unfilled row. -/
theorem exists_odd_or_split_clear {b : Board} {pls : List Placement}
    (h : wordGravity b pls ≠ 0) :
    ∃ (c : Board) (pl : Placement), ∀ t k : ℕ, Even k →
      Board.fullRows GameConfig.standard (pl.place c)
        ≠ Finset.Ico t (t + k) := by
  induction pls generalizing b with
  | nil => exact absurd rfl h
  | cons pl rest ih =>
    rw [wordGravity_cons] at h
    by_cases hhead : gravityWork (pl.place b) = 0
    · rw [hhead, zero_add] at h
      exact ih h
    · refine ⟨b, pl, fun t k hk hf => hhead ?_⟩
      exact gravityWork_of_contiguous_even hk hf

/-- The legal 35-cycle instance. -/
theorem legal_cycle_has_odd_or_split_clear {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    ∃ (c : Board) (pl : Placement), ∀ t k : ℕ, Even k →
      Board.fullRows GameConfig.standard (pl.place c)
        ≠ Finset.Ico t (t + k) := by
  apply exists_odd_or_split_clear (b := b) (pls := w)
  rw [legal_cycle_gravity_odd hwf hne hv hbag hfold hlen]
  decide

/-! ### The row charge: the other half of the checkerboard

Splitting the checkerboard colour `(c + r)` into its two coordinates
gives two independent invariants. The column half was done with the
weighted ledger; the row half is new, and it behaves better than the
full checkerboard in one respect: a cleared row contributes `10·t ≡ 0`,
so the row count drops out entirely and only the gravity work remains.

Comparing the two laws expresses the gravity work a second way — in
terms of the shapes' ROW moments — and so turns the parity debt into a
constraint on which ROTATIONS the cycle plays. -/

/-- The row charge of a board: its cells' row indices, mod two. -/
def rowCharge (b : Board) : ZMod 2 := ∑ p ∈ b, ((p.2 : ℕ) : ZMod 2)

/-- A shape's own row moment, mod two. -/
def shapeRowCharge (pl : Placement) : ZMod 2 :=
  ∑ cell ∈ pl.shapeUp, ((cell.2 : ℕ) : ZMod 2)

/-- **A drop adds its shape's row charge, wherever it lands**: the four
cells all shift by the drop offset, and four is even. -/
theorem rowCharge_place (b : Board) (pl : Placement) :
    rowCharge (pl.place b) = rowCharge b + shapeRowCharge pl := by
  classical
  have hdrop : ∑ p ∈ pl.dropped b, ((p.2 : ℕ) : ZMod 2)
      = shapeRowCharge pl := by
    rw [Placement.dropped_eq_image]
    rw [Finset.sum_image (fun x _ y _ hxy => by
      have h1 : pl.col + x.1 = pl.col + y.1 := congrArg Prod.fst hxy
      have h2 : pl.dropOffset b + x.2 = pl.dropOffset b + y.2 :=
        congrArg Prod.snd hxy
      exact Prod.ext_iff.mpr ⟨by omega, by omega⟩)]
    unfold shapeRowCharge
    have hterm : ∀ cell ∈ pl.shapeUp,
        (((pl.dropOffset b + cell.2 : ℕ)) : ZMod 2)
        = ((pl.dropOffset b : ℕ) : ZMod 2) + ((cell.2 : ℕ) : ZMod 2) := by
      intro cell _
      push_cast
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib,
      Finset.sum_const, pl.shapeUp_card]
    have h4 : (4 : ℕ) • ((pl.dropOffset b : ℕ) : ZMod 2) = 0 := by
      rw [nsmul_eq_mul]
      have : ((4 : ℕ) : ZMod 2) = 0 := by decide
      rw [this, zero_mul]
    rw [h4, zero_add]
  unfold rowCharge
  rw [Placement.place_eq_union_dropped,
    Finset.sum_union (pl.dropped_disjoint b).symm, hdrop]

/-- A cleared row contributes nothing to the row charge: ten cells all
in the same row, and ten is even. -/
theorem rowCharge_cleared_cells {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    ∑ p ∈ b.filter (fun p => Board.isFull GameConfig.standard b p.2),
      ((p.2 : ℕ) : ZMod 2) = 0 := by
  classical
  have hmaps : ∀ p ∈ b.filter
      (fun p => Board.isFull GameConfig.standard b p.2),
      p.2 ∈ Board.fullRows GameConfig.standard b := by
    intro p hp
    obtain ⟨hpb, hpf⟩ := Finset.mem_filter.mp hp
    unfold Board.fullRows
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_image.mpr ⟨p, hpb, rfl⟩, hpf⟩
  have hfib := Finset.sum_fiberwise_of_maps_to hmaps
    (fun p : Coord => ((p.2 : ℕ) : ZMod 2))
  have hrow : ∀ t ∈ Board.fullRows GameConfig.standard b,
      (∑ p ∈ (b.filter
          (fun p => Board.isFull GameConfig.standard b p.2)).filter
          (fun p => p.2 = t), ((p.2 : ℕ) : ZMod 2)) = 0 := by
    intro t ht
    rw [cleared_fiber_eq hwf ht]
    rw [Finset.sum_image (fun x _ y _ hxy => by
      simpa using congrArg Prod.fst hxy)]
    simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    rw [show ((10 : ℕ) : ZMod 2) = 0 from by decide, zero_mul]
  rw [← hfib, Finset.sum_congr rfl hrow, Finset.sum_const, smul_zero]

/-- **THE ROW-CHARGE LAW ACROSS A CLEAR**: the removed rows cancel
(ten cells apiece), so a clear changes the row charge by the gravity
work alone. -/
theorem rowCharge_clearLines {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    rowCharge (Board.clearLines GameConfig.standard b)
      = rowCharge b + gravityWork b := by
  classical
  have hinj : ∀ x ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ∀ y ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ((x.1, x.2 - Board.clearedBelow GameConfig.standard b x.2) : Coord)
        = ((y.1, y.2 - Board.clearedBelow GameConfig.standard b y.2)
            : Coord) → x = y := by
    intro x hx y hy hxy
    rw [Finset.mem_filter] at hx hy
    have hpair := Prod.ext_iff.mp hxy
    have h1 : x.1 = y.1 := hpair.1
    have h2 : x.2 - Board.clearedBelow GameConfig.standard b x.2
        = y.2 - Board.clearedBelow GameConfig.standard b y.2 := hpair.2
    have h3 : x.2 = y.2 := by
      rcases lt_trichotomy x.2 y.2 with hlt | heq | hgt
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hlt hx.2
        omega
      · exact heq
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hgt hy.2
        omega
    exact Prod.ext_iff.mpr ⟨h1, h3⟩
  have hcl : Board.clearLines GameConfig.standard b
      = (b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2)).image
        (fun p => ((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
          : Coord)) := rfl
  have hstep :
      (∑ p ∈ b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2),
        (((((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
            : Coord).2) : ℕ) : ZMod 2))
      = (∑ p ∈ b.filter
            (fun p => ¬ Board.isFull GameConfig.standard b p.2),
          ((p.2 : ℕ) : ZMod 2))
        + ∑ p ∈ b.filter
            (fun p => ¬ Board.isFull GameConfig.standard b p.2),
          ((Board.clearedBelow GameConfig.standard b p.2 : ℕ)
            : ZMod 2) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun p _ => ?_)
    have hle := clearedBelow_le GameConfig.standard b p.2
    show (((p.2 - Board.clearedBelow GameConfig.standard b p.2 : ℕ))
        : ZMod 2) = _
    rw [cast_sub_two hle]
  have hsplit := Finset.sum_filter_add_sum_filter_not b
    (fun p => Board.isFull GameConfig.standard b p.2)
    (fun p => ((p.2 : ℕ) : ZMod 2))
  have hcl2 := rowCharge_cleared_cells hwf
  unfold rowCharge gravityWork
  rw [hcl, Finset.sum_image hinj, hstep, ← hsplit, hcl2, zero_add]

/-- The row-charge law for a whole move. -/
theorem rowCharge_applyStep {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    rowCharge (Placement.applyStep GameConfig.standard b pl)
      = rowCharge b + shapeRowCharge pl + gravityWork (pl.place b) := by
  rw [Placement.applyStep_eq_clearLines_place,
    rowCharge_clearLines (Placement.place_wf hwf hv), rowCharge_place]

/-- The word's total shape row charge. -/
def wordRowCharge : List Placement → ZMod 2
  | [] => 0
  | pl :: rest => shapeRowCharge pl + wordRowCharge rest

@[simp] theorem wordRowCharge_nil : wordRowCharge [] = 0 := rfl

theorem wordRowCharge_cons (pl : Placement) (rest : List Placement) :
    wordRowCharge (pl :: rest)
      = shapeRowCharge pl + wordRowCharge rest := rfl

/-- The row-charge ledger along a word. -/
theorem rowCharge_word {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    rowCharge (pls.foldl (Placement.applyStep GameConfig.standard) b)
      = rowCharge b + wordRowCharge pls + wordGravity b pls := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    have hstep := rowCharge_applyStep hwf hvpl
    have hrec := ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, hrec, hstep, wordRowCharge_cons, wordGravity_cons]
    ring

/-- **THE GRAVITY WORK, READ A SECOND WAY**: around a cycle it equals
the total row moment of the shapes played. -/
theorem cycle_rowCharge_law {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordGravity b pls = wordRowCharge pls := by
  have h := rowCharge_word hwf hv
  rw [hfold] at h
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  have key := hchar (rowCharge b)
  have key2 := hchar (wordRowCharge pls)
  first
    | linear_combination -h - key2
    | linear_combination h - key2
    | linear_combination -h - key - key2

/-- **THE ROTATION CENSUS IS CONSTRAINED**: reading the gravity work
both ways forces the shapes' total row moment to match the T-count plus
the cleared-row count. Which ROTATIONS the cycle plays is no longer
free — it is pinned, mod two, by the piece census. -/
theorem cycle_rotation_census {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordRowCharge pls
      = wordTCharge pls + ((wordClears b pls : ℕ) : ZMod 2) := by
  rw [← cycle_rowCharge_law hwf hv hfold]
  exact cycle_charge_law hwf hv hfold

/-- On a legal 35-cycle the row moment is ODD: an odd number of its
placements present a shape whose cells' rows sum to an odd number. -/
theorem legal_cycle_rowCharge_odd {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordRowCharge w = 1 := by
  rw [← cycle_rowCharge_law hwf hv hfold]
  exact legal_cycle_gravity_odd hwf hne hv hbag hfold hlen

/-! ### Which shapes carry an odd row moment, and what that forces

The row-moment table is remarkably clean: I, O, S and Z are even in
every rotation; L and J are odd in every rotation; and T is odd exactly
in its two FLAT rotations. So the rotation census reduces to a single
question — how many of the cycle's T's are laid flat. -/

/-- **The row-moment table**: only L, J, and the flat T carry an odd row
moment. Verified by the kernel across all seven pieces and four
rotations. -/
theorem shapeRowCharge_eq_one_iff :
    ∀ (p : Piece) (r : Rotation),
      (∑ cell ∈ p.shapeUp r, ((cell.2 : ℕ) : ZMod 2)) = 1
        ↔ (p = Piece.L ∨ p = Piece.J
            ∨ (p = Piece.T ∧ (r = 0 ∨ r = 2))) := by
  decide

/-- Away from the flat T, the row moment is just the L/J indicator. -/
theorem shapeRowCharge_of_not_flat_T {pl : Placement}
    (h : ¬ (pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2))) :
    shapeRowCharge pl
      = if pl.piece = Piece.L ∨ pl.piece = Piece.J then 1 else 0 := by
  have hiff := shapeRowCharge_eq_one_iff pl.piece pl.rot
  have hval : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
  by_cases hLJ : pl.piece = Piece.L ∨ pl.piece = Piece.J
  · rw [if_pos hLJ]
    exact hiff.mpr (by tauto)
  · rw [if_neg hLJ]
    rcases hval (shapeRowCharge pl) with h0 | h1
    · exact h0
    · exact absurd (hiff.mp h1) (by tauto)

/-- The word's tally of one particular piece, mod two. -/
def wordPieceCharge (q : Piece) : List Placement → ZMod 2
  | [] => 0
  | pl :: rest => (if pl.piece = q then 1 else 0) + wordPieceCharge q rest

@[simp] theorem wordPieceCharge_nil (q : Piece) :
    wordPieceCharge q [] = 0 := rfl

theorem wordPieceCharge_cons (q : Piece) (pl : Placement)
    (rest : List Placement) :
    wordPieceCharge q (pl :: rest)
      = (if pl.piece = q then 1 else 0)
        + wordPieceCharge q rest := rfl

theorem wordPieceCharge_eq_count (q : Piece) (pls : List Placement) :
    wordPieceCharge q pls
      = (((pls.map (·.piece)).count q : ℕ) : ZMod 2) := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    rw [wordPieceCharge_cons, ih, List.map_cons, List.count_cons]
    by_cases hp : pl.piece = q
    · rw [if_pos hp, if_pos (show
        (((fun x : Placement => x.piece) pl) == q) = true from by
          simp [hp])]
      push_cast
      ring
    · rw [if_neg hp, if_neg (show
        ¬ ((((fun x : Placement => x.piece) pl) == q) = true) from by
          simp [hp])]
      push_cast
      ring

/-- A word with no flat T carries exactly the L and J tallies. -/
theorem wordRowCharge_of_no_flat_T {pls : List Placement}
    (h : ∀ pl ∈ pls,
      ¬ (pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2))) :
    wordRowCharge pls
      = wordPieceCharge Piece.L pls + wordPieceCharge Piece.J pls := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    rw [wordRowCharge_cons, ih (fun q hq => h q (by simp [hq])),
      wordPieceCharge_cons, wordPieceCharge_cons,
      shapeRowCharge_of_not_flat_T (h pl (by simp))]
    by_cases hL : pl.piece = Piece.L
    · rw [if_pos (Or.inl hL), if_pos hL,
        if_neg (show ¬ (pl.piece = Piece.J) from by rw [hL]; decide)]
      ring
    · by_cases hJ : pl.piece = Piece.J
      · rw [if_pos (Or.inr hJ), if_neg hL, if_pos hJ]
        ring
      · rw [if_neg (by tauto), if_neg hL, if_neg hJ]
        ring

/-- **EVERY LEGAL 35-CYCLE LAYS A T FLAT.** L and J contribute five
apiece — an even total — so the whole odd row moment must come from the
T's, and only a T in one of its two flat rotations carries any. An odd
number of the cycle's five T's are therefore laid flat; in particular at
least one is. A concrete, checkable demand on the M2 witness, obtained
from parity alone. -/
theorem legal_cycle_has_flat_T {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    ∃ pl ∈ w, pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2) := by
  by_contra hcon
  have hcon' : ∀ pl ∈ w,
      ¬ (pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2)) := by
    intro pl hpl hbad
    exact hcon ⟨pl, hpl, hbad⟩
  have h1 := wordRowCharge_of_no_flat_T hcon'
  have hL := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.L
  have hJ := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.J
  rw [census_eq_count, hlen] at hL hJ
  norm_num at hL hJ
  rw [wordPieceCharge_eq_count, wordPieceCharge_eq_count, hL, hJ] at h1
  have hodd := legal_cycle_rowCharge_odd hwf hne hv hbag hfold hlen
  rw [h1] at hodd
  revert hodd
  decide

/-! ### The potential balance: how high things happen

Every law so far is blind to HEIGHT. Summing the row indices of the
board's cells gives a potential: drops raise it by four times the drop
offset plus the shape's own moment, and clears lower it by ten times
the height of each row removed plus the gravity work. Around a cycle
the two must balance EXACTLY — not merely modulo something.

This is the first cycle law that mentions how high the pieces land and
how high the rows are that they complete. -/

/-- The board's potential: the sum of its cells' row indices. -/
def rowMoment (b : Board) : ℕ := ∑ p ∈ b, p.2

/-- A shape's own row moment. -/
def shapeRowMoment (pl : Placement) : ℕ := ∑ cell ∈ pl.shapeUp, cell.2

/-- The integer gravity work: how far, in total, the survivors fall. -/
def gravityInt (b : Board) : ℕ :=
  ∑ p ∈ b.filter (fun p => ¬ Board.isFull GameConfig.standard b p.2),
    Board.clearedBelow GameConfig.standard b p.2

/-- The total height of the rows a clear removes. -/
def clearedRowSum (b : Board) : ℕ :=
  ∑ t ∈ Board.fullRows GameConfig.standard b, t

/-- **A drop raises the potential by four times its landing height plus
the shape's own moment.** -/
theorem rowMoment_place (b : Board) (pl : Placement) :
    rowMoment (pl.place b)
      = rowMoment b + 4 * pl.dropOffset b + shapeRowMoment pl := by
  classical
  have hdrop : ∑ p ∈ pl.dropped b, p.2
      = 4 * pl.dropOffset b + shapeRowMoment pl := by
    rw [Placement.dropped_eq_image]
    rw [Finset.sum_image (fun x _ y _ hxy => by
      have h1 : pl.col + x.1 = pl.col + y.1 := congrArg Prod.fst hxy
      have h2 : pl.dropOffset b + x.2 = pl.dropOffset b + y.2 :=
        congrArg Prod.snd hxy
      exact Prod.ext_iff.mpr ⟨by omega, by omega⟩)]
    unfold shapeRowMoment
    rw [Finset.sum_congr rfl (fun cell _ => show
      ((pl.col + cell.1, pl.dropOffset b + cell.2) : Coord).2
        = pl.dropOffset b + cell.2 from rfl)]
    rw [Finset.sum_add_distrib, Finset.sum_const, pl.shapeUp_card,
      smul_eq_mul]
  unfold rowMoment
  rw [Placement.place_eq_union_dropped,
    Finset.sum_union (pl.dropped_disjoint b).symm, hdrop]
  omega

/-- The cells a clear removes carry total height `10` per cleared row. -/
theorem clearedCells_rowMoment {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    ∑ p ∈ b.filter (fun p => Board.isFull GameConfig.standard b p.2), p.2
      = 10 * clearedRowSum b := by
  classical
  have hmaps : ∀ p ∈ b.filter
      (fun p => Board.isFull GameConfig.standard b p.2),
      p.2 ∈ Board.fullRows GameConfig.standard b := by
    intro p hp
    obtain ⟨hpb, hpf⟩ := Finset.mem_filter.mp hp
    unfold Board.fullRows
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_image.mpr ⟨p, hpb, rfl⟩, hpf⟩
  have hfib := Finset.sum_fiberwise_of_maps_to hmaps (fun p : Coord => p.2)
  have hrow : ∀ t ∈ Board.fullRows GameConfig.standard b,
      (∑ p ∈ (b.filter
          (fun p => Board.isFull GameConfig.standard b p.2)).filter
          (fun p => p.2 = t), p.2) = 10 * t := by
    intro t ht
    rw [cleared_fiber_eq hwf ht]
    rw [Finset.sum_image (fun x _ y _ hxy => by
      simpa using congrArg Prod.fst hxy)]
    simp only [Finset.sum_const, Finset.card_range, smul_eq_mul]
  rw [← hfib, Finset.sum_congr rfl hrow, ← Finset.mul_sum]
  rfl

/-- **THE POTENTIAL LAW ACROSS A CLEAR**: the potential lost is ten per
unit height of each removed row, plus the distance the survivors fall. -/
theorem rowMoment_clearLines {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    rowMoment (Board.clearLines GameConfig.standard b)
      + 10 * clearedRowSum b + gravityInt b = rowMoment b := by
  classical
  have hinj : ∀ x ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ∀ y ∈ b.filter
      (fun p => ¬ Board.isFull GameConfig.standard b p.2),
      ((x.1, x.2 - Board.clearedBelow GameConfig.standard b x.2) : Coord)
        = ((y.1, y.2 - Board.clearedBelow GameConfig.standard b y.2)
            : Coord) → x = y := by
    intro x hx y hy hxy
    rw [Finset.mem_filter] at hx hy
    have hpair := Prod.ext_iff.mp hxy
    have h3 : x.2 = y.2 := by
      rcases lt_trichotomy x.2 y.2 with hlt | heq | hgt
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hlt hx.2
        have := hpair.2
        omega
      · exact heq
      · have := clearedBelow_shift_strictMono
          (cfg := GameConfig.standard) (b := b) hgt hy.2
        have := hpair.2
        omega
    exact Prod.ext_iff.mpr ⟨hpair.1, h3⟩
  have hcl : Board.clearLines GameConfig.standard b
      = (b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2)).image
        (fun p => ((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
          : Coord)) := rfl
  have hsurv :
      (∑ p ∈ b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2),
        ((p.1, p.2 - Board.clearedBelow GameConfig.standard b p.2)
          : Coord).2) + gravityInt b
      = ∑ p ∈ b.filter
          (fun p => ¬ Board.isFull GameConfig.standard b p.2), p.2 := by
    unfold gravityInt
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun p _ => ?_)
    have hle := clearedBelow_le GameConfig.standard b p.2
    show (p.2 - Board.clearedBelow GameConfig.standard b p.2)
      + Board.clearedBelow GameConfig.standard b p.2 = p.2
    omega
  have hsplit := Finset.sum_filter_add_sum_filter_not b
    (fun p => Board.isFull GameConfig.standard b p.2)
    (fun p : Coord => p.2)
  have hclr := clearedCells_rowMoment hwf
  unfold rowMoment
  rw [hcl, Finset.sum_image hinj]
  omega

/-- The potential law for a whole move. -/
theorem rowMoment_applyStep {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    rowMoment (Placement.applyStep GameConfig.standard b pl)
      + 10 * clearedRowSum (pl.place b) + gravityInt (pl.place b)
      = rowMoment b + 4 * pl.dropOffset b + shapeRowMoment pl := by
  rw [Placement.applyStep_eq_clearLines_place]
  have h1 := rowMoment_clearLines (Placement.place_wf hwf hv)
  have h2 := rowMoment_place b pl
  omega

/-- The potential a word's drops deliver. -/
def wordLift (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (4 * pl.dropOffset b + shapeRowMoment pl)
      + wordLift (Placement.applyStep GameConfig.standard b pl) rest

/-- The potential a word's clears release. -/
def wordRelease (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (10 * clearedRowSum (pl.place b) + gravityInt (pl.place b))
      + wordRelease (Placement.applyStep GameConfig.standard b pl) rest

theorem wordLift_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordLift b (pl :: rest)
      = (4 * pl.dropOffset b + shapeRowMoment pl)
        + wordLift (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

theorem wordRelease_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordRelease b (pl :: rest)
      = (10 * clearedRowSum (pl.place b) + gravityInt (pl.place b))
        + wordRelease (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- The potential ledger along a word. -/
theorem rowMoment_word {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    rowMoment (pls.foldl (Placement.applyStep GameConfig.standard) b)
      + wordRelease b pls = rowMoment b + wordLift b pls := by
  induction pls generalizing b with
  | nil => simp [wordLift, wordRelease]
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    have hstep := rowMoment_applyStep hwf hvpl
    have hrec := ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, wordLift_cons, wordRelease_cons]
    omega

/-- **THE POTENTIAL BALANCE OF A CYCLE**: the height the drops deliver
is exactly the height the clears give back. Four times the total
landing height plus the shapes' moments equals ten times the total
height of the rows removed plus the total fall of the survivors. An
exact identity, not a congruence — and the first cycle law that sees
how HIGH the play happens. -/
theorem cycle_potential_balance {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordLift b pls = wordRelease b pls := by
  have h := rowMoment_word hwf hv
  rw [hfold] at h
  omega

/-! ### Reading the potential balance

Three consequences: the survivors' fall is bounded by the potential
itself; the shapes' own moments are small and computable; and the total
landing height of a cycle is capped by what its clears release. -/

/-- **A survivor never falls further than it stood**: the total fall is
bounded by the board's own potential. -/
theorem gravityInt_le_rowMoment (B : Board) :
    gravityInt B ≤ rowMoment B := by
  classical
  unfold gravityInt rowMoment
  calc ∑ p ∈ B.filter
        (fun p => ¬ Board.isFull GameConfig.standard B p.2),
        Board.clearedBelow GameConfig.standard B p.2
      ≤ ∑ p ∈ B.filter
          (fun p => ¬ Board.isFull GameConfig.standard B p.2), p.2 :=
        Finset.sum_le_sum (fun p _ => clearedBelow_le GameConfig.standard B p.2)
    _ ≤ ∑ p ∈ B, p.2 :=
        Finset.sum_le_sum_of_subset (Finset.filter_subset _ _)

/-- Every shape's own row moment is at most six. -/
theorem shapeRowMoment_le_six :
    ∀ (p : Piece) (r : Rotation), (∑ cell ∈ p.shapeUp r, cell.2) ≤ 6 := by
  decide

/-- The square's row moment is two, whichever way you turn it. -/
theorem shapeRowMoment_O :
    ∀ (r : Rotation), (∑ cell ∈ Piece.O.shapeUp r, cell.2) = 2 := by
  decide

/-- Only the vertical I reaches the maximum moment six. -/
theorem shapeRowMoment_eq_six_iff :
    ∀ (p : Piece) (r : Rotation),
      (∑ cell ∈ p.shapeUp r, cell.2) = 6 ↔ (p = Piece.I ∧ (r = 1 ∨ r = 3)) := by
  decide

/-- The total landing height of a word. -/
def wordDropSum (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      pl.dropOffset b
      + wordDropSum (Placement.applyStep GameConfig.standard b pl) rest

theorem wordDropSum_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordDropSum b (pl :: rest)
      = pl.dropOffset b
        + wordDropSum (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- The lift is at least four times the total landing height. -/
theorem wordLift_ge_drops (b : Board) (pls : List Placement) :
    4 * wordDropSum b pls ≤ wordLift b pls := by
  induction pls generalizing b with
  | nil => simp [wordLift, wordDropSum]
  | cons pl rest ih =>
    rw [wordLift_cons, wordDropSum_cons, Nat.mul_add]
    have := ih (Placement.applyStep GameConfig.standard b pl)
    omega

/-- The lift is also at most four times the landing height plus six per
move — the shapes contribute at most six apiece. -/
theorem wordLift_le_drops (b : Board) (pls : List Placement) :
    wordLift b pls ≤ 4 * wordDropSum b pls + 6 * pls.length := by
  induction pls generalizing b with
  | nil => simp [wordLift, wordDropSum]
  | cons pl rest ih =>
    rw [wordLift_cons, wordDropSum_cons, List.length_cons]
    have hsh : shapeRowMoment pl ≤ 6 :=
      shapeRowMoment_le_six pl.piece pl.rot
    have := ih (Placement.applyStep GameConfig.standard b pl)
    omega

/-- **THE LANDING-HEIGHT CAP**: around a cycle, four times the total
height at which the pieces land is at most what the clears give back.
Playing high costs exactly as much as the clears can repay. -/
theorem cycle_drop_height_cap {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    4 * wordDropSum b pls ≤ wordRelease b pls := by
  have hbal := cycle_potential_balance hwf hv hfold
  have hge := wordLift_ge_drops b pls
  omega

/-- **…and the release is bounded by the play**: what the clears give
back is at most four times the landing height plus six per move. The
two sides of the potential are pinned to each other within a fixed
budget. -/
theorem cycle_release_cap {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordRelease b pls ≤ 4 * wordDropSum b pls + 6 * pls.length := by
  have hbal := cycle_potential_balance hwf hv hfold
  have hle := wordLift_le_drops b pls
  omega

/-- For a legal 35-cycle the release sits within `210` of four times the
landing height. -/
theorem legal_cycle_release_window {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    4 * wordDropSum b w ≤ wordRelease b w
      ∧ wordRelease b w ≤ 4 * wordDropSum b w + 210 := by
  refine ⟨cycle_drop_height_cap hwf hv hfold, ?_⟩
  have h := cycle_release_cap hwf hv hfold
  rw [hlen] at h
  omega

/-! ### The bottom row is frozen at EVERY moment of the cycle

The dichotomy so far pinned only the opening move. Splitting the word
at an arbitrary point instead of indexing into it upgrades this to the
whole loop: the bottom row of every board the cycle visits is the same
set of cells. -/

/-- Bottom-row clearing splits along concatenation. -/
theorem wordBottomClear_append (b : Board) (w1 w2 : List Placement) :
    wordBottomClear b (w1 ++ w2)
      ↔ wordBottomClear b w1
        ∨ wordBottomClear
            (w1.foldl (Placement.applyStep GameConfig.standard) b) w2 := by
  induction w1 generalizing b with
  | nil => simp [wordBottomClear]
  | cons pl rest ih =>
    rw [List.cons_append, wordBottomClear_cons, wordBottomClear_cons,
      List.foldl_cons, ih]
    tauto

/-- A word that never fills the bottom row only ever adds to it. -/
theorem bottomCells_word_subset {b : Board} {pls : List Placement}
    (hnb : ¬ wordBottomClear b pls) :
    bottomCells b
      ⊆ bottomCells (pls.foldl
          (Placement.applyStep GameConfig.standard) b) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordBottomClear_cons] at hnb
    push Not at hnb
    have hstep : bottomCells b
        ⊆ bottomCells (Placement.applyStep GameConfig.standard b pl) := by
      rw [Placement.applyStep_eq_clearLines_place,
        bottomCells_clearLines hnb.1]
      exact bottomCells_subset_place b pl
    rw [List.foldl_cons]
    exact subset_trans hstep (ih hnb.2)

/-- **THE BOTTOM ROW IS FROZEN THROUGHOUT**: if a cycle never fills its
bottom row, then every board it visits carries exactly the same bottom
row — cell for cell, at every moment, not merely after the first move. -/
theorem cycle_bottom_frozen_split {b : Board} {w1 w2 : List Placement}
    (hnb : ¬ wordBottomClear b (w1 ++ w2))
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b) :
    bottomCells (w1.foldl (Placement.applyStep GameConfig.standard) b)
      = bottomCells b := by
  have hsplit := wordBottomClear_append b w1 w2
  have hn1 : ¬ wordBottomClear b w1 := fun h => hnb (hsplit.mpr (Or.inl h))
  have hn2 : ¬ wordBottomClear
      (w1.foldl (Placement.applyStep GameConfig.standard) b) w2 :=
    fun h => hnb (hsplit.mpr (Or.inr h))
  have hsub1 := bottomCells_word_subset hn1
  have hsub2 := bottomCells_word_subset hn2
  rw [← List.foldl_append, hfold] at hsub2
  exact Finset.Subset.antisymm hsub2 hsub1

/-- The floor gap of such a cycle is permanent: a column missing from
the bottom row of the base board is missing from it at every moment. -/
theorem cycle_bottom_gap_permanent {b : Board} {w1 w2 : List Placement}
    (hnb : ¬ wordBottomClear b (w1 ++ w2))
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b)
    {c : ℕ} (hc : ((c, 0) : Coord) ∉ b) :
    ((c, 0) : Coord)
      ∉ w1.foldl (Placement.applyStep GameConfig.standard) b := by
  intro hmem
  have hfrozen := cycle_bottom_frozen_split hnb hfold
  have hin : ((c, 0) : Coord)
      ∈ bottomCells (w1.foldl
          (Placement.applyStep GameConfig.standard) b) := by
    unfold bottomCells
    exact Finset.mem_filter.mpr ⟨hmem, rfl⟩
  rw [hfrozen] at hin
  exact hc (Finset.mem_filter.mp hin).1

/-- **A LEGAL CYCLE THAT SPARES ITS FLOOR CARRIES A PERMANENT GAP**: the
base board of any cycle is clear-free, so its bottom row is missing a
column; if the loop never completes that row, the gap survives every
single move — while the column above it must still take delivery of
fourteen cells. -/
theorem legal_cycle_permanent_floor_gap {b : Board} {w : List Placement}
    (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hnb : ¬ wordBottomClear b w) :
    ∃ c < 10, ((c, 0) : Coord) ∉ b
      ∧ ∀ w1 w2 : List Placement, w = w1 ++ w2 →
          ((c, 0) : Coord)
            ∉ w1.foldl (Placement.applyStep GameConfig.standard) b := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have hnf := board_on_cycle_clear_free
    (b := b) (n := w.length) ⟨w, rfl, hpos, hv, hfold⟩
  have h0 := hnf 0
  have hex : ∃ c ∈ Finset.range GameConfig.standard.cols,
      ((c, 0) : Coord) ∉ b := by
    by_contra hcon
    push Not at hcon
    exact h0 hcon
  obtain ⟨c, hcmem, hcnot⟩ := hex
  rw [GameConfig.standard_cols, Finset.mem_range] at hcmem
  refine ⟨c, hcmem, hcnot, ?_⟩
  intro w1 w2 hw
  subst hw
  exact cycle_bottom_gap_permanent hnb hfold hcnot

/-! ### Splitting, applied twice more

The split technique upgrades any "first move" statement to "every
moment" for free. Two more: the debt of a non-recycling cycle, and the
frozen foundation at arbitrary depth. -/

/-- Burial splits along concatenation. -/
theorem wordBuried_append (b : Board) (w1 w2 : List Placement) :
    wordBuried b (w1 ++ w2)
      = wordBuried b w1
        + wordBuried (w1.foldl
            (Placement.applyStep GameConfig.standard) b) w2 := by
  induction w1 generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [List.cons_append, wordBuried_cons, wordBuried_cons,
      List.foldl_cons, ih]
    omega

/-- Unburial splits along concatenation. -/
theorem wordUnburied_append (b : Board) (w1 w2 : List Placement) :
    wordUnburied b (w1 ++ w2)
      = wordUnburied b w1
        + wordUnburied (w1.foldl
            (Placement.applyStep GameConfig.standard) b) w2 := by
  induction w1 generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [List.cons_append, wordUnburied_cons, wordUnburied_cons,
      List.foldl_cons, ih]
    omega

/-- A word that frees no holes never lowers the debt. -/
theorem debt_word_ge {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hnu : wordUnburied b pls = 0) :
    HoleDebt.debt GameConfig.standard b
      ≤ HoleDebt.debt GameConfig.standard
          (pls.foldl (Placement.applyStep GameConfig.standard) b) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    rw [wordUnburied_cons] at hnu
    have h1 := debt_le_debt_place b pl
    have h2 := debt_applyStep_le b pl hwf hvpl
    have hstep : HoleDebt.debt GameConfig.standard b
        ≤ HoleDebt.debt GameConfig.standard
            (Placement.applyStep GameConfig.standard b pl) := by omega
    rw [List.foldl_cons]
    exact le_trans hstep (ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq])) (by omega))

/-- **THE DEBT IS FROZEN THROUGHOUT**: a cycle that frees no holes
carries exactly the same hole count on every board it visits. It cannot
even temporarily dig and repay — the debt never moves at all. -/
theorem cycle_debt_frozen_split {b : Board} {w1 w2 : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w1 ++ w2, pl.Valid GameConfig.standard)
    (hnu : wordUnburied b (w1 ++ w2) = 0)
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b) :
    HoleDebt.debt GameConfig.standard
        (w1.foldl (Placement.applyStep GameConfig.standard) b)
      = HoleDebt.debt GameConfig.standard b := by
  rw [wordUnburied_append] at hnu
  have hv1 : ∀ pl ∈ w1, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_append_left _ hpl)
  have hv2 : ∀ pl ∈ w2, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_append_right _ hpl)
  have hwf1 := foldl_applyStep_wf hwf hv1
  have hge1 := debt_word_ge hwf hv1 (by omega)
  have hge2 := debt_word_ge hwf1 hv2 (by omega)
  rw [← List.foldl_append, hfold] at hge2
  omega

/-- The word fills some row at or below depth `r` at some point. -/
def wordLowClear (r : ℕ) (b : Board) : List Placement → Prop
  | [] => False
  | pl :: rest =>
      (∃ s, s ≤ r ∧ Board.isFull GameConfig.standard (pl.place b) s)
      ∨ wordLowClear r (Placement.applyStep GameConfig.standard b pl) rest

theorem wordLowClear_cons (r : ℕ) (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordLowClear r b (pl :: rest)
      ↔ (∃ s, s ≤ r ∧ Board.isFull GameConfig.standard (pl.place b) s)
        ∨ wordLowClear r
            (Placement.applyStep GameConfig.standard b pl) rest := Iff.rfl

/-- Low bands only grow under placement. -/
theorem lowCells_subset_place (r : ℕ) (b : Board) (pl : Placement) :
    lowCells r b ⊆ lowCells r (pl.place b) := by
  unfold lowCells
  apply Finset.filter_subset_filter
  rw [Placement.place_eq_union_dropped]
  exact Finset.subset_union_left

/-- A word that never fills a low row only ever adds to the low band. -/
theorem lowCells_word_subset {r : ℕ} {b : Board} {pls : List Placement}
    (hnb : ¬ wordLowClear r b pls) :
    lowCells r b
      ⊆ lowCells r (pls.foldl
          (Placement.applyStep GameConfig.standard) b) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordLowClear_cons] at hnb
    push Not at hnb
    have hnf : ∀ s, s ≤ r →
        ¬ Board.isFull GameConfig.standard (pl.place b) s := by
      intro s hs hfull
      exact hnb.1 s hs hfull
    have hstep : lowCells r b
        ⊆ lowCells r (Placement.applyStep GameConfig.standard b pl) := by
      rw [Placement.applyStep_eq_clearLines_place,
        lowCells_clearLines hnf]
      exact lowCells_subset_place r b pl
    rw [List.foldl_cons]
    exact subset_trans hstep (ih hnb.2)

/-- Low clearing splits along concatenation. -/
theorem wordLowClear_append (r : ℕ) (b : Board) (w1 w2 : List Placement) :
    wordLowClear r b (w1 ++ w2)
      ↔ wordLowClear r b w1
        ∨ wordLowClear r
            (w1.foldl (Placement.applyStep GameConfig.standard) b) w2 := by
  induction w1 generalizing b with
  | nil => simp [wordLowClear]
  | cons pl rest ih =>
    rw [List.cons_append, wordLowClear_cons, wordLowClear_cons,
      List.foldl_cons, ih, or_assoc]

/-- **THE FOUNDATION IS FROZEN THROUGHOUT**: below the lowest row a
cycle ever completes, the board is not merely undisturbed at the end —
it is identical at every single moment of the loop. A permanent
substructure the play never touches. -/
theorem cycle_lowCells_frozen_split {r : ℕ} {b : Board}
    {w1 w2 : List Placement}
    (hnb : ¬ wordLowClear r b (w1 ++ w2))
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b) :
    lowCells r (w1.foldl (Placement.applyStep GameConfig.standard) b)
      = lowCells r b := by
  have hsplit := wordLowClear_append r b w1 w2
  have hn1 : ¬ wordLowClear r b w1 := fun h => hnb (hsplit.mpr (Or.inl h))
  have hn2 : ¬ wordLowClear r
      (w1.foldl (Placement.applyStep GameConfig.standard) b) w2 :=
    fun h => hnb (hsplit.mpr (Or.inr h))
  have hsub1 := lowCells_word_subset hn1
  have hsub2 := lowCells_word_subset hn2
  rw [← List.foldl_append, hfold] at hsub2
  exact Finset.Subset.antisymm hsub2 hsub1

/-! ### From a safe legal word to a certified closed cycle

Everything the word theory needs is now in place: the orbit's thirty-five
states are distinct, the table policy reproduces the word on them, every
draw along the way is legal, and the loop closes. Assembling these gives
the library's canonical `ClosedCycle` — and with it infinite play. -/

/-- Membership in a legal cycle's orbit names a position. -/
theorem mem_wordOrbit_iff {b : Board} {w : List Placement}
    (hlen : w.length = 35) (s : GameState) :
    s ∈ (wordOrbit b w).toFinset
      ↔ ∃ i, i < 35 ∧ wordPlay ⟨b, Bag.full⟩ w i = s := by
  rw [List.mem_toFinset]
  unfold wordOrbit
  rw [List.mem_map]
  constructor
  · rintro ⟨i, hi, hval⟩
    rw [List.mem_range, hlen] at hi
    exact ⟨i, hi, hval⟩
  · rintro ⟨i, hi, hval⟩
    exact ⟨i, by rw [List.mem_range, hlen]; exact hi, hval⟩

/-- **THE CERTIFIED CLOSED CYCLE**: a bag-legal 35-word that folds a
well-formed board back to itself and never tops out along the way is a
`ClosedCycle` of the game — states, policy, validity, legal draws,
safety and closure all discharged. -/
theorem legal_safe_word_closedCycle {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35)
    (hsafe : ∀ i, i < 35 →
      ¬ (wordPlay ⟨b, Bag.full⟩ w i).lost GameConfig.standard) :
    ∃ C : ClosedCycle GameConfig.standard,
      (⟨b, Bag.full⟩ : GameState) ∈ C.states := by
  classical
  have h35 := legal_cycle_word_thirty_five_dvd hwf hne hv hbag hfold
  have h7 : 7 ∣ w.length := by omega
  have hcyc := legal_cycle_word_state_cycle hwf hne hv hbag hfold
  have hgetD : ∀ i, i < 35 →
      w.getD i ⟨Piece.O, 0, 0⟩ ∈ w := by
    intro i hi
    rw [List.getD_eq_getElem w _ (by omega)]
    exact List.getElem_mem _
  have hstep : ∀ i, i < 35 →
      (wordPlay ⟨b, Bag.full⟩ w i).step GameConfig.standard
        (wordPolicy b w (wordPlay ⟨b, Bag.full⟩ w i))
        = wordPlay ⟨b, Bag.full⟩ w (i + 1) := by
    intro i hi
    rw [wordPolicy_eval hwf hne hv hbag hfold hlen hi, wordPlay_succ,
      Nat.mod_eq_of_lt (by omega)]
  refine ⟨⟨(wordOrbit b w).toFinset, wordPolicy b w, ?_, ?_, ?_, ?_⟩, ?_⟩
  · intro s hs
    obtain ⟨i, hi, rfl⟩ := (mem_wordOrbit_iff hlen s).mp hs
    rw [wordPolicy_eval hwf hne hv hbag hfold hlen hi]
    exact hv _ (hgetD i hi)
  · intro s hs
    obtain ⟨i, hi, rfl⟩ := (mem_wordOrbit_iff hlen s).mp hs
    rw [wordPolicy_eval hwf hne hv hbag hfold hlen hi]
    exact legal_word_draw_legal (b := b) hbag h7 (by omega)
  · intro s hs
    obtain ⟨i, hi, rfl⟩ := (mem_wordOrbit_iff hlen s).mp hs
    exact hsafe i hi
  · intro s hs
    obtain ⟨i, hi, rfl⟩ := (mem_wordOrbit_iff hlen s).mp hs
    rw [hstep i hi, mem_wordOrbit_iff hlen]
    by_cases hlt : i + 1 < 35
    · exact ⟨i + 1, hlt, rfl⟩
    · refine ⟨0, by omega, ?_⟩
      have hi34 : i + 1 = 35 := by omega
      rw [hi34, wordPlay_mod hne hcyc 35, hlen]
  · rw [mem_wordOrbit_iff hlen]
    exact ⟨0, by omega, rfl⟩

/-- **…and therefore infinite play.** A safe, bag-legal 35-cycle word is
a complete M2 certificate: the game entered at its base board survives
forever under the table policy the word induces. -/
theorem legal_safe_word_survives {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35)
    (hsafe : ∀ i, i < 35 →
      ¬ (wordPlay ⟨b, Bag.full⟩ w i).lost GameConfig.standard) :
    ∃ (C : ClosedCycle GameConfig.standard),
      SurvivesForever GameConfig.standard C.policy ⟨b, Bag.full⟩ := by
  obtain ⟨C, hC⟩ := legal_safe_word_closedCycle hwf hne hv hbag hfold
    hlen hsafe
  exact ⟨C, closed_cycle_survives C hC⟩

/-! ### The profile of an M2 witness

Everything proved about legal 35-cycles, collected. Any word that could
serve as the M2 certificate must satisfy every clause below — and, if it
is also safe, `legal_safe_word_survives` turns it into the certificate.
Necessary conditions on one side, sufficient on the other. -/

/-- Moves of a word that clear nothing. -/
def wordDryMoves (b : Board) : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (if (Board.fullRows GameConfig.standard (pl.place b)).card = 0
        then 1 else 0)
      + wordDryMoves (Placement.applyStep GameConfig.standard b pl) rest

@[simp] theorem wordDryMoves_nil (b : Board) : wordDryMoves b [] = 0 := rfl

theorem wordDryMoves_cons (b : Board) (pl : Placement)
    (rest : List Placement) :
    wordDryMoves b (pl :: rest)
      = (if (Board.fullRows GameConfig.standard (pl.place b)).card = 0
          then 1 else 0)
        + wordDryMoves (Placement.applyStep GameConfig.standard b pl)
            rest := rfl

/-- Every move either clears or is dry. -/
theorem wordClearMoves_add_dry (b : Board) (pls : List Placement) :
    wordClearMoves b pls + wordDryMoves b pls = pls.length := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordClearMoves_cons, wordDryMoves_cons, List.length_cons]
    have hrec := ih (Placement.applyStep GameConfig.standard b pl)
    by_cases hc : 0 < (Board.fullRows GameConfig.standard
        (pl.place b)).card
    · rw [if_pos hc, if_neg (by omega)]
      omega
    · rw [if_neg hc, if_pos (by omega)]
      omega

/-- **A legal 35-cycle is mostly quiet**: at least twenty-one of its
thirty-five moves clear nothing at all. -/
theorem legal_cycle_dry_moves_ge {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    21 ≤ wordDryMoves b w := by
  have hsum := wordClearMoves_add_dry b w
  have hbr := legal_cycle_word_clearing_moves_bracket hwf hne hv hbag
    hfold hlen
  rw [hlen] at hsum
  omega

/-- **THE M2 WITNESS PROFILE.** Every clause is forced. A bag-legal
35-cycle word on a well-formed board:

1. lives on a clear-free base board;
2. delivers exactly fourteen cells to each of the ten columns;
3. clears exactly fourteen rows, on between four and fourteen moves,
   leaving at least twenty-one moves dry;
4. plays at most five tetrises, one per bag;
5. satisfies the exact drop-column equation `∑(4·col + shapeMoment) = 630`;
6. balances its potential exactly: the height its drops deliver is the
   height its clears give back;
7. makes at least one PARTIAL clear — one that removes a row and leaves
   a cell standing — so it is neither all-dry nor a chain of perfect
   clears;
8. cannot be built from doubles and tetrises alone;
9. lays at least one T flat.

Clauses 7 to 9 come from the checkerboard charge surviving the passage
through clears, which is what lifts this list beyond pure counting. -/
theorem legal_cycle_profile {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    (∀ r, ¬ Board.isFull GameConfig.standard b r)
    ∧ (∀ c, c < 10 → wordColProfile c w = 14)
    ∧ (wordClears b w = 14
        ∧ 4 ≤ wordClearMoves b w ∧ wordClearMoves b w ≤ 14
        ∧ 21 ≤ wordDryMoves b w)
    ∧ wordTetrises b w ≤ 5
    ∧ (w.map (fun pl => 4 * pl.col + shapeMoment pl)).sum = 630
    ∧ wordLift b w = wordRelease b w
    ∧ (∃ (c : Board) (pl : Placement),
        Board.fullRows GameConfig.standard (pl.place c) ≠ ∅
        ∧ ∃ p ∈ pl.place c,
            ¬ Board.isFull GameConfig.standard (pl.place c) p.2)
    ∧ (∃ (c : Board) (pl : Placement), ∀ t k : ℕ, Even k →
        Board.fullRows GameConfig.standard (pl.place c)
          ≠ Finset.Ico t (t + k))
    ∧ (∃ pl ∈ w, pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2)) := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  refine ⟨board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_⟩
  · intro c hc
    exact legal_cycle_column_fourteen hwf hne hv hbag hfold hlen hc
  · refine ⟨legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen,
      ?_, ?_, legal_cycle_dry_moves_ge hwf hne hv hbag hfold hlen⟩
    · exact (legal_cycle_word_clearing_moves_bracket hwf hne hv hbag
        hfold hlen).1
    · exact (legal_cycle_word_clearing_moves_bracket hwf hne hv hbag
        hfold hlen).2
  · have h := legal_cycle_word_tetris_cap hwf hne hv hbag hfold
    rw [hlen] at h
    omega
  · exact legal_cycle_moment_630 hwf hne hv hbag hfold hlen
  · exact cycle_potential_balance hwf hv hfold
  · exact legal_cycle_has_partial_clear hwf hne hv hbag hfold hlen
  · exact legal_cycle_has_odd_or_split_clear hwf hne hv hbag hfold hlen
  · exact legal_cycle_has_flat_T hwf hne hv hbag hfold hlen

/-! ### What an empty floor costs

Suppose a cycle never completes its bottom row and starts with that row
empty. The freeze then keeps row zero empty for the entire loop — so
every occupied column stands over a hole. At any clearing moment a full
row touches all ten columns at once, so all ten are occupied, and the
board is carrying ten holes simultaneously. An empty floor is not free;
it is a standing debt of ten. -/

/-- A column with a gap at the floor and anything above it holds a hole. -/
theorem colHoles_pos_of_floor_gap {b : Board} {c : ℕ}
    (hgap : ((c, 0) : Coord) ∉ b) (hpos : 0 < b.colHeight c) :
    1 ≤ HoleDebt.colHoles b c := by
  classical
  have hsub : b.colRows c ⊆ (Finset.range (b.colHeight c)).erase 0 := by
    intro x hx
    rw [Finset.mem_erase, Finset.mem_range]
    constructor
    · intro hx0
      apply hgap
      unfold Board.colRows at hx
      rw [Finset.mem_image] at hx
      obtain ⟨q, hq, hq2⟩ := hx
      obtain ⟨hqb, hq1⟩ := Finset.mem_filter.mp hq
      have : q = (c, 0) := Prod.ext_iff.mpr ⟨hq1, by omega⟩
      rw [← this]
      exact hqb
    · have hle : x + 1 ≤ b.colHeight c := by
        unfold Board.colHeight
        exact Finset.le_sup (f := (· + 1)) hx
      omega
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_erase_of_mem (Finset.mem_range.mpr hpos),
    Finset.card_range] at hcard
  unfold HoleDebt.colHoles
  omega

/-- **AN EMPTY FLOOR UNDER A FULL ROW COSTS TEN HOLES**: if the bottom
row is empty while some row is complete, every one of the ten columns
stands over a gap. -/
theorem debt_ge_ten_of_empty_floor {B : Board}
    (hempty : ∀ c, ((c, 0) : Coord) ∉ B)
    {t : ℕ} (hfull : Board.isFull GameConfig.standard B t) :
    10 ≤ HoleDebt.debt GameConfig.standard B := by
  classical
  have hcol : ∀ c ∈ Finset.range 10, 1 ≤ HoleDebt.colHoles B c := by
    intro c hc
    have hmem := hfull c (by rw [GameConfig.standard_cols]; exact hc)
    have hpos : 0 < B.colHeight c := by
      have := Board.lt_colHeight hmem
      omega
    exact colHoles_pos_of_floor_gap (hempty c) hpos
  have hsum : ∑ _c ∈ Finset.range 10, 1
      ≤ ∑ c ∈ Finset.range 10, HoleDebt.colHoles B c :=
    Finset.sum_le_sum hcol
  rw [Finset.sum_const, Finset.card_range, smul_eq_mul, mul_one] at hsum
  unfold HoleDebt.debt
  rw [GameConfig.standard_cols]
  exact hsum

/-- The frozen floor of an empty-floor cycle stays empty at every
moment. -/
theorem cycle_floor_stays_empty {b : Board} {w1 w2 : List Placement}
    (hnb : ¬ wordBottomClear b (w1 ++ w2))
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b)
    (hempty : ∀ c, ((c, 0) : Coord) ∉ b) :
    ∀ c, ((c, 0) : Coord)
      ∉ w1.foldl (Placement.applyStep GameConfig.standard) b :=
  fun c => cycle_bottom_gap_permanent hnb hfold (hempty c)

/-- **THE STANDING DEBT OF TEN**: a cycle that begins with an empty
bottom row and never completes it carries ten holes at every clearing
moment — one under each column. Playing above an empty floor is paid
for continuously, not once. -/
theorem cycle_empty_floor_debt {b : Board} {w1 w2 : List Placement}
    (hnb : ¬ wordBottomClear b (w1 ++ w2))
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b)
    (hempty : ∀ c, ((c, 0) : Coord) ∉ b)
    {t : ℕ}
    (hfull : Board.isFull GameConfig.standard
      (w1.foldl (Placement.applyStep GameConfig.standard) b) t) :
    10 ≤ HoleDebt.debt GameConfig.standard
      (w1.foldl (Placement.applyStep GameConfig.standard) b) :=
  debt_ge_ten_of_empty_floor (cycle_floor_stays_empty hnb hfold hempty)
    hfull

/-! ### Counting the flat T's

The row-moment table splits cleanly: a shape's row charge is the L
indicator plus the J indicator plus the flat-T indicator, with nothing
else contributing. So the parity debt of a legal cycle reads directly as
a count of flat T's — and since only five T's are played at all, that
count is one, three or five. -/

/-- Casting a zero-or-one through to `ZMod 2` commutes with the test. -/
theorem cast_ite_one_zero (P : Prop) [Decidable P] :
    (((if P then 1 else 0 : ℕ)) : ZMod 2) = if P then 1 else 0 := by
  split <;> simp

/-- A natural casts to one in `ZMod 2` exactly when it is odd. -/
theorem natCast_two_eq_one_iff (n : ℕ) :
    ((n : ℕ) : ZMod 2) = 1 ↔ n % 2 = 1 := by
  have h : ((n : ℕ) : ZMod 2) = ((n % 2 : ℕ) : ZMod 2) :=
    (ZMod.natCast_mod n 2).symm
  rcases (show n % 2 = 0 ∨ n % 2 = 1 by omega) with h0 | h1
  · rw [h, h0]
    constructor
    · intro hc
      exact absurd hc (by decide)
    · intro hc
      exact absurd hc (by decide)
  · rw [h, h1]
    simp

/-- The row charge splits into an L part, a J part and a flat-T part. -/
theorem shapeRowCharge_split3 :
    ∀ (p : Piece) (r : Rotation),
      (∑ cell ∈ p.shapeUp r, ((cell.2 : ℕ) : ZMod 2))
        = (if p = Piece.L then 1 else 0) + (if p = Piece.J then 1 else 0)
          + (if p = Piece.T ∧ (r = 0 ∨ r = 2) then 1 else 0) := by
  decide

/-- How many of a word's placements lay a T flat. -/
def wordFlatTCount : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (if pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2) then 1 else 0)
        + wordFlatTCount rest

@[simp] theorem wordFlatTCount_nil : wordFlatTCount [] = 0 := rfl

theorem wordFlatTCount_cons (pl : Placement) (rest : List Placement) :
    wordFlatTCount (pl :: rest)
      = (if pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2) then 1 else 0)
        + wordFlatTCount rest := rfl

/-- **The word's row charge counts its L's, J's and flat T's.** -/
theorem wordRowCharge_eq_counts (pls : List Placement) :
    wordRowCharge pls
      = wordPieceCharge Piece.L pls + wordPieceCharge Piece.J pls
        + ((wordFlatTCount pls : ℕ) : ZMod 2) := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    have hs : shapeRowCharge pl
        = (if pl.piece = Piece.L then 1 else 0)
          + (if pl.piece = Piece.J then 1 else 0)
          + (if pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2)
              then 1 else 0) := shapeRowCharge_split3 pl.piece pl.rot
    rw [wordRowCharge_cons, ih, wordPieceCharge_cons, wordPieceCharge_cons,
      wordFlatTCount_cons, Nat.cast_add, cast_ite_one_zero, hs]
    ring

/-- **THE FLAT-T COUNT IS ODD**: a legal 35-cycle lays an odd number of
its T's flat. Ten L's and J's cancel in pairs, leaving the entire parity
debt on the flat T's. -/
theorem legal_cycle_flatT_odd {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordFlatTCount w % 2 = 1 := by
  have hodd := legal_cycle_rowCharge_odd hwf hne hv hbag hfold hlen
  have hL := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.L
  have hJ := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.J
  rw [census_eq_count, hlen] at hL hJ
  norm_num at hL hJ
  rw [wordRowCharge_eq_counts, wordPieceCharge_eq_count,
    wordPieceCharge_eq_count, hL, hJ] at hodd
  rw [show (((5 : ℕ)) : ZMod 2) + (((5 : ℕ)) : ZMod 2) = 0 from by decide,
    zero_add] at hodd
  exact (natCast_two_eq_one_iff _).mp hodd

/-- A flat T is a T, so the flat count never exceeds the T count. -/
theorem wordFlatTCount_le_T (pls : List Placement) :
    wordFlatTCount pls ≤ (pls.map (·.piece)).count Piece.T := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    rw [wordFlatTCount_cons, List.map_cons, List.count_cons]
    by_cases hF : pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2)
    · rw [if_pos hF, if_pos (show
        (((fun x : Placement => x.piece) pl) == Piece.T) = true from by
          simp [hF.1])]
      omega
    · rw [if_neg hF]
      split <;> omega

/-- **ONE, THREE OR FIVE**: a legal 35-cycle lays an odd number of its
five T's flat, and there are only five T's — so the flat count is
exactly one, three or five. -/
theorem legal_cycle_flatT_one_three_or_five {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordFlatTCount w = 1 ∨ wordFlatTCount w = 3
      ∨ wordFlatTCount w = 5 := by
  have hodd := legal_cycle_flatT_odd hwf hne hv hbag hfold hlen
  have hT := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.T
  rw [census_eq_count, hlen] at hT
  norm_num at hT
  have hle := wordFlatTCount_le_T w
  rw [hT] at hle
  omega

/-! ### The column charge, and the upright T's

The checkerboard colour is the sum of the two coordinates, so the column
charge is the checkerboard charge plus the row charge — no new induction
needed, just character-two algebra on the two laws already proved. Its
shape table is the exact mirror of the row table: I, O, S, Z always
even, L and J always odd, and T odd precisely in its UPRIGHT rotations.

Where the row charge counted flat T's, the column charge counts upright
ones — and it forces their number to be even. -/

/-- The column charge of a board: its cells' column indices, mod two. -/
def colCharge (b : Board) : ZMod 2 := ∑ p ∈ b, ((p.1 : ℕ) : ZMod 2)

/-- A shape's own column moment, mod two. -/
def shapeColCharge (pl : Placement) : ZMod 2 :=
  ∑ cell ∈ pl.shapeUp, ((cell.1 : ℕ) : ZMod 2)

/-- The checkerboard charge splits into column and row parts. -/
theorem charge_eq_col_add_row (s : Finset Coord) :
    BagGrowth.charge s
      = (∑ p ∈ s, ((p.1 : ℕ) : ZMod 2))
        + ∑ p ∈ s, ((p.2 : ℕ) : ZMod 2) := by
  unfold BagGrowth.charge
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun p _ => by push_cast; ring)

/-- Hence the column charge is the checkerboard charge plus the row
charge. -/
theorem colCharge_eq (b : Board) :
    colCharge b = BagGrowth.charge b + rowCharge b := by
  have h := charge_eq_col_add_row b
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  have key := hchar (∑ p ∈ b, ((p.2 : ℕ) : ZMod 2))
  unfold colCharge rowCharge
  first
    | linear_combination -h - key
    | linear_combination h - key
    | linear_combination -h + key

/-- The same decomposition for a shape. -/
theorem shape_charge_eq (pl : Placement) :
    BagGrowth.charge pl.shapeUp
      = shapeColCharge pl + shapeRowCharge pl := by
  unfold shapeColCharge shapeRowCharge
  exact charge_eq_col_add_row pl.shapeUp

/-- **A drop adds its shape's column moment.** -/
theorem colCharge_place (b : Board) (pl : Placement) :
    colCharge (pl.place b) = colCharge b + shapeColCharge pl := by
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  rw [colCharge_eq, colCharge_eq, BagGrowth.charge_place, rowCharge_place,
    shape_charge_eq]
  have key := hchar (shapeRowCharge pl)
  first
    | linear_combination key
    | linear_combination -key

/-- **THE COLUMN-CHARGE LAW ACROSS A CLEAR**: the gravity work cancels
between the two halves, leaving only the row count. Columns do not care
how far anything falls. -/
theorem colCharge_clearLines {b : Board}
    (hwf : Board.WF GameConfig.standard b) :
    colCharge (Board.clearLines GameConfig.standard b)
      = colCharge b
        + ((Board.fullRows GameConfig.standard b).card : ZMod 2) := by
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  rw [colCharge_eq, colCharge_eq, charge_clearLines hwf,
    rowCharge_clearLines hwf]
  have key := hchar (gravityWork b)
  first
    | linear_combination key
    | linear_combination -key

/-- The column-charge law for a whole move. -/
theorem colCharge_applyStep {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    colCharge (Placement.applyStep GameConfig.standard b pl)
      = colCharge b + shapeColCharge pl
        + ((Board.fullRows GameConfig.standard (pl.place b)).card
            : ZMod 2) := by
  rw [Placement.applyStep_eq_clearLines_place,
    colCharge_clearLines (Placement.place_wf hwf hv), colCharge_place]

/-- The word's total shape column charge. -/
def wordColCharge : List Placement → ZMod 2
  | [] => 0
  | pl :: rest => shapeColCharge pl + wordColCharge rest

@[simp] theorem wordColCharge_nil : wordColCharge [] = 0 := rfl

theorem wordColCharge_cons (pl : Placement) (rest : List Placement) :
    wordColCharge (pl :: rest)
      = shapeColCharge pl + wordColCharge rest := rfl

/-- The column-charge ledger along a word. -/
theorem colCharge_word {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) :
    colCharge (pls.foldl (Placement.applyStep GameConfig.standard) b)
      = colCharge b + wordColCharge pls
        + ((wordClears b pls : ℕ) : ZMod 2) := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    have hvpl := hv pl (by simp)
    have hstep := colCharge_applyStep hwf hvpl
    have hrec := ih (Placement.applyStep_wf hwf hvpl)
      (fun q hq => hv q (by simp [hq]))
    rw [List.foldl_cons, hrec, hstep, wordColCharge_cons, wordClears_cons]
    push_cast
    ring

/-- **The column cycle law**: around a loop the shapes' total column
moment equals the cleared-row count. -/
theorem cycle_colCharge_law {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard)
    (hfold : pls.foldl (Placement.applyStep GameConfig.standard) b = b) :
    wordColCharge pls = ((wordClears b pls : ℕ) : ZMod 2) := by
  have h := colCharge_word hwf hv
  rw [hfold] at h
  have hchar : ∀ x : ZMod 2, x + x = 0 := by decide
  have key := hchar (colCharge b)
  have key2 := hchar (((wordClears b pls : ℕ) : ZMod 2))
  first
    | linear_combination -h - key2
    | linear_combination h - key2
    | linear_combination -h - key - key2

/-- The column charge splits into L, J and UPRIGHT-T indicators — the
exact mirror of the row table. -/
theorem shapeColCharge_split3 :
    ∀ (p : Piece) (r : Rotation),
      (∑ cell ∈ p.shapeUp r, ((cell.1 : ℕ) : ZMod 2))
        = (if p = Piece.L then 1 else 0) + (if p = Piece.J then 1 else 0)
          + (if p = Piece.T ∧ (r = 1 ∨ r = 3) then 1 else 0) := by
  decide

/-- How many of a word's placements stand a T upright. -/
def wordUprightTCount : List Placement → ℕ
  | [] => 0
  | pl :: rest =>
      (if pl.piece = Piece.T ∧ (pl.rot = 1 ∨ pl.rot = 3) then 1 else 0)
        + wordUprightTCount rest

@[simp] theorem wordUprightTCount_nil : wordUprightTCount [] = 0 := rfl

theorem wordUprightTCount_cons (pl : Placement) (rest : List Placement) :
    wordUprightTCount (pl :: rest)
      = (if pl.piece = Piece.T ∧ (pl.rot = 1 ∨ pl.rot = 3) then 1 else 0)
        + wordUprightTCount rest := rfl

theorem wordColCharge_eq_counts (pls : List Placement) :
    wordColCharge pls
      = wordPieceCharge Piece.L pls + wordPieceCharge Piece.J pls
        + ((wordUprightTCount pls : ℕ) : ZMod 2) := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    have hs : shapeColCharge pl
        = (if pl.piece = Piece.L then 1 else 0)
          + (if pl.piece = Piece.J then 1 else 0)
          + (if pl.piece = Piece.T ∧ (pl.rot = 1 ∨ pl.rot = 3)
              then 1 else 0) := shapeColCharge_split3 pl.piece pl.rot
    rw [wordColCharge_cons, ih, wordPieceCharge_cons, wordPieceCharge_cons,
      wordUprightTCount_cons, Nat.cast_add, cast_ite_one_zero, hs]
    ring

/-- An upright T is a T. -/
theorem wordUprightTCount_le_T (pls : List Placement) :
    wordUprightTCount pls ≤ (pls.map (·.piece)).count Piece.T := by
  induction pls with
  | nil => simp
  | cons pl rest ih =>
    rw [wordUprightTCount_cons, List.map_cons, List.count_cons]
    by_cases hF : pl.piece = Piece.T ∧ (pl.rot = 1 ∨ pl.rot = 3)
    · rw [if_pos hF, if_pos (show
        (((fun x : Placement => x.piece) pl) == Piece.T) = true from by
          simp [hF.1])]
      omega
    · rw [if_neg hF]
      split <;> omega

/-- **ZERO, TWO OR FOUR**: where the row charge made the flat T's odd,
the column charge makes the UPRIGHT T's even. A legal 35-cycle stands an
even number of its five T's on end. -/
theorem legal_cycle_uprightT_even {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordUprightTCount w = 0 ∨ wordUprightTCount w = 2
      ∨ wordUprightTCount w = 4 := by
  have hlaw := cycle_colCharge_law hwf hv hfold
  have hclears := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold
    hlen
  have hL := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.L
  have hJ := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.J
  have hT := legal_cycle_word_piece_census hwf hne hv hbag hfold Piece.T
  rw [census_eq_count, hlen] at hL hJ hT
  norm_num at hL hJ hT
  rw [wordColCharge_eq_counts, wordPieceCharge_eq_count,
    wordPieceCharge_eq_count, hL, hJ, hclears] at hlaw
  rw [show (((5 : ℕ)) : ZMod 2) + (((5 : ℕ)) : ZMod 2) = 0 from by decide,
    zero_add] at hlaw
  have hne1 : ¬ (wordUprightTCount w % 2 = 1) := by
    intro hodd
    rw [(natCast_two_eq_one_iff _).mpr hodd] at hlaw
    revert hlaw
    decide
  have hle := wordUprightTCount_le_T w
  rw [hT] at hle
  omega

/-! ### A clear-free board holds at most 180

The usual ceiling on the board's mass is the capacity `10 × 20 = 200`.
But a board with no complete row is missing a cell in every row it
occupies, so it can hold at most nine per row — a hundred and eighty in
all. Every board a cycle visits is clear-free, so that sharper ceiling
holds at every moment of the loop, not merely at its base. -/

/-- **A clear-free row is short of full**: on a well-formed board, a row
that is not complete holds at most nine cells. -/
theorem rowCount_le_nine {b : Board}
    (hwf : Board.WF GameConfig.standard b) {r : ℕ}
    (hnf : ¬ Board.isFull GameConfig.standard b r) :
    Board.rowCount b r ≤ 9 := by
  classical
  by_contra hgt
  push Not at hgt
  have hsub : b.filter (fun p => p.2 = r)
      ⊆ (Finset.range 10).image (fun c => ((c, r) : Coord)) := by
    intro p hp
    obtain ⟨hpb, hpr⟩ := Finset.mem_filter.mp hp
    rw [Finset.mem_image]
    refine ⟨p.1, ?_, ?_⟩
    · have := hwf p hpb
      rw [GameConfig.standard_cols] at this
      exact Finset.mem_range.mpr this
    · exact Prod.ext_iff.mpr ⟨rfl, hpr.symm⟩
  have hcard : (b.filter (fun p => p.2 = r)).card
      ≤ ((Finset.range 10).image (fun c => ((c, r) : Coord))).card :=
    Finset.card_le_card hsub
  have himg : ((Finset.range 10).image
      (fun c => ((c, r) : Coord))).card = 10 := by
    rw [Finset.card_image_of_injective _ (fun x y hxy => by
      simpa using congrArg Prod.fst hxy), Finset.card_range]
  have heq : b.filter (fun p => p.2 = r)
      = (Finset.range 10).image (fun c => ((c, r) : Coord)) := by
    apply Finset.eq_of_subset_of_card_le hsub
    unfold Board.rowCount at hgt
    omega
  apply hnf
  intro c hc
  rw [GameConfig.standard_cols] at hc
  have hmem : ((c, r) : Coord)
      ∈ (Finset.range 10).image (fun c => ((c, r) : Coord)) := by
    rw [Finset.mem_image]
    exact ⟨c, hc, rfl⟩
  rw [← heq, Finset.mem_filter] at hmem
  exact hmem.1

/-- The board's mass is its rows' masses, over the field. -/
theorem count_eq_sum_rowCount {b : Board}
    (hin : ∀ p ∈ b, p.2 < 20) :
    b.card = ∑ r ∈ Finset.range 20, Board.rowCount b r := by
  classical
  unfold Board.rowCount
  rw [Finset.card_eq_sum_card_fiberwise
    (f := fun p : Coord => p.2) (t := Finset.range 20)
    (fun p hp => Finset.mem_range.mpr (hin p hp))]

/-- **THE CLEAR-FREE CEILING**: a well-formed, in-field board with no
complete row holds at most a hundred and eighty cells — nine per row,
not ten. -/
theorem count_le_180 {b : Board}
    (hwf : Board.WF GameConfig.standard b)
    (hin : ∀ p ∈ b, p.2 < 20)
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    b.card ≤ 180 := by
  classical
  rw [count_eq_sum_rowCount hin]
  calc ∑ r ∈ Finset.range 20, Board.rowCount b r
      ≤ ∑ _r ∈ Finset.range 20, 9 :=
        Finset.sum_le_sum (fun r _ => rowCount_le_nine hwf (hnf r))
    _ = 180 := by simp

/-- **A CYCLE NEVER CARRIES MORE THAN 180**: every board a loop visits
is the image of a move, hence clear-free, hence nine-per-row at most.
The mass ceiling for cyclic play is a tenth lower than the raw
capacity. -/
theorem cycle_mass_le_180 {b : Board} {w1 w2 : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w1 ++ w2, pl.Valid GameConfig.standard)
    (hfold : (w1 ++ w2).foldl
      (Placement.applyStep GameConfig.standard) b = b)
    (hne : w1 ≠ [])
    (hin : ∀ p ∈ w1.foldl (Placement.applyStep GameConfig.standard) b,
      p.2 < 20) :
    (w1.foldl (Placement.applyStep GameConfig.standard) b).card ≤ 180 := by
  have hv1 : ∀ pl ∈ w1, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_append_left _ hpl)
  have hwf1 := foldl_applyStep_wf hwf hv1
  have hnf : ∀ r, ¬ Board.isFull GameConfig.standard
      (w1.foldl (Placement.applyStep GameConfig.standard) b) r := by
    rcases w1.eq_nil_or_concat with hnil | ⟨ys, y, hys⟩
    · exact absurd hnil hne
    · rw [hys]
      simp only [List.concat_eq_append, List.foldl_append, List.foldl]
      exact applyStep_clear_free _ _
  exact count_le_180 hwf1 hin hnf

/-- The base board of a cycle obeys the same ceiling. -/
theorem cycle_base_mass_le_180 {b : Board} {w : List Placement}
    (hne : w ≠ [])
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hin : ∀ p ∈ b, p.2 < 20) :
    b.card ≤ 180 := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  exact count_le_180 hwf hin
    (board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩)

/-! ### Consequences of the sharper ceiling, and repeated boards

Two uses of the hundred-and-eighty ceiling and one new reading of the
mass clock: within a loop the same BOARD can recur, but only at times
five apart, so any five consecutive moments show five different
boards. -/

/-- A non-empty play ends on a clear-free board. -/
theorem foldl_clear_free_of_ne_nil {b : Board} {pls : List Placement}
    (hne : pls ≠ []) :
    ∀ r, ¬ Board.isFull GameConfig.standard
      (pls.foldl (Placement.applyStep GameConfig.standard) b) r := by
  rcases pls.eq_nil_or_concat with hnil | ⟨ys, y, hys⟩
  · exact absurd hnil hne
  · rw [hys]
    simp only [List.concat_eq_append, List.foldl_append, List.foldl]
    exact applyStep_clear_free _ _

/-- **The prefix band, tightened**: with the clear-free ceiling the mass
delivered can outrun the mass cleared by at most a hundred and eighty,
not two hundred. -/
theorem prefix_mass_bound_180 {b : Board} {pls : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ pls, pl.Valid GameConfig.standard) {i : ℕ}
    (hi : i ≤ pls.length) (hpos : 0 < i)
    (hin : ∀ p ∈ (pls.take i).foldl
      (Placement.applyStep GameConfig.standard) b, p.2 < 20) :
    b.count + 4 * i ≤ 180 + 10 * wordClears b (pls.take i) := by
  have hvt : ∀ pl ∈ pls.take i, pl.Valid GameConfig.standard :=
    fun pl hpl => hv pl (List.mem_of_mem_take hpl)
  have hnet : pls.take i ≠ [] := by
    intro hnil
    have hlen0 : (pls.take i).length = 0 := by rw [hnil]; rfl
    rw [List.length_take] at hlen0
    omega
  have hcap : Board.count ((pls.take i).foldl
      (Placement.applyStep GameConfig.standard) b) ≤ 180 :=
    count_le_180 (foldl_applyStep_wf hwf hvt) hin
      (foldl_clear_free_of_ne_nil hnet)
  have h := foldl_count_ledger_exact (b := b) (pls := pls.take i) hwf hvt
  have hlen : (pls.take i).length = i := by
    rw [List.length_take]
    omega
  rw [hlen] at h
  omega

/-- **REPEATED BOARDS ARE FIVE APART**: the mass clock reads the time
modulo five, so if a loop revisits the same board the gap between the
visits is a multiple of five. -/
theorem legal_cycle_board_repeat_five_dvd {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard) {i j : ℕ}
    (hij : i ≤ j) (hj : j ≤ w.length)
    (heq : (wordPlay ⟨b, Bag.full⟩ w i).board
      = (wordPlay ⟨b, Bag.full⟩ w j).board) :
    5 ∣ (j - i) := by
  have hci := wordPlay_count_mod (b := b) (w := w) hwf hv
    (show i ≤ w.length by omega)
  have hcj := wordPlay_count_mod (b := b) (w := w) hwf hv hj
  rw [heq] at hci
  omega

/-- **FIVE CONSECUTIVE MOMENTS, FIVE DIFFERENT BOARDS**: inside a loop
no board can recur within four moves of itself. The orbit therefore
shows at least five distinct boards, however many times the loop
revisits them afterwards. -/
theorem legal_cycle_consecutive_boards_distinct {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard) {i j : ℕ}
    (hij : i < j) (hlt : j < i + 5) (hj : j ≤ w.length) :
    (wordPlay ⟨b, Bag.full⟩ w i).board
      ≠ (wordPlay ⟨b, Bag.full⟩ w j).board := by
  intro heq
  have h5 := legal_cycle_board_repeat_five_dvd hwf hv (by omega) hj heq
  omega

/-- The board at any moment differs from the board one move later: a
single placement always changes the board. -/
theorem legal_cycle_board_ne_succ {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard) {i : ℕ}
    (hi : i + 1 ≤ w.length) :
    (wordPlay ⟨b, Bag.full⟩ w i).board
      ≠ (wordPlay ⟨b, Bag.full⟩ w (i + 1)).board :=
  legal_cycle_consecutive_boards_distinct hwf hv (by omega) (by omega) hi

/-! ### How a column's fourteen cells arrive

Each column takes delivery of exactly fourteen cells per cycle. That
fixed budget limits how COARSELY the deliveries can come: a move
dropping four cells into one column (only a vertical I does that) can
happen at most three times there, and a column that never receives one
must be fed on at least five separate moves. -/

/-- Moves delivering at least `k` cells to column `c` each pay `k`. -/
theorem mul_bigMoves_le_colProfile (c k : ℕ) (pls : List Placement) :
    k * (pls.filter (fun pl => decide (k ≤ pl.colProfile c))).length
      ≤ wordColProfile c pls := by
  induction pls with
  | nil => simp [wordColProfile]
  | cons pl rest ih =>
    unfold wordColProfile at ih ⊢
    rw [List.map_cons, List.sum_cons, List.filter_cons]
    by_cases hk : k ≤ pl.colProfile c
    · rw [if_pos (by simpa using hk), List.length_cons, Nat.mul_add,
        Nat.mul_one]
      omega
    · rw [if_neg (by simpa using hk)]
      omega

/-- If no move delivers more than `m` cells to column `c`, the feeding
moves must be numerous enough to cover the total. -/
theorem colProfile_le_mul_feeds {c m : ℕ} {pls : List Placement}
    (hm : ∀ pl ∈ pls, pl.colProfile c ≤ m) :
    wordColProfile c pls
      ≤ m * (pls.filter (fun pl => decide (0 < pl.colProfile c))).length := by
  induction pls with
  | nil => simp [wordColProfile]
  | cons pl rest ih =>
    have hrec := ih (fun q hq => hm q (by simp [hq]))
    unfold wordColProfile at hrec ⊢
    rw [List.map_cons, List.sum_cons, List.filter_cons]
    by_cases hpos : 0 < pl.colProfile c
    · rw [if_pos (by simpa using hpos), List.length_cons, Nat.mul_add,
        Nat.mul_one]
      have := hm pl (by simp)
      omega
    · rw [if_neg (by simpa using hpos)]
      omega

/-- **NO COLUMN TAKES FOUR VERTICAL I'S**: a move dropping four cells
into a single column can happen at most three times there, since four
such moves would deliver sixteen cells against a budget of fourteen. -/
theorem legal_cycle_column_quad_le_three {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {c : ℕ} (hc : c < 10) :
    (w.filter (fun pl => decide (4 ≤ pl.colProfile c))).length ≤ 3 := by
  have h14 := legal_cycle_column_fourteen hwf hne hv hbag hfold hlen hc
  have hmul := mul_bigMoves_le_colProfile c 4 w
  rw [h14] at hmul
  omega

/-- **…and at most four triples**: five moves of three cells apiece
would already overshoot the budget. -/
theorem legal_cycle_column_triple_le_four {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {c : ℕ} (hc : c < 10) :
    (w.filter (fun pl => decide (3 ≤ pl.colProfile c))).length ≤ 4 := by
  have h14 := legal_cycle_column_fourteen hwf hne hv hbag hfold hlen hc
  have hmul := mul_bigMoves_le_colProfile c 3 w
  rw [h14] at hmul
  omega

/-- **A COLUMN WITHOUT A VERTICAL I IS FED FIVE TIMES**: if no move ever
drops four cells into column `c`, then at most three arrive per move, so
the fourteen require at least five separate deliveries. -/
theorem legal_cycle_column_no_quad_feeds_five {b : Board}
    {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) {c : ℕ} (hc : c < 10)
    (hno : ∀ pl ∈ w, pl.colProfile c ≤ 3) :
    5 ≤ (w.filter (fun pl => decide (0 < pl.colProfile c))).length := by
  have h14 := legal_cycle_column_fourteen hwf hne hv hbag hfold hlen hc
  have hle := colProfile_le_mul_feeds (c := c) (m := 3) hno
  rw [h14] at hle
  omega

/-! ### Bag-legality is a finite check

`IsBagStream` quantifies over infinitely many blocks, which makes it
awkward to establish for a candidate word. For a word of length
thirty-five it need not be: the repeated stream's block `j` is the
word's own block `j mod 5`, so checking the five blocks of the word
settles every block of the stream. Bag-legality becomes decidable. -/

/-- **THE FIVE-BLOCK TEST**: a 35-word whose five seven-blocks each
contain all seven pieces generates a bag-legal stream. An infinite
condition reduced to a finite one. -/
theorem isBagStream_of_blocks {w : List Placement} (hlen : w.length = 35)
    (hblocks : ∀ jj, jj < 5 → ∀ p : Piece, ∃ i, i < 7 ∧
      (w.getD (7 * jj + i) ⟨Piece.O, 0, 0⟩).piece = p) :
    IsBagStream (wordStream w) := by
  intro j p
  obtain ⟨i, hi, hval⟩ := hblocks (j % 5) (by omega) p
  refine ⟨i, hi, ?_⟩
  unfold wordStream
  rw [hlen, show (7 * j + i) % 35 = 7 * (j % 5) + i from by omega]
  exact hval

/-- One bag's worth of placements, all dropped at column `c`. -/
def sampleBlock (c : ℕ) : List Placement :=
  [⟨Piece.I, 1, c⟩, ⟨Piece.O, 0, c⟩, ⟨Piece.S, 0, c⟩, ⟨Piece.Z, 0, c⟩,
   ⟨Piece.T, 0, c⟩, ⟨Piece.L, 0, c⟩, ⟨Piece.J, 0, c⟩]

/-- Five bags, one per column of the left half: a concrete 35-word. -/
def sampleWord : List Placement :=
  sampleBlock 0 ++ sampleBlock 1 ++ sampleBlock 2 ++ sampleBlock 3
    ++ sampleBlock 4

theorem sampleWord_length : sampleWord.length = 35 := by decide

theorem sampleWord_valid :
    ∀ pl ∈ sampleWord, pl.Valid GameConfig.standard := by decide

/-- **A CONCRETE BAG-LEGAL WORD**: the five-block test discharges
`IsBagStream` for `sampleWord` by kernel evaluation alone. The
machinery has something to bite on. -/
theorem sampleWord_isBagStream : IsBagStream (wordStream sampleWord) := by
  apply isBagStream_of_blocks sampleWord_length
  decide

/-- Its census is the forced one: five of every piece. -/
theorem sampleWord_census (p : Piece) :
    (sampleWord.map (·.piece)).count p = 5 := by
  cases p <;> decide

/-- Every piece of the sample word is played, in particular a flat T is
present — as any legal cycle would need. -/
theorem sampleWord_has_flat_T :
    ∃ pl ∈ sampleWord, pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2) := by
  refine ⟨⟨Piece.T, 0, 0⟩, ?_, rfl, Or.inl rfl⟩
  decide

/-! ### The certificate, as one object

Everything the M2 argument consumes, packaged. Four of the six fields
are finite decidable checks — length, validity, the five-block test,
and (for a concrete board) the safety of thirty-five states. The two
that are not are the ones that carry the mathematics: that the word
CLOSES the loop, and that it never tops out. -/

/-- A complete M2 certificate: a board, a bag-legal 35-word that folds
it back to itself, and safety along the way. -/
structure LegalCycleWitness where
  /-- The board the loop is anchored at. -/
  base : Board
  /-- The thirty-five placements of one period. -/
  word : List Placement
  /-- The board is well formed. -/
  wf : Board.WF GameConfig.standard base
  /-- One period is thirty-five moves. -/
  len : word.length = 35
  /-- Every placement is in-bounds. -/
  valid : ∀ pl ∈ word, pl.Valid GameConfig.standard
  /-- Each of the five seven-blocks deals every piece. -/
  blocks : ∀ jj, jj < 5 → ∀ p : Piece, ∃ i, i < 7 ∧
    (word.getD (7 * jj + i) ⟨Piece.O, 0, 0⟩).piece = p
  /-- The word returns the board exactly. -/
  cycles : word.foldl (Placement.applyStep GameConfig.standard) base = base
  /-- No state of the period is lost. -/
  safe : ∀ i, i < 35 →
    ¬ (wordPlay ⟨base, Bag.full⟩ word i).lost GameConfig.standard

namespace LegalCycleWitness

theorem word_ne_nil (W : LegalCycleWitness) : W.word ≠ [] := by
  intro hnil
  have := W.len
  rw [hnil] at this
  simp at this

theorem isBagStream (W : LegalCycleWitness) :
    IsBagStream (wordStream W.word) :=
  isBagStream_of_blocks W.len W.blocks

/-- **A WITNESS IS A CLOSED CYCLE.** -/
theorem closedCycle (W : LegalCycleWitness) :
    ∃ C : ClosedCycle GameConfig.standard,
      (⟨W.base, Bag.full⟩ : GameState) ∈ C.states :=
  legal_safe_word_closedCycle W.wf W.word_ne_nil W.valid W.isBagStream
    W.cycles W.len W.safe

/-- **…AND THEREFORE INFINITE PLAY.** Producing one inhabitant of this
structure settles M2 for the game entered at its base board. -/
theorem survives (W : LegalCycleWitness) :
    ∃ (C : ClosedCycle GameConfig.standard),
      SurvivesForever GameConfig.standard C.policy ⟨W.base, Bag.full⟩ :=
  legal_safe_word_survives W.wf W.word_ne_nil W.valid W.isBagStream
    W.cycles W.len W.safe

/-- Every witness carries the forced profile: the nine clauses hold of
it automatically. -/
theorem profile (W : LegalCycleWitness) :
    (∀ r, ¬ Board.isFull GameConfig.standard W.base r)
    ∧ (∀ c, c < 10 → wordColProfile c W.word = 14)
    ∧ (wordClears W.base W.word = 14
        ∧ 4 ≤ wordClearMoves W.base W.word
        ∧ wordClearMoves W.base W.word ≤ 14
        ∧ 21 ≤ wordDryMoves W.base W.word)
    ∧ wordTetrises W.base W.word ≤ 5
    ∧ (W.word.map (fun pl => 4 * pl.col + shapeMoment pl)).sum = 630
    ∧ wordLift W.base W.word = wordRelease W.base W.word
    ∧ (∃ (c : Board) (pl : Placement),
        Board.fullRows GameConfig.standard (pl.place c) ≠ ∅
        ∧ ∃ p ∈ pl.place c,
            ¬ Board.isFull GameConfig.standard (pl.place c) p.2)
    ∧ (∃ (c : Board) (pl : Placement), ∀ t k : ℕ, Even k →
        Board.fullRows GameConfig.standard (pl.place c)
          ≠ Finset.Ico t (t + k))
    ∧ (∃ pl ∈ W.word, pl.piece = Piece.T ∧ (pl.rot = 0 ∨ pl.rot = 2)) :=
  legal_cycle_profile W.wf W.word_ne_nil W.valid W.isBagStream W.cycles
    W.len

/-- A witness lays an odd number of its T's flat and stands an even
number upright. -/
theorem T_split (W : LegalCycleWitness) :
    (wordFlatTCount W.word = 1 ∨ wordFlatTCount W.word = 3
      ∨ wordFlatTCount W.word = 5)
    ∧ (wordUprightTCount W.word = 0 ∨ wordUprightTCount W.word = 2
      ∨ wordUprightTCount W.word = 4) :=
  ⟨legal_cycle_flatT_one_three_or_five W.wf W.word_ne_nil W.valid
      W.isBagStream W.cycles W.len,
    legal_cycle_uprightT_even W.wf W.word_ne_nil W.valid W.isBagStream
      W.cycles W.len⟩

end LegalCycleWitness

/-! ### Only three tetrises fit

The bag argument caps a cycle's tetrises at five, one per bag. The clear
budget caps them lower: fourteen rows at four rows apiece leaves room
for only three. And three tetrises cannot be the whole story, since
fourteen is not a multiple of four — some clearing move must take fewer
than four rows. -/

/-- Each tetris move takes four rows off the budget. -/
theorem four_mul_tetrises_le_clears (b : Board) (pls : List Placement) :
    4 * wordTetrises b pls ≤ wordClears b pls := by
  induction pls generalizing b with
  | nil => simp
  | cons pl rest ih =>
    rw [wordTetrises_cons, wordClears_cons, Nat.mul_add]
    have hrec := ih (Placement.applyStep GameConfig.standard b pl)
    by_cases h4 : 4 ≤ (Board.fullRows GameConfig.standard
        (pl.place b)).card
    · rw [if_pos h4]
      omega
    · rw [if_neg h4]
      omega

/-- **AT MOST THREE TETRISES**: fourteen rows at four apiece leave room
for three, not the five the bag count allows. The clear budget is the
binding constraint, not the piece supply. -/
theorem legal_cycle_tetris_le_three {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordTetrises b w ≤ 3 := by
  have h14 := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen
  have h4 := four_mul_tetrises_le_clears b w
  omega

/-- **A LEGAL CYCLE ALWAYS CLEARS SMALL SOMEWHERE**: its clearing moves
cannot all be tetrises, because fourteen is not a multiple of four.
Every M2 witness contains a single, a double or a triple. -/
theorem legal_cycle_tetrises_lt_clearMoves {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    wordTetrises b w < wordClearMoves b w := by
  have hpos : 0 < w.length := List.length_pos_iff.mpr hne
  have h14 := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen
  have h4 := four_mul_tetrises_le_clears b w
  have hnfb : ∀ r, ¬ Board.isFull GameConfig.standard b r :=
    board_on_cycle_clear_free ⟨w, rfl, hpos, hv, hfold⟩
  have hmix := word_clear_mix_bound (w := w) hnfb
  have hle : wordTetrises b w ≤ wordClearMoves b w := by
    have hcap := wordClears_le_four_mul_moves (w := w) hnfb
    omega
  omega

/-- The non-tetris clearing moves of a legal cycle take at least two
rows between them. -/
theorem legal_cycle_small_clears_ge_two {b : Board} {w : List Placement}
    (hwf : Board.WF GameConfig.standard b) (hne : w ≠ [])
    (hv : ∀ pl ∈ w, pl.Valid GameConfig.standard)
    (hbag : IsBagStream (wordStream w))
    (hfold : w.foldl (Placement.applyStep GameConfig.standard) b = b)
    (hlen : w.length = 35) :
    2 ≤ wordClears b w - 4 * wordTetrises b w := by
  have h14 := legal_cycle_word_clears_fourteen hwf hne hv hbag hfold hlen
  have h3 := legal_cycle_tetris_le_three hwf hne hv hbag hfold hlen
  omega

/-! ### A clear happens where the piece landed

On a clear-free board every completed row owes its last cell to the
piece just dropped, and that piece occupies only four consecutive rows
starting at its landing height. So the rows a move clears all lie in a
four-row window at the landing height — never below it, never far above
it. High clears therefore cost height, and the potential balance has to
pay for them. -/

/-- **Cleared rows sit in the piece's own four-row window**: on a
clear-free board a move can only complete rows it has just touched. -/
theorem fullRows_subset_span {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    Board.fullRows GameConfig.standard (pl.place b)
      ⊆ Finset.Icc (pl.dropOffset b) (pl.dropOffset b + 3) := by
  classical
  intro r hr
  simp only [Board.fullRows, Finset.mem_filter] at hr
  obtain ⟨c, hc, hcb⟩ : ∃ c ∈ Finset.range GameConfig.standard.cols,
      ((c, r) : Coord) ∉ b := by
    by_contra hcon
    push Not at hcon
    exact hnf r hcon
  have hcplace : ((c, r) : Coord) ∈ pl.place b := hr.2 c hc
  have hcdrop : ((c, r) : Coord) ∈ pl.dropped b := by
    rw [Placement.place_eq_union_dropped, Finset.mem_union] at hcplace
    rcases hcplace with h | h
    · exact absurd h hcb
    · exact h
  rw [Placement.dropped_eq_image, Finset.mem_image] at hcdrop
  obtain ⟨cell, hcell, hEq⟩ := hcdrop
  have hrow : pl.dropOffset b + cell.2 = r := congrArg Prod.snd hEq
  have hlt := Piece.shapeUp_row_lt_four pl.piece pl.rot cell hcell
  rw [Finset.mem_Icc]
  omega

/-- **No clear happens below the landing height**: the rows a move
completes are all at or above the height its piece came to rest at. -/
theorem cleared_row_ge_dropOffset {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) {r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    pl.dropOffset b ≤ r :=
  (Finset.mem_Icc.mp (fullRows_subset_span hnf hr)).1

/-- **THE HEIGHT OF A HARVEST**: the total height of the rows a move
clears is at least its landing height times the number of rows taken.
Clearing high is expensive, and the potential balance must pay for
it. -/
theorem clearedRowSum_ge_mul {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    (Board.fullRows GameConfig.standard (pl.place b)).card
        * pl.dropOffset b
      ≤ clearedRowSum (pl.place b) := by
  classical
  unfold clearedRowSum
  calc (Board.fullRows GameConfig.standard (pl.place b)).card
        * pl.dropOffset b
      = ∑ _t ∈ Board.fullRows GameConfig.standard (pl.place b),
          pl.dropOffset b := by
        rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ t ∈ Board.fullRows GameConfig.standard (pl.place b), t :=
        Finset.sum_le_sum (fun t ht => cleared_row_ge_dropOffset hnf ht)

/-- **…and at most three rows above it**: a harvest's total height is
also bounded by the landing height plus three, per row taken. -/
theorem clearedRowSum_le_mul {b : Board} {pl : Placement}
    (hnf : ∀ r, ¬ Board.isFull GameConfig.standard b r) :
    clearedRowSum (pl.place b)
      ≤ (Board.fullRows GameConfig.standard (pl.place b)).card
          * (pl.dropOffset b + 3) := by
  classical
  unfold clearedRowSum
  calc ∑ t ∈ Board.fullRows GameConfig.standard (pl.place b), t
      ≤ ∑ _t ∈ Board.fullRows GameConfig.standard (pl.place b),
          (pl.dropOffset b + 3) :=
        Finset.sum_le_sum (fun t ht =>
          (Finset.mem_Icc.mp (fullRows_subset_span hnf ht)).2)
    _ = (Board.fullRows GameConfig.standard (pl.place b)).card
          * (pl.dropOffset b + 3) := by
        rw [Finset.sum_const, smul_eq_mul]

/-! ### What a clear must already own

A piece brings four cells. A completed row needs ten. So a row cleared
by a move was already holding at least six, and a `k`-clear was holding
at least `10k − 4` across its rows — thirty-six for a tetris. Harvests
are collected, not created. -/

/-- A full row of a well-formed board holds exactly ten cells. -/
theorem rowCount_of_mem_fullRows {B : Board}
    (hwf : Board.WF GameConfig.standard B) {t : ℕ}
    (ht : t ∈ Board.fullRows GameConfig.standard B) :
    Board.rowCount B t = 10 := by
  classical
  have hfib : (B.filter
      (fun p => Board.isFull GameConfig.standard B p.2)).filter
      (fun p => p.2 = t)
      = (Finset.range 10).image (fun c => ((c, t) : Coord)) :=
    cleared_fiber_eq hwf ht
  have heq : B.filter (fun p => p.2 = t)
      = (Finset.range 10).image (fun c => ((c, t) : Coord)) := by
    rw [← hfib]
    ext p
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hpb, hpt⟩
      exact ⟨⟨hpb, by rw [hpt]; exact Board.isFull_of_mem_fullRows ht⟩,
        hpt⟩
    · rintro ⟨⟨hpb, -⟩, hpt⟩
      exact ⟨hpb, hpt⟩
  unfold Board.rowCount
  rw [heq, Finset.card_image_of_injective _ (fun x y hxy => by
    simpa using congrArg Prod.fst hxy), Finset.card_range]

/-- Placing a piece adds its own cells to each row. -/
theorem rowCount_place (b : Board) (pl : Placement) (r : ℕ) :
    Board.rowCount (pl.place b) r
      = Board.rowCount b r + Board.rowCount (pl.dropped b) r := by
  classical
  unfold Board.rowCount
  rw [Placement.place_eq_union_dropped, Finset.filter_union,
    Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (pl.dropped_disjoint b).symm)]

/-- **A CLEARED ROW WAS ALREADY SIX-TENTHS FULL**: the piece brings at
most four cells, so any row it completes held at least six before. -/
theorem cleared_row_prior_six {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) {r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard (pl.place b)) :
    6 ≤ Board.rowCount b r := by
  classical
  have hten := rowCount_of_mem_fullRows (Placement.place_wf hwf hv) hr
  have hsplit := rowCount_place b pl r
  have hle : Board.rowCount (pl.dropped b) r ≤ 4 := by
    unfold Board.rowCount
    calc ((pl.dropped b).filter (fun p => p.2 = r)).card
        ≤ (pl.dropped b).card := Finset.card_filter_le _ _
      _ = 4 := Placement.card_dropped b pl
  omega

/-- **A `k`-CLEAR HAD `10k − 4` BANKED**: the four cells of one piece
cannot supply more than four of the `10k` needed. -/
theorem cleared_rows_prior_inventory_board {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard) :
    10 * (Board.fullRows GameConfig.standard (pl.place b)).card
      ≤ (∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
          Board.rowCount b r) + 4 := by
  classical
  have hten : ∀ t ∈ Board.fullRows GameConfig.standard (pl.place b),
      Board.rowCount (pl.place b) t = 10 :=
    fun t ht => rowCount_of_mem_fullRows (Placement.place_wf hwf hv) ht
  have hsum : ∑ t ∈ Board.fullRows GameConfig.standard (pl.place b),
      Board.rowCount (pl.place b) t
      = 10 * (Board.fullRows GameConfig.standard (pl.place b)).card := by
    rw [Finset.sum_congr rfl hten, Finset.sum_const, smul_eq_mul,
      mul_comm]
  have hsplit : ∀ t ∈ Board.fullRows GameConfig.standard (pl.place b),
      Board.rowCount (pl.place b) t
        = Board.rowCount b t + Board.rowCount (pl.dropped b) t :=
    fun t _ => rowCount_place b pl t
  rw [Finset.sum_congr rfl hsplit, Finset.sum_add_distrib] at hsum
  have hdrop : ∑ t ∈ Board.fullRows GameConfig.standard (pl.place b),
      Board.rowCount (pl.dropped b) t ≤ 4 := by
    unfold Board.rowCount
    have hmaps : ∀ p ∈ (pl.dropped b).filter
        (fun p => p.2 ∈ Board.fullRows GameConfig.standard (pl.place b)),
        p.2 ∈ Board.fullRows GameConfig.standard (pl.place b) :=
      fun p hp => (Finset.mem_filter.mp hp).2
    have hfib := Finset.sum_fiberwise_of_maps_to hmaps (fun _ => 1)
    have hcongr : ∀ t ∈ Board.fullRows GameConfig.standard (pl.place b),
        ((pl.dropped b).filter (fun p => p.2 = t)).card
          = ∑ _p ∈ ((pl.dropped b).filter
              (fun p => p.2 ∈ Board.fullRows GameConfig.standard
                (pl.place b))).filter (fun p => p.2 = t), 1 := by
      intro t ht
      rw [Finset.sum_const, smul_eq_mul, mul_one]
      congr 1
      ext p
      simp only [Finset.mem_filter]
      constructor
      · rintro ⟨hpd, hpt⟩
        exact ⟨⟨hpd, by rw [hpt]; exact ht⟩, hpt⟩
      · rintro ⟨⟨hpd, -⟩, hpt⟩
        exact ⟨hpd, hpt⟩
    rw [Finset.sum_congr rfl hcongr, hfib, Finset.sum_const, smul_eq_mul,
      mul_one]
    calc ((pl.dropped b).filter
          (fun p => p.2 ∈ Board.fullRows GameConfig.standard
            (pl.place b))).card
        ≤ (pl.dropped b).card := Finset.card_filter_le _ _
      _ = 4 := Placement.card_dropped b pl
  omega

/-- **A TETRIS NEEDS THIRTY-SIX BANKED**: four rows at ten cells apiece,
less the four the piece itself supplies. -/
theorem tetris_prior_thirtysix {b : Board} {pl : Placement}
    (hwf : Board.WF GameConfig.standard b)
    (hv : pl.Valid GameConfig.standard)
    (h4 : (Board.fullRows GameConfig.standard (pl.place b)).card = 4) :
    36 ≤ ∑ r ∈ Board.fullRows GameConfig.standard (pl.place b),
      Board.rowCount b r := by
  have h := cleared_rows_prior_inventory_board hwf hv
  rw [h4] at h
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
