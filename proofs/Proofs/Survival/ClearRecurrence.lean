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

end ClearRate
end Tetris
