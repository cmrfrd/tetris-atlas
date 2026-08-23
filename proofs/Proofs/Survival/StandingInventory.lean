import Mathlib
import Proofs.Survival.ClearRecurrence

/-!
# Standing inventory: the ideal solver cannot play an empty board

Everything so far bounded the solver's occupancy from *above*. This file proves
the surprising converse: survival forces a **floor on the time-averaged board
occupancy**. An immortal solver is required to carry standing inventory — it
can never be an "empty-board" player, not even on average.

The mechanism is the interaction of two prior facts:

* clearing `k` rows requires `10k − 4` cells already banked (`clear_step_le`),
  so every clearing moment sits on at least **6 cells** of inventory
  (`six_le_count_of_clearing`);
* clearing moments are *frequent* — at least ten percent of all placements
  (`le_clearingSteps`).

Summing the banked-mass requirement over the clearing moments and balancing
against the clearing duty gives the floor (`standing_inventory_floor`):

  `∑_{t<n} count(t)  ≥  2.4·n − 200`

— the time-averaged occupancy of any surviving play is at least **2.4 cells**
(0.24 rows), forever. A singles-only sawtooth achieves average 4, so the floor
is within a factor of two of achievable.

A sharper pointwise companion (`card_empty_times_le`): the board can be empty
only at placement counts divisible by 5 (`five_dvd_of_count_eq_zero`), so an
immortal solver's board is **occupied at least 80% of the time** — even a
perfect-clear-loop strategy spends at least four fifths of its life holding
cells.

Solver-design reading: "keep the board as empty as possible" is not merely
conservative, it is *impossible* beyond a hard limit. The ideal solver's
occupancy hovers in the band `[2.4, 200]` with mandatory returns toward the
bottom (recurrence) and mandatory stock before every clear. Evaluation
functions that monotonically reward emptiness fight a provable floor.
-/

namespace Tetris
namespace ClearRate

/-! ## The cumulative occupancy -/

/-- Sum of the board occupancy over the first `n` checkpoints of a trace. -/
def sumCount (π : Policy GameConfig.standard) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      sumCount π n + (trace GameConfig.standard π GameState.init n).board.count

@[simp] theorem sumCount_zero (π : Policy GameConfig.standard) :
    sumCount π 0 = 0 := rfl

theorem sumCount_succ (π : Policy GameConfig.standard) (n : ℕ) :
    sumCount π (n + 1)
      = sumCount π n
        + (trace GameConfig.standard π GameState.init n).board.count := rfl

/-- **Every clearing moment sits on six banked cells.** A drop that completes a
row needs `10·1 − 4 = 6` cells already on the board. -/
theorem six_le_count_of_clearing {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hc : 0 < (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init n)).place
        (trace GameConfig.standard π GameState.init n).board)).card) :
    6 ≤ (trace GameConfig.standard π GameState.init n).board.count := by
  have hstep := clear_step_le hv n
  rw [cleared_succ] at hstep
  omega

/-- **The banked-mass ledger.** Summing the per-clear requirement over a trace:
the cleared rows are financed by the standing inventory, up to 4 cells of
same-drop credit per clearing moment. -/
theorem ten_cleared_le_sumCount {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    10 * cleared GameConfig.standard π GameState.init n
      ≤ sumCount π n
        + 4 * clearingSteps GameConfig.standard π GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := clear_step_le hv k
    rw [cleared_succ] at hstep
    rw [cleared_succ, clearingSteps_succ, sumCount_succ]
    split_ifs with hc
    · omega
    · omega

/-- **The standing-inventory floor.** Any live legal play satisfies
`12·n ≤ 5·∑ count + 1000`, i.e. the time-averaged occupancy is at least
`2.4 − 200/n` cells. Survival does not permit an empty-board style: the clear
duty is ten percent of placements, each clearing moment demands six banked
cells, and the arithmetic closes only if the solver *carries stock*. -/
theorem standing_inventory_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init n).lost
      GameConfig.standard) :
    12 * n ≤ 5 * sumCount π n + 1000 := by
  have h1 := ten_cleared_le_sumCount hv n
  have h2 := le_cols_mul_cleared hv hlive
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h2
  have h3 := clearingSteps_le hv n
  omega

/-! ## Occupied at least 80% of the time -/

/-- **The board is empty at most a fifth of the time.** Emptiness forces
`5 ∣ t` (`five_dvd_of_count_eq_zero`), and multiples of five have density
one fifth: `5·#{t < n | empty} ≤ n + 4`. Equivalently, an immortal solver's
board is occupied on at least four fifths of all checkpoints — even a
perfect-clear loop holds cells 80% of its life. -/
theorem card_empty_times_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    5 * ((Finset.range n).filter
        (fun t => (trace GameConfig.standard π GameState.init t).board.count = 0)).card
      ≤ n + 4 := by
  classical
  set S := (Finset.range n).filter
    (fun t => (trace GameConfig.standard π GameState.init t).board.count = 0) with hS
  have hmod : ∀ t ∈ S, t % 5 = 0 := by
    intro t ht
    obtain ⟨-, hzero⟩ := Finset.mem_filter.mp ht
    obtain ⟨q, rfl⟩ := five_dvd_of_count_eq_zero hv hzero
    simp [Nat.mul_mod_right]
  have hlt : ∀ t ∈ S, t < n := fun t ht =>
    Finset.mem_range.mp (Finset.mem_filter.mp ht).1
  have hcard : S.card ≤ (Finset.range ((n + 4) / 5)).card := by
    refine Finset.card_le_card_of_injOn (· / 5) ?_ ?_
    · intro t ht
      have h1 := hmod t ht
      have h2 := hlt t ht
      simp only [Finset.coe_range, Set.mem_Iio]
      omega
    · intro t₁ ht₁ t₂ ht₂ hEq
      have h1 := hmod t₁ ht₁
      have h2 := hmod t₂ ht₂
      dsimp only at hEq
      omega
  rw [Finset.card_range] at hcard
  omega

end ClearRate
end Tetris
