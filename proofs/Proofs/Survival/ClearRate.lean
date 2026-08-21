import Mathlib
import Proofs.Model.Game
import Proofs.Invariants.BagGrowth
import Proofs.Survival.Survival

/-!
# The clearing-rate law: 2.8 rows per bag, exactly

Every 7-bag delivers `7 × 4 = 28` cells; every cleared row removes `cols = 10`
of them. So the *only* rate at which a surviving player can clear is
`28 / 10 = 2.8` rows per bag — and mass conservation pins it from **both**
sides:

* **Floor.** A live board holds at most `cols · rows = 200` cells. Whatever is
  not cleared is still sitting on the board, so after `m` bags the cumulative
  clears satisfy `10 · cleared + 200 ≥ 28 m`, i.e. `cleared ≥ 2.8 m − 20`
  (`le_cols_mul_cleared`, `bags_le_cleared`). Fall further than one boardful
  behind the treadmill and you are, provably, already dead
  (`lost_of_clear_deficit`).
* **Ceiling.** A row must be filled before it can be cleared, and from the
  empty board the only source of cells is the pieces. So `10 · cleared ≤ 28 m`,
  i.e. `cleared ≤ 2.8 m` (`cols_mul_cleared_le`, `bags_cleared_le`), with
  equality exactly when the board is empty — a perfect clear
  (`cleared_eq_iff_board_empty`).

Together: `2.8 m − 20 ≤ cleared ≤ 2.8 m`. The per-bag rate is trapped in a
window of width `20/m` around `2.8`, so **any** policy that survives forever
clears at asymptotic rate exactly `2.8` rows per bag
(`survival_forces_clear_rate`), and any policy whose rate is eventually
`≤ 2.8 − ε` tops out (`not_survivesForever_of_rate_lt`).

The often-stated intuition "you need to clear *more* than 2.8 rows per bag to
play forever" is half right and half wrong, and this file says exactly which
half: `2.8` is a hard **necessary** rate (you may never durably clear less),
but it is also a hard **ceiling** (you may never durably clear more). The
long-run clearing rate of an immortal Tetris player is not `> 2.8`; it is `2.8`
on the nose. The whole gap between the floor and the ceiling is the mass
currently resting on the board — at most 200 cells, i.e. at most 20 rows of
credit, ever (`init_ledger`).

Because the bounds are pathwise and hold for every trajectory, they survive
averaging: over any finite ensemble of runs (`average_clears_bounds`) or any
probability distribution over randomized strategies and piece orders
(`expected_clears_bounds`), the *expected* clears in `m` bags lie in
`[2.8 m − 20, 2.8 m]`.

**Scope.** The trace results are stated for the cooperative `Policy` setting
(`Proofs/Survival/Survival`), which is the *weakest* player discipline in this
library: a policy deals itself its own pieces. A necessity theorem proved
against the weakest discipline binds every stronger one, so the rate law
applies verbatim to adversarial solvers. `play_bag_sandwich` makes that
explicit — it quantifies over an arbitrary list of placements, so it covers
every realizable trajectory of every solver against every piece sequence.

Downstream of `Proofs/Invariants/BagGrowth`, which supplies the per-move mass
balance (`count_applyStep_add`) and the capacity bound (`count_le_capacity`).
-/

namespace Tetris
namespace ClearRate

open Filter Topology MeasureTheory

/-! ## Cumulative clears along a policy trace -/

/-- Total rows cleared over the first `n` moves of the trace of `π` from `g0`.
Each move contributes the number of rows that are full after its hard drop and
before the clear. -/
def cleared (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      cleared cfg π g0 n
        + (Board.fullRows cfg
            ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card

@[simp] theorem cleared_zero (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) :
    cleared cfg π g0 0 = 0 := rfl

theorem cleared_succ (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (n : ℕ) :
    cleared cfg π g0 (n + 1)
      = cleared cfg π g0 n
        + (Board.fullRows cfg
            ((π (trace cfg π g0 n)).place (trace cfg π g0 n).board)).card := rfl

/-- Cumulative clears never decrease. -/
theorem cleared_mono (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) :
    Monotone (cleared cfg π g0) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [cleared_succ]
  exact Nat.le_add_right _ _

/-! ## The mass ledger along a trace -/

/-- A legal policy keeps every trace board well-formed. -/
theorem trace_board_wf {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hv : ∀ g, (π g).Valid cfg) (hwf : Board.WF cfg g0.board) (n : ℕ) :
    Board.WF cfg (trace cfg π g0 n).board := by
  induction n with
  | zero => simpa using hwf
  | succ k ih =>
    rw [trace_succ, GameState.step_board]
    exact Placement.applyStep_wf ih (hv _)

/-- **Mass conservation along a trace.** After `n` moves, the cells still on the
board plus the `cols` cells removed by each cleared row account exactly for the
`4` cells delivered by each placement. Nothing is created, nothing vanishes: the
clear ledger and the board mass are two halves of one conserved quantity. -/
theorem mass_ledger {cfg : GameConfig} {π : Policy cfg} {g0 : GameState}
    (hv : ∀ g, (π g).Valid cfg) (hwf : Board.WF cfg g0.board) (n : ℕ) :
    (trace cfg π g0 n).board.count + cfg.cols * cleared cfg π g0 n
      = g0.board.count + 4 * n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := BagGrowth.count_applyStep_add (trace_board_wf hv hwf k)
      (hv (trace cfg π g0 k))
    rw [trace_succ, GameState.step_board, cleared_succ, Nat.mul_add]
    omega

/-- **The ledger from the empty board**: board mass plus cleared mass equals
delivered mass. The board's current cell count *is* the clearing deficit. -/
theorem init_ledger {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    (trace cfg π GameState.init n).board.count
      + cfg.cols * cleared cfg π GameState.init n = 4 * n := by
  have h := mass_ledger hv (GameState.init_board_wf cfg) n
  rw [GameState.init_board_count] at h
  omega

/-! ## The two-sided rate bound -/

/-- **The ceiling.** From the empty board a row can only be cleared after it has
been filled, and the pieces are the only source of cells: `cols · cleared ≤ 4n`.
At standard width that is `2.8` rows per bag — an upper bound no strategy, however
good, can beat. -/
theorem cols_mul_cleared_le {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    cfg.cols * cleared cfg π GameState.init n ≤ 4 * n := by
  have h := init_ledger hv n
  omega

/-- **The floor.** A board that has not topped out holds at most `cols · rows`
cells, so the clears can never fall more than one boardful behind the delivered
mass: `4n ≤ cols · cleared + cols · rows`. -/
theorem le_cols_mul_cleared {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {n : ℕ}
    (hlive : ¬ (trace cfg π GameState.init n).lost cfg) :
    4 * n ≤ cfg.cols * cleared cfg π GameState.init n + cfg.cols * cfg.rows := by
  have h := init_ledger hv n
  have hcap := BagGrowth.count_le_capacity
    (trace_board_wf hv (GameState.init_board_wf cfg) n)
    ((GameState.not_lost_iff_forall_row_lt cfg _).mp hlive)
  omega

/-- **The ceiling is attained exactly at a perfect clear.** Clearing the full
`4n / cols` is possible only with an empty board: any surplus is mass still
sitting on the stack. -/
theorem cleared_eq_iff_board_empty {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    cfg.cols * cleared cfg π GameState.init n = 4 * n
      ↔ (trace cfg π GameState.init n).board = ∅ := by
  have h := init_ledger hv n
  constructor
  · intro he
    have hz : (trace cfg π GameState.init n).board.count = 0 := by omega
    exact Finset.card_eq_zero.mp hz
  · intro hb
    have hz : (trace cfg π GameState.init n).board.count = 0 := by
      rw [hb]; simp [Board.count]
    omega

/-! ## Per-bag form at standard width -/

/-- Per bag, the ceiling: `m` bags of legal play from the empty board clear at
most `2.8 m` rows (`10 · cleared ≤ 28 m`). -/
theorem bags_cleared_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    10 * cleared GameConfig.standard π GameState.init (7 * m) ≤ 28 * m := by
  have h := cols_mul_cleared_le hv (7 * m)
  rw [GameConfig.standard_cols] at h
  omega

/-- Per bag, the floor: `m` bags of legal play that has not topped out have
cleared at least `2.8 m − 20` rows (`28 m ≤ 10 · cleared + 200`). -/
theorem bags_le_cleared {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    28 * m ≤ 10 * cleared GameConfig.standard π GameState.init (7 * m) + 200 := by
  have h := le_cols_mul_cleared hv hlive
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  omega

/-- **The bag sandwich.** `m` bags of live legal play clear between `2.8 m − 20`
and `2.8 m` rows. The whole width of the window is the board's capacity. -/
theorem bags_sandwich {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    28 * m ≤ 10 * cleared GameConfig.standard π GameState.init (7 * m) + 200
      ∧ 10 * cleared GameConfig.standard π GameState.init (7 * m) ≤ 28 * m :=
  ⟨bags_le_cleared hv hlive, bags_cleared_le hv m⟩

/-! ## Necessity: falling behind the treadmill is fatal -/

/-- **The clear-deficit death law.** If after `m` bags the cumulative clears have
fallen more than one boardful behind the `2.8`-rows-per-bag treadmill, the state
is *already* lost — no board can absorb the surplus. This is the finite,
certificate-shaped form of "you must clear 2.8 rows per bag to stay alive". -/
theorem lost_of_clear_deficit {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hdef : 10 * cleared GameConfig.standard π GameState.init (7 * m) + 200 < 28 * m) :
    (trace GameConfig.standard π GameState.init (7 * m)).lost GameConfig.standard := by
  by_contra hlive
  have h := bags_le_cleared hv hlive
  omega

/-- A single bag count at which the clears fall a boardful behind refutes
infinite play outright. -/
theorem not_survivesForever_of_clear_deficit {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hdef : 10 * cleared GameConfig.standard π GameState.init (7 * m) + 200 < 28 * m) :
    ¬ SurvivesForever GameConfig.standard π GameState.init :=
  fun hs => hs (7 * m) (lost_of_clear_deficit hv hdef)

/-! ## The asymptotic rate -/

/-- Rows cleared per bag over the first `m` bags. -/
noncomputable def bagRate (π : Policy GameConfig.standard) (m : ℕ) : ℝ :=
  (cleared GameConfig.standard π GameState.init (7 * m) : ℝ) / m

/-- The per-bag clearing rate never exceeds `2.8`, at any horizon. -/
theorem bagRate_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    bagRate π m ≤ 2.8 := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · norm_num [bagRate]
  · have hm' : (0 : ℝ) < m := by exact_mod_cast hm
    have hcast : (10 : ℝ) * (cleared GameConfig.standard π GameState.init (7 * m) : ℝ)
        ≤ 28 * m := by exact_mod_cast bags_cleared_le hv m
    rw [bagRate, div_le_iff₀ hm']
    linarith

/-- While alive, the per-bag clearing rate is at least `2.8 − 20/m`. -/
theorem le_bagRate {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 0 < m)
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    2.8 - 20 / (m : ℝ) ≤ bagRate π m := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hne : (m : ℝ) ≠ 0 := ne_of_gt hm'
  have hcast : (28 : ℝ) * m
      ≤ 10 * (cleared GameConfig.standard π GameState.init (7 * m) : ℝ) + 200 := by
    exact_mod_cast bags_le_cleared hv hlive
  rw [bagRate, le_div_iff₀ hm']
  have hkey : (2.8 - 20 / (m : ℝ)) * m = 2.8 * m - 20 := by
    field_simp
  rw [hkey]
  linarith

/-- **The rate window.** While alive, the per-bag clearing rate sits within
`20/m` of `2.8` — the board's capacity, amortised. -/
theorem abs_bagRate_sub_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 0 < m)
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    |bagRate π m - 2.8| ≤ 20 / (m : ℝ) := by
  have hlow := le_bagRate hv hm hlive
  have hhigh := bagRate_le hv m
  have hpos : (0 : ℝ) ≤ 20 / (m : ℝ) := by positivity
  rw [abs_le]
  exact ⟨by linarith, by linarith⟩

/-- **Survival forces the rate.** Any legal policy that plays forever clears at
asymptotic rate exactly `2.8` rows per bag. Not "at least": the ceiling closes
the window from above, so the limit exists and equals `28/10`. -/
theorem survival_forces_clear_rate {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) :
    Tendsto (bagRate π) atTop (𝓝 2.8) := by
  have hlow : Tendsto (fun m : ℕ => (2.8 : ℝ) - 20 / m) atTop (𝓝 2.8) := by
    have h0 := tendsto_const_div_atTop_nhds_zero_nat (20 : ℝ)
    simpa using (tendsto_const_nhds (x := (2.8 : ℝ)) (f := atTop)).sub h0
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hlow tendsto_const_nhds ?_ ?_
  · filter_upwards [eventually_gt_atTop 0] with m hm
    exact le_bagRate hv hm (hsurv (7 * m))
  · exact Eventually.of_forall fun m => bagRate_le hv m

/-- **Sub-`2.8` clearing is fatal.** If the per-bag clearing rate is eventually
bounded by `2.8 − ε` for some fixed `ε > 0`, the policy cannot play forever.
This is the precise sense in which `2.8` rows per bag is *required*: you may dip
below it for a while — the board is a 200-cell buffer — but not durably. -/
theorem not_survivesForever_of_rate_lt {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {ε : ℝ} (hε : 0 < ε)
    (hslow : ∀ᶠ m in atTop, bagRate π m ≤ 2.8 - ε) :
    ¬ SurvivesForever GameConfig.standard π GameState.init := by
  intro hsurv
  have hlim := le_of_tendsto (survival_forces_clear_rate hv hsurv) hslow
  linarith

/-- **Super-`2.8` clearing is impossible.** Symmetrically, no legal policy can
hold its per-bag rate at `2.8 + ε` for any `ε > 0` — not even for one bag. The
popular reading "infinite play needs *more* than 2.8 rows per bag" is therefore
strictly unsatisfiable; the correct statement is the equality. -/
theorem not_rate_ge_add {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {ε : ℝ} (hε : 0 < ε) (m : ℕ) :
    ¬ (2.8 + ε ≤ bagRate π m) := by
  intro hge
  have := bagRate_le hv m
  linarith

/-! ## Sequence-free form: every trajectory, adversarial included -/

/-- **The rate law without a policy.** Any list of `7m` legal placements played
from the empty board that has not topped out obeys the same sandwich. Quantifying
over placement lists covers every trajectory of every solver against every piece
sequence, so the `2.8`-rows-per-bag law is a property of the *dynamics*, not of a
particular player discipline. -/
theorem play_bag_sandwich {L : List Placement} {m : ℕ}
    (hL : ∀ pl ∈ L, pl.Valid GameConfig.standard) (hlen : L.length = 7 * m)
    (hfield : ∀ p ∈ BagGrowth.playFrom GameConfig.standard (∅ : Board) L,
      p.2 < GameConfig.standard.rows) :
    28 * m ≤ 10 * BagGrowth.clearedRows GameConfig.standard ∅ L + 200
      ∧ 10 * BagGrowth.clearedRows GameConfig.standard ∅ L ≤ 28 * m := by
  have hlow := BagGrowth.clearing_rate hL hfield
  have hbal := BagGrowth.count_playFrom_add_cleared
    (Board.empty_wf GameConfig.standard) hL
  have hz : (∅ : Board).count = 0 := by simp [Board.count]
  rw [GameConfig.standard_cols, GameConfig.standard_rows, hlen] at hlow
  rw [GameConfig.standard_cols, hlen, hz] at hbal
  omega

/-! ## Averages and expectations

The bounds above are pathwise, so they pass through any averaging operator
unchanged: no probabilistic structure on the piece sequence is needed, and the
conclusion holds for every distribution rather than for a typical one. -/

/-- **The rate law on a finite ensemble of runs.** Average the clears of any
finite family of live legal policies and the mean still lands in
`[2.8 m − 20, 2.8 m]`. -/
theorem average_clears_bounds {ι : Type*} (S : Finset ι) (hS : S.Nonempty)
    (strat : ι → Policy GameConfig.standard) (m : ℕ)
    (hv : ∀ i g, (strat i g).Valid GameConfig.standard)
    (hlive : ∀ i, ¬ (trace GameConfig.standard (strat i) GameState.init (7 * m)).lost
      GameConfig.standard) :
    2.8 * m - 20
        ≤ (∑ i ∈ S, (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ))
            / S.card
      ∧ (∑ i ∈ S, (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ))
            / S.card ≤ 2.8 * m := by
  have hcard : (0 : ℝ) < S.card := by
    exact_mod_cast Finset.card_pos.mpr hS
  have hlowptwise : ∀ i ∈ S,
      (2.8 * m - 20 : ℝ)
        ≤ (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ) := by
    intro i _
    have : (28 : ℝ) * m
        ≤ 10 * (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ) + 200 := by
      exact_mod_cast bags_le_cleared (hv i) (hlive i)
    linarith
  have hhighptwise : ∀ i ∈ S,
      (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ) ≤ 2.8 * m := by
    intro i _
    have : (10 : ℝ) * (cleared GameConfig.standard (strat i) GameState.init (7 * m) : ℝ)
        ≤ 28 * m := by exact_mod_cast bags_cleared_le (hv i) m
    linarith
  have hsumlow := Finset.sum_le_sum hlowptwise
  have hsumhigh := Finset.sum_le_sum hhighptwise
  rw [Finset.sum_const, nsmul_eq_mul] at hsumlow hsumhigh
  constructor
  · rw [le_div_iff₀ hcard]
    linarith
  · rw [div_le_iff₀ hcard]
    linarith

/-- A constant below an integrable function is below its integral, on a
probability space. -/
theorem const_le_integral {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {f : Ω → ℝ} (hf : Integrable f μ) {c : ℝ}
    (h : ∀ ω, c ≤ f ω) : c ≤ ∫ ω, f ω ∂μ := by
  have hmono := integral_mono (integrable_const c) hf h
  simpa using hmono

/-- A constant above an integrable function is above its integral, on a
probability space. -/
theorem integral_le_const {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {f : Ω → ℝ} (hf : Integrable f μ) {c : ℝ}
    (h : ∀ ω, f ω ≤ c) : (∫ ω, f ω ∂μ) ≤ c := by
  have hmono := integral_mono hf (integrable_const c) h
  simpa using hmono

/-- **The rate law in expectation.** Randomise the strategy however you like —
`strat` assigns a policy to each sample point of an arbitrary probability space — and
as long as every realisation is legal and alive at bag `m`, the *expected* number
of rows cleared in `m` bags is trapped in `[2.8 m − 20, 2.8 m]`. Averaging cannot
escape a pathwise conservation law, so no randomisation, no distribution over
piece orders, and no mixed strategy can shift the clearing rate off `2.8`. -/
theorem expected_clears_bounds {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (strat : Ω → Policy GameConfig.standard) (m : ℕ)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hlive : ∀ ω, ¬ (trace GameConfig.standard (strat ω) GameState.init (7 * m)).lost
      GameConfig.standard)
    (hint : Integrable
      (fun ω => (cleared GameConfig.standard (strat ω) GameState.init (7 * m) : ℝ)) μ) :
    2.8 * m - 20
        ≤ ∫ ω, (cleared GameConfig.standard (strat ω) GameState.init (7 * m) : ℝ) ∂μ
      ∧ (∫ ω, (cleared GameConfig.standard (strat ω) GameState.init (7 * m) : ℝ) ∂μ)
        ≤ 2.8 * m := by
  refine ⟨const_le_integral hint ?_, integral_le_const hint ?_⟩
  · intro ω
    have : (28 : ℝ) * m
        ≤ 10 * (cleared GameConfig.standard (strat ω) GameState.init (7 * m) : ℝ) + 200 := by
      exact_mod_cast bags_le_cleared (hv ω) (hlive ω)
    linarith
  · intro ω
    have : (10 : ℝ) * (cleared GameConfig.standard (strat ω) GameState.init (7 * m) : ℝ)
        ≤ 28 * m := by exact_mod_cast bags_cleared_le (hv ω) m
    linarith

end ClearRate
end Tetris
