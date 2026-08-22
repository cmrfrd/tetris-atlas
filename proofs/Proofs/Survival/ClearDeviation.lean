import Mathlib
import Proofs.Survival.ClearRate

/-!
# The deviation calculus: how far off 2.8 a solver may drift

`Proofs/Survival/ClearRate` pins the asymptotic clearing rate of any immortal
policy at exactly `2.8` rows per bag. This file quantifies the *deviations*:
how far from that pace a solver can stray, for how long, and which observable
statistics of a running solver certify that it must eventually die.

The whole file rests on one identity. Write

  `deficit n = 4n − cols · cleared n`

for the cells delivered but not yet cleared after `n` placements. The trace
ledger (`ClearRate.init_ledger`) says

  **`deficit n = board.count n`** (`deficit_eq_mass`)

— the clearing deficit *is* the board occupancy, cell for cell. Since a live
board holds between `0` and `cols · rows = 200` cells, the deficit is confined
to `[0, 200]` forever. Every result below is a reading of that confinement.

## What it buys

* **Uniform window bound** (`window_bags_ge`, `window_bags_le`): over *any*
  window of `w` bags — not just windows starting at the origin — the rows
  cleared land within **20 rows** of `2.8 w`. The board is a 20-row buffer and
  that is the entire budget for deviation, at every time scale simultaneously.
* **Never ahead** (`centered_nonpos`): the centered clear count
  `cleared − 2.8m` is always `≤ 0`. A solver can run a deficit but can never
  bank a surplus, so "clear extra now to spend later" is not a strategy that
  exists. Its **maximum drawdown is 20 rows** (`abs_centered_sub_le`) — plot it
  and the whole curve lives in a 20-row band, forever.
* **The shortfall/duration hyperbola** (`sustained_shortfall_window_le`): a
  sustained shortfall of `β` rows per bag can persist for at most `20/β` bags.
  Contrapositive `lost_of_sustained_shortfall`: exceed that and the solver is
  provably dead. This is the practical death-horizon test.
* **The dry-spell bound** (`lost_of_dry_spell`): eight consecutive bags deliver
  `224 > 200` cells, so **no solver can go eight bags without clearing a row**.
  Seven is the hard maximum.
* **A wide marginal range** (`bagClears_le_twentytwo`): a single bag may clear
  anywhere from `0` to `22` rows, so there is **no useful bound on the per-bag
  variance**. Any statistical test of a solver must therefore look at the
  long-run variance, not the marginal one.
* **Degenerate CLT scaling** (`centered_div_sqrt_tendsto_zero`,
  `centered_sq_div_tendsto_zero`): the centered clear count divided by `√m`
  tends to `0`, equivalently the horizon-`m` long-run-variance estimator
  `(∑ deviations)²/m` tends to `0`. For any stochastic process obeying a central
  limit theorem the former converges to a nondegenerate Gaussian, so an immortal
  solver's clear series has **long-run variance exactly zero**.
* **A covariance budget** (`covariance_sum_le`): with no dependence assumption
  at all, the entire `m × m` covariance matrix of the per-bag counts sums to at
  most `400`. Its diagonal is the total marginal variance, so the off-diagonal
  autocovariance mass must cancel that variance completely and grow ever more
  negative with the horizon.
* **No independent increments** (`variance_zero_of_bounded_partial_sums`,
  `survival_forces_indep_variance_zero`, `survival_forces_indep_ae_const`): if a
  solver's per-bag clear counts are pairwise independent with a common variance,
  that variance must be `0` and the counts are almost surely deterministic.
  Independent-increment behaviour with any spread whatsoever is fatal — the
  per-bag deviations of an immortal solver must be strongly negatively
  correlated, exactly cancelling at every horizon.
* **Not even nonnegative correlation** (`variance_zero_of_nonneg_covariance`):
  independence can be weakened to "no pair of bags is negatively correlated" and
  the same conclusion holds. Genuine negative autocovariance — self-correction
  after a bad bag — is forced, not optional.

## What it does not buy

The deficit is bounded by `200` *because* the board is alive, so no finite
observation of a *live* trajectory can exhibit a violation: the deterministic
certificates fire exactly at death, never before it. Genuine prediction needs
an extrapolation hypothesis — a sustained rate (`lost_of_sustained_shortfall`)
or a stochastic model (`survival_forces_indep_variance_zero`). Those are the
honest shapes of "this solver will lose".
-/

namespace Tetris
namespace ClearRate

open Filter Topology MeasureTheory ProbabilityTheory

/-! ## The deficit and its identification with board mass -/

/-- The clearing deficit after `n` placements: cells delivered minus cells
removed by clears, as an integer. -/
def deficit (cfg : GameConfig) (π : Policy cfg) (g0 : GameState) (n : ℕ) : ℤ :=
  4 * n - cfg.cols * cleared cfg π g0 n

/-- **The deficit is the board.** Whatever a player has failed to clear is
exactly what is still sitting on the stack — one identity, no inequality. -/
theorem deficit_eq_mass {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    deficit cfg π GameState.init n
      = ((trace cfg π GameState.init n).board.count : ℤ) := by
  have h : ((trace cfg π GameState.init n).board.count : ℤ)
      + (cfg.cols : ℤ) * (cleared cfg π GameState.init n : ℤ) = 4 * n := by
    exact_mod_cast init_ledger hv n
  unfold deficit
  linarith

/-- The deficit is never negative: you cannot clear cells you were not given. -/
theorem deficit_nonneg {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) (n : ℕ) :
    0 ≤ deficit cfg π GameState.init n := by
  rw [deficit_eq_mass hv]
  exact Int.natCast_nonneg _

/-- While alive the deficit is capped by the playfield capacity. -/
theorem deficit_le_capacity {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {n : ℕ}
    (hlive : ¬ (trace cfg π GameState.init n).lost cfg) :
    deficit cfg π GameState.init n ≤ (cfg.cols * cfg.rows : ℕ) := by
  rw [deficit_eq_mass hv]
  exact_mod_cast BagGrowth.count_le_capacity
    (trace_board_wf hv (GameState.init_board_wf cfg) n)
    ((GameState.not_lost_iff_forall_row_lt cfg _).mp hlive)

/-! ## Windows: the bound holds at every time scale, not just from the origin -/

/-- The window identity: clears over `[a, b]` equal the delivered mass over the
window minus the change in deficit. -/
theorem window_identity (cfg : GameConfig) (π : Policy cfg) (g0 : GameState)
    (a b : ℕ) :
    (cfg.cols : ℤ) * ((cleared cfg π g0 b : ℤ) - (cleared cfg π g0 a : ℤ))
      = 4 * ((b : ℤ) - (a : ℤ))
        - (deficit cfg π g0 b - deficit cfg π g0 a) := by
  unfold deficit
  ring

/-- **Window floor.** Over any window ending at a live state, the clears trail
the delivered mass by at most one boardful — *regardless of where the window
starts*. -/
theorem window_clears_ge {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {a b : ℕ}
    (hb : ¬ (trace cfg π GameState.init b).lost cfg) :
    4 * ((b : ℤ) - (a : ℤ)) - (cfg.cols * cfg.rows : ℕ)
      ≤ (cfg.cols : ℤ) * ((cleared cfg π GameState.init b : ℤ)
          - (cleared cfg π GameState.init a : ℤ)) := by
  have hid := window_identity cfg π GameState.init a b
  have h1 := deficit_le_capacity hv hb
  have h2 := deficit_nonneg hv a
  linarith

/-- **Window ceiling.** Over any window starting at a live state, the clears
exceed the delivered mass by at most one boardful. -/
theorem window_clears_le {cfg : GameConfig} {π : Policy cfg}
    (hv : ∀ g, (π g).Valid cfg) {a b : ℕ}
    (ha : ¬ (trace cfg π GameState.init a).lost cfg) :
    (cfg.cols : ℤ) * ((cleared cfg π GameState.init b : ℤ)
        - (cleared cfg π GameState.init a : ℤ))
      ≤ 4 * ((b : ℤ) - (a : ℤ)) + (cfg.cols * cfg.rows : ℕ) := by
  have hid := window_identity cfg π GameState.init a b
  have h1 := deficit_le_capacity hv ha
  have h2 := deficit_nonneg hv b
  linarith

/-- Bag form of the window floor at standard width: over any `w`-bag window
ending alive, `10 · clears ≥ 28 w − 200`. -/
theorem window_bags_ge {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ}
    (hb : ¬ (trace GameConfig.standard π GameState.init (7 * b)).lost
      GameConfig.standard) :
    28 * ((b : ℤ) - (a : ℤ)) - 200
      ≤ 10 * ((cleared GameConfig.standard π GameState.init (7 * b) : ℤ)
          - (cleared GameConfig.standard π GameState.init (7 * a) : ℤ)) := by
  have h := window_clears_ge hv (a := 7 * a) hb
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  push_cast at h ⊢
  linarith

/-- Bag form of the window ceiling at standard width. -/
theorem window_bags_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ}
    (ha : ¬ (trace GameConfig.standard π GameState.init (7 * a)).lost
      GameConfig.standard) :
    10 * ((cleared GameConfig.standard π GameState.init (7 * b) : ℤ)
        - (cleared GameConfig.standard π GameState.init (7 * a) : ℤ))
      ≤ 28 * ((b : ℤ) - (a : ℤ)) + 200 := by
  have h := window_clears_le hv (b := 7 * b) ha
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  push_cast at h ⊢
  linarith

/-! ## Finite death certificates -/

/-- **Window shortfall is fatal.** If over some window the clears fall more than
a boardful behind the `2.8`-per-bag pace, the endpoint is lost. No hypothesis on
the window's start: the certificate needs only the two ends of the ledger. -/
theorem lost_of_window_shortfall {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ}
    (hshort : 10 * ((cleared GameConfig.standard π GameState.init (7 * b) : ℤ)
        - (cleared GameConfig.standard π GameState.init (7 * a) : ℤ)) + 200
      < 28 * ((b : ℤ) - (a : ℤ))) :
    (trace GameConfig.standard π GameState.init (7 * b)).lost GameConfig.standard := by
  by_contra hb
  have h := window_bags_ge hv (a := a) hb
  linarith

/-- **No eight dry bags.** Eight bags deliver `8 · 28 = 224` cells and the board
holds only `200`, so no legal play can go eight consecutive bags without
clearing a row. Seven is the hard maximum dry spell — a one-line, fully
observable liveness test on any solver's clear log. -/
theorem lost_of_dry_spell {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ} (hlen : a + 8 ≤ b)
    (hdry : cleared GameConfig.standard π GameState.init (7 * b)
      = cleared GameConfig.standard π GameState.init (7 * a)) :
    (trace GameConfig.standard π GameState.init (7 * b)).lost GameConfig.standard := by
  have hb8 : (8 : ℤ) ≤ (b : ℤ) - (a : ℤ) := by
    have : (a : ℤ) + 8 ≤ (b : ℤ) := by exact_mod_cast hlen
    linarith
  refine lost_of_window_shortfall hv (a := a) ?_
  rw [hdry, sub_self]
  linarith

/-! ## The centered clear count -/

/-- The centered clear count after `m` bags: rows actually cleared minus the
`2.8`-per-bag pace. This is the single statistic every result here is about. -/
noncomputable def centered (π : Policy GameConfig.standard) (m : ℕ) : ℝ :=
  (cleared GameConfig.standard π GameState.init (7 * m) : ℝ) - 2.8 * m

/-- **The centered count is minus a tenth of the board.** Every statistical
question about a solver's clear rate is the same question about its board
occupancy — there is no independent information in the rate series. -/
theorem centered_eq_neg_deficit (π : Policy GameConfig.standard) (m : ℕ) :
    centered π m
      = -((deficit GameConfig.standard π GameState.init (7 * m) : ℝ) / 10) := by
  unfold centered deficit
  rw [GameConfig.standard_cols]
  push_cast
  ring

/-- **A solver is never ahead.** The centered clear count is always `≤ 0`: no
strategy can bank a surplus of clears to spend later. -/
theorem centered_nonpos {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (m : ℕ) :
    centered π m ≤ 0 := by
  rw [centered_eq_neg_deficit]
  have h : (0 : ℤ) ≤ deficit GameConfig.standard π GameState.init (7 * m) :=
    deficit_nonneg hv _
  have h' : (0 : ℝ) ≤ (deficit GameConfig.standard π GameState.init (7 * m) : ℝ) := by
    exact_mod_cast h
  linarith

/-- **And never more than 20 rows behind.** -/
theorem neg_twenty_le_centered {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    -20 ≤ centered π m := by
  rw [centered_eq_neg_deficit]
  have h := deficit_le_capacity hv hlive
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  have h' : (deficit GameConfig.standard π GameState.init (7 * m) : ℝ) ≤ 200 := by
    exact_mod_cast h
  linarith

/-- **The master statistic is bounded.** `|cleared − 2.8m| ≤ 20`, for every `m`,
for every legal policy that is still alive. -/
theorem abs_centered_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    |centered π m| ≤ 20 := by
  rw [abs_le]
  exact ⟨neg_twenty_le_centered hv hlive, le_trans (centered_nonpos hv m) (by norm_num)⟩

/-- The centered count is `m` times the rate deviation: the sample mean of the
per-bag clear counts converges to `2.8` at rate `O(1/m)`, not `O(1/√m)`. -/
theorem mul_abs_bagRate_sub_eq {π : Policy GameConfig.standard} {m : ℕ} (hm : 0 < m) :
    (m : ℝ) * |bagRate π m - 2.8| = |centered π m| := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hne : (m : ℝ) ≠ 0 := ne_of_gt hm'
  have hkey : bagRate π m - 2.8 = centered π m / m := by
    rw [bagRate, centered, sub_div, mul_div_assoc, div_self hne, mul_one]
  rw [hkey, abs_div, abs_of_pos hm']
  field_simp

/-- **Maximum drawdown.** Between any two live checkpoints the centered clear
count moves by at most 20 rows. Plot `centered` against bag index for a running
solver and the entire curve is confined to a 20-row band forever: it is the one
diagnostic worth watching, and leaving the band *is* death. -/
theorem abs_centered_sub_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ}
    (ha : ¬ (trace GameConfig.standard π GameState.init (7 * a)).lost
      GameConfig.standard)
    (hb : ¬ (trace GameConfig.standard π GameState.init (7 * b)).lost
      GameConfig.standard) :
    |centered π b - centered π a| ≤ 20 := by
  have h1 := centered_nonpos hv a
  have h2 := centered_nonpos hv b
  have h3 := neg_twenty_le_centered hv ha
  have h4 := neg_twenty_le_centered hv hb
  rw [abs_le]
  constructor <;> linarith

/-! ## The shortfall–duration hyperbola -/

/-- **How far out of line a solver may be.** A shortfall of `β` rows per bag,
sustained across a window, can last at most `20/β` bags. Deviating twice as far
buys half as long: the product of size and duration is capped by the board.
The `β → 0` limit recovers "the asymptotic rate is 2.8". -/
theorem sustained_shortfall_window_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ} {β : ℝ} (hβ : 0 < β)
    (hb : ¬ (trace GameConfig.standard π GameState.init (7 * b)).lost
      GameConfig.standard)
    (hslow : (cleared GameConfig.standard π GameState.init (7 * b) : ℝ)
        - (cleared GameConfig.standard π GameState.init (7 * a) : ℝ)
      ≤ (2.8 - β) * ((b : ℝ) - (a : ℝ))) :
    (b : ℝ) - (a : ℝ) ≤ 20 / β := by
  have hint := window_bags_ge hv (a := a) hb
  have hreal : (28 : ℝ) * ((b : ℝ) - (a : ℝ)) - 200
      ≤ 10 * ((cleared GameConfig.standard π GameState.init (7 * b) : ℝ)
          - (cleared GameConfig.standard π GameState.init (7 * a) : ℝ)) := by
    exact_mod_cast hint
  have hkey : β * ((b : ℝ) - (a : ℝ)) ≤ 20 := by nlinarith
  rw [le_div_iff₀ hβ]
  linarith

/-- Contrapositive: **the death horizon.** Sustain a shortfall of `β` rows per
bag for more than `20/β` bags and the solver is provably dead. This is the
predictive test — fit `β` to a running solver's clear log and read off the bag
count by which it must have topped out. -/
theorem lost_of_sustained_shortfall {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {a b : ℕ} {β : ℝ} (hβ : 0 < β)
    (hslow : (cleared GameConfig.standard π GameState.init (7 * b) : ℝ)
        - (cleared GameConfig.standard π GameState.init (7 * a) : ℝ)
      ≤ (2.8 - β) * ((b : ℝ) - (a : ℝ)))
    (hlong : 20 / β < (b : ℝ) - (a : ℝ)) :
    (trace GameConfig.standard π GameState.init (7 * b)).lost GameConfig.standard := by
  by_contra hb
  exact absurd (sustained_shortfall_window_le hv hβ hb hslow) (not_le.mpr hlong)

/-! ## Degenerate CLT scaling: the long-run variance is zero -/

/-- **The `√m`-scaled deviation vanishes.** For any process satisfying a central
limit theorem, `(S_m − mμ)/√m` converges to a nondegenerate Gaussian; here it
converges to `0`. So an immortal solver's per-bag clear series has long-run
variance (spectral density at zero) exactly `0` — its autocovariances must sum
to `−σ²/2`, i.e. the deviations cancel at every horizon rather than accumulating
like a random walk. -/
theorem centered_div_sqrt_tendsto_zero {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) :
    Tendsto (fun m : ℕ => |centered π m| / Real.sqrt m) atTop (𝓝 0) := by
  have hsq : Tendsto (fun m : ℕ => Real.sqrt m) atTop atTop :=
    Real.tendsto_sqrt_atTop.comp tendsto_natCast_atTop_atTop
  have hub : Tendsto (fun m : ℕ => (20 : ℝ) / Real.sqrt m) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop hsq
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hub ?_ ?_
  · exact Eventually.of_forall fun m => by positivity
  · filter_upwards [eventually_gt_atTop 0] with m hm
    have hmr : (0 : ℝ) < m := by exact_mod_cast hm
    have hm' : (0 : ℝ) < Real.sqrt m := Real.sqrt_pos.mpr hmr
    gcongr
    exact abs_centered_le hv (hsurv (7 * m))

/-- The same statement as a uniform bound: `|centered| / √m ≤ 20/√m`. -/
theorem abs_centered_div_sqrt_le {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 0 < m)
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * m)).lost
      GameConfig.standard) :
    |centered π m| / Real.sqrt m ≤ 20 / Real.sqrt m := by
  have hmr : (0 : ℝ) < m := by exact_mod_cast hm
  have hm' : (0 : ℝ) < Real.sqrt m := Real.sqrt_pos.mpr hmr
  gcongr
  exact abs_centered_le hv hlive

/-! ## Per-bag clear counts -/

/-- Rows cleared during bag `k` (bags are 7 placements). -/
def bagClears (π : Policy GameConfig.standard) (k : ℕ) : ℕ :=
  cleared GameConfig.standard π GameState.init (7 * (k + 1))
    - cleared GameConfig.standard π GameState.init (7 * k)

/-- The centered clear count is the partial sum of the centered per-bag counts:
`centered m = ∑_{k<m} (c_k − 2.8)`. So every bound on `centered` is a bound on a
partial sum of the deviation series. -/
theorem sum_bagClears_centered (π : Policy GameConfig.standard) (m : ℕ) :
    ∑ k ∈ Finset.range m, ((bagClears π k : ℝ) - 2.8) = centered π m := by
  induction m with
  | zero => simp [centered]
  | succ j ih =>
    have hmono : cleared GameConfig.standard π GameState.init (7 * j)
        ≤ cleared GameConfig.standard π GameState.init (7 * (j + 1)) :=
      cleared_mono GameConfig.standard π GameState.init (by omega)
    have hcast : (bagClears π j : ℝ)
        = (cleared GameConfig.standard π GameState.init (7 * (j + 1)) : ℝ)
          - (cleared GameConfig.standard π GameState.init (7 * j) : ℝ) := by
      unfold bagClears
      rw [Nat.cast_sub hmono]
    rw [Finset.sum_range_succ, ih, hcast]
    unfold centered
    push_cast
    ring

/-- **At most 22 rows in a single bag.** A bag delivers 28 cells and the board
can supply at most another 200, so `10 c ≤ 228`. Together with
`lost_of_dry_spell` the per-bag clear count of a live solver is pinned to
`{0, …, 22}` with running mean exactly `2.8` and no seven-bag gap. Note the
range is wide: **there is no useful bound on the per-bag variance**, which is
why the honest statistical constraint is on the *long-run* variance
(`centered_sq_div_tendsto_zero`), not the marginal one. -/
theorem bagClears_le_twentytwo {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {k : ℕ}
    (hlive : ¬ (trace GameConfig.standard π GameState.init (7 * k)).lost
      GameConfig.standard) :
    bagClears π k ≤ 22 := by
  have hmono : cleared GameConfig.standard π GameState.init (7 * k)
      ≤ cleared GameConfig.standard π GameState.init (7 * (k + 1)) :=
    cleared_mono GameConfig.standard π GameState.init (by omega)
  have h := window_bags_le hv (a := k) (b := k + 1) hlive
  have hcast : ((bagClears π k : ℕ) : ℤ)
      = (cleared GameConfig.standard π GameState.init (7 * (k + 1)) : ℤ)
        - (cleared GameConfig.standard π GameState.init (7 * k) : ℤ) := by
    unfold bagClears
    exact Nat.cast_sub hmono
  push_cast at h
  omega

/-! ## The long-run variance estimator -/

/-- **The long-run variance vanishes.** `(∑ deviations)² / m` is the textbook
horizon-`m` estimator of the long-run variance (the spectral density at zero) of
the per-bag clear series. For an immortal solver it tends to `0`, so the series
has long-run variance exactly zero — the diagnostic to run on a solver's clear
log: estimate it, and if it is bounded away from `0`, the solver is mortal. -/
theorem centered_sq_div_tendsto_zero {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hsurv : SurvivesForever GameConfig.standard π GameState.init) :
    Tendsto (fun m : ℕ => (centered π m) ^ 2 / m) atTop (𝓝 0) := by
  have hub : Tendsto (fun m : ℕ => (400 : ℝ) / m) atTop (𝓝 0) :=
    tendsto_const_div_atTop_nhds_zero_nat 400
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hub ?_ ?_
  · exact Eventually.of_forall fun m => by positivity
  · filter_upwards [eventually_gt_atTop 0] with m hm
    have hmr : (0 : ℝ) < m := by exact_mod_cast hm
    have habs := abs_centered_le hv (hsurv (7 * m))
    have hsq : (centered π m) ^ 2 ≤ 400 := by
      nlinarith [abs_nonneg (centered π m), sq_abs (centered π m)]
    gcongr

/-! ## No independent increments -/

/-- **Bounded partial sums force zero variance.** A sequence of pairwise
independent, square-integrable random variables whose partial sums are uniformly
bounded by `B`, and which share a common variance `σ²`, must have `σ² = 0`.

The proof is one line of statistics: independence makes the variance of the
`m`-th partial sum equal `m σ²`, while the uniform bound caps it at `B²`. Let
`m → ∞`. -/
theorem variance_zero_of_bounded_partial_sums {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hmem : ∀ i, MemLp (X i) 2 μ)
    (hindep : Pairwise fun i j => IndepFun (X i) (X j) μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B)
    {σ2 : ℝ} (hvar : ∀ i, variance (X i) μ = σ2) :
    σ2 = 0 := by
  have hσ : 0 ≤ σ2 := by
    have h := variance_nonneg (X 0) μ
    rwa [hvar 0] at h
  have hkey : ∀ m : ℕ, (m : ℝ) * σ2 ≤ B ^ 2 := by
    intro m
    have hSmem : MemLp (∑ i ∈ Finset.range m, X i) 2 μ :=
      memLp_finset_sum' (Finset.range m) (fun i _ => hmem i)
    have hsum : variance (∑ i ∈ Finset.range m, X i) μ = (m : ℝ) * σ2 := by
      rw [ProbabilityTheory.IndepFun.variance_sum (fun i _ => hmem i)
        (fun i _ j _ hij => hindep hij)]
      simp [hvar]
    have hexp : μ[(∑ i ∈ Finset.range m, X i) ^ 2] ≤ B ^ 2 := by
      refine integral_le_const hSmem.integrable_sq ?_
      intro ω
      have hb := abs_le.mp (hB m ω)
      have happ : (∑ i ∈ Finset.range m, X i) ω = ∑ i ∈ Finset.range m, X i ω :=
        Finset.sum_apply ω (Finset.range m) X
      simp only [happ]
      exact sq_le_sq' hb.1 hb.2
    have hvle := variance_le_expectation_sq (μ := μ)
      (X := ∑ i ∈ Finset.range m, X i) hSmem.aestronglyMeasurable
    rw [hsum] at hvle
    linarith
  by_contra hne
  have hpos : 0 < σ2 := lt_of_le_of_ne hσ (Ne.symm hne)
  obtain ⟨m, hm⟩ := exists_nat_gt (B ^ 2 / σ2)
  have hkm := hkey m
  rw [div_lt_iff₀ hpos] at hm
  linarith

/-- **The covariance budget.** Drop the independence assumption entirely: for
*any* dependence structure, the full `m × m` covariance matrix of the centered
per-bag counts sums to at most `B²`. The diagonal of that matrix is the total
marginal variance `∑ Var[X i]`, so the off-diagonal autocovariance mass must be
at least as negative — and grows without bound in the negative direction as the
horizon grows. This is the exact sense in which an immortal solver's clear
series must be anti-correlated: the autocovariances do not merely fail to be
positive, they must cancel the marginal variances completely. -/
theorem covariance_sum_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ) (hmem : ∀ i, MemLp (X i) 2 μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B) (m : ℕ) :
    ∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m,
        ProbabilityTheory.covariance (X i) (X j) μ ≤ B ^ 2 := by
  have hSmem : MemLp (∑ i ∈ Finset.range m, X i) 2 μ :=
    memLp_finset_sum' (Finset.range m) (fun i _ => hmem i)
  have hdec := ProbabilityTheory.variance_sum' (μ := μ) (s := Finset.range m)
    (X := X) (fun i _ => hmem i)
  have hexp : μ[(∑ i ∈ Finset.range m, X i) ^ 2] ≤ B ^ 2 := by
    refine integral_le_const hSmem.integrable_sq ?_
    intro ω
    have hb := abs_le.mp (hB m ω)
    have happ : (∑ i ∈ Finset.range m, X i) ω = ∑ i ∈ Finset.range m, X i ω :=
      Finset.sum_apply ω (Finset.range m) X
    simp only [happ]
    exact sq_le_sq' hb.1 hb.2
  have hvle := variance_le_expectation_sq (μ := μ)
    (X := ∑ i ∈ Finset.range m, X i) hSmem.aestronglyMeasurable
  rw [hdec] at hvle
  linarith

/-- **Nonnegative autocorrelation is fatal.** Weaken independence all the way
down to "no pair of bags is *negatively* correlated" and the conclusion
survives: with a common variance and bounded partial sums, that variance is
zero. So an immortal solver must carry genuinely negative autocovariance —
self-correction after a bad bag is not a design choice, it is forced. -/
theorem card_mul_variance_le_of_nonneg_covariance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hmem : ∀ i, MemLp (X i) 2 μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B)
    {σ2 : ℝ} (hvar : ∀ i, variance (X i) μ = σ2)
    {L : ℕ} (hcov : ∀ i ∈ Finset.range L, ∀ j ∈ Finset.range L, i ≠ j →
      0 ≤ ProbabilityTheory.covariance (X i) (X j) μ) :
    (L : ℝ) * σ2 ≤ B ^ 2 := by
  have hsum := covariance_sum_le μ X hmem hB L
  have hdiag : ∀ i ∈ Finset.range L,
      ProbabilityTheory.covariance (X i) (X i) μ
        ≤ ∑ j ∈ Finset.range L, ProbabilityTheory.covariance (X i) (X j) μ := by
    intro i hi
    refine Finset.single_le_sum (f := fun j =>
      ProbabilityTheory.covariance (X i) (X j) μ) ?_ hi
    intro j hj
    dsimp only
    rcases eq_or_ne i j with rfl | hij
    · rw [ProbabilityTheory.covariance_self (hmem i).aemeasurable]
      exact variance_nonneg _ _
    · exact hcov i hi j hj hij
  have hlow := Finset.sum_le_sum hdiag
  have hconst : ∑ i ∈ Finset.range L,
      ProbabilityTheory.covariance (X i) (X i) μ = (L : ℝ) * σ2 := by
    have hpt : ∀ i ∈ Finset.range L,
        ProbabilityTheory.covariance (X i) (X i) μ = σ2 := by
      intro i _
      rw [ProbabilityTheory.covariance_self (hmem i).aemeasurable, hvar i]
    rw [Finset.sum_congr rfl hpt, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul]
  rw [hconst] at hlow
  linarith

/-- Letting the horizon grow: if the non-negativity holds at every lag, the
variance must vanish. -/
theorem variance_zero_of_nonneg_covariance {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hmem : ∀ i, MemLp (X i) 2 μ)
    (hcov : ∀ i j, i ≠ j → 0 ≤ ProbabilityTheory.covariance (X i) (X j) μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B)
    {σ2 : ℝ} (hvar : ∀ i, variance (X i) μ = σ2) :
    σ2 = 0 := by
  have hσ : 0 ≤ σ2 := by
    have h := variance_nonneg (X 0) μ
    rwa [hvar 0] at h
  by_contra hne
  have hpos : 0 < σ2 := lt_of_le_of_ne hσ (Ne.symm hne)
  obtain ⟨m, hm⟩ := exists_nat_gt (B ^ 2 / σ2)
  have hkm := card_mul_variance_le_of_nonneg_covariance μ X hmem hB hvar
    (L := m) (fun i _ j _ hij => hcov i j hij)
  rw [div_lt_iff₀ hpos] at hm
  linarith

/-- **Independent per-bag clearing is fatal unless it is deterministic.**
Randomise a solver however you like; if its per-bag clear counts are pairwise
independent with a common variance and it survives forever, that variance is
zero. Independence plus spread makes the deviation series a random walk, whose
partial sums grow like `√m` — but the board caps them at 20 rows forever.

So an immortal solver's bag-to-bag clear counts cannot be independent draws:
they must be strongly negatively correlated, cancelling exactly at every
horizon. -/
theorem survival_forces_indep_variance_zero {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (strat : Ω → Policy GameConfig.standard)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hsurv : ∀ ω, SurvivesForever GameConfig.standard (strat ω) GameState.init)
    (X : ℕ → Ω → ℝ)
    (hX : ∀ k ω, X k ω = (bagClears (strat ω) k : ℝ) - 2.8)
    (hmem : ∀ k, MemLp (X k) 2 μ)
    (hindep : Pairwise fun i j => IndepFun (X i) (X j) μ)
    {σ2 : ℝ} (hvar : ∀ k, variance (X k) μ = σ2) :
    σ2 = 0 := by
  refine variance_zero_of_bounded_partial_sums μ X hmem hindep (B := 20) ?_ hvar
  intro m ω
  have hs : ∑ k ∈ Finset.range m, X k ω = centered (strat ω) m := by
    rw [← sum_bagClears_centered (strat ω) m]
    exact Finset.sum_congr rfl fun k _ => hX k ω
  rw [hs]
  exact abs_centered_le (hv ω) (hsurv ω (7 * m))

/-- The partial sums of an immortal randomized solver's centered per-bag counts
never leave the 20-row band. -/
theorem abs_sum_bagDeviation_le {Ω : Type*}
    (strat : Ω → Policy GameConfig.standard)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hsurv : ∀ ω, SurvivesForever GameConfig.standard (strat ω) GameState.init)
    (X : ℕ → Ω → ℝ)
    (hX : ∀ k ω, X k ω = (bagClears (strat ω) k : ℝ) - 2.8)
    (m : ℕ) (ω : Ω) :
    |∑ i ∈ Finset.range m, X i ω| ≤ 20 := by
  have hs : ∑ k ∈ Finset.range m, X k ω = centered (strat ω) m := by
    rw [← sum_bagClears_centered (strat ω) m]
    exact Finset.sum_congr rfl fun k _ => hX k ω
  rw [hs]
  exact abs_centered_le (hv ω) (hsurv ω (7 * m))

/-- **The off-diagonal covariance budget.** Peeling the diagonal off
`covariance_sum_le`: across the first `L` bags the *cross* terms must absorb the
entire marginal variance `L σ²`, up to the board's `B²`. -/
theorem offDiag_covariance_sum_le {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hmem : ∀ i, MemLp (X i) 2 μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B)
    {σ2 : ℝ} (hvar : ∀ i, variance (X i) μ = σ2) (L : ℕ) :
    ∑ i ∈ Finset.range L, ∑ j ∈ (Finset.range L).erase i,
        ProbabilityTheory.covariance (X i) (X j) μ
      ≤ B ^ 2 - (L : ℝ) * σ2 := by
  have hsum := covariance_sum_le μ X hmem hB L
  have hsplit : ∀ i ∈ Finset.range L,
      ∑ j ∈ Finset.range L, ProbabilityTheory.covariance (X i) (X j) μ
        = σ2 + ∑ j ∈ (Finset.range L).erase i,
            ProbabilityTheory.covariance (X i) (X j) μ := by
    intro i hi
    rw [← Finset.add_sum_erase _ (fun j => ProbabilityTheory.covariance (X i) (X j) μ) hi,
      ProbabilityTheory.covariance_self (hmem i).aemeasurable, hvar i]
  rw [Finset.sum_congr rfl hsplit, Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_range, nsmul_eq_mul] at hsum
  linarith

/-- **Correction is located, not merely eventual.** Once the horizon `L` exceeds
`B²/σ²`, some *specific pair* of bags inside the first `L` has negatively
correlated clear counts. The solver does not just self-correct "in the limit" —
a correcting pair sits within every sufficiently long window. -/
theorem exists_neg_covariance_of_horizon {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (X : ℕ → Ω → ℝ)
    (hmem : ∀ i, MemLp (X i) 2 μ)
    {B : ℝ} (hB : ∀ (m : ℕ) (ω : Ω), |∑ i ∈ Finset.range m, X i ω| ≤ B)
    {σ2 : ℝ} (hvar : ∀ i, variance (X i) μ = σ2)
    {L : ℕ} (hL : B ^ 2 < (L : ℝ) * σ2) :
    ∃ i ∈ Finset.range L, ∃ j ∈ Finset.range L, i ≠ j ∧
      ProbabilityTheory.covariance (X i) (X j) μ < 0 := by
  by_contra hcon
  push Not at hcon
  have h := card_mul_variance_le_of_nonneg_covariance μ X hmem hB hvar
    (L := L) (fun i hi j hj hij => hcon i hi j hj hij)
  linarith

/-- **The recovery deadline.** The previous theorems say an immortal solver must
correct itself; this one says *when*. Suppose its per-bag clear counts have
variance `σ² > 0` and that across a window of `L` bags no pair is negatively
correlated — that is, for `L` bags the solver never systematically compensates
for a bad bag. Then

  `L ≤ 400 / σ²`.

Correction cannot be postponed past that horizon: the deviations would otherwise
accumulate like a random walk and overrun the board's 20-row budget. The noisier
a solver's clearing, the sooner it must correct — stddev 1 row buys 400 bags,
stddev 2 buys 100, stddev 4 buys 25.

Design reading: the accumulated clear debt has to be an *input* to the policy on
a timescale of `400/σ²` bags. A policy that ignores it for longer is not merely
suboptimal — it provably tops out. -/
theorem recovery_deadline {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (strat : Ω → Policy GameConfig.standard)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hsurv : ∀ ω, SurvivesForever GameConfig.standard (strat ω) GameState.init)
    (X : ℕ → Ω → ℝ)
    (hX : ∀ k ω, X k ω = (bagClears (strat ω) k : ℝ) - 2.8)
    (hmem : ∀ k, MemLp (X k) 2 μ)
    {σ2 : ℝ} (hvar : ∀ k, variance (X k) μ = σ2) (hσ : 0 < σ2)
    {L : ℕ} (hcov : ∀ i ∈ Finset.range L, ∀ j ∈ Finset.range L, i ≠ j →
      0 ≤ ProbabilityTheory.covariance (X i) (X j) μ) :
    (L : ℝ) ≤ 400 / σ2 := by
  have h := card_mul_variance_le_of_nonneg_covariance μ X hmem
    (abs_sum_bagDeviation_le strat hv hsurv X hX) hvar hcov
  norm_num at h
  rw [le_div_iff₀ hσ]
  linarith

/-- **The per-lag recovery law.** Take an immortal randomized solver whose
per-bag clear counts have variance `σ² > 0`, and any window of `L > 400/σ²`
bags. Then two *named* bags inside that window have negatively correlated clear
counts. Self-correction is not an asymptotic tendency — it is scheduled, and the
schedule's period is at most `400/σ²` bags. -/
theorem exists_correcting_pair {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (strat : Ω → Policy GameConfig.standard)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hsurv : ∀ ω, SurvivesForever GameConfig.standard (strat ω) GameState.init)
    (X : ℕ → Ω → ℝ)
    (hX : ∀ k ω, X k ω = (bagClears (strat ω) k : ℝ) - 2.8)
    (hmem : ∀ k, MemLp (X k) 2 μ)
    {σ2 : ℝ} (hvar : ∀ k, variance (X k) μ = σ2) (hσ : 0 < σ2)
    {L : ℕ} (hL : 400 / σ2 < (L : ℝ)) :
    ∃ i ∈ Finset.range L, ∃ j ∈ Finset.range L, i ≠ j ∧
      ProbabilityTheory.covariance (X i) (X j) μ < 0 := by
  refine exists_neg_covariance_of_horizon μ X hmem
    (abs_sum_bagDeviation_le strat hv hsurv X hX) hvar ?_
  rw [div_lt_iff₀ hσ] at hL
  norm_num
  linarith

/-- Spelled out: under independence, an immortal solver's per-bag clear count is
almost surely equal to its own mean — a deterministic schedule, not a
distribution. -/
theorem survival_forces_indep_ae_const {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (strat : Ω → Policy GameConfig.standard)
    (hv : ∀ ω g, (strat ω g).Valid GameConfig.standard)
    (hsurv : ∀ ω, SurvivesForever GameConfig.standard (strat ω) GameState.init)
    (X : ℕ → Ω → ℝ)
    (hX : ∀ k ω, X k ω = (bagClears (strat ω) k : ℝ) - 2.8)
    (hmem : ∀ k, MemLp (X k) 2 μ)
    (hindep : Pairwise fun i j => IndepFun (X i) (X j) μ)
    {σ2 : ℝ} (hvar : ∀ k, variance (X k) μ = σ2) (k : ℕ) :
    ∀ᵐ ω ∂μ, X k ω = μ[X k] := by
  have hz : σ2 = 0 :=
    survival_forces_indep_variance_zero μ strat hv hsurv X hX hmem hindep hvar
  refine ae_eq_integral_of_variance_eq_zero (hmem k) ?_
  rw [hvar k, hz]

end ClearRate
end Tetris
