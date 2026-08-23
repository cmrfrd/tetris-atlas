import Mathlib
import Proofs.Survival.ClearMix
import Proofs.Safety.BagCadence
import Proofs.Safety.CycleQuantum

/-!
# Can the adversary force a clear rate above 2.8?

The question: give the adversary full control of the piece order (within legal
7-bag draws). Can it *force* the player's clearing rate strictly above `2.8`
rows per bag — always, or ever?

**Answer: no — and it cannot force it even once.** The ceiling
`10 · cleared ≤ 4 · pieces` is a conservation law of the dynamics, not a
strategic quantity: a row must be filled before it clears, and from the empty
board the pieces are the only source of cells. The adversary chooses *which*
piece arrives, but every piece carries exactly 4 cells, so the ledger

  `board.count + 10 · cleared = 4 · pieces`   (`clearedAdv_ledger`)

holds along **every** adversarial trace, whatever the piece order and whatever
the solver. Hence (`adversary_cleared_le`, `advBagRate_le`,
`adversary_cannot_force_gt`): the cumulative rate from the empty board never
reaches `2.8 + ε`, for any adversary, any solver, any horizon, any `ε > 0`.

What the adversary *does* get — for free, with no strategy at all — is the
other side: any solver that survives is dragged to exactly `2.8` in the limit
(`adversary_forces_ge`, `adversary_forces_rate`). Every piece sequence forces
this automatically, because the floor comes from the board's 200-cell capacity,
not from adversarial pressure.

So the adversary's power is orthogonal to the rate: it can contest *survival*
(whether the floor can be met geometrically), but the rate of any survivor is
pinned at `2.8` from both sides regardless of who picks the pieces. A
"rate-attacking" adversary is attacking a conserved quantity.
-/

namespace Tetris
namespace ClearRate

open Filter Topology

/-! ## Cumulative clears along an adversarial trace -/

/-- Total rows cleared over the first `n` moves of solver `σ` against piece
sequence `s`, from `g0`. Each move contributes the rows full after the forced
drop (`adversarialStep` plays `{σ g (s n) with piece := s n}`). -/
def clearedAdv (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g0 : GameState) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      clearedAdv cfg σ s g0 n
        + (Board.fullRows cfg
            (({ σ (adversarialTrace cfg σ s g0 n) (s n) with piece := s n }
                : Placement).place
              (adversarialTrace cfg σ s g0 n).board)).card

@[simp] theorem clearedAdv_zero (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (g0 : GameState) : clearedAdv cfg σ s g0 0 = 0 := rfl

theorem clearedAdv_succ (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g0 : GameState) (n : ℕ) :
    clearedAdv cfg σ s g0 (n + 1)
      = clearedAdv cfg σ s g0 n
        + (Board.fullRows cfg
            (({ σ (adversarialTrace cfg σ s g0 n) (s n) with piece := s n }
                : Placement).place
              (adversarialTrace cfg σ s g0 n).board)).card := rfl

/-- **Mass conservation along every adversarial trace.** The adversary picks the
piece, but every piece carries exactly 4 cells and every cleared row removes
`cols`, so the ledger closes whatever the sequence. -/
theorem clearedAdv_ledger {cfg : GameConfig} {σ : Solver cfg} {s : ℕ → Piece}
    {g0 : GameState} (hwf : Board.WF cfg g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace cfg σ s g0 n) (s n) with piece := s n }
      : Placement).Valid cfg) (n : ℕ) :
    (adversarialTrace cfg σ s g0 n).board.count
        + cfg.cols * clearedAdv cfg σ s g0 n
      = g0.board.count + 4 * n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := BagGrowth.count_applyStep_add
      (adversarialTrace_board_wf hwf hv k) (hv k)
    rw [adversarialTrace_succ, adversarialStep_board, clearedAdv_succ, Nat.mul_add]
    dsimp only at hstep ⊢
    omega

/-! ## The ceiling: no adversary can force more than 2.8 -/

/-- **The adversarial ceiling.** From the empty board, `10 · cleared ≤ 4n`
against every piece sequence: the adversary cannot push the cumulative rate
above `0.4` rows per piece even momentarily, because the cells to clear more
simply have not been delivered. -/
theorem adversary_cleared_le {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    10 * clearedAdv GameConfig.standard σ s GameState.init n ≤ 4 * n := by
  have h := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n
  rw [GameConfig.standard_cols, GameState.init_board_count] at h
  omega

/-- Per bag: `m` bags of adversarial play clear at most `2.8 m` rows. -/
theorem adversary_bags_le {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (m : ℕ) :
    10 * clearedAdv GameConfig.standard σ s GameState.init (7 * m) ≤ 28 * m := by
  have h := adversary_cleared_le hv (7 * m)
  omega

/-! ## The floor: every adversary drags a survivor to 2.8 anyway -/

/-- **The adversarial floor.** If the solver is still alive after `m` bags, the
clears trail `2.8 m` by at most one boardful — against every piece sequence.
The adversary needs no strategy to force this; the board's capacity does it. -/
theorem adversary_forces_ge {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {m : ℕ}
    (hlive : ¬ (adversarialTrace GameConfig.standard σ s GameState.init (7 * m)).lost
      GameConfig.standard) :
    28 * m ≤ 10 * clearedAdv GameConfig.standard σ s GameState.init (7 * m) + 200 := by
  have h := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv (7 * m)
  rw [GameConfig.standard_cols, GameState.init_board_count] at h
  have hcap := BagGrowth.count_le_capacity
    (adversarialTrace_board_wf (GameState.init_board_wf GameConfig.standard) hv (7 * m))
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at hcap
  omega

/-! ## The rate, in the limit -/

/-- Rows cleared per bag over the first `m` bags of adversarial play. -/
noncomputable def advBagRate (σ : Solver GameConfig.standard) (s : ℕ → Piece)
    (m : ℕ) : ℝ :=
  (clearedAdv GameConfig.standard σ s GameState.init (7 * m) : ℝ) / m

/-- The adversarial per-bag rate never exceeds `2.8`, at any horizon. -/
theorem advBagRate_le {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (m : ℕ) :
    advBagRate σ s m ≤ 2.8 := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · norm_num [advBagRate]
  · have hm' : (0 : ℝ) < m := by exact_mod_cast hm
    have hcast : (10 : ℝ) * (clearedAdv GameConfig.standard σ s GameState.init (7 * m) : ℝ)
        ≤ 28 * m := by exact_mod_cast adversary_bags_le hv m
    rw [advBagRate, div_le_iff₀ hm']
    linarith

/-- **The answer.** No adversary can force the rate to `2.8 + ε` — for any
`ε > 0`, at any horizon, against any solver. "Forcing more than 2.8" names a
state of the ledger that cannot exist. -/
theorem adversary_cannot_force_gt {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    {ε : ℝ} (hε : 0 < ε) (m : ℕ) :
    ¬ (2.8 + ε ≤ advBagRate σ s m) := by
  intro hge
  have := advBagRate_le hv m
  linarith

/-- While alive, the adversarial rate is within `20/m` of `2.8` from below. -/
theorem le_advBagRate {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    {m : ℕ} (hm : 0 < m)
    (hlive : ¬ (adversarialTrace GameConfig.standard σ s GameState.init (7 * m)).lost
      GameConfig.standard) :
    2.8 - 20 / (m : ℝ) ≤ advBagRate σ s m := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hcast : (28 : ℝ) * m
      ≤ 10 * (clearedAdv GameConfig.standard σ s GameState.init (7 * m) : ℝ) + 200 := by
    exact_mod_cast adversary_forces_ge hv hlive
  rw [advBagRate, le_div_iff₀ hm']
  have hkey : (2.8 - 20 / (m : ℝ)) * m = 2.8 * m - 20 := by field_simp
  rw [hkey]
  linarith

/-- **Every adversary pins a survivor at exactly 2.8.** If the solver survives
the whole sequence, its per-bag rate converges to `2.8` — the same limit as in
the cooperative game, for every piece order. The adversary cannot move the rate
of a survivor in either direction; its only lever is survival itself. -/
theorem adversary_forces_rate {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    (hsurv : ∀ n, ¬ (adversarialTrace GameConfig.standard σ s GameState.init n).lost
      GameConfig.standard) :
    Tendsto (advBagRate σ s) atTop (𝓝 2.8) := by
  have hlow : Tendsto (fun m : ℕ => (2.8 : ℝ) - 20 / m) atTop (𝓝 2.8) := by
    have h0 := tendsto_const_div_atTop_nhds_zero_nat (20 : ℝ)
    simpa using (tendsto_const_nhds (x := (2.8 : ℝ)) (f := atTop)).sub h0
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hlow tendsto_const_nhds ?_ ?_
  · filter_upwards [eventually_gt_atTop 0] with m hm
    exact le_advBagRate hv hm (hsurv (7 * m))
  · exact Eventually.of_forall fun m => advBagRate_le hv m

/-- **Adversarially too, the first clear arrives by placement fifty-one.**
Along any adversarial trace that is still alive at `n ≥ 51`, at least one row
has been cleared: with zero clears the delivered mass `4n > 200` cannot fit a
live board. Certificates past depth 50 must clear, whoever picks the pieces. -/
theorem adversary_first_clear_by_fiftyone {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    {n : ℕ} (hn : 51 ≤ n)
    (hlive : ¬ (adversarialTrace GameConfig.standard σ s GameState.init n).lost
      GameConfig.standard) :
    0 < clearedAdv GameConfig.standard σ s GameState.init n := by
  have h := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n
  rw [GameConfig.standard_cols, GameState.init_board_count] at h
  have hcap := BagGrowth.count_le_capacity
    (adversarialTrace_board_wf (GameState.init_board_wf GameConfig.standard) hv n)
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at hcap
  omega

/-! ## Standing inventory, adversarially

The occupancy statistics of `Survival/StandingInventory` are properties of the
dynamics, not of the cooperative setting: they transfer verbatim to
adversarial traces. An immortal solver carries at least `2.4` cells of
time-averaged inventory *against every piece order*, banks at least six cells
before every clearing moment, and leaves its board empty at most a fifth of
the time. -/

/-- Cumulative occupancy over the first `n` checkpoints of an adversarial
trace. -/
def sumCountAdv (σ : Solver GameConfig.standard) (s : ℕ → Piece) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      sumCountAdv σ s n
        + (adversarialTrace GameConfig.standard σ s GameState.init n).board.count

@[simp] theorem sumCountAdv_zero (σ : Solver GameConfig.standard)
    (s : ℕ → Piece) : sumCountAdv σ s 0 = 0 := rfl

theorem sumCountAdv_succ (σ : Solver GameConfig.standard) (s : ℕ → Piece)
    (n : ℕ) :
    sumCountAdv σ s (n + 1)
      = sumCountAdv σ s n
        + (adversarialTrace GameConfig.standard σ s GameState.init n).board.count :=
  rfl

/-- Number of the first `n` adversarial moves that cleared at least one row. -/
def clearingStepsAdv (σ : Solver GameConfig.standard) (s : ℕ → Piece) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      clearingStepsAdv σ s n
        + (if 0 < (Board.fullRows GameConfig.standard
              (({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
                  with piece := s n } : Placement).place
                (adversarialTrace GameConfig.standard σ s GameState.init n).board)).card
           then 1 else 0)

@[simp] theorem clearingStepsAdv_zero (σ : Solver GameConfig.standard)
    (s : ℕ → Piece) : clearingStepsAdv σ s 0 = 0 := rfl

theorem clearingStepsAdv_succ (σ : Solver GameConfig.standard) (s : ℕ → Piece)
    (n : ℕ) :
    clearingStepsAdv σ s (n + 1)
      = clearingStepsAdv σ s n
        + (if 0 < (Board.fullRows GameConfig.standard
              (({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
                  with piece := s n } : Placement).place
                (adversarialTrace GameConfig.standard σ s GameState.init n).board)).card
           then 1 else 0) := rfl

/-- Adversarial per-step clear bound: a drop clearing `k` rows needs `10k`
cells present after it, of which the piece supplies four. -/
theorem clearAdv_step_le {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    10 * (clearedAdv GameConfig.standard σ s GameState.init (n + 1)
          - clearedAdv GameConfig.standard σ s GameState.init n)
      ≤ (adversarialTrace GameConfig.standard σ s GameState.init n).board.count + 4 := by
  have hwf := adversarialTrace_board_wf
    (GameState.init_board_wf GameConfig.standard) hv n
  have h := BagGrowth.count_clearLines_add_cols
    (Placement.place_wf hwf (hv n))
  rw [Placement.count_place, GameConfig.standard_cols] at h
  rw [clearedAdv_succ]
  omega

/-- Adversarial banked-mass ledger: clears are financed by standing inventory,
up to four cells of same-drop credit per clearing moment. -/
theorem ten_clearedAdv_le_sumCountAdv {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    10 * clearedAdv GameConfig.standard σ s GameState.init n
      ≤ sumCountAdv σ s n + 4 * clearingStepsAdv σ s n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := clearAdv_step_le hv k
    rw [clearedAdv_succ] at hstep
    rw [clearedAdv_succ, clearingStepsAdv_succ, sumCountAdv_succ]
    split_ifs with hc
    · omega
    · omega

/-- Adversarial clearing-frequency ceiling: at most two placements in five can
clear, whoever picks the pieces. -/
theorem clearingStepsAdv_le_clearedAdv (σ : Solver GameConfig.standard)
    (s : ℕ → Piece) (n : ℕ) :
    clearingStepsAdv σ s n
      ≤ clearedAdv GameConfig.standard σ s GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [clearingStepsAdv_succ, clearedAdv_succ]
    split_ifs with hc
    · omega
    · omega

theorem clearingStepsAdv_le {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    10 * clearingStepsAdv σ s n ≤ 4 * n := by
  have h1 := clearingStepsAdv_le_clearedAdv σ s n
  have h2 := adversary_cleared_le hv n
  omega

/-- **The standing-inventory floor, adversarially.** Against every legal piece
order, a live trace's time-averaged occupancy is at least `2.4 − 200/n` cells:
the adversary can neither starve the solver's inventory nor excuse it. -/
theorem adversary_standing_inventory_floor {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (adversarialTrace GameConfig.standard σ s GameState.init n).lost
      GameConfig.standard) :
    12 * n ≤ 5 * sumCountAdv σ s n + 1000 := by
  have h1 := ten_clearedAdv_le_sumCountAdv hv n
  have h3 := clearingStepsAdv_le hv n
  have hled := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n
  rw [GameConfig.standard_cols, GameState.init_board_count] at hled
  have hcap := BagGrowth.count_le_capacity
    (adversarialTrace_board_wf (GameState.init_board_wf GameConfig.standard) hv n)
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at hcap
  omega

/-- Every clearing moment of an adversarial trace sits on six banked cells. -/
theorem adversary_six_le_count_of_clearing {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hc : 0 < clearedAdv GameConfig.standard σ s GameState.init (n + 1)
          - clearedAdv GameConfig.standard σ s GameState.init n) :
    6 ≤ (adversarialTrace GameConfig.standard σ s GameState.init n).board.count := by
  have h := clearAdv_step_le hv n
  omega

/-- Adversarial trace boards never carry a full row: the initial board is empty
and every later board is a `clearLines` image. -/
theorem adversarialTrace_board_no_full {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (n r : ℕ) :
    ¬ Board.isFull cfg (adversarialTrace cfg σ s GameState.init n).board r := by
  cases n with
  | zero =>
    intro hfull
    have h0 := hfull 0 (Finset.mem_range.2 cfg.cols_pos)
    rw [adversarialTrace_zero] at h0
    exact GameState.init_board_no_mem _ h0
  | succ k =>
    rw [adversarialTrace_succ, adversarialStep_board]
    unfold Placement.applyStep
    exact Board.clearLines_no_full _ cfg.cols_pos r

/-- **A tetris identifies the adversary's piece.** If the drop at adversarial
step `n` completes four rows, the piece announced at step `n` was an I: the
tetris constraint reads the piece sequence directly off the clear log. -/
theorem adversary_tetris_step_I {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {n : ℕ}
    (h4 : 4 ≤ (Board.fullRows cfg
      (({ σ (adversarialTrace cfg σ s GameState.init n) (s n) with piece := s n }
          : Placement).place
        (adversarialTrace cfg σ s GameState.init n).board)).card) :
    s n = Piece.I := by
  have hI := tetris_requires_I
    (b := (adversarialTrace cfg σ s GameState.init n).board)
    (pl := { σ (adversarialTrace cfg σ s GameState.init n) (s n) with piece := s n })
    (fun r => adversarialTrace_board_no_full n r) h4
  exact hI

/-- **The board is empty at most a fifth of the time, adversarially.** The
occupancy residue clock (`adversarialTrace_count_mod_ten`) forces `5 ∣ n` at
every empty checkpoint, whoever picks the pieces. -/
theorem adversary_card_empty_times_le {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    5 * ((Finset.range n).filter (fun t =>
        (adversarialTrace GameConfig.standard σ s GameState.init t).board.count
          = 0)).card
      ≤ n + 4 := by
  classical
  set S := (Finset.range n).filter (fun t =>
    (adversarialTrace GameConfig.standard σ s GameState.init t).board.count = 0)
    with hS
  have hmod : ∀ t ∈ S, t % 5 = 0 := by
    intro t ht
    obtain ⟨htr, hzero⟩ := Finset.mem_filter.mp ht
    have h := adversarialTrace_count_mod_ten
      (GameState.init_board_wf GameConfig.standard) hv t
    rw [hzero, GameState.init_board_count] at h
    omega
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

/-- Tetris steps are I steps: the set of adversarial moves that clear four
rows embeds in the set of moves where the sequence dealt an I. -/
theorem adversary_tetris_steps_subset {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (a : ℕ) :
    (Finset.range 7).filter (fun k => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init (a + k)) (s (a + k))
            with piece := s (a + k) } : Placement).place
          (adversarialTrace cfg σ s GameState.init (a + k)).board)).card)
      ⊆ (Finset.range 7).filter (fun k => s (a + k) = Piece.I) := by
  intro k hk
  obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hk
  exact Finset.mem_filter.mpr ⟨h1, adversary_tetris_step_I h2⟩

/-- **At most two tetrises in any seven adversarial placements.** Four-row
clears require an I (`adversary_tetris_step_I`), and the 7-bag deals at most
two I's per seven draws (`window_same_piece_card_le_two`) — so tetris bursts
are capped at pairs on every window, a purely cadence-driven bound with no
board reasoning. -/
theorem adversary_two_tetris_per_seven {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (hl : LegalSequence s) (a : ℕ) :
    ((Finset.range 7).filter (fun k => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init (a + k)) (s (a + k))
            with piece := s (a + k) } : Placement).place
          (adversarialTrace cfg σ s GameState.init (a + k)).board)).card)).card
      ≤ 2 :=
  le_trans (Finset.card_le_card (adversary_tetris_steps_subset a))
    (BagCadence.window_same_piece_card_le_two hl a Piece.I)

/-! ## The clear-size mix, adversarially -/

/-- An adversarial drop clears at most four rows (the pre-drop board never
carries a full row). -/
theorem adversary_fullRows_card_le_four {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (n : ℕ) :
    (Board.fullRows cfg
      (({ σ (adversarialTrace cfg σ s GameState.init n) (s n) with piece := s n }
          : Placement).place
        (adversarialTrace cfg σ s GameState.init n).board)).card ≤ 4 := by
  have h := linesCleared_place_le_four cfg
    (adversarialTrace cfg σ s GameState.init n).board
    ({ σ (adversarialTrace cfg σ s GameState.init n) (s n) with piece := s n })
    (fun r => adversarialTrace_board_no_full n r)
  rwa [Board.linesCleared] at h

/-- **Triples require I, L or J — adversarially.** A drop clearing three or
more rows at adversarial step `n` forces the announced piece into `{I, L, J}`:
another read of the piece sequence off the clear log. -/
theorem adversary_three_clear_ILJ {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace cfg σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid cfg) {n : ℕ}
    (h3 : 3 ≤ (Board.fullRows cfg
      (({ σ (adversarialTrace cfg σ s GameState.init n) (s n) with piece := s n }
          : Placement).place
        (adversarialTrace cfg σ s GameState.init n).board)).card) :
    s n = Piece.I ∨ s n = Piece.L ∨ s n = Piece.J := by
  have h := three_clear_requires_I_L_or_J (hv n)
    (fun r => adversarialTrace_board_no_full n r) h3
  exact h

/-- Number of the first `n` adversarial drops that cleared exactly `k` rows. -/
def sizeCountAdv (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (k : ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
      sizeCountAdv cfg σ s k n
        + (if (Board.fullRows cfg
              (({ σ (adversarialTrace cfg σ s GameState.init n) (s n)
                  with piece := s n } : Placement).place
                (adversarialTrace cfg σ s GameState.init n).board)).card = k
           then 1 else 0)

@[simp] theorem sizeCountAdv_zero (cfg : GameConfig) (σ : Solver cfg)
    (s : ℕ → Piece) (k : ℕ) : sizeCountAdv cfg σ s k 0 = 0 := rfl

theorem sizeCountAdv_succ (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (k n : ℕ) :
    sizeCountAdv cfg σ s k (n + 1)
      = sizeCountAdv cfg σ s k n
        + (if (Board.fullRows cfg
              (({ σ (adversarialTrace cfg σ s GameState.init n) (s n)
                  with piece := s n } : Placement).place
                (adversarialTrace cfg σ s GameState.init n).board)).card = k
           then 1 else 0) := rfl

/-- **The mix identity, adversarially**: every cleared row belongs to exactly
one drop, and drops clear at most four rows. -/
theorem mix_identity_adv {cfg : GameConfig} {σ : Solver cfg} {s : ℕ → Piece}
    (n : ℕ) :
    sizeCountAdv cfg σ s 1 n + 2 * sizeCountAdv cfg σ s 2 n
        + 3 * sizeCountAdv cfg σ s 3 n + 4 * sizeCountAdv cfg σ s 4 n
      = clearedAdv cfg σ s GameState.init n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have h4 := adversary_fullRows_card_le_four (cfg := cfg) (σ := σ) (s := s) k
    rw [sizeCountAdv_succ, sizeCountAdv_succ, sizeCountAdv_succ,
      sizeCountAdv_succ, clearedAdv_succ]
    split_ifs <;> omega

/-- **The mix law, adversarially**: one linear equation in four unknowns —
the clear-size mix stays free against every piece order. -/
theorem adversary_mix_law {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    10 * (sizeCountAdv GameConfig.standard σ s 1 n
          + 2 * sizeCountAdv GameConfig.standard σ s 2 n
          + 3 * sizeCountAdv GameConfig.standard σ s 3 n
          + 4 * sizeCountAdv GameConfig.standard σ s 4 n)
        + (adversarialTrace GameConfig.standard σ s GameState.init n).board.count
      = 4 * n := by
  rw [mix_identity_adv]
  have h := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n
  rw [GameConfig.standard_cols, GameState.init_board_count] at h
  omega

/-- **Tetris steps embed in I steps, on any index set.** Generalises the
7-window form: over an arbitrary finite set of step indices, the moves that
clear four rows are among the moves where the sequence dealt an I — so any
window's tetris count is bounded by its I count. -/
theorem adversary_tetris_filter_subset {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (F : Finset ℕ) :
    F.filter (fun n => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init n) (s n)
            with piece := s n } : Placement).place
          (adversarialTrace cfg σ s GameState.init n).board)).card)
      ⊆ F.filter (fun n => s n = Piece.I) := by
  intro n hn
  obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hn
  exact Finset.mem_filter.mpr ⟨h1, adversary_tetris_step_I h2⟩

/-- Card form: the tetris count of any index window is at most its I count. -/
theorem adversary_tetris_card_le_I_card {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (F : Finset ℕ) :
    (F.filter (fun n => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init n) (s n)
            with piece := s n } : Placement).place
          (adversarialTrace cfg σ s GameState.init n).board)).card)).card
      ≤ (F.filter (fun n => s n = Piece.I)).card :=
  Finset.card_le_card (adversary_tetris_filter_subset F)

/-- The recursive size counter agrees with the windowed filter cardinality:
`sizeCountAdv k n` counts exactly the steps below `n` that cleared `k` rows. -/
theorem sizeCountAdv_eq_card_filter {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (k n : ℕ) :
    sizeCountAdv cfg σ s k n
      = ((Finset.range n).filter (fun m => (Board.fullRows cfg
          (({ σ (adversarialTrace cfg σ s GameState.init m) (s m)
              with piece := s m } : Placement).place
            (adversarialTrace cfg σ s GameState.init m).board)).card = k)).card := by
  classical
  induction n with
  | zero => simp
  | succ m ih =>
    rw [sizeCountAdv_succ, ih, Finset.range_add_one, Finset.filter_insert]
    split_ifs with h
    · rw [Finset.card_insert_of_notMem (by simp)]
    · omega

/-- **Adversarial tetris count is capped by the I supply.** The cumulative
number of four-row clears never exceeds the number of I's the sequence has
dealt: `sizeCountAdv 4 n ≤ #{m < n | s m = I}`. -/
theorem sizeCountAdv_four_le_I_card {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (n : ℕ) :
    sizeCountAdv cfg σ s 4 n
      ≤ ((Finset.range n).filter (fun m => s m = Piece.I)).card := by
  rw [sizeCountAdv_eq_card_filter]
  refine Finset.card_le_card ?_
  intro m hm
  obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hm
  exact Finset.mem_filter.mpr ⟨h1, adversary_tetris_step_I h2.ge⟩

/-- **Adversarial cycle windows balance exactly too**: between two equal
adversarial states, `10·Δcleared = 4·Δn` — every adversarial cycle period
clears exactly `14k` rows, whoever picks the pieces. -/
theorem adversarialTrace_eq_clears {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : adversarialTrace GameConfig.standard σ s GameState.init n₁
        = adversarialTrace GameConfig.standard σ s GameState.init n₂) :
    10 * (clearedAdv GameConfig.standard σ s GameState.init n₂
          - clearedAdv GameConfig.standard σ s GameState.init n₁)
      = 4 * (n₂ - n₁) := by
  have h1 := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n₁
  have h2 := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n₂
  rw [GameConfig.standard_cols, GameState.init_board_count] at h1 h2
  have hc : (adversarialTrace GameConfig.standard σ s GameState.init n₁).board.count
      = (adversarialTrace GameConfig.standard σ s GameState.init n₂).board.count := by
    rw [h]
  omega

/-- Adversarial size counters never decrease. -/
theorem sizeCountAdv_mono (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (k : ℕ) : Monotone (sizeCountAdv cfg σ s k) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [sizeCountAdv_succ]
  exact Nat.le_add_right _ _

/-- **The adversarial period mix**: over any 35-placement adversarial cycle
period, `Δa₁ + 2Δa₂ + 3Δa₃ + 4Δa₄ = 14`, whoever picks the pieces. -/
theorem adversary_period_mix_fourteen {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    (sizeCountAdv GameConfig.standard σ s 1 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 1 n)
      + 2 * (sizeCountAdv GameConfig.standard σ s 2 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 2 n)
      + 3 * (sizeCountAdv GameConfig.standard σ s 3 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 3 n)
      + 4 * (sizeCountAdv GameConfig.standard σ s 4 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 4 n)
      = 14 := by
  have h1 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ) (s := s) n
  have h2 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ) (s := s) (n + 35)
  have hbal := adversarialTrace_eq_clears hv (Nat.le_add_right n 35) hcyc
  have hm1 := sizeCountAdv_mono GameConfig.standard σ s 1 (Nat.le_add_right n 35)
  have hm2 := sizeCountAdv_mono GameConfig.standard σ s 2 (Nat.le_add_right n 35)
  have hm3 := sizeCountAdv_mono GameConfig.standard σ s 3 (Nat.le_add_right n 35)
  have hm4 := sizeCountAdv_mono GameConfig.standard σ s 4 (Nat.le_add_right n 35)
  omega

/-- **At most six tetrises per 35 adversarial placements**: four-row clears
require an I and any 35-window deals at most six I's. -/
theorem adversary_window_tetris_le_six {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} (hl : LegalSequence s) (a : ℕ) :
    ((Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init (a + k)) (s (a + k))
            with piece := s (a + k) } : Placement).place
          (adversarialTrace cfg σ s GameState.init (a + k)).board)).card)).card
      ≤ 6 := by
  have hsub : (Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows cfg
        (({ σ (adversarialTrace cfg σ s GameState.init (a + k)) (s (a + k))
            with piece := s (a + k) } : Placement).place
          (adversarialTrace cfg σ s GameState.init (a + k)).board)).card)
      ⊆ (Finset.range 35).filter (fun k => s (a + k) = Piece.I) := by
    intro k hk
    obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hk
    exact Finset.mem_filter.mpr ⟨h1, adversary_tetris_step_I h2⟩
  exact le_trans (Finset.card_le_card hsub)
    (BagCadence.window_thirtyfive_le_six hl a Piece.I)

/-- From any seed, adversarial trace boards after the first step carry no
full row. -/
theorem adversarialTrace_board_no_full_of_pos {cfg : GameConfig}
    {σ : Solver cfg} {s : ℕ → Piece} {g0 : GameState} {m : ℕ} (hm : 1 ≤ m)
    (r : ℕ) :
    ¬ Board.isFull cfg (adversarialTrace cfg σ s g0 m).board r := by
  obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
  rw [adversarialTrace_succ, adversarialStep_board]
  unfold Placement.applyStep
  exact Board.clearLines_no_full _ cfg.cols_pos r

/-- **At most five tetrises per adversarial cycle period** — the period deals
exactly five I's, and every tetris consumes one. -/
theorem adversarialClosedCycle_period_tetris_le_five
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    {t : ℕ → Piece} (hl : LegalSequenceFrom g0.bag t) {n : ℕ} (hn : 1 ≤ n)
    (hcyc : adversarialTrace GameConfig.standard C.solver t g0 n
        = adversarialTrace GameConfig.standard C.solver t g0 (n + 35)) :
    ((Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows GameConfig.standard
        (({ C.solver (adversarialTrace GameConfig.standard C.solver t g0 (n + k))
              (t (n + k)) with piece := t (n + k) } : Placement).place
          (adversarialTrace GameConfig.standard C.solver t g0 (n + k)).board)).card)).card
      ≤ 5 := by
  have hbal := adversarialClosedCycle_period_piece_balanced C hl hcyc Piece.I
  have hsub : (Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows GameConfig.standard
        (({ C.solver (adversarialTrace GameConfig.standard C.solver t g0 (n + k))
              (t (n + k)) with piece := t (n + k) } : Placement).place
          (adversarialTrace GameConfig.standard C.solver t g0 (n + k)).board)).card)
      ⊆ (Finset.range 35).filter (fun k => t (n + k) = Piece.I) := by
    intro k hk
    obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hk
    refine Finset.mem_filter.mpr ⟨h1, ?_⟩
    exact tetris_requires_I
      (fun r => adversarialTrace_board_no_full_of_pos (by omega : 1 ≤ n + k) r) h2
  calc ((Finset.range 35).filter _).card
      ≤ ((Finset.range 35).filter (fun k => t (n + k) = Piece.I)).card :=
        Finset.card_le_card hsub
    _ = 5 := hbal

/-- **At most five tetrises per cooperative cycle period** (windows past the
seed): the period plays exactly five I's. -/
theorem closedCycle_period_tetris_le_five (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ} (hn : 1 ≤ n)
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) :
    ((Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows GameConfig.standard
        ((C.policy (trace GameConfig.standard C.policy g0 (n + k))).place
          (trace GameConfig.standard C.policy g0 (n + k)).board)).card)).card
      ≤ 5 := by
  have hbal := closedCycle_period_piece_balanced C h0 hcyc Piece.I
  have hsub : (Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows GameConfig.standard
        ((C.policy (trace GameConfig.standard C.policy g0 (n + k))).place
          (trace GameConfig.standard C.policy g0 (n + k)).board)).card)
      ⊆ (Finset.range 35).filter (fun k =>
        (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
          = Piece.I) := by
    intro k hk
    obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hk
    exact Finset.mem_filter.mpr ⟨h1, trace_tetris_step_I (by omega) h2⟩
  calc ((Finset.range 35).filter _).card
      ≤ ((Finset.range 35).filter (fun k =>
          (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
            = Piece.I)).card := Finset.card_le_card hsub
    _ = 5 := hbal

/-- **At most three tetrises per adversarial cycle period** — fourteen rows
cannot absorb a fourth, whoever picks the pieces. Sharper than the five-I
supply. -/
theorem adversary_period_tetris_le_three {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    sizeCountAdv GameConfig.standard σ s 4 (n + 35)
      - sizeCountAdv GameConfig.standard σ s 4 n ≤ 3 := by
  have h := adversary_period_mix_fourteen hv hcyc
  omega

/-- Adversarial period caps for the other sizes: triples ≤ 4, doubles ≤ 7,
singles ≤ 14. -/
theorem adversary_period_size_caps {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    sizeCountAdv GameConfig.standard σ s 3 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 3 n ≤ 4
      ∧ sizeCountAdv GameConfig.standard σ s 2 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 2 n ≤ 7
      ∧ sizeCountAdv GameConfig.standard σ s 1 (n + 35)
        - sizeCountAdv GameConfig.standard σ s 1 n ≤ 14 := by
  have h := adversary_period_mix_fourteen hv hcyc
  omega

/-- **Adversarial periodicity needs a periodic stream**: a single 35-return
does *not* iterate on its own (the adversary may continue arbitrarily), but
if the announced stream is 35-periodic, determinism pushes the return around
the loop — the adversarial analog of `trace_eq_of_state_eq`. -/
theorem adversarialTrace_periodic {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState}
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + 35)) :
    ∀ k, adversarialTrace cfg σ s g0 (n + k)
      = adversarialTrace cfg σ s g0 (n + 35 + k) := by
  intro k
  induction k with
  | zero => simpa using hcyc
  | succ k ih =>
    have hs : s (n + 35 + k) = s (n + k) := by
      rw [show n + 35 + k = (n + k) + 35 by omega]
      exact hper (n + k)
    rw [show n + (k + 1) = (n + k) + 1 by omega,
      show n + 35 + (k + 1) = (n + 35 + k) + 1 by omega,
      adversarialTrace_succ, adversarialTrace_succ, ih, hs]

/-- Periodic-stream returns iterate to every multiple of the period. -/
theorem adversarialTrace_period_multiples {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState}
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + 35)) :
    ∀ j, adversarialTrace cfg σ s g0 n
      = adversarialTrace cfg σ s g0 (n + 35 * j) := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hstep := adversarialTrace_periodic hper hcyc (35 * j)
    rw [show n + 35 * (j + 1) = n + 35 + 35 * j by ring]
    exact ih.trans hstep

/-- **The adversarial linear clearing law**: against a 35-periodic piece
stream, a solver that returns after one period clears exactly `14·j` rows
over `j` periods — the cooperative multi-period law survives adversarial
piece choice whenever the loop witness is periodic (as any concrete
adversarial cycle certificate is). -/
theorem adversarial_multi_period_clears {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    (j : ℕ) :
    clearedAdv GameConfig.standard σ s GameState.init (n + 35 * j)
      - clearedAdv GameConfig.standard σ s GameState.init n = 14 * j := by
  have hiter := adversarialTrace_period_multiples hper hcyc j
  have hbal := adversarialTrace_eq_clears hv (Nat.le_add_right n (35 * j)) hiter
  omega

/-- **The adversarial multi-period mix**: against a 35-periodic stream, a
returning solver's clear-size increments weight-sum to exactly `14·j` over
`j` periods. -/
theorem adversarial_multi_period_mix {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    (j : ℕ) :
    (sizeCountAdv GameConfig.standard σ s 1 (n + 35 * j)
        - sizeCountAdv GameConfig.standard σ s 1 n)
      + 2 * (sizeCountAdv GameConfig.standard σ s 2 (n + 35 * j)
        - sizeCountAdv GameConfig.standard σ s 2 n)
      + 3 * (sizeCountAdv GameConfig.standard σ s 3 (n + 35 * j)
        - sizeCountAdv GameConfig.standard σ s 3 n)
      + 4 * (sizeCountAdv GameConfig.standard σ s 4 (n + 35 * j)
        - sizeCountAdv GameConfig.standard σ s 4 n)
      = 14 * j := by
  have h1 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ) (s := s) n
  have h2 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ) (s := s)
    (n + 35 * j)
  have hcl := adversarial_multi_period_clears hv hper hcyc j
  have hm1 := sizeCountAdv_mono GameConfig.standard σ s 1
    (Nat.le_add_right n (35 * j))
  have hm2 := sizeCountAdv_mono GameConfig.standard σ s 2
    (Nat.le_add_right n (35 * j))
  have hm3 := sizeCountAdv_mono GameConfig.standard σ s 3
    (Nat.le_add_right n (35 * j))
  have hm4 := sizeCountAdv_mono GameConfig.standard σ s 4
    (Nat.le_add_right n (35 * j))
  omega

/-- **At most `3·j` adversarial tetrises over `j` periods** — the telescoped
row budget survives adversarial piece choice under a periodic stream. -/
theorem adversarial_multi_period_tetris_le {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    ∀ j, sizeCountAdv GameConfig.standard σ s 4 (n + 35 * j)
      - sizeCountAdv GameConfig.standard σ s 4 n ≤ 3 * j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hj := adversarialTrace_period_multiples hper hcyc j
    have hj1 := adversarialTrace_period_multiples hper hcyc (j + 1)
    have hcycj : adversarialTrace GameConfig.standard σ s GameState.init
          (n + 35 * j)
        = adversarialTrace GameConfig.standard σ s GameState.init
          ((n + 35 * j) + 35) := by
      rw [show (n + 35 * j) + 35 = n + 35 * (j + 1) by ring]
      exact hj.symm.trans hj1
    have hstep := adversary_period_tetris_le_three hv hcycj
    have hmono := sizeCountAdv_mono GameConfig.standard σ s 4
      (Nat.le_add_right n (35 * j))
    rw [show n + 35 * (j + 1) = (n + 35 * j) + 35 by ring]
    omega

end ClearRate
end Tetris
