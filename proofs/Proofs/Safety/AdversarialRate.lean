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

/-- Adversarial cumulative clears never decrease. -/
theorem clearedAdv_mono (cfg : GameConfig) (σ : Solver cfg) (s : ℕ → Piece)
    (g0 : GameState) : Monotone (clearedAdv cfg σ s g0) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [clearedAdv_succ]
  exact Nat.le_add_right _ _

/-- **The adversarial clearing bracket**: against a 35-periodic stream, a
returning solver's cleared count stays within fourteen rows of the linear
2.8-per-bag law at *every* horizon. -/
theorem adversarial_clears_bracket {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m : ℕ} (hnm : n ≤ m) :
    14 * ((m - n) / 35)
        ≤ clearedAdv GameConfig.standard σ s GameState.init m
          - clearedAdv GameConfig.standard σ s GameState.init n
      ∧ clearedAdv GameConfig.standard σ s GameState.init m
          - clearedAdv GameConfig.standard σ s GameState.init n
        ≤ 14 * ((m - n) / 35) + 14 := by
  set j := (m - n) / 35 with hj
  have hlo : n + 35 * j ≤ m := by omega
  have hhi : m ≤ n + 35 * (j + 1) := by omega
  have hjlaw := adversarial_multi_period_clears hv hper hcyc j
  have hjlaw' := adversarial_multi_period_clears hv hper hcyc (j + 1)
  have hm1 := clearedAdv_mono GameConfig.standard σ s GameState.init hlo
  have hm2 := clearedAdv_mono GameConfig.standard σ s GameState.init hhi
  have hm0 := clearedAdv_mono GameConfig.standard σ s GameState.init
    (Nat.le_add_right n (35 * j))
  exact ⟨by omega, by omega⟩

/-- **The adversarial mass band**: against a 35-periodic stream, a returning
solver's board occupancy is trapped within a fourteen-row band of its
boundary value at every horizon — `count(n) − 140 ≤ count(m) ≤ count(n) + 136`.
The bounded-occupancy character of cycle play is adversary-proof. -/
theorem adversarial_mass_band {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m : ℕ} (hnm : n ≤ m) :
    (adversarialTrace GameConfig.standard σ s GameState.init m).board.count
        ≤ (adversarialTrace GameConfig.standard σ s GameState.init n).board.count
          + 136
      ∧ (adversarialTrace GameConfig.standard σ s GameState.init n).board.count
        ≤ (adversarialTrace GameConfig.standard σ s GameState.init m).board.count
          + 140 := by
  have hln := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv n
  have hlm := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard) hv m
  rw [GameConfig.standard_cols, GameState.init_board_count] at hln hlm
  obtain ⟨hlo, hhi⟩ := adversarial_clears_bracket hv hper hcyc hnm
  have hclm := clearedAdv_mono GameConfig.standard σ s GameState.init hnm
  exact ⟨by omega, by omega⟩

/-- **Silence dominates adversarially too**: of the 35 placements in an
adversarial cycle period, only between 4 and 14 clear anything, whoever
picks the pieces. -/
theorem adversary_period_clear_events_bounds {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    4 ≤ (sizeCountAdv GameConfig.standard σ s 1 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 1 n)
        + (sizeCountAdv GameConfig.standard σ s 2 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 2 n)
        + (sizeCountAdv GameConfig.standard σ s 3 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 3 n)
        + (sizeCountAdv GameConfig.standard σ s 4 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 4 n)
      ∧ (sizeCountAdv GameConfig.standard σ s 1 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 1 n)
        + (sizeCountAdv GameConfig.standard σ s 2 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 2 n)
        + (sizeCountAdv GameConfig.standard σ s 3 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 3 n)
        + (sizeCountAdv GameConfig.standard σ s 4 (n + 35)
          - sizeCountAdv GameConfig.standard σ s 4 n)
        ≤ 14 := by
  have h := adversary_period_mix_fourteen hv hcyc
  exact ⟨by omega, by omega⟩

/-- Every 69-window of a periodic-stream adversarial cycle clears at least
fourteen rows — the window contains a complete aligned period. -/
theorem adversarial_window_clears_fourteen {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) :
    14 ≤ clearedAdv GameConfig.standard σ s GameState.init (m₀ + 69)
      - clearedAdv GameConfig.standard σ s GameState.init m₀ := by
  set i := (m₀ - n + 34) / 35 with hi
  have hb1 : m₀ ≤ n + 35 * i := by omega
  have hlaw1 := adversarial_multi_period_clears hv hper hcyc i
  have hlaw2 := adversarial_multi_period_clears hv hper hcyc (i + 1)
  have hmono0 := clearedAdv_mono GameConfig.standard σ s GameState.init
    (Nat.le_add_right n (35 * i))
  have hmono1 := clearedAdv_mono GameConfig.standard σ s GameState.init hb1
  have hmono2 := clearedAdv_mono GameConfig.standard σ s GameState.init
    (show n + 35 * (i + 1) ≤ m₀ + 69 by omega)
  omega

/-- **Adversarial dry spells last at most 68 placements too**: under a
periodic stream, a returning solver clears within every 69-placement
stretch, whoever orders the pieces. -/
theorem adversarial_dry_spell_le {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ L : ℕ} (hm : n ≤ m₀)
    (hdry : clearedAdv GameConfig.standard σ s GameState.init (m₀ + L)
      = clearedAdv GameConfig.standard σ s GameState.init m₀) :
    L ≤ 68 := by
  by_contra hcon
  have h69 := adversarial_window_clears_fourteen hv hper hcyc hm
  have hmono := clearedAdv_mono GameConfig.standard σ s GameState.init
    (show m₀ + 69 ≤ m₀ + L by omega)
  have hmono0 := clearedAdv_mono GameConfig.standard σ s GameState.init
    (Nat.le_add_right m₀ 69)
  omega

/-- No pure-tetris adversarial cycle: `4 ∤ 14`, whoever picks the pieces. -/
theorem adversary_no_pure_tetris_period {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    (h1 : sizeCountAdv GameConfig.standard σ s 1 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 1 n)
    (h2 : sizeCountAdv GameConfig.standard σ s 2 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 2 n)
    (h3 : sizeCountAdv GameConfig.standard σ s 3 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 3 n) :
    False := by
  have h := adversary_period_mix_fourteen hv hcyc
  omega

/-- No pure-triple adversarial cycle: `3 ∤ 14`. -/
theorem adversary_no_pure_triple_period {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    (h1 : sizeCountAdv GameConfig.standard σ s 1 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 1 n)
    (h2 : sizeCountAdv GameConfig.standard σ s 2 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 2 n)
    (h4 : sizeCountAdv GameConfig.standard σ s 4 (n + 35)
      = sizeCountAdv GameConfig.standard σ s 4 n) :
    False := by
  have h := adversary_period_mix_fourteen hv hcyc
  omega

/-- The adversarial mass diameter: under a periodic stream, any two states of
a returning solver's trace differ by at most 276 cells of occupancy. -/
theorem adversarial_mass_diameter {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₁ m₂ : ℕ} (h1 : n ≤ m₁) (h2 : n ≤ m₂) :
    (adversarialTrace GameConfig.standard σ s GameState.init m₁).board.count
      ≤ (adversarialTrace GameConfig.standard σ s GameState.init m₂).board.count
        + 276 := by
  obtain ⟨hup1, hlo1⟩ := adversarial_mass_band hv hper hcyc h1
  obtain ⟨hup2, hlo2⟩ := adversarial_mass_band hv hper hcyc h2
  omega

/-- Adversarial tail periodicity, packaged: under a periodic stream, every
later index anchors the return. -/
theorem adversarialTrace_tail_periodic {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState}
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    adversarialTrace cfg σ s g0 m = adversarialTrace cfg σ s g0 (m + 35) := by
  have h := adversarialTrace_periodic hper hcyc (m - n)
  rw [show n + (m - n) = m by omega] at h
  rw [show n + 35 + (m - n) = m + 35 by omega] at h
  exact h

/-- The adversarial clearing bracket from every anchor:
`[14⌊w/35⌋, 14⌊w/35⌋ + 14]` in every window, whoever picks the pieces. -/
theorem adversarial_clears_bracket_stationary {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) (w : ℕ) :
    14 * (w / 35)
        ≤ clearedAdv GameConfig.standard σ s GameState.init (m₀ + w)
          - clearedAdv GameConfig.standard σ s GameState.init m₀
      ∧ clearedAdv GameConfig.standard σ s GameState.init (m₀ + w)
          - clearedAdv GameConfig.standard σ s GameState.init m₀
        ≤ 14 * (w / 35) + 14 := by
  have hbr := adversarial_clears_bracket hv hper
    (adversarialTrace_tail_periodic hper hcyc hm) (Nat.le_add_right m₀ w)
  rw [show m₀ + w - m₀ = w by omega] at hbr
  exact hbr

/-- The sharp adversarial mass diameter: `+136/−140` cells between any two
ordered horizons under a periodic stream. -/
theorem adversarial_mass_diameter_sharp {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₁ m₂ : ℕ} (h1 : n ≤ m₁) (h12 : m₁ ≤ m₂) :
    (adversarialTrace GameConfig.standard σ s GameState.init m₂).board.count
        ≤ (adversarialTrace GameConfig.standard σ s GameState.init m₁).board.count
          + 136
      ∧ (adversarialTrace GameConfig.standard σ s GameState.init m₁).board.count
        ≤ (adversarialTrace GameConfig.standard σ s GameState.init m₂).board.count
          + 140 :=
  adversarial_mass_band hv hper
    (adversarialTrace_tail_periodic hper hcyc h1) h12

/-- Every 35-window of a periodic-stream adversarial cycle clears exactly
fourteen rows, from any starting point. -/
theorem adversarial_window_clears_exact {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) :
    clearedAdv GameConfig.standard σ s GameState.init (m₀ + 35)
      - clearedAdv GameConfig.standard σ s GameState.init m₀ = 14 := by
  have h := adversarial_multi_period_clears hv hper
    (adversarialTrace_tail_periodic hper hcyc hm) 1
  simpa using h

/-- Adversarial dry spells last at most 34 placements — the sharp form. -/
theorem adversarial_dry_spell_le_thirtyfour {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ L : ℕ} (hm : n ≤ m₀)
    (hdry : clearedAdv GameConfig.standard σ s GameState.init (m₀ + L)
      = clearedAdv GameConfig.standard σ s GameState.init m₀) :
    L ≤ 34 := by
  by_contra hcon
  have hex := adversarial_window_clears_exact hv hper hcyc hm
  have hmono := clearedAdv_mono GameConfig.standard σ s GameState.init
    (show m₀ + 35 ≤ m₀ + L by omega)
  have hmono0 := clearedAdv_mono GameConfig.standard σ s GameState.init
    (Nat.le_add_right m₀ 35)
  omega

/-- Every 35-window's mix weight-sums to exactly fourteen, adversarially,
from any starting point under a periodic stream. -/
theorem adversarial_window_mix_stationary {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) :
    (sizeCountAdv GameConfig.standard σ s 1 (m₀ + 35)
        - sizeCountAdv GameConfig.standard σ s 1 m₀)
      + 2 * (sizeCountAdv GameConfig.standard σ s 2 (m₀ + 35)
        - sizeCountAdv GameConfig.standard σ s 2 m₀)
      + 3 * (sizeCountAdv GameConfig.standard σ s 3 (m₀ + 35)
        - sizeCountAdv GameConfig.standard σ s 3 m₀)
      + 4 * (sizeCountAdv GameConfig.standard σ s 4 (m₀ + 35)
        - sizeCountAdv GameConfig.standard σ s 4 m₀)
      = 14 :=
  adversary_period_mix_fourteen hv
    (adversarialTrace_tail_periodic hper hcyc hm)

/-- Every 35-window holds at most three adversarial tetrises, from any
starting point under a periodic stream. -/
theorem adversarial_window_tetris_le_three_stationary
    {σ : Solver GameConfig.standard} {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) :
    sizeCountAdv GameConfig.standard σ s 4 (m₀ + 35)
      - sizeCountAdv GameConfig.standard σ s 4 m₀ ≤ 3 :=
  adversary_period_tetris_le_three hv
    (adversarialTrace_tail_periodic hper hcyc hm)

/-- Every 35-window clears on 4–14 of its placements, adversarially, from
any starting point under a periodic stream. -/
theorem adversarial_window_events_stationary {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    {m₀ : ℕ} (hm : n ≤ m₀) :
    4 ≤ (sizeCountAdv GameConfig.standard σ s 1 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 1 m₀)
        + (sizeCountAdv GameConfig.standard σ s 2 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 2 m₀)
        + (sizeCountAdv GameConfig.standard σ s 3 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 3 m₀)
        + (sizeCountAdv GameConfig.standard σ s 4 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 4 m₀)
      ∧ (sizeCountAdv GameConfig.standard σ s 1 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 1 m₀)
        + (sizeCountAdv GameConfig.standard σ s 2 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 2 m₀)
        + (sizeCountAdv GameConfig.standard σ s 3 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 3 m₀)
        + (sizeCountAdv GameConfig.standard σ s 4 (m₀ + 35)
          - sizeCountAdv GameConfig.standard σ s 4 m₀)
        ≤ 14 :=
  adversary_period_clear_events_bounds hv
    (adversarialTrace_tail_periodic hper hcyc hm)

/-- Any 35-window of an adversarial trace shows at least five distinct
boards — the mass clock is adversary-proof. -/
theorem adversarialTrace_window_boards_ge_five {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s g0 n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    5 ≤ ((Finset.range 35).image
      (fun k => (adversarialTrace GameConfig.standard σ s g0 (n + k)).board)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 7 * ((Finset.range 35).image
      (fun k => (adversarialTrace GameConfig.standard σ s g0 (n + k)).board)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace GameConfig.standard σ s g0 (n + k)).board = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace GameConfig.standard σ s g0 (n + k)).board = a),
        i / 5 = k / 5 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have h1 := adversarialTrace_count_mod_ten hwf hv (n + i)
      have h2 := adversarialTrace_count_mod_ten hwf hv (n + k)
      rw [hi.2] at h1
      rw [hk.2] at h2
      omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace GameConfig.standard σ s g0 (n + k)).board = a),
        i / 5 ∈ Finset.range 7 := by
      intro i hi
      rw [Finset.mem_filter, Finset.mem_range] at hi
      rw [Finset.mem_range]
      omega
    have := Finset.card_le_card_of_injOn (fun i => i / 5) hmap hinj
    rw [Finset.card_range] at this
    exact this
  rw [Finset.card_range] at hle
  omega

/-- Any 35-window of a legally-drawn adversarial trace shows at least seven
distinct bag states — the bag clock is adversary-proof too. -/
theorem adversarialTrace_window_bags_ge_seven {cfg : GameConfig}
    {σ : Solver cfg} {s : ℕ → Piece} {g0 : GameState}
    (hdraw : ∀ n, s n ∈ (adversarialTrace cfg σ s g0 n).bag) (n : ℕ) :
    7 ≤ ((Finset.range 35).image
      (fun k => (adversarialTrace cfg σ s g0 (n + k)).bag)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 5 * ((Finset.range 35).image
      (fun k => (adversarialTrace cfg σ s g0 (n + k)).bag)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace cfg σ s g0 (n + k)).bag = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace cfg σ s g0 (n + k)).bag = a),
        i / 7 = k / 7 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have h1 := bag_card_adversarialTrace hdraw (n + i)
      have h2 := bag_card_adversarialTrace hdraw (n + k)
      rw [hi.2] at h1
      rw [hk.2] at h2
      have hle7 : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
      have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
      omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (adversarialTrace cfg σ s g0 (n + k)).bag = a),
        i / 7 ∈ Finset.range 5 := by
      intro i hi
      rw [Finset.mem_filter, Finset.mem_range] at hi
      rw [Finset.mem_range]
      omega
    have := Finset.card_le_card_of_injOn (fun i => i / 7) hmap hinj
    rw [Finset.card_range] at this
    exact this
  rw [Finset.card_range] at hle
  omega

/-- **A closed Atlas set spans at least five distinct boards**: the diversity
floor of any legal trace window transfers to the M4 witness set. -/
theorem isClosedOn_boards_ge_five {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) (hbag : g₀.bag.Nonempty) :
    5 ≤ (S.image (fun g => g.board)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hv := isClosedOn_trace_forced_valid h hg₀ ht
  have hwin := adversarialTrace_window_boards_ge_five
    (σ := A.toSolver) (s := t) (g0 := g₀) hwf hv 0
  refine le_trans hwin (Finset.card_le_card ?_)
  intro b hb
  rw [Finset.mem_image] at hb ⊢
  obtain ⟨k, -, rfl⟩ := hb
  exact ⟨_, h.toSolver_adversarialTrace_mem hg₀ ht (0 + k), rfl⟩

/-- **A closed Atlas set spans at least seven distinct bag states** — it must
carry a full bag-clock cycle. -/
theorem isClosedOn_bags_ge_seven {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S) (hbag : g₀.bag.Nonempty) :
    7 ≤ (S.image (fun g => g.bag)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard A.toSolver t g₀ n).bag :=
    fun n => by rw [adversarialTrace_bag_from]; exact ht n
  have hwin := adversarialTrace_window_bags_ge_seven hdraw 0
  refine le_trans hwin (Finset.card_le_card ?_)
  intro b hb
  rw [Finset.mem_image] at hb ⊢
  obtain ⟨k, -, rfl⟩ := hb
  exact ⟨_, h.toSolver_adversarialTrace_mem hg₀ ht (0 + k), rfl⟩

/-- The M4 witness diversity: any init-containing closed Atlas spans at least
five boards and seven bag states. -/
theorem init_closed_atlas_diversity {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    (hinit : GameState.init ∈ S) :
    5 ≤ (S.image (fun g => g.board)).card
      ∧ 7 ≤ (S.image (fun g => g.bag)).card :=
  ⟨isClosedOn_boards_ge_five h hinit
      (GameState.init_board_wf GameConfig.standard) GameState.init_bag_nonempty,
    isClosedOn_bags_ge_seven h hinit GameState.init_bag_nonempty⟩

/-- The five-board diversity floor from any well-formed seed (not just
`init`): the mass clock runs on the general ledger. -/
theorem trace_window_boards_ge_five_from {π : Policy GameConfig.standard}
    {g0 : GameState}
    (hv : ∀ k, (π (trace GameConfig.standard π g0 k)).Valid GameConfig.standard)
    (hwf : Board.WF GameConfig.standard g0.board) (n : ℕ) :
    5 ≤ ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π g0 (n + k)).board)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 7 * ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π g0 (n + k)).board)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π g0 (n + k)).board = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π g0 (n + k)).board = a),
        i / 5 = k / 5 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have hcnt : (trace GameConfig.standard π g0 (n + i)).board.count
          = (trace GameConfig.standard π g0 (n + k)).board.count := by
        rw [hi.2, hk.2]
      rcases Nat.lt_or_ge k i with hik | hik
      · have hd := five_dvd_of_count_eq_from hv hwf
          (show n + k ≤ n + i by omega) hcnt.symm
        omega
      · have hd := five_dvd_of_count_eq_from hv hwf
          (show n + i ≤ n + k by omega) hcnt
        omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π g0 (n + k)).board = a),
        i / 5 ∈ Finset.range 7 := by
      intro i hi
      rw [Finset.mem_filter, Finset.mem_range] at hi
      rw [Finset.mem_range]
      omega
    have := Finset.card_le_card_of_injOn (fun i => i / 5) hmap hinj
    rw [Finset.card_range] at this
    exact this
  rw [Finset.card_range] at hle
  omega

/-- The seven-bag diversity floor from any legally-drawn seed. -/
theorem trace_window_bags_ge_seven_from {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace cfg π g0 k)).piece ∈ (trace cfg π g0 k).bag)
    (n : ℕ) :
    7 ≤ ((Finset.range 35).image
      (fun k => (trace cfg π g0 (n + k)).bag)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 5 * ((Finset.range 35).image
      (fun k => (trace cfg π g0 (n + k)).bag)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace cfg π g0 (n + k)).bag = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (trace cfg π g0 (n + k)).bag = a),
        i / 7 = k / 7 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have h1 := bag_card_trace_from hdraw (n + i)
      have h2 := bag_card_trace_from hdraw (n + k)
      rw [hi.2] at h1
      rw [hk.2] at h2
      have hle7 : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
      have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
      omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace cfg π g0 (n + k)).bag = a),
        i / 7 ∈ Finset.range 5 := by
      intro i hi
      rw [Finset.mem_filter, Finset.mem_range] at hi
      rw [Finset.mem_range]
      omega
    have := Finset.card_le_card_of_injOn (fun i => i / 7) hmap hinj
    rw [Finset.card_range] at this
    exact this
  rw [Finset.card_range] at hle
  omega

/-- **The M2 artifact spans five boards**: every closed cycle's state set
shows at least five distinct boards. -/
theorem closedCycle_boards_ge_five (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (hg0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) :
    5 ≤ (C.states.image (fun g => g.board)).card := by
  classical
  have hv : ∀ k, (C.policy (trace GameConfig.standard C.policy g0 k)).Valid
      GameConfig.standard :=
    fun k => C.valid _ (C.trace_mem_states hg0 k)
  have hwin := trace_window_boards_ge_five_from hv hwf 0
  refine le_trans hwin (Finset.card_le_card ?_)
  intro b hb
  rw [Finset.mem_image] at hb ⊢
  obtain ⟨k, -, rfl⟩ := hb
  exact ⟨_, C.trace_mem_states hg0 (0 + k), rfl⟩

/-- **The M2 artifact spans seven bag states** — a full bag-clock cycle. -/
theorem closedCycle_bags_ge_seven (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (hg0 : g0 ∈ C.states) :
    7 ≤ (C.states.image (fun g => g.bag)).card := by
  classical
  have hdraw : ∀ k, (C.policy (trace GameConfig.standard C.policy g0 k)).piece
      ∈ (trace GameConfig.standard C.policy g0 k).bag :=
    fun k => C.legal_draw _ (C.trace_mem_states hg0 k)
  have hwin := trace_window_bags_ge_seven_from hdraw 0
  refine le_trans hwin (Finset.card_le_card ?_)
  intro b hb
  rw [Finset.mem_image] at hb ⊢
  obtain ⟨k, -, rfl⟩ := hb
  exact ⟨_, C.trace_mem_states hg0 (0 + k), rfl⟩

/-- The adversarial M2 artifact spans five boards and seven bag states. -/
theorem adversarialClosedCycle_diversity
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    (hg0 : g0 ∈ C.states) (hwf : Board.WF GameConfig.standard g0.board)
    (hbag : g0.bag.Nonempty) :
    5 ≤ (C.states.image (fun g => g.board)).card
      ∧ 7 ≤ (C.states.image (fun g => g.bag)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard C.solver t g0 n).bag := by
    intro n
    have hn := ht n
    rw [Bag.canDraw_iff_mem] at hn
    rw [adversarialTrace_bag_from]
    exact hn
  have hv : ∀ n, ({ C.solver (adversarialTrace GameConfig.standard C.solver t
      g0 n) (t n) with piece := t n } : Placement).Valid GameConfig.standard := by
    intro n
    obtain ⟨hp, hval⟩ :=
      C.valid _ (C.adversarialTrace_mem_states_from_mem hg0 ht n) (t n) (hdraw n)
    rw [placement_with_piece_self hp]
    exact hval
  constructor
  · have hwin := adversarialTrace_window_boards_ge_five
      (σ := C.solver) (s := t) (g0 := g0) hwf hv 0
    refine le_trans hwin (Finset.card_le_card ?_)
    intro b hb
    rw [Finset.mem_image] at hb ⊢
    obtain ⟨k, -, rfl⟩ := hb
    exact ⟨_, C.adversarialTrace_mem_states_from_mem hg0 ht (0 + k), rfl⟩
  · have hwin := adversarialTrace_window_bags_ge_seven hdraw 0
    refine le_trans hwin (Finset.card_le_card ?_)
    intro b hb
    rw [Finset.mem_image] at hb ⊢
    obtain ⟨k, -, rfl⟩ := hb
    exact ⟨_, C.adversarialTrace_mem_states_from_mem hg0 ht (0 + k), rfl⟩

set_option maxRecDepth 4000 in
/-- **The adversarial idle-I law**: even choosing the piece order, an
adversary facing a returning solver sees at least two of each period's five
I's do lesser work than a tetris — the row budget is order-independent. -/
theorem adversary_period_idle_I_ge_two {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hdraw : ∀ n, s n
      ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35)) :
    2 ≤ ((Finset.range 35).filter (fun k =>
        s (n + k) = Piece.I
          ∧ (Board.fullRows GameConfig.standard
              (({ σ (adversarialTrace GameConfig.standard σ s GameState.init
                    (n + k)) (s (n + k)) with piece := s (n + k) }
                : Placement).place
                (adversarialTrace GameConfig.standard σ s GameState.init
                  (n + k)).board)).card ≠ 4)).card := by
  classical
  -- the announced stream is legal from the initial bag
  have hl : LegalSequenceFrom GameState.init.bag s := by
    intro m
    have := hdraw m
    rwa [adversarialTrace_bag_from] at this
  -- bags at the return endpoints agree
  have hbag : bagAt GameState.init.bag s (n + 35)
      = bagAt GameState.init.bag s n := by
    rw [← adversarialTrace_bag_from GameConfig.standard σ s GameState.init,
      ← adversarialTrace_bag_from GameConfig.standard σ s GameState.init, hcyc]
  have hbal := BagCadence.window_thirtyfive_balanced hl hbag Piece.I
  -- split the five I's by whether they clear four rows
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := (Finset.range 35).filter (fun k => s (n + k) = Piece.I))
    (p := fun k => (Board.fullRows GameConfig.standard
      (({ σ (adversarialTrace GameConfig.standard σ s GameState.init (n + k))
          (s (n + k)) with piece := s (n + k) } : Placement).place
        (adversarialTrace GameConfig.standard σ s GameState.init
          (n + k)).board)).card = 4)
  rw [Finset.filter_filter, Finset.filter_filter, hbal] at hsplit
  -- the (I ∧ 4-clear) fiber is exactly the tetris fiber
  have hIfour : ((Finset.range 35).filter (fun k =>
        s (n + k) = Piece.I
          ∧ (Board.fullRows GameConfig.standard
              (({ σ (adversarialTrace GameConfig.standard σ s GameState.init
                    (n + k)) (s (n + k)) with piece := s (n + k) }
                : Placement).place
                (adversarialTrace GameConfig.standard σ s GameState.init
                  (n + k)).board)).card = 4)).card
      = ((Finset.range 35).filter (fun k =>
        (Board.fullRows GameConfig.standard
          (({ σ (adversarialTrace GameConfig.standard σ s GameState.init
                (n + k)) (s (n + k)) with piece := s (n + k) }
            : Placement).place
            (adversarialTrace GameConfig.standard σ s GameState.init
              (n + k)).board)).card = 4)).card := by
    refine congrArg Finset.card (Finset.filter_congr ?_)
    intro k hk
    constructor
    · intro h
      exact h.2
    · intro h
      exact ⟨adversary_tetris_step_I (le_of_eq h.symm), h⟩
  -- the tetris fiber is the counter increment, capped at three
  have h1 := sizeCountAdv_eq_card_filter (cfg := GameConfig.standard)
    (σ := σ) (s := s) 4 n
  have h2 := sizeCountAdv_eq_card_filter (cfg := GameConfig.standard)
    (σ := σ) (s := s) 4 (n + 35)
  have hspl := card_filter_range_add (fun m => (Board.fullRows GameConfig.standard
    (({ σ (adversarialTrace GameConfig.standard σ s GameState.init m) (s m)
        with piece := s m } : Placement).place
      (adversarialTrace GameConfig.standard σ s GameState.init m).board)).card
      = 4) n 35
  have htet := adversary_period_tetris_le_three hv hcyc
  simp only [ne_eq] at hsplit hIfour ⊢
  dsimp only at h1 h2 hspl hsplit hIfour ⊢
  omega

/-- **The stratified floor**: a closed Atlas set holds at least five states
at *every* bag fill level `c ∈ {1, …, 7}` — the 35-state floor decomposed
into its seven bag strata of five. The greedy trace passes through the
target stratum once per bag, and the quantum keeps those visits distinct. -/
theorem isClosedOn_stratum_ge_five {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) (hbag : g₀.bag.Nonempty)
    {c : ℕ} (hc1 : 1 ≤ c) (hc7 : c ≤ 7) :
    5 ≤ (S.filter (fun s => s.bag.card = c)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard A.toSolver t g₀ n).bag :=
    fun n => by rw [adversarialTrace_bag_from]; exact ht n
  have hcard := bag_card_adversarialTrace hdraw
  have hle7 : g₀.bag.card ≤ 7 := Bag.card_le_seven g₀.bag
  have hpos : 0 < g₀.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  -- the residue r < 7 whose indices carry bag card c
  set r := (7 - c + g₀.bag.card) % 7 with hr
  have hrc : ∀ j, (adversarialTrace GameConfig.standard A.toSolver t g₀
      (r + 7 * j)).bag.card = c := by
    intro j
    rw [hcard]
    omega
  refine le_trans ?_ (Finset.card_le_card_of_injOn
    (s := Finset.range 5)
    (fun j => adversarialTrace GameConfig.standard A.toSolver t g₀ (r + 7 * j))
    ?_ ?_)
  · rw [Finset.card_range]
  · intro j hj
    exact Finset.mem_filter.mpr
      ⟨h.toSolver_adversarialTrace_mem hg₀ ht (r + 7 * j), hrc j⟩
  · intro i hi j hj hEq
    rw [Finset.coe_range, Set.mem_Iio] at hi hj
    rcases Nat.lt_or_ge i j with hij | hij
    · have hd := isClosedOn_thirtyfive_dvd h hg₀ hwf ht
        (show r + 7 * i ≤ r + 7 * j by omega) hEq
      omega
    · rcases Nat.eq_or_lt_of_le hij with heq | hij'
      · omega
      · have hd := isClosedOn_thirtyfive_dvd h hg₀ hwf ht
          (show r + 7 * j ≤ r + 7 * i by omega) hEq.symm
        omega

/-- The stratified floor on the cooperative M2 artifact: at least five
states at every bag fill level. -/
theorem closedCycle_stratum_ge_five (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (hg0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) {c : ℕ}
    (hc1 : 1 ≤ c) (hc7 : c ≤ 7) :
    5 ≤ (C.states.filter (fun s => s.bag.card = c)).card := by
  classical
  have hdraw : ∀ k, (C.policy (trace GameConfig.standard C.policy g0 k)).piece
      ∈ (trace GameConfig.standard C.policy g0 k).bag :=
    fun k => C.legal_draw _ (C.trace_mem_states hg0 k)
  have hcard := bag_card_trace_from hdraw
  have hle7 : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  set r := (7 - c + g0.bag.card) % 7 with hr
  have hrc : ∀ j, (trace GameConfig.standard C.policy g0 (r + 7 * j)).bag.card
      = c := by
    intro j
    rw [hcard]
    omega
  refine le_trans ?_ (Finset.card_le_card_of_injOn
    (s := Finset.range 5)
    (fun j => trace GameConfig.standard C.policy g0 (r + 7 * j)) ?_ ?_)
  · rw [Finset.card_range]
  · intro j hj
    exact Finset.mem_filter.mpr ⟨C.trace_mem_states hg0 (r + 7 * j), hrc j⟩
  · intro i hi j hj hEq
    rw [Finset.coe_range, Set.mem_Iio] at hi hj
    rcases Nat.lt_or_ge i j with hij | hij
    · have hd := closedCycle_thirtyfive_dvd C hg0 hwf
        (show r + 7 * i ≤ r + 7 * j by omega) hEq
      omega
    · rcases Nat.eq_or_lt_of_le hij with heq | hij'
      · omega
      · have hd := closedCycle_thirtyfive_dvd C hg0 hwf
          (show r + 7 * j ≤ r + 7 * i by omega) hEq.symm
        omega

/-- The stratified floor on the adversarial M2 artifact. -/
theorem adversarialClosedCycle_stratum_ge_five
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    (hg0 : g0 ∈ C.states) (hwf : Board.WF GameConfig.standard g0.board)
    (hbag : g0.bag.Nonempty) {c : ℕ} (hc1 : 1 ≤ c) (hc7 : c ≤ 7) :
    5 ≤ (C.states.filter (fun s => s.bag.card = c)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard C.solver t g0 n).bag := by
    intro n
    have hn := ht n
    rw [Bag.canDraw_iff_mem] at hn
    rw [adversarialTrace_bag_from]
    exact hn
  have hcard := bag_card_adversarialTrace hdraw
  have hle7 : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  set r := (7 - c + g0.bag.card) % 7 with hr
  have hrc : ∀ j, (adversarialTrace GameConfig.standard C.solver t g0
      (r + 7 * j)).bag.card = c := by
    intro j
    rw [hcard]
    omega
  refine le_trans ?_ (Finset.card_le_card_of_injOn
    (s := Finset.range 5)
    (fun j => adversarialTrace GameConfig.standard C.solver t g0 (r + 7 * j))
    ?_ ?_)
  · rw [Finset.card_range]
  · intro j hj
    exact Finset.mem_filter.mpr
      ⟨C.adversarialTrace_mem_states_from_mem hg0 ht (r + 7 * j), hrc j⟩
  · intro i hi j hj hEq
    rw [Finset.coe_range, Set.mem_Iio] at hi hj
    rcases Nat.lt_or_ge i j with hij | hij
    · have hd := adversarialClosedCycle_thirtyfive_dvd C hg0 hwf ht
        (show r + 7 * i ≤ r + 7 * j by omega) hEq
      omega
    · rcases Nat.eq_or_lt_of_le hij with heq | hij'
      · omega
      · have hd := adversarialClosedCycle_thirtyfive_dvd C hg0 hwf ht
          (show r + 7 * j ≤ r + 7 * i by omega) hEq.symm
        omega

/-- **The CRT grid**: a closed Atlas set inhabits *every* cell of the
7 × 5 grid (bag fill level × mass phase) — for each bag level `c` and each
of the five mass residues the trajectory carries, some state realises both
simultaneously. The 35-state floor is exactly this grid, by the Chinese
remainders of the two clocks. -/
theorem isClosedOn_grid_inhabited {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) (hbag : g₀.bag.Nonempty)
    {c : ℕ} (hc1 : 1 ≤ c) (hc7 : c ≤ 7) {i : ℕ} (hi : i < 5) :
    ∃ s ∈ S, s.bag.card = c
      ∧ s.board.count % 10 = (g₀.board.count + 4 * i) % 10 := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard A.toSolver t g₀ n).bag :=
    fun n => by rw [adversarialTrace_bag_from]; exact ht n
  have hv := isClosedOn_trace_forced_valid h hg₀ ht
  have hbagcard := bag_card_adversarialTrace hdraw
  have hcnt := adversarialTrace_count_mod_ten hwf hv
  have hle7 : g₀.bag.card ≤ 7 := Bag.card_le_seven g₀.bag
  have hpos : 0 < g₀.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  -- the bag-level residue and the CRT-matched block index
  set r := (7 - c + g₀.bag.card) % 7 with hr
  set j := (3 * (i + 5 - r % 5)) % 5 with hj
  refine ⟨adversarialTrace GameConfig.standard A.toSolver t g₀ (r + 7 * j),
    h.toSolver_adversarialTrace_mem hg₀ ht (r + 7 * j), ?_, ?_⟩
  · rw [hbagcard]
    omega
  · rw [hcnt]
    omega

/-- **The adversarial tetris frequency cap**: at most `⌊Δn/7⌋ + 2` tetrises
in any window, whoever picks the pieces — every four-clear reads `I` off the
announced stream, and the stream obeys the frequency law. -/
theorem adversary_tetris_frequency_cap {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hdraw : ∀ n, s n
      ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag)
    {n m : ℕ} (hnm : n ≤ m) :
    sizeCountAdv GameConfig.standard σ s 4 m
        - sizeCountAdv GameConfig.standard σ s 4 n
      ≤ (m - n) / 7 + 2 := by
  classical
  have hl : LegalSequenceFrom GameState.init.bag s := by
    intro k
    have := hdraw k
    rwa [adversarialTrace_bag_from] at this
  have h1 := sizeCountAdv_eq_card_filter (cfg := GameConfig.standard)
    (σ := σ) (s := s) 4 n
  have h2 := sizeCountAdv_eq_card_filter (cfg := GameConfig.standard)
    (σ := σ) (s := s) 4 (m)
  have hsplit := card_filter_range_add (fun k => (Board.fullRows
    GameConfig.standard
    (({ σ (adversarialTrace GameConfig.standard σ s GameState.init k) (s k)
        with piece := s k } : Placement).place
      (adversarialTrace GameConfig.standard σ s GameState.init k).board)).card
      = 4) n (m - n)
  rw [show n + (m - n) = m by omega] at hsplit
  have hsub : ((Finset.range (m - n)).filter (fun j => (Board.fullRows
        GameConfig.standard
        (({ σ (adversarialTrace GameConfig.standard σ s GameState.init (n + j))
            (s (n + j)) with piece := s (n + j) } : Placement).place
          (adversarialTrace GameConfig.standard σ s GameState.init
            (n + j)).board)).card = 4)).card
      ≤ ((Finset.range (m - n)).filter
          (fun j => s (n + j) = Piece.I)).card := by
    apply Finset.card_le_card
    intro j hj
    rw [Finset.mem_filter] at hj ⊢
    exact ⟨hj.1, adversary_tetris_step_I (le_of_eq hj.2.symm)⟩
  have hwin := BagCadence.window_frequency_law hl n Piece.I (m - n)
  omega

/-- The adversarial live-window capacity bound on occupancy. -/
theorem adversarialTrace_count_lt {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard) {n : ℕ}
    (hlive : ¬ (adversarialTrace GameConfig.standard σ s GameState.init n).lost
      GameConfig.standard) :
    (adversarialTrace GameConfig.standard σ s GameState.init n).board.count
      < 201 := by
  have h := BagGrowth.count_le_capacity
    (adversarialTrace_board_wf (GameState.init_board_wf GameConfig.standard)
      hv n)
    ((GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp hlive)
  rw [GameConfig.standard_cols, GameConfig.standard_rows] at h
  omega

/-- **The adversarial survivor's clearing bracket, windowed**: on any window
of a live adversarial trace, `4w − 200 ≤ 10·ΔclearedAdv ≤ 4w + 200` —
the 2.8-per-bag law with one-boardful slack, whoever picks the pieces. -/
theorem adversary_survivor_window_clears {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    {n w : ℕ}
    (hlive_n : ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      n).lost GameConfig.standard)
    (hlive_m : ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      (n + w)).lost GameConfig.standard) :
    4 * w ≤ 10 * (clearedAdv GameConfig.standard σ s GameState.init (n + w)
        - clearedAdv GameConfig.standard σ s GameState.init n) + 200
      ∧ 10 * (clearedAdv GameConfig.standard σ s GameState.init (n + w)
        - clearedAdv GameConfig.standard σ s GameState.init n)
        ≤ 4 * w + 200 := by
  have h1 := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard)
    hv n
  have h2 := clearedAdv_ledger (GameState.init_board_wf GameConfig.standard)
    hv (n + w)
  rw [GameConfig.standard_cols, GameState.init_board_count] at h1 h2
  have hcap1 := adversarialTrace_count_lt hv hlive_n
  have hcap2 := adversarialTrace_count_lt hv hlive_m
  have hmono := clearedAdv_mono GameConfig.standard σ s GameState.init
    (Nat.le_add_right n w)
  exact ⟨by omega, by omega⟩

/-- The adversarial event-rate bracket: clearing events on a live window
number between `(4w − 200)/40` and `(4w + 200)/10`. -/
theorem adversary_survivor_window_events {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    {n w : ℕ}
    (hlive_n : ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      n).lost GameConfig.standard)
    (hlive_m : ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      (n + w)).lost GameConfig.standard) :
    4 * w ≤ 40 * ((sizeCountAdv GameConfig.standard σ s 1 (n + w)
          - sizeCountAdv GameConfig.standard σ s 1 n)
        + (sizeCountAdv GameConfig.standard σ s 2 (n + w)
          - sizeCountAdv GameConfig.standard σ s 2 n)
        + (sizeCountAdv GameConfig.standard σ s 3 (n + w)
          - sizeCountAdv GameConfig.standard σ s 3 n)
        + (sizeCountAdv GameConfig.standard σ s 4 (n + w)
          - sizeCountAdv GameConfig.standard σ s 4 n)) + 200
      ∧ 10 * ((sizeCountAdv GameConfig.standard σ s 1 (n + w)
          - sizeCountAdv GameConfig.standard σ s 1 n)
        + (sizeCountAdv GameConfig.standard σ s 2 (n + w)
          - sizeCountAdv GameConfig.standard σ s 2 n)
        + (sizeCountAdv GameConfig.standard σ s 3 (n + w)
          - sizeCountAdv GameConfig.standard σ s 3 n)
        + (sizeCountAdv GameConfig.standard σ s 4 (n + w)
          - sizeCountAdv GameConfig.standard σ s 4 n))
        ≤ 4 * w + 200 := by
  obtain ⟨hfl, hce⟩ := adversary_survivor_window_clears hv hlive_n hlive_m
  have hm1 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ)
    (s := s) n
  have hm2 := mix_identity_adv (cfg := GameConfig.standard) (σ := σ)
    (s := s) (n + w)
  have hs1 := sizeCountAdv_mono GameConfig.standard σ s 1
    (Nat.le_add_right n w)
  have hs2 := sizeCountAdv_mono GameConfig.standard σ s 2
    (Nat.le_add_right n w)
  have hs3 := sizeCountAdv_mono GameConfig.standard σ s 3
    (Nat.le_add_right n w)
  have hs4 := sizeCountAdv_mono GameConfig.standard σ s 4
    (Nat.le_add_right n w)
  exact ⟨by omega, by omega⟩

/-- **The mass-phase stratification**: a closed Atlas set holds at least
seven states at every mass phase the trajectory carries — the dual of the
bag stratification, completing the 35 = 7 × 5 grid decomposition from the
other axis. -/
theorem isClosedOn_count_stratum_ge_seven {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) (hbag : g₀.bag.Nonempty)
    (i : ℕ) :
    7 ≤ (S.filter (fun g => g.board.count % 10
      = (g₀.board.count + 4 * i) % 10)).card := by
  classical
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hdraw : ∀ n, t n
      ∈ (adversarialTrace GameConfig.standard A.toSolver t g₀ n).bag :=
    fun n => by rw [adversarialTrace_bag_from]; exact ht n
  have hv := isClosedOn_trace_forced_valid h hg₀ ht
  have hcnt := adversarialTrace_count_mod_ten hwf hv
  have hrc : ∀ j, (adversarialTrace GameConfig.standard A.toSolver t g₀
      (i + 5 * j)).board.count % 10 = (g₀.board.count + 4 * i) % 10 := by
    intro j
    rw [hcnt]
    omega
  refine le_trans ?_ (Finset.card_le_card_of_injOn
    (s := Finset.range 7)
    (fun j => adversarialTrace GameConfig.standard A.toSolver t g₀ (i + 5 * j))
    ?_ ?_)
  · rw [Finset.card_range]
  · intro j hj
    exact Finset.mem_filter.mpr
      ⟨h.toSolver_adversarialTrace_mem hg₀ ht (i + 5 * j), hrc j⟩
  · intro a ha b hb hEq
    rw [Finset.coe_range, Set.mem_Iio] at ha hb
    rcases Nat.lt_or_ge a b with hab | hab
    · have hd := isClosedOn_thirtyfive_dvd h hg₀ hwf ht
        (show i + 5 * a ≤ i + 5 * b by omega) hEq
      omega
    · rcases Nat.eq_or_lt_of_le hab with heq | hab'
      · omega
      · have hd := isClosedOn_thirtyfive_dvd h hg₀ hwf ht
          (show i + 5 * b ≤ i + 5 * a by omega) hEq.symm
        omega

/-- The adversarial bag content law: the bag at step `n` is the full bag
minus the pieces announced since the block boundary. -/
theorem adversarialTrace_bag_eq_sdiff {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hdraw : ∀ n, s n
      ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag)
    (n : ℕ) :
    (adversarialTrace GameConfig.standard σ s GameState.init n).bag
      = Bag.full \ ((Finset.range (n % 7)).image
          (fun j => s (7 * (n / 7) + j))) := by
  have hl : LegalSequenceFrom Bag.full s := by
    intro k
    have := hdraw k
    rw [adversarialTrace_bag_from] at this
    rwa [show GameState.init.bag = Bag.full from GameState.init_bag] at this
  have h := BagCadence.bagAt_eq_sdiff hl n
  rw [adversarialTrace_bag_from,
    show GameState.init.bag = Bag.full from GameState.init_bag]
  exact h

/-- **The adversary's announcement constraint, explicit**: a piece is
announceable at step `n` iff it was not announced in the last `n mod 7`
steps — the adversary's entire freedom is a no-repeat-within-block rule. -/
theorem adversary_piece_available_iff {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hdraw : ∀ n, s n
      ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag)
    (n : ℕ) (p : Piece) :
    p ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag
      ↔ ∀ j, j < n % 7 → s (7 * (n / 7) + j) ≠ p := by
  rw [adversarialTrace_bag_eq_sdiff hdraw, Finset.mem_sdiff]
  constructor
  · rintro ⟨-, hnot⟩ j hj hp
    exact hnot (Finset.mem_image.mpr ⟨j, Finset.mem_range.mpr hj, hp⟩)
  · intro h
    refine ⟨Bag.mem_full p, ?_⟩
    intro hmem
    obtain ⟨j, hj, hp⟩ := Finset.mem_image.mp hmem
    exact h j (Finset.mem_range.mp hj) hp

/-- **Solving Tetris is surviving every permutation schedule**: `σ` solves
iff it survives the concatenation of every sequence of block permutations —
the adversary's strategy space reduced to its combinatorial core,
`(7!)^ℕ`. -/
theorem solvesTetris_iff_forall_patterns (σ : Solver GameConfig.standard) :
    SolvesTetris GameConfig.standard σ
      ↔ ∀ F : ℕ → ℕ → Piece,
          (∀ b : ℕ, ∀ j < 7, ∀ j' < 7, j ≠ j' → F b j ≠ F b j')
          → ∀ n, ¬ (adversarialTrace GameConfig.standard σ
              (fun m => F (m / 7) (m % 7)) GameState.init n).lost
              GameConfig.standard := by
  constructor
  · intro hσ F hF n
    exact hσ _ (BagCadence.pattern_seq_legal F hF) n
  · intro h s hs n
    obtain ⟨F, hF, hseq⟩ :=
      (BagCadence.legalSequenceFrom_iff_exists_pattern_seq s).mp hs
    rw [hseq]
    exact h F hF n

/-- **The mission statement as a permutation game**: Tetris is solvable iff
some solver survives every infinite sequence of bag permutations — the
project's headline proposition with its adversary in combinatorial normal
form. -/
theorem tetrisSolvable_iff_pattern_game :
    TetrisSolvable
      ↔ ∃ σ : Solver GameConfig.standard,
          ∀ F : ℕ → ℕ → Piece,
            (∀ b : ℕ, ∀ j < 7, ∀ j' < 7, j ≠ j' → F b j ≠ F b j')
            → ∀ n, ¬ (adversarialTrace GameConfig.standard σ
                (fun m => F (m / 7) (m % 7)) GameState.init n).lost
                GameConfig.standard := by
  unfold TetrisSolvable
  constructor
  · rintro ⟨σ, hσ⟩
    exact ⟨σ, (solvesTetris_iff_forall_patterns σ).mp hσ⟩
  · rintro ⟨σ, hσ⟩
    exact ⟨σ, (solvesTetris_iff_forall_patterns σ).mpr hσ⟩

/-- Aligned perfect clears close the loop adversarially too: empty boards
with equal bags are the same state, whoever picked the pieces. -/
theorem adversarial_perfect_clear_pair_return {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState} {n₁ n₂ : ℕ}
    (h1 : (adversarialTrace GameConfig.standard σ s g0 n₁).board.count = 0)
    (h2 : (adversarialTrace GameConfig.standard σ s g0 n₂).board.count = 0)
    (hbag : (adversarialTrace GameConfig.standard σ s g0 n₁).bag
        = (adversarialTrace GameConfig.standard σ s g0 n₂).bag) :
    adversarialTrace GameConfig.standard σ s g0 n₁
      = adversarialTrace GameConfig.standard σ s g0 n₂ := by
  have hb : (adversarialTrace GameConfig.standard σ s g0 n₁).board
      = (adversarialTrace GameConfig.standard σ s g0 n₂).board := by
    rw [(Board.count_eq_zero_iff_eq_empty _).mp h1,
      (Board.count_eq_zero_iff_eq_empty _).mp h2]
  calc adversarialTrace GameConfig.standard σ s g0 n₁
      = ⟨(adversarialTrace GameConfig.standard σ s g0 n₁).board,
        (adversarialTrace GameConfig.standard σ s g0 n₁).bag⟩ := rfl
    _ = ⟨(adversarialTrace GameConfig.standard σ s g0 n₂).board,
        (adversarialTrace GameConfig.standard σ s g0 n₂).bag⟩ := by
        rw [hb, hbag]
    _ = adversarialTrace GameConfig.standard σ s g0 n₂ := rfl

/-- Tail periodicity iterates: every later index returns at every multiple
of the period. -/
theorem adversarialTrace_tail_period_multiples {cfg : GameConfig}
    {σ : Solver cfg} {s : ℕ → Piece} {g0 : GameState}
    (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    ∀ j, adversarialTrace cfg σ s g0 m
      = adversarialTrace cfg σ s g0 (m + 35 * j) := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hstep := adversarialTrace_tail_periodic hper hcyc
      (show n ≤ m + 35 * j by omega)
    rw [show m + 35 * (j + 1) = (m + 35 * j) + 35 by ring]
    exact ih.trans hstep

/-- **Adversarial survival from finite evidence**: against a 35-periodic
stream, liveness on `[0, n + 35)` plus a 35-return at `n` proves the trace
lives forever — the perfect-clear route's finite check, adversarially. -/
theorem adversarial_survives_of_return {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} (hper : ∀ k, s (k + 35) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + 35))
    (hlive : ∀ k, k < n + 35 →
      ¬ (adversarialTrace GameConfig.standard σ s GameState.init k).lost
        GameConfig.standard) :
    ∀ m, ¬ (adversarialTrace GameConfig.standard σ s GameState.init m).lost
      GameConfig.standard := by
  intro m
  rcases Nat.lt_or_ge m n with hm | hm
  · exact hlive m (by omega)
  · have hx : adversarialTrace GameConfig.standard σ s GameState.init
        (n + (m - n) % 35)
        = adversarialTrace GameConfig.standard σ s GameState.init m := by
      have := adversarialTrace_tail_period_multiples hper hcyc
        (show n ≤ n + (m - n) % 35 by omega) ((m - n) / 35)
      rw [show n + (m - n) % 35 + 35 * ((m - n) / 35) = m by omega] at this
      exact this
    rw [← hx]
    exact hlive (n + (m - n) % 35) (by omega)

/-- Adversarial determinism at any period: a `T`-return of the trace against
a `T`-periodic stream pushes forward. -/
theorem adversarialTrace_periodic_T {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState} {T : ℕ}
    (hper : ∀ k, s (k + T) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + T)) :
    ∀ k, adversarialTrace cfg σ s g0 (n + k)
      = adversarialTrace cfg σ s g0 (n + T + k) := by
  intro k
  induction k with
  | zero => simpa using hcyc
  | succ k ih =>
    have hs : s (n + T + k) = s (n + k) := by
      rw [show n + T + k = (n + k) + T by omega]
      exact hper (n + k)
    rw [show n + (k + 1) = (n + k) + 1 by omega,
      show n + T + (k + 1) = (n + T + k) + 1 by omega,
      adversarialTrace_succ, adversarialTrace_succ, ih, hs]

/-- Tail `T`-returns iterate. -/
theorem adversarialTrace_tail_period_multiples_T {cfg : GameConfig}
    {σ : Solver cfg} {s : ℕ → Piece} {g0 : GameState} {T : ℕ}
    (hper : ∀ k, s (k + T) = s k) {n : ℕ}
    (hcyc : adversarialTrace cfg σ s g0 n
        = adversarialTrace cfg σ s g0 (n + T)) {m : ℕ} (hnm : n ≤ m) :
    ∀ j, adversarialTrace cfg σ s g0 m
      = adversarialTrace cfg σ s g0 (m + T * j) := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have htail : adversarialTrace cfg σ s g0 (m + T * j)
        = adversarialTrace cfg σ s g0 (m + T * j + T) := by
      have := adversarialTrace_periodic_T hper hcyc (m + T * j - n)
      rw [show n + (m + T * j - n) = m + T * j by omega,
        show n + T + (m + T * j - n) = m + T * j + T by omega] at this
      exact this
    rw [show m + T * (j + 1) = (m + T * j) + T by ring]
    exact ih.trans htail

/-- **Adversarial survival from finite evidence, any period**: against a
`T`-periodic stream (`T > 0`), liveness on `[0, n + T)` plus a `T`-return at
`n` proves the trace lives forever. -/
theorem adversarial_survives_of_return_T {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {T : ℕ} (hT : 0 < T) (hper : ∀ k, s (k + T) = s k) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard σ s GameState.init n
        = adversarialTrace GameConfig.standard σ s GameState.init (n + T))
    (hlive : ∀ k, k < n + T →
      ¬ (adversarialTrace GameConfig.standard σ s GameState.init k).lost
        GameConfig.standard) :
    ∀ m, ¬ (adversarialTrace GameConfig.standard σ s GameState.init m).lost
      GameConfig.standard := by
  intro m
  rcases Nat.lt_or_ge m n with hm | hm
  · exact hlive m (by omega)
  · have hx : adversarialTrace GameConfig.standard σ s GameState.init
        (n + (m - n) % T)
        = adversarialTrace GameConfig.standard σ s GameState.init m := by
      have := adversarialTrace_tail_period_multiples_T hper hcyc
        (show n ≤ n + (m - n) % T by omega) ((m - n) / T)
      rw [show n + (m - n) % T + T * ((m - n) / T) = m by
        have := Nat.div_add_mod (m - n) T
        omega] at this
      exact this
    rw [← hx]
    have hmod : (m - n) % T < T := Nat.mod_lt _ hT
    exact hlive (n + (m - n) % T) (by omega)

/-- A forever-live adversarial trace must return — pigeonhole on the
in-field states. -/
theorem adversarial_survives_exists_return {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hs : ∀ m, ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      m).lost GameConfig.standard) :
    ∃ n₁ n₂, n₁ < n₂
      ∧ adversarialTrace GameConfig.standard σ s GameState.init n₁
        = adversarialTrace GameConfig.standard σ s GameState.init n₂ := by
  have hfin : (Set.range
      (adversarialTrace GameConfig.standard σ s GameState.init)).Finite := by
    apply Set.Finite.subset (Set.finite_univ.image
      (fun q : InFieldBoard GameConfig.standard × Bag =>
        GameState.mk q.1.val q.2))
    rintro g ⟨n, rfl⟩
    have hwfn := adversarialTrace_board_wf
      (GameState.init_board_wf GameConfig.standard) hv n
    have hif : ∀ p ∈ (adversarialTrace GameConfig.standard σ s GameState.init
        n).board, p.2 < GameConfig.standard.rows :=
      (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp (hs n)
    exact ⟨(⟨(adversarialTrace GameConfig.standard σ s GameState.init n).board,
      hwfn, hif⟩,
      (adversarialTrace GameConfig.standard σ s GameState.init n).bag),
      Set.mem_univ _, rfl⟩
  have hninj : ¬ Function.Injective
      (adversarialTrace GameConfig.standard σ s GameState.init) := by
    intro hinj
    exact Set.infinite_range_of_injective hinj hfin
  rw [Function.not_injective_iff] at hninj
  obtain ⟨a, b, heq, hne⟩ := hninj
  rcases Nat.lt_or_ge a b with hab | hab
  · exact ⟨a, b, hab, heq⟩
  · exact ⟨b, a, by omega, heq.symm⟩

/-- 35-periodicity iterates to every multiple. -/
theorem stream_periodic_iterate {s : ℕ → Piece}
    (hper : ∀ k, s (k + 35) = s k) :
    ∀ m k, s (k + 35 * m) = s k := by
  intro m
  induction m with
  | zero => simp
  | succ m ih =>
    intro k
    rw [show k + 35 * (m + 1) = (k + 35 * m) + 35 by ring, hper, ih]

/-- **Surviving a periodic adversary ⟺ finite evidence**: against a legal
35-periodic stream, a valid solver's trace lives forever iff it exhibits a
live prefix ending in a state revisit. One fixed periodic adversary's game
reduces to a finite check, both ways — the quantum supplies the periodicity
at the return separation. -/
theorem adversarial_survives_iff_return {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} (hl : LegalSequence s)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hper : ∀ k, s (k + 35) = s k) :
    (∀ m, ¬ (adversarialTrace GameConfig.standard σ s GameState.init m).lost
      GameConfig.standard)
      ↔ ∃ n₁ n₂, n₁ < n₂
          ∧ adversarialTrace GameConfig.standard σ s GameState.init n₁
            = adversarialTrace GameConfig.standard σ s GameState.init n₂
          ∧ ∀ k, k < n₂ →
            ¬ (adversarialTrace GameConfig.standard σ s GameState.init k).lost
              GameConfig.standard := by
  constructor
  · intro hs
    obtain ⟨n₁, n₂, hlt, hret⟩ := adversarial_survives_exists_return hv hs
    exact ⟨n₁, n₂, hlt, hret, fun k _ => hs k⟩
  · rintro ⟨n₁, n₂, hlt, hret, hlive⟩
    have hdraw : ∀ n, s n
        ∈ (adversarialTrace GameConfig.standard σ s GameState.init n).bag := by
      intro n
      rw [adversarialTrace_bag_from,
        show GameState.init.bag = Bag.full from GameState.init_bag]
      have h := hl n
      rwa [Bag.canDraw_iff_mem] at h
    have hdvd : 35 ∣ (n₂ - n₁) :=
      thirtyfive_dvd_of_adversarialTrace_eq
        (GameState.init_board_wf GameConfig.standard) hv hdraw
        (le_of_lt hlt) hret
    obtain ⟨mm, hmm⟩ := hdvd
    have hperT : ∀ k, s (k + (n₂ - n₁)) = s k := by
      intro k
      rw [hmm]
      exact stream_periodic_iterate hper mm k
    refine adversarial_survives_of_return_T (n := n₁) (by omega) hperT ?_ ?_
    · rw [show n₁ + (n₂ - n₁) = n₂ by omega]
      exact hret
    · intro k hk
      exact hlive k (by omega)

/-- The canonical periodic stream `I O S Z T L J` repeated forever. -/
def canonicalStream : ℕ → Piece := fun n => BagCadence.sevenPattern (n % 7)

theorem canonicalStream_legal : LegalSequence canonicalStream :=
  BagCadence.periodic_stream_legal BagCadence.sevenPattern (by decide)

theorem canonicalStream_periodic : ∀ k, canonicalStream (k + 35)
    = canonicalStream k := by
  intro k
  unfold canonicalStream
  rw [show (k + 35) % 7 = k % 7 by omega]

/-- **A finite necessary condition for solvability**: if Tetris is solvable
by a valid solver, then some solver exhibits a live state-revisit against
the canonical periodic stream — a single concrete, finitely checkable
certificate shape that MUST exist if the mission can succeed at all. -/
theorem tetrisSolvableValid_implies_canonical_evidence :
    TetrisSolvableValid →
      ∃ (σ : Solver GameConfig.standard) (n₁ n₂ : ℕ), n₁ < n₂
        ∧ adversarialTrace GameConfig.standard σ canonicalStream
            GameState.init n₁
          = adversarialTrace GameConfig.standard σ canonicalStream
              GameState.init n₂
        ∧ ∀ k, k < n₂ →
          ¬ (adversarialTrace GameConfig.standard σ canonicalStream
              GameState.init k).lost GameConfig.standard := by
  rintro ⟨σ, hval, hsolve⟩
  have hdraw : ∀ n, canonicalStream n
      ∈ (adversarialTrace GameConfig.standard σ canonicalStream
        GameState.init n).bag := by
    intro n
    rw [adversarialTrace_bag_from,
      show GameState.init.bag = Bag.full from GameState.init_bag]
    have h := canonicalStream_legal n
    rwa [Bag.canDraw_iff_mem] at h
  have hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ canonicalStream
      GameState.init n) (canonicalStream n) with piece := canonicalStream n }
      : Placement).Valid GameConfig.standard := by
    intro n
    obtain ⟨hp, hval'⟩ := hval _ (canonicalStream n) (hdraw n)
    rw [placement_with_piece_self hp]
    exact hval'
  have hs : ∀ m, ¬ (adversarialTrace GameConfig.standard σ canonicalStream
      GameState.init m).lost GameConfig.standard :=
    hsolve canonicalStream canonicalStream_legal
  obtain ⟨n₁, n₂, hlt, hret⟩ := adversarial_survives_exists_return hv hs
  exact ⟨σ, n₁, n₂, hlt, hret, fun k _ => hs k⟩

/-- The bounded adversarial pigeonhole: a forever-live adversarial trace
revisits a state within `2^207` steps. -/
theorem adversarial_survives_return_within {σ : Solver GameConfig.standard}
    {s : ℕ → Piece}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s GameState.init n)
      (s n) with piece := s n } : Placement).Valid GameConfig.standard)
    (hs : ∀ m, ¬ (adversarialTrace GameConfig.standard σ s GameState.init
      m).lost GameConfig.standard) :
    ∃ n₁ n₂, n₁ < n₂ ∧ n₂ ≤ 2 ^ 207
      ∧ adversarialTrace GameConfig.standard σ s GameState.init n₁
        = adversarialTrace GameConfig.standard σ s GameState.init n₂ := by
  classical
  have hwf : ∀ n, Board.WF GameConfig.standard
      (adversarialTrace GameConfig.standard σ s GameState.init n).board :=
    adversarialTrace_board_wf (GameState.init_board_wf GameConfig.standard) hv
  have hif : ∀ n, ∀ p ∈ (adversarialTrace GameConfig.standard σ s
      GameState.init n).board, p.2 < GameConfig.standard.rows :=
    fun n => (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp
      (hs n)
  have hlt : Fintype.card (InFieldBoard GameConfig.standard × Bag)
      < Fintype.card (Fin (2 ^ 207 + 1)) := by
    rw [Fintype.card_fin, ClearRate.card_infield_times_bag]
    exact Nat.lt_succ_self _
  obtain ⟨i, j, hne, hfeq⟩ := Fintype.exists_ne_map_eq_of_card_lt
    (fun i : Fin (2 ^ 207 + 1) =>
      ((⟨(adversarialTrace GameConfig.standard σ s GameState.init i).board,
        hwf i, hif i⟩,
      (adversarialTrace GameConfig.standard σ s GameState.init i).bag)
        : InFieldBoard GameConfig.standard × Bag)) hlt
  have hb : (adversarialTrace GameConfig.standard σ s GameState.init i).board
      = (adversarialTrace GameConfig.standard σ s GameState.init j).board :=
    congrArg (fun q : InFieldBoard GameConfig.standard × Bag => q.1.val) hfeq
  have hg : (adversarialTrace GameConfig.standard σ s GameState.init i).bag
      = (adversarialTrace GameConfig.standard σ s GameState.init j).bag :=
    congrArg Prod.snd hfeq
  have hstates : adversarialTrace GameConfig.standard σ s GameState.init i
      = adversarialTrace GameConfig.standard σ s GameState.init j := by
    calc adversarialTrace GameConfig.standard σ s GameState.init i
        = ⟨(adversarialTrace GameConfig.standard σ s GameState.init i).board,
          (adversarialTrace GameConfig.standard σ s GameState.init i).bag⟩ :=
        rfl
      _ = ⟨(adversarialTrace GameConfig.standard σ s GameState.init j).board,
          (adversarialTrace GameConfig.standard σ s GameState.init j).bag⟩ :=
        by rw [hb, hg]
      _ = adversarialTrace GameConfig.standard σ s GameState.init j := rfl
  have hij : (i : ℕ) ≠ (j : ℕ) := fun h => hne (Fin.ext h)
  rcases Nat.lt_or_ge (i : ℕ) (j : ℕ) with hlt' | hge
  · exact ⟨i, j, hlt', by have := j.isLt; omega, hstates⟩
  · exact ⟨j, i, by omega, by have := i.isLt; omega, hstates.symm⟩

/-- **The canonical game, characterized and bounded**: a valid solver
survives the canonical periodic adversary forever iff it exhibits a live
revisit — and if it does survive, the revisit occurs within `2^207` steps.
The mission's necessary condition is a bounded search. -/
theorem canonical_evidence_bounded {σ : Solver GameConfig.standard}
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ canonicalStream
      GameState.init n) (canonicalStream n) with piece := canonicalStream n }
      : Placement).Valid GameConfig.standard)
    (hs : ∀ m, ¬ (adversarialTrace GameConfig.standard σ canonicalStream
      GameState.init m).lost GameConfig.standard) :
    ∃ n₁ n₂, n₁ < n₂ ∧ n₂ ≤ 2 ^ 207
      ∧ adversarialTrace GameConfig.standard σ canonicalStream
          GameState.init n₁
        = adversarialTrace GameConfig.standard σ canonicalStream
            GameState.init n₂
      ∧ ∀ k, k < n₂ →
        ¬ (adversarialTrace GameConfig.standard σ canonicalStream
            GameState.init k).lost GameConfig.standard := by
  obtain ⟨n₁, n₂, hlt, hbound, hret⟩ :=
    adversarial_survives_return_within hv hs
  exact ⟨n₁, n₂, hlt, hbound, hret, fun k _ => hs k⟩

end ClearRate
end Tetris
