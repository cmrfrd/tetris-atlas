import Mathlib
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

end ClearRate
end Tetris
