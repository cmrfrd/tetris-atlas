import Mathlib
import Proofs.Survival.ClearRecurrence
import Proofs.Safety.BagCadence
import Proofs.Safety.Adversarial
import Proofs.Safety.SafeSet

/-!
# The five-bag quantum, adversarially

`Proofs/Survival/ClearRecurrence` proves that a *cooperative* trace can only
revisit a state after a multiple of 35 placements — five bags — and lands that
on `ClosedCycle`. But `ClosedCycle` is the cooperative M2 artifact: its policy
deals itself pieces. The object `TetrisSolvable` actually needs is the
adversarial one, where the bag announces the piece and the solver must answer.

This file transports the quantum across that gap. Both clocks survive the
move unchanged, because both are counting arguments and the adversary controls
only *which* piece arrives, never how many cells it carries:

* **The mass clock.** Every placement adds 4 cells and every cleared row removes
  10, so `count ≡ count₀ + 4n (mod 10)` along any adversarial trace
  (`adversarialTrace_count_mod_ten`). Returning to a board therefore forces
  `5 ∣ Δn` (`five_dvd_of_adversarial_count_eq`).
* **The bag clock.** A legal draw takes the bag `7, 6, …, 1, 7, …`, so its size
  after `n` placements is `7 − (7 − c₀ + n) mod 7` (`bag_card_adversarialTrace`)
  and returning to a bag forces `7 ∣ Δn`.

Hence `thirtyfive_dvd_of_adversarialTrace_eq`, and on the certificate itself
**`adversarialClosedCycle_thirtyfive_dvd`** / `adversarialClosedCycle_thirtyfive_le`:
*every* adversarial closed cycle has period a multiple of 35 placements, and
none is shorter than 35.

The practical content is a search constraint on the real proof target. An M2/M3
hunt over adversarial cycles never has to test a separation that is not a
multiple of 35, and can reject any candidate certificate shorter than five bags
without examining its geometry.
-/

namespace Tetris
namespace ClearRate

/-! ## Adversarial placements -/

/-- Setting a placement's piece to the piece it already plays is a no-op. This is
what makes `adversarialStep`'s piece-forcing invisible when the solver already
answers with the announced piece. -/
theorem placement_with_piece_self {pl : Placement} {p : Piece} (h : pl.piece = p) :
    ({ pl with piece := p } : Placement) = pl := by
  cases pl
  simp_all

/-- Adversarial trace boards stay well-formed. -/
theorem adversarialTrace_board_wf {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState} (hwf : Board.WF cfg g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace cfg σ s g0 n) (s n) with piece := s n }
      : Placement).Valid cfg) (n : ℕ) :
    Board.WF cfg (adversarialTrace cfg σ s g0 n).board := by
  induction n with
  | zero => simpa using hwf
  | succ k ih =>
    rw [adversarialTrace_succ, adversarialStep_board]
    exact Placement.applyStep_wf ih (hv k)

/-! ## The mass clock -/

/-- **Occupancy mod 10 is a clock, adversarially too.** The adversary picks the
piece but not its size: every drop adds 4 cells and every clear removes 10, so
the residue advances by 4 per placement no matter what arrives. -/
theorem adversarialTrace_count_mod_ten {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s g0 n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard) (n : ℕ) :
    (adversarialTrace GameConfig.standard σ s g0 n).board.count % 10
      = (g0.board.count + 4 * n) % 10 := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hstep := Board.applyStep_count_mod_cols
      (adversarialTrace_board_wf hwf hv k) (hv k)
    rw [GameConfig.standard_cols] at hstep
    rw [adversarialTrace_succ, adversarialStep_board, hstep]
    omega

/-- Returning to a cell count forces a multiple of 5 placements. -/
theorem five_dvd_of_adversarial_count_eq {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s g0 n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    {n₁ n₂ : ℕ}
    (hc : (adversarialTrace GameConfig.standard σ s g0 n₁).board.count
        = (adversarialTrace GameConfig.standard σ s g0 n₂).board.count) :
    5 ∣ (n₂ - n₁) := by
  have h1 := adversarialTrace_count_mod_ten hwf hv n₁
  have h2 := adversarialTrace_count_mod_ten hwf hv n₂
  rw [hc] at h1
  omega

/-! ## The bag clock -/

/-- **The bag counter, adversarially.** Legal draws cycle the bag size through
`7, 6, …, 1, 7, …` whoever is choosing the pieces. -/
theorem bag_card_adversarialTrace {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState}
    (hdraw : ∀ n, s n ∈ (adversarialTrace cfg σ s g0 n).bag) (n : ℕ) :
    (adversarialTrace cfg σ s g0 n).bag.card = 7 - ((7 - g0.bag.card) + n) % 7 := by
  have hle : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  induction n with
  | zero =>
    simp only [adversarialTrace_zero, Nat.add_zero]
    omega
  | succ k ih =>
    have hmod : ((7 - g0.bag.card) + k) % 7 < 7 := Nat.mod_lt _ (by omega)
    rw [adversarialTrace_succ, adversarialStep_bag, card_draw (hdraw k), ih]
    split <;> omega

/-- Equal bag sizes force equal placement counts mod 7. -/
theorem seven_mod_eq_of_adversarial_bag_card_eq {cfg : GameConfig} {σ : Solver cfg}
    {s : ℕ → Piece} {g0 : GameState}
    (hdraw : ∀ n, s n ∈ (adversarialTrace cfg σ s g0 n).bag) {n₁ n₂ : ℕ}
    (h : (adversarialTrace cfg σ s g0 n₁).bag.card
        = (adversarialTrace cfg σ s g0 n₂).bag.card) :
    n₁ % 7 = n₂ % 7 := by
  have hle : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  have h1 := bag_card_adversarialTrace hdraw n₁
  have h2 := bag_card_adversarialTrace hdraw n₂
  have hm1 : ((7 - g0.bag.card) + n₁) % 7 < 7 := Nat.mod_lt _ (by omega)
  have hm2 : ((7 - g0.bag.card) + n₂) % 7 < 7 := Nat.mod_lt _ (by omega)
  rw [h1, h2] at h
  omega

/-! ## The quantum -/

/-- **The five-bag quantum, adversarially.** Any legal adversarial trace that
revisits a state does so after a multiple of 35 placements. -/
theorem thirtyfive_dvd_of_adversarialTrace_eq {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s g0 n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    (hdraw : ∀ n, s n ∈ (adversarialTrace GameConfig.standard σ s g0 n).bag)
    {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : adversarialTrace GameConfig.standard σ s g0 n₁
        = adversarialTrace GameConfig.standard σ s g0 n₂) :
    35 ∣ (n₂ - n₁) := by
  have h5 : 5 ∣ (n₂ - n₁) := five_dvd_of_adversarial_count_eq hwf hv (by rw [h])
  have h7 : n₁ % 7 = n₂ % 7 :=
    seven_mod_eq_of_adversarial_bag_card_eq hdraw (by rw [h])
  omega

/-- **The adversarial M2 certificate is quantised.** An `AdversarialClosedCycle`
— the artifact `TetrisSolvable` actually needs — can only revisit a state after
a multiple of **35 placements = 5 bags**, against every legal piece sequence.
A cycle hunt never has to test any other separation. -/
theorem adversarialClosedCycle_thirtyfive_dvd
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    (hg0 : g0 ∈ C.states) (hwf : Board.WF GameConfig.standard g0.board)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g0.bag t) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h : adversarialTrace GameConfig.standard C.solver t g0 n₁
        = adversarialTrace GameConfig.standard C.solver t g0 n₂) :
    35 ∣ (n₂ - n₁) := by
  have hdraw : ∀ n, t n ∈ (adversarialTrace GameConfig.standard C.solver t g0 n).bag := by
    intro n
    have hn := hl n
    rw [Bag.canDraw_iff_mem] at hn
    rw [adversarialTrace_bag_from]
    exact hn
  have hv : ∀ n, ({ C.solver (adversarialTrace GameConfig.standard C.solver t g0 n) (t n)
      with piece := t n } : Placement).Valid GameConfig.standard := by
    intro n
    obtain ⟨hp, hval⟩ :=
      C.valid _ (C.adversarialTrace_mem_states_from_mem hg0 hl n) (t n) (hdraw n)
    rw [placement_with_piece_self hp]
    exact hval
  exact thirtyfive_dvd_of_adversarialTrace_eq hwf hv hdraw h12 h

/-- A nontrivial adversarial closed cycle is at least five bags long. -/
theorem adversarialClosedCycle_thirtyfive_le
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    (hg0 : g0 ∈ C.states) (hwf : Board.WF GameConfig.standard g0.board)
    {t : ℕ → Piece} (hl : LegalSequenceFrom g0.bag t) {n₁ n₂ : ℕ} (h12 : n₁ < n₂)
    (h : adversarialTrace GameConfig.standard C.solver t g0 n₁
        = adversarialTrace GameConfig.standard C.solver t g0 n₂) :
    n₁ + 35 ≤ n₂ := by
  have hd := adversarialClosedCycle_thirtyfive_dvd C hg0 hwf hl (le_of_lt h12) h
  omega

/-- **Every adversarial closed cycle holds at least 35 states.** Run any legal
sequence from a cycle member (one exists — `exists_legalSequenceFrom`): the
first 35 trace states are pairwise distinct by the quantum and all lie in the
cycle. The counting lower bound on the *adversarial* M2 artifact — the one
`TetrisSolvable` needs. (The bag-nonemptiness hypothesis excludes the
degenerate empty-bag singleton cycles, which certify nothing.) -/
theorem adversarialClosedCycle_card_ge_thirtyfive
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    (hg0 : g0 ∈ C.states) (hwf : Board.WF GameConfig.standard g0.board)
    (hbag : g0.bag.Nonempty) :
    35 ≤ C.states.card := by
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hcalc : (Finset.range 35).card ≤ C.states.card := by
    refine Finset.card_le_card_of_injOn
      (fun i => adversarialTrace GameConfig.standard C.solver t g0 i) ?_ ?_
    · intro i _
      exact C.adversarialTrace_mem_states_from_mem hg0 ht i
    · intro i hi j hj hEq
      simp only [Finset.coe_range, Set.mem_Iio] at hi hj
      dsimp only at hEq
      rcases le_total i j with h | h
      · have := adversarialClosedCycle_thirtyfive_dvd C hg0 hwf ht h hEq
        omega
      · have := adversarialClosedCycle_thirtyfive_dvd C hg0 hwf ht h hEq.symm
        omega
  rwa [Finset.card_range] at hcalc

/-- The adversarial init-cycle floor: an `AdversarialClosedCycle` through the
initial state holds at least 35 states. -/
theorem init_adversarialClosedCycle_card_ge_thirtyfive
    (C : AdversarialClosedCycle GameConfig.standard)
    (h0 : GameState.init ∈ C.states) : 35 ≤ C.states.card :=
  adversarialClosedCycle_card_ge_thirtyfive C h0
    (GameState.init_board_wf GameConfig.standard) GameState.init_bag_nonempty

/-! ## The quantum on the M4 artifact itself -/

/-- Along any legal trace from a closed-atlas state, the materialised solver's
forced placement is valid: the atlas is total on drawable pieces and its
answers play the announced piece in bounds. -/
theorem isClosedOn_trace_forced_valid {cfg : GameConfig} {A : Atlas cfg}
    {S : Finset GameState} (h : A.IsClosedOn cfg S) {g₀ : GameState}
    (hg₀ : g₀ ∈ S) {t : ℕ → Piece} (hl : LegalSequenceFrom g₀.bag t) (n : ℕ) :
    ({ A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) (t n)
        with piece := t n } : Placement).Valid cfg := by
  have hgS : adversarialTrace cfg A.toSolver t g₀ n ∈ S :=
    h.toSolver_adversarialTrace_mem hg₀ hl n
  have hp : t n ∈ (adversarialTrace cfg A.toSolver t g₀ n).bag := by
    rw [adversarialTrace_bag_from]
    exact hl n
  obtain ⟨pl, hpl⟩ := Option.isSome_iff_exists.mp (h.total _ hgS (t n) hp)
  obtain ⟨hpiece, hvalid⟩ := h.valid _ hgS (t n) hp pl hpl
  have hts : A.toSolver (adversarialTrace cfg A.toSolver t g₀ n) (t n) = pl :=
    Atlas.toSolver_apply_of_some hpl
  rw [hts, placement_with_piece_self hpiece]
  exact hvalid

/-- **The five-bag quantum on a closed Atlas.** The materialised solver's trace
through a closed atlas revisits a state only at multiples of 35 placements. -/
theorem isClosedOn_thirtyfive_dvd {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) {t : ℕ → Piece}
    (hl : LegalSequenceFrom g₀.bag t) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (heq : adversarialTrace GameConfig.standard A.toSolver t g₀ n₁
        = adversarialTrace GameConfig.standard A.toSolver t g₀ n₂) :
    35 ∣ (n₂ - n₁) :=
  thirtyfive_dvd_of_adversarialTrace_eq hwf
    (isClosedOn_trace_forced_valid h hg₀ hl)
    (fun n => by rw [adversarialTrace_bag_from]; exact hl n) h12 heq

/-- **A closed Atlas covering a real state holds at least 35 states.** Running
the greedy legal sequence from any member with a well-formed board and a
nonempty bag, the first 35 trace states are pairwise distinct and all lie in
`S`. -/
theorem isClosedOn_card_ge_thirtyfive {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    {g₀ : GameState} (hg₀ : g₀ ∈ S)
    (hwf : Board.WF GameConfig.standard g₀.board) (hbag : g₀.bag.Nonempty) :
    35 ≤ S.card := by
  obtain ⟨t, ht⟩ := BagCadence.exists_legalSequenceFrom hbag
  have hcalc : (Finset.range 35).card ≤ S.card := by
    refine Finset.card_le_card_of_injOn
      (fun i => adversarialTrace GameConfig.standard A.toSolver t g₀ i) ?_ ?_
    · intro i _
      exact h.toSolver_adversarialTrace_mem hg₀ ht i
    · intro i hi j hj hEq
      simp only [Finset.coe_range, Set.mem_Iio] at hi hj
      dsimp only at hEq
      rcases le_total i j with hij | hij
      · have := isClosedOn_thirtyfive_dvd h hg₀ hwf ht hij hEq
        omega
      · have := isClosedOn_thirtyfive_dvd h hg₀ hwf ht hij hEq.symm
        omega
  rwa [Finset.card_range] at hcalc

/-- **The Atlas has at least 35 entries.** Any init-containing closed Atlas —
the exact M4 witness shape of `tetrisSolvable_of_exists_init_closed_atlas` —
covers at least 35 states. Together with `solvable_implies_bounded_atlas` the
M4 artifact's size is pinned to `[35, 2^207]` by counting alone. -/
theorem init_closed_atlas_card_ge_thirtyfive {A : Atlas GameConfig.standard}
    {S : Finset GameState} (h : A.IsClosedOn GameConfig.standard S)
    (hinit : GameState.init ∈ S) :
    35 ≤ S.card :=
  isClosedOn_card_ge_thirtyfive h hinit
    (GameState.init_board_wf GameConfig.standard) GameState.init_bag_nonempty

end ClearRate
end Tetris
