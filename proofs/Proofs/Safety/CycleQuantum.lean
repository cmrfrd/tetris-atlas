import Mathlib
import Proofs.Survival.ClearRecurrence
import Proofs.Survival.ClearMix
import Proofs.Invariants.StateSpace
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

/-! ## The balance theorem on the cycle artifacts -/

/-- Policy-trace bags are the `bagAt` stream of the pieces the policy plays. -/
theorem trace_bag_eq_bagAt {cfg : GameConfig} (π : Policy cfg) (g0 : GameState) :
    ∀ k, (trace cfg π g0 k).bag
      = bagAt g0.bag (fun m => (π (trace cfg π g0 m)).piece) k
  | 0 => rfl
  | k + 1 => by
      rw [trace_succ, GameState.step_bag]
      change (trace cfg π g0 k).bag.draw _
        = (bagAt g0.bag (fun m => (π (trace cfg π g0 m)).piece) k).draw _
      rw [trace_bag_eq_bagAt π g0 k]

/-- Legal draws make the played-piece stream a legal sequence. -/
theorem legalSequence_of_trace_draws {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace cfg π g0 k)).piece ∈ (trace cfg π g0 k).bag) :
    LegalSequenceFrom g0.bag (fun m => (π (trace cfg π g0 m)).piece) := by
  intro k
  change (π (trace cfg π g0 k)).piece
    ∈ bagAt g0.bag (fun m => (π (trace cfg π g0 m)).piece) k
  rw [← trace_bag_eq_bagAt]
  exact hdraw k

/-- **Every closed-cycle period plays each piece exactly five times.** The
balance theorem landed on the cooperative M2 artifact: over any 35-placement
period of a `ClosedCycle`, the policy's piece stream contains each of the
seven pieces exactly five times — in particular exactly five T's and exactly
five I's per period. -/
theorem closedCycle_period_piece_balanced (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ}
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) (p : Piece) :
    ((Finset.range 35).filter (fun k =>
        (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
          = p)).card = 5 := by
  have hdraw : ∀ k, (C.policy (trace GameConfig.standard C.policy g0 k)).piece
      ∈ (trace GameConfig.standard C.policy g0 k).bag :=
    fun k => C.legal_draw _ (C.trace_mem_states h0 k)
  have hl := legalSequence_of_trace_draws hdraw
  have hbag : bagAt g0.bag
        (fun m => (C.policy (trace GameConfig.standard C.policy g0 m)).piece)
        (n + 35)
      = bagAt g0.bag
        (fun m => (C.policy (trace GameConfig.standard C.policy g0 m)).piece)
        n := by
    rw [← trace_bag_eq_bagAt, ← trace_bag_eq_bagAt, hcyc]
  exact BagCadence.window_thirtyfive_balanced hl hbag p

/-- Splitting a filtered range count at an interior point: the count over
`[0, a + b)` is the count over `[0, a)` plus the shifted count over `[0, b)`. -/
theorem card_filter_range_add (P : ℕ → Prop) [DecidablePred P] (a : ℕ) :
    ∀ b, ((Finset.range (a + b)).filter P).card
      = ((Finset.range a).filter P).card
        + ((Finset.range b).filter (fun k => P (a + k))).card := by
  intro b
  induction b with
  | zero => simp
  | succ b ih =>
    rw [show a + (b + 1) = (a + b) + 1 by ring, Finset.range_add_one,
      Finset.range_add_one, Finset.filter_insert, Finset.filter_insert]
    by_cases h : P (a + b)
    · rw [if_pos h, if_pos h, Finset.card_insert_of_notMem (by simp),
        Finset.card_insert_of_notMem (by simp), ih]
      omega
    · rw [if_neg h, if_neg h, ih]

/-- **Multi-period balance**: over `j` cycle periods a closed cycle plays
each piece exactly `5·j` times — in particular exactly `5j` I's and `5j` T's.
Periodicity re-arms the balance theorem on every lap and the windows sum. -/
theorem closedCycle_multi_period_piece_balanced
    (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ}
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) (p : Piece) :
    ∀ j, ((Finset.range (35 * j)).filter (fun k =>
        (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
          = p)).card = 5 * j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hsplit := card_filter_range_add (fun k =>
      (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece = p)
      (35 * j) 35
    rw [show 35 * (j + 1) = 35 * j + 35 by ring, hsplit, ih]
    have hj := trace_period_multiples C.policy g0 hcyc j
    have hj1 := trace_period_multiples C.policy g0 hcyc (j + 1)
    have hcycj : trace GameConfig.standard C.policy g0 (n + 35 * j)
        = trace GameConfig.standard C.policy g0 ((n + 35 * j) + 35) := by
      rw [show (n + 35 * j) + 35 = n + (j + 1) * 35 by ring,
        show n + 35 * j = n + j * 35 by ring]
      exact hj.symm.trans hj1
    have hbal := closedCycle_period_piece_balanced C h0 hcycj p
    have harg : ∀ x : ℕ, n + (35 * j + x) = (n + 35 * j) + x := fun x => by ring
    simp only [harg]
    rw [hbal]
    ring

/-- The balance theorem at policy-trace level: a 35-return of a legally-drawn
policy trace deals each piece exactly five times. (`ClosedCycle` supplies the
`hdraw` hypothesis from its `legal_draw` field.) -/
theorem trace_period_piece_balanced {π : Policy GameConfig.standard}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + 35)) (p : Piece) :
    ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card = 5 := by
  have hl := legalSequence_of_trace_draws hdraw
  have hbag : bagAt g0.bag
        (fun m => (π (trace GameConfig.standard π g0 m)).piece) (n + 35)
      = bagAt g0.bag
        (fun m => (π (trace GameConfig.standard π g0 m)).piece) n := by
    rw [← trace_bag_eq_bagAt, ← trace_bag_eq_bagAt, hcyc]
  exact BagCadence.window_thirtyfive_balanced hl hbag p

/-- Multi-period balance at policy-trace level: `j` periods deal each piece
exactly `5·j` times. -/
theorem trace_multi_period_piece_balanced {π : Policy GameConfig.standard}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + 35)) (p : Piece) :
    ∀ j, ((Finset.range (35 * j)).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card
      = 5 * j := by
  intro j
  induction j with
  | zero => simp
  | succ j ih =>
    have hsplit := card_filter_range_add (fun k =>
      (π (trace GameConfig.standard π g0 (n + k))).piece = p) (35 * j) 35
    rw [show 35 * (j + 1) = 35 * j + 35 by ring, hsplit, ih]
    have hj := trace_period_multiples π g0 hcyc j
    have hj1 := trace_period_multiples π g0 hcyc (j + 1)
    have hcycj : trace GameConfig.standard π g0 (n + 35 * j)
        = trace GameConfig.standard π g0 ((n + 35 * j) + 35) := by
      rw [show (n + 35 * j) + 35 = n + (j + 1) * 35 by ring,
        show n + 35 * j = n + j * 35 by ring]
      exact hj.symm.trans hj1
    have hbal := trace_period_piece_balanced hdraw hcycj p
    have harg : ∀ x : ℕ, n + (35 * j + x) = (n + 35 * j) + x := fun x => by ring
    simp only [harg]
    rw [hbal]
    ring

/-- **The I-counter linear law**: on a periodic legally-drawn trace from
`init`, the cumulative I count advances by exactly `5·j` over `j` periods —
the counter feeding the tetris caps (`sizeCount_four_le_iCount`) is pinned
linearly by the balance theorem. -/
theorem cycle_iCount_linear {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (j : ℕ) :
    iCount GameConfig.standard π GameState.init (n + 35 * j)
      - iCount GameConfig.standard π GameState.init n = 5 * j := by
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π)
    (n + 35 * j)
  have hsplit := card_filter_range_add (fun m =>
    (π (trace GameConfig.standard π GameState.init m)).piece = Piece.I)
    n (35 * j)
  have hbal := trace_multi_period_piece_balanced hdraw hcyc Piece.I j
  rw [h1, h2, hsplit, hbal]
  omega

/-- **The multi-block window law**: any `35·q` consecutive legal draws hold
between `4q` and `6q` of every piece — no periodicity hypothesis. Split into
`q` 35-blocks and sum the two-sided window law. -/
theorem window_multiblock_bounds {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (n : ℕ) (p : Piece) :
    ∀ q, 4 * q ≤ ((Finset.range (35 * q)).filter
          (fun k => s (n + k) = p)).card
      ∧ ((Finset.range (35 * q)).filter (fun k => s (n + k) = p)).card
        ≤ 6 * q := by
  intro q
  induction q with
  | zero => simp
  | succ q ih =>
    have hsplit := card_filter_range_add (fun k => s (n + k) = p) (35 * q) 35
    have hge := BagCadence.window_thirtyfive_ge_four hl (n + 35 * q) p
    have hle := BagCadence.window_thirtyfive_le_six hl (n + 35 * q) p
    have harg : ∀ x : ℕ, n + (35 * q + x) = (n + 35 * q) + x := fun x => by ring
    rw [show 35 * (q + 1) = 35 * q + 35 by ring, hsplit]
    simp only [harg]
    omega

/-- **The I-counter bracket at every aligned horizon — no cycle needed**: on
any legally-drawn trace from `init`, the cumulative I count over `q` bag
quints advances by between `4q` and `6q`. Pure cadence; a cycle sharpens it
to exactly `5q` (`cycle_iCount_linear`). -/
theorem iCount_window_bounds {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n q : ℕ) :
    4 * q ≤ iCount GameConfig.standard π GameState.init (n + 35 * q)
        - iCount GameConfig.standard π GameState.init n
      ∧ iCount GameConfig.standard π GameState.init (n + 35 * q)
        - iCount GameConfig.standard π GameState.init n ≤ 6 * q := by
  have hl := legalSequence_of_trace_draws hdraw
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π)
    (n + 35 * q)
  have hsplit := card_filter_range_add (fun m =>
    (π (trace GameConfig.standard π GameState.init m)).piece = Piece.I)
    n (35 * q)
  have hblock := window_multiblock_bounds hl n Piece.I q
  omega

/-- **The window law at every length**: any `w` consecutive legal draws hold
between `4·⌊w/35⌋` and `6·⌊w/35⌋ + 6` of every piece — the piece frequency is
sandwiched in `[4/35, 6/35]` at every scale, no alignment or periodicity
required. Squeeze between the enclosed and enclosing block windows. -/
theorem window_bounds_any_length {initBag : Bag} {s : ℕ → Piece}
    (hl : LegalSequenceFrom initBag s) (n w : ℕ) (p : Piece) :
    4 * (w / 35) ≤ ((Finset.range w).filter (fun k => s (n + k) = p)).card
      ∧ ((Finset.range w).filter (fun k => s (n + k) = p)).card
        ≤ 6 * (w / 35) + 6 := by
  classical
  set q := w / 35 with hq
  have hlo : 35 * q ≤ w := by omega
  have hhi : w ≤ 35 * (q + 1) := by omega
  have hin : ((Finset.range (35 * q)).filter (fun k => s (n + k) = p)).card
      ≤ ((Finset.range w).filter (fun k => s (n + k) = p)).card :=
    Finset.card_le_card (Finset.filter_subset_filter _
      (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega))
  have hout : ((Finset.range w).filter (fun k => s (n + k) = p)).card
      ≤ ((Finset.range (35 * (q + 1))).filter (fun k => s (n + k) = p)).card :=
    Finset.card_le_card (Finset.filter_subset_filter _
      (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega))
  obtain ⟨hg, _⟩ := window_multiblock_bounds hl n p q
  obtain ⟨_, hle⟩ := window_multiblock_bounds hl n p (q + 1)
  constructor
  · omega
  · omega

/-- **The I-counter bracket at every horizon** — aligned or not: on any
legally-drawn trace from `init`, `4·⌊Δn/35⌋ ≤ ΔiCount ≤ 6·⌊Δn/35⌋ + 6`.
The I supply is linear with slope in `[4/35, 6/35]` at all scales. -/
theorem iCount_bracket_any {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n m : ℕ}
    (hnm : n ≤ m) :
    4 * ((m - n) / 35) ≤ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
      ∧ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
        ≤ 6 * ((m - n) / 35) + 6 := by
  have hl := legalSequence_of_trace_draws hdraw
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) m
  have hsplit := card_filter_range_add (fun k =>
    (π (trace GameConfig.standard π GameState.init k)).piece = Piece.I)
    n (m - n)
  rw [show n + (m - n) = m by omega] at hsplit
  have hwin := window_bounds_any_length hl n (m - n) Piece.I
  omega

/-- **The adversarial period is balanced too**: over any 35-placement period of
an `AdversarialClosedCycle`, the announced piece sequence contains each piece
exactly five times — the adversary's freedom inside a cycle is limited to the
order of a fixed multiset. -/
theorem adversarialClosedCycle_period_piece_balanced
    (C : AdversarialClosedCycle GameConfig.standard) {g0 : GameState}
    {t : ℕ → Piece} (hl : LegalSequenceFrom g0.bag t) {n : ℕ}
    (hcyc : adversarialTrace GameConfig.standard C.solver t g0 n
        = adversarialTrace GameConfig.standard C.solver t g0 (n + 35))
    (p : Piece) :
    ((Finset.range 35).filter (fun k => t (n + k) = p)).card = 5 := by
  have hbag : bagAt g0.bag t (n + 35) = bagAt g0.bag t n := by
    rw [← adversarialTrace_bag_from, ← adversarialTrace_bag_from, hcyc]
  exact BagCadence.window_thirtyfive_balanced hl hbag p

/-- The I-cadence on policy traces: with legal draws, an I is played within
every window of thirteen placements. -/
theorem trace_exists_I_within_thirteen {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace cfg π g0 k)).piece ∈ (trace cfg π g0 k).bag)
    (n : ℕ) :
    ∃ k, k < 13 ∧ (π (trace cfg π g0 (n + k))).piece = Piece.I := by
  obtain ⟨k, hk, hks⟩ := BagCadence.exists_I_within_thirteen
    (legalSequence_of_trace_draws hdraw) n
  exact ⟨k, hk, hks⟩

/-- **A closed cycle plays an I within every thirteen placements** — the
tetris-fuel cadence holds along the M2 artifact's own trajectory. -/
theorem closedCycle_exists_I_within_thirteen (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) (n : ℕ) :
    ∃ k, k < 13 ∧
      (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
        = Piece.I :=
  trace_exists_I_within_thirteen
    (fun k => C.legal_draw _ (C.trace_mem_states h0 k)) n

/-- The two-sided window law on policy traces: over any 35 consecutive
placements with legal draws, each piece is played between four and six
times. -/
theorem trace_window_piece_bounds {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace cfg π g0 k)).piece ∈ (trace cfg π g0 k).bag)
    (n : ℕ) (p : Piece) :
    4 ≤ ((Finset.range 35).filter (fun k =>
        (π (trace cfg π g0 (n + k))).piece = p)).card
      ∧ ((Finset.range 35).filter (fun k =>
        (π (trace cfg π g0 (n + k))).piece = p)).card ≤ 6 := by
  have hl := legalSequence_of_trace_draws hdraw
  exact ⟨BagCadence.window_thirtyfive_ge_four hl n p,
    BagCadence.window_thirtyfive_le_six hl n p⟩

/-- **The window law on the M2 artifact**: along a closed cycle's trajectory,
every 35-placement window plays each piece between four and six times — and
exactly five on full periods (`closedCycle_period_piece_balanced`). -/
theorem closedCycle_window_piece_bounds (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) (n : ℕ) (p : Piece) :
    4 ≤ ((Finset.range 35).filter (fun k =>
        (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
          = p)).card
      ∧ ((Finset.range 35).filter (fun k =>
        (C.policy (trace GameConfig.standard C.policy g0 (n + k))).piece
          = p)).card ≤ 6 :=
  trace_window_piece_bounds
    (fun k => C.legal_draw _ (C.trace_mem_states h0 k)) n p

/-- From any seed, trace boards after the first step carry no full row: every
successor board is a `clearLines` image. -/
theorem trace_board_no_full_of_pos {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {m : ℕ} (hm : 1 ≤ m) (r : ℕ) :
    ¬ Board.isFull cfg (trace cfg π g0 m).board r := by
  obtain ⟨k, rfl⟩ : ∃ k, m = k + 1 := ⟨m - 1, by omega⟩
  rw [trace_succ, GameState.step_board, Placement.applyStep_eq_clearLines_place]
  exact Board.clearLines_no_full _ cfg.cols_pos r

/-- A four-row clear at any positive trace step is played with an I — from any
seed, not just `init`. -/
theorem trace_tetris_step_I {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {m : ℕ} (hm : 1 ≤ m)
    (h4 : 4 ≤ (Board.fullRows cfg
      ((π (trace cfg π g0 m)).place (trace cfg π g0 m).board)).card) :
    (π (trace cfg π g0 m)).piece = Piece.I :=
  tetris_requires_I (fun r => trace_board_no_full_of_pos hm r) h4

/-- **At most six tetrises per 35 placements, along any legal trace** (windows
past the seed): four-row clears require an I and the window law caps the I's
at six. -/
theorem trace_window_tetris_le_six {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace cfg π g0 k)).piece ∈ (trace cfg π g0 k).bag)
    {n : ℕ} (hn : 1 ≤ n) :
    ((Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows cfg
        ((π (trace cfg π g0 (n + k))).place
          (trace cfg π g0 (n + k)).board)).card)).card ≤ 6 := by
  have hsub : (Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows cfg
        ((π (trace cfg π g0 (n + k))).place
          (trace cfg π g0 (n + k)).board)).card)
      ⊆ (Finset.range 35).filter (fun k =>
        (π (trace cfg π g0 (n + k))).piece = Piece.I) := by
    intro k hk
    obtain ⟨h1, h2⟩ := Finset.mem_filter.mp hk
    exact Finset.mem_filter.mpr ⟨h1, trace_tetris_step_I (by omega) h2⟩
  exact le_trans (Finset.card_le_card hsub)
    (trace_window_piece_bounds hdraw n Piece.I).2

/-- The tetris window cap along the M2 artifact's trajectory. -/
theorem closedCycle_window_tetris_le_six (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ} (hn : 1 ≤ n) :
    ((Finset.range 35).filter (fun k => 4 ≤ (Board.fullRows GameConfig.standard
        ((C.policy (trace GameConfig.standard C.policy g0 (n + k))).place
          (trace GameConfig.standard C.policy g0 (n + k)).board)).card)).card
      ≤ 6 :=
  trace_window_tetris_le_six
    (fun k => C.legal_draw _ (C.trace_mem_states h0 k)) hn

/-- **The windowed tetris–I embedding**: past the seed, the tetris increment
over any window never exceeds the I increment — each four-clear step *is* an
I step, windowed. -/
theorem tetris_le_I_window {π : Policy GameConfig.standard} {n m : ℕ}
    (hn : 1 ≤ n) (hnm : n ≤ m) :
    sizeCount GameConfig.standard π GameState.init 4 m
        - sizeCount GameConfig.standard π GameState.init 4 n
      ≤ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n := by
  classical
  have hsz := sizeCount_window (cfg := GameConfig.standard) (π := π) 4 n (m - n)
  rw [show n + (m - n) = m by omega] at hsz
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) m
  have hsplit := card_filter_range_add (fun k =>
    (π (trace GameConfig.standard π GameState.init k)).piece = Piece.I)
    n (m - n)
  rw [show n + (m - n) = m by omega] at hsplit
  have hsub : ((Finset.range (m - n)).filter (fun j =>
        (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init (n + j))).place
            (trace GameConfig.standard π GameState.init (n + j)).board)).card
          = 4)).card
      ≤ ((Finset.range (m - n)).filter (fun j =>
        (π (trace GameConfig.standard π GameState.init (n + j))).piece
          = Piece.I)).card := by
    apply Finset.card_le_card
    intro j hj
    rw [Finset.mem_filter] at hj ⊢
    exact ⟨hj.1, trace_tetris_step_I (by omega) (le_of_eq hj.2.symm)⟩
  omega

/-- **The tetris bracket at every horizon — no cycle needed**: on any
legally-drawn trace, at most `6⌊Δn/35⌋ + 6` tetrises past the seed, because
the I supply itself is capped. A cycle sharpens the slope to `3/35`
(`cycle_tetris_density`). -/
theorem tetris_bracket_any {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n m : ℕ}
    (hn : 1 ≤ n) (hnm : n ≤ m) :
    sizeCount GameConfig.standard π GameState.init 4 m
        - sizeCount GameConfig.standard π GameState.init 4 n
      ≤ 6 * ((m - n) / 35) + 6 := by
  have hemb := tetris_le_I_window (π := π) hn hnm
  obtain ⟨_, hI⟩ := iCount_bracket_any hdraw hnm
  omega

/-- **The per-piece bracket on cycles, at every horizon**: on a periodic
legally-drawn trace, every piece's count over `[n, m)` lies within
`[5⌊Δn/35⌋, 5⌊Δn/35⌋ + 5]` — frequency exactly `1/7` with error at most one
period's worth, sharpening the general `[4/35, 6/35]` sandwich. -/
theorem cycle_piece_bracket {π : Policy GameConfig.standard} {g0 : GameState}
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + 35)) (p : Piece) (m : ℕ) :
    5 * ((m - n) / 35)
        ≤ ((Finset.range (m - n)).filter (fun k =>
            (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card
      ∧ ((Finset.range (m - n)).filter (fun k =>
            (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card
        ≤ 5 * ((m - n) / 35) + 5 := by
  classical
  set j := (m - n) / 35 with hj
  have hbal := trace_multi_period_piece_balanced hdraw hcyc p j
  have hbal' := trace_multi_period_piece_balanced hdraw hcyc p (j + 1)
  have hin : ((Finset.range (35 * j)).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card
      ≤ ((Finset.range (m - n)).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card :=
    Finset.card_le_card (Finset.filter_subset_filter _
      (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega))
  have hout : ((Finset.range (m - n)).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card
      ≤ ((Finset.range (35 * (j + 1))).filter (fun k =>
        (π (trace GameConfig.standard π g0 (n + k))).piece = p)).card :=
    Finset.card_le_card (Finset.filter_subset_filter _
      (by intro x hx; rw [Finset.mem_range] at hx ⊢; omega))
  omega

/-- **The I-counter bracket on cycles**: `5⌊Δn/35⌋ ≤ ΔiCount ≤ 5⌊Δn/35⌋ + 5`
at every horizon — the cycle sharpens `iCount_bracket_any` to slope exactly
`1/7`. -/
theorem cycle_iCount_bracket {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m : ℕ}
    (hnm : n ≤ m) :
    5 * ((m - n) / 35) ≤ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
      ∧ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
        ≤ 5 * ((m - n) / 35) + 5 := by
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) m
  have hsplit := card_filter_range_add (fun k =>
    (π (trace GameConfig.standard π GameState.init k)).piece = Piece.I)
    n (m - n)
  rw [show n + (m - n) = m by omega] at hsplit
  have hbr := cycle_piece_bracket hdraw hcyc Piece.I m
  omega

/-- **The forward orbit of a cycle is its first 35 states**: past the entry
point every visited state already appears in the entry window — the orbit of
a five-bag cycle is a finite set traversed in lockstep. -/
theorem cycle_orbit_subset {π : Policy GameConfig.standard} {g0 : GameState}
    {n : ℕ}
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    trace GameConfig.standard π g0 m
      ∈ (Finset.range 35).image (fun k => trace GameConfig.standard π g0 (n + k)) := by
  classical
  have hq := trace_period_multiples π g0 hcyc ((m - n) / 35)
  have hshift := trace_eq_of_state_eq π g0 hq ((m - n) % 35)
  rw [Finset.mem_image]
  refine ⟨(m - n) % 35, Finset.mem_range.mpr (Nat.mod_lt _ (by omega)), ?_⟩
  rw [hshift, show n + (m - n) / 35 * 35 + (m - n) % 35 = m by omega]

/-- **Any 35 consecutive trace states are pairwise distinct** — the quantum
forbids any return shorter than five bags, so every 35-window is an injective
run. On a cycle, `cycle_orbit_subset` adds that every later state revisits
this window: **a minimal five-bag cycle visits exactly 35 states**, matching
`closedCycle_card_ge_thirtyfive` from the other side. -/
theorem trace_window_image_card_thirtyfive {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ) :
    ((Finset.range 35).image
      (fun k => trace GameConfig.standard π GameState.init (n + k))).card
      = 35 := by
  classical
  rw [Finset.card_image_of_injOn, Finset.card_range]
  intro i hi k hk hik
  rw [Finset.coe_range, Set.mem_Iio] at hi hk
  rcases Nat.lt_or_ge i k with hlt | hge
  · have hd := thirtyfive_dvd_of_trace_eq hv hdraw
      (show n + i ≤ n + k by omega) hik
    omega
  · rcases Nat.eq_or_lt_of_le hge with heq | hlt'
    · omega
    · have hd := thirtyfive_dvd_of_trace_eq hv hdraw
        (show n + k ≤ n + i by omega) hik.symm
      omega

/-- **The minimal sub-cycle**: a five-bag return inside a closed cycle carves
out a `ClosedCycle` of its own — the 35-state orbit window. Every field is
inherited from the ambient cycle; closure is `cycle_orbit_subset` applied to
the trace successor. -/
def orbitCycle (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ}
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) :
    ClosedCycle GameConfig.standard where
  states := (Finset.range 35).image
    (fun k => trace GameConfig.standard C.policy g0 (n + k))
  policy := C.policy
  valid := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.valid _ (C.trace_mem_states h0 (n + k))
  legal_draw := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.legal_draw _ (C.trace_mem_states h0 (n + k))
  not_lost := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.not_lost _ (C.trace_mem_states h0 (n + k))
  closed := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    have hstep : (trace GameConfig.standard C.policy g0 (n + k)).step
          GameConfig.standard
          (C.policy (trace GameConfig.standard C.policy g0 (n + k)))
        = trace GameConfig.standard C.policy g0 ((n + k) + 1) :=
      (trace_succ GameConfig.standard C.policy g0 (n + k)).symm
    rw [hstep]
    exact cycle_orbit_subset hcyc (by omega)

/-- The minimal sub-cycle's states sit inside the ambient cycle. -/
theorem orbitCycle_subset (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n : ℕ}
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) :
    (orbitCycle C h0 hcyc).states ⊆ C.states := by
  intro s hs
  have hs' : s ∈ (Finset.range 35).image
      (fun k => trace GameConfig.standard C.policy g0 (n + k)) := hs
  rw [Finset.mem_image] at hs'
  obtain ⟨k, -, rfl⟩ := hs'
  exact C.trace_mem_states h0 (n + k)

/-- **The minimal sub-cycle has exactly 35 states** (well-formed seed):
every closed cycle admitting a five-bag return contains a closed cycle of
the exact minimal size. -/
theorem orbitCycle_card (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) {n : ℕ}
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + 35)) :
    (orbitCycle C h0 hcyc).states.card = 35 := by
  classical
  change ((Finset.range 35).image
    (fun k => trace GameConfig.standard C.policy g0 (n + k))).card = 35
  rw [Finset.card_image_of_injOn, Finset.card_range]
  intro i hi k hk hik
  rw [Finset.coe_range, Set.mem_Iio] at hi hk
  rcases Nat.lt_or_ge i k with hlt | hge
  · have hd := closedCycle_thirtyfive_dvd C h0 hwf
      (show n + i ≤ n + k by omega) hik
    omega
  · rcases Nat.eq_or_lt_of_le hge with heq | hlt'
    · omega
    · have hd := closedCycle_thirtyfive_dvd C h0 hwf
        (show n + k ≤ n + i by omega) hik.symm
      omega

/-- **Every closed cycle's trace returns**: the trace lives inside the finite
state set, so within `card + 1` steps two indices collide — a return exists,
by pigeonhole, with both indices at most the cycle's size. -/
theorem closedCycle_exists_return (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) :
    ∃ n₁ n₂, n₁ < n₂ ∧ n₂ ≤ C.states.card ∧
      trace GameConfig.standard C.policy g0 n₁
        = trace GameConfig.standard C.policy g0 n₂ := by
  classical
  have hmaps : ∀ m ∈ Finset.range (C.states.card + 1),
      trace GameConfig.standard C.policy g0 m ∈ C.states :=
    fun m _ => C.trace_mem_states h0 m
  have hcard : C.states.card < (Finset.range (C.states.card + 1)).card := by
    rw [Finset.card_range]
    omega
  obtain ⟨i, hi, j, hj, hij, heq⟩ :=
    Finset.exists_ne_map_eq_of_card_lt_of_maps_to hcard hmaps
  rw [Finset.mem_range] at hi hj
  rcases Nat.lt_or_ge i j with hlt | hge
  · exact ⟨i, j, hlt, by omega, heq⟩
  · have hlt : j < i := by omega
    exact ⟨j, i, hlt, by omega, heq.symm⟩

/-- **Every closed cycle carries a five-bag-quantised period**: there is a
return of period `P` with `35 ∣ P`, `0 < P ≤ card`. The M2 artifact always
contains its own quantised loop — combined with `orbitCycle`, a cycle with a
*minimal* (35-placement) return contains the exactly-35-state certificate. -/
theorem closedCycle_exists_period (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) :
    ∃ n P, 0 < P ∧ P ≤ C.states.card ∧ 35 ∣ P ∧
      trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + P) := by
  obtain ⟨n₁, n₂, hlt, hle, heq⟩ := closedCycle_exists_return C h0
  refine ⟨n₁, n₂ - n₁, by omega, by omega, ?_, ?_⟩
  · exact closedCycle_thirtyfive_dvd C h0 hwf (le_of_lt hlt) heq
  · rw [show n₁ + (n₂ - n₁) = n₂ by omega]
    exact heq

/-- The orbit-window lemma at any period: past the entry point, a `P`-periodic
trace stays inside its `P`-state entry window. -/
theorem cycle_orbit_subset_period {π : Policy GameConfig.standard}
    {g0 : GameState} {n P : ℕ} (hP : 0 < P)
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + P)) {m : ℕ} (hnm : n ≤ m) :
    trace GameConfig.standard π g0 m
      ∈ (Finset.range P).image (fun k => trace GameConfig.standard π g0 (n + k)) := by
  classical
  have hq := trace_period_multiples π g0 hcyc ((m - n) / P)
  have hshift := trace_eq_of_state_eq π g0 hq ((m - n) % P)
  have hdm := Nat.div_add_mod (m - n) P
  have hidx : n + (m - n) / P * P + (m - n) % P = m := by
    rw [Nat.add_assoc, Nat.mul_comm, hdm]
    omega
  rw [Finset.mem_image]
  refine ⟨(m - n) % P, Finset.mem_range.mpr (Nat.mod_lt _ hP), ?_⟩
  rw [hshift, hidx]

/-- The orbit sub-cycle at any period: a `P`-periodic return inside a closed
cycle carves out a closed cycle on its `P`-state orbit window. -/
def orbitCycleP (C : ClosedCycle GameConfig.standard)
    {g0 : GameState} (h0 : g0 ∈ C.states) {n P : ℕ} (hP : 0 < P)
    (hcyc : trace GameConfig.standard C.policy g0 n
        = trace GameConfig.standard C.policy g0 (n + P)) :
    ClosedCycle GameConfig.standard where
  states := (Finset.range P).image
    (fun k => trace GameConfig.standard C.policy g0 (n + k))
  policy := C.policy
  valid := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.valid _ (C.trace_mem_states h0 (n + k))
  legal_draw := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.legal_draw _ (C.trace_mem_states h0 (n + k))
  not_lost := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact C.not_lost _ (C.trace_mem_states h0 (n + k))
  closed := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    have hstep : (trace GameConfig.standard C.policy g0 (n + k)).step
          GameConfig.standard
          (C.policy (trace GameConfig.standard C.policy g0 (n + k)))
        = trace GameConfig.standard C.policy g0 ((n + k) + 1) :=
      (trace_succ GameConfig.standard C.policy g0 (n + k)).symm
    rw [hstep]
    exact cycle_orbit_subset_period hP hcyc (by omega)

/-- **Every closed cycle contains a minimal orbit cycle**: a sub-`ClosedCycle`
whose state count is *exactly* the trace's minimal period — positive, a
multiple of 35, and at most the ambient size. The M2 artifact always contains
a tight certificate: no state is wasted, and its size still obeys the
quantum. -/
theorem closedCycle_contains_minimal_orbit
    (C : ClosedCycle GameConfig.standard) {g0 : GameState} (h0 : g0 ∈ C.states)
    (hwf : Board.WF GameConfig.standard g0.board) :
    ∃ D : ClosedCycle GameConfig.standard, D.states ⊆ C.states ∧
      0 < D.states.card ∧ 35 ∣ D.states.card ∧
      D.states.card ≤ C.states.card ∧
      ∀ s ∈ D.states, ∃ m, s = trace GameConfig.standard C.policy g0 m := by
  classical
  obtain ⟨n, P, hPpos, hPle, _, hret⟩ := closedCycle_exists_period C h0 hwf
  have hex : ∃ Q, 0 < Q ∧ ∃ m, trace GameConfig.standard C.policy g0 m
      = trace GameConfig.standard C.policy g0 (m + Q) := ⟨P, hPpos, n, hret⟩
  obtain ⟨hQpos, n₀, hret₀⟩ := Nat.find_spec hex
  have hQle : Nat.find hex ≤ P := Nat.find_min' hex ⟨hPpos, n, hret⟩
  refine ⟨orbitCycleP C h0 hQpos hret₀, ?_, ?_, ?_, ?_, ?_⟩
  · intro s hs
    have hs' : s ∈ (Finset.range (Nat.find hex)).image
        (fun k => trace GameConfig.standard C.policy g0 (n₀ + k)) := hs
    rw [Finset.mem_image] at hs'
    obtain ⟨k, -, rfl⟩ := hs'
    exact C.trace_mem_states h0 (n₀ + k)
  all_goals
    have hcard : (orbitCycleP C h0 hQpos hret₀).states.card = Nat.find hex := by
      change ((Finset.range (Nat.find hex)).image
        (fun k => trace GameConfig.standard C.policy g0 (n₀ + k))).card = _
      rw [Finset.card_image_of_injOn, Finset.card_range]
      intro i hi k hk hik
      rw [Finset.coe_range, Set.mem_Iio] at hi hk
      by_contra hne
      rcases Nat.lt_or_ge i k with hlt | hge
      · exact Nat.find_min hex (show k - i < Nat.find hex by omega)
          ⟨by omega, n₀ + i, by rw [show n₀ + i + (k - i) = n₀ + k by omega]; exact hik⟩
      · have hlt : k < i := by omega
        exact Nat.find_min hex (show i - k < Nat.find hex by omega)
          ⟨by omega, n₀ + k, by rw [show n₀ + k + (i - k) = n₀ + i by omega]; exact hik.symm⟩
  · rw [hcard]
    exact hQpos
  · rw [hcard]
    have hd := closedCycle_thirtyfive_dvd C h0 hwf
      (Nat.le_add_right n₀ (Nat.find hex)) hret₀
    simpa using hd
  · rw [hcard]
    omega
  · intro s hs
    have hs' : s ∈ (Finset.range (Nat.find hex)).image
        (fun k => trace GameConfig.standard C.policy g0 (n₀ + k)) := hs
    rw [Finset.mem_image] at hs'
    obtain ⟨k, -, rfl⟩ := hs'
    exact ⟨n₀ + k, rfl⟩

/-- **The M3 bridge**: a closed cycle through the initial state contains a
tight quantised sub-cycle every state of which is *reached from the empty
board* by an explicit trace index — the minimal certificate is not only
present but reachable. -/
theorem init_closedCycle_reachable_minimal_orbit
    (C : ClosedCycle GameConfig.standard) (h0 : GameState.init ∈ C.states) :
    ∃ D : ClosedCycle GameConfig.standard, D.states ⊆ C.states ∧
      0 < D.states.card ∧ 35 ∣ D.states.card ∧
      D.states.card ≤ C.states.card ∧
      ∀ s ∈ D.states, ∃ m,
        s = trace GameConfig.standard C.policy GameState.init m :=
  closedCycle_contains_minimal_orbit C h0
    (GameState.init_board_wf GameConfig.standard)

/-- **The whole tail is periodic**: one 35-return makes the trace 35-periodic
at every later index, not just at the entry point — determinism carries the
return forward. The packaged form of `trace_eq_of_state_eq` for cycles. -/
theorem trace_tail_periodic {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n : ℕ}
    (hcyc : trace cfg π g0 n = trace cfg π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    trace cfg π g0 m = trace cfg π g0 (m + 35) := by
  have h := trace_eq_of_state_eq π g0 hcyc (m - n)
  rw [show n + (m - n) = m by omega] at h
  rw [show n + 35 + (m - n) = m + 35 by omega] at h
  exact h

/-- Every scalar face of the state is periodic on the tail — occupancy in
particular: `count(m + 35) = count(m)` for all `m ≥ n`. -/
theorem cycle_count_periodic {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n : ℕ}
    (hcyc : trace cfg π g0 n = trace cfg π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    (trace cfg π g0 (m + 35)).board.count = (trace cfg π g0 m).board.count := by
  rw [← trace_tail_periodic hcyc hnm]

/-- **A cycle is its own periodic adversary**: the piece stream a cooperative
cycle deals itself is 35-periodic on the tail — exactly the `hper` hypothesis
the adversarial multi-period theory requires. The two theories meet: any
concrete cooperative loop witness automatically provides a periodic
adversarial stream. -/
theorem cycle_piece_stream_periodic {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n : ℕ}
    (hcyc : trace cfg π g0 n = trace cfg π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    (π (trace cfg π g0 (m + 35))).piece = (π (trace cfg π g0 m)).piece := by
  rw [← trace_tail_periodic hcyc hnm]

/-- **Every point of a cycle is an anchor**, so the anchored clearing bracket
holds verbatim from any `m₀ ≥ n`: `14⌊w/35⌋ ≤ Δcleared ≤ 14⌊w/35⌋ + 14` —
strictly sharper than subtracting boundary brackets. -/
theorem cycle_clears_bracket_stationary {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) (w : ℕ) :
    14 * (w / 35)
        ≤ cleared GameConfig.standard π GameState.init (m₀ + w)
          - cleared GameConfig.standard π GameState.init m₀
      ∧ cleared GameConfig.standard π GameState.init (m₀ + w)
          - cleared GameConfig.standard π GameState.init m₀
        ≤ 14 * (w / 35) + 14 := by
  have hbr := cycle_clears_bracket hv (trace_tail_periodic hcyc hm)
    (Nat.le_add_right m₀ w)
  rw [show m₀ + w - m₀ = w by omega] at hbr
  exact hbr

/-- The tetris density from every anchor: `≤ 3⌊w/35⌋ + 3` in every window. -/
theorem cycle_tetris_density_stationary {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) (w : ℕ) :
    sizeCount GameConfig.standard π GameState.init 4 (m₀ + w)
        - sizeCount GameConfig.standard π GameState.init 4 m₀
      ≤ 3 * (w / 35) + 3 := by
  have hd := cycle_tetris_density hv (trace_tail_periodic hcyc hm)
    (Nat.le_add_right m₀ w)
  rw [show m₀ + w - m₀ = w by omega] at hd
  exact hd

/-- **The sharp mass diameter**: between any two ordered horizons of a cycle
the occupancy moves by at most `+136 / −140` cells — the 276-cell corridor
(`cycle_mass_diameter`) tightened to the band's own asymmetric width. -/
theorem cycle_mass_diameter_sharp {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₁ m₂ : ℕ}
    (h1 : n ≤ m₁) (h12 : m₁ ≤ m₂) :
    (trace GameConfig.standard π GameState.init m₂).board.count
        ≤ (trace GameConfig.standard π GameState.init m₁).board.count + 136
      ∧ (trace GameConfig.standard π GameState.init m₁).board.count
        ≤ (trace GameConfig.standard π GameState.init m₂).board.count + 140 :=
  cycle_mass_band hv (trace_tail_periodic hcyc h1) h12

/-- **Every 35-window of a cycle clears exactly fourteen rows** — from any
starting point, not just the entry boundary: every point is an anchor and
the ledger balances exactly on each return. -/
theorem cycle_window_clears_exact {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) :
    cleared GameConfig.standard π GameState.init (m₀ + 35)
      - cleared GameConfig.standard π GameState.init m₀ = 14 :=
  trace_eq_thirtyfive_clears_fourteen hv (trace_tail_periodic hcyc hm)

/-- **Dry spells on a cycle last at most 34 placements** — the sharp form:
any 35 consecutive placements clear (exactly fourteen rows), so a clear-free
stretch never reaches one full period. Halves the pre-anchor 68 bound. -/
theorem cycle_dry_spell_le_thirtyfour {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ L : ℕ}
    (hm : n ≤ m₀)
    (hdry : cleared GameConfig.standard π GameState.init (m₀ + L)
      = cleared GameConfig.standard π GameState.init m₀) :
    L ≤ 34 := by
  by_contra hcon
  have hex := cycle_window_clears_exact hv hcyc hm
  have hmono := cleared_mono GameConfig.standard π GameState.init
    (show m₀ + 35 ≤ m₀ + L by omega)
  have hmono0 := cleared_mono GameConfig.standard π GameState.init
    (Nat.le_add_right m₀ 35)
  omega

/-- **Every 35-window of a cycle is piece-balanced**: from any starting point
each of the seven pieces is dealt exactly five times — the cycle is
statistically homogeneous, window by window. -/
theorem cycle_window_piece_balanced_stationary {π : Policy GameConfig.standard}
    {g0 : GameState}
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π g0 n
        = trace GameConfig.standard π g0 (n + 35)) {m₀ : ℕ} (hm : n ≤ m₀)
    (p : Piece) :
    ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π g0 (m₀ + k))).piece = p)).card = 5 :=
  trace_period_piece_balanced hdraw (trace_tail_periodic hcyc hm) p

/-- Every 35-window's clear-size mix weight-sums to exactly fourteen, from
any starting point. -/
theorem cycle_window_mix_stationary {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) :
    (sizeCount GameConfig.standard π GameState.init 1 (m₀ + 35)
        - sizeCount GameConfig.standard π GameState.init 1 m₀)
      + 2 * (sizeCount GameConfig.standard π GameState.init 2 (m₀ + 35)
        - sizeCount GameConfig.standard π GameState.init 2 m₀)
      + 3 * (sizeCount GameConfig.standard π GameState.init 3 (m₀ + 35)
        - sizeCount GameConfig.standard π GameState.init 3 m₀)
      + 4 * (sizeCount GameConfig.standard π GameState.init 4 (m₀ + 35)
        - sizeCount GameConfig.standard π GameState.init 4 m₀)
      = 14 :=
  period_mix_fourteen hv (trace_tail_periodic hcyc hm)

/-- Every 35-window of a cycle holds at most three tetrises, from any
starting point — sharper on cycles than the general six-per-window law. -/
theorem cycle_window_tetris_le_three_stationary {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) :
    sizeCount GameConfig.standard π GameState.init 4 (m₀ + 35)
      - sizeCount GameConfig.standard π GameState.init 4 m₀ ≤ 3 :=
  period_tetris_le_three hv (trace_tail_periodic hcyc hm)

/-- Every 35-window of a cycle clears on 4–14 of its placements, from any
starting point — silence dominates every window, not just the periods. -/
theorem cycle_window_events_stationary {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) {m₀ : ℕ}
    (hm : n ≤ m₀) :
    4 ≤ (sizeCount GameConfig.standard π GameState.init 1 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 1 m₀)
        + (sizeCount GameConfig.standard π GameState.init 2 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 2 m₀)
        + (sizeCount GameConfig.standard π GameState.init 3 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 3 m₀)
        + (sizeCount GameConfig.standard π GameState.init 4 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 4 m₀)
      ∧ (sizeCount GameConfig.standard π GameState.init 1 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 1 m₀)
        + (sizeCount GameConfig.standard π GameState.init 2 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 2 m₀)
        + (sizeCount GameConfig.standard π GameState.init 3 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 3 m₀)
        + (sizeCount GameConfig.standard π GameState.init 4 (m₀ + 35)
          - sizeCount GameConfig.standard π GameState.init 4 m₀)
        ≤ 14 :=
  period_clear_events_bounds hv (trace_tail_periodic hcyc hm)

/-- **Any 35-window shows at least five distinct boards**: equal boards force
equal counts, the mass clock forces index separation in multiples of five, a
residue class meets a 35-range at most seven times — pigeonhole. -/
theorem trace_window_boards_ge_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    5 ≤ ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π GameState.init (n + k)).board)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 7 * ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π GameState.init (n + k)).board)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).board = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).board = a),
        i / 5 = k / 5 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have hcnt : (trace GameConfig.standard π GameState.init (n + i)).board.count
          = (trace GameConfig.standard π GameState.init (n + k)).board.count := by
        rw [hi.2, hk.2]
      rcases Nat.lt_or_ge k i with hik | hik
      · have hd := (five_dvd_of_count_eq hv
          (show n + k ≤ n + i by omega) hcnt.symm).1
        omega
      · have hd := (five_dvd_of_count_eq hv
          (show n + i ≤ n + k by omega) hcnt).1
        omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).board = a),
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

/-- **Any 35-window shows at least seven distinct bag states** — tight: the
bag clock cycles through exactly seven fill levels. Equal bags force equal
cards, the card determines the index mod 7, and a residue class meets a
35-range at most five times. -/
theorem trace_window_bags_ge_seven {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ) :
    7 ≤ ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π GameState.init (n + k)).bag)).card := by
  classical
  by_contra hcon
  push Not at hcon
  have hle : (Finset.range 35).card ≤ 5 * ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π GameState.init (n + k)).bag)).card := by
    apply Finset.card_le_mul_card_image
    intro a ha
    have hinj : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).bag = a),
        ∀ k ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).bag = a),
        i / 7 = k / 7 → i = k := by
      intro i hi k hk hdiv
      rw [Finset.mem_filter, Finset.mem_range] at hi hk
      have hc1 := bag_card_trace hdraw (n + i)
      have hc2 := bag_card_trace hdraw (n + k)
      rw [hi.2] at hc1
      rw [hk.2] at hc2
      omega
    have hmap : ∀ i ∈ (Finset.range 35).filter (fun k =>
          (trace GameConfig.standard π GameState.init (n + k)).bag = a),
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

/-- **At least two of every period's five I's are idle**: a cycle period
deals exactly five I pieces but its row budget admits at most three
tetrises, so at least two I placements per period do lesser work. The I
piece cannot serve as a pure tetris tool on any cycle. -/
theorem period_idle_I_ge_two {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hn : 1 ≤ n)
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    2 ≤ ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.I
          ∧ (Board.fullRows GameConfig.standard
              ((π (trace GameConfig.standard π GameState.init (n + k))).place
                (trace GameConfig.standard π GameState.init (n + k)).board)).card
            ≠ 4)).card := by
  classical
  have hbal := trace_period_piece_balanced hdraw hcyc Piece.I
  -- split the five I's by whether they clear four rows
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := (Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece = Piece.I))
    (p := fun k => (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init (n + k))).place
        (trace GameConfig.standard π GameState.init (n + k)).board)).card = 4)
  rw [Finset.filter_filter, Finset.filter_filter, hbal] at hsplit
  -- the (I ∧ 4-clear) fiber is exactly the tetris fiber
  have hIfour : ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.I
          ∧ (Board.fullRows GameConfig.standard
              ((π (trace GameConfig.standard π GameState.init (n + k))).place
                (trace GameConfig.standard π GameState.init (n + k)).board)).card
            = 4)).card
      = ((Finset.range 35).filter (fun k =>
        (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init (n + k))).place
            (trace GameConfig.standard π GameState.init (n + k)).board)).card
          = 4)).card := by
    congr 1
    apply Finset.filter_congr
    intro k hk
    constructor
    · intro h
      exact h.2
    · intro h
      exact ⟨trace_tetris_step_I (by omega) (le_of_eq h.symm), h⟩
  have hszw := sizeCount_window (cfg := GameConfig.standard) (π := π) 4 n 35
  have htet := period_tetris_le_three hv hcyc
  simp only [ne_eq] at hsplit hIfour ⊢
  omega

/-- The bag is periodic on the tail of a cycle. -/
theorem cycle_bag_periodic {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n : ℕ}
    (hcyc : trace cfg π g0 n = trace cfg π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    (trace cfg π g0 (m + 35)).bag = (trace cfg π g0 m).bag := by
  rw [← trace_tail_periodic hcyc hnm]

/-- **The clearing pattern loops**: the number of rows cleared at step
`m + 35` equals the number cleared at step `m`, for every `m ≥ n` — the
observable clear-size sequence of a cycle is 35-periodic. -/
theorem cycle_clear_size_periodic {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n : ℕ}
    (hcyc : trace cfg π g0 n = trace cfg π g0 (n + 35)) {m : ℕ} (hnm : n ≤ m) :
    (Board.fullRows cfg ((π (trace cfg π g0 (m + 35))).place
        (trace cfg π g0 (m + 35)).board)).card
      = (Board.fullRows cfg ((π (trace cfg π g0 m)).place
        (trace cfg π g0 m).board)).card := by
  rw [← trace_tail_periodic hcyc hnm]

set_option maxRecDepth 4000 in
/-- **Windowed size counters are shift-invariant by a period**: the number of
`k`-row clears in `[m + 35, m + 35 + w)` equals the number in `[m, m + w)` —
every statistic of the clearing process repeats verbatim one period later. -/
theorem cycle_window_sizeCount_shift {π : Policy GameConfig.standard} {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) (k : ℕ) {m : ℕ}
    (hnm : n ≤ m) (w : ℕ) :
    sizeCount GameConfig.standard π GameState.init k (m + 35 + w)
        - sizeCount GameConfig.standard π GameState.init k (m + 35)
      = sizeCount GameConfig.standard π GameState.init k (m + w)
        - sizeCount GameConfig.standard π GameState.init k m := by
  classical
  have h1 := sizeCount_window (cfg := GameConfig.standard) (π := π) k (m + 35) w
  have h2 := sizeCount_window (cfg := GameConfig.standard) (π := π) k m w
  rw [h1, h2]
  refine congrArg Finset.card (Finset.filter_congr ?_)
  intro j hj
  have hper := cycle_clear_size_periodic (π := π) (g0 := GameState.init) hcyc
    (show n ≤ m + j by omega)
  rw [show m + 35 + j = (m + j) + 35 by omega, hper]

/-- **A tetris-only clearer can never cycle**: a policy whose every clearing
step is a four-clear admits no 35-return — `4 ∤ 14`. Impossibility at the
solver-design level: perpetual pure-tetris play is not a loop. -/
theorem no_cycle_of_only_tetris_clears {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hpol : ∀ m, (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 0
      ∨ (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 4)
    (n : ℕ) :
    trace GameConfig.standard π GameState.init n
      ≠ trace GameConfig.standard π GameState.init (n + 35) := by
  classical
  intro hcyc
  have hzero : ∀ k, k = 1 ∨ k = 2 ∨ k = 3 →
      sizeCount GameConfig.standard π GameState.init k (n + 35)
        = sizeCount GameConfig.standard π GameState.init k n := by
    intro k hk
    have hw := sizeCount_window (cfg := GameConfig.standard) (π := π) k n 35
    have hmono := sizeCount_mono GameConfig.standard π GameState.init k
      (Nat.le_add_right n 35)
    have hempty : ((Finset.range 35).filter (fun j =>
        (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init (n + j))).place
            (trace GameConfig.standard π GameState.init (n + j)).board)).card
          = k)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro j hj
      rcases hpol (n + j) with h0 | h4
      · rw [h0]
        omega
      · rw [h4]
        omega
    omega
  exact no_pure_tetris_period hv hcyc (hzero 1 (by omega)) (hzero 2 (by omega))
    (hzero 3 (by omega))

/-- A triple-only clearer can never cycle either: `3 ∤ 14`. -/
theorem no_cycle_of_only_triple_clears {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hpol : ∀ m, (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 0
      ∨ (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card = 3)
    (n : ℕ) :
    trace GameConfig.standard π GameState.init n
      ≠ trace GameConfig.standard π GameState.init (n + 35) := by
  classical
  intro hcyc
  have hzero : ∀ k, k = 1 ∨ k = 2 ∨ k = 4 →
      sizeCount GameConfig.standard π GameState.init k (n + 35)
        = sizeCount GameConfig.standard π GameState.init k n := by
    intro k hk
    have hw := sizeCount_window (cfg := GameConfig.standard) (π := π) k n 35
    have hmono := sizeCount_mono GameConfig.standard π GameState.init k
      (Nat.le_add_right n 35)
    have hempty : ((Finset.range 35).filter (fun j =>
        (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init (n + j))).place
            (trace GameConfig.standard π GameState.init (n + j)).board)).card
          = k)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro j hj
      rcases hpol (n + j) with h0 | h3
      · rw [h0]
        omega
      · rw [h3]
        omega
    omega
  exact no_pure_triple_period hv hcyc (hzero 1 (by omega)) (hzero 2 (by omega))
    (hzero 4 (by omega))

/-- **A policy that plays I only for tetrises can never cycle** (past the
seed): every period must contain at least two idle I's, but this discipline
forbids them. The contrapositive of `period_idle_I_ge_two` as a solver-design
impossibility. -/
theorem no_cycle_of_I_strictly_tetris {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag)
    (hpol : ∀ m, (π (trace GameConfig.standard π GameState.init m)).piece
        = Piece.I
      → (Board.fullRows GameConfig.standard
          ((π (trace GameConfig.standard π GameState.init m)).place
            (trace GameConfig.standard π GameState.init m).board)).card = 4)
    {n : ℕ} (hn : 1 ≤ n) :
    trace GameConfig.standard π GameState.init n
      ≠ trace GameConfig.standard π GameState.init (n + 35) := by
  classical
  intro hcyc
  have hidle := period_idle_I_ge_two hv hdraw hn hcyc
  have hempty : ((Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece = Piece.I
        ∧ (Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init (n + k))).place
              (trace GameConfig.standard π GameState.init (n + k)).board)).card
          ≠ 4)).card = 0 := by
    rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro j hj hcon
    exact hcon.2 (hpol (n + j) hcon.1)
  omega

/-- **Three tetrises span at least seven placements**: tetrises ride I's,
three draws of one piece must straddle two bag refills, and refills sit
seven apart. Tetris bursts come at most in pairs — a triple burst inside
seven placements is impossible on any legally-drawn trace. -/
theorem trace_three_tetrises_span {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag)
    {i j k : ℕ} (hi : 1 ≤ i) (hij : i < j) (hjk : j < k)
    (h4i : 4 ≤ (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init i)).place
        (trace GameConfig.standard π GameState.init i).board)).card)
    (h4j : 4 ≤ (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init j)).place
        (trace GameConfig.standard π GameState.init j).board)).card)
    (h4k : 4 ≤ (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init k)).place
        (trace GameConfig.standard π GameState.init k).board)).card) :
    i + 7 ≤ k := by
  have hl := legalSequence_of_trace_draws hdraw
  exact BagCadence.same_piece_three_apart hl
    (trace_tetris_step_I (by omega) h4i)
    (trace_tetris_step_I (by omega) h4j)
    (trace_tetris_step_I (by omega) h4k) hij hjk

/-- Two perfect clears sit a multiple of five placements apart — the mass
clock reads zero at both. -/
theorem perfect_clear_spacing {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {n₁ n₂ : ℕ} (h12 : n₁ ≤ n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0) :
    5 ∣ (n₂ - n₁) :=
  (five_dvd_of_count_eq hv h12 (by rw [h1, h2])).1

/-- **Aligned perfect clears close the loop**: empty boards with equal bags
are the *same state* — a policy that perfect-clears twice at the same bag
phase has returned. -/
theorem perfect_clear_pair_return {π : Policy GameConfig.standard} {n₁ n₂ : ℕ}
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag) :
    trace GameConfig.standard π GameState.init n₁
      = trace GameConfig.standard π GameState.init n₂ := by
  have hb : (trace GameConfig.standard π GameState.init n₁).board
      = (trace GameConfig.standard π GameState.init n₂).board := by
    rw [(Board.count_eq_zero_iff_eq_empty _).mp h1,
      (Board.count_eq_zero_iff_eq_empty _).mp h2]
  calc trace GameConfig.standard π GameState.init n₁
      = ⟨(trace GameConfig.standard π GameState.init n₁).board,
        (trace GameConfig.standard π GameState.init n₁).bag⟩ := rfl
    _ = ⟨(trace GameConfig.standard π GameState.init n₂).board,
        (trace GameConfig.standard π GameState.init n₂).bag⟩ := by
        rw [hb, hbag]
    _ = trace GameConfig.standard π GameState.init n₂ := rfl

/-- **Two aligned perfect clears prove infinite play**: if the policy stays
live between them, it stays live forever — the segment loops. A concrete,
checkable route to an M2 certificate: exhibit one live perfect-clear-to-
perfect-clear segment at matching bag phase. -/
theorem survives_forever_of_perfect_clear_pair {π : Policy GameConfig.standard}
    {n₁ n₂ : ℕ} (hlt : n₁ < n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag)
    (hlive : ∀ k, k < n₂ - n₁ →
      ¬ (trace GameConfig.standard π GameState.init (n₁ + k)).lost
        GameConfig.standard) :
    ∀ m, n₁ ≤ m →
      ¬ (trace GameConfig.standard π GameState.init m).lost
        GameConfig.standard := by
  intro m hm
  have hret : trace GameConfig.standard π GameState.init n₁
      = trace GameConfig.standard π GameState.init (n₁ + (n₂ - n₁)) := by
    rw [show n₁ + (n₂ - n₁) = n₂ by omega]
    exact perfect_clear_pair_return h1 h2 hbag
  have hmem := cycle_orbit_subset_period (show 0 < n₂ - n₁ by omega) hret hm
  rw [Finset.mem_image] at hmem
  obtain ⟨k, hk, heq⟩ := hmem
  rw [← heq]
  exact hlive k (Finset.mem_range.mp hk)

/-- **The perfect-clear cycle**: a live perfect-clear-to-perfect-clear
segment at matching bag phase, packaged as a genuine `ClosedCycle` — the M2
artifact itself, constructed from one checkable trace segment. -/
def perfectClearCycle {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n₁ n₂ : ℕ}
    (hlt : n₁ < n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag)
    (hlive : ∀ k, k < n₂ - n₁ →
      ¬ (trace GameConfig.standard π GameState.init (n₁ + k)).lost
        GameConfig.standard) :
    ClosedCycle GameConfig.standard where
  states := (Finset.range (n₂ - n₁)).image
    (fun k => trace GameConfig.standard π GameState.init (n₁ + k))
  policy := π
  valid := fun s _ => hv s
  legal_draw := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    rw [Bag.canDraw_iff_mem]
    exact hdraw (n₁ + k)
  not_lost := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, hk, rfl⟩ := hs
    exact hlive k (Finset.mem_range.mp hk)
  closed := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    have hret : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init (n₁ + (n₂ - n₁)) := by
      rw [show n₁ + (n₂ - n₁) = n₂ by omega]
      exact perfect_clear_pair_return h1 h2 hbag
    have hstep : (trace GameConfig.standard π GameState.init (n₁ + k)).step
          GameConfig.standard
          (π (trace GameConfig.standard π GameState.init (n₁ + k)))
        = trace GameConfig.standard π GameState.init ((n₁ + k) + 1) :=
      (trace_succ GameConfig.standard π GameState.init (n₁ + k)).symm
    rw [hstep]
    exact cycle_orbit_subset_period (show 0 < n₂ - n₁ by omega) hret (by omega)

/-- The first perfect clear is in the cycle. -/
theorem mem_perfectClearCycle {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n₁ n₂ : ℕ}
    (hlt : n₁ < n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag)
    (hlive : ∀ k, k < n₂ - n₁ →
      ¬ (trace GameConfig.standard π GameState.init (n₁ + k)).lost
        GameConfig.standard) :
    trace GameConfig.standard π GameState.init n₁
      ∈ (perfectClearCycle hv hdraw hlt h1 h2 hbag hlive).states := by
  have : trace GameConfig.standard π GameState.init n₁
      ∈ (Finset.range (n₂ - n₁)).image
        (fun k => trace GameConfig.standard π GameState.init (n₁ + k)) := by
    rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), by rw [Nat.add_zero]⟩
  exact this

/-- The perfect-clear cycle holds at least 35 states — the quantum floor on
the constructed certificate. -/
theorem perfectClearCycle_card_ge {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n₁ n₂ : ℕ}
    (hlt : n₁ < n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag)
    (hlive : ∀ k, k < n₂ - n₁ →
      ¬ (trace GameConfig.standard π GameState.init (n₁ + k)).lost
        GameConfig.standard) :
    35 ≤ (perfectClearCycle hv hdraw hlt h1 h2 hbag hlive).states.card :=
  closedCycle_card_ge_thirtyfive _
    (mem_perfectClearCycle hv hdraw hlt h1 h2 hbag hlive)
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) n₁)

/-- **Survival from finitely much evidence**: liveness on the finite prefix
`[0, n₂)` plus two aligned perfect clears proves `SurvivesForever` outright —
every hypothesis of this theorem is checkable by running the policy for `n₂`
steps. The cooperative infinite-play certificate reduced to finite data. -/
theorem survivesForever_of_perfect_clear_pair {π : Policy GameConfig.standard}
    {n₁ n₂ : ℕ} (hlt : n₁ < n₂)
    (h1 : (trace GameConfig.standard π GameState.init n₁).board.count = 0)
    (h2 : (trace GameConfig.standard π GameState.init n₂).board.count = 0)
    (hbag : (trace GameConfig.standard π GameState.init n₁).bag
        = (trace GameConfig.standard π GameState.init n₂).bag)
    (hlive : ∀ k, k < n₂ →
      ¬ (trace GameConfig.standard π GameState.init k).lost
        GameConfig.standard) :
    SurvivesForever GameConfig.standard π GameState.init := by
  intro m
  rcases Nat.lt_or_ge m n₁ with hm | hm
  · exact hlive m (by omega)
  · exact survives_forever_of_perfect_clear_pair hlt h1 h2 hbag
      (fun k hk => hlive (n₁ + k) (by omega)) m hm

/-- **Any live return proves survival**: a trace that revisits a state after
a live prefix survives forever — the segment loops. The general form behind
the perfect-clear route. -/
theorem survivesForever_of_trace_return {π : Policy GameConfig.standard}
    {n₁ n₂ : ℕ} (hlt : n₁ < n₂)
    (hret : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂)
    (hlive : ∀ k, k < n₂ →
      ¬ (trace GameConfig.standard π GameState.init k).lost
        GameConfig.standard) :
    SurvivesForever GameConfig.standard π GameState.init := by
  intro m
  rcases Nat.lt_or_ge m n₁ with hm | hm
  · exact hlive m (by omega)
  · have hret' : trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init (n₁ + (n₂ - n₁)) := by
      rw [show n₁ + (n₂ - n₁) = n₂ by omega]
      exact hret
    have hmem := cycle_orbit_subset_period (show 0 < n₂ - n₁ by omega) hret' hm
    rw [Finset.mem_image] at hmem
    obtain ⟨k, hk, heq⟩ := hmem
    rw [← heq]
    exact hlive (n₁ + k) (by have := Finset.mem_range.mp hk; omega)

/-- **A surviving trace must return**: the live states are in-field and
well-formed, hence finitely many — pigeonhole. -/
theorem survivesForever_exists_return {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hs : SurvivesForever GameConfig.standard π GameState.init) :
    ∃ n₁ n₂, n₁ < n₂ ∧ trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂ := by
  have hfin : (Set.range (trace GameConfig.standard π GameState.init)).Finite := by
    apply Set.Finite.subset (Set.finite_univ.image
      (fun q : InFieldBoard GameConfig.standard × Bag =>
        GameState.mk q.1.val q.2))
    rintro g ⟨n, rfl⟩
    have hwf := trace_board_wf hv
      (GameState.init_board_wf GameConfig.standard) n
    have hif : ∀ p ∈ (trace GameConfig.standard π GameState.init n).board,
        p.2 < GameConfig.standard.rows :=
      (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp (hs n)
    exact ⟨(⟨(trace GameConfig.standard π GameState.init n).board, hwf, hif⟩,
      (trace GameConfig.standard π GameState.init n).bag), Set.mem_univ _, rfl⟩
  have hninj : ¬ Function.Injective
      (trace GameConfig.standard π GameState.init) := by
    intro hinj
    exact Set.infinite_range_of_injective hinj hfin
  rw [Function.not_injective_iff] at hninj
  obtain ⟨a, b, heq, hne⟩ := hninj
  rcases Nat.lt_or_ge a b with hab | hab
  · exact ⟨a, b, hab, heq⟩
  · exact ⟨b, a, by omega, heq.symm⟩

/-- **The characterization: cooperative survival ⟺ a live return.** A valid
policy survives forever from the empty board iff its trace exhibits a
finite live prefix ending in a state revisit — infinite play is equivalent
to finite, checkable evidence. -/
theorem survivesForever_iff_live_return {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) :
    SurvivesForever GameConfig.standard π GameState.init
      ↔ ∃ n₁ n₂, n₁ < n₂
          ∧ trace GameConfig.standard π GameState.init n₁
            = trace GameConfig.standard π GameState.init n₂
          ∧ ∀ k, k < n₂ →
            ¬ (trace GameConfig.standard π GameState.init k).lost
              GameConfig.standard := by
  constructor
  · intro hs
    obtain ⟨n₁, n₂, hlt, hret⟩ := survivesForever_exists_return hv hs
    exact ⟨n₁, n₂, hlt, hret, fun k _ => hs k⟩
  · rintro ⟨n₁, n₂, hlt, hret, hlive⟩
    exact survivesForever_of_trace_return hlt hret hlive

/-- The full state space of live play: `2^200` in-field boards times `2^7`
bags is `2^207` states. -/
theorem card_infield_times_bag :
    Fintype.card (InFieldBoard GameConfig.standard × Bag) = 2 ^ 207 := by
  rw [Fintype.card_prod, InFieldBoard.standard_fintype_card]
  have hbag : Fintype.card Bag = 2 ^ 7 := by
    rw [Fintype.card_finset]
    congr 1
  rw [hbag, ← pow_add]

/-- **The return is bounded**: a surviving valid trace revisits a state
within `2^207` steps, and the revisit separation is a positive multiple of
35 — the characterization of survival made quantitative. Deciding
cooperative survival is a finite (astronomically finite) computation. -/
theorem survivesForever_return_within {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag)
    (hs : SurvivesForever GameConfig.standard π GameState.init) :
    ∃ n₁ n₂, n₁ < n₂ ∧ n₂ ≤ 2 ^ 207 ∧
      trace GameConfig.standard π GameState.init n₁
        = trace GameConfig.standard π GameState.init n₂
      ∧ 35 ∣ (n₂ - n₁) := by
  classical
  have hwf : ∀ n, Board.WF GameConfig.standard
      (trace GameConfig.standard π GameState.init n).board :=
    trace_board_wf hv (GameState.init_board_wf GameConfig.standard)
  have hif : ∀ n, ∀ p ∈ (trace GameConfig.standard π GameState.init n).board,
      p.2 < GameConfig.standard.rows :=
    fun n => (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp
      (hs n)
  set f : Fin (2 ^ 207 + 1) → InFieldBoard GameConfig.standard × Bag :=
    fun i => (⟨(trace GameConfig.standard π GameState.init i).board,
      hwf i, hif i⟩, (trace GameConfig.standard π GameState.init i).bag)
    with hf
  have hlt : Fintype.card (InFieldBoard GameConfig.standard × Bag)
      < Fintype.card (Fin (2 ^ 207 + 1)) := by
    rw [Fintype.card_fin, card_infield_times_bag]
    exact Nat.lt_succ_self _
  obtain ⟨i, j, hne, hfeq⟩ := Fintype.exists_ne_map_eq_of_card_lt f hlt
  have hstates : trace GameConfig.standard π GameState.init i
      = trace GameConfig.standard π GameState.init j := by
    have hb : (trace GameConfig.standard π GameState.init i).board
        = (trace GameConfig.standard π GameState.init j).board :=
      congrArg (fun q : InFieldBoard GameConfig.standard × Bag => q.1.val) hfeq
    have hg : (trace GameConfig.standard π GameState.init i).bag
        = (trace GameConfig.standard π GameState.init j).bag :=
      congrArg Prod.snd hfeq
    calc trace GameConfig.standard π GameState.init i
        = ⟨(trace GameConfig.standard π GameState.init i).board,
          (trace GameConfig.standard π GameState.init i).bag⟩ := rfl
      _ = ⟨(trace GameConfig.standard π GameState.init j).board,
          (trace GameConfig.standard π GameState.init j).bag⟩ := by
          rw [hb, hg]
      _ = trace GameConfig.standard π GameState.init j := rfl
  have hij : (i : ℕ) ≠ (j : ℕ) := fun h => hne (Fin.ext h)
  rcases Nat.lt_or_ge (i : ℕ) (j : ℕ) with hlt' | hge
  · refine ⟨i, j, hlt', by have := j.isLt; omega, hstates, ?_⟩
    exact thirtyfive_dvd_of_trace_eq hv hdraw (le_of_lt hlt') hstates
  · have hlt'' : (j : ℕ) < (i : ℕ) := by omega
    refine ⟨j, i, hlt'', by have := i.isLt; omega, hstates.symm, ?_⟩
    exact thirtyfive_dvd_of_trace_eq hv hdraw (le_of_lt hlt'') hstates.symm

/-- **Survival is a bounded formula**: `SurvivesForever` — a Π⁰₁ statement
over all time — is equivalent to a statement whose every quantifier is
bounded by `2^207`. Cooperative solvability of a fixed policy carries no
logical strength beyond finite verification. -/
theorem survivesForever_iff_bounded_evidence {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) :
    SurvivesForever GameConfig.standard π GameState.init
      ↔ ∃ n₂, n₂ ≤ 2 ^ 207 ∧ ∃ n₁, n₁ < n₂
          ∧ trace GameConfig.standard π GameState.init n₁
            = trace GameConfig.standard π GameState.init n₂
          ∧ ∀ k, k < n₂ →
            ¬ (trace GameConfig.standard π GameState.init k).lost
              GameConfig.standard := by
  constructor
  · intro hs
    obtain ⟨n₁, n₂, hlt, hbound, hret, -⟩ :=
      survivesForever_return_within hv hdraw hs
    exact ⟨n₂, hbound, n₁, hlt, hret, fun k _ => hs k⟩
  · rintro ⟨n₂, -, n₁, hlt, hret, hlive⟩
    exact survivesForever_of_trace_return hlt hret hlive

/-- The closed cycle carved from any live return — the general form of
`perfectClearCycle`, detecting the loop by state equality directly. -/
def closedCycleOfReturn {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {g0 : GameState}
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag) {n₁ n₂ : ℕ} (hlt : n₁ < n₂)
    (hret : trace GameConfig.standard π g0 n₁
        = trace GameConfig.standard π g0 n₂)
    (hlive : ∀ k, ¬ (trace GameConfig.standard π g0 k).lost
      GameConfig.standard) :
    ClosedCycle GameConfig.standard where
  states := (Finset.range (n₂ - n₁)).image
    (fun k => trace GameConfig.standard π g0 (n₁ + k))
  policy := π
  valid := fun s _ => hv s
  legal_draw := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    rw [Bag.canDraw_iff_mem]
    exact hdraw (n₁ + k)
  not_lost := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    exact hlive (n₁ + k)
  closed := by
    intro s hs
    rw [Finset.mem_image] at hs
    obtain ⟨k, -, rfl⟩ := hs
    have hret' : trace GameConfig.standard π g0 n₁
        = trace GameConfig.standard π g0 (n₁ + (n₂ - n₁)) := by
      rw [show n₁ + (n₂ - n₁) = n₂ by omega]
      exact hret
    have hstep : (trace GameConfig.standard π g0 (n₁ + k)).step
          GameConfig.standard (π (trace GameConfig.standard π g0 (n₁ + k)))
        = trace GameConfig.standard π g0 ((n₁ + k) + 1) :=
      (trace_succ GameConfig.standard π g0 (n₁ + k)).symm
    rw [hstep]
    exact cycle_orbit_subset_period (show 0 < n₂ - n₁ by omega) hret' (by omega)

/-- A surviving trace from a well-formed seed returns (general-seed form of
the pigeonhole). -/
theorem survivesForever_exists_return_from {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hs : SurvivesForever GameConfig.standard π g0) :
    ∃ n₁ n₂, n₁ < n₂ ∧ trace GameConfig.standard π g0 n₁
        = trace GameConfig.standard π g0 n₂ := by
  have hfin : (Set.range (trace GameConfig.standard π g0)).Finite := by
    apply Set.Finite.subset (Set.finite_univ.image
      (fun q : InFieldBoard GameConfig.standard × Bag =>
        GameState.mk q.1.val q.2))
    rintro g ⟨n, rfl⟩
    have hwfn := trace_board_wf hv hwf n
    have hif : ∀ p ∈ (trace GameConfig.standard π g0 n).board,
        p.2 < GameConfig.standard.rows :=
      (GameState.not_lost_iff_forall_row_lt GameConfig.standard _).mp (hs n)
    exact ⟨(⟨(trace GameConfig.standard π g0 n).board, hwfn, hif⟩,
      (trace GameConfig.standard π g0 n).bag), Set.mem_univ _, rfl⟩
  have hninj : ¬ Function.Injective (trace GameConfig.standard π g0) := by
    intro hinj
    exact Set.infinite_range_of_injective hinj hfin
  rw [Function.not_injective_iff] at hninj
  obtain ⟨a, b, heq, hne⟩ := hninj
  rcases Nat.lt_or_ge a b with hab | hab
  · exact ⟨a, b, hab, heq⟩
  · exact ⟨b, a, by omega, heq.symm⟩

/-- **Every surviving policy hides a closed cycle**: a valid, legally-drawn
policy surviving forever from any well-formed seed yields a nonempty
`ClosedCycle` — the M2 artifact is extracted, not assumed. With
`closed_cycle_survives` (the converse), cooperative infinite play and the
existence of cycle certificates are the same phenomenon. -/
theorem exists_closedCycle_of_survives {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hdraw : ∀ k, (π (trace GameConfig.standard π g0 k)).piece
      ∈ (trace GameConfig.standard π g0 k).bag)
    (hs : SurvivesForever GameConfig.standard π g0) :
    ∃ C : ClosedCycle GameConfig.standard, C.states.Nonempty := by
  obtain ⟨n₁, n₂, hlt, hret⟩ := survivesForever_exists_return_from hv hwf hs
  refine ⟨closedCycleOfReturn hv hdraw hlt hret (fun k => hs k), ?_⟩
  refine ⟨trace GameConfig.standard π g0 n₁, ?_⟩
  have : trace GameConfig.standard π g0 n₁
      ∈ (Finset.range (n₂ - n₁)).image
        (fun k => trace GameConfig.standard π g0 (n₁ + k)) := by
    rw [Finset.mem_image]
    exact ⟨0, Finset.mem_range.mpr (by omega), by rw [Nat.add_zero]⟩
  exact this

/-- The cycle's policy patched to be valid everywhere: on the cycle it plays
the cycle's move, off it a fixed in-bounds placement. -/
def ClosedCycle.globalPolicy (C : ClosedCycle GameConfig.standard) :
    Policy GameConfig.standard := fun g =>
  if g ∈ C.states then C.policy g else ⟨Piece.I, 0, 0⟩

/-- The patched policy is valid at every state. -/
theorem ClosedCycle.globalPolicy_valid (C : ClosedCycle GameConfig.standard)
    (g : GameState) : (ClosedCycle.globalPolicy C g).Valid GameConfig.standard := by
  unfold ClosedCycle.globalPolicy
  split
  · exact C.valid g ‹_›
  · intro cell hcell
    have h4 := Piece.shapeUp_col_lt_four Piece.I 0 cell hcell
    dsimp only
    rw [GameConfig.standard_cols]
    omega

/-- On cycle states the patched policy agrees with the cycle's. -/
theorem ClosedCycle.globalPolicy_eq_on (C : ClosedCycle GameConfig.standard)
    {g : GameState} (hg : g ∈ C.states) :
    ClosedCycle.globalPolicy C g = C.policy g := by
  unfold ClosedCycle.globalPolicy
  rw [if_pos hg]

/-- The cycle with its policy globalized — same states, same behaviour on
them, but the policy is now valid everywhere. -/
def ClosedCycle.globalize (C : ClosedCycle GameConfig.standard) :
    ClosedCycle GameConfig.standard where
  states := C.states
  policy := ClosedCycle.globalPolicy C
  valid := fun s _ => ClosedCycle.globalPolicy_valid C s
  legal_draw := by
    intro s hs
    rw [ClosedCycle.globalPolicy_eq_on C hs]
    exact C.legal_draw s hs
  not_lost := C.not_lost
  closed := by
    intro s hs
    rw [ClosedCycle.globalPolicy_eq_on C hs]
    exact C.closed s hs

/-- **Survivors and cycles are the same phenomenon**: there is a globally
valid, trace-legal policy surviving forever from some seed iff there is a
nonempty closed cycle. Forward extracts the orbit; backward globalizes the
cycle's policy and rides `closed_cycle_survives`. -/
theorem exists_survivor_iff_exists_nonempty_cycle :
    (∃ (π : Policy GameConfig.standard) (g0 : GameState),
        (∀ g, (π g).Valid GameConfig.standard)
      ∧ Board.WF GameConfig.standard g0.board
      ∧ (∀ k, (π (trace GameConfig.standard π g0 k)).piece
          ∈ (trace GameConfig.standard π g0 k).bag)
      ∧ SurvivesForever GameConfig.standard π g0)
      ↔ ∃ C : ClosedCycle GameConfig.standard, C.states.Nonempty
          ∧ ∀ s ∈ C.states, Board.WF GameConfig.standard s.board := by
  constructor
  · rintro ⟨π, g0, hv, hwf, hdraw, hs⟩
    obtain ⟨n₁, n₂, hlt, hret⟩ := survivesForever_exists_return_from hv hwf hs
    refine ⟨closedCycleOfReturn hv hdraw hlt hret (fun k => hs k), ?_, ?_⟩
    · refine ⟨trace GameConfig.standard π g0 n₁, ?_⟩
      have : trace GameConfig.standard π g0 n₁
          ∈ (Finset.range (n₂ - n₁)).image
            (fun k => trace GameConfig.standard π g0 (n₁ + k)) := by
        rw [Finset.mem_image]
        exact ⟨0, Finset.mem_range.mpr (by omega), by rw [Nat.add_zero]⟩
      exact this
    · intro s hs'
      have hs'' : s ∈ (Finset.range (n₂ - n₁)).image
          (fun k => trace GameConfig.standard π g0 (n₁ + k)) := hs'
      rw [Finset.mem_image] at hs''
      obtain ⟨k, -, rfl⟩ := hs''
      exact trace_board_wf hv hwf (n₁ + k)
  · rintro ⟨C, ⟨g0, hg0⟩, hwfs⟩
    refine ⟨ClosedCycle.globalPolicy C, g0,
      ClosedCycle.globalPolicy_valid C, hwfs g0 hg0, ?_, ?_⟩
    · intro k
      have hmem : trace GameConfig.standard (ClosedCycle.globalPolicy C) g0 k
          ∈ C.states :=
        (ClosedCycle.globalize C).trace_mem_states
          (show g0 ∈ (ClosedCycle.globalize C).states from hg0) k
      rw [ClosedCycle.globalPolicy_eq_on C hmem]
      have := C.legal_draw _ hmem
      rwa [Bag.canDraw_iff_mem] at this
    · have hs := closed_cycle_survives (ClosedCycle.globalize C)
        (show g0 ∈ (ClosedCycle.globalize C).states from hg0)
      exact hs

/-- **Any 35-window shows at least five distinct occupancy values** — the
finest diversity observable: the count's residue mod 10 steps through all
five even-cycle residues within any five consecutive placements. Refines
`trace_window_boards_ge_five` at the level of a single scalar. -/
theorem trace_window_counts_ge_five {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) (n : ℕ) :
    5 ≤ ((Finset.range 35).image
      (fun k => (trace GameConfig.standard π GameState.init (n + k)).board.count)).card := by
  classical
  have hsub : (Finset.range 5).image
        (fun k => (trace GameConfig.standard π GameState.init (n + k)).board.count)
      ⊆ (Finset.range 35).image
        (fun k => (trace GameConfig.standard π GameState.init (n + k)).board.count) := by
    apply Finset.image_subset_image
    intro x hx
    rw [Finset.mem_range] at hx ⊢
    omega
  refine le_trans ?_ (Finset.card_le_card hsub)
  have hinj : ∀ i ∈ Finset.range 5, ∀ k ∈ Finset.range 5,
      (trace GameConfig.standard π GameState.init (n + i)).board.count
        = (trace GameConfig.standard π GameState.init (n + k)).board.count
      → i = k := by
    intro i hi k hk heq
    rw [Finset.mem_range] at hi hk
    have h1 := count_mod_ten hv (n + i)
    have h2 := count_mod_ten hv (n + k)
    rw [heq] at h1
    omega
  have := Finset.card_image_of_injOn (s := Finset.range 5)
    (f := fun k => (trace GameConfig.standard π GameState.init (n + k)).board.count)
    (fun i hi k hk => hinj i hi k hk)
  rw [this, Finset.card_range]

/-- **The phase is observable**: within any 35-window, a state's bag fill
level and occupancy residue mod 10 determine its position — the two cheap
scalar observables jointly reconstruct the five-bag phase (CRT of the two
clocks). -/
theorem trace_window_phase_determined {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n i k : ℕ}
    (hi : i < 35) (hk : k < 35)
    (hbag : (trace GameConfig.standard π GameState.init (n + i)).bag.card
        = (trace GameConfig.standard π GameState.init (n + k)).bag.card)
    (hcnt : (trace GameConfig.standard π GameState.init (n + i)).board.count % 10
        = (trace GameConfig.standard π GameState.init (n + k)).board.count % 10) :
    i = k := by
  have h1 := bag_card_trace hdraw (n + i)
  have h2 := bag_card_trace hdraw (n + k)
  have h3 := count_mod_ten hv (n + i)
  have h4 := count_mod_ten hv (n + k)
  rw [h1, h2] at hbag
  rw [h3, h4] at hcnt
  omega

/-- The adversarial mirror: the two clocks reconstruct the phase whoever
picks the pieces (any well-formed seed). -/
theorem adversarialTrace_window_phase_determined {σ : Solver GameConfig.standard}
    {s : ℕ → Piece} {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board)
    (hv : ∀ n, ({ σ (adversarialTrace GameConfig.standard σ s g0 n) (s n)
      with piece := s n } : Placement).Valid GameConfig.standard)
    (hdraw : ∀ n, s n ∈ (adversarialTrace GameConfig.standard σ s g0 n).bag)
    {n i k : ℕ} (hi : i < 35) (hk : k < 35)
    (hbag : (adversarialTrace GameConfig.standard σ s g0 (n + i)).bag.card
        = (adversarialTrace GameConfig.standard σ s g0 (n + k)).bag.card)
    (hcnt : (adversarialTrace GameConfig.standard σ s g0 (n + i)).board.count % 10
        = (adversarialTrace GameConfig.standard σ s g0 (n + k)).board.count % 10) :
    i = k := by
  have h1 := bag_card_adversarialTrace hdraw (n + i)
  have h2 := bag_card_adversarialTrace hdraw (n + k)
  have h3 := adversarialTrace_count_mod_ten hwf hv (n + i)
  have h4 := adversarialTrace_count_mod_ten hwf hv (n + k)
  have hle7 : g0.bag.card ≤ 7 := Bag.card_le_seven g0.bag
  have hpos : 0 < g0.bag.card := Finset.card_pos.mpr ⟨_, hdraw 0⟩
  rw [h1, h2] at hbag
  rw [h3, h4] at hcnt
  omega

/-- **The I-counter frequency law**: `⌊(Δn−7)/7⌋ ≤ ΔiCount ≤ ⌊Δn/7⌋ + 2` at
every horizon — the I supply is `1/7` of the placements with error at most
two, at every scale. Supersedes the 35-block bracket. -/
theorem iCount_frequency_law {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n m : ℕ}
    (hnm : n ≤ m) :
    (m - n - 7) / 7 ≤ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
      ∧ iCount GameConfig.standard π GameState.init m
        - iCount GameConfig.standard π GameState.init n
        ≤ (m - n) / 7 + 2 := by
  have hl := legalSequence_of_trace_draws hdraw
  have h1 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) n
  have h2 := iCount_eq_card_filter (cfg := GameConfig.standard) (π := π) m
  have hsplit := card_filter_range_add (fun k =>
    (π (trace GameConfig.standard π GameState.init k)).piece = Piece.I)
    n (m - n)
  rw [show n + (m - n) = m by omega] at hsplit
  have hwin := BagCadence.window_frequency_law hl n Piece.I (m - n)
  omega

/-- **The tetris frequency cap**: at most `⌊Δn/7⌋ + 2` tetrises at every
horizon past the seed — each rides an I, and I's run at `1/7 + O(1)`. -/
theorem tetris_frequency_cap {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n m : ℕ}
    (hn : 1 ≤ n) (hnm : n ≤ m) :
    sizeCount GameConfig.standard π GameState.init 4 m
        - sizeCount GameConfig.standard π GameState.init 4 n
      ≤ (m - n) / 7 + 2 := by
  have hemb := tetris_le_I_window (π := π) hn hnm
  obtain ⟨-, hI⟩ := iCount_frequency_law hdraw hnm
  omega

/-- **The big-clear frequency cap**: triples and tetrises together number at
most `3⌊Δn/7⌋ + 6` in any window past the seed — every 3+-clear rides an I,
L or J, and each of those runs at `1/7 + O(1)`. -/
theorem big_clear_frequency_cap {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n m : ℕ}
    (hnm : n ≤ m) :
    (sizeCount GameConfig.standard π GameState.init 3 m
        - sizeCount GameConfig.standard π GameState.init 3 n)
      + (sizeCount GameConfig.standard π GameState.init 4 m
        - sizeCount GameConfig.standard π GameState.init 4 n)
      ≤ 3 * ((m - n) / 7) + 6 := by
  classical
  have hl := legalSequence_of_trace_draws hdraw
  set w := m - n with hw
  -- windowed counters as filter cards
  have h3 := sizeCount_window (cfg := GameConfig.standard) (π := π) 3 n w
  have h4 := sizeCount_window (cfg := GameConfig.standard) (π := π) 4 n w
  rw [show n + w = m by omega] at h3 h4
  -- the two big-clear fibers are disjoint and land in the ILJ fiber
  set A := (Finset.range w).filter (fun j =>
    (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init (n + j))).place
        (trace GameConfig.standard π GameState.init (n + j)).board)).card = 3)
    with hA
  set B := (Finset.range w).filter (fun j =>
    (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init (n + j))).place
        (trace GameConfig.standard π GameState.init (n + j)).board)).card = 4)
    with hB
  set C := (Finset.range w).filter (fun j =>
    (π (trace GameConfig.standard π GameState.init (n + j))).piece = Piece.I
      ∨ (π (trace GameConfig.standard π GameState.init (n + j))).piece = Piece.L
      ∨ (π (trace GameConfig.standard π GameState.init (n + j))).piece = Piece.J)
    with hC
  have hdisj : Disjoint A B := by
    rw [Finset.disjoint_left]
    intro j hjA hjB
    rw [hA, Finset.mem_filter] at hjA
    rw [hB, Finset.mem_filter] at hjB
    omega
  have hAC : A ⊆ C := by
    intro j hj
    rw [hA, Finset.mem_filter] at hj
    rw [hC, Finset.mem_filter]
    exact ⟨hj.1, three_clear_requires_I_L_or_J_trace hv (le_of_eq hj.2.symm)⟩
  have hBC : B ⊆ C := by
    intro j hj
    rw [hB, Finset.mem_filter] at hj
    rw [hC, Finset.mem_filter]
    refine ⟨hj.1, three_clear_requires_I_L_or_J_trace hv ?_⟩
    omega
  have hunion : A.card + B.card ≤ C.card := by
    rw [← Finset.card_union_of_disjoint hdisj]
    exact Finset.card_le_card (Finset.union_subset hAC hBC)
  -- the ILJ fiber splits into the three piece fibers
  have hCsplit : C.card
      ≤ ((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.I)).card
        + ((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.L)).card
        + ((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.J)).card := by
    have hsub : C ⊆ ((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.I))
        ∪ (((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.L))
        ∪ ((Finset.range w).filter (fun j =>
          (π (trace GameConfig.standard π GameState.init (n + j))).piece
            = Piece.J))) := by
      intro j hj
      rw [hC, Finset.mem_filter] at hj
      rw [Finset.mem_union, Finset.mem_union]
      rcases hj.2 with h | h | h
      · exact Or.inl (Finset.mem_filter.mpr ⟨hj.1, h⟩)
      · exact Or.inr (Or.inl (Finset.mem_filter.mpr ⟨hj.1, h⟩))
      · exact Or.inr (Or.inr (Finset.mem_filter.mpr ⟨hj.1, h⟩))
    calc C.card ≤ _ := Finset.card_le_card hsub
      _ ≤ _ := le_trans (Finset.card_union_le _ _)
        (by
          have := Finset.card_union_le
            ((Finset.range w).filter (fun j =>
              (π (trace GameConfig.standard π GameState.init (n + j))).piece
                = Piece.L))
            ((Finset.range w).filter (fun j =>
              (π (trace GameConfig.standard π GameState.init (n + j))).piece
                = Piece.J))
          omega)
  -- each piece fiber obeys the frequency law
  have hI := BagCadence.window_frequency_law hl n Piece.I w
  have hL := BagCadence.window_frequency_law hl n Piece.L w
  have hJ := BagCadence.window_frequency_law hl n Piece.J w
  omega

/-- **The window grid is a bijection**: within any 35-window, every
(bag level, mass phase) cell is realised at exactly one position — existence
by Chinese-remainder index choice, uniqueness by the observable phase.
The five-bag window *is* the 7 × 5 clock grid. -/
theorem trace_window_grid_unique {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ)
    {c : ℕ} (hc1 : 1 ≤ c) (hc7 : c ≤ 7) {i : ℕ} (hi : i < 5) :
    ∃! k, k < 35
      ∧ (trace GameConfig.standard π GameState.init (n + k)).bag.card = c
      ∧ (trace GameConfig.standard π GameState.init (n + k)).board.count % 10
        = (4 * (n + i)) % 10 := by
  have hbagf := bag_card_trace hdraw
  have hcntf := count_mod_ten hv
  -- the residue giving bag level c, then the CRT block choice for the phase
  set r := (7 - c + 6 * (n % 7)) % 7 with hr
  set j := (3 * (i + 5 - r % 5)) % 5 with hj
  refine ⟨r + 7 * j, ⟨by omega, ?_, ?_⟩, ?_⟩
  · rw [hbagf]
    omega
  · rw [hcntf]
    omega
  · rintro k ⟨hk35, hkbag, hkcnt⟩
    have h1 := hbagf (n + k)
    have h2 := hcntf (n + k)
    rw [h1] at hkbag
    rw [h2] at hkcnt
    omega

/-- **The trace bag content law**: from the empty board, the bag at step `n`
is the full bag minus the pieces played since the block boundary `7⌊n/7⌋`. -/
theorem trace_bag_eq_sdiff {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).bag
      = Bag.full \ ((Finset.range (n % 7)).image (fun j =>
          (π (trace GameConfig.standard π GameState.init
            (7 * (n / 7) + j))).piece)) := by
  have hl := legalSequence_of_trace_draws hdraw
  have hl' : LegalSequenceFrom Bag.full
      (fun m => (π (trace GameConfig.standard π GameState.init m)).piece) := by
    have hb : GameState.init.bag = Bag.full := GameState.init_bag
    rwa [hb] at hl
  have h := BagCadence.bagAt_eq_sdiff hl' n
  rw [trace_bag_eq_bagAt]
  rw [show GameState.init.bag = Bag.full from GameState.init_bag]
  exact h

/-- **The availability criterion**: a piece is drawable at step `n` iff the
policy has not played it since the current block began — bag membership is
an explicit statement about the last `n mod 7` moves. -/
theorem trace_piece_available_iff {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ)
    (p : Piece) :
    p ∈ (trace GameConfig.standard π GameState.init n).bag
      ↔ ∀ j, j < n % 7 →
        (π (trace GameConfig.standard π GameState.init
          (7 * (n / 7) + j))).piece ≠ p := by
  rw [trace_bag_eq_sdiff hdraw, Finset.mem_sdiff]
  constructor
  · rintro ⟨-, hnot⟩ j hj hp
    exact hnot (Finset.mem_image.mpr ⟨j, Finset.mem_range.mpr hj, hp⟩)
  · intro h
    refine ⟨Bag.mem_full p, ?_⟩
    intro hmem
    obtain ⟨j, hj, hp⟩ := Finset.mem_image.mp hmem
    exact h j (Finset.mem_range.mp hj) hp

/-- Any live return proves survival, from any seed. -/
theorem survivesForever_of_trace_return_from {π : Policy GameConfig.standard}
    {g0 : GameState} {n₁ n₂ : ℕ} (hlt : n₁ < n₂)
    (hret : trace GameConfig.standard π g0 n₁
        = trace GameConfig.standard π g0 n₂)
    (hlive : ∀ k, k < n₂ →
      ¬ (trace GameConfig.standard π g0 k).lost GameConfig.standard) :
    SurvivesForever GameConfig.standard π g0 := by
  intro m
  rcases Nat.lt_or_ge m n₁ with hm | hm
  · exact hlive m (by omega)
  · have hret' : trace GameConfig.standard π g0 n₁
        = trace GameConfig.standard π g0 (n₁ + (n₂ - n₁)) := by
      rw [show n₁ + (n₂ - n₁) = n₂ by omega]
      exact hret
    have hmem := cycle_orbit_subset_period (show 0 < n₂ - n₁ by omega) hret' hm
    rw [Finset.mem_image] at hmem
    obtain ⟨k, hk, heq⟩ := hmem
    rw [← heq]
    exact hlive (n₁ + k) (by have := Finset.mem_range.mp hk; omega)

/-- **The survival characterization at every well-formed seed**: a valid
policy survives forever from `g0` iff its trace shows a finite live prefix
ending in a revisit. The bank-75 init theorem, seed-general. -/
theorem survivesForever_iff_live_return_from {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {g0 : GameState}
    (hwf : Board.WF GameConfig.standard g0.board) :
    SurvivesForever GameConfig.standard π g0
      ↔ ∃ n₁ n₂, n₁ < n₂
          ∧ trace GameConfig.standard π g0 n₁
            = trace GameConfig.standard π g0 n₂
          ∧ ∀ k, k < n₂ →
            ¬ (trace GameConfig.standard π g0 k).lost GameConfig.standard := by
  constructor
  · intro hs
    obtain ⟨n₁, n₂, hlt, hret⟩ := survivesForever_exists_return_from hv hwf hs
    exact ⟨n₁, n₂, hlt, hret, fun k _ => hs k⟩
  · rintro ⟨n₁, n₂, hlt, hret, hlive⟩
    exact survivesForever_of_trace_return_from hlt hret hlive

/-- The tetris well, on traces: past the seed, a four-clear's four gaps sit
in one straight column of the current board. -/
theorem trace_tetris_well {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 1 ≤ m)
    (h4 : (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    ∃ c₀, ∀ r ∈ Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board),
      (c₀, r) ∉ (trace GameConfig.standard π GameState.init m).board
        ∧ (c₀, r) ∈ (π (trace GameConfig.standard π GameState.init m)).dropped
            (trace GameConfig.standard π GameState.init m).board :=
  tetris_gaps_share_column
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m)
    (hv _) (fun r => trace_board_no_full_of_pos hm r) h4

/-- The unified per-row floor, on traces: past the seed, each row a step
completes held at least `5 + k` cells. -/
theorem trace_cleared_row_pre_ge {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 1 ≤ m) {r : ℕ}
    (hr : r ∈ Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)) :
    5 + (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init m)).place
          (trace GameConfig.standard π GameState.init m).board)).card
      ≤ (trace GameConfig.standard π GameState.init m).board.rowCount r :=
  cleared_row_pre_ge
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m)
    (hv _) (fun r' => trace_board_no_full_of_pos hm r') hr

/-- The single-column feed, on traces: past the seed, a four-clear step
pours its entire four-cell feed into one column — profile `4` there, `0`
at the other nine. -/
theorem trace_tetris_feeds_single_column {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard) {m : ℕ} (hm : 1 ≤ m)
    (h4 : (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card = 4) :
    ∃ c₀ < 10,
      (π (trace GameConfig.standard π GameState.init m)).colProfile c₀ = 4
      ∧ ∀ j < 10, j ≠ c₀ →
        (π (trace GameConfig.standard π GameState.init m)).colProfile j
          = 0 :=
  tetris_feeds_single_column
    (trace_board_wf hv (GameState.init_board_wf GameConfig.standard) m)
    (hv _) (fun r => trace_board_no_full_of_pos hm r) h4

/-- The O clear cap, on traces: past the seed, a step playing the O clears
at most two rows. -/
theorem trace_clears_le_two_of_O {π : Policy GameConfig.standard}
    {m : ℕ} (hm : 1 ≤ m)
    (hO : (π (trace GameConfig.standard π GameState.init m)).piece
      = Piece.O) :
    (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card ≤ 2 :=
  clears_le_two_of_O (fun r => trace_board_no_full_of_pos hm r) hO

/-- The non-I clear cap, on traces: past the seed, a step not playing the
I clears at most three rows. -/
theorem trace_clears_le_three_of_ne_I {π : Policy GameConfig.standard}
    {m : ℕ} (hm : 1 ≤ m)
    (hI : (π (trace GameConfig.standard π GameState.init m)).piece
      ≠ Piece.I) :
    (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init m)).place
        (trace GameConfig.standard π GameState.init m).board)).card ≤ 3 :=
  clears_le_three_of_ne_I (fun r => trace_board_no_full_of_pos hm r) hI

/-- **The global tall-drop cap**: per cycle period, at most five placements
pour a full four-cell feed into *any* column at all — each such drop plays
the I (`full_feed_requires_I`), and the bag deals exactly five I's per
period. Sharper than summing the per-column caps (which would give thirty):
the whole board hosts at most five tall drops per period. -/
theorem cycle_tall_drop_total_cap {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ((Finset.range 35).filter (fun k =>
        ∃ j ∈ Finset.range 10,
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile j
            = 4)).card ≤ 5 := by
  classical
  have hbal := trace_period_piece_balanced hdraw hcyc Piece.I
  refine le_trans (Finset.card_le_card ?_) (le_of_eq hbal)
  intro k hk
  rw [Finset.mem_filter] at hk ⊢
  obtain ⟨hkr, j, -, hj4⟩ := hk
  exact ⟨hkr, full_feed_requires_I (le_of_eq hj4.symm)⟩

/-- **The global heavy-feed cap**: per cycle period, at most twenty
placements deliver three or more cells into any single column — each such
feed plays a tall piece (I, L, J or T), and the bag deals exactly five of
each per period. The supply-side companion of `cycle_tall_drop_total_cap`. -/
theorem cycle_heavy_feed_total_cap {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ((Finset.range 35).filter (fun k =>
        ∃ j ∈ Finset.range 10, 3 ≤
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
            j)).card ≤ 20 := by
  classical
  have hI := trace_period_piece_balanced hdraw hcyc Piece.I
  have hL := trace_period_piece_balanced hdraw hcyc Piece.L
  have hJ := trace_period_piece_balanced hdraw hcyc Piece.J
  have hT := trace_period_piece_balanced hdraw hcyc Piece.T
  have hsub : (Finset.range 35).filter (fun k =>
        ∃ j ∈ Finset.range 10, 3 ≤
          (π (trace GameConfig.standard π GameState.init (n + k))).colProfile
            j)
      ⊆ (Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.I
          ∨ (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.L
          ∨ (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.J
          ∨ (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.T) := by
    intro k hk
    rw [Finset.mem_filter] at hk ⊢
    obtain ⟨hkr, j, -, hj3⟩ := hk
    exact ⟨hkr, heavy_feed_requires_tall hj3⟩
  refine le_trans (Finset.card_le_card hsub) ?_
  rw [Finset.filter_or, Finset.filter_or, Finset.filter_or]
  have h1 := Finset.card_union_le
    ((Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece
        = Piece.I))
    (((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.L))
      ∪ (((Finset.range 35).filter (fun k =>
          (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.J))
        ∪ ((Finset.range 35).filter (fun k =>
          (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.T))))
  have h2 := Finset.card_union_le
    ((Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece
        = Piece.L))
    (((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.J))
      ∪ ((Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.T)))
  have h3 := Finset.card_union_le
    ((Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece
        = Piece.J))
    ((Finset.range 35).filter (fun k =>
      (π (trace GameConfig.standard π GameState.init (n + k))).piece
        = Piece.T))
  omega

/-- Window clears decompose as the sum of the window's per-step clear
counts. -/
theorem cleared_window_sum (π : Policy GameConfig.standard) (n : ℕ) :
    ∀ w, cleared GameConfig.standard π GameState.init (n + w)
        - cleared GameConfig.standard π GameState.init n
      = ∑ k ∈ Finset.range w,
          (Board.fullRows GameConfig.standard
            ((π (trace GameConfig.standard π GameState.init (n + k))).place
              (trace GameConfig.standard π GameState.init (n + k)).board)).card := by
  intro w
  induction w with
  | zero => simp
  | succ w ih =>
    have hmono := cleared_mono GameConfig.standard π GameState.init
      (Nat.le_add_right n w)
    rw [show n + (w + 1) = (n + w) + 1 by omega, cleared_succ,
      Finset.sum_range_succ, ← ih]
    omega

/-- **The O clears at most ten rows per period**: its five appearances can
each complete at most two rows. -/
theorem cycle_O_step_clears_le {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    ∑ k ∈ (Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.O),
      (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init (n + k))).place
          (trace GameConfig.standard π GameState.init (n + k)).board)).card
      ≤ 10 := by
  classical
  have hbal := trace_period_piece_balanced hdraw hcyc Piece.O
  calc ∑ k ∈ (Finset.range 35).filter (fun k =>
        (π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.O),
      (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init (n + k))).place
          (trace GameConfig.standard π GameState.init (n + k)).board)).card
      ≤ ∑ _k ∈ (Finset.range 35).filter (fun k =>
          (π (trace GameConfig.standard π GameState.init (n + k))).piece
            = Piece.O), 2 :=
        Finset.sum_le_sum (fun k hk =>
          clears_le_two_of_O (fun r => trace_board_no_full (n + k) r)
            (Finset.mem_filter.mp hk).2)
    _ = 10 := by
        rw [Finset.sum_const, hbal, smul_eq_mul]

/-- **Tall pieces must clear**: per cycle period, at least four of the
fourteen cleared rows fall at steps not playing the O — square-only
clearing cannot meet the rate law. -/
theorem cycle_non_O_clear_floor {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) {n : ℕ}
    (hcyc : trace GameConfig.standard π GameState.init n
        = trace GameConfig.standard π GameState.init (n + 35)) :
    4 ≤ ∑ k ∈ (Finset.range 35).filter (fun k =>
        ¬ ((π (trace GameConfig.standard π GameState.init (n + k))).piece
          = Piece.O)),
      (Board.fullRows GameConfig.standard
        ((π (trace GameConfig.standard π GameState.init (n + k))).place
          (trace GameConfig.standard π GameState.init (n + k)).board)).card := by
  classical
  have htot := multi_period_clears hv hcyc 1
  rw [mul_one] at htot
  have hwin := cleared_window_sum π n 35
  have hsplit := Finset.sum_filter_add_sum_filter_not (Finset.range 35)
    (fun k => (π (trace GameConfig.standard π GameState.init (n + k))).piece
      = Piece.O)
    (fun k => (Board.fullRows GameConfig.standard
      ((π (trace GameConfig.standard π GameState.init (n + k))).place
        (trace GameConfig.standard π GameState.init (n + k)).board)).card)
  have hO := cycle_O_step_clears_le hdraw hcyc
  omega

/-- The bag's cardinality on a trace follows the block clock: `7 − n % 7`. -/
theorem trace_bag_card {π : Policy GameConfig.standard}
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag) (n : ℕ) :
    (trace GameConfig.standard π GameState.init n).bag.card = 7 - n % 7 := by
  have hl := legalSequence_of_trace_draws hdraw
  rw [show GameState.init.bag = Bag.full from GameState.init_bag] at hl
  rw [trace_bag_eq_bagAt π GameState.init n,
    show GameState.init.bag = Bag.full from GameState.init_bag]
  exact BagCadence.bagAt_card hl n

/-- **The state wears the cycle clock**: two reachable states — under any
two policies — agreeing on bag size and on board mass mod ten sit at the
same step index mod 35. The bag reads the block clock (mod 7), the mass
reads the mass clock (mod 5), and the two residues CRT to the full cycle
quantum: a state revisit can only happen a multiple of 35 steps later. -/
theorem trace_state_reveals_step_mod_thirtyfive
    {π π' : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hv' : ∀ g, (π' g).Valid GameConfig.standard)
    (hdraw : ∀ k, (π (trace GameConfig.standard π GameState.init k)).piece
      ∈ (trace GameConfig.standard π GameState.init k).bag)
    (hdraw' : ∀ k, (π' (trace GameConfig.standard π' GameState.init k)).piece
      ∈ (trace GameConfig.standard π' GameState.init k).bag) {m n : ℕ}
    (hbag : (trace GameConfig.standard π GameState.init m).bag.card
      = (trace GameConfig.standard π' GameState.init n).bag.card)
    (hcount : (trace GameConfig.standard π GameState.init m).board.count % 10
      = (trace GameConfig.standard π' GameState.init n).board.count % 10) :
    m % 35 = n % 35 := by
  have hb1 := trace_bag_card hdraw m
  have hb2 := trace_bag_card hdraw' n
  have hm1 := trace_board_count_mod_ten hv m
  have hm2 := trace_board_count_mod_ten hv' n
  omega

/-- **Every visited state is visited early**: once the trace returns
(`trace n₁ = trace n₂`), every state it ever reaches already appears
before index `n₂` — the pre-return prefix exhausts the orbit. -/
theorem visited_state_early {cfg : GameConfig} {π : Policy cfg}
    {g0 : GameState} {n₁ n₂ : ℕ} (h12 : n₁ < n₂)
    (hret : trace cfg π g0 n₁ = trace cfg π g0 n₂) :
    ∀ m, ∃ m' < n₂, trace cfg π g0 m' = trace cfg π g0 m := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
    rcases Nat.lt_or_ge m n₂ with hm | hm
    · exact ⟨m, hm, rfl⟩
    · have hk := trace_eq_of_state_eq π g0 hret (m - n₂)
      rw [show n₂ + (m - n₂) = m by omega] at hk
      obtain ⟨m', hm', heq⟩ := ih (n₁ + (m - n₂)) (by omega)
      exact ⟨m', hm', heq.trans hk⟩

/-- **A survivor's orbit is finite**: the set of states a forever-surviving
policy visits from the empty board is a finite set — the minimal atlas of
any solution is a finite object, exhausted by the trace's pre-return
prefix. -/
theorem survivor_orbit_finite {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hs : SurvivesForever GameConfig.standard π GameState.init) :
    (Set.range (trace GameConfig.standard π GameState.init)).Finite := by
  obtain ⟨n₁, n₂, h12, hret⟩ := survivesForever_exists_return hv hs
  apply Set.Finite.subset
    (Set.Finite.image (trace GameConfig.standard π GameState.init)
      (Finset.range n₂).finite_toSet)
  rintro g ⟨m, rfl⟩
  obtain ⟨m', hm', heq⟩ := visited_state_early h12 hret m
  exact ⟨m', Finset.mem_coe.mpr (Finset.mem_range.mpr hm'), heq⟩

/-- **Finite-invariant extraction**: every cooperative survivor yields a
*finite* set of states containing the empty-board start, free of lost
states, and closed under its own policy's step — the survivor's orbit,
packaged as the finite certificate the Atlas construction is after. -/
theorem survivor_finite_invariant {π : Policy GameConfig.standard}
    (hv : ∀ g, (π g).Valid GameConfig.standard)
    (hs : SurvivesForever GameConfig.standard π GameState.init) :
    ∃ S : Set GameState, S.Finite ∧ GameState.init ∈ S
      ∧ (∀ g ∈ S, ¬ g.lost GameConfig.standard)
      ∧ ∀ g ∈ S, g.step GameConfig.standard (π g) ∈ S := by
  refine ⟨Set.range (trace GameConfig.standard π GameState.init),
    survivor_orbit_finite hv hs, ⟨0, rfl⟩, ?_, ?_⟩
  · rintro g ⟨n, rfl⟩
    exact hs n
  · rintro g ⟨n, rfl⟩
    exact ⟨n + 1, (trace_succ GameConfig.standard π GameState.init n).symm⟩

/-- The converse: a step-closed, lost-free set containing a state makes its
policy a survivor from that state — the finite invariant *is* the survival
certificate, in both directions. -/
theorem survivesForever_of_invariant {cfg : GameConfig} {π : Policy cfg}
    {S : Set GameState} {g0 : GameState} (hg0 : g0 ∈ S)
    (hsafe : ∀ g ∈ S, ¬ g.lost cfg)
    (hclosed : ∀ g ∈ S, g.step cfg (π g) ∈ S) :
    SurvivesForever cfg π g0 := by
  have hmem : ∀ n, trace cfg π g0 n ∈ S := by
    intro n
    induction n with
    | zero => exact hg0
    | succ k ih =>
      rw [trace_succ]
      exact hclosed _ ih
  intro n
  exact hsafe _ (hmem n)

end ClearRate
end Tetris
