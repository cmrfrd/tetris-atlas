import Mathlib
import Proofs.Survival.ClearRecurrence
import Proofs.Survival.ClearMix
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

end ClearRate
end Tetris
